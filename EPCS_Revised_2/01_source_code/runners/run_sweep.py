"""
Revision-3 sweep orchestrator.

Scheduling is derived from measurement, not assumption.

  * concurrency_probe.json: two concurrent EPCS-Net jobs give 0.71x the
    throughput of running them one after the other. The GPU is already
    saturated by a single job.
  * An initial two-lane design (one heavy, one light) was tried and ABANDONED
    after measuring it live: U-Net slowed from 11.6 to 42.7 s/epoch and
    U-Net3+ from 72.5 to 185.0 s/epoch, making the split 2.2x worse than
    serial. Data loading was separately measured at 4.3 ms/image, i.e. 2.4 s
    per NuInsSeg epoch, so I/O is not a factor.

So: ONE job at a time, ordered cheapest-first, which also means completed
results accumulate early and a late interruption costs the least.

Every run is a separate subprocess so a crash cannot take the sweep down, and
every run is skipped if its summary.json already exists, so the sweep is
resumable after an interruption.

Usage:
    python run_sweep.py --plan            # print the matrix and the estimate
    python run_sweep.py --run             # execute
    python run_sweep.py --run --only nuinsseg
    python run_sweep.py --status          # how far along
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from models_r3 import MODELS                                    # noqa: E402
from protocol_r3 import SEEDS, run_dir, run_id                  # noqa: E402

R3 = Path(__file__).resolve().parents[2]
PROTOCOL = R3 / "03_code" / "shared" / "protocol_r3.py"
LOGS = R3 / "04_results" / "logs"
BENCH = LOGS / "benchmark_r3.json"

PUBLISHED = ["unet", "unetpp", "unet3plus", "eunet3plus", "sam_unet", "epcs_net"]
REFERENCE21 = ["ref21_bottleneck", "ref21_pyramid"]
ABLATION = ["abl1_nosam", "abl2_bottleneck", "abl3_pspi", "abl4_bdsg", "abl5_dksr",
            "abl6_ftas", "ablA_acfb_nopspi", "ablB_random_embed",
            "ablC_pspi_concat", "ablD_gn_encoder"]

CRYO_FOLDS = (1, 2, 3)
ABLATION_FOLD = 1          # the ablation is a CryoNuSeg study, fold 1 only
HEAVY_STEP_S = 0.10        # above this a job saturates the GPU


def build_matrix() -> list[dict]:
    jobs = []

    def add(model, dataset, seed, fold, block):
        jobs.append({"model": model, "dataset": dataset, "seed": seed,
                     "fold": fold, "block": block,
                     "run_id": run_id(model, dataset, seed, fold)})

    for m in PUBLISHED + REFERENCE21:
        for s in SEEDS:
            add(m, "nuinsseg", s, None, "main-nuinsseg")
    for m in PUBLISHED + REFERENCE21:
        for f in CRYO_FOLDS:
            for s in SEEDS:
                add(m, "cryonuseg", s, f, "main-cryonuseg")
    for m in ABLATION:
        for s in SEEDS:
            add(m, "cryonuseg", s, ABLATION_FOLD, "ablation-cryonuseg")
    return jobs


def is_done(job: dict) -> bool:
    return (run_dir(job["dataset"], job["model"], job["seed"], job["fold"])
            / "summary.json").is_file()


def load_bench() -> dict:
    if not BENCH.is_file():
        return {}
    return json.loads(BENCH.read_text(encoding="utf-8"))


def estimate_seconds(job: dict, bench: dict) -> float:
    b = bench.get(job["model"])
    if b is None:                       # ref21 has no benchmark; use its nearest twin
        twin = {"ref21_bottleneck": "abl2_bottleneck",
                "ref21_pyramid": "abl3_pspi"}.get(job["model"])
        b = bench.get(twin)
    if b is None:
        return 0.0
    shape = ({"train": 459, "val": 103, "test": 103} if job["dataset"] == "nuinsseg"
             else {"train": 15, "val": 5, "test": 10})
    per_epoch = shape["train"] * b["train_step_s"] + shape["val"] * b["infer_s"]
    return 50 * per_epoch + (shape["val"] + shape["test"]) * b["infer_s"]


def is_heavy(job: dict, bench: dict) -> bool:
    b = bench.get(job["model"]) or bench.get(
        {"ref21_bottleneck": "abl2_bottleneck", "ref21_pyramid": "abl3_pspi"}.get(job["model"], ""))
    return bool(b) and b["train_step_s"] >= HEAVY_STEP_S


def run_job(job: dict, log_path: Path) -> tuple[bool, float]:
    cmd = [sys.executable, str(PROTOCOL),
           "--model", job["model"], "--dataset", job["dataset"],
           "--seed", str(job["seed"])]
    if job["fold"]:
        cmd += ["--fold", str(job["fold"])]
    t0 = time.time()
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n{'='*80}\n{datetime.now():%Y-%m-%d %H:%M:%S}  {job['run_id']}\n{'='*80}\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return proc.returncode == 0, time.time() - t0


class Lane(threading.Thread):
    def __init__(self, name: str, jobs: list[dict], state: dict, lock: threading.Lock):
        super().__init__(daemon=True)
        self.name_ = name
        self.jobs = jobs
        self.state = state
        self.lock = lock
        self.log = LOGS / f"sweep_{name}.log"

    def run(self):
        for job in self.jobs:
            if is_done(job):
                with self.lock:
                    self.state["skipped"] += 1
                continue
            ok, secs = run_job(job, self.log)
            with self.lock:
                self.state["done" if ok else "failed"] += 1
                self.state["elapsed"] += secs
                if not ok:
                    self.state["failures"].append(job["run_id"])
                left = self.state["total"] - (self.state["done"] + self.state["failed"]
                                              + self.state["skipped"])
                print(f"[{self.name_:>5}] {'ok ' if ok else 'FAIL'} "
                      f"{job['run_id']:<46} {secs/60:>6.1f} min   {left} left",
                      flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", default=None, help="restrict to a block substring")
    a = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    bench = load_bench()
    jobs = build_matrix()
    if a.only:
        jobs = [j for j in jobs if a.only in j["block"] or a.only in j["dataset"]]

    pending = [j for j in jobs if not is_done(j)]
    total_sec = sum(estimate_seconds(j, bench) for j in pending)

    if a.status or a.plan:
        done = len(jobs) - len(pending)
        print(f"Runs in matrix: {len(jobs)}   complete: {done}   pending: {len(pending)}")
        by_block = {}
        for j in jobs:
            b = by_block.setdefault(j["block"], [0, 0])
            b[0] += 1
            b[1] += int(is_done(j))
        for blk, (tot, dn) in sorted(by_block.items()):
            print(f"  {blk:<22} {dn:>4}/{tot:<4} complete")
        wall = total_sec / 3600
        print(f"\nSerial queue: {len(pending)} jobs pending, {wall:.1f} h of GPU work")
        print(f"\nEstimated wall clock: {wall:.1f} h  "
              f"(finishes about {(datetime.now()+timedelta(hours=wall)):%a %d %b %H:%M})")
        if a.plan and not a.run:
            return 0

    if not a.run:
        return 0

    state = {"total": len(jobs), "done": 0, "failed": 0, "skipped": 0,
             "elapsed": 0.0, "failures": []}
    lock = threading.Lock()

    # SERIAL. Measured: two concurrent jobs give 0.71x the throughput of running
    # them one after the other, and in the live sweep a two-lane split made both
    # jobs 2.6-3.7x slower, i.e. 2.2x worse overall. See the header of
    # 06_scripts/patch_sweep_serial.py for the measurements.
    #
    # Ordering is chosen so that ANY interruption leaves a usable result set:
    #   1. the two cheap blocks (CryoNuSeg main, ablation) finish inside ~4 h
    #   2. the expensive NuInsSeg block runs SEED-MAJOR, so after each seed
    #      there is a COMPLETE comparison across all models. Paired tests need
    #      N complete seeds for every model, not 5 seeds for some models.
    seed_rank = {s: i for i, s in enumerate(SEEDS)}

    def sort_key(j):
        block = {"main-cryonuseg": 0, "ablation-cryonuseg": 1,
                 "main-nuinsseg": 2}.get(j["block"], 3)
        if j["dataset"] == "nuinsseg":
            # seed first, then cheapest model within that seed
            return (block, seed_rank.get(j["seed"], 99), estimate_seconds(j, bench))
        # cheap blocks: fold, then seed, then cost
        return (block, j["fold"] or 0, seed_rank.get(j["seed"], 99),
                estimate_seconds(j, bench))

    ordered = sorted(jobs, key=sort_key)

    print(f"\nStarting sweep at {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  SERIAL, one job at a time, {len(ordered)} jobs, cheapest first")
    print(f"  logs: {LOGS}\n", flush=True)

    t0 = time.time()
    Lane("serial", ordered, state, lock).run()

    wall = time.time() - t0
    print(f"\nSweep finished in {wall/3600:.2f} h")
    print(f"  completed {state['done']}, skipped {state['skipped']}, failed {state['failed']}")
    if state["failures"]:
        print("  FAILED RUNS:")
        for f in state["failures"]:
            print(f"    {f}")
    (LOGS / "sweep_result.json").write_text(json.dumps(
        {"wall_hours": wall / 3600, **state}, indent=2), encoding="utf-8")
    return 1 if state["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
