"""
Run the whole revision-3 analysis chain in dependency order.

Safe to re-run at any time. Steps that need results not yet produced are
reported as skipped rather than crashing the chain, so this can be run while
the sweep is still going and again when it finishes.

Order and why
-------------
1. stats_r3            aggregate every run, paired tests, bootstrap intervals,
                       per-organ breakdown. Everything else reads its outputs.
2. latency_r3          needs an IDLE GPU, so it is skipped automatically while
                       the sweep is running and must be run afterwards.
3. gradcam_r3          needs a completed EPCS-Net run per dataset.
4. dpnp_r3             needs saved test predictions.
5. instances_r3        needs NuInsSeg test predictions and the instance masks.
6. build_paper_tables  consumes all of the above.

Usage:
    python run_all_analysis.py              # everything that can run now
    python run_all_analysis.py --final      # after the sweep, includes latency
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

R3 = Path(__file__).resolve().parents[2]
CODE = R3 / "03_code"
RESULTS = R3 / "04_results"
SPLITS = RESULTS / "splits"
LOGS = RESULTS / "logs"


def run(label: str, args: list[str], cwd: Path) -> tuple[bool, str]:
    print(f"\n{'='*78}\n{label}\n{'='*78}", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable] + args, cwd=str(cwd),
                          capture_output=True, text=True)
    out = proc.stdout
    tail = "\n".join(l for l in out.splitlines()
                     if "Warning" not in l and "WARN" not in l)[-2500:]
    print(tail)
    if proc.returncode != 0:
        err = "\n".join(proc.stderr.splitlines()[-15:])
        print(f"  FAILED (exit {proc.returncode})\n{err}")
    print(f"  [{time.time()-t0:.1f}s]")
    return proc.returncode == 0, out


def gpu_busy() -> bool:
    """Same utilisation/memory check as latency_r3 (process-name check alone is unreliable on WDDM)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
    from latency_r3 import gpu_is_busy
    return gpu_is_busy()[0]


def newest_epcs_run(dataset: str) -> Path | None:
    base = RESULTS / ("nuinsseg" if dataset == "nuinsseg" else "cryonuseg")
    cands = [p.parent for p in (base / dataset / "epcs_net").rglob("summary.json")] \
        if (base / dataset / "epcs_net").is_dir() else []
    cands = [c for c in cands if (c / "test_predictions").is_dir()
             and (c / "best_model.pt").is_file() and (c / "per_image_test.csv").is_file()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def manifest_for(run_dir: Path, dataset: str) -> Path:
    """Split manifest the run was actually trained/tested on (CryoNuSeg has 3 folds)."""
    if dataset == "nuinsseg":
        return SPLITS / "nuinsseg_R3_stratified_70_15_15.csv"
    fold = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")).get("fold", 1)
    return SPLITS / f"cryonuseg_R3_fold{fold}.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true",
                    help="the sweep has finished; include latency measurement")
    a = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    report = {"started": datetime.now().isoformat(timespec="seconds"), "steps": []}

    def record(name, ok, note=""):
        report["steps"].append({"step": name, "ok": ok, "note": note})

    ok, _ = run("1. Aggregate runs, paired tests, bootstrap intervals",
                [str(CODE / "analysis" / "stats_r3.py")], CODE / "analysis")
    record("stats_r3", ok)

    if a.final:
        if gpu_busy():
            print("\n2. Latency SKIPPED: the GPU is not idle. Measuring under "
                  "contention is exactly the error this revision corrects.")
            record("latency_r3", False, "GPU busy")
        else:
            ok, _ = run("2. Controlled latency measurement (idle GPU, with warm-up)",
                        [str(CODE / "analysis" / "latency_r3.py")], CODE / "analysis")
            record("latency_r3", ok)
    else:
        print("\n2. Latency skipped (pass --final once the sweep has finished).")
        record("latency_r3", False, "deferred")

    for ds in ("cryonuseg", "nuinsseg"):
        rd = newest_epcs_run(ds)
        if rd is None:
            print(f"\n3. Grad-CAM {ds} SKIPPED: no completed EPCS-Net run yet.")
            record(f"gradcam_{ds}", False, "no completed run")
            continue
        ok, _ = run(f"3. Grad-CAM and failure panel -- {ds}",
                    [str(CODE / "analysis" / "gradcam_r3.py"), "--run", str(rd)],
                    CODE / "analysis")
        record(f"gradcam_{ds}", ok)

    for ds in ("cryonuseg", "nuinsseg"):
        rd = newest_epcs_run(ds)
        if rd is None:
            print(f"\n4. DPNP {ds} SKIPPED: no completed EPCS-Net run yet.")
            record(f"dpnp_{ds}", False, "no completed run")
            continue
        ok, _ = run(f"4. DPNP clustering, loadings, stability, confounds -- {ds}",
                    [str(CODE / "analysis" / "dpnp_r3.py"),
                     "--predictions", str(rd / "test_predictions"),
                     "--split", str(manifest_for(rd, ds)),
                     "--only-split", "test", "--tag", f"r3_{ds}"], CODE / "analysis")
        record(f"dpnp_{ds}", ok)

    rd = newest_epcs_run("nuinsseg")
    if rd is None:
        print("\n5. Component-vs-instance SKIPPED: no completed NuInsSeg EPCS-Net run.")
        record("instances_r3", False, "no completed run")
    else:
        ok, _ = run("5. Predicted components vs ground-truth instances",
                    [str(CODE / "analysis" / "instances_r3.py"),
                     "--predictions", str(rd / "test_predictions"),
                     "--split", str(SPLITS / "nuinsseg_R3_stratified_70_15_15.csv"),
                     "--tag", "r3"], CODE / "analysis")
        record("instances_r3", ok)

    ok, _ = run("6. Build paper-ready tables (xlsx + LaTeX)",
                [str(CODE / "tables" / "build_paper_tables.py")], CODE / "tables")
    record("build_paper_tables", ok)

    report["finished"] = datetime.now().isoformat(timespec="seconds")
    (LOGS / "analysis_chain_report.json").write_text(json.dumps(report, indent=2),
                                                     encoding="utf-8")
    done = sum(1 for s in report["steps"] if s["ok"])
    print(f"\n{'='*78}\nAnalysis chain: {done}/{len(report['steps'])} steps completed")
    for s in report["steps"]:
        mark = "ok  " if s["ok"] else "SKIP"
        print(f"  {mark} {s['step']}" + (f"   ({s['note']})" if s["note"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
