"""
Measure the ACTUAL slowdown from running N training jobs concurrently on one
GPU, instead of assuming it. This decides the sweep's wall-clock schedule, and
it is also the direct evidence for why the previously published latency numbers
disagreed by 2.6x for the same model on the same images.

Usage:
    python probe_concurrency.py worker <model> <out_json>   # internal
    python probe_concurrency.py                             # run the probe
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from models_r3 import build_model, MODELS                       # noqa: E402
from protocol_r3 import FTASLoss, dice_loss, seed_everything    # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "04_results" / "logs"
WARMUP, STEPS = 5, 40
PROBE_MODEL = "epcs_net"
LEVELS = (1, 2, 3, 4)


def worker(model_id: str, out_path: str) -> int:
    device = torch.device("cuda:0")
    seed_everything(42)
    model = build_model(model_id).to(device)
    criterion = FTASLoss().to(device) if MODELS[model_id]["loss"] == "ftas" else None
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
    x = torch.rand(1, 3, 256, 256, device=device)
    y = (torch.rand(1, 1, 256, 256, device=device) > 0.5).float()

    def step():
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = (criterion(logits, y) if criterion
                else F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y))
        loss.backward()
        opt.step()

    model.train()
    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(STEPS):
        step()
    torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - t0) / STEPS

    Path(out_path).write_text(json.dumps({
        "model": model_id, "train_step_s": elapsed,
        "peak_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2)}), encoding="utf-8")
    return 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    me = Path(__file__).resolve()
    results = {}

    print(f"Concurrency probe: {PROBE_MODEL}, {STEPS} timed steps after {WARMUP} warm-up\n")
    print(f"{'jobs':>5}{'step s (mean)':>16}{'slowdown':>11}{'throughput':>13}{'peak MiB total':>17}")

    baseline = None
    for n in LEVELS:
        tmp = [OUT / f"_probe_{n}_{i}.json" for i in range(n)]
        procs = [subprocess.Popen([sys.executable, str(me), "worker", PROBE_MODEL, str(t)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                 for t in tmp]
        codes = [p.wait() for p in procs]
        if any(c != 0 for c in codes):
            print(f"{n:>5}   one or more workers failed (exit codes {codes})")
            for t in tmp:
                t.unlink(missing_ok=True)
            continue

        times, mem = [], []
        for t in tmp:
            d = json.loads(t.read_text(encoding="utf-8"))
            times.append(d["train_step_s"])
            mem.append(d["peak_mib"])
            t.unlink(missing_ok=True)

        mean = sum(times) / len(times)
        if baseline is None:
            baseline = mean
        slow = mean / baseline
        throughput = n / slow
        results[n] = {"mean_step_s": mean, "slowdown": slow, "throughput_x": throughput,
                      "per_worker_step_s": times, "peak_mib_total": sum(mem)}
        print(f"{n:>5}{mean:>16.4f}{slow:>11.2f}x{throughput:>12.2f}x{sum(mem):>17.0f}")

    if results:
        best = max(results, key=lambda k: results[k]["throughput_x"])
        results["recommended_concurrency"] = best
        print(f"\nBest throughput at {best} concurrent jobs "
              f"({results[best]['throughput_x']:.2f}x a single job)")
        (OUT / "concurrency_probe.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {OUT / 'concurrency_probe.json'}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        sys.exit(worker(sys.argv[2], sys.argv[3]))
    sys.exit(main())
