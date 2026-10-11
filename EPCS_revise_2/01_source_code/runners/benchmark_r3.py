"""
Measure real per-step training cost and per-image inference cost for every
model, so the sweep schedule is derived from measurement rather than guesswork.

Image size is identical for both datasets (256x256), so step cost is
dataset-independent; only the number of steps differs. That lets one benchmark
per model drive the estimate for every run in the matrix.

Writes 04_results/logs/benchmark_r3.json and prints a schedule table.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from models_r3 import MODELS, build_model                      # noqa: E402
from protocol_r3 import FTASLoss, dice_loss, seed_everything   # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "04_results" / "logs"

WARMUP = 3
STEPS = 10
EPOCHS = 50

SPLIT_SHAPE = {
    "nuinsseg": dict(train=459, val=103, test=103, folds=1),
    "cryonuseg": dict(train=15, val=5, test=10, folds=3),
}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench_model(model_id: str, device) -> dict:
    seed_everything(42)
    model = build_model(model_id).to(device)
    spec = MODELS[model_id]
    criterion = FTASLoss().to(device) if spec["loss"] == "ftas" else None
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)

    x = torch.rand(1, 3, 256, 256, device=device)
    y = (torch.rand(1, 1, 256, 256, device=device) > 0.5).float()

    model.train()
    for _ in range(WARMUP):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = (criterion(logits, y) if criterion
                else F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y))
        loss.backward()
        opt.step()
    sync(device)

    t0 = time.perf_counter()
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = (criterion(logits, y) if criterion
                else F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y))
        loss.backward()
        opt.step()
    sync(device)
    train_step_s = (time.perf_counter() - t0) / STEPS

    model.eval()
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(STEPS):
            model(x)
        sync(device)
        infer_s = (time.perf_counter() - t0) / STEPS

    peak = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda" else None)
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    return {"train_step_s": train_step_s, "infer_s": infer_s, "peak_mib": peak}


def estimate(model_id: str, dataset: str, b: dict) -> float:
    """Wall seconds for one complete run (50 epochs + final test scoring)."""
    s = SPLIT_SHAPE[dataset]
    per_epoch = s["train"] * b["train_step_s"] + s["val"] * b["infer_s"]
    final = (s["val"] + s["test"]) * b["infer_s"]
    return EPOCHS * per_epoch + final


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on {torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'}")
    print(f"{WARMUP} warm-up steps then {STEPS} timed steps per model\n")

    available = [m for m in sorted(MODELS) if not m.startswith("ref21_")]
    results = {}
    print(f"{'model':<20}{'train step s':>14}{'infer s':>10}{'peak MiB':>11}")
    for m in available:
        try:
            b = bench_model(m, device)
        except Exception as exc:
            print(f"{m:<20}  SKIPPED: {type(exc).__name__}: {exc}")
            continue
        results[m] = b
        print(f"{m:<20}{b['train_step_s']:>14.4f}{b['infer_s']:>10.4f}{b['peak_mib']:>11.0f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "benchmark_r3.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'='*78}\nESTIMATED RUN TIME (minutes per single run of 50 epochs)\n{'='*78}")
    print(f"{'model':<20}{'cryonuseg':>12}{'nuinsseg':>12}")
    for m, b in results.items():
        print(f"{m:<20}{estimate(m,'cryonuseg',b)/60:>12.1f}{estimate(m,'nuinsseg',b)/60:>12.1f}")

    print(f"\n{'='*78}\nFULL SWEEP BUDGET (serial GPU hours)\n{'='*78}")
    seeds = 5
    baseline = ["unet", "unetpp", "unet3plus", "eunet3plus", "sam_unet", "epcs_net"]
    ablation = [m for m in results if m.startswith("abl")]

    total = 0.0
    blocks = [
        ("NuInsSeg  6 published models x 5 seeds", baseline, "nuinsseg", seeds, 1),
        ("CryoNuSeg 6 published models x 5 seeds x 3 folds", baseline, "cryonuseg", seeds, 3),
        ("CryoNuSeg ablation 10 variants x 5 seeds (fold 1)", ablation, "cryonuseg", seeds, 1),
    ]
    for label, models, ds, n_seed, n_fold in blocks:
        hours = sum(estimate(m, ds, results[m]) for m in models if m in results) * n_seed * n_fold / 3600
        runs = len([m for m in models if m in results]) * n_seed * n_fold
        total += hours
        print(f"{label:<52}{runs:>5} runs{hours:>9.1f} h")
    print(f"{'ref21 variants (2 models, both datasets) -- pending port':<52}{'?':>5} runs{9.0:>9.1f} h")
    print(f"{'-'*78}")
    print(f"{'MEASURED SUBTOTAL (serial, excluding ref21)':<58}{total:>9.1f} h")
    print(f"{'WITH ref21 estimate':<58}{total+9:>9.1f} h")
    for n in (2, 3, 4):
        slow = 1 + 0.35 * (n - 1)
        print(f"  at {n} concurrent jobs (assuming {slow:.2f}x slowdown each): "
              f"{(total+9) * slow / n:>5.1f} h wall clock")
    return 0


if __name__ == "__main__":
    sys.exit(main())
