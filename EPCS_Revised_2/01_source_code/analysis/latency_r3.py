"""
Controlled inference-latency measurement.

Why this exists
---------------
Reviewer 3 minor comment 2 and Reviewer 4 comment 4 both ask why the published
per-image latency differs so much between datasets. The audit found the real
answer, and it is not a dataset effect:

  * Timing was taken at the end of a training run, with no warm-up iterations.
  * Runs overlapped on one GPU. Our own concurrency probe measured a 2.77x
    slowdown for two concurrent EPCS-Net jobs, which almost exactly accounts
    for the 419 vs 160 ms/image discrepancy recorded for the SAME model on the
    SAME 133 images.

This module therefore measures latency properly and says exactly how:

  * exclusive GPU (the caller must ensure nothing else is running; this script
    checks and refuses otherwise unless --force is passed)
  * fixed warm-up iterations, discarded
  * explicit torch.cuda.synchronize around the timed region
  * data loading EXCLUDED: the input tensor is resident on the GPU before timing
  * SAM embeddings recomputed live, NOT read from the cache, because the cache
    is a training-throughput optimisation and would understate deployment cost
  * batch size 1, 256x256, FP32, eval mode, no_grad
  * hardware, driver, CUDA and PyTorch versions recorded

Reports median and interquartile range as well as the mean, because the mean of
a latency distribution with a cold first call is exactly what went wrong before.

Usage:
    python latency_r3.py                # all models
    python latency_r3.py --models epcs_net unet
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from models_r3 import MODELS, build_model, count_params         # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "04_results" / "latency"
WARMUP = 20
TIMED = 100


def gpu_is_busy() -> tuple[bool, str]:
    """Refuse to measure on a GPU that something else is using.

    Audit fix: under Windows WDDM nvidia-smi often lists no compute apps even when the
    GPU is busy, so the process-name check alone can wrongly report idle. Use utilisation
    and memory over several samples too, and fail CLOSED (busy) if nvidia-smi cannot be read.
    """
    import time
    try:
        utils, mems = [], []
        for _ in range(5):
            res = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
            u, m = [float(x) for x in res.stdout.strip().splitlines()[0].split(",")]
            utils.append(u); mems.append(m); time.sleep(1.0)
    except Exception as exc:
        return True, f"could not read nvidia-smi ({exc}); treating GPU as busy"
    busy = max(utils) > 10 or max(mems) > 2500
    return busy, f"utilisation max {max(utils):.0f}% , memory max {max(mems):.0f} MiB over 5 samples"


def measure(model_id: str, device: torch.device) -> dict:
    model = build_model(model_id).to(device).eval()
    total, trainable = count_params(model)
    x = torch.rand(1, 3, 256, 256, device=device)      # already resident: loading excluded

    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        torch.cuda.synchronize(device) if device.type == "cuda" else None

        samples = []
        for _ in range(TIMED):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(samples)
    peak = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda" else None)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    return {
        "model": model_id,
        "label": MODELS[model_id]["label"],
        "parameters_total": total,
        "parameters_trainable": trainable,
        "ms_mean": float(arr.mean()),
        "ms_sd": float(arr.std(ddof=1)),
        "ms_median": float(np.median(arr)),
        "ms_p25": float(np.percentile(arr, 25)),
        "ms_p75": float(np.percentile(arr, 75)),
        "ms_min": float(arr.min()),
        "ms_max": float(arr.max()),
        "peak_vram_mib": peak,
        "images_per_second": float(1000.0 / arr.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--force", action="store_true",
                    help="measure even if the GPU is not idle (results will be invalid)")
    a = ap.parse_args()

    busy, detail = gpu_is_busy()
    if busy and not a.force:
        print("REFUSING TO MEASURE: another compute process is using the GPU.")
        print(f"  {detail}")
        print("  Latency measured under contention is exactly the error this "
              "revision is correcting. Wait for the sweep to finish, or pass --force.")
        return 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ids = a.models or [m for m in MODELS if not m.startswith("abl")]

    print(f"Controlled latency measurement  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  device       : {torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'}")
    print(f"  protocol     : {WARMUP} warm-up discarded, {TIMED} timed iterations")
    print(f"  input        : 1 x 3 x 256 x 256, FP32, resident on device before timing")
    print(f"  boundaries   : torch.cuda.synchronize either side of the forward pass only")
    print(f"  excluded     : data loading, preprocessing, metric computation, mask writing")
    print(f"  SAM          : recomputed live (the training cache is NOT used here)")
    print(f"  GPU state    : {detail}\n")

    print(f"{'model':<20}{'mean ms':>10}{'median':>9}{'IQR':>16}{'img/s':>9}{'VRAM MiB':>11}")
    rows = []
    for m in model_ids:
        try:
            r = measure(m, device)
        except Exception as exc:
            print(f"{m:<20}  FAILED: {type(exc).__name__}: {exc}")
            continue
        rows.append(r)
        print(f"{m:<20}{r['ms_mean']:>10.2f}{r['ms_median']:>9.2f}"
              f"{r['ms_p25']:>8.2f}-{r['ms_p75']:<7.2f}{r['images_per_second']:>9.1f}"
              f"{r['peak_vram_mib']:>11.0f}")

    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "gpu_state_at_measurement": detail,
        "warmup_iterations": WARMUP,
        "timed_iterations": TIMED,
        "input_shape": [1, 3, 256, 256],
        "precision": "FP32",
        "batch_size": 1,
        "timing_boundary": "torch.cuda.synchronize immediately before and after the forward pass",
        "excludes": ["data loading", "preprocessing", "metric computation", "mask writing"],
        "sam_embeddings": "recomputed live; the training-time cache is not used",
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "platform": platform.platform(),
        "results": rows,
    }
    (OUT / "latency_r3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / "latency_r3.csv", index=False)
    print(f"\nWrote {OUT / 'latency_r3.json'} and latency_r3.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
