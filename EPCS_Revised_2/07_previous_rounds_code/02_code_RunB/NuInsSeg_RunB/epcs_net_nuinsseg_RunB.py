#!/usr/bin/env python3
"""Run B -- EPCS-Net on NuInsSeg, five seeds.

Trains EPCS-Net five times (seeds 42, 123, 456, 789, 2024), 50 epochs each,
on the exact same 532/133 split Run A used -- see epcs_protocol_RunB.py for
why the split is held fixed while only the seed varies. Reports mean +/- SD
across the five runs.

NuInsSeg is much larger than CryoNuSeg (532 vs 24 training images); one seed
at 50 epochs took roughly 1.5-2.5 hours in Run A. Five seeds is a multi-hour
job -- run in batches of a few epochs at a time if there's a session/time
limit; every epoch is checkpointed and resumable.

Usage:

    python epcs_net_nuinsseg_RunB.py                   # all five seeds, 50 epochs each
    python epcs_net_nuinsseg_RunB.py --epochs 5         # first 5 epochs of every seed, resumable
    python epcs_net_nuinsseg_RunB.py --seeds 42 123      # only these seeds
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from epcs_protocol_RunB import SEEDS, run_multi_seed  # noqa: E402

DATASET = "nuinsseg"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help=f"default: {SEEDS}")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    run_multi_seed(DATASET, args.epochs, seeds=args.seeds, no_resume=args.no_resume)


if __name__ == "__main__":
    main()
