#!/usr/bin/env python3
"""Run B -- EPCS-Net on CryoNuSeg, five seeds.

Trains EPCS-Net five times (seeds 42, 123, 456, 789, 2024), 50 epochs each,
on the exact same 24/6 split Run A used -- see epcs_protocol_RunB.py for why
the split is held fixed while only the seed varies. Reports mean +/- SD across
the five runs.

Usage:

    python epcs_net_cryonuseg_RunB.py                  # all five seeds, 50 epochs each
    python epcs_net_cryonuseg_RunB.py --epochs 5        # first 5 epochs of every seed, resumable
    python epcs_net_cryonuseg_RunB.py --seeds 42 123     # only these seeds
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from epcs_protocol_RunB import SEEDS, run_multi_seed  # noqa: E402

DATASET = "cryonuseg"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help=f"default: {SEEDS}")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    run_multi_seed(DATASET, args.epochs, seeds=args.seeds, no_resume=args.no_resume)


if __name__ == "__main__":
    main()
