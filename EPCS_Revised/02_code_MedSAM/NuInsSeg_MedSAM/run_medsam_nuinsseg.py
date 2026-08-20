#!/usr/bin/env python3
"""MedSAM baseline on NuInsSeg -- frozen, zero-shot, no fine-tuning.

Scores the official MedSAM checkpoint (wanglab/medsam-vit-base) on the exact
same 532/133 validation split Run A and Run B use, via a whole-image box
prompt. See shared/medsam_baseline.py for the full rationale.

Requires Run A's NuInsSeg split to already exist (it reuses that manifest,
not a new one).

Usage:
    python run_medsam_nuinsseg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from medsam_baseline import run_medsam_dataset  # noqa: E402

if __name__ == "__main__":
    run_medsam_dataset("nuinsseg")
