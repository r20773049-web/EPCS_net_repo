#!/usr/bin/env python3
"""DPNP -- Grad-CAM, PCA nuclear phenotyping, HDBSCAN clustering, on CryoNuSeg.

Runs on EPCS-Net's best Run A checkpoint. Requires Run A's CryoNuSeg results
(checkpoint + predictions) to already exist.

Usage:
    python run_dpnp_cryonuseg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from dpnp_pipeline import run_dpnp  # noqa: E402

if __name__ == "__main__":
    run_dpnp("cryonuseg")
