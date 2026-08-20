#!/usr/bin/env python3
"""Run A -- U-Net3+ on CryoNuSeg.

Full-scale skip connections: every decoder node aggregates all five
encoder scales at once (Huang et al., 2020).

The shared protocol (split, augmentation, metrics, checkpoint selection,
reporting) is imported from ``epcs_protocol_RunA.py`` so that this model is
trained and scored identically to the other five. Only the architecture below
and, for EPCS-Net, the objective, differ between the six scripts.

Usage (from this folder):

    python unet3plus_cryonuseg_RunA.py                     # full 50-epoch Run A
    python unet3plus_cryonuseg_RunA.py --epochs 5          # first 5 epochs, resumable
    python unet3plus_cryonuseg_RunA.py --preflight         # wiring check, no training
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from epcs_protocol_RunA import (  # noqa: E402
    ConvBlock,
    Encoder,
    FullScaleFuse,
    run_single_model,
)

MODEL_ID = "unet3plus"
DATASET = "cryonuseg"


class UNet3Plus(nn.Module):
    """Every decoder node fuses all five encoder scales -- deeper maps are
    upsampled, shallower maps are max-pooled down."""

    def __init__(self, base: int = 32, block=ConvBlock):
        super().__init__()
        b = base
        self.enc = Encoder(b, block)
        d = 5 * b
        self.d4 = FullScaleFuse([b,2*b,4*b,8*b,16*b], b)
        self.d3 = FullScaleFuse([b,2*b,4*b,d,16*b], b)
        self.d2 = FullScaleFuse([b,2*b,d,d,16*b], b)
        self.d1 = FullScaleFuse([b,d,d,d,16*b], b)
        self.head = nn.Conv2d(d, 1, 1)

    def forward(self, x):
        e1,e2,e3,e4,e5 = self.enc(x)
        d4=self.d4([e1,e2,e3,e4,e5],e4); d3=self.d3([e1,e2,e3,d4,e5],e3)
        d2=self.d2([e1,e2,d3,d4,e5],e2); d1=self.d1([e1,d2,d3,d4,e5],e1)
        return self.head(d1)

def build(sam_checkpoint: str | None = None) -> nn.Module:
    """Factory handed to the shared runner. U-Net3+ does not use SAM."""
    return UNet3Plus()


if __name__ == "__main__":
    run_single_model(MODEL_ID, build, DATASET)
