#!/usr/bin/env python3
"""Run A -- U-Net on CryoNuSeg.

The original skip-connection encoder-decoder (Ronneberger et al., 2015).
Sets the CNN reference point for the comparison.

The shared protocol (split, augmentation, metrics, checkpoint selection,
reporting) is imported from ``epcs_protocol_RunA.py`` so that this model is
trained and scored identically to the other five. Only the architecture below
and, for EPCS-Net, the objective, differ between the six scripts.

Usage (from this folder):

    python unet_cryonuseg_RunA.py                     # full 50-epoch Run A
    python unet_cryonuseg_RunA.py --epochs 5          # first 5 epochs, resumable
    python unet_cryonuseg_RunA.py --preflight         # wiring check, no training
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
    Up,
    run_single_model,
)

MODEL_ID = "unet"
DATASET = "cryonuseg"


class UNet(nn.Module):
    """Five-stage encoder, four upsampling decoder stages with skip connections."""

    def __init__(self, base: int = 32):
        super().__init__()
        self.enc = Encoder(base)
        self.u4, self.u3 = Up(base * 16, base * 8, base * 8), Up(base * 8, base * 4, base * 4)
        self.u2, self.u1 = Up(base * 4, base * 2, base * 2), Up(base * 2, base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.enc(x)
        return self.head(self.u1(self.u2(self.u3(self.u4(e5, e4), e3), e2), e1))

def build(sam_checkpoint: str | None = None) -> nn.Module:
    """Factory handed to the shared runner. U-Net does not use SAM."""
    return UNet()


if __name__ == "__main__":
    run_single_model(MODEL_ID, build, DATASET)
