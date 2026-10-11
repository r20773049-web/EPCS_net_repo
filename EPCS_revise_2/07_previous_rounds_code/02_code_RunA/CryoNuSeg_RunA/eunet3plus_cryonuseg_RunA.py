#!/usr/bin/env python3
"""Run A -- eU-Net3+ (no SAM) on CryoNuSeg.

The EPCS-Net backbone with SAM removed entirely -- isolates what the
CNN achieves on its own. Uses gated linear unit (GLU) blocks throughout.

The shared protocol (split, augmentation, metrics, checkpoint selection,
reporting) is imported from ``epcs_protocol_RunA.py`` so that this model is
trained and scored identically to the other five. Only the architecture below
and, for EPCS-Net, the objective, differ between the six scripts.

Usage (from this folder):

    python eunet3plus_cryonuseg_RunA.py                     # full 50-epoch Run A
    python eunet3plus_cryonuseg_RunA.py --epochs 5          # first 5 epochs, resumable
    python eunet3plus_cryonuseg_RunA.py --preflight         # wiring check, no training
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from epcs_protocol_RunA import (  # noqa: E402
    GLUBlock,
    run_single_model,
)

MODEL_ID = "eunet3plus"
DATASET = "cryonuseg"


class EUNet3Plus(nn.Module):
    """GLU encoder-decoder: the EPCS-Net backbone without any SAM guidance."""

    def __init__(self, base: int = 32):
        super().__init__()
        b = base
        self.pool = nn.MaxPool2d(2)
        self.e1,self.e2,self.e3,self.e4,self.center = GLUBlock(3,b),GLUBlock(b,2*b),GLUBlock(2*b,4*b),GLUBlock(4*b,8*b),GLUBlock(8*b,16*b)
        self.d4,self.d3,self.d2,self.d1 = GLUBlock(24*b,8*b),GLUBlock(12*b,4*b),GLUBlock(6*b,2*b),GLUBlock(3*b,b)
        self.head = nn.Conv2d(b,1,1)

    def forward(self, x):
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3)); center=self.center(self.pool(e4))
        d4=self.d4(torch.cat([F.interpolate(center,size=e4.shape[-2:],mode="bilinear",align_corners=False),e4],1))
        d3=self.d3(torch.cat([F.interpolate(d4,size=e3.shape[-2:],mode="bilinear",align_corners=False),e3],1))
        d2=self.d2(torch.cat([F.interpolate(d3,size=e2.shape[-2:],mode="bilinear",align_corners=False),e2],1))
        d1=self.d1(torch.cat([F.interpolate(d2,size=e1.shape[-2:],mode="bilinear",align_corners=False),e1],1))
        return self.head(d1)

def build(sam_checkpoint: str | None = None) -> nn.Module:
    """Factory handed to the shared runner. eU-Net3+ deliberately does not use SAM."""
    return EUNet3Plus()


if __name__ == "__main__":
    run_single_model(MODEL_ID, build, DATASET)
