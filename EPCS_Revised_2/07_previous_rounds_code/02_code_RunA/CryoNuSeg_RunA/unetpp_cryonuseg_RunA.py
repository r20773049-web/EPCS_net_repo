#!/usr/bin/env python3
"""Run A -- U-Net++ on CryoNuSeg.

Nested dense skip pathways for progressive feature refinement
(Zhou et al., 2018).

The shared protocol (split, augmentation, metrics, checkpoint selection,
reporting) is imported from ``epcs_protocol_RunA.py`` so that this model is
trained and scored identically to the other five. Only the architecture below
and, for EPCS-Net, the objective, differ between the six scripts.

Usage (from this folder):

    python unetpp_cryonuseg_RunA.py                     # full 50-epoch Run A
    python unetpp_cryonuseg_RunA.py --epochs 5          # first 5 epochs, resumable
    python unetpp_cryonuseg_RunA.py --preflight         # wiring check, no training
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
    run_single_model,
)

MODEL_ID = "unetpp"
DATASET = "cryonuseg"


class UNetPP(nn.Module):
    """Nested dense skip pathways: every decoder node sees all shallower nodes
    at its own resolution plus the upsampled node from one level deeper."""

    def __init__(self, base: int = 16):
        super().__init__()
        b, p = base, nn.MaxPool2d(2)
        self.p = p
        self.x00, self.x10, self.x20, self.x30, self.x40 = ConvBlock(3,b), ConvBlock(b,2*b), ConvBlock(2*b,4*b), ConvBlock(4*b,8*b), ConvBlock(8*b,16*b)
        self.x01, self.x11, self.x21, self.x31 = ConvBlock(b+2*b,b), ConvBlock(2*b+4*b,2*b), ConvBlock(4*b+8*b,4*b), ConvBlock(8*b+16*b,8*b)
        self.x02, self.x12, self.x22 = ConvBlock(4*b,b), ConvBlock(8*b,2*b), ConvBlock(16*b,4*b)
        self.x03, self.x13 = ConvBlock(5*b,b), ConvBlock(10*b,2*b)
        self.x04, self.head = ConvBlock(6*b,b), nn.Conv2d(b,1,1)

    @staticmethod
    def up(x, target):
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        x00=self.x00(x); x10=self.x10(self.p(x00)); x20=self.x20(self.p(x10)); x30=self.x30(self.p(x20)); x40=self.x40(self.p(x30))
        x01=self.x01(torch.cat([x00,self.up(x10,x00)],1)); x11=self.x11(torch.cat([x10,self.up(x20,x10)],1)); x21=self.x21(torch.cat([x20,self.up(x30,x20)],1)); x31=self.x31(torch.cat([x30,self.up(x40,x30)],1))
        x02=self.x02(torch.cat([x00,x01,self.up(x11,x00)],1)); x12=self.x12(torch.cat([x10,x11,self.up(x21,x10)],1)); x22=self.x22(torch.cat([x20,x21,self.up(x31,x20)],1))
        x03=self.x03(torch.cat([x00,x01,x02,self.up(x12,x00)],1)); x13=self.x13(torch.cat([x10,x11,x12,self.up(x22,x10)],1))
        x04=self.x04(torch.cat([x00,x01,x02,x03,self.up(x13,x00)],1))
        return self.head(x04)

def build(sam_checkpoint: str | None = None) -> nn.Module:
    """Factory handed to the shared runner. U-Net++ does not use SAM."""
    return UNetPP()


if __name__ == "__main__":
    run_single_model(MODEL_ID, build, DATASET)
