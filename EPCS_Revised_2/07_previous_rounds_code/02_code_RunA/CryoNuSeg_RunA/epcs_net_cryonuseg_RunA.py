#!/usr/bin/env python3
"""Run A -- EPCS-Net (proposed) on CryoNuSeg.

The proposed architecture. Frozen SAM ViT-B priors are injected at all four
encoder scales (Pyramid-Scale Prior Injection, PSPI) and fused by the Adaptive
Cross-stream Fusion Block (ACFB): bidirectional gating, dual-kernel 3x3/5x5
refinement, squeeze-and-excitation, residual + GroupNorm.

This is the only model trained with the FTAS objective
(0.3 BCE + 0.7 focal-Tversky, alpha=0.7 beta=0.3 gamma=0.75); the shared runner
selects it automatically from MODEL_ID.

The shared protocol (split, augmentation, metrics, checkpoint selection,
reporting) is imported from ``epcs_protocol_RunA.py`` so that this model is
trained and scored identically to the other five. Only the architecture below
and, for EPCS-Net, the objective, differ between the six scripts.

Usage (from this folder):

    python epcs_net_cryonuseg_RunA.py                     # full 50-epoch Run A
    python epcs_net_cryonuseg_RunA.py --epochs 5          # first 5 epochs, resumable
    python epcs_net_cryonuseg_RunA.py --preflight         # wiring check, no training
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
    SAMImageEncoder,
    run_single_model,
)

MODEL_ID = "epcs_net"
DATASET = "cryonuseg"


class XGatedFusion(nn.Module):
    """Adaptive Cross-stream Fusion Block (ACFB), paper section 4.6.

    Step 1 channel projection, step 2 bidirectional dual-stream gating,
    step 3 dual-kernel spatial refinement (3x3 + 5x5), step 4 squeeze-and-
    excitation, step 5 residual + GroupNorm.

    Note: the paper's equation (14) places the residual around the DKSR+SE
    pair (``GroupNorm(W_out * x_att + x_cross)``); this implementation, carried
    over unchanged from the original EPCS notebook so the published numbers stay
    reproducible, adds the post-DKSR tensor instead. Flagged rather than
    silently changed.
    """

    def __init__(self, c_x, c_s=256):
        super().__init__()
        self.proj_s = nn.Conv2d(c_s, c_x, 1)
        self.proj_x = nn.Conv2d(c_x, c_x, 1)
        self.f_gate, self.s_gate = nn.Conv2d(c_x, c_x, 1), nn.Conv2d(c_x, c_x, 1)
        self.conv3 = nn.Conv2d(c_x, c_x, 3, padding=1)
        self.conv5 = nn.Conv2d(c_x, c_x, 5, padding=2)
        hidden = max(c_x // 4, 8)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c_x, hidden, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden, c_x, 1), nn.Sigmoid())
        self.out = nn.Conv2d(c_x, c_x, 1)
        self.norm = nn.GroupNorm(8 if c_x >= 8 else 1, c_x)

    def forward(self, x, s):
        x, s = self.proj_x(x), self.proj_s(s)
        fused = x * torch.sigmoid(self.s_gate(s)) + s * torch.sigmoid(self.f_gate(x))
        multi = self.conv3(fused) + self.conv5(fused)
        return self.norm(self.out(multi * self.se(multi)) + multi)


class EPCSNet(nn.Module):
    """eU-Net3+ style encoder with four-scale X-gated SAM fusion (PSPI + ACFB)
    and a full-scale fused decoder."""

    def __init__(self, checkpoint: str, base: int = 64):
        super().__init__()
        b = base
        self.sam = SAMImageEncoder(checkpoint)
        self.pool = nn.MaxPool2d(2)
        self.e1,self.e2,self.e3,self.e4 = ConvBlock(3,b),ConvBlock(b,2*b),ConvBlock(2*b,4*b),ConvBlock(4*b,8*b)
        self.f1,self.f2,self.f3,self.f4 = XGatedFusion(b),XGatedFusion(2*b),XGatedFusion(4*b),XGatedFusion(8*b)
        self.d4 = ConvBlock(8*b,8*b)
        self.d3 = ConvBlock(b+2*b+4*b+8*b,4*b)
        self.d2 = ConvBlock(b+2*b+4*b,2*b)
        self.d1 = ConvBlock(b+2*b,b)
        self.head = nn.Conv2d(b,1,1)

    def forward(self, x):
        sam=self.sam(x); e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))
        # PSPI: the single SAM embedding is resized to all four encoder scales.
        f1=self.f1(e1,F.interpolate(sam,size=e1.shape[-2:],mode="bilinear",align_corners=False))
        f2=self.f2(e2,F.interpolate(sam,size=e2.shape[-2:],mode="bilinear",align_corners=False))
        f3=self.f3(e3,F.interpolate(sam,size=e3.shape[-2:],mode="bilinear",align_corners=False))
        f4=self.f4(e4,F.interpolate(sam,size=e4.shape[-2:],mode="bilinear",align_corners=False))
        d4=self.d4(f4)
        d3=self.d3(torch.cat([F.max_pool2d(f1,4),F.max_pool2d(f2,2),f3,F.interpolate(d4,size=f3.shape[-2:],mode="bilinear",align_corners=False)],1))
        d2=self.d2(torch.cat([F.max_pool2d(f1,2),f2,F.interpolate(d3,size=f2.shape[-2:],mode="bilinear",align_corners=False)],1))
        d1=self.d1(torch.cat([f1,F.interpolate(d2,size=f1.shape[-2:],mode="bilinear",align_corners=False)],1))
        return self.head(d1)

def build(sam_checkpoint: str | None = None) -> nn.Module:
    """Factory handed to the shared runner. Requires the SAM ViT-B checkpoint."""
    if not sam_checkpoint or not Path(sam_checkpoint).is_file():
        raise FileNotFoundError(f"{MODEL_ID} requires --sam-checkpoint pointing at SAM ViT-B weights")
    return EPCSNet(sam_checkpoint)


if __name__ == "__main__":
    run_single_model(MODEL_ID, build, DATASET)
