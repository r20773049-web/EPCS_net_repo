#!/usr/bin/env python3
"""Run A -- SAM-guided baseline on CryoNuSeg.

Frozen SAM ViT-B fused with the encoder bottleneck only -- no PSPI, no
multi-scale injection. Isolates what plain SAM context contributes.

The shared protocol (split, augmentation, metrics, checkpoint selection,
reporting) is imported from ``epcs_protocol_RunA.py`` so that this model is
trained and scored identically to the other five. Only the architecture below
and, for EPCS-Net, the objective, differ between the six scripts.

Usage (from this folder):

    python sam_unet_cryonuseg_RunA.py                     # full 50-epoch Run A
    python sam_unet_cryonuseg_RunA.py --epochs 5          # first 5 epochs, resumable
    python sam_unet_cryonuseg_RunA.py --preflight         # wiring check, no training
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

MODEL_ID = "sam_unet"
DATASET = "cryonuseg"


class SAMBottleneckFusion(nn.Module):
    """Single-scale gated fusion at the bottleneck. This is the ablation
    control: SAM context arrives only at the coarsest resolution."""

    def __init__(self, sam_ch=256, unet_ch=512, out_ch=512):
        super().__init__()
        self.sam_proj, self.unet_proj = nn.Conv2d(sam_ch,256,1), nn.Conv2d(unet_ch,256,1)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(512,256,1),nn.ReLU(inplace=True),nn.Conv2d(256,512,1),nn.Sigmoid())
        self.out = nn.Conv2d(512,out_ch,1)

    def forward(self, sam, unet):
        fused = torch.cat([self.sam_proj(sam), self.unet_proj(unet)], 1)
        return self.out(fused * self.gate(fused))


class SAMUNet(nn.Module):
    """U-Net encoder-decoder with frozen SAM context injected at the bottleneck."""

    def __init__(self, checkpoint: str):
        super().__init__()
        self.sam = SAMImageEncoder(checkpoint)
        self.pool = nn.MaxPool2d(2)
        self.e1,self.e2,self.e3,self.e4 = ConvBlock(3,64),ConvBlock(64,128),ConvBlock(128,256),ConvBlock(256,512)
        self.fuse = SAMBottleneckFusion()
        self.d3, self.d2, self.d1 = ConvBlock(512+256,256), ConvBlock(256+128,128), ConvBlock(128+64,64)
        self.head = nn.Conv2d(64,1,1)

    def forward(self, x):
        sam=self.sam(x); e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))
        sam=F.interpolate(sam,size=e4.shape[-2:],mode="bilinear",align_corners=False)
        y=self.fuse(sam,e4)
        y=self.d3(torch.cat([F.interpolate(y,size=e3.shape[-2:],mode="bilinear",align_corners=False),e3],1))
        y=self.d2(torch.cat([F.interpolate(y,size=e2.shape[-2:],mode="bilinear",align_corners=False),e2],1))
        y=self.d1(torch.cat([F.interpolate(y,size=e1.shape[-2:],mode="bilinear",align_corners=False),e1],1))
        return self.head(y)

def build(sam_checkpoint: str | None = None) -> nn.Module:
    """Factory handed to the shared runner. Requires the SAM ViT-B checkpoint."""
    if not sam_checkpoint or not Path(sam_checkpoint).is_file():
        raise FileNotFoundError(f"{MODEL_ID} requires --sam-checkpoint pointing at SAM ViT-B weights")
    return SAMUNet(sam_checkpoint)


if __name__ == "__main__":
    run_single_model(MODEL_ID, build, DATASET)
