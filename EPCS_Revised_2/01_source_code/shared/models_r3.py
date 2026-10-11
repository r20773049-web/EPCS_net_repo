"""
Revision-3 model zoo.

Self-contained so the revision-3 tree does not depend on the Run A tree, but the
six published architectures are reproduced BYTE-FOR-BYTE in structure. A unit
test in this file asserts that every parameter count matches the value recorded
in the Run A summary JSON files, so any accidental drift fails loudly.

Published models (unchanged):
    unet, unetpp, unet3plus, eunet3plus, sam_unet, epcs_net

EPCS-Net is additionally exposed parametrically so that every ablation the
reviewers asked for is the SAME code path with one switch flipped, rather than a
separately maintained variant that could drift:

    EPCSNet(sam_source=, injection=, fusion=, encoder_norm=)

      sam_source   'sam'        frozen SAM ViT-B  (default)
                   'random'     frozen randomly-initialised encoder of identical
                                output shape. Reviewer 3's control for whether
                                the gain comes from SAM's semantics or merely
                                from having a second input stream.
      injection    'pyramid'    PSPI, all four encoder scales (default)
                   'bottleneck' deepest scale only
      fusion       'acfb'       full ACFB: BDSG + DKSR + SE + residual + GN
                   'bdsg'       ACFB without the dual-kernel stage
                   'concat'     plain projection + concatenation + 1x1
                   'xgated21'   the reference-21 X-Gated Fusion Block
      encoder_norm 'bn'         BatchNorm2d in DoubleConv (default, as published)
                   'gn'         GroupNorm(8), Reviewer 1's batch-size-1 objection

Verified against the code: the frozen SAM encoder upsamples each 256x256 patch
to 1024x1024 and applies the official preprocess, so the embedding is
[B, 256, 64, 64]. The manuscript's "16x16 token grid" was wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
SAM_CHECKPOINT = ROOT / "05_reference" / "sam_vit_b_01ec64.pth"


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------
def _norm(kind: str, channels: int) -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    if kind == "gn":
        return nn.GroupNorm(8 if channels >= 8 else 1, channels)
    raise ValueError(f"unknown norm kind: {kind}")


class ConvBlock(nn.Module):
    """DoubleConv. Published form is Conv-BatchNorm-ReLU twice."""

    def __init__(self, in_ch: int, out_ch: int, norm: str = "bn"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), _norm(norm, out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), _norm(norm, out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class GLUBlock(nn.Module):
    """Gated convolution block used only by eU-Net3+."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.gate = nn.Conv2d(in_ch, out_ch * 2, 3, padding=1, bias=False)
        self.refine = nn.Sequential(
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU())

    def forward(self, x):
        return self.refine(F.glu(self.gate(x), dim=1))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.block = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class Encoder(nn.Module):
    def __init__(self, base: int, block=ConvBlock):
        super().__init__()
        self.e1, self.e2 = block(3, base), block(base, base * 2)
        self.e3, self.e4 = block(base * 2, base * 4), block(base * 4, base * 8)
        self.e5, self.pool = block(base * 8, base * 16), nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        e5 = self.e5(self.pool(e4))
        return e1, e2, e3, e4, e5


class FullScaleFuse(nn.Module):
    def __init__(self, channels: list[int], out_ch: int):
        super().__init__()
        self.proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
                          nn.ReLU(inplace=True)) for c in channels)
        self.fuse = ConvBlock(out_ch * len(channels), out_ch * len(channels))

    def forward(self, xs, target):
        resized = []
        for x, project in zip(xs, self.proj):
            if x.shape[-2:] != target.shape[-2:]:
                x = (F.adaptive_max_pool2d(x, target.shape[-2:])
                     if x.shape[-2] > target.shape[-2]
                     else F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False))
            resized.append(project(x))
        return self.fuse(torch.cat(resized, 1))


# --------------------------------------------------------------------------
# frozen context encoders
# --------------------------------------------------------------------------
class SAMImageEncoder(nn.Module):
    """Frozen SAM ViT-B. The 256x256 patch is upsampled to 1024x1024 and passed
    through the OFFICIAL preprocess, so the checkpoint is used unmodified and
    the embedding is [B, 256, 64, 64]."""

    def __init__(self, checkpoint: str):
        super().__init__()
        from segment_anything import sam_model_registry
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        for p in self.sam.parameters():
            p.requires_grad_(False)
        self.sam.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.sam.eval()
        return self

    def forward(self, x):
        x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False).mul(255.0)
        with torch.no_grad():
            return self.sam.image_encoder(self.sam.preprocess(x))


class RandomFrozenEncoder(nn.Module):
    """Reviewer 3's control: a frozen, randomly-initialised encoder producing an
    embedding of exactly the same shape as SAM's, [B, 256, 64, 64]. It supplies
    an extra input stream with no pretrained semantic content, which isolates
    whether EPCS-Net's gain comes from SAM's representations or merely from the
    presence of a second branch."""

    RANDOM_SEED = 20261005

    def __init__(self, out_channels: int = 256):
        super().__init__()
        gen = torch.Generator().manual_seed(self.RANDOM_SEED)
        layers, in_ch = [], 3
        # 256 -> 128 -> 64, then widen to 256 channels, matching SAM's stride-16
        # output when fed the upsampled 1024 input (1024/16 = 64).
        for out_ch in (32, 64, out_channels):
            conv = nn.Conv2d(in_ch, out_ch, 3, stride=2 if in_ch != 64 else 1, padding=1, bias=False)
            with torch.no_grad():
                conv.weight.copy_(torch.randn(conv.weight.shape, generator=gen) * 0.05)
            layers += [conv, nn.GELU()]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, x):
        with torch.no_grad():
            y = self.net(x)
            if y.shape[-2:] != (64, 64):
                y = F.interpolate(y, size=(64, 64), mode="bilinear", align_corners=False)
            return y


# --------------------------------------------------------------------------
# fusion operators
# --------------------------------------------------------------------------
class ACFB(nn.Module):
    """Adaptive Cross-stream Fusion Block.

    full  : projection -> BDSG -> DKSR (3x3 + 5x5) -> SE -> residual + GroupNorm
    bdsg  : as above but a single 3x3 instead of the dual kernel

    The residual is taken on the post-DKSR tensor, which is what the released
    code does. Equation 14 of the manuscript places it around the DKSR+SE pair;
    that discrepancy is disclosed rather than silently changed.
    """

    def __init__(self, c_x: int, c_s: int = 256, dual_kernel: bool = True):
        super().__init__()
        self.dual_kernel = dual_kernel
        self.proj_s = nn.Conv2d(c_s, c_x, 1)
        self.proj_x = nn.Conv2d(c_x, c_x, 1)
        self.f_gate = nn.Conv2d(c_x, c_x, 1)
        self.s_gate = nn.Conv2d(c_x, c_x, 1)
        self.conv3 = nn.Conv2d(c_x, c_x, 3, padding=1)
        if dual_kernel:
            self.conv5 = nn.Conv2d(c_x, c_x, 5, padding=2)
        hidden = max(c_x // 4, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c_x, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c_x, 1), nn.Sigmoid())
        self.out = nn.Conv2d(c_x, c_x, 1)
        self.norm = nn.GroupNorm(8 if c_x >= 8 else 1, c_x)

    def forward(self, x, s):
        x, s = self.proj_x(x), self.proj_s(s)
        fused = x * torch.sigmoid(self.s_gate(s)) + s * torch.sigmoid(self.f_gate(x))
        multi = self.conv3(fused) + self.conv5(fused) if self.dual_kernel else self.conv3(fused)
        return self.norm(self.out(multi * self.se(multi)) + multi)


class ConcatFusion(nn.Module):
    """Plain projection + concatenation + 1x1. The 'no ACFB' control."""

    def __init__(self, c_x: int, c_s: int = 256):
        super().__init__()
        self.proj_s = nn.Conv2d(c_s, c_x, 1)
        self.proj_x = nn.Conv2d(c_x, c_x, 1)
        self.out = nn.Conv2d(2 * c_x, c_x, 1)

    def forward(self, x, s):
        return self.out(torch.cat([self.proj_x(x), self.proj_s(s)], 1))


def _xgated21(c_x: int, c_s: int = 256, sam_scale: int | None = None) -> nn.Module:
    """Reference-21 X-Gated Fusion Block.

    Imported from the separately documented port, which was verified
    bit-identical to the upstream implementation (commit 237669b4) by copying
    weights across and comparing outputs. See reference21/FIDELITY_REPORT.md.

    Two properties of their operator must be stated whenever the four-scale
    variant is reported:

    * `sam_to_cnn_scale` selects their exact learned ConvTranspose2d for
      upsampling the SAM feature. Without it the port falls back to bilinear
      upsampling, which is NOT what upstream does. We pass 4 at 256x256 and
      2 at 128x128.
    * Their cross-attention cannot run at 256x256 or 128x128: the attention
      matrix would be 65536 x 65536. Upstream confines attention to the
      bottleneck for exactly this reason. At those two scales only the gated
      squeeze-and-excitation half of the operator is active, and the paper
      must say so rather than implying the full block runs at every scale.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference21"))
    try:
        from xgated_fusion_swain import XGatedFusionSwain
    except ImportError as exc:                                      # pragma: no cover
        raise ImportError(
            "The reference-21 fusion port is not available yet. Expected "
            "03_code/reference21/xgated_fusion_swain.py. Variants 'xgated21' "
            "cannot be built until that port exists."
        ) from exc
    inner = XGatedFusionSwain(sam_channels=c_s, cnn_channels=c_x, out_channels=c_x,
                              sam_to_cnn_scale=sam_scale, cross_attention="auto")
    return _SwainAdapter(inner)


class _SwainAdapter(nn.Module):
    """Argument-order adapter. Our fusion convention is forward(cnn, sam); the
    upstream port's is forward(sam, cnn). Adapting explicitly rather than
    reordering at the call site keeps the port untouched and auditable."""

    # Their block performs its OWN learned resampling of the SAM feature
    # (ConvTranspose2d upstream), so it must receive the raw 64x64 embedding
    # rather than one PSPI has already bilinearly resized. Feeding it a
    # pre-resized tensor would stack two resamplings and is not their method.
    expects_raw_context = True

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x, s):
        return self.inner(s, x)


FUSIONS = {
    "acfb": lambda c_x, c_s, sam_scale=None: ACFB(c_x, c_s, dual_kernel=True),
    "bdsg": lambda c_x, c_s, sam_scale=None: ACFB(c_x, c_s, dual_kernel=False),
    "concat": lambda c_x, c_s, sam_scale=None: ConcatFusion(c_x, c_s),
    "xgated21": _xgated21,
}


# --------------------------------------------------------------------------
# published baselines
# --------------------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.enc = Encoder(base)
        self.u4 = Up(base * 16, base * 8, base * 8)
        self.u3 = Up(base * 8, base * 4, base * 4)
        self.u2 = Up(base * 4, base * 2, base * 2)
        self.u1 = Up(base * 2, base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.enc(x)
        return self.head(self.u1(self.u2(self.u3(self.u4(e5, e4), e3), e2), e1))


class UNetPP(nn.Module):
    def __init__(self, base: int = 16):
        super().__init__()
        b = base
        self.p = nn.MaxPool2d(2)
        self.x00, self.x10, self.x20, self.x30, self.x40 = (
            ConvBlock(3, b), ConvBlock(b, 2*b), ConvBlock(2*b, 4*b),
            ConvBlock(4*b, 8*b), ConvBlock(8*b, 16*b))
        self.x01, self.x11, self.x21, self.x31 = (
            ConvBlock(b+2*b, b), ConvBlock(2*b+4*b, 2*b),
            ConvBlock(4*b+8*b, 4*b), ConvBlock(8*b+16*b, 8*b))
        self.x02, self.x12, self.x22 = ConvBlock(4*b, b), ConvBlock(8*b, 2*b), ConvBlock(16*b, 4*b)
        self.x03, self.x13 = ConvBlock(5*b, b), ConvBlock(10*b, 2*b)
        self.x04, self.head = ConvBlock(6*b, b), nn.Conv2d(b, 1, 1)

    @staticmethod
    def up(x, target):
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        p = self.p
        x00 = self.x00(x); x10 = self.x10(p(x00)); x20 = self.x20(p(x10))
        x30 = self.x30(p(x20)); x40 = self.x40(p(x30))
        x01 = self.x01(torch.cat([x00, self.up(x10, x00)], 1))
        x11 = self.x11(torch.cat([x10, self.up(x20, x10)], 1))
        x21 = self.x21(torch.cat([x20, self.up(x30, x20)], 1))
        x31 = self.x31(torch.cat([x30, self.up(x40, x30)], 1))
        x02 = self.x02(torch.cat([x00, x01, self.up(x11, x00)], 1))
        x12 = self.x12(torch.cat([x10, x11, self.up(x21, x10)], 1))
        x22 = self.x22(torch.cat([x20, x21, self.up(x31, x20)], 1))
        x03 = self.x03(torch.cat([x00, x01, x02, self.up(x12, x00)], 1))
        x13 = self.x13(torch.cat([x10, x11, x12, self.up(x22, x10)], 1))
        x04 = self.x04(torch.cat([x00, x01, x02, x03, self.up(x13, x00)], 1))
        return self.head(x04)


class UNet3Plus(nn.Module):
    def __init__(self, base: int = 32, block=ConvBlock):
        super().__init__()
        b = base
        self.enc = Encoder(b, block)
        d = 5 * b
        self.d4 = FullScaleFuse([b, 2*b, 4*b, 8*b, 16*b], b)
        self.d3 = FullScaleFuse([b, 2*b, 4*b, d, 16*b], b)
        self.d2 = FullScaleFuse([b, 2*b, d, d, 16*b], b)
        self.d1 = FullScaleFuse([b, d, d, d, 16*b], b)
        self.head = nn.Conv2d(d, 1, 1)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.enc(x)
        d4 = self.d4([e1, e2, e3, e4, e5], e4)
        d3 = self.d3([e1, e2, e3, d4, e5], e3)
        d2 = self.d2([e1, e2, d3, d4, e5], e2)
        d1 = self.d1([e1, d2, d3, d4, e5], e1)
        return self.head(d1)


class EUNet3Plus(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        b = base
        self.pool = nn.MaxPool2d(2)
        self.e1, self.e2, self.e3, self.e4, self.center = (
            GLUBlock(3, b), GLUBlock(b, 2*b), GLUBlock(2*b, 4*b),
            GLUBlock(4*b, 8*b), GLUBlock(8*b, 16*b))
        self.d4, self.d3, self.d2, self.d1 = (
            GLUBlock(24*b, 8*b), GLUBlock(12*b, 4*b), GLUBlock(6*b, 2*b), GLUBlock(3*b, b))
        self.head = nn.Conv2d(b, 1, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3)); center = self.center(self.pool(e4))
        d4 = self.d4(torch.cat([self._up(center, e4), e4], 1))
        d3 = self.d3(torch.cat([self._up(d4, e3), e3], 1))
        d2 = self.d2(torch.cat([self._up(d3, e2), e2], 1))
        d1 = self.d1(torch.cat([self._up(d2, e1), e1], 1))
        return self.head(d1)


class SAMBottleneckFusion(nn.Module):
    def __init__(self, sam_ch=256, unet_ch=512, out_ch=512):
        super().__init__()
        self.sam_proj, self.unet_proj = nn.Conv2d(sam_ch, 256, 1), nn.Conv2d(unet_ch, 256, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(512, 256, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 1), nn.Sigmoid())
        self.out = nn.Conv2d(512, out_ch, 1)

    def forward(self, sam, unet):
        fused = torch.cat([self.sam_proj(sam), self.unet_proj(unet)], 1)
        return self.out(fused * self.gate(fused))


class SAMUNet(nn.Module):
    """Our simplified bottleneck SAM baseline, unchanged from the published run."""

    def __init__(self, checkpoint: str):
        super().__init__()
        self.sam = SAMImageEncoder(checkpoint)
        self.pool = nn.MaxPool2d(2)
        self.e1, self.e2, self.e3, self.e4 = (
            ConvBlock(3, 64), ConvBlock(64, 128), ConvBlock(128, 256), ConvBlock(256, 512))
        self.fuse = SAMBottleneckFusion()
        self.d3, self.d2, self.d1 = ConvBlock(768, 256), ConvBlock(384, 128), ConvBlock(192, 64)
        self.head = nn.Conv2d(64, 1, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x, ctx=None):
        sam = self.sam(x) if ctx is None else ctx
        e1 = self.e1(x); e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
        y = self.fuse(self._up(sam, e4), e4)
        y = self.d3(torch.cat([self._up(y, e3), e3], 1))
        y = self.d2(torch.cat([self._up(y, e2), e2], 1))
        y = self.d1(torch.cat([self._up(y, e1), e1], 1))
        return self.head(y)


# --------------------------------------------------------------------------
# EPCS-Net, parametric
# --------------------------------------------------------------------------
class EPCSNet(nn.Module):
    def __init__(self, checkpoint: str | None = None, base: int = 64,
                 sam_source: str = "sam", injection: str = "pyramid",
                 fusion: str = "acfb", encoder_norm: str = "bn"):
        super().__init__()
        if sam_source not in ("sam", "random", "none"):
            raise ValueError(sam_source)
        if injection not in ("pyramid", "bottleneck"):
            raise ValueError(injection)
        if fusion not in FUSIONS:
            raise ValueError(fusion)

        b = base
        self.injection = injection
        # xgated21 resamples internally; ACFB/concat expect PSPI-resized input.
        self.raw_context = (fusion == "xgated21")
        self.sam_source = sam_source
        self.no_context = (sam_source == "none")
        if self.no_context:
            self.context = None      # no SAM, no checkpoint, no fusion blocks
        else:
            self.context = (SAMImageEncoder(checkpoint) if sam_source == "sam"
                            else RandomFrozenEncoder(256))
        self.pool = nn.MaxPool2d(2)
        self.e1 = ConvBlock(3, b, encoder_norm)
        self.e2 = ConvBlock(b, 2*b, encoder_norm)
        self.e3 = ConvBlock(2*b, 4*b, encoder_norm)
        self.e4 = ConvBlock(4*b, 8*b, encoder_norm)

        make = FUSIONS[fusion]
        if self.no_context:
            injection = "none"        # skip both fusion-construction branches below
            self.injection = "none"
        # SAM's embedding is 64x64. Encoder scales are 256, 128, 64, 32, so the
        # SAM feature is upsampled x4 and x2 at the two shallow scales, matched
        # at scale 3, and downsampled at the bottleneck.
        if injection == "pyramid":
            self.f1 = make(b, 256, 4)
            self.f2 = make(2*b, 256, 2)
            self.f3 = make(4*b, 256, None)
            self.f4 = make(8*b, 256, None)
        elif injection == "bottleneck":
            self.f4 = make(8*b, 256, None)

        self.d4 = ConvBlock(8*b, 8*b, encoder_norm)
        self.d3 = ConvBlock(b + 2*b + 4*b + 8*b, 4*b, encoder_norm)
        self.d2 = ConvBlock(b + 2*b + 4*b, 2*b, encoder_norm)
        self.d1 = ConvBlock(b + 2*b, b, encoder_norm)
        self.head = nn.Conv2d(b, 1, 1)

    @staticmethod
    def _to(ctx, ref):
        return F.interpolate(ctx, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x, ctx=None):
        # ctx lets a precomputed frozen embedding be supplied. The encoder is
        # frozen and in eval mode, so a cached embedding is numerically
        # identical to recomputing it; sam_cache_r3.py asserts that bit-for-bit.
        if not self.no_context:
            ctx = self.context(x) if ctx is None else ctx
        e1 = self.e1(x); e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))

        def ctx_for(ref):
            return ctx if self.raw_context else self._to(ctx, ref)

        if self.no_context:
            f1, f2, f3, f4 = e1, e2, e3, e4      # identity at all four scales
        elif self.injection == "pyramid":
            f1 = self.f1(e1, ctx_for(e1))
            f2 = self.f2(e2, ctx_for(e2))
            f3 = self.f3(e3, ctx_for(e3))
            f4 = self.f4(e4, ctx_for(e4))
        else:
            f1, f2, f3 = e1, e2, e3
            f4 = self.f4(e4, ctx_for(e4))

        d4 = self.d4(f4)
        d3 = self.d3(torch.cat([F.max_pool2d(f1, 4), F.max_pool2d(f2, 2), f3,
                                self._to(d4, f3)], 1))
        d2 = self.d2(torch.cat([F.max_pool2d(f1, 2), f2, self._to(d3, f2)], 1))
        d1 = self.d1(torch.cat([f1, self._to(d2, f1)], 1))
        return self.head(d1)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
# name -> (builder, loss_id, needs_sam_checkpoint, human label)
def _epcs(**kw):
    return lambda ckpt: EPCSNet(ckpt, **kw)


MODELS: dict[str, dict] = {
    # --- published comparison set -----------------------------------------
    "unet":       dict(build=lambda c: UNet(),       loss="bce_dice", sam=False, group="baseline", label="U-Net"),
    "unetpp":     dict(build=lambda c: UNetPP(),     loss="bce_dice", sam=False, group="baseline", label="U-Net++"),
    "unet3plus":  dict(build=lambda c: UNet3Plus(),  loss="bce_dice", sam=False, group="baseline", label="U-Net3+"),
    "eunet3plus": dict(build=lambda c: EUNet3Plus(), loss="bce_dice", sam=False, group="baseline", label="eU-Net3+ (no SAM)"),
    "sam_unet":   dict(build=lambda c: SAMUNet(c),   loss="bce_dice", sam=True,  group="baseline", label="SAM-guided baseline (bottleneck)"),
    "epcs_net":   dict(build=_epcs(),                loss="ftas",     sam=True,  group="proposed", label="EPCS-Net (proposed)"),

    # --- reference 21, our codebase, controlled placement -----------------
    "ref21_bottleneck": dict(build=_epcs(injection="bottleneck", fusion="xgated21"), loss="ftas", sam=True,
                             group="reference21", label="Ref-21 X-Gated fusion, bottleneck"),
    "ref21_pyramid":    dict(build=_epcs(injection="pyramid", fusion="xgated21"), loss="ftas", sam=True,
                             group="reference21", label="Ref-21 X-Gated fusion, four scales"),

    # --- ablation chain, matching the published Table 13 ordering ---------
    "abl0_nosam_epcsbackbone": dict(build=_epcs(sam_source="none"), loss="bce_dice", sam=False,
                              group="ablation", label="0. EPCS backbone, no SAM"),
    "abl1_nosam":        dict(build=lambda c: EUNet3Plus(), loss="bce_dice", sam=False, group="ablation",
                              label="1. eU-Net3+ only (no SAM, no ACFB)"),
    "abl2_bottleneck":   dict(build=_epcs(injection="bottleneck", fusion="concat"), loss="bce_dice", sam=True,
                              group="ablation", label="2. + SAM bottleneck fusion"),
    "abl3_pspi":         dict(build=_epcs(injection="pyramid", fusion="concat"), loss="bce_dice", sam=True,
                              group="ablation", label="3. + PSPI (four-scale)"),
    "abl4_bdsg":         dict(build=_epcs(injection="pyramid", fusion="bdsg"), loss="bce_dice", sam=True,
                              group="ablation", label="4. + ACFB bidirectional gating"),
    "abl5_dksr":         dict(build=_epcs(injection="pyramid", fusion="acfb"), loss="bce_dice", sam=True,
                              group="ablation", label="5. + dual-kernel refinement"),
    "abl6_ftas":         dict(build=_epcs(), loss="ftas", sam=True, group="ablation",
                              label="6. + FTAS loss (full EPCS-Net)"),

    # --- controls the reviewers asked for ---------------------------------
    "ablA_acfb_nopspi":  dict(build=_epcs(injection="bottleneck", fusion="acfb"), loss="ftas", sam=True,
                              group="ablation_control", label="A. ACFB without PSPI"),
    "ablB_random_embed": dict(build=_epcs(sam_source="random"), loss="ftas", sam=False,
                              group="ablation_control", label="B. Random frozen embedding"),
    "ablC_pspi_concat":  dict(build=_epcs(fusion="concat"), loss="ftas", sam=True,
                              group="ablation_control", label="C. PSPI without ACFB"),
    "ablD_gn_encoder":   dict(build=_epcs(encoder_norm="gn"), loss="ftas", sam=True,
                              group="ablation_control", label="D. GroupNorm encoder"),
}

PUBLISHED_PARAMS = {       # from the Run A summary JSONs: (total, trainable)
    "unet": (7_849_601, 7_849_601),
    "unetpp": (2_291_969, 2_291_969),
    "unet3plus": (6_684_289, 6_684_289),
    "eunet3plus": (11_767_265, 11_767_265),
    "sam_unet": (102_243_889, 8_508_417),
    "epcs_net": (120_417_697, 26_682_225),
}


def build_model(model_id: str, sam_checkpoint: str | None = None) -> nn.Module:
    if model_id not in MODELS:
        raise ValueError(f"unknown model id '{model_id}'. known: {sorted(MODELS)}")
    spec = MODELS[model_id]
    ckpt = sam_checkpoint or str(SAM_CHECKPOINT)
    if spec["sam"] and not Path(ckpt).is_file():
        raise FileNotFoundError(f"{model_id} needs the SAM ViT-B checkpoint at {ckpt}")
    return spec["build"](ckpt)


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def verify_published_parameter_counts(verbose: bool = True) -> bool:
    """Regression test: the six published architectures must be reproduced
    exactly. Any structural drift shows up here as a parameter-count mismatch."""
    ok = True
    for model_id, (exp_total, exp_trainable) in PUBLISHED_PARAMS.items():
        model = build_model(model_id)
        total, trainable = count_params(model)
        good = (total == exp_total) and (trainable == exp_trainable)
        ok &= good
        if verbose:
            mark = "OK  " if good else "FAIL"
            print(f"  {mark} {model_id:<12} total {total:>11,} (expected {exp_total:>11,})  "
                  f"trainable {trainable:>11,} (expected {exp_trainable:>11,})")
        del model
    return ok


if __name__ == "__main__":
    print("Verifying the six published architectures reproduce exactly:")
    ok = verify_published_parameter_counts()
    print("\nALL PARAMETER COUNTS MATCH" if ok else "\nMISMATCH -- do not proceed")
    sys.exit(0 if ok else 1)
