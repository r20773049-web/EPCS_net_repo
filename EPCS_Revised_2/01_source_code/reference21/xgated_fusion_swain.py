"""
Faithful standalone PyTorch port of the "X-Gated Fusion Block" (X-GFB) of

    B. R. Swain, K. J. Cheoi, J. Ko,
    "SAM Guided Task-Specific Enhanced Nuclei Segmentation in Digital Pathology",
    MICCAI 2024.

Upstream source (MIT licence, (c) 2024 CVPR, KIT):
    https://github.com/CVPR-KIT/SAM-Guided-Enhanced-Nuclei-Segmentation
    cloned to ./upstream/ at commit 237669b4d2e8eca6b1517bd522b8d4c0246ffa50

All line references below are to that commit.

--------------------------------------------------------------------------
WHAT THE UPSTREAM CODE ACTUALLY DOES
--------------------------------------------------------------------------
The upstream repository never defines a module called "X-Gated Fusion Block".
The operator is written inline inside `UNet_3Plus`, split between the
constructor (upstream/networkModules/modelUnet3p.py:99-150) and two forward
methods (`guidedForward`, lines 311-415, and `advGuidedForward`, lines
418-545).

Bottleneck fusion, as executed by `guidedForward` (modelUnet3p.py:346-374):

    samcd   = MaxBlurPool2d(MaxBlurPool2d(SAM_Enc))        # L347-348
    combined = cat([hd5, samcd], dim=1)                     # L351
    w        = squeeze_excite_h5(combined)                  # L355
               # = AdaptiveAvgPool2d(1)
               #   Conv2d(cnn+sam, (cnn+sam)//16, k=1)
               #   ReLU(inplace=True)
               #   Conv2d((cnn+sam)//16, cnn, k=1)
               #   Sigmoid                                  # L102-108
    gated    = w * hd5                                      # L359
    q        = flatten(gated)                               # L364
    kv       = flatten(hd5)                                 # L365   <-- UNGATED
    out, _   = MultiheadAttention(embed_dim=cnn, num_heads=1,
                                  batch_first=True)(q, kv, kv)  # L109, L368
    hd5      = unflatten(out)                               # L369, L374

There is NO residual connection, NO normalisation, and NO output projection:
the attention output *replaces* the encoder feature entirely (L374).

--------------------------------------------------------------------------
THREE THINGS THAT DO NOT MATCH THE PAPER / PROJECT PAGE
--------------------------------------------------------------------------
1. GLU.  The project page (docs/index.html, "Method") states the X-GFB
   "first uses GLU in gated squeeze and excitation block".  It does not.
   The SE block in the released code is Conv-ReLU-Conv-Sigmoid
   (modelUnet3p.py:102-108, and identically at L114-120, L124-130, L134-140,
   L144-150).  `grep -rn GLU --include=*.py` over the whole repo matches ONLY
   conv_modules_unet3p.py:23-26 and 35-37, i.e. the plain encoder conv blocks
   `unetConv2`, which is a different thing entirely.  This port follows the
   CODE (ReLU), not the paper text.  See FIDELITY_REPORT.md.

2. Cross-attention is applied at the BOTTLENECK ONLY.  At scales h1-h4
   (`advGuidedForward`, L425-461) only the gated-SE part runs; there is no
   attention.  The paper figure (docs/static/images/model_arch.png) likewise
   shows a single fusion box between encoder and decoder.

3. `advGuidedForward` contains a copy-paste bug.  At L489-490 both the query
   and the key/value are built from `gatedFeatures`, so the "cross"-attention
   degenerates to self-attention on the gated tensor.  `guidedForward`
   (L364-365) uses the ungated `hd5` as key/value and is the genuine
   cross-attention.  This port implements the `guidedForward` version, which
   is also the one selected by configs/config_cryonuseg.sys
   (advancedSAM_Guided = "False", L38) and configs/config_connic.sys (L35).

--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

__all__ = ["XGatedFusionSwain", "MaxBlurPool2d"]

# Upstream commit this port was written against.
UPSTREAM_COMMIT = "237669b4d2e8eca6b1517bd522b8d4c0246ffa50"


class MaxBlurPool2d(nn.Module):
    """Verbatim port of upstream/networkModules/conv_modules_unet3p.py:121-142.

    2x max-pool followed by a fixed (non-learned) 3x3 binomial blur applied
    depthwise with padding=1, so the spatial size is exactly halved.  Note the
    upstream class keeps `blur_kernel` as a plain attribute, not a registered
    buffer (L129-130), and moves it to the input device inside forward
    (L137); that behaviour is reproduced, except that we register it as a
    non-persistent buffer so `.to(device)` works as users expect.  This is
    numerically identical.
    """

    def __init__(self, kernel_size: int = 2, stride: int = 2, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        k = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 16.0
        self.register_buffer("blur_kernel", k, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )
        w = self.blur_kernel.to(dtype=x.dtype, device=x.device).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, w, groups=x.size(1), padding=1)


def _kaiming_fan_in_(module: nn.Module) -> None:
    """Replicates upstream/networkModules/init_weights.py:29-38 as applied by
    modelUnet3p.py:301-305 (every nn.Conv2d / nn.BatchNorm2d in the model is
    re-initialised with kaiming_normal_, a=0, mode='fan_in'; biases are left
    at the PyTorch default).  nn.MultiheadAttention is deliberately NOT
    touched upstream -- the isinstance checks at L302/L304 do not match it --
    so it keeps PyTorch's own xavier_uniform_ initialisation here too.
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)


class XGatedFusionSwain(nn.Module):
    """X-Gated Fusion Block of Swain et al. (MICCAI 2024), generalised to any
    encoder scale.

    Args
    ----
    sam_channels : channels of the frozen-SAM feature map (256 for ViT-B).
    cnn_channels : channels of the task-specific encoder feature map.
    out_channels : channels of the returned tensor.
    sam_branch_channels : channels the SAM branch is projected to before the
        concatenation.  Default None -> cnn_channels // 2, see ASSUMPTION 2.
    sam_to_cnn_scale : if given, the integer factor by which the SAM feature
        must be UPsampled to reach the CNN resolution (2 or 4 in the upstream
        model).  Supplying it reproduces upstream's learned ConvTranspose2d
        exactly; leaving it None uses the generic resampler, see ASSUMPTION 3.
    cross_attention : "auto" (default), "always" or "never".  See ASSUMPTION 5.
    max_attention_tokens : token budget for "auto".  Default 4096 = 64x64.
    reduction_ratio : SE bottleneck divisor, 16 upstream (modelUnet3p.py:30).
    num_heads : attention heads, 1 upstream (modelUnet3p.py:109).

    forward(sam_feat, cnn_feat) -> Tensor
        sam_feat : (B, sam_channels, Hs, Ws)
        cnn_feat : (B, cnn_channels, Hc, Wc)
        returns  : (B, out_channels, Hc, Wc)
    """

    def __init__(
        self,
        sam_channels: int,
        cnn_channels: int,
        out_channels: int,
        *,
        sam_branch_channels: Optional[int] = None,
        sam_to_cnn_scale: Optional[int] = None,
        cross_attention: str = "auto",
        max_attention_tokens: int = 4096,
        reduction_ratio: int = 16,
        num_heads: int = 1,
    ) -> None:
        super().__init__()

        if cross_attention not in ("auto", "always", "never"):
            raise ValueError("cross_attention must be 'auto', 'always' or 'never'")

        self.sam_channels = sam_channels
        self.cnn_channels = cnn_channels
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio
        self.cross_attention_mode = cross_attention
        self.max_attention_tokens = max_attention_tokens

        # ------------------------------------------------------------------
        # ASSUMPTION 1 (SAM branch width rule).
        # Upstream hard-codes a different SAM-branch width per scale, in the
        # constructor: 16 for h1 (ConvTranspose2d out_channels=16, L77-83),
        # 32 for h2 (L85-91), 64 for h3 (Conv2d 1x1, L94), 128 for h4 (L96),
        # and 256 for the bottleneck (no projection at all -- the raw SAM
        # embedding is used, L351).  With the paper's `channel = 32`
        # (configs/config_cryonuseg.sys:45) the encoder widths are
        # filters = [32, 64, 128, 256, 512] (L43), so in EVERY one of the five
        # cases the SAM branch width equals cnn_channels // 2.  I therefore
        # generalise the rule as sam_branch = cnn_channels // 2.  This is an
        # inference from five consistent data points, not something upstream
        # states; it is exact at all five upstream scales but is still an
        # extrapolation.  Override with `sam_branch_channels=` if you disagree.
        # ------------------------------------------------------------------
        if sam_branch_channels is None:
            sam_branch_channels = max(1, cnn_channels // 2)
        self.sam_branch_channels = sam_branch_channels

        # ------------------------------------------------------------------
        # SAM-branch projection + resampling.
        # ------------------------------------------------------------------
        self.sam_to_cnn_scale = sam_to_cnn_scale
        if sam_to_cnn_scale is not None:
            if sam_to_cnn_scale < 2 or (sam_to_cnn_scale & (sam_to_cnn_scale - 1)) != 0:
                raise ValueError(
                    "sam_to_cnn_scale must be a power of two >= 2 (upstream uses 2 and 4)"
                )
            # EXACT reproduction of modelUnet3p.py:77-91.  Upstream builds
            #   f=4: ConvTranspose2d(256, 16, kernel_size=8, stride=4, padding=2)
            #   f=2: ConvTranspose2d(256, 32, kernel_size=4, stride=2, padding=1)
            # Both are the single formula k=2f, s=f, p=f//2, which is what we
            # use.  Note that upstream's deconv performs the channel
            # projection AND the upsampling in one layer, so no separate 1x1
            # conv is created in this branch.
            f = sam_to_cnn_scale
            self.sam_proj = nn.ConvTranspose2d(
                in_channels=sam_channels,
                out_channels=sam_branch_channels,
                kernel_size=2 * f,
                stride=f,
                padding=f // 2,
            )
            self._resample = "none"
        elif sam_branch_channels == sam_channels:
            # Bottleneck case: upstream applies NO channel projection to the
            # SAM embedding at h5 -- `samcd` is fed straight into the concat
            # (modelUnet3p.py:347-351).  Reproduce that exactly.
            self.sam_proj = nn.Identity()
            self._resample = "generic"
        else:
            # Same-resolution / downsampling case: upstream uses a plain 1x1
            # conv with bias and no normalisation or activation
            # (modelUnet3p.py:94 for h3, L96 for h4), then -- for h4 only --
            # a MaxBlurPool2d (L458).  Reproduce that ordering: project, then
            # resample.
            self.sam_proj = nn.Conv2d(sam_channels, sam_branch_channels, kernel_size=1)
            self._resample = "generic"

        self.sam_down = MaxBlurPool2d(kernel_size=2)

        # ------------------------------------------------------------------
        # Gated squeeze-and-excitation.
        # Verbatim from modelUnet3p.py:102-108 (and the identical blocks at
        # L114-120 / L124-130 / L134-140 / L144-150).
        # ------------------------------------------------------------------
        fused_channels = cnn_channels + sam_branch_channels
        hidden = fused_channels // reduction_ratio
        if hidden < 1:
            # ASSUMPTION 2 (degenerate SE bottleneck).
            # Upstream would silently build a Conv2d with out_channels=0 and
            # crash if fused_channels < 16.  That never happens at their five
            # scales (smallest is 32+16=48 -> 3).  We clamp to 1 so the module
            # is usable at narrow scales; at every upstream scale this clamp
            # is inactive and the port is bit-identical.
            hidden = 1
        self.se_hidden_channels = hidden
        self.squeeze_excite = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fused_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, cnn_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        # NOTE: AdaptiveAvgPool2d(1) comes FIRST, so the gate is a purely
        # channel-wise vector of shape (B, cnn_channels, 1, 1) -- there is no
        # spatial attention anywhere in this block.

        # ------------------------------------------------------------------
        # Cross-attention.  modelUnet3p.py:109.
        # embed_dim is the CNN width, num_heads=1, batch_first=True.
        # ------------------------------------------------------------------
        if cnn_channels % num_heads != 0:
            raise ValueError("cnn_channels must be divisible by num_heads")
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=cnn_channels, num_heads=num_heads, batch_first=True
        )

        # ------------------------------------------------------------------
        # ASSUMPTION 3 (output projection).
        # Upstream has NO output projection -- the block returns exactly
        # cnn_channels (modelUnet3p.py:374).  The caller's required signature
        # exposes `out_channels`, so when out_channels != cnn_channels we add
        # a bare 1x1 conv (no BN, no activation, bias=True, kaiming fan-in
        # init, consistent with every other 1x1 in their model, e.g. L94/L96).
        # When out_channels == cnn_channels this is an Identity and the port
        # is parameter-identical to upstream.  ALWAYS prefer
        # out_channels == cnn_channels for a like-for-like comparison.
        # ------------------------------------------------------------------
        if out_channels == cnn_channels:
            self.out_proj: nn.Module = nn.Identity()
        else:
            self.out_proj = nn.Conv2d(cnn_channels, out_channels, kernel_size=1)

        _kaiming_fan_in_(self)

    # ----------------------------------------------------------------------

    def _align(self, sam_feat: torch.Tensor, size: "tuple[int, int]") -> torch.Tensor:
        """Bring the SAM branch to the CNN spatial size."""
        if sam_feat.shape[-2:] == size:
            return sam_feat

        hs, ws = sam_feat.shape[-2:]
        hc, wc = size

        if hc < hs and wc < ws:
            # DOWNSAMPLING -- exact upstream behaviour.  At the bottleneck
            # upstream applies MaxBlurPool2d twice (modelUnet3p.py:347-348,
            # 64x64 -> 32x32 -> 16x16) and at h4 once (L458, 64x64 -> 32x32).
            # Chain MaxBlurPool2d while it gets us closer; this is exact for
            # every power-of-two ratio, which covers all upstream scales.
            x = sam_feat
            while x.shape[-2] // 2 >= hc and x.shape[-1] // 2 >= wc:
                x = self.sam_down(x)
            if x.shape[-2:] != size:
                # ASSUMPTION 4a: non-power-of-two ratio, which upstream never
                # encounters.  Fall back to bilinear interpolation.
                x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            return x

        # ------------------------------------------------------------------
        # ASSUMPTION 4b (generic UPsampling).
        # Upstream upsamples the SAM embedding with a LEARNED ConvTranspose2d
        # (modelUnet3p.py:77-91).  This module cannot build that layer in
        # __init__ because the required factor is only known from the runtime
        # tensor shapes, and lazily creating parameters after the optimiser
        # has been constructed silently drops them from training.  So the
        # generic path uses parameter-free bilinear interpolation, which is
        # NOT what upstream does.  Pass `sam_to_cnn_scale=` to get the exact
        # upstream ConvTranspose2d instead.  This difference affects only the
        # scales where SAM must be upsampled (h1 and h2 upstream).
        # ------------------------------------------------------------------
        return F.interpolate(sam_feat, size=size, mode="bilinear", align_corners=False)

    def _use_attention(self, tokens: int) -> bool:
        if self.cross_attention_mode == "never":
            return False
        if self.cross_attention_mode == "always":
            return True
        # ASSUMPTION 5 ("auto").
        # Upstream applies cross-attention at the BOTTLENECK ONLY
        # (modelUnet3p.py:368 is the sole call site; the h1-h4 fusions at
        # L425-461 are gating-only).  Applying O(N^2) attention at 256x256
        # would need a 65536x65536 attention matrix (~17 GB in fp32), which
        # is why they never did it.  "auto" therefore runs attention when the
        # token count is within budget and degrades to gating-only otherwise
        # -- which is exactly upstream's own behaviour at coarse vs fine
        # scales.  Set cross_attention="always" to force it and accept the
        # memory cost, or "never" for a pure gating ablation.
        return tokens <= self.max_attention_tokens

    def forward(self, sam_feat: torch.Tensor, cnn_feat: torch.Tensor) -> torch.Tensor:
        if sam_feat.dim() != 4 or cnn_feat.dim() != 4:
            raise ValueError("sam_feat and cnn_feat must both be 4-D (B, C, H, W)")
        if cnn_feat.shape[1] != self.cnn_channels:
            raise ValueError(
                f"cnn_feat has {cnn_feat.shape[1]} channels, expected {self.cnn_channels}"
            )
        if sam_feat.shape[1] != self.sam_channels:
            raise ValueError(
                f"sam_feat has {sam_feat.shape[1]} channels, expected {self.sam_channels}"
            )

        b, c, h, w = cnn_feat.shape

        # --- SAM branch: project then resample -----------------------------
        # Upstream order at h4 is project (L457) then pool (L458); at h1/h2
        # the ConvTranspose2d does both at once; at the bottleneck it is pool
        # only with no projection (L347-348).  self.sam_proj is Identity /
        # 1x1 conv / ConvTranspose2d accordingly, matching all three cases.
        sam = self.sam_proj(sam_feat)
        if self._resample != "none":
            sam = self._align(sam, (h, w))
        elif sam.shape[-2:] != (h, w):
            raise RuntimeError(
                f"sam_to_cnn_scale={self.sam_to_cnn_scale} produced {tuple(sam.shape[-2:])}, "
                f"expected {(h, w)}"
            )

        # --- Gated squeeze-and-excitation (modelUnet3p.py:351-359) ---------
        combined = torch.cat((cnn_feat, sam), dim=1)   # L351: (cnn, sam) order
        gating_weights = self.squeeze_excite(combined)  # L355 -> (B, C, 1, 1)
        gated = gating_weights * cnn_feat               # L359

        # --- Cross-attention (modelUnet3p.py:363-369) ----------------------
        if not self._use_attention(h * w):
            # Gating-only, i.e. upstream's h1-h4 behaviour
            # (modelUnet3p.py:429 / 441 / 451 / 461).
            return self.out_proj(gated)

        q = gated.view(b, c, -1).permute(0, 2, 1)       # L364
        kv = cnn_feat.view(b, c, -1).permute(0, 2, 1)   # L365: UNGATED feature
        attn_out, _ = self.cross_attention(query=q, key=kv, value=kv)  # L368
        attn_out = attn_out.permute(0, 2, 1).view(b, c, h, w)          # L369

        # L374: the attention output REPLACES the encoder feature.  No
        # residual add, no LayerNorm, no dropout.  Reproduced verbatim --
        # this is unusual for an attention block and is a real property of
        # their design, not an omission on our part.
        return self.out_proj(attn_out)

    def extra_repr(self) -> str:
        return (
            f"sam_channels={self.sam_channels}, cnn_channels={self.cnn_channels}, "
            f"out_channels={self.out_channels}, sam_branch_channels={self.sam_branch_channels}, "
            f"se_hidden={self.se_hidden_channels}, cross_attention={self.cross_attention_mode}"
        )
