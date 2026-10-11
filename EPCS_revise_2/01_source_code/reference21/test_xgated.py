"""Smoke test for the XGatedFusionSwain port.

Instantiates the block at the four encoder scales of our model, runs a CPU
forward pass with random tensors, asserts the output shapes and prints the
parameter count at each scale.

SAM ViT-B always emits a 256 x 64 x 64 embedding for a 1024 x 1024 input
(see upstream/genSAMEncoding.py:49-51 and the shape comment at
upstream/networkModules/modelUnet3p.py:420), so sam_channels=256 and the SAM
spatial size is 64 x 64 at every scale.
"""

import torch

from xgated_fusion_swain import UPSTREAM_COMMIT, XGatedFusionSwain

# (cnn_channels, cnn_spatial) -- our encoder's four scales.
SCALES = [
    (64, 256),
    (128, 128),
    (256, 64),
    (512, 32),
]

SAM_CHANNELS = 256
SAM_SPATIAL = 64
BATCH = 2


def main() -> None:
    torch.manual_seed(0)
    print(f"upstream commit: {UPSTREAM_COMMIT}")
    print(f"torch {torch.__version__} | device cpu | batch {BATCH}")
    print(f"SAM feature: {SAM_CHANNELS} x {SAM_SPATIAL} x {SAM_SPATIAL}\n")

    header = (
        f"{'cnn_ch':>7} {'spatial':>9} {'sam_branch':>11} {'se_hidden':>10} "
        f"{'x-attn':>7} {'params':>12} {'out shape':>22}"
    )
    print(header)
    print("-" * len(header))

    total = 0
    for cnn_channels, spatial in SCALES:
        block = XGatedFusionSwain(
            sam_channels=SAM_CHANNELS,
            cnn_channels=cnn_channels,
            out_channels=cnn_channels,
        ).eval()

        sam_feat = torch.randn(BATCH, SAM_CHANNELS, SAM_SPATIAL, SAM_SPATIAL)
        cnn_feat = torch.randn(BATCH, cnn_channels, spatial, spatial)

        with torch.no_grad():
            out = block(sam_feat, cnn_feat)

        expected = (BATCH, cnn_channels, spatial, spatial)
        assert out.shape == expected, f"got {tuple(out.shape)}, expected {expected}"
        assert torch.isfinite(out).all(), "non-finite values in output"

        n_params = sum(p.numel() for p in block.parameters())
        total += n_params
        used_attn = block._use_attention(spatial * spatial)
        print(
            f"{cnn_channels:>7} {spatial:>4}x{spatial:<4} {block.sam_branch_channels:>11} "
            f"{block.se_hidden_channels:>10} {str(used_attn):>7} {n_params:>12,} "
            f"{str(tuple(out.shape)):>22}"
        )

    print("-" * len(header))
    print(f"{'total':>7} {'':>9} {'':>11} {'':>10} {'':>7} {total:>12,}\n")

    # Extra check: out_channels != cnn_channels exercises the added 1x1
    # out_proj (ASSUMPTION 3 in xgated_fusion_swain.py).
    block = XGatedFusionSwain(sam_channels=256, cnn_channels=512, out_channels=256).eval()
    with torch.no_grad():
        out = block(torch.randn(1, 256, 64, 64), torch.randn(1, 512, 32, 32))
    assert out.shape == (1, 256, 32, 32), out.shape
    print(f"out_channels != cnn_channels -> {tuple(out.shape)}  OK")

    # Extra check: the exact upstream ConvTranspose2d upsampling path
    # (modelUnet3p.py:77-91) for the two scales where SAM must be upsampled.
    for factor, cnn_channels, spatial in ((4, 64, 256), (2, 128, 128)):
        block = XGatedFusionSwain(
            sam_channels=256,
            cnn_channels=cnn_channels,
            out_channels=cnn_channels,
            sam_to_cnn_scale=factor,
        ).eval()
        with torch.no_grad():
            out = block(
                torch.randn(1, 256, 64, 64), torch.randn(1, cnn_channels, spatial, spatial)
            )
        assert out.shape == (1, cnn_channels, spatial, spatial), out.shape
        print(
            f"sam_to_cnn_scale={factor} (exact upstream deconv) -> {tuple(out.shape)}  OK"
        )

    # Extra check: gating-only ablation, i.e. upstream's h1-h4 behaviour.
    block = XGatedFusionSwain(
        sam_channels=256, cnn_channels=64, out_channels=64, cross_attention="never"
    ).eval()
    with torch.no_grad():
        out = block(torch.randn(1, 256, 64, 64), torch.randn(1, 64, 256, 256))
    assert out.shape == (1, 64, 256, 256), out.shape
    print(f"cross_attention='never' (gating only)  -> {tuple(out.shape)}  OK")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
