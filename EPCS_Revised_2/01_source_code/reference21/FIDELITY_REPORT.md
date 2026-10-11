# Fidelity report: port of the X-Gated Fusion Block of reference [21]

**Reference [21].** B. R. Swain, K. J. Cheoi, J. Ko, *"SAM Guided Task-Specific Enhanced
Nuclei Segmentation in Digital Pathology"*, MICCAI 2024.

**Official code.** <https://github.com/CVPR-KIT/SAM-Guided-Enhanced-Nuclei-Segmentation>

**Clone.** `./upstream/`, cloned successfully with `git clone`.

| | |
|---|---|
| Commit hash | `237669b4d2e8eca6b1517bd522b8d4c0246ffa50` |
| Commit date | Wed 16 Oct 2024 17:16:44 +0900 |
| Commit subject | `Update index.html` |
| Branch | `main` (repository HEAD at time of clone) |
| Licence | MIT, © 2024 CVPR, KIT (`upstream/LICENSE`) |

The MIT licence permits use, modification and redistribution provided the copyright
notice and permission notice are retained. Our port carries attribution in its module
docstring. Nothing in the licence restricts use for a benchmark comparison or
publication.

**Our port.** `xgated_fusion_swain.py`, class `XGatedFusionSwain`.
**Smoke test.** `test_xgated.py`.

All line numbers below refer to the commit above. Anyone checking this report can run
`git -C upstream checkout 237669b4` and read the cited lines directly.

---

## 1. Summary of what we found

The upstream repository **does not contain a module named "X-Gated Fusion Block"**.
`grep -rniE "x-gfb|xgated|gated fusion|x-gated" --include=*.py` over the repository
returns **zero matches**; the only occurrence anywhere in the repository is in the
project-page prose at `docs/index.html`. The operator
is written inline inside the `UNet_3Plus` class, split between the constructor
(`upstream/networkModules/modelUnet3p.py:99-150`) and two alternative forward methods
(`guidedForward`, L311-415, and `advGuidedForward`, L418-545). Our port is a
reconstruction of that inline code into a reusable module, not a re-export of an
upstream class.

In the configuration that corresponds to the paper's headline CryoNuSeg and CoNIC
numbers (`advancedSAM_Guided = "False"`), the block is

```
samcd    = MaxBlurPool2d(MaxBlurPool2d(SAM_Enc))              # L347-348  64x64 -> 16x16
combined = cat([hd5, samcd], dim=1)                           # L351      512 + 256 = 768 ch
w        = AdaptiveAvgPool2d(1)                               # L103      -> 768 x 1 x 1
           Conv2d(768, 48, k=1) -> ReLU -> Conv2d(48, 512, k=1) -> Sigmoid   # L104-107
gated    = w * hd5                                            # L359      channel-wise gate
q        = flatten(gated)                                     # L364
k = v    = flatten(hd5)                                       # L365      UNGATED feature
out      = MultiheadAttention(embed_dim=512, num_heads=1)(q, k, v)   # L109, L368
hd5      = unflatten(out)                                     # L369, L374  REPLACES hd5
```

There is **no residual connection, no normalisation layer, no dropout inside the block,
and no output projection**. The attention output replaces the encoder bottleneck
feature outright.

### 1.1 Verified numerical equivalence

We transcribed the upstream bottleneck code verbatim, copied weights into our port, and
compared outputs on random inputs:

| Path | Upstream lines | Max abs difference | Bit-identical | Parameter count |
|---|---|---|---|---|
| Bottleneck (SE gate + cross-attention), cnn=512 / sam=256 | L102-109, L347-374 | `0.0` | yes (`torch.equal` True) | 1,112,624 both |
| h1 path (deconv ×4 + SE gate, no attention), cnn=32 / sam=16 | L77-83, L144-150, L425-429 | `0.0` | yes | — |
| h2 path (deconv ×2 + SE gate, no attention), cnn=64 / sam=32 | L85-91, L134-140, L437-441 | `0.0` | yes | — |

So for the three configurations where upstream and our port are directly comparable,
the port is **exact, not approximate**.

---

## 2. Architectural element table

"Exact" below means verified bit-identical against transcribed upstream code, or a
line-for-line transcription with no behavioural change.

### 2.1 X-Gated Fusion Block

| Element | Upstream file:line | Our port | Status |
|---|---|---|---|
| SAM branch, bottleneck: no channel projection | `modelUnet3p.py:347-351` | `sam_proj = nn.Identity()` when `sam_branch == sam_channels` | **Exact** |
| SAM branch, 2× / 4× upsample: `ConvTranspose2d(256, C, k=2f, s=f, p=f//2)` | `modelUnet3p.py:77-83` (f=4), `:85-91` (f=2) | same layer, built when `sam_to_cnn_scale` is given | **Exact** (when the arg is passed) |
| SAM branch, same-resolution / downsample: `Conv2d(256, C, k=1)` bias=True, no BN, no act | `modelUnet3p.py:94` (h3), `:96` (h4) | `nn.Conv2d(sam_channels, sam_branch, 1)` | **Exact** |
| Spatial downsample: `MaxBlurPool2d(kernel_size=2)` chained | `modelUnet3p.py:73, 347-348, 458` | `MaxBlurPool2d` chained in `_align` | **Exact** |
| `MaxBlurPool2d`: maxpool 2×2 then fixed `[1 2 1; 2 4 2; 1 2 1]/16` depthwise blur, padding=1 | `conv_modules_unet3p.py:121-142` | verbatim port; kernel registered as a non-persistent buffer instead of a plain attribute (numerically identical) | **Exact** |
| Spatial upsample, generic path | — | `F.interpolate(bilinear)` | **Approximate** — see ASSUMPTION 4b |
| Concatenation order `cat([cnn, sam], dim=1)` | `modelUnet3p.py:351, 427, 439, 449, 459, 479` | same | **Exact** |
| SE: `AdaptiveAvgPool2d(1)` **first** (so the gate is purely channel-wise) | `modelUnet3p.py:103` | same | **Exact** |
| SE: `Conv2d(cnn+sam, (cnn+sam)//16, k=1)` | `modelUnet3p.py:104`; ratio 16 set at `:30` | same, with a clamp to ≥1 (inactive at all upstream widths) | **Exact** / ASSUMPTION 2 |
| SE: `ReLU(inplace=True)` | `modelUnet3p.py:105` | same | **Exact** — *and contradicts the paper, see §4* |
| SE: `Conv2d((cnn+sam)//16, cnn, k=1)` | `modelUnet3p.py:106` | same | **Exact** |
| SE: `Sigmoid()` | `modelUnet3p.py:107` | same | **Exact** |
| Gating: `gated = w * cnn_feat` (multiplicative, no residual) | `modelUnet3p.py:359` | same | **Exact** |
| Cross-attention: `nn.MultiheadAttention(embed_dim=cnn, num_heads=1, batch_first=True)` | `modelUnet3p.py:109` | same | **Exact** |
| Query = flattened **gated** feature | `modelUnet3p.py:364` | same | **Exact** |
| Key/Value = flattened **ungated** encoder feature | `modelUnet3p.py:365` | same | **Exact** |
| Attention output replaces the feature; no residual, no LayerNorm | `modelUnet3p.py:369, 374` | same | **Exact** |
| Cross-attention applied at the bottleneck only | sole call site `modelUnet3p.py:368` (and the duplicate at `:493`) | `cross_attention="auto"` | ASSUMPTION 5 |
| Output projection | *does not exist upstream* | `Identity` when `out_channels == cnn_channels`, else a 1×1 conv | ASSUMPTION 3 |
| Weight init: `kaiming_normal_(a=0, mode='fan_in')` on every Conv2d; biases left at PyTorch default; `MultiheadAttention` left at its own default | `init_weights.py:29-38` applied via `modelUnet3p.py:301-305` | replicated in `_kaiming_fan_in_` | **Exact** |

### 2.2 Encoder (`eU-Net3+`), for context — **not** ported

`modelUnet3p.py:43` sets `filters = [ch, ch*2, ch*4, ch*8, ch*16]` with
`ch = config["channel"] = 32` in all four shipped configs, i.e.
**[32, 64, 128, 256, 512]**. This is a quarter of the original UNet3+ width; the code
comment at `:41` says so explicitly ("original paper uses channel size of 64, while we
use 16", which matches neither the code nor the configs — the configs say 32).

Each stage is `unetConv2` (`conv_modules_unet3p.py:8-51`): two repetitions of
`Conv2d(k=3, s=1, p=1) -> BatchNorm2d -> activation`. With `activation = "GLU"` (all
four configs) each conv emits `2*out_size` channels and `nn.GLU(dim=1)` halves them
back (`conv_modules_unet3p.py:23-26`). Downsampling between stages is
`MaxBlurPool2d(kernel_size=2)` because `use_maxblurpool = "True"` in all configs
(`modelUnet3p.py:48-69`). At the 256×256 input used throughout, the encoder feature maps
are 32@256², 64@128², 128@64², 256@32², 512@16².

The "`e`" (enhanced) in `eU-Net3+` therefore amounts to: quarter width, GLU instead of
ReLU, MaxBlurPool instead of MaxPool, and `Dropout2d(p=0.3)` on the bottleneck
(`modelUnet3p.py:25, 379`). A `FastGuidedFilter(r=2, eps=1e-2)` is constructed at
`:39` and a `use_guidedFilter` flag is read at `:35`, **but the guided filter module is
never called anywhere in the forward passes** — it is dead code, despite
`config_connic.sys:51` setting `use_guidedFilter = "True"`.

### 2.3 Decoder — **not** ported

Standard UNet3+ full-scale skip decoder (`modelUnet3p.py:152-298`). `CatChannels =
filters[0] = 32`, `CatBlocks = 5`, `UpChannels = 160`. Each decoder stage `d` gathers
five inputs — finer encoder stages via `MaxPool2d(ceil_mode=True)`, the same-scale
encoder stage directly, coarser stages via `nn.Upsample(mode='bilinear')` — each passed
through `Conv2d(·, 32, k=3, p=1) -> BatchNorm2d -> ReLU`, concatenated to 160 channels,
then `Conv2d(160, 160, k=3, p=1) -> BatchNorm2d -> ReLU`. Output head is
`Conv2d(160, num_classes=2, k=3, p=1)` (`:298`) followed by **`torch.sigmoid`**, not
softmax (`:415, :545`). Note the decoder uses plain `nn.Upsample` without
`align_corners`, which emits a PyTorch warning, and plain `MaxPool2d` rather than the
MaxBlurPool used in the encoder.

### 2.4 SAM embeddings

| Item | Finding | Source |
|---|---|---|
| SAM variant | `vit_b`, checkpoint `sam_vit_b_01ec64.pth` | `genSAMEncoding.py:35-36` |
| Frozen? | Yes, trivially — SAM is never loaded during training at all. Embeddings are precomputed offline and read from disk. | `genSAMEncoding.py`; `dataset.py:49, 108, 159` |
| How obtained | `SamPredictor.set_image(img)` then `predictor.features` | `genSAMEncoding.py:49-50` |
| Colour order | `cv2.imread` (BGR) → `cv2.cvtColor(..., COLOR_BGR2RGB)`; **not** divided by 255 (SAM's own preprocessing handles it) | `genSAMEncoding.py:47-48` |
| SAM input resolution | 1024×1024. `SamPredictor.set_image` resizes the longest side to 1024 and pads; the stored 256×256 tiles are therefore upsampled 4× before entering SAM. | `segment_anything` library behaviour, consistent with the 64×64 output |
| Embedding shape on disk | `torch.Size([1, 256, 64, 64])` | `debug_SAM.ipynb` (recorded cell output); comment at `modelUnet3p.py:420`; `dataset.py:70` calls `.squeeze(0)` |
| Cache file | `<index>_en.pt`, one per image, written next to the image with `torch.save` | `genSAMEncoding.py:51` |
| dtype | float32 (SAM default; no cast or quantisation) | `genSAMEncoding.py:51` |

Note `genSAMEncoding.py` as shipped has its directory list hard-coded to a single
MoNuSeg path (`:23`, `:40`) with the CryoNuSeg/NuInsSeg paths commented out — it must be
edited before it will generate encodings for any dataset.

### 2.5 Loss function

`focalDiceLoss` (`auxilary/lossFunctions.py:772-808`), selected by `loss = "focaldice"`
in all four configs and dispatched at `main.py:212-213`.

```
pt         = p*t + (1-p)*(1-t)
focal      = mean( -((1-pt)**2) * log(pt + 1e-8) * class_weights )     # gamma fixed at 2
intersect  = sum(p*t, dim=(0,2,3))
union      = sum(p) + sum(t) - intersect                                # NB: IoU-style union
dice       = mean( (1 - (2*intersect + 1e-6)/(union + 1e-6)) * class_weights )
loss       = alpha*dice + beta*focal
```

Two things to flag:

* The "dice" term uses an **IoU-style denominator** (`sum(p) + sum(t) - intersection`)
  inside a Dice-style `2*intersection/·` numerator (`:796-798`). This is neither the
  standard Dice nor the standard Jaccard loss.
* `alpha` and `beta` are declared as `nn.Parameter(0.5)` (`:778-779`) and described as
  "learnable", but the optimiser is constructed as
  `torch.optim.Adam(model.parameters(), ...)` (`main.py:217`) — **the criterion's
  parameters are not passed to the optimiser**, so `alpha` and `beta` never move from
  0.5. The effective loss is a plain unweighted `0.5*dice + 0.5*focal`.
* Class weights are recomputed per batch as `total / (num_classes * class_counts)`
  (`main.py:48-56, 81-83`).

### 2.6 Training hyperparameters

Read from `configs/*.sys` by the ad-hoc parser `auxilary/utils.py:63-105`.

| Parameter | CryoNuSeg | NuInsSeg | CoNIC | MoNuSeg | Source |
|---|---|---|---|---|---|
| Optimiser | Adam, `weight_decay=1e-5` | same | same | same | `main.py:217` (hard-coded, not configurable) |
| Learning rate | 1e-4 | 1e-4 | 1e-5 | 1e-5 | `config_*.sys` `learning_rate` |
| Batch size (train) | 16 | 16 | 16 | 16 | `config_*.sys`; `main.py:204` |
| Batch size (val/test) | 1 | 1 | 1 | 1 | `main.py:205, 312` |
| Epochs | 50 | 50 | 50 | 50 | `config_*.sys` |
| LR scheduler | **None** | None | None | None | see below |
| Encoder width `channel` | 32 | 32 | 32 | 32 | `config_*.sys` |
| Kernel size | 3 | 3 | 3 | 3 | `config_*.sys` |
| Activation | GLU | GLU | GLU | GLU | `config_*.sys` |
| Dropout (bottleneck) | 0.3 | 0.3 | 0.3 | 0.3 | `config_*.sys` |
| Loss | focaldice | focaldice | focaldice | focaldice | `config_*.sys` |
| `num_classes` | 2 | 2 | 2 | 2 | `config_*.sys` |
| Input | RGB 256×256, `/255` only | same | same | same | `dataset.py:38, 56-57` |
| `use_maxblurpool` | True | True | True | True | `config_*.sys` |
| `SAM_Guided` | True | True | True | True | `config_*.sys` |
| **`advancedSAM_Guided`** | **False** | **True** | **False** | *key absent* † | `config_cryonuseg.sys:38`, `config_nuinsseg.sys:38`, `config_connic.sys:35` |
| Augmentations per image | 100 | 50 | 50 | 50 | `config_*.sys` `augmentPerImage` |
| Tile size | 256×256 | 256×256 | 256×256 | 256×256 | `config_*.sys` |
| Train/val/test split | 0.7 / 0.15 / 0.15 | same | same | same | `organize_dataset.py:49-63` (the `splitRatio = 0.7` config key is unused) |
| Seeds | `torch`/`numpy`/`random` = 0, `cudnn.deterministic = True` | | | | `main.py:130-136` |

† `config_monuseg.sys` has no `advancedSAM_Guided` key at all. `modelUnet3p.py:29` reads
it unguarded (`self.advSamGuided = config["advancedSAM_Guided"]`), unlike
`use_guidedFilter` which is wrapped in a `try/except` at `:34-37`. **The shipped MoNuSeg
config therefore raises `KeyError` in the model constructor.** MoNuSeg is not one of the
three datasets evaluated in the paper, so this does not affect the published results, but
it does show the configs were not all exercised against the released code.

**No learning-rate scheduler exists.** `grep -rn "lr_decay\|scheduler\|StepLR\|CosineAnneal" --include=*.py` over the whole repository returns **zero matches**. The `lr_decay = "False"` key in every config file is read into the config dict and never used.

**Model selection** is by best validation *accuracy*, and only on epochs divisible by 5
(`main.py:286-296`) — so if the best epoch is not a multiple of 5 the checkpoint on disk
is from an earlier, worse epoch. The final test loop then evaluates `model` (the last
epoch) rather than the `best_model` it just loaded, because of a variable mix-up at
`main.py:321` (`pred = model(...)` where `best_model` was intended, constructed at
`:304-307`).

**Augmentation is entirely offline and the script that performs it is missing.** The
README points to `genAugmentation.py`; that file **is not in the repository**. The only
trace left is `log/train_augmentationLog.txt`, which is a bare list of output filenames
with no parameters. The configs record only *how many* augmentations per image
(`augmentPerImage`) and the tile size, not *which* transforms, their probabilities, or
their magnitudes. The presence of `auxilary/simplex.py` (an OpenSimplex noise fork)
suggests simplex-noise-based perturbation was involved, but nothing in the repository
calls it. **We cannot reconstruct their augmentation pipeline from the released code.**

There is no augmentation in the dataloader (`dataset.py`) — it only reads the
precomputed tiles.

### 2.7 Expected dataset folder layout

From `README.md` and confirmed by `dataset.py:38-51`:

```
<dataset>/
  ├── train/
  │     0.png          input image, 256x256 RGB
  │     0_label.png    binary groundtruth, grayscale
  │     0_en.pt        SAM ViT-B embedding, torch.Size([1, 256, 64, 64])
  │     1.png ...
  ├── val/   (same three-file pattern)
  └── test/  (same three-file pattern)
```

Indices must be a contiguous `0..N-1` integer range. The dataset length is computed as
`len(os.listdir(img_dir)) // 3 - 1` (`dataset.py:34, 93`) — i.e. it *counts files in the
directory and divides by three*, then drops one sample. Any stray file (a manifest,
`.DS_Store`, a leftover `.npy`) corrupts the sample count silently. There is a special
case bumping the divisor to 4 when `wandbProjectName == "monunet-segmenation"`
(`dataset.py:32-33`), which is a project-name string being overloaded as a data-format
switch.

---

## 3. Every ASSUMPTION made in the port

Each is marked with an `ASSUMPTION:` comment at the corresponding place in
`xgated_fusion_swain.py`.

**ASSUMPTION 1 — SAM-branch width rule (`sam_branch_channels = cnn_channels // 2`).**
Upstream hard-codes a *different* SAM-branch width at each of its five scales: 16
(`:77-83`), 32 (`:85-91`), 64 (`:94`), 128 (`:96`), and 256 (no projection, `:351`).
With the configs' `channel = 32`, the encoder widths are [32, 64, 128, 256, 512], so in
all five cases the SAM branch is exactly `cnn_channels // 2`. We adopt that as the
general rule. *Justification:* five consistent data points, and it is the only rule that
reproduces all five upstream widths with one expression. *Honest caveat:* upstream never
states this rule; it is our inference. It is exact at the upstream scales and an
extrapolation elsewhere. Overridable via `sam_branch_channels=`.

**ASSUMPTION 2 — SE bottleneck clamp.** `hidden = fused_channels // 16` would be 0 if
`fused_channels < 16`, which would make upstream crash. We clamp to 1. *Justification:*
makes the module usable at narrow widths. *Caveat:* the clamp is inactive at every
upstream scale (smallest is 32+16=48 → 3) and at all four of our scales (smallest is
64+32=96 → 6), so it changes nothing in this experiment.

**ASSUMPTION 3 — output projection.** Upstream has **no** output projection; the block
returns `cnn_channels`. Our mandated signature exposes `out_channels`, so when
`out_channels != cnn_channels` we add a bare 1×1 conv (bias, no BN, no activation,
kaiming fan-in init, matching their other 1×1 convs at `:94, :96`). When
`out_channels == cnn_channels` it is `nn.Identity()` and the port is
parameter-identical to upstream. **For a like-for-like comparison, always call the block
with `out_channels == cnn_channels`** — which is what `test_xgated.py` does for the four
scales.

**ASSUMPTION 4a — non-power-of-two downsampling.** `MaxBlurPool2d` chaining is exact for
power-of-two ratios (all upstream and all our scales). For any other ratio we fall back
to bilinear interpolation. Never triggered in this experiment.

**ASSUMPTION 4b — generic upsampling (the most significant deviation).** Upstream
upsamples the SAM embedding with a **learned** `ConvTranspose2d` (`:77-91`). Our module
cannot construct that layer in `__init__` because the required factor is only known from
runtime tensor shapes, and lazily creating parameters after the optimiser has been built
would silently exclude them from training. The default path therefore uses
**parameter-free bilinear interpolation**, which is *not* what upstream does. We provide
`sam_to_cnn_scale=` to build the exact upstream `ConvTranspose2d` (verified bit-identical
for f=2 and f=4), and the smoke test exercises both. **Any experiment that compares our
block at the 256×256 and 128×128 scales must pass `sam_to_cnn_scale=4` and
`sam_to_cnn_scale=2` respectively, or it is not comparing their operator.** The
bottleneck and 64×64 scales are unaffected.

**ASSUMPTION 5 — when cross-attention runs (`cross_attention="auto"`).** Upstream
applies cross-attention **only at the bottleneck**; the h1–h4 fusions (`:425-461`) are
gating-only. This is almost certainly forced by cost: full self/cross-attention at
256×256 needs a 65536×65536 attention matrix (~17 GB fp32 per head per sample). Our
`"auto"` mode runs attention when the token count is ≤ `max_attention_tokens` (default
4096 = 64×64) and degrades to gating-only above that — which reproduces upstream's own
coarse-vs-fine behaviour. **Consequence for the four-scale experiment: attention is
active at 64×64 and 32×32 but not at 128×128 or 256×256** (see the smoke-test output's
`x-attn` column). `"always"` and `"never"` are available for forcing either behaviour;
`"always"` at 256×256 will exhaust memory on any normal machine.

**ASSUMPTION 6 — which of the two upstream forward paths to port.** We implement
`guidedForward` (`:364-365`, key/value from the **ungated** feature = genuine
cross-attention). `advGuidedForward` (`:489-490`) builds query *and* key/value from the
gated tensor, which is self-attention, not cross-attention. We treat that as a
copy-paste bug rather than a design choice because (a) it contradicts the method's own
name, (b) `guidedForward` does it the other way two hundred lines earlier, and (c) the
CryoNuSeg and CoNIC configs select `guidedForward`. **This is a judgement call and we
flag it as such.** If a reviewer prefers the other reading, the change is one line.

**ASSUMPTION 7 — ReLU, not GLU, in the SE block.** See §4.1. We followed the code.

---

## 4. Where their code does not match their paper

These are not defects in our port. They are discrepancies in reference [21] itself, and
they are stated here because the response letter may need to be precise about what "their
architecture" means.

### 4.1 The paper says GLU in the gating block; the code uses ReLU

The project page (`docs/index.html`, Method section) states the X-GFB *"first uses GLU in
gated squeeze and excitation block and then uses cross-attention block"*, and the
architecture figure `docs/static/images/model_arch.png` carries a box labelled **GLU**
inside the fusion block.

The released SE block is `AdaptiveAvgPool2d → Conv1×1 → **ReLU** → Conv1×1 → Sigmoid`
(`modelUnet3p.py:102-108`, and identically at `:114-120`, `:124-130`, `:134-140`,
`:144-150`). `grep -rn "GLU" --include=*.py` over the entire repository matches **only**
`conv_modules_unet3p.py:23-26` and `:35-37`, which is `unetConv2` — the plain encoder
convolution block, a different component. The same is true in the superseded
`modelUnet3p_old.py`.

So the "GLU" the paper refers to in the X-GFB either (a) describes a version not
released, or (b) is a mislabelling of the encoder's GLU activation. **Our port follows
the code (ReLU).** We state this explicitly rather than quietly "fixing" the port to
match the prose, because a port that used GLU would not reproduce anything that exists.

### 4.2 The paper figure shows bottleneck-only fusion; one config enables multi-scale fusion

`model_arch.png` shows a single fusion block between encoder and decoder. That matches
`advancedSAM_Guided = "False"` (CryoNuSeg, CoNIC, MoNuSeg). But
`config_nuinsseg.sys:38` sets `advancedSAM_Guided = "True"`, which routes to
`advGuidedForward` and injects SAM gating at **all five** encoder stages. **The NuInsSeg
result and the CryoNuSeg/CoNIC results therefore come from two structurally different
models**, which the paper figure does not convey. We found no ablation in the repository
distinguishing them.

### 4.3 `advGuidedForward`'s cross-attention is self-attention

`modelUnet3p.py:489-490`:

```python
gated_unet_features_flat = gatedFeatures.view(batch_size, channels, -1).permute(0, 2, 1)
unet_features_flat       = gatedFeatures.view(batch_size, channels, -1).permute(0, 2, 1)
```

Both operands are `gatedFeatures`. The equivalent lines in `guidedForward` (`:364-365`)
use `gatedFeatures` and `hd5`. Since NuInsSeg is the config that uses
`advGuidedForward`, **the NuInsSeg number in the paper was produced with self-attention
at the bottleneck, not cross-attention.**

### 4.4 Other defects noted in passing

* `use_guidedFilter` is read and a `FastGuidedFilter` is instantiated
  (`modelUnet3p.py:35-39`) but never called. `config_connic.sys:51` sets it to `True`,
  so the CoNIC configuration advertises a component that does nothing.
* The criterion's `alpha`/`beta` are `nn.Parameter`s never given to the optimiser
  (§2.5), so the "learnable loss weighting" is fixed at 0.5/0.5.
* The final test loop evaluates the last-epoch model, not the loaded best model
  (`main.py:321` vs `:304-307`).
* Checkpoints are only saved on epochs divisible by 5 (`main.py:291`).
* The output head is `sigmoid` over 2 channels (`:415`) rather than softmax, so the two
  class scores are independent and need not sum to 1 — while the loss treats them as a
  one-hot distribution.
* `README.md` instructs `python train_test.py --expt_dir ...`; **`train_test.py` does not
  exist**. The working entry point is `python inference.py --expt_dir ...`.
* `genAugmentation.py`, referenced in the README, **does not exist** (§2.6).
* `inference_cryonuseg.py:11-20` hard-codes ten absolute experiment paths from the
  authors' own machine.
* No pretrained weights are released; the repository contains no checkpoint and no link
  to one.

---

## 5. What we deliberately did not reproduce, and why

| Not reproduced | Why |
|---|---|
| The `eU-Net3+` encoder and the UNet3+ full-scale decoder | Out of scope. The controlled experiment substitutes *their fusion operator* into *our* encoder/decoder at four scales. Porting their backbone as well would confound the fusion comparison with a backbone comparison. |
| The guided filter | Dead code upstream — never called (§4.4). Reproducing it would be reproducing nothing. |
| `focalDiceLoss` | We compare under our own training protocol; mixing their loss into our pipeline would confound the fusion comparison. Fully documented in §2.5 so it can be added if a reviewer asks. |
| Their offline augmentation pipeline | **Impossible** — `genAugmentation.py` is absent from the repository (§2.6). |
| Their data split | Their `organize_dataset.py` performs a non-stratified, filename-order 70/15/15 split over a different directory structure. We use our own pre-registered split manifests. |
| Precomputation of SAM embeddings to `_en.pt` files | Documented (§2.4) but not ported; our pipeline computes or caches SAM features by its own route. The embedding *content* (ViT-B, 1024×1024 input, 256×64×64 output) is matched. |
| The `advGuidedForward` self-attention variant | We implement the `guidedForward` (true cross-attention) variant, per ASSUMPTION 6. |

---

## 6. Running the official code end to end

**Assessment: no — not without substantial modification, and even then the result would
not be a faithful reproduction of their published numbers.** Per the task brief we did
not attempt to run their training; this assessment is from reading the code.

### 6.1 Hard blockers

1. **The augmentation script is missing.** `genAugmentation.py` is referenced by the
   README and absent from the repository (§2.6). Their configs call for 100
   augmentations per image on CryoNuSeg and 50 on NuInsSeg/CoNIC, and *all* their
   training data is these offline augmented tiles — the dataloader applies no
   augmentation of its own. Without that script we cannot build the training set they
   trained on, and we cannot even describe the transforms they used. **This alone makes
   a true end-to-end reproduction of their published numbers impossible from the
   released code.**

2. **Dependencies are not installable as pinned.** `requirements.txt` omits
   `segment_anything` (required by `genSAMEncoding.py:10`) and `guided_filter_pytorch`
   (imported unconditionally at `modelUnet3p.py:8`, so the model file will not import
   without it, even though the filter is never used). It also pins `torch==2.2.0` with
   a full set of `nvidia-*-cu12` Linux wheels that do not resolve on Windows. A working
   environment has to be assembled by hand.

3. **No pretrained weights.** Nothing can be evaluated without full retraining.

### 6.2 Precomputed SAM encodings: required

`dataset.py:49, 108, 159` unconditionally `torch.load(<index>_en.pt)` in all three
dataset classes. There is no on-the-fly SAM path anywhere in the codebase. We would have
to run `genSAMEncoding.py` (after editing its hard-coded directory list at `:23`/`:40`)
over every augmented tile, writing one ~4 MB float32 `.pt` per tile. For CryoNuSeg this
is 30 images × 100 augmentations ≈ 3,000 files ≈ **12 GB**; for CoNIC's 4,981 images × 50
≈ 249,000 files ≈ **1 TB**. The storage cost is a practical obstacle for CoNIC, less so
for CryoNuSeg/NuInsSeg.

Note also that the failed-load path is a bare `except:` that only *prints* and then falls
through (`dataset.py:50-51`), leaving `samencoding` undefined — so a missing `_en.pt`
produces a confusing `UnboundLocalError` rather than a clear error.

### 6.3 Can their loader be pointed at our split manifests? No, not as written

Their loader is **not** manifest-driven. It:

* enumerates samples as contiguous integers `0..N-1` (`dataset.py:38, 43, 49`);
* derives the sample count by counting *files in the directory* and dividing by 3, then
  subtracting 1 (`dataset.py:34, 93`);
* requires image, label and embedding to be siblings in one flat directory;
* has a divisor that changes to 4 based on the **wandb project name** string
  (`dataset.py:32-33`).

Pointing it at our splits requires either (a) materialising each of our splits as a flat
renumbered directory of `i.png` / `i_label.png` / `i_en.pt` triplets — duplicating all
image data and all SAM embeddings per split — or (b) replacing `nucleiDataset`,
`nucleiValDataset` and `nucleiTestDataset` with manifest-driven versions. Option (b) is
~60 lines and is the sane choice, but it means we are no longer running *their*
dataloader.

### 6.4 Config and harness friction

* The config parser (`auxilary/utils.py:63-105`) is a hand-rolled line splitter, not
  TOML/INI: it splits on the first `=` and the first `#`, and infers type by looking for
  a `"` or a `.`. A path containing `#`, or a float written without a decimal point
  (e.g. `learning_rate = 1e-4`), silently parses wrong. `1e-4` would be read as the
  *string-less* branch, find no `.`, and crash in `int()`.
* `logWandb = "True"` in all four configs; `main.py:161-162` calls `wandb.init`
  unconditionally when set. Needs a wandb login or `WANDB_MODE=offline`.
* Dataset paths are absolute `/mnt/Datasets/...` Linux paths in every config.
* `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` is set at `main.py:143`, *after* imports but
  before device selection; single-GPU only.
* Test-time images are resized to 256×256 with `INTER_CUBIC` **including the label mask**
  (`dataset.py:167-169`) before binarisation — cubic interpolation of a binary mask
  introduces ringing, then `toBinary` thresholds it. This is a real evaluation-protocol
  difference from ours.
* `config_cryonuseg.sys` alone cannot reproduce their CryoNuSeg table entry, which
  `inference_cryonuseg.py` shows was a 10-fold cross-validation over ten separate
  training runs. No fold definition is in the repository.

### 6.5 Realistic options

| Option | Effort | What it yields |
|---|---|---|
| **A.** Run their `main.py` unmodified | Not possible | Blocked by §6.1. |
| **B.** Rebuild environment, write our own augmentation, precompute `_en.pt`, replace the dataloader, run their model + loss on our splits | High (several days + ~12 GB for CryoNuSeg; CoNIC impractical) | A genuine end-to-end baseline of **their architecture under our data protocol and a guessed augmentation pipeline**. Will not reproduce their published numbers and should not be presented as doing so. |
| **C.** Our port of the fusion operator into our encoder/decoder (what we have built) | Done | A clean, controlled comparison of **the fusion operator**, with everything else held constant. |

**Recommendation.** Report option C as the primary comparison and describe option A's
blockers factually in the response letter. If Reviewer 4 insists on a full
architecture-level baseline, option B is feasible for CryoNuSeg and NuInsSeg but must be
reported with the augmentation gap stated plainly, because that gap is in *their*
release, not in our effort.

---

## 7. What can and cannot be claimed from a comparison against this port

### Can be claimed

* That we benchmarked **the X-Gated Fusion operator of [21] as implemented in the
  authors' own released code**, at commit `237669b4`, and that our implementation of the
  bottleneck configuration is **bit-identical** to theirs given identical weights
  (§1.1) — not a simplified stand-in, and not a reimplementation from the paper text.
* That the comparison isolates the fusion operator: encoder, decoder, loss, data,
  augmentation, optimiser, schedule and evaluation protocol are held constant across
  arms, so a difference in results is attributable to the fusion operator and not to a
  backbone or training-recipe difference.
* That we additionally evaluated their operator at three scales beyond the single
  bottleneck placement they used, which is strictly more favourable to [21] than their
  own published configuration for CryoNuSeg and CoNIC.
* Specific, checkable statements about their design: channel-only gating with no spatial
  component; cross-attention with ungated key/value; no residual, no normalisation
  inside the block; ReLU rather than the GLU claimed in the paper; `vit_b` SAM at
  1024×1024 producing 256×64×64 embeddings, frozen by virtue of being precomputed
  offline.

### Cannot be claimed

* **That we reproduced the numbers in [21].** We did not retrain their full model on
  their data with their augmentation. Their augmentation script is not in the release,
  so nobody can.
* **That our result is a verdict on their complete method.** We ported one block. Their
  reported gains also depend on the `eU-Net3+` backbone (quarter width, GLU,
  MaxBlurPool), their `focalDiceLoss`, 50–100× offline augmentation, and their
  split/evaluation protocol. A difference we measure is a difference in the fusion
  operator **within our system**, not a refutation of their paper.
* **That the fine-scale arms test their operator in full.** At 128×128 and 256×256 the
  block runs gating-only, because cross-attention at those resolutions is
  computationally infeasible — the same reason they confined it to the bottleneck.
  Those two arms test their *gated-SE* component, not the full X-GFB. This must be
  stated wherever those two rows appear.
* **That the 256×256 and 128×128 arms used their exact SAM-upsampling layer**, unless
  `sam_to_cnn_scale=4` / `sam_to_cnn_scale=2` was passed. With the default the SAM branch
  is bilinearly interpolated rather than passed through their learned transposed
  convolution (ASSUMPTION 4b).
* **That performance differences reflect tuning effort parity.** We did not tune
  hyperparameters for their operator beyond matching their stated values. If their
  operator underperforms in our system, the honest statement is that it underperforms
  *under our training protocol*, not that it is inferior in general.
* **Anything about the GLU variant described in their paper.** It is not in the released
  code and we did not invent it.

---

## 8. Reproducing this report

```bash
cd <LOCAL_PATH>
git -C upstream rev-parse HEAD      # 237669b4d2e8eca6b1517bd522b8d4c0246ffa50
python test_xgated.py
```

Smoke-test output at the time of writing (PyTorch 2.10.0+cu130, CPU, batch 2,
SAM feature 256×64×64):

```
 cnn_ch   spatial  sam_branch  se_hidden  x-attn       params              out shape
------------------------------------------------------------------------------------
     64  256x256           32          6   False       25,894      (2, 64, 256, 256)
    128  128x128           64         12   False       86,476     (2, 128, 128, 128)
    256   64x64           128         24    True      311,704       (2, 256, 64, 64)
    512   32x32           256         48    True    1,112,624       (2, 512, 32, 32)
------------------------------------------------------------------------------------
  total                                             1,536,698
```
