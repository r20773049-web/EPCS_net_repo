# Figure specifications

Texts are extracted verbatim from the .excalidraw files (they match the tex captions' facts). Line breaks inside a block are shown as new lines.


## Figure 1: Figure_01.excalidraw -> fig1_pipeline.png (\label{fig:pipeline})

**Purpose.** Explain the whole method end to end: data loader, two branches, PSPI, ACFB, FSFD, loss, training and evaluation protocol, post-hoc DPNP.

**Caption in the tex (EPCS_revise3.tex).**

> EPCS-Net complete methodology pipeline showing the dual-branch architecture with PSPI, ACFB fusion, full-scale decoder, FTAS loss, DPNP explainability, and the training protocol actually used to produce the results in Section~ef{sec:results} (batch size 1, fixed learning rate, no scheduler, FP32 throughout).

**Layout.** Tall single column, top to bottom: title and subtitle; OSPT loader; preprocessing; two parallel branches (SAM left, local encoder right) with branch labels; PSPI box (left) and four-scale box (right); ACFB with step list; fused maps; FSFD; output; FTAS; training protocol; evaluation protocol; DPNP header with three boxes (Grad-CAM, PCA, HDBSCAN) in a row. 18 arrows.

**Text blocks** (top to bottom; [free] = standalone label, [box] = inside a rectangle):

1. [free]
```
Figure 1.  EPCS-Net methodology pipeline
```

2. [free]
```
Dual-branch architecture: frozen SAM global prior + EPCS-Net local DoubleConv features, fused at four scales by ACFB,
decoded full-scale, supervised by FTAS. Geometry and hyperparameters below are those of the released implementation.
```

3. [box]
```
Organ-Stratified Pan-Cancer Training (OSPT) data loader
WeightedRandomSampler   weight(i) = 1 / count(organ of i)   with replacement
Every organ therefore has equal draw probability 1/K per sample (K = number of organs)
```

4. [box]
```
H&E preprocessing
Resize to 256 x 256 (bilinear)   scale to [0,1], no mean/std on the local branch
SAM branch applies its own preprocess   only augmentation: horizontal flip, p = 0.5 (training)
```

5. [free]
```
SAM branch
```

6. [free]
```
local branch
```

7. [box]
```
FROZEN SAM ViT-B ENCODER
patch upsampled 256 -> 1024 (bilinear),
then official SAM preprocess
output embedding  [B, 256, 64, 64]
image encoder only, frozen, single precision, no gradient
```

8. [box]
```
EPCS-Net LOCAL ENCODER
4 x DoubleConv stages (base 64: 64/128/256/512), MaxPool2d
DoubleConv = [Conv 3x3 - BatchNorm2d - ReLU] x 2
local morphological features
```

9. [box]
```
PSPI  --  Pyramid-Scale Prior Injection
SAM embedding, bilinear (align_corners=False) to all four scales
-> 256x256   -> 128x128   -> 64x64   -> 32x32
```

10. [box]
```
Four encoder scales
x1 [B, 64, 256, 256]      x2 [B, 128, 128, 128]
x3 [B, 256, 64, 64]        x4 [B, 512, 32, 32]
```

11. [box]
```
Step 1   channel projection, 1x1 conv both streams
Step 2   bidirectional dual-stream gating (BDSG)
Step 3   dual-kernel refinement, 3x3 + 5x5 (DKSR)
Step 4   squeeze-and-excitation channel attention
Step 5   residual + GroupNorm(8)
```

12. [box]
```
ACFB  --  Adaptive Cross-stream Fusion Block
one independent module per encoder scale, four in total
fuses the SAM prior with the CNN stream at that scale
```

13. [box]
```
Fused feature maps at all four scales   Phi1 [64ch]   Phi2 [128ch]   Phi3 [256ch]   Phi4 [512ch]
```

14. [box]
```
FSFD  --  Full-Scale Fused Decoder
every encoder scale feeds every decoder level
960-channel fusion input at scale 3, decoder widths 512/256/128/64, upsampling to 256 x 256
```

15. [box]
```
Segmentation output   1x1 conv head -> sigmoid -> binary nuclei mask at threshold 0.5
```

16. [box]
```
FTAS loss  --  Focal Tversky Asymmetric Supervision
0.3 x BCE  +  0.7 x focal-Tversky   (alpha = 0.7, beta = 0.3, gamma = 0.75)
```

17. [box]
```
TRAINING PROTOCOL
50 epochs   batch size 1   AdamW, fixed LR 3e-4, weight decay 1e-4
no scheduler   no gradient clipping   single precision   deterministic CUDA seeding
```

18. [box]
```
EVALUATION PROTOCOL
organ-stratified train / validation / test split   checkpoint chosen on VALIDATION Dice only
all reported scores computed on the untouched TEST split   every model x 5 matched seeds   paired tests across seeds
```

19. [box]
```
DPNP  --  Deep Phenotypic Nuclear Profiling   (post-hoc, exploratory, inference only)
```

20. [box]
```
Grad-CAM
spatial agreement between
model attention and nuclei
```

21. [box]
```
PCA
connected-region feature
variance structure
```

22. [box]
```
HDBSCAN
density clustering and
outlier flagging
```


## Figure 2: Figure_02.excalidraw -> fig2_pspi.png (\label{fig:pspi})

**Purpose.** Show that one SAM embedding is resized to every encoder scale and fused at each scale, not only at the bottleneck.

**Caption in the tex (EPCS_revise3.tex).**

> Pyramid-Scale Prior Injection (PSPI) showing SAM embedding interpolated to all four encoder resolution scales with corresponding ACFB fusion at each level.

**Layout.** Wide: SAM embedding box on top with the interpolation note to its right; four scale boxes in a row, each an arrow target; encoder-features note; row of four ACFB boxes; row of four fused Phi boxes; FSFD box at the bottom with arrows converging. 18 arrows.

**Text blocks** (top to bottom; [free] = standalone label, [box] = inside a rectangle):

1. [free]
```
Figure 2.  Pyramid-Scale Prior Injection (PSPI)
```

2. [free]
```
A single frozen SAM embedding is bilinearly resized to all four encoder resolutions and fused independently at each
scale, instead of being injected only at the bottleneck as in prior SAM-guided designs.
```

3. [free]
```
bilinear interpolation to each encoder scale
(F.interpolate, align_corners = False)
```

4. [box]
```
Frozen SAM ViT-B encoder output
[B, 256, 64, 64]      (256x256 patch upsampled to 1024x1024 before the encoder)
```

5. [box]
```
Scale 1
256 x 256
```

6. [box]
```
Scale 2
128 x 128
```

7. [box]
```
Scale 3
64 x 64
```

8. [box]
```
Scale 4
32 x 32
```

9. [free]
```
EPCS-Net local encoder features
x1 .. x4  enter here
```

10. [box]
```
ACFB fuse 1
64 ch  .  256 x 256
```

11. [box]
```
ACFB fuse 2
128 ch  .  128 x 128
```

12. [box]
```
ACFB fuse 3
256 ch  .  64 x 64
```

13. [box]
```
ACFB fuse 4
512 ch  .  32 x 32
```

14. [box]
```
Phi 1 fused
64 ch  .  256 x 256
```

15. [box]
```
Phi 2 fused
128 ch  .  128 x 128
```

16. [box]
```
Phi 3 fused
256 ch  .  64 x 64
```

17. [box]
```
Phi 4 fused
512 ch  .  32 x 32
```

18. [box]
```
Full-Scale Fused Decoder (FSFD)
```


## Figure 3: Figure_03.excalidraw -> fig3_acfb.png (\label{fig:acfb})

**Purpose.** Show the five internal steps of one ACFB with their equations.

**Caption in the tex (EPCS_revise3.tex).**

> Adaptive Cross-stream Fusion Block (ACFB) showing the five sequential steps: channel projection, bidirectional dual-stream gating (BDSG), dual-kernel spatial refinement (DKSR), squeeze-and-excitation channel attention, and residual connection (on the post-DKSR tensor) with Group Normalisation.

**Layout.** Tall vertical stack: two input boxes side by side (SAM prior left, local features right) merging by elbow arrows into Step 1; Steps 2 to 5 stacked; fused output; implementation note at the bottom. 7 arrows.

**Text blocks** (top to bottom; [free] = standalone label, [box] = inside a rectangle):

1. [free]
```
Figure 3.  Adaptive Cross-stream Fusion Block (ACFB)
```

2. [free]
```
One ACFB instance per encoder scale k. Shapes are written for scale k with C_k channels at H_k x W_k.
Normalisation inside this block is GroupNorm(8); the encoder DoubleConv elsewhere uses BatchNorm2d.
```

3. [box]
```
SAM prior at scale k
F_S^k   .   256 channels   .   H_k x W_k
```

4. [box]
```
Local encoder features at scale k
X_L^k   .   C_k channels   .   H_k x W_k
```

5. [box]
```
STEP 1   Channel projection
1x1 conv brings both streams to C_k channels
X^ = W_x * X_L^k        S^ = W_s * F_S^k
```

6. [box]
```
STEP 2   Bidirectional Dual-Stream Gating  (BDSG)
g_s = sigmoid(W_gs * S^)        g_x = sigmoid(W_gx * X^)
X_cross = X^ (*) g_s  +  S^ (*) g_x
each stream gates the other, so neither can dominate
```

7. [box]
```
STEP 3   Dual-Kernel Spatial Refinement  (DKSR)
X_ms = Conv3x3(X_cross) + Conv5x5(X_cross)
3x3 captures nuclear boundary detail, 5x5 widens the context
```

8. [box]
```
STEP 4   Squeeze-and-excitation channel attention
z = GAP(X_ms) -> Conv(C_k -> h) -> ReLU -> Conv(h -> C_k) -> sigmoid
h = max(C_k / 4, 8)        X_att = X_ms (*) s
```

9. [box]
```
STEP 5   Residual shortcut + GroupNorm(8 groups)
Phi^k = GroupNorm( W_out * X_att  +  X_ms )
note: the residual is taken on the post-DKSR tensor X_ms
```

10. [box]
```
Fused output   Phi^k   .   C_k channels   .   H_k x W_k
```

11. [free]
```
Implementation note: the residual adds the post-DKSR tensor X_ms (not X_cross).
GroupNorm(8) is used only inside ACFB; the DoubleConv blocks use BatchNorm2d.
```


## Figure 4: Figure_04.excalidraw -> fig4_fsfd.png (\label{fig:fsfd})

**Purpose.** Show that each decoder level receives every encoder scale and where the 960-channel input arises.

**Caption in the tex (EPCS_revise3.tex).**

> Full-Scale Fused Decoder (FSFD) with multi-scale skip connections showing the 960-channel concatenation at decoder scale 3 and progressive reconstruction to $256{	imes}256$ output resolution.

**Layout.** Four Phi boxes in a row at top; decoder column d4 (right) down to d1 with dashed full-scale skip arrows from each Phi to each decoder level; 1x1 head at the bottom; key at right. 11 arrows (solid = decoder path, dashed = skip).

**Text blocks** (top to bottom; [free] = standalone label, [box] = inside a rectangle):

1. [free]
```
Figure 4.  Full-Scale Fused Decoder (FSFD)
```

2. [free]
```
Every decoder level receives every encoder scale. All skip connections carry SAM-fused features Phi^k,
so global context is present at every reconstruction step.
```

3. [free]
```
SAM-enriched fused encoder features, output of ACFB
```

4. [box]
```
Phi 1
64 ch  .  256 x 256
```

5. [box]
```
Phi 2
128 ch  .  128 x 128
```

6. [box]
```
Phi 3
256 ch  .  64 x 64
```

7. [box]
```
Phi 4
512 ch  .  32 x 32
```

8. [box]
```
d4 = DoubleConv(Phi 4)
512 ch  .  32 x 32
```

9. [box]
```
d3 = DoubleConv( cat[ maxpool4(Phi1), maxpool2(Phi2), Phi3, up(d4) ] )
64 + 128 + 256 + 512 = 960 channels  ->  256 ch  .  64 x 64
```

10. [box]
```
d2 = DoubleConv( cat[ maxpool2(Phi1), Phi2, up(d3) ] )
64 + 128 + 256 = 448 channels  ->  128 ch  .  128 x 128
```

11. [free]
```
Key
solid  = decoder path
dashed = full-scale skip
```

12. [box]
```
d1 = DoubleConv( cat[ Phi1, up(d2) ] )
64 + 128 = 192 channels  ->  64 ch  .  256 x 256
```

13. [box]
```
1x1 conv head -> logits  .  1 x 256 x 256
```


## Optional extra diagram the paper could use: Study design

Purpose: one picture of how data are split and used, so a reader sees at a glance that checkpoint selection never touches the test split. Not created; outline for you to draw (suggest about 1000 x 800 px, wide, Helvetica, same colours as the other figures).

Blocks, top to bottom:
1. Two dataset boxes side by side.
   - Left: "NuInsSeg: 665 images, 31 organs; one organ-stratified split: train 459 / validation 103 / test 103".
   - Right: "CryoNuSeg: 30 images, 10 tissues; 3-fold tissue-stratified cross-validation: per fold train 15 / validation 5 / test 10".
2. Arrow down from each to a box "Five seeds per split or fold (NuInsSeg 5 runs; CryoNuSeg 3 folds x 5 = 15 runs), every model on the same matched seeds".
3. Box "Training (train split only): 50 epochs, batch size 1, AdamW lr 3e-4, wd 1e-4, flip p=0.5; EPCS-Net uses FTAS, baselines use BCE+Dice".
4. Box "Validation split: Dice after each epoch -> best-validation-Dice checkpoint kept (the only selection step)".
5. Box "Test split: used once, with the selected checkpoint; test Dice/IoU and other measures".
6. Side column "Models compared": EPCS-Net; two reference-21 variants (FTAS loss); five baselines (BCE+Dice); MedSAM frozen (zero-shot and oracle-box, not ranked, no seeds).
7. Bottom box "Statistics: paired tests across matched runs, Holm correction; fold-level tests on CryoNuSeg".
Use a dashed "never used for selection" arrow marker from Test split back to the checkpoint box, crossed out. Verify the exact baseline names and counts against the tex Tables before drawing.
