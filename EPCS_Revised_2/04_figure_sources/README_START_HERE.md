# Redrawing the four EPCS-Net schematic figures in Excalidraw

Files here: Figure_01.excalidraw ... Figure_04.excalidraw (editable sources), `_exported_previews/` (the current PNGs, for comparison), FIGURE_SPECS.md (captions, every text block, layout).

## Open a file
1. Go to https://excalidraw.com
2. Menu (top-left, three lines) > Open, choose the .excalidraw file. Or drag the file onto the canvas.
3. Press Shift+1 ("zoom to fit") to see the whole figure.

The files use roughness 0 (clean straight lines) and the Helvetica font (Excalidraw fontFamily 2). Excalidraw will show them exactly in that style; no hand-drawn wobble unless you change "Sloppiness" in the style panel (keep it at Architect for the paper).

## Edit text
Double-click a box (or any text) and type. Text inside a box stays centred; the box does not auto-grow in every case, so drag its handles if text overflows. Use Enter for new lines. Keep the font Helvetica ("Normal") and sizes consistent (body 16, small 14, titles 20).

## Export (for each figure)
Menu > Export image (or Ctrl+Shift+E), then:
- Background: ON, white
- Scale: 3x
- Embed scene: OFF
- Dark mode: OFF
- Export "Selected only" is irrelevant if nothing is selected.
Save as PNG; then export again as SVG.

## File names to send back
| Figure | Send back | Goes to |
|---|---|---|
| 1 pipeline | fig1_pipeline.png + fig1_pipeline.svg | PAPER_OVERLEAF/figures/ |
| 2 PSPI | fig2_pspi.png + fig2_pspi.svg | PAPER_OVERLEAF/figures/ |
| 3 ACFB | fig3_acfb.png + fig3_acfb.svg | PAPER_OVERLEAF/figures/ |
| 4 FSFD | fig4_fsfd.png + fig4_fsfd.svg | PAPER_OVERLEAF/figures/ |

Same names, so EPCS_revise3.tex needs no edit. Please also send back your edited .excalidraw files.

## Width the tex uses (keep the aspect ratio close to the current one)
| Figure | includegraphics options | Current PNG px (w x h) | w:h |
|---|---|---|---|
| fig1 (fig:pipeline) | width=0.997\linewidth, height=0.82\textheight, keepaspectratio | 1103 x 1690 | 0.65 (tall; height-limited, a page of its own) |
| fig2 (fig:pspi) | width=0.460\linewidth | 1262 x 846 | 1.49 (wide, small on the page, so keep text large) |
| fig3 (fig:acfb) | width=0.486\linewidth, height=0.55\textheight, keepaspectratio | 928 x 1103 | 0.84 |
| fig4 (fig:fsfd) | width=0.545\linewidth | 1038 x 976 | 1.06 |

Figures 2, 3 and 4 print at roughly half the text width, so minimum font should stay readable (about 7 pt printed). Do not shrink text when redrawing; make boxes larger instead. All figures use [H] placement.

NOTE (v3 redesign): new PNG sizes: fig1 1308x1512 (0.86), fig2 1032x820 (1.26), fig3 748x1147 (0.65), fig4 956x888 (1.08). See 07_revision3/05_manuscript/figures/FIGURE_REDESIGN_NOTES.md.

## Facts that must stay true when redrawing
- SAM input: the 256x256 patch is upsampled to 1024x1024; frozen ViT-B gives a 64x64 embedding with 256 channels ([B,256,64,64]).
- Local encoder: DoubleConv stages, base 64, channels 64/128/256/512 (BatchNorm2d inside DoubleConv).
- ACFB: one per scale (four in total); GroupNorm(8) only inside ACFB; the residual is taken on the post-DKSR tensor (X_ms), not X_cross.
- Decoder widths 512/256/128/64; the fusion input at scale 3 has 960 channels (64+128+256+512); output 256x256.
- Training: batch size 1, AdamW, lr 3e-4, weight decay 1e-4, 50 epochs, no scheduler.
- Loss FTAS = 0.3 BCE + 0.7 focal-Tversky.
- Only augmentation: horizontal flip, p = 0.5.
- Checkpoint selection on validation Dice only; the test split is used once.
(Source of truth: 07_revision3/00_plan/METHODS_CODE_FIDELITY.md and the code in 02_code_RunA.)

## Things to know when editing (added after validation)
* **Arrows are not attached to boxes.** The files open fine, but if you move a box its arrows will not follow; move the arrows by hand (or delete and redraw them with the arrow tool, which snaps to boxes).
* **Text may shift slightly** when you edit it, because Excalidraw re-measures text boxes. Check each figure after editing.
* **Figure titles are drawn inside the figures** ("Figure N. ..."). The paper's caption already carries this, so you can delete the title and subtitle text before exporting if you want a clean figure.
* The files use clean straight lines (roughness 0), not Excalidraw's hand-drawn style. Keep this for a journal figure.
* Figures 2 to 4 print at about half the text width, so keep fonts large when you redraw them.
