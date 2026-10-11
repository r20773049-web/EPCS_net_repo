# EPCS_Revised_2 - EPCS-Net: SAM-guided multi-scale nuclei segmentation

This folder (`EPCS_Revised_2`) of the repository `EPCS_net_repo` holds the code, per-run results and checkpoint manifest of the **second revision** of the paper

> *EPCS-Net: SAM-Guided Multi-Scale Semantic Nuclei Segmentation and Exploratory Phenotypic Profiling for Pan-Cancer Analysis* (Scientific Reports).

Code and results of the earlier submission stay in the other folders of the repository and are not changed. Numbers in this folder replace those of the earlier submission.

* Archived release (code and results, Zenodo DOI): `10.5281/zenodo.XXXXXXX`
* Trained checkpoints (Zenodo DOI): `10.5281/zenodo.YYYYYYY`
* Release tag: `vX.Y.Z`, commit: `<hash>`

## What the project is
EPCS-Net segments cell nuclei in H&E-stained histopathology images. A frozen Segment Anything (SAM ViT-B) image encoder supplies global context; a local DoubleConv encoder supplies fine detail. Three components combine them:
* **PSPI** (pyramid-scale prior injection): the SAM embedding is resized to every encoder scale (256, 128, 64, 32 px).
* **ACFB** (adaptive cross-stream fusion block, modified from the X-Gated Fusion Block of reference 21): bidirectional gating, 3x3 and 5x5 refinement, squeeze-and-excitation, residual with GroupNorm (GroupNorm only inside ACFB).
* **FSFD** decoder (full-scale fused decoder) and the **FTAS** loss (0.3 BCE + 0.7 focal Tversky).
Training sampler OSPT (organ-balanced sampling in expectation); exploratory DPNP phenotyping (PCA, HDBSCAN, stability and confounder checks) and Grad-CAM are provided as analyses, not as clinical claims.

## Evaluation protocol of this revision
* Datasets: NuInsSeg (665 images, 31 organs; organ-stratified split 459/103/103) and CryoNuSeg (30 images, 10 tissues; three tissue-stratified folds 15/5/10, so that every tissue is tested).
* Five seeds (42, 123, 456, 789, 2024); the checkpoint is chosen on validation Dice only; the test split is scored once.
* Models: U-Net, U-Net++, U-Net3+, eU-Net3+ (no SAM), SAM-guided baseline, reference-21 fusion (bottleneck and four scales), EPCS-Net. The baselines use BCE+Dice and EPCS-Net and reference-21 use FTAS, so loss and architecture differ together.

## Main results (test Dice, mean +- SD over runs; full tables in `MASTER_RESULTS.xlsx`)
| Model | CryoNuSeg (15 runs) | NuInsSeg (5 seeds) |
|---|---|---|
| U-Net | 0.7389 +- 0.0256 | 0.6070 +- 0.0171 |
| U-Net++ | 0.7152 +- 0.0262 | 0.5950 +- 0.0146 |
| U-Net3+ | 0.7230 +- 0.0415 | 0.6071 +- 0.0220 |
| eU-Net3+ (no SAM) | 0.7454 +- 0.0213 | 0.5900 +- 0.0197 |
| SAM-guided baseline | 0.7777 +- 0.0153 | 0.6372 +- 0.0235 |
| Reference-21 fusion, bottleneck | 0.7637 +- 0.0217 | 0.6049 +- 0.0190 |
| Reference-21 fusion, four scales | 0.7590 +- 0.0187 | 0.5676 +- 0.0341 |
| **EPCS-Net** | **0.7899 +- 0.0170** | **0.6797 +- 0.0208** |

EPCS-Net has the highest mean Dice on both datasets. The margin over the SAM-guided baseline is +0.012 on CryoNuSeg (ahead in 3 of 3 folds) and +0.043 on NuInsSeg; on NuInsSeg the seed-level paired test is not significant (Holm p = 0.070) while the image-level bootstrap excludes zero. The ablation (CryoNuSeg fold 1, five seeds) is non-monotonic and does not show each component to be necessary. Paired tests, bootstrap, Grad-CAM, phenotyping, instance-merge analysis, MedSAM evaluation and latency are in `03_results/analysis/`, `03_results/medsam_r3/` and `03_results/latency/`.

## Folder map
| Path | Content |
|---|---|
| `MASTER_RESULTS.xlsx` | master workbook: every run, averages per model, best values, ablation, MedSAM, significance tests, evaluation log |
| `MASTER_RESULTS_all_runs.csv` | one row per trained run (215 runs): dataset, model, fold, seed, best epoch, test and validation metrics, parameters, time |
| `MASTER_RESULTS_summary_by_model.csv` | mean, SD and 95% CI per model and dataset |
| `01_source_code/` | `shared/` (models, protocol, splits, SAM cache), `runners/` (sweep and benchmark), `analysis/` (aggregation, tests, instances, Grad-CAM, DPNP, latency), `reference21/` (re-implementation with fidelity tests), `medsam_r3/`, `patches/`, `tables/` |
| `02_pipeline_scripts/` | pipeline driver `post_sweep_chain.py`, builders for tables, figures and workbooks, checkers |
| `03_results/` | per-run metrics and logs (`cryonuseg/`, `nuinsseg/`, `ablation/`, `reference21/`), `analysis/`, `epoch_workbooks/`, `latency/`, `medsam_r3/`, `paper_tables/`, `logs/`, `splits/` |
| `04_figure_sources/` | Excalidraw sources of Figures 1-4 |
| `05_checkpoints_manifest/` | `CHECKPOINT_MANIFEST.csv`: path, size, SHA-256 of all 215 checkpoints (the files are on Zenodo, not in git) |
| `06_data_info/` | split manifests (which image is in train, validation, test) |
| `07_previous_rounds_code/` | code of earlier review rounds (no results; superseded) |

## Data (not included)
* CryoNuSeg: https://www.kaggle.com/datasets/ipateam/segmentation-of-nuclei-in-cryosectioned-he-images
* NuInsSeg: https://www.kaggle.com/datasets/ipateam/nuinsseg
* SAM ViT-B weights `sam_vit_b_01ec64.pth`: https://github.com/facebookresearch/segment-anything
* MedSAM weights (`wanglab/medsam-vit-base`) for the MedSAM evaluation.

## Checkpoints
Four archives on the checkpoint Zenodo record (`EPCS_checkpoints_cryonuseg.zip`, `..._nuinsseg.zip`, `..._ablation.zip`, `..._reference21.zip`). Verify each file with the SHA-256 in `05_checkpoints_manifest/CHECKPOINT_MANIFEST.csv`. Paths inside the zips follow the `03_results/` layout.

## Reproducing
`pip install -r requirements.txt` (Python 3.11, PyTorch 2). Training one model needs a GPU with about 3 GB of memory (measured peak 2.9 GB for EPCS-Net). The scripts were written in a working tree with the folder names `03_code`, `04_results`, `06_scripts`; here they are `01_source_code`, `03_results`, `02_pipeline_scripts`. Scripts find the project root from their own location, so to re-run them recreate the original names or edit the root constant at the top of each script. All tables and figures in the paper can be regenerated from `03_results/` without retraining.

## Limitations stated in the paper
Pairs are not independent across seeds within a fold; the ablation is one CryoNuSeg fold (10 test images); phenotyping uses one seed-2024 model per dataset and connected components merge touching nuclei (18.0% of components on NuInsSeg); no whole-slide or clinical claim is made.

## Citation and licence
See `CITATION.cff`. Code: MIT (`LICENSE`). The datasets and the SAM and MedSAM weights keep their own licences.
