# EPCS-Net: SAM-guided multi-scale nuclei segmentation

Code, per-run results and checkpoint manifest for the paper *EPCS-Net: SAM-Guided Multi-Scale Semantic Nuclei Segmentation and Exploratory Phenotypic Profiling for Pan-Cancer Analysis* (Scientific Reports).

* Archived release (Zenodo DOI): `10.5281/zenodo.XXXXXXX` (code and results)
* Trained checkpoints (Zenodo DOI): `10.5281/zenodo.YYYYYYY`
* Release tag: `v1.0.0`, commit: `<hash>`

## Start here
| File / folder | Content |
|---|---|
| `MASTER_RESULTS.xlsx` | master workbook: every run, averages per model, best values, ablation, MedSAM, significance tests, evaluation log |
| `MASTER_RESULTS_all_runs.csv` | one row per trained run (215 runs): dataset, model, fold, seed, best epoch, test and validation metrics, parameters, time |
| `MASTER_RESULTS_summary_by_model.csv` | mean, SD and 95% CI per model and dataset |
| `01_source_code/` | `shared/` (models, protocol, splits, SAM cache), `runners/` (sweep and benchmark), `analysis/` (aggregation, tests, instances, Grad-CAM, DPNP, latency), `reference21/` (re-implementation with fidelity tests), `medsam_r3/`, `patches/`, `tables/` |
| `02_pipeline_scripts/` | pipeline driver `post_sweep_chain.py`, builders for tables, figures and workbooks, checkers |
| `03_results/` | per-run metrics and logs (`cryonuseg/`, `nuinsseg/`, `ablation/`, `reference21/`), `analysis/` (CSV/JSON used in the paper), `epoch_workbooks/`, `latency/`, `medsam_r3/`, `paper_tables/`, `logs/` |
| `04_figure_sources/` | Excalidraw sources of Figures 1-4 |
| `05_checkpoints_manifest/` | `CHECKPOINT_MANIFEST.csv`: path, size and SHA-256 of all 215 checkpoints (files are on Zenodo, not in git) |
| `06_data_info/` | split manifests (which image is in train, validation and test) |
| `07_previous_rounds_code/` | code of earlier review rounds (no results; superseded) |

## Protocol
Frozen SAM ViT-B image encoder, local DoubleConv encoder, pyramid-scale prior injection (PSPI), adaptive cross-stream fusion blocks (ACFB), full-scale fused decoder, FTAS loss. Seeds 42, 123, 456, 789, 2024. NuInsSeg: organ-stratified 459/103/103 split, five seeds. CryoNuSeg: three tissue-stratified folds (15/5/10 images), five seeds per fold. The checkpoint is chosen on validation Dice only and the test split is scored once. Every number in the paper comes from `03_results/analysis/`.

## Data (not included)
* CryoNuSeg: https://www.kaggle.com/datasets/ipateam/segmentation-of-nuclei-in-cryosectioned-he-images
* NuInsSeg: https://www.kaggle.com/datasets/ipateam/nuinsseg
* SAM ViT-B weights `sam_vit_b_01ec64.pth`: https://github.com/facebookresearch/segment-anything
* MedSAM weights (`wanglab/medsam-vit-base`) for the MedSAM evaluation.

## Checkpoints
Four archives on Zenodo (`EPCS_checkpoints_cryonuseg.zip`, `..._nuinsseg.zip`, `..._ablation.zip`, `..._reference21.zip`). Verify with the SHA-256 values in `05_checkpoints_manifest/CHECKPOINT_MANIFEST.csv`. Paths inside the zips follow the `03_results/` layout.

## Reproducing
`pip install -r requirements.txt`. Scripts were written in a working tree with the layout `03_code`, `04_results`, `06_scripts`; in this release they are named `01_source_code`, `03_results`, `02_pipeline_scripts`. Scripts find the project root from their own location, so to re-run them recreate the original names or edit the root constant at the top of the script. Tables and figures can be regenerated from `03_results/` without retraining; training needs a GPU with about 3 GB of memory.

## Citation and licence
See `CITATION.cff`. Code: MIT (`LICENSE`). Datasets and the SAM and MedSAM weights keep their own licences.
