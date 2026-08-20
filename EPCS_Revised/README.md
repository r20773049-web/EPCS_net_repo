# EPCS-Net — Revised Code and Results

Code and results only. This folder is meant to be pushed into the existing
`EPCS_net_repo` GitHub repository as a clearly separated addition to the
older codebase already there.

## Layout

| Folder | Contents |
|---|---|
| `02_code_RunA/` | Six models (U-Net, U-Net++, U-Net3+, eU-Net3+, SAM-guided baseline, EPCS-Net), one script per model per dataset, over a shared training/evaluation module (`shared/epcs_protocol_RunA.py`). |
| `02_code_RunB/` | EPCS-Net retrained from 5 fixed seeds per dataset, reusing Run A's split, for the reproducibility statistics reported in the paper. |
| `02_code_MedSAM/` | Zero-shot MedSAM baseline evaluation (official checkpoint, whole-image box prompt). |
| `02_code_DPNP/` | Grad-CAM, PCA nuclear phenotyping, and HDBSCAN clustering (Deep Phenotypic Nuclear Profiling). |
| `03_results_RunA/`, `03_results_RunB/`, `03_results_MedSAM/`, `03_results_DPNP/` | Per-model metrics (CSV/JSON), per-epoch training histories, saved prediction masks, and DPNP outputs. |
| `EPCS_Master_Results_ForPaper.csv` | Every reportable number from the paper, in one file. |

The 13 EPCS-Net checkpoints (flagship + 5-seed reproducibility runs + the
OSPT ablation model, ~1.4 GB) **are** present in this copy of the folder,
sitting in their natural `03_results_RunA/`/`03_results_RunB/` locations —
but see "Trained model checkpoints" below before running `git add`.
The manuscript, response letter, and project log are not here either — see
`../EPCS_Documentation/`, one level up.

## Setting up datasets and pretrained weights (needed to re-run anything)

Neither is included here (large, and both already exist elsewhere):

- **CryoNuSeg** and **NuInsSeg** (~2.2 GB combined, public):
  - <https://www.kaggle.com/datasets/ipateam/segmentation-of-nuclei-in-cryosectioned-he-images>
  - <https://www.kaggle.com/datasets/ipateam/nuinsseg>

  Place them at `01_datasets/CryoNuSeg/` and `01_datasets/NuInsSeg/`,
  as siblings of `02_code_RunA/` — the scripts resolve paths relative to
  that layout and need no editing.

- **SAM ViT-B checkpoint** (`sam_vit_b_01ec64.pth`, ~375 MB, third-party):
  download from the official Segment Anything repository and place at
  `05_reference/sam_vit_b_01ec64.pth`.

## Trained model checkpoints

Every EPCS-Net checkpoint is ~102 MB — over GitHub's 100 MB hard limit for
a plain `git push`, which fails outright on any file past that size, not
just a warning. That's true even for a single checkpoint, let alone all 13.

This folder's `.gitignore` excludes `*.pt` for exactly that reason, so a
normal `git add .` / `git push` will commit the code and skip the
checkpoints safely rather than fail partway through. The checkpoint files
themselves stay physically present here for your own local use.

To actually get them onto GitHub attached to this repository, use
**Releases**, not a normal push: after pushing the code (below), go to the
repo's Releases page → Draft a new release → attach the `.pt` files as
release assets (2 GB per-file limit, comfortably enough). Zenodo archives
release assets the same way it archives the rest of the repo, so this still
ends up under the same DOI.

The same 13 files also sit in `../EPCS_revised_checkpoints_for_zenodo/` for
convenience if you'd rather upload from there.

## Pushing to GitHub

```bash
cd EPCS_revised
git init
git add .
git commit -m "Add revised code, results, and predictions for Scientific Reports revision"
git branch -M revision-2
git remote add origin https://github.com/r20773049-web/EPCS_net_repo.git
git push -u origin revision-2
```

`git branch -M revision-2` pushes to a *new* branch rather than `main`,
since the repo already has the old code on `main` — this way nothing gets
overwritten, and the two can be merged later via a pull request if that's
the plan. If you specifically want this to replace `main` directly instead,
use `git push -u origin main` in the last line — but confirm with your
senior first, since that's a one-way door on a shared repo.

If `git remote add origin` errors saying `origin` already exists, use
`git remote set-url origin https://github.com/r20773049-web/EPCS_net_repo.git`
instead.
