"""
Revision-3 split construction.

Two designs, both fully deterministic, each fixing a specific reviewer objection.

NuInsSeg -- organ-stratified 70/15/15 train/validation/test.
    Fixes Reviewer 3 major 1 and Reviewer 4 comment 1: the reported set must not
    be the set used to choose the checkpoint. Also fixes the audit finding that
    the old random 80/20 split left two of the 31 organs with zero validation
    patches. Every organ appears in all three partitions wherever its count
    allows.

CryoNuSeg -- tissue-stratified three-fold cross-validation.
    Fixes Reviewer 2 comment 2: the split is not tissue-stratified and does not
    contain all ten tissue types. The dataset is exactly 10 tissues x 3
    replicates. Fold k holds out replicate k of EVERY tissue as test, so the
    reported set always covers all ten tissues. Across the three folds each
    image serves as test exactly once.

      fold k: test  = replicate k                      (10 images, all tissues)
              val   = replicate (k+1) % 3, 5 tissues   ( 5 images)
              train = everything else                  (15 images)

    Validation is only a checkpoint-selection set, so it does not need full
    tissue coverage. The test set does, and it has it.

Both manifests carry an ``organ`` column. CryoNuSeg never had one before, which
is why OSPT was silently never applied to it. From revision 3 OSPT applies to
both datasets.

Image/mask pairing reproduces the verified logic of the Run A protocol exactly,
including the ``name (1).png`` mask variant, but is kept self-contained here so
the revision-3 tree does not depend on the old one.

Usage:
    python splits_r3.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]              # ...\Desktop\EPCS
OUT = Path(__file__).resolve().parents[2] / "04_results" / "splits"

# Fixed, and deliberately independent of every training seed, so that all models
# and all seeds see byte-identical partitions. Changing it invalidates
# comparability with any run already completed.
SPLIT_SEED = 20261005

IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}

CRYO_IMAGES = ROOT / "01_datasets" / "CryoNuSeg" / "tissue images"
CRYO_MASKS = ROOT / "01_datasets" / "CryoNuSeg" / "Annotator 1 (biologist)" / "mask binary"
NUIN_ROOT = ROOT / "01_datasets" / "NuInsSeg"
NUIN_IMAGE_SUBDIR = "tissue images"
NUIN_MASK_SUBDIR = "mask binary without border erode"

VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


# --------------------------------------------------------------------------
# pairing (identical behaviour to the Run A protocol)
# --------------------------------------------------------------------------
def stem_key(path: Path) -> str:
    stem = re.sub(r"\s*\(1\)$", "", path.stem, flags=re.IGNORECASE)
    return stem.lower().replace(" ", "_").replace("-", "_")


def index_folder(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        return {}
    out: dict[str, Path] = {}
    for path in sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
        key = stem_key(path)
        if key in out:
            raise ValueError(f"Duplicate stem '{key}' in {folder}: {out[key]} and {path}")
        out[key] = path
    return out


def pair(images: dict[str, Path], masks: dict[str, Path], where: str) -> list[dict]:
    missing = sorted(set(images) - set(masks))
    extra = sorted(set(masks) - set(images))
    if missing or extra:
        raise ValueError(f"{where}: unmatched stems. missing masks {missing[:5]}, "
                         f"extra masks {extra[:5]}")
    return [{"filename": images[k].name,
             "image_path": str(images[k].resolve()),
             "mask_path": str(masks[k].resolve())} for k in sorted(images)]


def stable_seed(text: str) -> int:
    """Deterministic across processes, unlike hash() on str."""
    digest = hashlib.sha256(f"{SPLIT_SEED}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


# --------------------------------------------------------------------------
# CryoNuSeg
# --------------------------------------------------------------------------
def cryo_records() -> pd.DataFrame:
    images = index_folder(CRYO_IMAGES)
    masks = index_folder(CRYO_MASKS)
    if not images:
        raise FileNotFoundError(f"no CryoNuSeg images under {CRYO_IMAGES}")
    rows = pair(images, masks, "cryonuseg")
    for r in rows:
        m = re.fullmatch(r"Human_([A-Za-z]+)_(\d+)", Path(r["filename"]).stem)
        if not m:
            raise RuntimeError(f"unexpected CryoNuSeg filename: {r['filename']}")
        r["organ"] = m.group(1)
        r["replicate"] = int(m.group(2))
    return pd.DataFrame(rows)


def build_cryonuseg_folds() -> dict:
    df = cryo_records()
    tissues = sorted(df["organ"].unique())
    replicates = sorted(df["replicate"].unique())
    print(f"  {len(df)} patches, {len(tissues)} tissues, {len(replicates)} replicates")

    report = {"dataset": "CryoNuSeg", "design": "tissue-stratified 3-fold CV",
              "n_patches": int(len(df)), "tissues": tissues,
              "replicates": [int(r) for r in replicates], "folds": []}

    for k, test_rep in enumerate(replicates):
        val_rep = replicates[(k + 1) % len(replicates)]
        val_tissues = [t for i, t in enumerate(tissues) if (i + k) % 2 == 0][:5]

        def assign(row):
            if row["replicate"] == test_rep:
                return "test"
            if row["replicate"] == val_rep and row["organ"] in val_tissues:
                return "validation"
            return "train"

        fold = df.copy()
        fold["split"] = fold.apply(assign, axis=1)
        cols = ["filename", "image_path", "mask_path", "organ", "replicate", "split"]
        path = OUT / f"cryonuseg_R3_fold{k + 1}.csv"
        fold[cols].to_csv(path, index=False)

        counts = {s: int((fold["split"] == s).sum()) for s in ("train", "validation", "test")}
        test_tissues = sorted(fold[fold.split == "test"]["organ"].unique())
        covers = len(test_tissues) == len(tissues)
        report["folds"].append({
            "fold": k + 1, "manifest": path.name,
            "test_replicate": int(test_rep), "val_replicate": int(val_rep),
            "val_tissues": val_tissues, "counts": counts,
            "test_tissue_count": len(test_tissues),
            "test_covers_all_tissues": covers,
        })
        print(f"  fold {k + 1}: train {counts['train']:>2}  val {counts['validation']:>2}  "
              f"test {counts['test']:>2}   test covers {len(test_tissues)}/{len(tissues)} tissues "
              f"{'OK' if covers else 'FAIL'}")
    return report


# --------------------------------------------------------------------------
# NuInsSeg
# --------------------------------------------------------------------------
def nuin_records() -> pd.DataFrame:
    organ_dirs = sorted(p for p in NUIN_ROOT.iterdir() if p.is_dir())
    rows = []
    for organ_dir in organ_dirs:
        images = index_folder(organ_dir / NUIN_IMAGE_SUBDIR)
        if not images:
            continue
        masks = index_folder(organ_dir / NUIN_MASK_SUBDIR)
        for r in pair(images, masks, organ_dir.name):
            r["organ"] = organ_dir.name
            # Organ-prefixed key so per-image metric files cannot collide
            # between organs that reuse a filename.
            r["filename"] = f"{organ_dir.name}__{r['filename']}"
            rows.append(r)
    if not rows:
        raise FileNotFoundError(f"no paired NuInsSeg files under {NUIN_ROOT}")
    return pd.DataFrame(rows)


def build_nuinsseg_split() -> dict:
    df = nuin_records()
    blocks = []
    for organ in sorted(df["organ"].unique()):
        sub = df[df["organ"] == organ].sort_values("filename")
        shuffled = sub.sample(frac=1.0, random_state=stable_seed(organ))
        n = len(shuffled)
        if n >= 3:
            n_test = max(1, round(TEST_FRACTION * n))
            n_val = max(1, round(VAL_FRACTION * n))
        elif n == 2:
            n_test, n_val = 1, 0
        else:
            n_test, n_val = 0, 0
        if n_test + n_val >= n:                      # always keep >= 1 train patch
            n_val = max(0, n - n_test - 1)
        labels = ["test"] * n_test + ["validation"] * n_val + ["train"] * (n - n_test - n_val)
        block = shuffled.copy()
        block["split"] = labels
        blocks.append(block)

    out = (pd.concat(blocks).sort_values(["organ", "filename"]).reset_index(drop=True))
    cols = ["filename", "image_path", "mask_path", "organ", "split"]
    path = OUT / "nuinsseg_R3_stratified_70_15_15.csv"
    out[cols].to_csv(path, index=False)

    per_organ = (out.groupby(["organ", "split"]).size().unstack(fill_value=0)
                 .reindex(columns=["train", "validation", "test"], fill_value=0))
    per_organ["total"] = per_organ.sum(axis=1)
    per_organ.to_csv(OUT / "nuinsseg_R3_per_organ_counts.csv")

    zero_test = per_organ.index[per_organ["test"] == 0].tolist()
    zero_val = per_organ.index[per_organ["validation"] == 0].tolist()
    counts = {s: int((out["split"] == s).sum()) for s in ("train", "validation", "test")}
    total = len(out)
    print(f"  {total} patches, {out['organ'].nunique()} organs")
    print(f"  train {counts['train']}  validation {counts['validation']}  test {counts['test']}"
          f"   ({counts['train']/total:.1%} / {counts['validation']/total:.1%} / {counts['test']/total:.1%})")
    print(f"  organs with zero test patches: {len(zero_test)}  "
          f"zero validation patches: {len(zero_val)}")

    return {"dataset": "NuInsSeg", "design": "organ-stratified 70/15/15",
            "manifest": path.name, "per_organ_counts": "nuinsseg_R3_per_organ_counts.csv",
            "n_patches": total, "n_organs": int(out["organ"].nunique()),
            "counts": counts, "organs_without_test": zero_test,
            "organs_without_validation": zero_val}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building revision-3 splits")
    print(f"  split seed (fixed, independent of training seeds): {SPLIT_SEED}\n")
    print("CryoNuSeg -- tissue-stratified 3-fold CV")
    cryo = build_cryonuseg_folds()
    print("\nNuInsSeg -- organ-stratified 70/15/15")
    nuin = build_nuinsseg_split()

    report = {"split_seed": SPLIT_SEED, "cryonuseg": cryo, "nuinsseg": nuin}
    (OUT / "SPLIT_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote manifests + SPLIT_REPORT.json to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
