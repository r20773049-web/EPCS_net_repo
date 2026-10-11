"""
Do predicted connected components correspond to individual nuclei?

Reviewer 4 comment 3 and Reviewer 3 major comment 5 both say the same thing:
the DPNP features are extracted from connected components of a BINARY semantic
mask, touching nuclei merge into one component, and therefore the descriptors
cannot be presented as per-nucleus measurements.

The manuscript currently concedes this in words. This module measures it, using
the ground-truth INSTANCE masks that NuInsSeg ships in its `label masks`
folders, so the concession becomes a number instead of a caveat.

What is computed
----------------
Counting
    n_gt_instances, n_pred_components, and the ratio. A ratio below 1 means the
    pipeline systematically under-counts nuclei by merging them.

Merging and splitting
    merge  : a predicted component that overlaps 2 or more ground-truth
             instances. This is the failure the reviewers name.
    split  : a ground-truth instance covered by 2 or more predicted components.
    clean  : one-to-one.
    Overlap is counted only when it exceeds MIN_OVERLAP_PX pixels, so a few
    pixels of boundary bleed do not register as a merge.

Feature bias
    The same descriptors DPNP uses (area, roundness, mean and variance of
    intensity) computed on predicted components and on ground-truth instances,
    compared with a Mann-Whitney U test and a rank-biserial effect size. If
    merging inflates area and depresses roundness, that is the quantitative
    statement the paper needs.

CryoNuSeg is excluded on purpose: it ships binary masks only, so there is no
instance ground truth to compare against. That exclusion is reported rather
than worked around.

Usage:
    python instances_r3.py                       # newest revision-3 runs
    python instances_r3.py --predictions <dir>   # any folder of predicted PNGs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats

R3 = Path(__file__).resolve().parents[2]
ROOT = R3.parent
RESULTS = R3 / "04_results"
OUT = RESULTS / "analysis"
NUIN = ROOT / "01_datasets" / "NuInsSeg"

IMAGE_SIZE = 256
MIN_OVERLAP_PX = 10        # ignore a few pixels of boundary bleed
MIN_COMPONENT_AREA = 5     # same floor DPNP uses


# --------------------------------------------------------------------------
def instance_mask_for(image_path: str) -> np.ndarray | None:
    """Locate the ground-truth instance mask beside a NuInsSeg tissue image."""
    p = Path(image_path)
    organ_dir = p.parent.parent
    for folder in ("label masks", "label masks modify"):
        for ext in (".tif", ".tiff", ".png"):
            cand = organ_dir / folder / f"{p.stem}{ext}"
            if cand.is_file():
                m = cv2.imread(str(cand), cv2.IMREAD_UNCHANGED)
                if m is None:
                    continue
                if m.ndim == 3:
                    m = m[..., 0]
                return cv2.resize(m, (IMAGE_SIZE, IMAGE_SIZE),
                                  interpolation=cv2.INTER_NEAREST)
    return None


def region_features(label_map: np.ndarray, gray: np.ndarray) -> pd.DataFrame:
    """Area, roundness and intensity statistics, matching the DPNP definitions."""
    rows = []
    for lid in np.unique(label_map):
        if lid == 0:
            continue
        m = (label_map == lid).astype(np.uint8)
        area = int(m.sum())
        if area < MIN_COMPONENT_AREA:
            continue
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perim = sum(cv2.arcLength(c, True) for c in contours) or 1.0
        vals = gray[m.astype(bool)]
        rows.append({"label": int(lid), "area": area,
                     "roundness": float(4 * np.pi * area / (perim ** 2)),
                     "mean_intensity": float(vals.mean()),
                     "var_intensity": float(vals.var())})
    return pd.DataFrame(rows)


def analyse_image(image_path: str, pred_png: Path) -> dict | None:
    gt = instance_mask_for(image_path)
    if gt is None:
        return None
    pred = cv2.imread(str(pred_png), cv2.IMREAD_GRAYSCALE)
    if pred is None:
        return None
    if pred.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        pred = cv2.resize(pred, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    binary = (pred > 127).astype(np.uint8)

    n_comp, comp = cv2.connectedComponents(binary, connectivity=8)
    gt_ids = [i for i in np.unique(gt) if i != 0]
    comp_ids = [i for i in np.unique(comp) if i != 0]

    merges, clean, comp_gt_counts = 0, 0, []
    for cid in comp_ids:
        m = comp == cid
        if m.sum() < MIN_COMPONENT_AREA:
            continue
        overlapped = gt[m]
        ids, counts = np.unique(overlapped[overlapped != 0], return_counts=True)
        hit = int((counts >= MIN_OVERLAP_PX).sum())
        comp_gt_counts.append(hit)
        if hit >= 2:
            merges += 1
        elif hit == 1:
            clean += 1

    splits = 0
    for gid in gt_ids:
        m = gt == gid
        ids, counts = np.unique(comp[m][comp[m] != 0], return_counts=True)
        if int((counts >= MIN_OVERLAP_PX).sum()) >= 2:
            splits += 1

    n_valid = len(comp_gt_counts)
    return {
        "n_gt_instances": len(gt_ids),
        "n_pred_components": n_valid,
        "count_ratio": (n_valid / len(gt_ids)) if gt_ids else np.nan,
        "merged_components": merges,
        "merge_rate": merges / n_valid if n_valid else np.nan,
        "clean_components": clean,
        "clean_rate": clean / n_valid if n_valid else np.nan,
        "split_gt_instances": splits,
        "split_rate": splits / len(gt_ids) if gt_ids else np.nan,
        "mean_gt_per_component": float(np.mean(comp_gt_counts)) if comp_gt_counts else np.nan,
        "max_gt_per_component": int(np.max(comp_gt_counts)) if comp_gt_counts else 0,
        "_comp_label_map": comp,
        "_gt_label_map": gt,
    }


def find_latest_epcs_run() -> tuple[Path, pd.DataFrame] | None:
    """Newest completed revision-3 EPCS-Net NuInsSeg run."""
    cands = sorted((RESULTS / "nuinsseg" / "nuinsseg" / "epcs_net").glob("seed*"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for d in cands:
        if (d / "summary.json").is_file() and (d / "test_predictions").is_dir():
            split = pd.read_csv(RESULTS / "splits" / "nuinsseg_R3_stratified_70_15_15.csv")
            return d / "test_predictions", split[split.split == "test"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=None,
                    help="folder of predicted PNG masks (defaults to the newest "
                         "revision-3 EPCS-Net NuInsSeg run)")
    ap.add_argument("--split", default=None, help="split manifest to pair against")
    ap.add_argument("--tag", default="r3", help="suffix for the output files")
    a = ap.parse_args()

    if a.predictions:
        pred_dir = Path(a.predictions)
        split_path = Path(a.split) if a.split else (
            RESULTS / "splits" / "nuinsseg_R3_stratified_70_15_15.csv")
        frame = pd.read_csv(split_path)
        if "split" in frame.columns:      # fix: ALWAYS restrict to the test partition (a --split manifest used to bypass this)
            frame = frame[frame.split == "test"]
    else:
        found = find_latest_epcs_run()
        if not found:
            print("No completed revision-3 EPCS-Net NuInsSeg run with predictions yet.")
            print("Either wait for the sweep, or pass --predictions explicitly.")
            return 1
        pred_dir, frame = found

    print(f"Predictions : {pred_dir}")
    print(f"Images      : {len(frame)}")

    rows, pred_feats, gt_feats, missing = [], [], [], 0
    for _, rec in frame.iterrows():
        stem = Path(rec.image_path).stem
        # fix: exact match on the manifest filename (e.g. 'human bladder__human_bladder_03.png'); the old substring
        # glob paired some training images with other images' predictions (3 of 106 analysed images were not test images)
        png = pred_dir / (rec.filename if "filename" in rec.index else f"{stem}.png")
        if not png.is_file():
            png = pred_dir / f"{stem}.png"
        if not png.is_file():
            missing += 1
            continue
        res = analyse_image(rec.image_path, png)
        if res is None:
            missing += 1
            continue

        comp_map = res.pop("_comp_label_map")
        gt_map = res.pop("_gt_label_map")
        img = cv2.imread(rec.image_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        pf = region_features(comp_map, gray)
        gf = region_features(gt_map, gray)
        pf["filename"], gf["filename"] = rec.filename, rec.filename
        pf["organ"] = gf["organ"] = rec.get("organ", "")
        pred_feats.append(pf)
        gt_feats.append(gf)

        res.update({"filename": rec.filename, "organ": rec.get("organ", "")})
        rows.append(res)

    if not rows:
        print("No images could be analysed.")
        return 1

    per_image = pd.DataFrame(rows)
    pred_df = pd.concat(pred_feats, ignore_index=True)
    gt_df = pd.concat(gt_feats, ignore_index=True)

    OUT.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(OUT / f"instance_vs_component_per_image_{a.tag}.csv", index=False)
    pred_df.to_csv(OUT / f"features_predicted_components_{a.tag}.csv", index=False)
    gt_df.to_csv(OUT / f"features_gt_instances_{a.tag}.csv", index=False)

    # ---- feature bias ----
    feat_rows = []
    for col in ("area", "roundness", "mean_intensity", "var_intensity"):
        a_vals = pred_df[col].to_numpy(float)
        b_vals = gt_df[col].to_numpy(float)
        u, p = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
        rb = 1 - (2 * u) / (len(a_vals) * len(b_vals))     # rank-biserial correlation
        feat_rows.append({
            "feature": col,
            "predicted_n": len(a_vals), "gt_n": len(b_vals),
            "predicted_mean": float(a_vals.mean()), "gt_mean": float(b_vals.mean()),
            "predicted_median": float(np.median(a_vals)), "gt_median": float(np.median(b_vals)),
            "ratio_of_means": float(a_vals.mean() / b_vals.mean()) if b_vals.mean() else np.nan,
            "mannwhitney_u": float(u), "p_value": float(p),
            "rank_biserial": float(rb),
        })
    feat = pd.DataFrame(feat_rows)
    feat.to_csv(OUT / f"feature_bias_components_vs_instances_{a.tag}.csv", index=False)

    summary = {
        "predictions_dir": str(pred_dir),
        "images_analysed": int(len(per_image)),
        "images_skipped_no_instance_mask": int(missing),
        "min_overlap_px": MIN_OVERLAP_PX,
        "min_component_area_px": MIN_COMPONENT_AREA,
        "total_gt_instances": int(per_image.n_gt_instances.sum()),
        "total_pred_components": int(per_image.n_pred_components.sum()),
        "overall_count_ratio": float(per_image.n_pred_components.sum()
                                     / max(per_image.n_gt_instances.sum(), 1)),
        "mean_merge_rate": float(per_image.merge_rate.mean()),
        "mean_clean_rate": float(per_image.clean_rate.mean()),
        "mean_split_rate": float(per_image.split_rate.mean()),
        "mean_gt_per_component": float(per_image.mean_gt_per_component.mean()),
        "max_gt_in_one_component": int(per_image.max_gt_per_component.max()),
        "cryonuseg": "excluded: ships binary masks only, no instance ground truth",
    }
    (OUT / f"instance_vs_component_summary_{a.tag}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print("DO PREDICTED COMPONENTS CORRESPOND TO INDIVIDUAL NUCLEI?")
    print("=" * 74)
    print(f"  ground-truth instances          {summary['total_gt_instances']:>8,}")
    print(f"  predicted components            {summary['total_pred_components']:>8,}")
    print(f"  count ratio                     {summary['overall_count_ratio']:>8.3f}")
    print(f"  components merging >=2 nuclei   {summary['mean_merge_rate']:>8.1%}")
    print(f"  components matching 1 nucleus   {summary['mean_clean_rate']:>8.1%}")
    print(f"  nuclei split across components  {summary['mean_split_rate']:>8.1%}")
    print(f"  mean nuclei per component       {summary['mean_gt_per_component']:>8.2f}")
    print(f"  worst single component held     {summary['max_gt_in_one_component']:>8d} nuclei")
    print("\nFEATURE BIAS (predicted components vs ground-truth instances)")
    print(f"  {'feature':<16}{'pred mean':>12}{'gt mean':>12}{'ratio':>9}{'p':>12}{'effect':>9}")
    for _, r in feat.iterrows():
        print(f"  {r['feature']:<16}{r['predicted_mean']:>12.3f}{r['gt_mean']:>12.3f}"
              f"{r['ratio_of_means']:>9.3f}{r['p_value']:>12.3g}{r['rank_biserial']:>9.3f}")
    print(f"\nOutputs in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
