"""
DPNP rebuilt to answer Reviewer 3 major comment 5 and Reviewer 4 comment 3.

The objections
--------------
R3.5  "Under the primary HDBSCAN rule, min_cluster_size = max(5, N/10), the
       revised manuscript reports that every connected component is labelled as
       noise and no clusters are found. Figure 12 instead displays a sensitivity
       analysis using min_cluster_size = 5. These are materially different
       outcomes. Please identify which configuration constitutes the
       prespecified analysis, present the alternative only as a sensitivity
       check, and avoid treating a silhouette score as biological validation."

R4.3  "Support the biological labels assigned to PCA components through loading
       analyses. The main HDBSCAN configuration produces no clusters; clusters
       recovered in a sensitivity analysis require stability assessment and
       checks for organ, species, stain, or image effects."

What this module does about them
--------------------------------
1. PRESPECIFIED FIRST. The prespecified rule is computed and reported as the
   primary result whatever it returns. The fixed min_cluster_size = 5 run is
   computed too, and is labelled a sensitivity check everywhere it appears.

2. LOADING ANALYSIS. The feature vector is 7 interpretable morphological terms
   plus 512 ResNet18 embedding dimensions. The manuscript labels PC1 "shape and
   size", PC2 "staining intensity" and PC3 "texture and chromatin". This
   computes (a) how much of each principal component's loading mass sits on the
   interpretable terms versus the embedding, and (b) the Spearman correlation
   between each component and each interpretable feature. If the embedding
   dominates, the labels are not supportable and the paper must drop them.

3. STABILITY. The sensitivity clustering is repeated on bootstrap subsamples
   and scored with the adjusted Rand index against the full-sample labels,
   restricted to the shared points. A cluster structure that does not survive
   resampling is not a finding.

4. CONFOUNDS. Cluster assignment is tested for association with organ, with
   species (NuInsSeg contains both human and mouse tissue), and with source
   image, using chi-square and Cramer's V. A "phenotype" cluster that is really
   an organ or a species label is a confound, not biology.

5. NO BIOLOGICAL CLAIM. A silhouette score measures geometric separation in a
   feature space, not biological validity. Outputs are named accordingly, and
   the module prints that statement so it cannot be quietly dropped.

Usage:
    python dpnp_r3.py --predictions <dir> --split <manifest> --tag <name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet18_Weights, resnet18

R3 = Path(__file__).resolve().parents[2]
RESULTS = R3 / "04_results"
OUT = RESULTS / "analysis"

IMAGE_SIZE = 256
MIN_COMPONENT_AREA = 5
MORPH = ["area", "roundness", "mean_intensity", "var_intensity", "L", "a", "b"]
HDBSCAN_MIN_SAMPLES = 5
HDBSCAN_METRIC = "euclidean"
PCA_MAX_DIM = 40
N_BOOTSTRAP = 25
BOOTSTRAP_FRACTION = 0.8
RNG = np.random.default_rng(20261005)


def build_embedder(device):
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    pre = ResNet18_Weights.IMAGENET1K_V1.transforms()
    return net, pre


@torch.no_grad()
def component_features(image_rgb, binary_mask, net, pre, device) -> pd.DataFrame:
    n, labels, stats_cc, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    rows = []
    for lid in range(1, n):
        area = int(stats_cc[lid, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_AREA:
            continue
        m = (labels == lid).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perim = sum(cv2.arcLength(c, True) for c in contours) or 1.0
        vals = gray[m.astype(bool)]
        lvals = lab[m.astype(bool)]
        x, y, w, h = (stats_cc[lid, cv2.CC_STAT_LEFT], stats_cc[lid, cv2.CC_STAT_TOP],
                      stats_cc[lid, cv2.CC_STAT_WIDTH], stats_cc[lid, cv2.CC_STAT_HEIGHT])
        roi = image_rgb[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        # Match the published DPNP pipeline exactly: convert to float and
        # scale to [0,1] BEFORE the torchvision transform. Passing uint8
        # instead makes the internal resize run at 8-bit precision, which
        # perturbs the embedding and changes the clustering downstream.
        roi_t = torch.from_numpy(roi.transpose(2, 0, 1).copy()).float().div(255.0)
        t = pre(roi_t).unsqueeze(0).to(device)
        emb = net(t)[0].cpu().numpy()
        row = {"component_id": lid, "area": area,
               "roundness": float(4 * np.pi * area / (perim ** 2)),
               "mean_intensity": float(vals.mean()), "var_intensity": float(vals.var()),
               "L": float(lvals[:, 0].mean()), "a": float(lvals[:, 1].mean()),
               "b": float(lvals[:, 2].mean())}
        for i, v in enumerate(emb):
            row[f"resnet18_{i}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def loading_analysis(pca: PCA, cols: list[str], scores: np.ndarray,
                     features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Where does each component's loading mass sit, and what does it track?"""
    morph_idx = [i for i, c in enumerate(cols) if c in MORPH]
    emb_idx = [i for i, c in enumerate(cols) if c.startswith("resnet18_")]

    mass_rows = []
    for k in range(min(3, pca.n_components_)):
        load = pca.components_[k]
        sq = load ** 2
        total = sq.sum()
        mass_rows.append({
            "component": f"PC{k+1}",
            "explained_variance_pct": float(pca.explained_variance_ratio_[k] * 100),
            "n_interpretable_features": len(morph_idx),
            "n_embedding_features": len(emb_idx),
            "loading_mass_interpretable_pct": float(sq[morph_idx].sum() / total * 100),
            "loading_mass_embedding_pct": float(sq[emb_idx].sum() / total * 100),
            "top_interpretable_feature": cols[morph_idx[int(np.argmax(sq[morph_idx]))]] if morph_idx else "",
        })

    corr_rows = []
    for k in range(min(3, scores.shape[1])):
        for f in MORPH:
            if f not in features.columns:
                continue
            rho, p = stats.spearmanr(scores[:, k], features[f].to_numpy(float))
            corr_rows.append({"component": f"PC{k+1}", "feature": f,
                              "spearman_rho": float(rho), "p_value": float(p),
                              "abs_rho": abs(float(rho))})
    return pd.DataFrame(mass_rows), pd.DataFrame(corr_rows)


def cluster(scores: np.ndarray, min_cluster_size: int) -> dict:
    cl = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=HDBSCAN_MIN_SAMPLES,
                 metric=HDBSCAN_METRIC)
    labels = cl.fit_predict(scores)
    uniq = [c for c in np.unique(labels) if c != -1]
    sil = None
    if len(uniq) >= 2:
        keep = labels != -1
        if keep.sum() > len(uniq):
            sil = float(silhouette_score(scores[keep], labels[keep]))
    return {"min_cluster_size": int(min_cluster_size),
            "min_samples": HDBSCAN_MIN_SAMPLES, "metric": HDBSCAN_METRIC,
            "n_points": int(len(labels)), "n_clusters": len(uniq),
            "outlier_pct": float((labels == -1).mean() * 100),
            "silhouette_geometric_separation": sil,
            "cluster_sizes": {int(c): int((labels == c).sum()) for c in uniq},
            "labels": labels}


def stability(scores: np.ndarray, min_cluster_size: int, reference: np.ndarray) -> dict:
    """Adjusted Rand index between the full-sample labelling and bootstrap
    subsample labellings, on the points they share."""
    n = len(scores)
    size = int(BOOTSTRAP_FRACTION * n)
    aris, counts = [], []
    for _ in range(N_BOOTSTRAP):
        idx = RNG.choice(n, size=size, replace=False)
        lab = HDBSCAN(min_cluster_size=min_cluster_size,
                      min_samples=HDBSCAN_MIN_SAMPLES,
                      metric=HDBSCAN_METRIC).fit_predict(scores[idx])
        aris.append(adjusted_rand_score(reference[idx], lab))
        counts.append(len([c for c in np.unique(lab) if c != -1]))
    return {"n_bootstrap": N_BOOTSTRAP, "subsample_fraction": BOOTSTRAP_FRACTION,
            "ari_mean": float(np.mean(aris)), "ari_sd": float(np.std(aris, ddof=1)),
            "ari_min": float(np.min(aris)), "ari_max": float(np.max(aris)),
            "n_clusters_mean": float(np.mean(counts)),
            "n_clusters_range": [int(np.min(counts)), int(np.max(counts))]}


def cramers_v(table: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n and k else float("nan")


def confounds(labels: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keep = labels != -1
    if keep.sum() < 2:
        return pd.DataFrame()
    lab = labels[keep]
    for field in ("organ", "species", "filename"):
        if field not in meta.columns:
            continue
        vals = meta.loc[keep, field].astype(str).to_numpy()
        if len(np.unique(vals)) < 2:
            continue
        table = pd.crosstab(vals, lab).to_numpy()
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
        rows.append({"confound": field, "n_levels": int(len(np.unique(vals))),
                     "chi2": float(chi2), "dof": int(dof), "p_value": float(p),
                     "cramers_v": cramers_v(table),
                     "interpretation": "cluster assignment is associated with this variable"
                                       if p < 0.05 else "no detected association"})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--only-split", default=None,
                    help="restrict the manifest to this split value, e.g. test")
    ap.add_argument("--device", default="auto", help="auto (default), cpu, cuda:0")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N images (smoke tests only)")
    ap.add_argument("--out-dir", default=None,
                    help="write outputs here instead of 04_results/analysis")
    a = ap.parse_args()
    global OUT
    if a.out_dir:
        OUT = Path(a.out_dir)

    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu")
                          if a.device == "auto" else a.device)
    pred_dir = Path(a.predictions)
    frame = pd.read_csv(a.split)
    if a.only_split and "split" in frame.columns:
        frame = frame[frame.split == a.only_split]
    if a.limit:
        frame = frame.head(a.limit)

    net, pre = build_embedder(device)
    blocks, meta = [], []
    for _, rec in frame.iterrows():
        stem = Path(rec.image_path).stem
        png = pred_dir / f"{stem}.png"
        if not png.is_file():
            alt = list(pred_dir.glob(f"*{stem}*.png"))
            if not alt:
                continue
            png = alt[0]
        pred = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(rec.image_path, cv2.IMREAD_COLOR)
        if pred is None or img is None:
            continue
        img = cv2.cvtColor(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2RGB)
        if pred.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            pred = cv2.resize(pred, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        f = component_features(img, (pred > 127).astype(np.uint8), net, pre, device)
        if f.empty:
            continue
        organ = str(rec.get("organ", ""))
        f["filename"] = rec.filename
        f["organ"] = organ
        f["species"] = ("mouse" if organ.lower().startswith("mouse")
                        else "human" if organ.lower().startswith("human") else "unknown")
        blocks.append(f)
    if not blocks:
        print("No components extracted.")
        return 1

    feats = pd.concat(blocks, ignore_index=True)
    meta = feats[["filename", "organ", "species"]].reset_index(drop=True)
    cols = MORPH + [c for c in feats.columns if c.startswith("resnet18_")]
    X = StandardScaler().fit_transform(feats[cols].to_numpy(float))
    n_comp = min(PCA_MAX_DIM, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    scores = pca.fit_transform(X)

    mass, corr = loading_analysis(pca, cols, scores, feats)

    n = len(feats)
    prespecified_mcs = max(5, n // 10)
    primary = cluster(scores, prespecified_mcs)
    sensitivity = cluster(scores, 5)

    stab = (stability(scores, 5, sensitivity["labels"])
            if sensitivity["n_clusters"] >= 2 else None)
    conf = confounds(sensitivity["labels"], meta) if sensitivity["n_clusters"] >= 2 else pd.DataFrame()

    OUT.mkdir(parents=True, exist_ok=True)
    feats.assign(**{f"PC{i+1}": scores[:, i] for i in range(min(3, scores.shape[1],))}) \
         .to_csv(OUT / f"dpnp_components_{a.tag}.csv", index=False)
    mass.to_csv(OUT / f"dpnp_pca_loading_mass_{a.tag}.csv", index=False)
    corr.to_csv(OUT / f"dpnp_pca_feature_correlations_{a.tag}.csv", index=False)
    if not conf.empty:
        conf.to_csv(OUT / f"dpnp_cluster_confounds_{a.tag}.csv", index=False)

    report = {
        "tag": a.tag, "predictions_dir": str(pred_dir),
        "n_images": int(feats.filename.nunique()), "n_components": int(n),
        "feature_vector": {"interpretable": MORPH,
                           "embedding_dims": len([c for c in cols if c.startswith("resnet18_")]),
                           "total_dims": len(cols)},
        "pca": {"n_components": int(n_comp),
                "explained_variance_pct": [float(v * 100) for v in pca.explained_variance_ratio_[:3]],
                "cumulative_first_three_pct": float(pca.explained_variance_ratio_[:3].sum() * 100)},
        "clustering_primary_prespecified": {k: v for k, v in primary.items() if k != "labels"},
        "clustering_sensitivity_fixed5": {k: v for k, v in sensitivity.items() if k != "labels"},
        "sensitivity_stability": stab,
        "confounds": conf.to_dict("records") if not conf.empty else [],
        "statement": ("A silhouette score measures geometric separation in a feature "
                      "space. It is not biological validation. These outputs are "
                      "exploratory connected-region descriptors, not per-nucleus "
                      "phenotypes; see instances_r3.py for the measured merge rate."),
    }
    (OUT / f"dpnp_report_{a.tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 76)
    print(f"DPNP  {a.tag}   {n:,} components from {feats.filename.nunique()} images")
    print("=" * 76)
    print(f"feature vector: {len(MORPH)} interpretable + "
          f"{len(cols)-len(MORPH)} embedding = {len(cols)} dims")
    print(f"\nPCA explained variance: PC1 {pca.explained_variance_ratio_[0]*100:.2f}%  "
          f"PC2 {pca.explained_variance_ratio_[1]*100:.2f}%  "
          f"PC3 {pca.explained_variance_ratio_[2]*100:.2f}%")
    print("\nLOADING MASS -- can the component labels be supported?")
    print(f"  {'PC':<5}{'var %':>8}{'interpretable %':>18}{'embedding %':>14}{'top feature':>18}")
    for _, r in mass.iterrows():
        print(f"  {r['component']:<5}{r['explained_variance_pct']:>8.2f}"
              f"{r['loading_mass_interpretable_pct']:>18.2f}"
              f"{r['loading_mass_embedding_pct']:>14.2f}{r['top_interpretable_feature']:>18}")
    print("\nSTRONGEST INTERPRETABLE CORRELATIONS")
    for pcname, g in corr.groupby("component"):
        top = g.nlargest(2, "abs_rho")
        s = "   ".join(f"{r['feature']} rho={r['spearman_rho']:+.3f}" for _, r in top.iterrows())
        print(f"  {pcname}: {s}")
    print(f"\nCLUSTERING -- PRIMARY (prespecified, min_cluster_size = max(5, N/10) = {prespecified_mcs})")
    print(f"  clusters {primary['n_clusters']}   outliers {primary['outlier_pct']:.1f}%   "
          f"silhouette {primary['silhouette_geometric_separation']}")
    print(f"CLUSTERING -- SENSITIVITY CHECK (min_cluster_size = 5)")
    print(f"  clusters {sensitivity['n_clusters']}   outliers {sensitivity['outlier_pct']:.1f}%   "
          f"silhouette {sensitivity['silhouette_geometric_separation']}")
    if stab:
        print(f"  stability: ARI {stab['ari_mean']:.3f} +/- {stab['ari_sd']:.3f} "
              f"over {stab['n_bootstrap']} subsamples, clusters {stab['n_clusters_range']}")
    if not conf.empty:
        print("  confound checks:")
        for _, r in conf.iterrows():
            print(f"    {r['confound']:<10} Cramer's V {r['cramers_v']:.3f}  "
                  f"p {r['p_value']:.3g}  -> {r['interpretation']}")
    print(f"\n{report['statement']}")
    print(f"\nOutputs in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
