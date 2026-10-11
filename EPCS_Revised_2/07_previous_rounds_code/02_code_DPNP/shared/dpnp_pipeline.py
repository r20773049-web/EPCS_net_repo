#!/usr/bin/env python3
"""Deep Phenotypic Nuclear Profiling (DPNP) -- Grad-CAM, PCA phenotyping, HDBSCAN.

Added in response to reviewer feedback (Reviewer 2 Issues 4.2/4.3): the paper
describes this pipeline but the manuscript doesn't report the parameters or
per-sample statistics needed to evaluate it. This is a from-scratch
implementation of the three pieces described in the paper (section 5.6),
run on EPCS-Net's own best Run A checkpoint, that explicitly reports
everything the reviewer asked for:

  - Grad-CAM: per-sample overlap IoU with ground truth (not just a single
    mean), the exact binarisation threshold, and how "overlap" is computed.
  - HDBSCAN: every parameter used (min_cluster_size, min_samples, metric),
    plus a silhouette score, which the manuscript did not report.

Honest framing throughout: this is exploratory feature visualisation, not a
clinically validated diagnostic tool. No cluster or outlier label is given
any diagnostic interpretation anywhere in this code or its outputs -- that
mirrors the correction already made in the manuscript draft (removing
language like "malignant transformation") and reflects the real state of
validation (none against pathologist annotations).
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code_RunA" / "shared"))
from epcs_protocol_RunA import DATASETS, SAM_CHECKPOINT, checkpoint_state  # noqa: E402

# HDBSCAN parameters, reported explicitly here and in every output file --
# Reviewer 2 Issue 4.2 specifically flagged that the manuscript did not state
# these anywhere.
HDBSCAN_MIN_SAMPLES = 5
HDBSCAN_METRIC = "euclidean"
GRADCAM_THRESHOLD = 0.5  # matches the segmentation threshold used everywhere else in this project
PCA_CLUSTERING_DIMS = 40  # matches the "40-dimensional PCA-reduced feature space" referenced in review
MIN_COMPONENT_AREA_PX = 5  # drop near-empty connected components as noise, not real nuclei

RESULTS = {
    "cryonuseg": ROOT / "03_results_DPNP" / "CryoNuSeg_DPNP",
    "nuinsseg": ROOT / "03_results_DPNP" / "NuInsSeg_DPNP",
}


# --------------------------------------------------------------------------
# model loading (reuses the exact EPCS-Net architecture already validated in Run A)
# --------------------------------------------------------------------------
def load_epcs_net(dataset: str, device: torch.device):
    tag = DATASETS[dataset]["tag"]
    model_dir = ROOT / "02_code_RunA" / ("CryoNuSeg_RunA" if dataset == "cryonuseg" else "NuInsSeg_RunA")
    sys.path.insert(0, str(model_dir))
    module = importlib.import_module(f"epcs_net_{tag}_RunA")
    model = module.build(str(SAM_CHECKPOINT)).to(device)
    checkpoint_path = ROOT / "03_results_RunA" / ("CryoNuSeg_RunA" if dataset == "cryonuseg" else "NuInsSeg_RunA") \
        / "checkpoints" / f"best_model_epcs_net_{tag}_RunA.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Run A's best EPCS-Net checkpoint is missing: {checkpoint_path}. Run Run A first.")
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    incompatible = model.load_state_dict(saved["state_dict"], strict=False)
    missing = [k for k in incompatible.missing_keys if not k.startswith("sam.sam.")]
    if incompatible.unexpected_keys or missing:
        raise RuntimeError(f"Checkpoint mismatch loading EPCS-Net for DPNP: missing={missing}, unexpected={incompatible.unexpected_keys}")
    model.eval()
    return model, saved


# --------------------------------------------------------------------------
# 1. Segmentation-specific Grad-CAM (paper eq. 27-28)
# --------------------------------------------------------------------------
def gradcam_for_image(model, image: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Hooks model.d1 (the last feature map before the 1x1 output head) and
    computes CAM = ReLU(sum_c alpha_c * A^l,c) where alpha_c is the
    spatially-averaged gradient of the mean foreground probability S = mean(P)
    with respect to channel c. Returns (cam in [0,1] at input resolution,
    predicted probability map)."""
    activations = {}

    def hook(_module, _input, output):
        activations["value"] = output

    handle = model.d1.register_forward_hook(hook)
    image = image.clone().requires_grad_(False)
    logits = model(image)
    handle.remove()

    probs = torch.sigmoid(logits)
    score = probs.mean()  # S = mean(P), paper eq. 27
    grads = torch.autograd.grad(score, activations["value"], retain_graph=False)[0]

    alpha = grads.mean(dim=(2, 3), keepdim=True)  # spatial average of the gradient, per channel
    cam = F.relu((alpha * activations["value"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam_max = cam.max()
    cam = cam / cam_max if cam_max > 0 else cam
    return cam, probs[0, 0].detach().cpu().numpy()


def gradcam_overlap_iou(cam: np.ndarray, mask_gt: np.ndarray, threshold: float = GRADCAM_THRESHOLD) -> float:
    """Overlap is computed as IoU between the thresholded CAM and the ground
    truth nuclei mask -- stated explicitly because Reviewer 2 Issue 4.3 asked
    how 'overlap' was computed and it was not specified in the manuscript."""
    cam_bin = (cam >= threshold)
    gt_bin = (mask_gt > 0)
    intersection = np.logical_and(cam_bin, gt_bin).sum()
    union = np.logical_or(cam_bin, gt_bin).sum()
    return float(intersection / union) if union > 0 else float("nan")


# --------------------------------------------------------------------------
# 2. Nuclear phenotype feature extraction + ResNet18 embeddings
# --------------------------------------------------------------------------
def build_resnet18_embedder(device: torch.device):
    weights = ResNet18_Weights.IMAGENET1K_V1
    net = resnet18(weights=weights)
    net.fc = torch.nn.Identity()  # drop the classification head; keep the 512-d pooled feature
    net = net.to(device).eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    preprocess = weights.transforms()
    return net, preprocess


def extract_component_features(image_rgb: np.ndarray, binary_mask: np.ndarray, resnet, preprocess, device) -> pd.DataFrame:
    """Connected-component post-hoc feature extraction, paper eq. 29.

    NOTE ON DIMENSIONALITY: the paper's prose calls this a "six-dimensional
    feature vector" but its own equation (29) lists seven scalars
    [A, R, mu_I, Var_I, L*, a*, b*]. This implementation follows the equation
    literally (7 morphological scalars), not the prose label -- flagged here
    rather than silently reconciled either way."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary_mask > 0).astype(np.uint8), connectivity=8)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    rows = []
    for label_id in range(1, num_labels):  # 0 is background
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_AREA_PX:
            continue
        component_mask = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours) or 1.0
        roundness = float(4 * np.pi * area / (perimeter ** 2))

        pixel_values = gray[component_mask > 0].astype(np.float32)
        mean_intensity, var_intensity = float(pixel_values.mean()), float(pixel_values.var())
        lab_pixels = lab[component_mask > 0]
        mean_l, mean_a, mean_b = (float(v) for v in lab_pixels.mean(axis=0))

        x, y, w, h = stats[label_id, cv2.CC_STAT_LEFT], stats[label_id, cv2.CC_STAT_TOP], \
            stats[label_id, cv2.CC_STAT_WIDTH], stats[label_id, cv2.CC_STAT_HEIGHT]
        roi = image_rgb[y:y + h, x:x + w]
        roi_tensor = preprocess(torch.from_numpy(roi.transpose(2, 0, 1)).float().div(255.0)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(roi_tensor)[0].cpu().numpy()

        row = {
            "component_id": label_id, "area": area, "roundness": roundness,
            "mean_intensity": mean_intensity, "var_intensity": var_intensity,
            "L": mean_l, "a": mean_a, "b": mean_b,
        }
        for i, value in enumerate(embedding):
            row[f"resnet18_{i}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. PCA + HDBSCAN
# --------------------------------------------------------------------------
def phenotype_cluster(features: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    morphological_cols = ["area", "roundness", "mean_intensity", "var_intensity", "L", "a", "b"]
    embedding_cols = [c for c in features.columns if c.startswith("resnet18_")]
    feature_matrix = features[morphological_cols + embedding_cols].to_numpy()

    scaled = StandardScaler().fit_transform(feature_matrix)

    pca_report = PCA(n_components=3, random_state=42).fit(scaled)
    pca_table = {
        "PC1_explained_variance": float(pca_report.explained_variance_ratio_[0]),
        "PC2_explained_variance": float(pca_report.explained_variance_ratio_[1]),
        "PC3_explained_variance": float(pca_report.explained_variance_ratio_[2]),
        "PC1_PC2_PC3_total": float(pca_report.explained_variance_ratio_[:3].sum()),
    }

    n_components = max(2, min(PCA_CLUSTERING_DIMS, scaled.shape[0] - 1, scaled.shape[1]))
    reduced = PCA(n_components=n_components, random_state=42).fit_transform(scaled)

    min_cluster_size = max(5, len(features) // 10)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=HDBSCAN_MIN_SAMPLES, metric=HDBSCAN_METRIC)
    labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = int((labels == -1).sum())
    try:
        non_outlier = labels != -1
        silhouette = float(silhouette_score(reduced[non_outlier], labels[non_outlier])) if n_clusters >= 2 and non_outlier.sum() > n_clusters else None
    except ValueError:
        silhouette = None

    features = features.copy()
    features["pca_dims_used_for_clustering"] = n_components
    features["cluster_label"] = labels
    features["is_outlier"] = labels == -1
    for i in range(3):
        features[f"pc{i+1}"] = pca_report.transform(scaled)[:, i]

    # Sensitivity check: the paper's own max(5, N/10) formula scales
    # min_cluster_size with the pooled nucleus count, which can be stricter
    # than the data supports for a small/heterogeneous pool. Re-run with a
    # fixed, commonly-used min_cluster_size=5 so a degenerate result under the
    # paper's formula (e.g. 0 clusters) can be distinguished from a broken
    # pipeline -- reported alongside, not instead of, the primary result.
    fixed_clusterer = HDBSCAN(min_cluster_size=5, min_samples=HDBSCAN_MIN_SAMPLES, metric=HDBSCAN_METRIC)
    fixed_labels = fixed_clusterer.fit_predict(reduced)
    fixed_n_clusters = len(set(fixed_labels)) - (1 if -1 in fixed_labels else 0)
    fixed_n_outliers = int((fixed_labels == -1).sum())
    try:
        fixed_non_outlier = fixed_labels != -1
        fixed_silhouette = float(silhouette_score(reduced[fixed_non_outlier], fixed_labels[fixed_non_outlier])) \
            if fixed_n_clusters >= 2 and fixed_non_outlier.sum() > fixed_n_clusters else None
    except ValueError:
        fixed_silhouette = None

    hdbscan_params = {
        "primary_config": {
            "min_cluster_size": min_cluster_size,
            "min_cluster_size_formula": "max(5, N/10), as stated in the paper",
            "min_samples": HDBSCAN_MIN_SAMPLES,
            "metric": HDBSCAN_METRIC,
            "n_clusters_found": n_clusters,
            "n_outliers": n_outliers,
            "outlier_fraction": n_outliers / len(features) if len(features) else None,
            "silhouette_score": silhouette,
        },
        "sensitivity_check_fixed_min_cluster_size_5": {
            "min_cluster_size": 5,
            "min_samples": HDBSCAN_MIN_SAMPLES,
            "metric": HDBSCAN_METRIC,
            "n_clusters_found": fixed_n_clusters,
            "n_outliers": fixed_n_outliers,
            "outlier_fraction": fixed_n_outliers / len(features) if len(features) else None,
            "silhouette_score": fixed_silhouette,
        },
        "pca_dims_used_for_clustering": n_components,
        "n_nuclei_total": len(features),
        "silhouette_score_note": "computed excluding outlier-labelled (-1) points; None if fewer than 2 clusters were found",
    }
    return features, hdbscan_params, pca_table


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def run_dpnp(dataset: str) -> None:
    config = DATASETS[dataset]
    tag = config["tag"]
    results_dir = RESULTS[dataset]
    overlays_dir = results_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _ = load_epcs_net(dataset, device)
    resnet, preprocess = build_resnet18_embedder(device)

    split_path = Path(config["split_csv"])
    split = pd.read_csv(split_path)
    val_frame = split[split["split"] == "validation"].reset_index(drop=True)
    predictions_dir = ROOT / "03_results_RunA" / ("CryoNuSeg_RunA" if dataset == "cryonuseg" else "NuInsSeg_RunA") \
        / "predictions" / "epcs_net"

    print(f"[{tag}] DPNP: {len(val_frame)} validation images")

    # ---- 1. Grad-CAM, per image ----
    gradcam_rows = []
    for i, row in val_frame.iterrows():
        image = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
        mask_gt = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_gt, (256, 256), interpolation=cv2.INTER_NEAREST)
        image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1).copy()).float().div(255.0).unsqueeze(0).to(device)

        cam, _ = gradcam_for_image(model, image_tensor)
        overlap = gradcam_overlap_iou(cam, mask_resized)
        gradcam_rows.append({"filename": row.filename, "gradcam_iou_overlap": overlap, "threshold": GRADCAM_THRESHOLD})

        if i < 5:  # save a handful of overlays, not all -- these are illustrative, not a bulk output
            heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR), 0.5, heat, 0.5, 0)
            cv2.imwrite(str(overlays_dir / f"gradcam_{Path(row.filename).stem}.png"), overlay)

    gradcam_df = pd.DataFrame(gradcam_rows)
    gradcam_df.to_csv(results_dir / f"gradcam_per_image_{tag}_DPNP.csv", index=False)
    gradcam_summary = {
        "dataset": tag, "n_images": len(gradcam_df),
        "threshold_used": GRADCAM_THRESHOLD,
        "overlap_metric": "IoU between thresholded CAM and ground-truth nuclei mask",
        "target_layer": "model.d1 (final feature map before the 1x1 output head)",
        "gradcam_score_definition": "S = mean(sigmoid(logits)) over the full image (paper eq. 27)",
        "mean_overlap_iou": float(gradcam_df["gradcam_iou_overlap"].mean()),
        "sd_overlap_iou": float(gradcam_df["gradcam_iou_overlap"].std(ddof=1)) if len(gradcam_df) > 1 else None,
        "min_overlap_iou": float(gradcam_df["gradcam_iou_overlap"].min()),
        "max_overlap_iou": float(gradcam_df["gradcam_iou_overlap"].max()),
    }
    (results_dir / f"gradcam_summary_{tag}_DPNP.json").write_text(json.dumps(gradcam_summary, indent=2), encoding="utf-8")
    print(f"[{tag}] Grad-CAM: mean IoU overlap {gradcam_summary['mean_overlap_iou']:.4f} +/- {gradcam_summary['sd_overlap_iou']:.4f}")

    # ---- 2. Phenotype features from the model's own predicted masks, pooled across all validation images ----
    all_features = []
    for _, row in val_frame.iterrows():
        pred_path = predictions_dir / f"{Path(row.filename).stem}.png"
        if not pred_path.is_file():
            continue
        image = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        features = extract_component_features(image_resized, pred_mask, resnet, preprocess, device)
        if len(features):
            features.insert(0, "source_image", row.filename)
            all_features.append(features)

    if not all_features:
        print(f"[{tag}] WARNING: no nuclei components found across any validation image; skipping phenotyping.")
        return
    pooled = pd.concat(all_features, ignore_index=True)
    print(f"[{tag}] {len(pooled)} nuclear components pooled from {len(val_frame)} images for phenotyping")

    clustered, hdbscan_params, pca_table = phenotype_cluster(pooled)
    embedding_cols = [c for c in clustered.columns if c.startswith("resnet18_")]
    clustered.drop(columns=embedding_cols).to_csv(results_dir / f"phenotype_features_{tag}_DPNP.csv", index=False)
    (results_dir / f"hdbscan_params_{tag}_DPNP.json").write_text(json.dumps(hdbscan_params, indent=2), encoding="utf-8")
    pd.DataFrame([pca_table]).to_csv(results_dir / f"pca_explained_variance_{tag}_DPNP.csv", index=False)
    primary, sensitivity = hdbscan_params["primary_config"], hdbscan_params["sensitivity_check_fixed_min_cluster_size_5"]
    print(f"[{tag}] HDBSCAN (paper formula, min_cluster_size={primary['min_cluster_size']}): "
          f"{primary['n_clusters_found']} clusters, {primary['outlier_fraction']:.1%} outliers, silhouette={primary['silhouette_score']}")
    print(f"[{tag}] HDBSCAN (sensitivity check, min_cluster_size=5): "
          f"{sensitivity['n_clusters_found']} clusters, {sensitivity['outlier_fraction']:.1%} outliers, silhouette={sensitivity['silhouette_score']}")
    print(f"[{tag}] DPNP DONE -> {results_dir}")
