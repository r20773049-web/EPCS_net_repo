#!/usr/bin/env python3
"""MedSAM baseline: frozen, zero-shot inference, no fine-tuning.

Added in response to reviewer feedback (Reviewer 1 Issue 4, Reviewer 2 Issue
2.2) asking for a comparison against a real SAM-based medical segmentation
method, not just the paper's own internal SAM-guided ablation.

Design choice, stated plainly: MedSAM (Ma et al., Nature Communications 2024)
is a box-prompted model -- its intended workflow is "give me a box around one
object, I'll segment it," not "find every nucleus in this image." Our task is
dense multi-instance binary segmentation with no natural single bounding box.
Rather than engineer something bespoke that wouldn't be reproducible, this
uses the single most defensible, simplest reproducible protocol: one
whole-image box prompt, frozen official MedSAM weights, no fine-tuning, no
retraining, no threshold tuning. This is explicitly a stress test of a strong
existing SAM-based method on this exact task, not a claim that MedSAM is being
used the way its authors intended -- worth stating in the manuscript exactly
this plainly.

Uses the official authors' checkpoint via Hugging Face
(wanglab/medsam-vit-base, matching the bowang-lab GitHub org) through the
transformers library's SamModel/SamProcessor, which is a more robustly tested
loading path than hand-adapting the raw checkpoint into this project's
existing segment_anything-based loader.

Reuses the shared split/metric/reporting code from Run A so this baseline is
scored on the exact same validation images, with the exact same metric
definitions, as every other model in this project.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code_RunA" / "shared"))

from epcs_protocol_RunA import DATASETS, METRIC_COLUMNS, metric_row  # noqa: E402

MEDSAM_HF_ID = "wanglab/medsam-vit-base"
RESULTS = {
    "cryonuseg": ROOT / "03_results_MedSAM" / "CryoNuSeg_MedSAM",
    "nuinsseg": ROOT / "03_results_MedSAM" / "NuInsSeg_MedSAM",
}


def load_medsam(device: torch.device):
    from transformers import SamModel, SamProcessor

    print(f"Loading MedSAM ({MEDSAM_HF_ID}) -- first call downloads ~375MB from Hugging Face...")
    model = SamModel.from_pretrained(MEDSAM_HF_ID).to(device).eval()
    processor = SamProcessor.from_pretrained(MEDSAM_HF_ID)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, processor


@torch.no_grad()
def segment_one(model, processor, device, image_rgb: np.ndarray) -> np.ndarray:
    """Whole-image box prompt, frozen inference. Returns a binary mask (uint8,
    0/1) at the original image resolution."""
    h, w = image_rgb.shape[:2]
    box = [[0, 0, w, h]]
    inputs = processor(image_rgb, input_boxes=[box], return_tensors="pt").to(device)
    outputs = model(**inputs, multimask_output=True)
    # SAM/MedSAM proposes 3 candidate masks per prompt with predicted IoU
    # scores for each; take the one the model itself is most confident in.
    iou_scores = outputs.iou_scores[0, 0]
    best = int(torch.argmax(iou_scores).item())
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    mask = masks[0][0, best].numpy().astype(np.uint8)
    return mask


def run_medsam_dataset(dataset: str) -> None:
    config = DATASETS[dataset]
    tag = config["tag"]
    results_dir = RESULTS[dataset]
    predictions_dir = results_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, processor = load_medsam(device)

    split_path = Path(config["split_csv"])
    if not split_path.exists():
        raise FileNotFoundError(
            f"Run A's split manifest is missing: {split_path}. MedSAM must be scored on the "
            f"same validation split every other model uses -- run Run A first."
        )
    split = pd.read_csv(split_path)
    val_frame = split[split["split"] == "validation"].reset_index(drop=True)
    print(f"[{tag}] MedSAM zero-shot evaluation on {len(val_frame)} validation images from {split_path.name}")

    rows = []
    elapsed_total = 0.0
    for _, row in val_frame.iterrows():
        image = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
        mask_gt = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        start = time.perf_counter()
        pred = segment_one(model, processor, device, image_rgb)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_total += time.perf_counter() - start

        if pred.shape != mask_gt.shape:
            pred = cv2.resize(pred, (mask_gt.shape[1], mask_gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        pred_t = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        gt_t = torch.from_numpy((mask_gt > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        rows.append(metric_row(row.filename, pred_t, gt_t))
        cv2.imwrite(str(predictions_dir / f"{Path(row.filename).stem}.png"), (pred * 255).astype(np.uint8))

    per_image = pd.DataFrame(rows)
    per_image.to_csv(results_dir / f"per_image_metrics_medsam_{tag}.csv", index=False)

    ms_per_image = elapsed_total * 1000 / len(val_frame)
    columns = list(METRIC_COLUMNS)
    summary = {
        "model": "medsam", "dataset": tag, "checkpoint": MEDSAM_HF_ID,
        "protocol": "frozen zero-shot inference, whole-image box prompt, no fine-tuning",
        "split_csv": str(split_path.resolve()),
        "n_validation_images": len(val_frame),
        "validation_milliseconds_per_image": ms_per_image,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None,
        "metric_mean": per_image[columns].mean().to_dict(),
        "metric_sample_sd": per_image[columns].std(ddof=1).to_dict(),
    }
    (results_dir / f"summary_medsam_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{tag}] MedSAM DONE: Dice {summary['metric_mean']['dice']:.4f}, IoU {summary['metric_mean']['iou']:.4f}")
