#!/usr/bin/env python3
"""MedSAM on the ROUND-3 TEST splits (Reviewer 1, comment 4).

Modes
-----
zero_shot_wholebox : round-2 protocol (one whole-image box, 3 candidate masks,
                     keep the highest predicted-IoU one). Sanity check only.
oracle_boxes       : one box per ground-truth connected component of the binary
                     mask (8-connectivity); the per-box masks are unioned into
                     one binary prediction. ORACLE UPPER BOUND: it uses
                     ground-truth localisation, so it is not a competitor and
                     not a deployable pipeline. Component counts are recorded.

Metrics come from shared/protocol_r3.py::metric_row (same definitions as all
other models). Scoring is at the ORIGINAL image resolution, as in round 2.
No network access unless --allow-download is given. Runs on CPU unless
--device cuda is given explicitly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
from protocol_r3 import METRIC_COLUMNS, SPLITS, RESULTS, metric_row  # noqa: E402

MEDSAM_HF_ID = "wanglab/medsam-vit-base"
MODES = ("zero_shot_wholebox", "oracle_boxes")
OUT_ROOT = RESULTS / "medsam_r3"
BOX_CHUNK = 32


# ---------------------------------------------------------------- splits
def split_jobs():
    """(tag, csv path) for every round-3 evaluation set."""
    return [(f"cryonuseg_fold{k}", SPLITS / f"cryonuseg_R3_fold{k}.csv") for k in (1, 2, 3)] + \
           [("nuinsseg", SPLITS / "nuinsseg_R3_stratified_70_15_15.csv")]


def read_test_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing split manifest {path}")
    df = pd.read_csv(path)
    return df[df["split"] == "test"].reset_index(drop=True)


# ---------------------------------------------------------------- boxes
def gt_boxes(mask_bin: np.ndarray):
    """Boxes [x0,y0,x1,y1] (inclusive pixel extents) of each 8-connected
    component of a binary mask. Returns (boxes, n_components)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask_bin > 0).astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i][:4])
        boxes.append([x, y, x + w - 1, y + h - 1])
    return boxes, n - 1


# ---------------------------------------------------------------- models
class StubModel:
    """Dry-run stand-in: the 'mask' for a box is the box region itself."""
    name = "stub"

    @staticmethod
    def _fill(shape, box):
        m = np.zeros(shape, np.uint8)
        m[box[1]:box[3] + 1, box[0]:box[2] + 1] = 1
        return m

    def wholebox(self, image_rgb):
        h, w = image_rgb.shape[:2]
        return self._fill((h, w), [0, 0, w - 1, h - 1])

    def boxes(self, image_rgb, boxes):
        h, w = image_rgb.shape[:2]
        return [self._fill((h, w), b) for b in boxes]


class MedSAMModel:
    name = MEDSAM_HF_ID

    def __init__(self, device, allow_download=False):
        from transformers import SamModel, SamProcessor
        local = not allow_download
        try:
            self.model = SamModel.from_pretrained(MEDSAM_HF_ID, local_files_only=local).to(device).eval()
            self.processor = SamProcessor.from_pretrained(MEDSAM_HF_ID, local_files_only=local)
        except Exception as e:  # noqa: BLE001
            if local:
                raise SystemExit(
                    f"ERROR: MedSAM weights '{MEDSAM_HF_ID}' are not in the local Hugging Face cache "
                    f"({type(e).__name__}: {e}).\nNothing was downloaded. Pre-fetch them yourself, or "
                    f"re-run with --allow-download to permit a download.") from None
            raise
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def wholebox(self, image_rgb):
        """Identical to round 2: box [0,0,w,h], multimask, argmax predicted IoU."""
        h, w = image_rgb.shape[:2]
        inputs = self.processor(image_rgb, input_boxes=[[[0, 0, w, h]]], return_tensors="pt").to(self.device)
        out = self.model(**inputs, multimask_output=True)
        best = int(torch.argmax(out.iou_scores[0, 0]).item())
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        return masks[0][0, best].numpy().astype(np.uint8)

    @torch.no_grad()
    def boxes(self, image_rgb, boxes):
        """Embed once, prompt in chunks; single-mask output (MedSAM's native box usage).
        Boxes are rescaled to the processor's resized frame like SamProcessor does."""
        if not boxes:
            return []
        h, w = image_rgb.shape[:2]
        inputs = self.processor(image_rgb, input_boxes=[[boxes[0]]], return_tensors="pt").to(self.device)
        emb = self.model.get_image_embeddings(inputs["pixel_values"])
        rh, rw = (int(v) for v in inputs["reshaped_input_sizes"][0].tolist())
        sx, sy = rw / w, rh / h
        res = []
        for i in range(0, len(boxes), BOX_CHUNK):
            chunk = torch.tensor([[b[0] * sx, b[1] * sy, (b[2] + 1) * sx, (b[3] + 1) * sy]
                                  for b in boxes[i:i + BOX_CHUNK]], dtype=torch.float32, device=self.device)
            out = self.model(image_embeddings=emb, input_boxes=chunk.unsqueeze(0), multimask_output=False)
            m = self.processor.image_processor.post_process_masks(
                out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
            res += [m[j, 0].numpy().astype(np.uint8) for j in range(m.shape[0])]
        return res


# ---------------------------------------------------------------- evaluation
def evaluate_frame(model, frame, mode, pred_dir=None):
    rows = []
    for _, r in frame.iterrows():
        img = cv2.imread(r.image_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(r.mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            raise RuntimeError(f"could not read {r.image_path} / {r.mask_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_bin = (gt > 0).astype(np.uint8)
        t0 = time.perf_counter()
        ncomp = None
        if mode == "zero_shot_wholebox":
            pred = model.wholebox(img)
        else:
            boxes, ncomp = gt_boxes(gt_bin)
            pred = np.zeros_like(gt_bin)
            for m in model.boxes(img, boxes):
                pred |= (m > 0).astype(np.uint8)
        dt = time.perf_counter() - t0
        if pred.shape != gt_bin.shape:
            pred = cv2.resize(pred, (gt_bin.shape[1], gt_bin.shape[0]), interpolation=cv2.INTER_NEAREST)
        row = metric_row(r.filename, torch.from_numpy(pred.astype(np.float32))[None, None],
                         torch.from_numpy(gt_bin.astype(np.float32))[None, None], r.get("organ", ""))
        row.update(n_gt_components=ncomp, seconds=dt)
        rows.append(row)
        if pred_dir is not None:
            cv2.imwrite(str(pred_dir / f"{Path(r.filename).stem.replace('/', '_')}.png"), pred * 255)
    return pd.DataFrame(rows)


def summarise(df, mode, tag, split_path, model_name, dry):
    cols = list(METRIC_COLUMNS)
    s = {"model": "medsam", "mode": mode, "dataset": tag, "checkpoint": model_name, "dry_run": dry,
         "split_csv": str(split_path), "split_used": "test", "n_test_images": int(len(df)),
         "metric_mean": df[cols].mean().to_dict(),
         "metric_sample_sd": df[cols].std(ddof=1).to_dict() if len(df) > 1 else None}
    if mode == "oracle_boxes":
        s["ORACLE_UPPER_BOUND"] = "uses ground-truth connected-component boxes; not a competitor"
        c = df.n_gt_components
        s["components_per_image"] = {"mean": float(c.mean()), "median": float(c.median()),
                                     "min": int(c.min()), "max": int(c.max()), "total": int(c.sum())}
        s["images_with_zero_components"] = int((c == 0).sum())
    else:
        s["protocol"] = "round-2 protocol: one whole-image box, best-predicted-IoU of 3 masks (sanity check)"
    return s


def unit_check_synthetic():
    """Rectangular GT components: the bounding box IS the component, so the stub
    oracle must reproduce the GT exactly (Dice ~ 1) and count components correctly."""
    gt = np.zeros((64, 80), np.uint8)
    gt[5:15, 5:20] = 1
    gt[30:40, 50:70] = 1
    gt[50:52, 10:12] = 1
    boxes, n = gt_boxes(gt)
    pred = np.zeros_like(gt)
    for m in StubModel().boxes(np.zeros((64, 80, 3), np.uint8), boxes):
        pred |= m
    row = metric_row("syn", torch.from_numpy(pred.astype(np.float32))[None, None],
                     torch.from_numpy(gt.astype(np.float32))[None, None], "")
    return {"n_components": n, "boxes": boxes, "dice": row["dice"],
            "ok": n == 3 and row["dice"] > 0.999999}


def unit_check_stub_oracle(frame):
    """On real masks the stub returns bounding boxes, so the union must contain
    the GT (recall == 1); Dice is < 1 by construction (boxes overfill)."""
    df = evaluate_frame(StubModel(), frame, "oracle_boxes")
    return {"recall_all_one": bool((df.recall > 0.999).all()), "mean_dice": float(df.dice.mean()),
            "min_dice": float(df.dice.min()), "mean_recall": float(df.recall.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--datasets", nargs="+", default=None, help="subset of tags, e.g. cryonuseg_fold1 nuinsseg")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda; cuda must be explicit")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="stub model, CPU, writes to medsam_r3/_dry_run")
    ap.add_argument("--limit", type=int, default=None, help="first N test images per set (debug)")
    a = ap.parse_args()

    out_root = OUT_ROOT / "_dry_run" if a.dry_run else OUT_ROOT
    device = torch.device("cpu" if a.dry_run else a.device)
    jobs = [j for j in split_jobs() if a.datasets is None or j[0] in a.datasets]
    if a.datasets is not None:
        known = [j[0] for j in split_jobs()]
        bad = [d for d in a.datasets if d not in known]
        if bad or not jobs:
            raise SystemExit(f"unknown --datasets {bad}; valid tags: {known}")
    model = StubModel() if a.dry_run else MedSAMModel(device, a.allow_download)
    print(f"model={model.name} device={device} out={out_root}")

    all_summ, failed = [], False
    if a.dry_run:
        syn = unit_check_synthetic()
        print(f"synthetic unit check: {syn}")
        failed = not syn["ok"]
    for tag, path in jobs:
        frame = read_test_frame(path)
        if a.limit:
            frame = frame.head(a.limit)
        print(f"[{tag}] {len(frame)} test images ({path.name})")
        if a.dry_run:
            chk = unit_check_stub_oracle(frame)
            print(f"[{tag}] unit check (stub oracle): {chk}")
            if not (chk["recall_all_one"] and chk["mean_dice"] > 0.5):
                failed = True
        for mode in a.modes:
            d = out_root / mode / tag
            (d / "predictions").mkdir(parents=True, exist_ok=True)
            df = evaluate_frame(model, frame, mode, None if a.dry_run else d / "predictions")
            df.to_csv(d / f"per_image_{mode}_{tag}.csv", index=False)
            s = summarise(df, mode, tag, path, model.name, a.dry_run)
            (d / f"summary_{mode}_{tag}.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
            all_summ.append(s)
            print(f"[{tag}] {mode}: dice {s['metric_mean']['dice']:.4f} iou {s['metric_mean']['iou']:.4f}")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary_all.json").write_text(json.dumps(all_summ, indent=2), encoding="utf-8")
    if failed:
        print("DRY-RUN CHECK FAILED")
        sys.exit(1)
    if a.dry_run:
        print("DRY-RUN PASSED")


if __name__ == "__main__":
    main()
