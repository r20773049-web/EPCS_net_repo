"""
Segmentation Grad-CAM, with the low-overlap failure panel.

Reviewer 2 comment 3, raised twice now:

   "The Grad-CAM methodology has been substantially improved, but the
    low-overlap failure-case panel requested previously has not been included.
    Since the authors themselves report a wide overlap range, particularly on
    NuInsSeg, including several representative low-overlap examples would
    further strengthen the transparency of this analysis."

The CAM itself reproduces the published method exactly, so the numbers stay
comparable: hook the last decoder feature map `d1` before the 1x1 output head,
take the score S as the mean foreground probability, average the gradient
spatially per channel to get the channel weights, sum, ReLU, upsample, and
normalise to [0, 1]. Overlap with ground truth is the IoU of the CAM
thresholded at 0.5 against the ground-truth mask.

What is added
-------------
* The full per-sample overlap distribution, not just a mean and a range.
* A figure whose BOTTOM half is the worst cases, selected by lowest overlap,
  shown at the same scale and with the same annotations as the best cases.
  The best cases alone are what the reviewer objected to.
* A short written characterisation of what the worst cases have in common
  (foreground fraction, number of components, Dice), so the panel is evidence
  rather than decoration.

Usage:
    python gradcam_r3.py --run <run_dir>          # a revision-3 run directory
    python gradcam_r3.py --list                   # show available runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

R3 = Path(__file__).resolve().parents[2]
RESULTS = R3 / "04_results"
OUT = RESULTS / "analysis"
FIGS = R3 / "05_manuscript" / "figures"

sys.path.insert(0, str(R3 / "03_code" / "shared"))
from models_r3 import build_model                                 # noqa: E402
from protocol_r3 import load_split, seed_everything               # noqa: E402

IMAGE_SIZE = 256
GRADCAM_THRESHOLD = 0.5
N_BEST = 3
N_WORST = 4


def gradcam_for_image(model, image: torch.Tensor):
    """Published method, unchanged: hook d1, score = mean foreground probability."""
    activations = {}

    def hook(_m, _i, output):
        activations["value"] = output

    handle = model.d1.register_forward_hook(hook)
    logits = model(image)
    handle.remove()

    probs = torch.sigmoid(logits)
    score = probs.mean()
    grads = torch.autograd.grad(score, activations["value"], retain_graph=False)[0]
    alpha = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((alpha * activations["value"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    mx = cam.max()
    return (cam / mx if mx > 0 else cam), probs[0, 0].detach().cpu().numpy()


def overlap_iou(cam, gt, threshold=GRADCAM_THRESHOLD) -> float:
    a = cam >= threshold
    b = gt > 0.5
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float("nan")


def load_run(run_dir: Path, device_name: str = "auto"):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if device_name == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    seed_everything(summary["seed"])
    model = build_model(summary["model"]).to(device)
    state = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    inc = model.load_state_dict(state["state_dict"], strict=False)
    missing = [k for k in inc.missing_keys
               if not (k.startswith("context.sam.") or k.startswith("sam.sam."))]
    if inc.unexpected_keys or missing:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:4]}, "
                           f"unexpected={inc.unexpected_keys[:4]}")
    model.eval()
    return model, summary, device


def read_pair(rec):
    img = cv2.imread(rec.image_path, cv2.IMREAD_COLOR)
    gt = cv2.imread(rec.mask_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE),
                                  interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    gt = cv2.resize(gt, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    return img, (gt > 0).astype(np.float32)


def panel(rows, title, path):
    """rows: list of dicts with image, gt, pred, cam, label lines."""
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(13.5, 3.25 * n))
    if n == 1:
        axes = axes[None, :]
    headers = ["Input (H&E)", "Ground truth", "Prediction", "Grad-CAM overlay"]
    for r, row in enumerate(rows):
        panels = [row["image"], row["gt"], row["pred"], None]
        for c in range(4):
            ax = axes[r, c]
            if c == 3:
                ax.imshow(row["image"])
                ax.imshow(row["cam"], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            elif c == 0:
                ax.imshow(panels[c])
            else:
                ax.imshow(panels[c], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(headers[c], fontsize=12, pad=8)
        axes[r, 0].set_ylabel(row["label"], fontsize=9, rotation=0,
                              ha="right", va="center", labelpad=90)
        for s in axes[r, 0].spines.values():
            s.set_visible(True)
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="auto", help="auto (default), cpu, cuda:0")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N images of the split (smoke tests only)")
    ap.add_argument("--out-dir", default=None,
                    help="write CSV/JSON/figure here instead of the default locations")
    a = ap.parse_args()
    global OUT, FIGS
    if a.out_dir:
        OUT = FIGS = Path(a.out_dir)

    # Completed runs only: has best_model.pt, not inside an archive folder (leading "_").
    rel = lambda p: p.relative_to(RESULTS).parts
    runs = sorted(p.parent for p in RESULTS.rglob("summary.json")
                  if "epcs_net" in p.parent.parts
                  and not any(x.startswith("_") for x in rel(p))
                  and (p.parent / "best_model.pt").is_file())
    if a.list or not runs:
        print("Completed EPCS-Net runs:")
        for r in runs:
            print(f"  {r}")
        if not runs:
            print("  (none yet; the sweep is still running)")
        return 0

    run_dir = Path(a.run) if a.run else runs[-1]
    model, summary, device = load_run(run_dir, a.device)
    split, _ = load_split(summary["dataset"], summary.get("fold"))
    frame = split[split.split == a.split]
    if a.limit:
        frame = frame.head(a.limit)
    print(f"Run     : {summary['run_id']}")
    print(f"Split   : {a.split}  ({len(frame)} images)")

    records = []
    for _, rec in frame.iterrows():
        img, gt = read_pair(rec)
        x = torch.from_numpy(img.transpose(2, 0, 1).copy()).float().div_(255.0) \
            .unsqueeze(0).to(device)
        cam, prob = gradcam_for_image(model, x)
        pred = (prob >= 0.5).astype(np.float32)
        inter = float((pred * gt).sum())
        dice = 2 * inter / (pred.sum() + gt.sum() + 1e-8)
        records.append({
            "filename": rec.filename, "organ": rec.get("organ", ""),
            "gradcam_iou_overlap": overlap_iou(cam, gt),
            "dice": dice,
            "gt_foreground_fraction": float(gt.mean()),
            "pred_foreground_fraction": float(pred.mean()),
            "n_gt_components": int(cv2.connectedComponents(
                (gt > 0.5).astype(np.uint8), connectivity=8)[0] - 1),
            "_image": img, "_gt": gt, "_pred": pred, "_cam": cam,
        })

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in records])
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    tag = summary["run_id"]
    df.to_csv(OUT / f"gradcam_per_image_{tag}.csv", index=False)

    order = sorted(range(len(records)), key=lambda i: records[i]["gradcam_iou_overlap"])
    worst = order[:N_WORST]
    best = order[-N_BEST:][::-1]

    def mk(i, kind):
        r = records[i]
        return {"image": r["_image"], "gt": r["_gt"], "pred": r["_pred"], "cam": r["_cam"],
                "label": (f"{kind}\n{Path(r['filename']).stem[:26]}\n"
                          f"CAM IoU {r['gradcam_iou_overlap']:.3f}\n"
                          f"Dice {r['dice']:.3f}\n"
                          f"GT fg {r['gt_foreground_fraction']*100:.1f}%")}

    rows = [mk(i, "HIGHEST OVERLAP") for i in best] + \
           [mk(i, "LOWEST OVERLAP") for i in worst]
    fig_path = FIGS / f"fig13_gradcam_best_and_failure_cases_{summary['dataset']}.png"
    panel(rows, f"Grad-CAM agreement with ground truth, {summary['dataset']} test split "
                f"({N_BEST} highest and {N_WORST} lowest overlap cases)", fig_path)

    lo = df.nsmallest(N_WORST, "gradcam_iou_overlap")
    hi = df.nlargest(N_BEST, "gradcam_iou_overlap")
    report = {
        "run_id": tag, "dataset": summary["dataset"], "split": a.split,
        "n_images": int(len(df)), "threshold": GRADCAM_THRESHOLD,
        "method": "hook d1, score = mean foreground probability, channel weights = "
                  "spatial mean of gradients, ReLU of weighted sum, bilinear upsample, "
                  "normalise to [0,1]; identical to the previously published method",
        "overlap_mean": float(df.gradcam_iou_overlap.mean()),
        "overlap_sd": float(df.gradcam_iou_overlap.std(ddof=1)),
        "overlap_median": float(df.gradcam_iou_overlap.median()),
        "overlap_p10": float(df.gradcam_iou_overlap.quantile(0.10)),
        "overlap_p90": float(df.gradcam_iou_overlap.quantile(0.90)),
        "overlap_min": float(df.gradcam_iou_overlap.min()),
        "overlap_max": float(df.gradcam_iou_overlap.max()),
        "failure_panel_figure": str(fig_path),
        "worst_cases": lo[["filename", "organ", "gradcam_iou_overlap", "dice",
                           "gt_foreground_fraction", "n_gt_components"]].to_dict("records"),
        "best_cases": hi[["filename", "organ", "gradcam_iou_overlap", "dice",
                          "gt_foreground_fraction", "n_gt_components"]].to_dict("records"),
        "what_distinguishes_failures": {
            "mean_gt_foreground_fraction_worst": float(lo.gt_foreground_fraction.mean()),
            "mean_gt_foreground_fraction_best": float(hi.gt_foreground_fraction.mean()),
            "mean_dice_worst": float(lo.dice.mean()),
            "mean_dice_best": float(hi.dice.mean()),
            "mean_n_components_worst": float(lo.n_gt_components.mean()),
            "mean_n_components_best": float(hi.n_gt_components.mean()),
        },
    }
    (OUT / f"gradcam_report_{tag}.json").write_text(json.dumps(report, indent=2),
                                                    encoding="utf-8")

    print(f"\nCAM overlap IoU over {len(df)} images")
    print(f"  mean {report['overlap_mean']:.4f}  sd {report['overlap_sd']:.4f}  "
          f"median {report['overlap_median']:.4f}")
    print(f"  range {report['overlap_min']:.4f} to {report['overlap_max']:.4f}   "
          f"p10 {report['overlap_p10']:.4f}  p90 {report['overlap_p90']:.4f}")
    w = report["what_distinguishes_failures"]
    print(f"\nWorst {N_WORST} vs best {N_BEST} cases")
    print(f"  GT foreground fraction  {w['mean_gt_foreground_fraction_worst']*100:>6.2f}%  vs "
          f"{w['mean_gt_foreground_fraction_best']*100:>6.2f}%")
    print(f"  Dice                    {w['mean_dice_worst']:>6.3f}   vs {w['mean_dice_best']:>6.3f}")
    print(f"  GT components           {w['mean_n_components_worst']:>6.1f}   vs "
          f"{w['mean_n_components_best']:>6.1f}")
    print(f"\nFailure panel: {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
