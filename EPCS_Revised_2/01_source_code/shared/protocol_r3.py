"""
Revision-3 training and evaluation protocol.

What changed from Run A, and why
--------------------------------
1. THREE-WAY SPLITS. The checkpoint is selected on VALIDATION only, and every
   reported score comes from an untouched TEST split that the selection never
   saw. Fixes Reviewer 3 major comment 1 and Reviewer 4 comment 1.
2. MATCHED SEEDS FOR EVERY MODEL. Every architecture runs on the same seed list
   over byte-identical splits, so paired tests are possible. Fixes Reviewer 3
   major comment 3, Reviewer 1, and Reviewer 2 comment 1.
3. OSPT ON BOTH DATASETS. CryoNuSeg manifests now carry tissue labels, so the
   organ-balanced sampler applies to both. Previously it was silently NuInsSeg
   only.
4. LATENCY IS NOT MEASURED HERE. Timing during a training sweep is
   contaminated by concurrent jobs, which is exactly how the published numbers
   came to disagree by 2.6x for the same model on the same images. Latency is
   measured separately by latency_r3.py on an idle GPU with warm-up.

Everything else -- batch size 1, AdamW at a fixed 3e-4, no scheduler, no
gradient clipping, FP32, 50 epochs, deterministic seeding -- is unchanged from
the published protocol so the comparison stays like for like.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models_r3 import MODELS, build_model, count_params          # noqa: E402
from sam_cache_r3 import SamCache                                # noqa: E402

R3 = Path(__file__).resolve().parents[2]
SPLITS = R3 / "04_results" / "splits"
RESULTS = R3 / "04_results"

METRIC_COLUMNS = ("accuracy", "precision", "recall", "specificity", "dice", "iou")
SEEDS = (42, 123, 456, 789, 2024)


# --------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


class NucleiDataset(Dataset):
    """Identical loading to Run A: resize to a square, binarise the mask, one
    random horizontal flip during training only."""

    def __init__(self, frame: pd.DataFrame, train: bool, image_size: int, cache=None):
        self.frame = frame.reset_index(drop=True)
        self.train = train
        self.image_size = image_size
        # Optional precomputed frozen SAM embedding. Numerically identical
        # to recomputing it; sam_cache_r3.verify asserts that bit-for-bit.
        self.cache = cache

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"could not read {row.image_path} / {row.mask_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        flipped = False
        # The RNG is consumed identically whether or not a cache is present,
        # so a cached run reproduces an uncached run exactly for one seed.
        if self.train and random.random() < 0.5:
            flipped = True
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)
        mask = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        ctx = (self.cache.get(row.filename, flipped) if self.cache is not None
               else torch.empty(0))
        return image, mask, row.filename, ctx


def organ_balanced_sampler(train_frame: pd.DataFrame) -> WeightedRandomSampler:
    """OSPT. weight(i) = 1 / count(organ of i), sampled WITH replacement.
    The weights sum to K (the number of organs), so every organ has draw
    probability exactly 1/K and an expected N_train/K draws per epoch."""
    counts = train_frame["organ"].value_counts()
    weights = train_frame["organ"].map(lambda o: 1.0 / counts[o]).to_numpy()
    return WeightedRandomSampler(weights=weights, num_samples=len(train_frame), replacement=True)


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------
def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (probs.sum((1, 2, 3)) + targets.sum((1, 2, 3)) + eps)).mean()


class FTASLoss(torch.nn.Module):
    """0.3 * BCE + 0.7 * focal-Tversky, alpha 0.7, beta 0.3, gamma 0.75."""

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        tp = (p * targets).sum((1, 2, 3))
        fp = (p * (1 - targets)).sum((1, 2, 3))
        fn = ((1 - p) * targets).sum((1, 2, 3))
        t = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        focal_tversky = ((1 - t).pow(self.gamma)).mean()
        return 0.3 * F.binary_cross_entropy_with_logits(logits, targets) + 0.7 * focal_tversky


def make_criterion(loss_id: str):
    if loss_id == "ftas":
        return FTASLoss(), "FTAS (0.3 BCE + 0.7 focal-Tversky)"
    return None, "BCEWithLogits + Dice"


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def metric_row(name: str, prob: torch.Tensor, target: torch.Tensor, organ: str) -> dict:
    pred = (prob >= 0.5).float()
    tp = float((pred * target).sum())
    fp = float((pred * (1 - target)).sum())
    fn = float(((1 - pred) * target).sum())
    tn = float(((1 - pred) * (1 - target)).sum())
    eps = 1e-8
    return {
        "filename": name,
        "organ": organ,
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
    }


def forward(model, images, ctx):
    """Pass the cached frozen embedding only to models whose forward takes it."""
    if ctx is not None and getattr(model, "accepts_ctx", False):
        return model(images, ctx)
    return model(images)


@torch.no_grad()
def evaluate(model, loader, device, organs: dict, predictions_dir: Path | None = None) -> pd.DataFrame:
    model.eval()
    rows = []
    for images, masks, names, ctx in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        ctx = ctx.to(device, non_blocking=True) if ctx.numel() else None
        probs = torch.sigmoid(forward(model, images, ctx))
        for i, name in enumerate(names):
            rows.append(metric_row(name, probs[i:i + 1].cpu(), masks[i:i + 1].cpu(),
                                   organs.get(name, "")))
            if predictions_dir is not None:
                pred = (probs[i, 0].cpu().numpy() >= 0.5).astype(np.uint8) * 255
                safe = Path(name).stem.replace("/", "_")
                cv2.imwrite(str(predictions_dir / f"{safe}.png"), pred)
    return pd.DataFrame(rows)


def checkpoint_state(model) -> dict:
    """Do not duplicate the frozen, publicly available SAM weights in outputs."""
    return {k: v for k, v in model.state_dict().items() if not k.startswith("context.sam.")
            and not k.startswith("sam.sam.")}


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------
def run_id(model_id: str, dataset: str, seed: int, fold: int | None) -> str:
    return (f"{model_id}__{dataset}__fold{fold}__seed{seed}" if fold
            else f"{model_id}__{dataset}__seed{seed}")


def run_dir(dataset: str, model_id: str, seed: int, fold: int | None) -> Path:
    group = MODELS[model_id]["group"]
    bucket = {"ablation": "ablation", "ablation_control": "ablation",
              "reference21": "reference21"}.get(group, dataset)
    sub = f"fold{fold}_seed{seed}" if fold else f"seed{seed}"
    return RESULTS / bucket / dataset / model_id / sub


def load_split(dataset: str, fold: int | None) -> tuple[pd.DataFrame, Path]:
    if dataset == "nuinsseg":
        path = SPLITS / "nuinsseg_R3_stratified_70_15_15.csv"
    elif dataset == "cryonuseg":
        if fold is None:
            raise ValueError("cryonuseg requires --fold (1, 2 or 3)")
        path = SPLITS / f"cryonuseg_R3_fold{fold}.csv"
    else:
        raise ValueError(dataset)
    if not path.is_file():
        raise FileNotFoundError(f"missing split manifest {path}; run splits_r3.py first")
    return pd.read_csv(path), path


def train_one(model_id: str, dataset: str, seed: int, fold: int | None,
              epochs: int = 50, image_size: int = 256, batch_size: int = 1,
              lr: float = 3e-4, weight_decay: float = 1e-4, ospt: bool = True,
              no_resume: bool = False, preflight: bool = False,
              use_sam_cache: bool = True) -> dict | None:
    spec = MODELS[model_id]
    split, split_path = load_split(dataset, fold)
    out = run_dir(dataset, model_id, seed, fold)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)

    organs = dict(zip(split["filename"], split.get("organ", pd.Series([""] * len(split)))))
    train_frame = split[split["split"] == "train"]
    val_frame = split[split["split"] == "validation"]
    test_frame = split[split["split"] == "test"]
    if len(train_frame) == 0 or len(val_frame) == 0 or len(test_frame) == 0:
        raise RuntimeError(f"empty partition in {split_path.name}")

    cache = None
    if spec["sam"] and use_sam_cache:
        try:
            cache = SamCache(dataset)
        except FileNotFoundError as exc:
            print(f"  (no SAM cache: {exc}; falling back to live SAM forwards)")

    sampler = organ_balanced_sampler(train_frame) if (ospt and "organ" in split) else None
    train_loader = DataLoader(NucleiDataset(train_frame, True, image_size, cache),
                              batch_size=batch_size, shuffle=sampler is None,
                              sampler=sampler, num_workers=0,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(NucleiDataset(val_frame, False, image_size, cache), batch_size=1,
                            shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(NucleiDataset(test_frame, False, image_size, cache), batch_size=1,
                             shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    model = build_model(model_id).to(device)
    import inspect
    model.accepts_ctx = "ctx" in inspect.signature(model.forward).parameters

    if preflight:
        images, masks, _, ctx = next(iter(val_loader))
        with torch.no_grad():
            output = forward(model, images.to(device),
                             ctx.to(device) if ctx.numel() else None)
        if output.shape != masks.shape:
            raise RuntimeError(f"{model_id}: output {tuple(output.shape)} != mask {tuple(masks.shape)}")
        total, trainable = count_params(model)
        print(f"PREFLIGHT OK  {model_id:<18} {dataset:<10} fold={fold} "
              f"out={tuple(output.shape)} params total={total:,} trainable={trainable:,}")
        return {"model": model_id, "dataset": dataset, "fold": fold,
                "params_total": total, "params_trainable": trainable}

    criterion, loss_label = make_criterion(spec["loss"])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=lr, weight_decay=weight_decay)
    total, trainable = count_params(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_path = out / "best_model.pt"
    resume_path = out / "resume.pt"
    history_path = out / "epoch_history.csv"
    best_dice, best_epoch, history, start_epoch = -1.0, None, [], 1

    if resume_path.exists() and not no_resume:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        inc = model.load_state_dict(state["model_state"], strict=False)
        missing = [k for k in inc.missing_keys
                   if not (k.startswith("context.sam.") or k.startswith("sam.sam."))]
        if inc.unexpected_keys or missing:
            raise RuntimeError(f"resume mismatch; missing={missing}, unexpected={inc.unexpected_keys}")
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = state["epoch"] + 1
        best_dice, best_epoch = state["best_dice"], state["best_epoch"]
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")

    wall_start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        losses = []
        for images, masks, _, ctx in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            ctx = ctx.to(device, non_blocking=True) if ctx.numel() else None
            optimizer.zero_grad(set_to_none=True)
            logits = forward(model, images, ctx)
            loss = (criterion(logits, masks) if criterion
                    else F.binary_cross_entropy_with_logits(logits, masks) + dice_loss(logits, masks))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(model, val_loader, device, organs)
        val_dice = float(val_metrics.dice.mean())
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        for c in METRIC_COLUMNS:
            row[f"validation_{c}"] = float(val_metrics[c].mean())
        history.append(row)

        if val_dice > best_dice:
            best_dice, best_epoch = val_dice, epoch
            torch.save({"model_id": model_id, "dataset": dataset, "fold": fold,
                        "epoch": epoch, "validation_dice": val_dice, "seed": seed,
                        "state_dict": checkpoint_state(model)}, best_path)
        torch.save({"epoch": epoch, "best_dice": best_dice, "best_epoch": best_epoch,
                    "model_state": checkpoint_state(model),
                    "optimizer_state": optimizer.state_dict()}, resume_path)
        pd.DataFrame(history).to_csv(history_path, index=False)

    wall_seconds = time.time() - wall_start

    # Restore the validation-selected checkpoint, then score the untouched test split.
    saved = torch.load(best_path, map_location=device, weights_only=False)
    inc = model.load_state_dict(saved["state_dict"], strict=False)
    missing = [k for k in inc.missing_keys
               if not (k.startswith("context.sam.") or k.startswith("sam.sam."))]
    if inc.unexpected_keys or missing:
        raise RuntimeError(f"checkpoint mismatch; missing={missing}, unexpected={inc.unexpected_keys}")

    pred_dir = out / "test_predictions"
    pred_dir.mkdir(exist_ok=True)
    test_metrics = evaluate(model, test_loader, device, organs, pred_dir)
    val_metrics = evaluate(model, val_loader, device, organs)
    test_metrics.to_csv(out / "per_image_test.csv", index=False)
    val_metrics.to_csv(out / "per_image_validation.csv", index=False)

    cols = list(METRIC_COLUMNS)
    summary = {
        "run_id": run_id(model_id, dataset, seed, fold),
        "model": model_id,
        "model_label": spec["label"],
        "group": spec["group"],
        "dataset": dataset,
        "fold": fold,
        "seed": seed,
        "split_manifest": str(split_path.resolve()),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "n_train": int(len(train_frame)), "n_validation": int(len(val_frame)),
        "n_test": int(len(test_frame)),
        "ospt_enabled": sampler is not None,
        "best_epoch": best_epoch,
        "best_validation_dice": best_dice,
        "epochs": epochs,
        "loss": loss_label,
        "parameters_total": total,
        "parameters_trainable": trainable,
        "cuda_peak_allocated_mib": (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                                    if device.type == "cuda" else None),
        "wall_seconds": round(wall_seconds, 1),
        "image_size": image_size, "batch_size": batch_size,
        "learning_rate": lr, "weight_decay": weight_decay,
        "python": sys.version.split()[0], "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "note": "latency is NOT measured here; see latency_r3.py (idle GPU, with warm-up)",
        "sam_embeddings_cached": cache is not None,
        "test_mean": test_metrics[cols].mean().to_dict(),
        "test_sd_across_images": test_metrics[cols].std(ddof=1).to_dict(),
        "validation_mean": val_metrics[cols].mean().to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    resume_path.unlink(missing_ok=True)

    print(f"DONE  {summary['run_id']:<46} val {best_dice:.4f} @ep{best_epoch:<3} "
          f"TEST dice {summary['test_mean']['dice']:.4f}  ({wall_seconds/60:.1f} min)")
    return summary


# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="EPCS-Net revision 3 single run")
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--dataset", required=True, choices=("cryonuseg", "nuinsseg"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--no-ospt", dest="ospt", action="store_false")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--no-sam-cache", dest="use_sam_cache", action="store_false")
    p.set_defaults(ospt=True, use_sam_cache=True)
    a = p.parse_args()

    train_one(a.model, a.dataset, a.seed, a.fold, a.epochs, a.image_size,
              a.batch_size, a.learning_rate, a.weight_decay, a.ospt,
              a.no_resume, a.preflight, a.use_sam_cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
