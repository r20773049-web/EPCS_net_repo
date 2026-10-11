#!/usr/bin/env python3
"""Shared Run A protocol for the EPCS-Net comparison study.

Every per-model script (``unet_cryonuseg_RunA.py``, ``epcs_net_nuinsseg_RunA.py``,
and the ten others) defines only its own architecture and then hands it to
``run_single_model`` here. Everything that must be *identical* across the six
models lives in this one file:

    * the seed and deterministic CUDA settings
    * the saved train/validation split manifest, created once and reused
    * image loading, resizing and the single horizontal-flip augmentation
    * the metric definitions (TP/FP/TN/FN, accuracy, precision, recall,
      specificity, Dice, IoU) and how they are averaged
    * checkpoint-at-best-validation-Dice selection
    * the summary/report files that get written

That is deliberate. If each model owned a private copy of the metric or split
code it would be easy for one of them to drift, and the six results would stop
being comparable. Only the objective differs by design: EPCS-Net uses the FTAS
loss from the paper, every baseline uses BCE + Dice.

Both datasets share this file, so CryoNuSeg and NuInsSeg are trained and scored
by exactly the same code -- only the data-loading configuration differs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

MODEL_IDS = ("unet", "unetpp", "unet3plus", "eunet3plus", "sam_unet", "epcs_net")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
METRIC_COLUMNS = ("accuracy", "precision", "recall", "specificity", "dice", "iou")
SAM_CHECKPOINT = ROOT / "05_reference" / "sam_vit_b_01ec64.pth"

DATASETS = {
    "cryonuseg": {
        "tag": "cryonuseg",
        "kind": "flat",
        "images_dir": ROOT / "01_datasets" / "CryoNuSeg" / "tissue images",
        "masks_dir": ROOT / "01_datasets" / "CryoNuSeg" / "Annotator 1 (biologist)" / "mask binary",
        "split_csv": ROOT / "02_code_RunA" / "CryoNuSeg_RunA" / "splits" / "split_cryonuseg_24_6_seed42_RunA.csv",
        "results_dir": ROOT / "03_results_RunA" / "CryoNuSeg_RunA",
        "expected_total": 30,
        "expected_validation": 6,
        "ospt": False,
    },
    "nuinsseg": {
        "tag": "nuinsseg",
        "kind": "organ",
        "dataset_root": ROOT / "01_datasets" / "NuInsSeg",
        "image_subdir": "tissue images",
        "mask_subdir": "mask binary without border erode",
        "split_csv": ROOT / "02_code_RunA" / "NuInsSeg_RunA" / "splits" / "split_nuinsseg_80_20_seed42_RunA.csv",
        "results_dir": ROOT / "03_results_RunA" / "NuInsSeg_RunA",
        "expected_total": 665,
        "expected_validation": 133,
        "ospt": True,
    },
}


# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    """Apply the same seed and deterministic CUDA settings before every model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def stem_key(path: Path) -> str:
    # Masks are exported both as `name.png` and `name (1).png`; both must pair
    # with the same `name.tif` source image.
    stem = re.sub(r"\s*\(1\)$", "", path.stem, flags=re.IGNORECASE)
    return stem.lower().replace(" ", "_").replace("-", "_")


# --------------------------------------------------------------------------
# split manifests
# --------------------------------------------------------------------------
def _index_folder(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        return {}
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    output: dict[str, Path] = {}
    for path in files:
        key = stem_key(path)
        if key in output:
            raise ValueError(f"Duplicate filename stem '{key}' in {folder}: {output[key]} and {path}")
        output[key] = path
    return output


def _pair(images: dict[str, Path], masks: dict[str, Path], where: str) -> list[dict[str, str]]:
    missing_masks = sorted(set(images) - set(masks))
    extra_masks = sorted(set(masks) - set(images))
    if missing_masks or extra_masks:
        raise ValueError(
            f"{where}: images and masks must have matching filename stems. "
            f"Missing masks: {missing_masks[:5]}; extra masks: {extra_masks[:5]}"
        )
    return [
        {"filename": images[key].name, "image_path": str(images[key].resolve()), "mask_path": str(masks[key].resolve())}
        for key in sorted(images)
    ]


def build_records(config: dict) -> list[dict[str, str]]:
    if config["kind"] == "flat":
        images = _index_folder(Path(config["images_dir"]))
        masks = _index_folder(Path(config["masks_dir"]))
        if not images:
            raise FileNotFoundError(f"No images found under {config['images_dir']}")
        return _pair(images, masks, config["tag"])

    root = Path(config["dataset_root"])
    organ_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not organ_dirs:
        raise FileNotFoundError(f"No organ subdirectories found under {root}")
    records: list[dict[str, str]] = []
    for organ_dir in organ_dirs:
        images = _index_folder(organ_dir / config["image_subdir"])
        if not images:
            continue
        masks = _index_folder(organ_dir / config["mask_subdir"])
        for record in _pair(images, masks, organ_dir.name):
            record["organ"] = organ_dir.name
            records.append(record)
    if not records:
        raise FileNotFoundError(f"No paired image/mask files found under {root}")
    return records


def make_or_load_split(args: argparse.Namespace, config: dict) -> pd.DataFrame:
    """Load the saved manifest, or create it once. Every model reuses this file,
    so all six are trained and scored on exactly the same images."""
    split_path = Path(args.split_csv)
    if split_path.exists():
        df = pd.read_csv(split_path)
        if "split" not in df.columns:
            raise ValueError(f"Existing split has no 'split' column: {split_path}")
        n_val = int((df["split"] == "validation").sum())
        if len(df) != config["expected_total"] or n_val != config["expected_validation"]:
            raise ValueError(
                f"Existing split is not the expected {config['expected_total']}-image manifest "
                f"with {config['expected_validation']} validation images: {split_path}"
            )
        for column in ("image_path", "mask_path"):
            missing = [p for p in df[column] if not Path(p).is_file()]
            if missing:
                raise FileNotFoundError(f"Saved split points to missing files, e.g. {missing[0]}")
        return df
    if not args.create_split:
        raise FileNotFoundError(f"Split does not exist: {split_path}. Run once with --create-split.")

    records = build_records(config)
    if len(records) != config["expected_total"]:
        raise ValueError(f"{config['tag']} expects {config['expected_total']} paired images; found {len(records)}.")
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(records))
    n_val = config["expected_validation"]
    val_positions = set(order[:n_val].tolist())
    for i, record in enumerate(records):
        record["split"] = "validation" if i in val_positions else "train"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records).sort_values("filename").reset_index(drop=True)
    df.to_csv(split_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Created immutable {len(df) - n_val}/{n_val} split: {split_path}")
    return df


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
class NucleiDataset(Dataset):
    """Identical loading for both datasets: resize to a square, binarise the
    mask, and apply one horizontal flip during training."""

    def __init__(self, frame: pd.DataFrame, train: bool, image_size: int):
        self.frame, self.train, self.image_size = frame.reset_index(drop=True), train, image_size

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Could not read {row.image_path} / {row.mask_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.train and random.random() < 0.5:
            image, mask = np.ascontiguousarray(image[:, ::-1]), np.ascontiguousarray(mask[:, ::-1])
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)
        mask = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        return image, mask, row.filename


def organ_balanced_sampler(train_frame: pd.DataFrame) -> WeightedRandomSampler:
    """Organ-Stratified Pan-Cancer Training (OSPT), paper section 4.2: make every
    organ equally likely to be drawn each epoch so the model does not overfit to
    whichever organ contributes the most patches. The paper specifies the goal
    ("balanced multi-tissue representation within every training epoch") but no
    exact algorithm, so this weighted sampler is an approximation of that goal
    rather than a literal reproduction."""
    counts = train_frame["organ"].value_counts()
    weights = train_frame["organ"].map(lambda organ: 1.0 / counts[organ]).to_numpy()
    return WeightedRandomSampler(weights=weights, num_samples=len(train_frame), replacement=True)


# --------------------------------------------------------------------------
# shared building blocks (used by more than one architecture)
# --------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x): return self.net(x)


class GLUBlock(nn.Module):
    """Gated convolution block used only by eU-Net3+."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.gate = nn.Conv2d(in_ch, out_ch * 2, 3, padding=1, bias=False)
        self.refine = nn.Sequential(nn.BatchNorm2d(out_ch), nn.GELU(), nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU())

    def forward(self, x): return self.refine(F.glu(self.gate(x), dim=1))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.block = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class Encoder(nn.Module):
    def __init__(self, base: int, block=ConvBlock):
        super().__init__()
        self.e1, self.e2 = block(3, base), block(base, base * 2)
        self.e3, self.e4 = block(base * 2, base * 4), block(base * 4, base * 8)
        self.e5, self.pool = block(base * 8, base * 16), nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3)); e5 = self.e5(self.pool(e4))
        return e1, e2, e3, e4, e5


class FullScaleFuse(nn.Module):
    def __init__(self, channels: list[int], out_ch: int):
        super().__init__()
        self.proj = nn.ModuleList(nn.Sequential(nn.Conv2d(c, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)) for c in channels)
        self.fuse = ConvBlock(out_ch * len(channels), out_ch * len(channels))

    def forward(self, xs, target):
        resized = []
        for x, project in zip(xs, self.proj):
            if x.shape[-2:] != target.shape[-2:]:
                x = F.adaptive_max_pool2d(x, target.shape[-2:]) if x.shape[-2] > target.shape[-2] else F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)
            resized.append(project(x))
        return self.fuse(torch.cat(resized, 1))


class SAMImageEncoder(nn.Module):
    """Real frozen SAM ViT-B embeddings; no prompt-free SAM mask is claimed."""
    def __init__(self, checkpoint: str):
        super().__init__()
        from segment_anything import sam_model_registry
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        for parameter in self.sam.parameters():
            parameter.requires_grad_(False)
        self.sam.eval()

    def train(self, mode: bool = True):
        # The parent segmentation model trains, but SAM remains frozen in eval mode.
        super().train(False)
        self.sam.eval()
        return self

    def forward(self, x):
        x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False).mul(255.0)
        with torch.no_grad():
            return self.sam.image_encoder(self.sam.preprocess(x))


# --------------------------------------------------------------------------
# objectives
# --------------------------------------------------------------------------
def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (probs.sum((1, 2, 3)) + targets.sum((1, 2, 3)) + eps)).mean()


class FTASLoss(nn.Module):
    """Focal Tversky Asymmetric Supervision, paper equations (22)-(24):
    0.3 * BCE + 0.7 * focal-Tversky, with false negatives penalised more heavily
    than false positives. Used by EPCS-Net only."""

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


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def binary_counts(probabilities, masks):
    pred = (probabilities >= 0.5)
    truth = (masks >= 0.5)
    return int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & ~truth).sum()), int((~pred & truth).sum())


def metric_row(filename, probability, mask):
    tp, fp, tn, fn = binary_counts(probability, mask)
    eps = 1e-12
    return {
        "filename": filename, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": (tp + tn) / (tp + fp + tn + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "dice": (2 * tp) / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
    }


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def checkpoint_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Do not duplicate the frozen, publicly available SAM weights in outputs."""
    return {name: tensor for name, tensor in model.state_dict().items() if not name.startswith("sam.sam.")}


@torch.no_grad()
def evaluate(model, loader, device, predictions_dir: Path | None = None):
    model.eval()
    rows, elapsed, count = [], 0.0, 0
    for images, masks, names in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        cuda_sync(device)
        start = time.perf_counter()
        probs = torch.sigmoid(model(images))
        cuda_sync(device)
        elapsed += time.perf_counter() - start
        for i, name in enumerate(names):
            rows.append(metric_row(name, probs[i:i + 1].cpu(), masks[i:i + 1].cpu()))
            count += 1
            if predictions_dir:
                pred = ((probs[i, 0].cpu().numpy() >= 0.5).astype(np.uint8) * 255)
                cv2.imwrite(str(predictions_dir / f"{Path(name).stem}.png"), pred)
    return pd.DataFrame(rows), (elapsed * 1000 / count if count else float("nan"))


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def refresh_combined_summary(config: dict) -> None:
    """Rebuild the one-row-per-model comparison table from each model's summary."""
    results_dir = Path(config["results_dir"])
    tag = config["tag"]
    rows = []
    for model_id in MODEL_IDS:
        path = results_dir / "model_summaries" / f"summary_{model_id}_{tag}_RunA.json"
        if not path.exists():
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        row = {key: value for key, value in item.items() if key not in ("metric_mean", "metric_sample_sd")}
        for metric, value in item["metric_mean"].items():
            row[metric] = value
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(results_dir / f"run_summary_{tag}_RunA.csv", index=False)


# --------------------------------------------------------------------------
# CLI + training entry point
# --------------------------------------------------------------------------
def parse_args(dataset: str, model_id: str) -> argparse.Namespace:
    config = DATASETS[dataset]
    p = argparse.ArgumentParser(description=f"Run A: {model_id} on {dataset}")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--sam-checkpoint", default=str(SAM_CHECKPOINT))
    p.add_argument("--split-csv", default=str(config["split_csv"]))
    p.add_argument("--results-dir", default=str(config["results_dir"]))
    p.add_argument("--create-split", action="store_true", help="Create the split manifest (only needed once, ever)")
    p.add_argument("--preflight", action="store_true", help="Forward one real image through the model and exit")
    p.add_argument("--no-resume", action="store_true", help="Ignore any saved resume checkpoint and train from epoch 1")
    if config["ospt"]:
        p.add_argument("--no-ospt", dest="ospt", action="store_false", help="Disable organ-balanced sampling")
        p.set_defaults(ospt=True)
    else:
        p.set_defaults(ospt=False)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error("--epochs and --batch-size must be positive")
    return args


def run_single_model(model_id: str, model_factory, dataset: str) -> None:
    """Train and score one model under the shared Run A protocol.

    ``model_factory`` takes the SAM checkpoint path (ignored by the four
    non-SAM baselines) and returns the model.
    """
    if model_id not in MODEL_IDS:
        raise ValueError(f"unknown model id: {model_id}")
    config = DATASETS[dataset]
    tag = config["tag"]
    args = parse_args(dataset, model_id)
    seed_everything(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("Run A is intended for the CUDA workstation; CUDA was not detected.")
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(device)} | CUDA {torch.version.cuda}")
    print(f"Model: {model_id} | Dataset: {tag} | Epochs: {args.epochs} | Seed: {args.seed}")

    split = make_or_load_split(args, config)
    results_dir = Path(args.results_dir)
    for sub in ("epoch_histories", "per_image_metrics", "model_summaries", "checkpoints", "predictions"):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)

    train_frame = split[split["split"] == "train"]
    val_frame = split[split["split"] == "validation"]
    sampler = organ_balanced_sampler(train_frame) if args.ospt else None
    train_loader = DataLoader(
        NucleiDataset(train_frame, True, args.image_size),
        batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        NucleiDataset(val_frame, False, args.image_size),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=device.type == "cuda",
    )

    model = model_factory(args.sam_checkpoint).to(device)

    if args.preflight:
        images, masks, _ = next(iter(val_loader))
        with torch.no_grad():
            output = model(images.to(device))
        if output.shape != masks.shape:
            raise RuntimeError(f"{model_id}: output {tuple(output.shape)} != mask {tuple(masks.shape)}")
        print(f"PREFLIGHT OK: {model_id} -> {tuple(output.shape)}")
        return

    criterion = FTASLoss() if model_id == "epcs_net" else None
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_path = results_dir / "checkpoints" / f"best_model_{model_id}_{tag}_RunA.pt"
    resume_path = results_dir / "checkpoints" / f"resume_{model_id}_{tag}_RunA.pt"
    history_path = results_dir / "epoch_histories" / f"epoch_history_{model_id}_{tag}_RunA.csv"
    best_dice, best_epoch, history, start_epoch = -1.0, None, [], 1

    if resume_path.exists() and not args.no_resume:
        # Long runs are split into batches of epochs; pick up exactly where the
        # previous invocation stopped. Model and optimizer state are restored
        # exactly; only the augmentation RNG stream restarts.
        state = torch.load(resume_path, map_location=device, weights_only=False)
        incompatible = model.load_state_dict(state["model_state"], strict=False)
        missing = [k for k in incompatible.missing_keys if not k.startswith("sam.sam.")]
        if incompatible.unexpected_keys or missing:
            raise RuntimeError(f"Resume checkpoint mismatch; missing={missing}, unexpected={incompatible.unexpected_keys}")
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch, best_dice, best_epoch = state["epoch"] + 1, state["best_dice"], state["best_epoch"]
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")
        print(f"{model_id}: resuming after epoch {start_epoch - 1} (best: epoch {best_epoch}, Dice {best_dice:.4f})")

    if start_epoch > args.epochs:
        print(f"{model_id}: already trained through epoch {start_epoch - 1}; refreshing reports only.")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        for images, masks, _ in tqdm(train_loader, desc=f"{model_id} epoch {epoch}", leave=False):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks) if criterion else F.binary_cross_entropy_with_logits(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics, _ = evaluate(model, val_loader, device)
        val_dice = float(metrics.dice.mean())
        epoch_row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        for column in METRIC_COLUMNS:
            epoch_row[f"validation_{column}"] = float(metrics[column].mean())
        history.append(epoch_row)
        if val_dice > best_dice:
            best_dice, best_epoch = val_dice, epoch
            torch.save(
                {"model_id": model_id, "dataset": tag, "epoch": epoch, "validation_dice": val_dice, "seed": args.seed,
                 "sam_backbone": "vit_b" if model_id in ("sam_unet", "epcs_net") else None,
                 "state_dict": checkpoint_state(model)},
                best_path,
            )
        torch.save(
            {"epoch": epoch, "best_dice": best_dice, "best_epoch": best_epoch,
             "model_state": checkpoint_state(model), "optimizer_state": optimizer.state_dict()},
            resume_path,
        )
        pd.DataFrame(history).to_csv(history_path, index=False)

    saved = torch.load(best_path, map_location=device, weights_only=False)
    incompatible = model.load_state_dict(saved["state_dict"], strict=False)
    missing = [k for k in incompatible.missing_keys if not k.startswith("sam.sam.")]
    if incompatible.unexpected_keys or missing:
        raise RuntimeError(f"Checkpoint mismatch; missing={missing}, unexpected={incompatible.unexpected_keys}")

    predictions_dir = results_dir / "predictions" / model_id
    predictions_dir.mkdir(parents=True, exist_ok=True)
    per_image, ms_per_image = evaluate(model, val_loader, device, predictions_dir)
    per_image.to_csv(results_dir / "per_image_metrics" / f"per_image_metrics_{model_id}_{tag}_RunA.csv", index=False)

    columns = list(METRIC_COLUMNS)
    summary = {
        "model": model_id,
        "dataset": tag,
        "seed": args.seed,
        "split_csv": str(Path(args.split_csv).resolve()),
        "split_sha256": hashlib.sha256(Path(args.split_csv).read_bytes()).hexdigest(),
        "ospt_enabled": bool(args.ospt),
        "best_epoch": best_epoch,
        "best_validation_dice": best_dice,
        "epochs_requested": args.epochs,
        "loss": "FTAS (0.3 BCE + 0.7 focal-Tversky)" if model_id == "epcs_net" else "BCEWithLogits + Dice",
        "parameters_total": total,
        "parameters_trainable": trainable,
        "validation_milliseconds_per_image": ms_per_image,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "metric_mean": per_image[columns].mean().to_dict(),
        "metric_sample_sd": per_image[columns].std(ddof=1).to_dict(),
    }
    (results_dir / "model_summaries" / f"summary_{model_id}_{tag}_RunA.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    refresh_combined_summary(config)
    print(f"{model_id} ({tag}): best epoch {best_epoch}, Dice {best_dice:.4f} -> {results_dir}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
