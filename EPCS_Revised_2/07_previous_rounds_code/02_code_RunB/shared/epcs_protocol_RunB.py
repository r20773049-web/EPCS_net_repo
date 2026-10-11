#!/usr/bin/env python3
"""Shared Run B protocol: EPCS-Net repeated across five seeds.

Run A answers "which architecture wins under one training run." Run B answers
a different question: "how much does EPCS-Net's own result vary just from
random initialisation and training stochasticity?" Answering that requires
holding everything else fixed and changing only the seed -- which is why this
module deliberately reuses, unchanged, the exact split manifest Run A created
(see ``SEEDS`` and ``run_multi_seed`` below). If the train/validation split
also changed per seed, split-variance and initialisation-variance would be
mixed together and the resulting standard deviation would not answer the
question Run B is for.

Everything dataset-agnostic (image loading, augmentation, metric definitions,
checkpoint-at-best-Dice selection, the resume mechanism) is imported from
Run A's ``epcs_protocol_RunA.py`` rather than duplicated. The EPCS-Net
architecture itself is imported directly from Run A's own
``epcs_net_<dataset>_RunA.py`` scripts, so the model is defined in exactly one
place -- not re-typed a third time for Run B.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code_RunA" / "shared"))

from epcs_protocol_RunA import (  # noqa: E402
    DATASETS,
    METRIC_COLUMNS,
    SAM_CHECKPOINT,
    NucleiDataset,
    checkpoint_state,
    dice_loss,
    evaluate,
    make_or_load_split,
    organ_balanced_sampler,
    seed_everything,
)
from epcs_protocol_RunA import FTASLoss  # noqa: E402  (kept separate: it's the one deliberate difference)

MODEL_ID = "epcs_net"

# The five seeds used for every Run B execution, on both datasets. Fixed here,
# not passed on the command line, so "Run B" always means these five values --
# there is no risk of accidentally running a different set on one dataset.
SEEDS = (42, 123, 456, 789, 2024)

RUNB_RESULTS = {
    "cryonuseg": ROOT / "03_results_RunB" / "CryoNuSeg_RunB",
    "nuinsseg": ROOT / "03_results_RunB" / "NuInsSeg_RunB",
}


class _Args:
    """Minimal stand-in for argparse.Namespace, built directly instead of
    parsed from the command line, since Run B's seeds and paths are fixed by
    this module rather than chosen per invocation."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _split_args(dataset: str, config: dict) -> _Args:
    # create_split is deliberately False: Run B must never generate its own
    # split. It fails loudly if Run A's manifest is missing rather than
    # silently creating a different one.
    return _Args(split_csv=str(config["split_csv"]), create_split=False, seed=SEEDS[0])


def run_one_seed(dataset: str, seed: int, epochs: int, no_resume: bool = False) -> dict:
    """Train and score EPCS-Net for one seed. Returns that seed's summary dict."""
    config = DATASETS[dataset]
    tag = config["tag"]
    results_dir = RUNB_RESULTS[dataset] / f"seed_{seed}"
    for sub in ("checkpoints", "predictions"):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Run B is intended for the CUDA workstation; CUDA was not detected.")
    device = torch.device("cuda:0")
    print(f"[{tag} seed {seed}] Device: {torch.cuda.get_device_name(device)} | CUDA {torch.version.cuda}")

    split = make_or_load_split(_split_args(dataset, config), config)
    train_frame = split[split["split"] == "train"]
    val_frame = split[split["split"] == "validation"]
    sampler = organ_balanced_sampler(train_frame) if config["ospt"] else None
    train_loader = DataLoader(
        NucleiDataset(train_frame, True, 256), batch_size=1,
        shuffle=sampler is None, sampler=sampler, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(NucleiDataset(val_frame, False, 256), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model_module_dir = ROOT / "02_code_RunA" / ("CryoNuSeg_RunA" if dataset == "cryonuseg" else "NuInsSeg_RunA")
    sys.path.insert(0, str(model_module_dir))
    import importlib
    model_module = importlib.import_module(f"epcs_net_{tag}_RunA")
    model = model_module.build(str(SAM_CHECKPOINT)).to(device)

    criterion = FTASLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4, weight_decay=1e-4)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    torch.cuda.reset_peak_memory_stats(device)

    best_path = results_dir / "checkpoints" / f"best_model_epcs_net_seed{seed}_{tag}_RunB.pt"
    resume_path = results_dir / "checkpoints" / f"resume_epcs_net_seed{seed}_{tag}_RunB.pt"
    history_path = results_dir / f"epoch_history_epcs_net_seed{seed}_{tag}_RunB.csv"
    best_dice, best_epoch, history, start_epoch = -1.0, None, [], 1

    if resume_path.exists() and not no_resume:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        incompatible = model.load_state_dict(state["model_state"], strict=False)
        missing = [k for k in incompatible.missing_keys if not k.startswith("sam.sam.")]
        if incompatible.unexpected_keys or missing:
            raise RuntimeError(f"Resume checkpoint mismatch; missing={missing}, unexpected={incompatible.unexpected_keys}")
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch, best_dice, best_epoch = state["epoch"] + 1, state["best_dice"], state["best_epoch"]
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")
        print(f"[{tag} seed {seed}] resuming after epoch {start_epoch - 1} (best: epoch {best_epoch}, Dice {best_dice:.4f})")

    if start_epoch > epochs:
        print(f"[{tag} seed {seed}] already trained through epoch {start_epoch - 1}; skipping training.")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        losses = []
        for images, masks, _ in tqdm(train_loader, desc=f"{tag} seed{seed} epoch {epoch}", leave=False):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
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
                {"model_id": MODEL_ID, "dataset": tag, "seed": seed, "epoch": epoch, "validation_dice": val_dice,
                 "sam_backbone": "vit_b", "state_dict": checkpoint_state(model)},
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

    predictions_dir = results_dir / "predictions"
    per_image, ms_per_image = evaluate(model, val_loader, device, predictions_dir)
    per_image.to_csv(results_dir / f"per_image_metrics_epcs_net_seed{seed}_{tag}_RunB.csv", index=False)

    columns = list(METRIC_COLUMNS)
    summary = {
        "model": MODEL_ID, "dataset": tag, "seed": seed,
        "split_csv": str(Path(config["split_csv"]).resolve()),
        "split_sha256": hashlib.sha256(Path(config["split_csv"]).read_bytes()).hexdigest(),
        "best_epoch": best_epoch, "best_validation_dice": best_dice, "epochs_requested": epochs,
        "loss": "FTAS (0.3 BCE + 0.7 focal-Tversky)",
        "parameters_total": total, "parameters_trainable": trainable,
        "validation_milliseconds_per_image": ms_per_image,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "metric_mean": per_image[columns].mean().to_dict(),
        "metric_sample_sd": per_image[columns].std(ddof=1).to_dict(),
    }
    (results_dir / f"summary_epcs_net_seed{seed}_{tag}_RunB.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # The resume checkpoint is deliberately NOT deleted here. Run B is driven
    # through the same "--epochs 5, then --epochs 10, ..." batching pattern
    # Run A used, and each of those calls reaches this point having only
    # trained *some* of the eventual target -- deleting the resume state after
    # every one of them would silently restart the next batch from epoch 1.
    # Cleanup of finished resume checkpoints is a separate, explicit step
    # (see drop_finished_resume_checkpoints below), run only once every seed
    # has genuinely reached its final target epoch count.
    print(f"[{tag} seed {seed}] DONE: best epoch {best_epoch}, Dice {best_dice:.4f}")
    del model
    torch.cuda.empty_cache()

    row = {"seed": seed, "best_epoch": best_epoch, "best_validation_dice": best_dice,
           "epochs_requested": epochs, "parameters_trainable": trainable,
           "validation_milliseconds_per_image": ms_per_image,
           "cuda_peak_allocated_mib": summary["cuda_peak_allocated_mib"]}
    row.update(summary["metric_mean"])
    return row


def refresh_runb_reports(dataset: str) -> None:
    """Rebuild the per-seed table, the mean+/-SD aggregate, and the seed/split
    tracking file from whatever seed folders currently exist -- safe to call
    after every seed, whether or not all five are done yet."""
    config = DATASETS[dataset]
    tag = config["tag"]
    results_dir = RUNB_RESULTS[dataset]
    rows = []
    for seed in SEEDS:
        path = results_dir / f"seed_{seed}" / f"summary_epcs_net_seed{seed}_{tag}_RunB.json"
        if not path.exists():
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        row = {key: value for key, value in item.items() if key not in ("metric_mean", "metric_sample_sd")}
        for metric, value in item["metric_mean"].items():
            row[metric] = value
        rows.append(row)
    if not rows:
        return

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(results_dir / f"run_summary_{tag}_RunB.csv", index=False)

    columns = list(METRIC_COLUMNS)
    agg = {"dataset": tag, "seeds_completed": len(per_seed), "seeds_planned": len(SEEDS)}
    for column in columns:
        agg[f"{column}_mean"] = per_seed[column].mean()
        agg[f"{column}_sd"] = per_seed[column].std(ddof=1) if len(per_seed) > 1 else float("nan")
    agg["best_epoch_mean"] = per_seed["best_epoch"].mean()
    agg["best_epoch_sd"] = per_seed["best_epoch"].std(ddof=1) if len(per_seed) > 1 else float("nan")
    pd.DataFrame([agg]).to_csv(results_dir / f"run_b_aggregate_{tag}_RunB.csv", index=False)

    # Seed + split tracking: proves every seed used the identical Run A split.
    split_path = Path(config["split_csv"])
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest() if split_path.exists() else None
    n_val = int((pd.read_csv(split_path)["split"] == "validation").sum()) if split_path.exists() else None
    n_total = int(len(pd.read_csv(split_path))) if split_path.exists() else None
    tracking_rows = [
        {
            "dataset": tag, "seed_index": i + 1, "seed_value": seed,
            "completed": (results_dir / f"seed_{seed}" / f"summary_epcs_net_seed{seed}_{tag}_RunB.json").exists(),
            "split_csv": str(split_path.resolve()), "split_sha256": split_hash,
            "n_total_images": n_total, "n_train": (n_total - n_val) if n_total and n_val else None, "n_validation": n_val,
        }
        for i, seed in enumerate(SEEDS)
    ]
    pd.DataFrame(tracking_rows).to_csv(results_dir / f"seeds_and_splits_{tag}_RunB.csv", index=False)


def drop_finished_resume_checkpoints(dataset: str, target_epochs: int = 50, seeds=None) -> None:
    """Delete a seed's resume checkpoint only once it has actually reached
    ``target_epochs`` -- verified by reading the checkpoint's own epoch number,
    not assumed from the caller. Safe to call at any time; does nothing to a
    seed that is not yet finished."""
    seeds = seeds if seeds is not None else SEEDS
    config = DATASETS[dataset]
    tag = config["tag"]
    for seed in seeds:
        resume_path = RUNB_RESULTS[dataset] / f"seed_{seed}" / "checkpoints" / f"resume_epcs_net_seed{seed}_{tag}_RunB.pt"
        if not resume_path.exists():
            continue
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["epoch"] >= target_epochs:
            resume_path.unlink()
            print(f"[{tag} seed {seed}] reached epoch {state['epoch']} >= {target_epochs}; removed resume checkpoint.")
        else:
            print(f"[{tag} seed {seed}] only at epoch {state['epoch']} of {target_epochs}; resume checkpoint kept.")


def run_multi_seed(dataset: str, epochs: int, seeds=None, no_resume: bool = False) -> None:
    seeds = seeds if seeds is not None else SEEDS
    RUNB_RESULTS[dataset].mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        run_one_seed(dataset, seed, epochs, no_resume=no_resume)
        try:
            refresh_runb_reports(dataset)
        except PermissionError as error:
            # A locked report CSV (e.g. open in Excel) must not lose an
            # otherwise-successful seed. Each seed's own summary JSON already
            # has the real numbers; the next successful refresh rebuilds the
            # combined tables from every seed's summary, not just this one's.
            print(f"WARNING: could not refresh Run B reports ({error}); per-seed results are unaffected, continuing.")
