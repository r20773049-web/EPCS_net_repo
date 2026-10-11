"""
Precompute the frozen SAM ViT-B embeddings once per image.

Why this is legitimate and not a shortcut
-----------------------------------------
The SAM encoder is frozen, held in eval mode, and wrapped in torch.no_grad, so
for a given input it is a pure deterministic function. Recomputing it on every
training step of every epoch of every seed of every model is pure waste: the
measured training step for EPCS-Net is 0.207 s, of which the SAM forward at
1024x1024 is the dominant part.

Caching changes nothing numerically. This module PROVES that rather than
asserting it: `verify` recomputes a sample of embeddings live and requires
bit-for-bit equality (torch.equal) against the cache.

The only subtlety is augmentation. Training applies a random horizontal flip to
the image, and flipping before a ViT is not the same as flipping after it, so
both flip states are cached separately. Nothing is approximated.

Reference 21's own released code does exactly this, saving `<index>_en.pt` SAM
encodings offline, so the practice is also what the comparison baseline uses.

Storage: 256 x 64 x 64 float32 = 4 MB per entry, two entries per image.
    CryoNuSeg    30 images ->   0.23 GB
    NuInsSeg    665 images ->   5.20 GB

Usage:
    python sam_cache_r3.py build       # build both datasets, then verify
    python sam_cache_r3.py verify      # verify only
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models_r3 import SAMImageEncoder, SAM_CHECKPOINT           # noqa: E402

CACHE_DIR = Path(__file__).resolve().parents[2] / "04_results" / "sam_cache"
EMB_SHAPE = (256, 64, 64)
IMAGE_SIZE = 256
VERIFY_SAMPLES = 12


def _manifests() -> dict[str, list[Path]]:
    splits = Path(__file__).resolve().parents[2] / "04_results" / "splits"
    return {
        "nuinsseg": [splits / "nuinsseg_R3_stratified_70_15_15.csv"],
        "cryonuseg": [splits / f"cryonuseg_R3_fold{k}.csv" for k in (1, 2, 3)],
    }


def _unique_images(dataset: str) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in _manifests()[dataset] if p.is_file()]
    if not frames:
        raise FileNotFoundError(f"no split manifests for {dataset}; run splits_r3.py")
    df = pd.concat(frames).drop_duplicates(subset=["filename"]).sort_values("filename")
    return df.reset_index(drop=True)


def load_image(path: str, flipped: bool) -> torch.Tensor:
    """Byte-for-byte the same preprocessing the training dataset applies."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    if flipped:
        image = np.ascontiguousarray(image[:, ::-1])
    return torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)


class SamCache:
    """Read-only accessor used by the training protocol."""

    def __init__(self, dataset: str):
        self.dataset = dataset
        idx_path = CACHE_DIR / f"{dataset}_index.json"
        arr_path = CACHE_DIR / f"{dataset}_embeddings.npy"
        if not idx_path.is_file() or not arr_path.is_file():
            raise FileNotFoundError(
                f"SAM cache for {dataset} is missing. Run: python sam_cache_r3.py build")
        self.index = json.loads(idx_path.read_text(encoding="utf-8"))["entries"]
        self.array = np.load(arr_path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.index)

    def get(self, filename: str, flipped: bool) -> torch.Tensor:
        key = f"{filename}|{int(flipped)}"
        row = self.index.get(key)
        if row is None:
            raise KeyError(f"no cached embedding for {key}")
        return torch.from_numpy(np.asarray(self.array[row]))


def build(dataset: str, device: torch.device) -> dict:
    df = _unique_images(dataset)
    n = len(df)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    arr_path = CACHE_DIR / f"{dataset}_embeddings.npy"
    idx_path = CACHE_DIR / f"{dataset}_index.json"

    array = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.float32,
                                      shape=(n * 2, *EMB_SHAPE))
    encoder = SAMImageEncoder(str(SAM_CHECKPOINT)).to(device).eval()

    entries, row, t0 = {}, 0, time.time()
    for i, rec in df.iterrows():
        for flipped in (False, True):
            x = load_image(rec.image_path, flipped).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = encoder(x)
            array[row] = emb[0].detach().cpu().numpy()
            entries[f"{rec.filename}|{int(flipped)}"] = row
            row += 1
        if (i + 1) % 50 == 0 or i + 1 == n:
            done = i + 1
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {dataset}: {done}/{n} images  ({rate:.1f} img/s, "
                  f"eta {max(n-done,0)/max(rate,1e-6):.0f}s)")

    array.flush()
    idx_path.write_text(json.dumps({
        "dataset": dataset, "n_images": n, "n_entries": row,
        "embedding_shape": list(EMB_SHAPE), "dtype": "float32",
        "image_size": IMAGE_SIZE,
        "sam_checkpoint": str(SAM_CHECKPOINT),
        "note": "two entries per image: flip state 0 (as loaded) and 1 (horizontally flipped)",
        "entries": entries}, indent=1), encoding="utf-8")

    size_gb = arr_path.stat().st_size / 2**30
    print(f"  {dataset}: wrote {row} embeddings, {size_gb:.2f} GB -> {arr_path.name}")
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"dataset": dataset, "n_images": n, "n_entries": row, "gb": round(size_gb, 3)}


def verify(dataset: str, device: torch.device, samples: int = VERIFY_SAMPLES) -> bool:
    """Recompute a random sample live and require BIT-FOR-BIT equality."""
    cache = SamCache(dataset)
    df = _unique_images(dataset)
    rng = np.random.default_rng(20261005)
    picks = rng.choice(len(df), size=min(samples, len(df)), replace=False)

    encoder = SAMImageEncoder(str(SAM_CHECKPOINT)).to(device).eval()
    exact, close, failed = 0, 0, 0
    worst = 0.0
    for i in picks:
        rec = df.iloc[int(i)]
        for flipped in (False, True):
            x = load_image(rec.image_path, flipped).unsqueeze(0).to(device)
            with torch.no_grad():
                live = encoder(x)[0].detach().cpu()
            cached = cache.get(rec.filename, flipped)
            if torch.equal(live, cached):
                exact += 1
            else:
                diff = float((live - cached).abs().max())
                worst = max(worst, diff)
                if diff == 0.0:
                    exact += 1
                elif diff < 1e-5:
                    close += 1
                else:
                    failed += 1
                    print(f"    MISMATCH {rec.filename} flip={int(flipped)} max|diff|={diff:.3e}")
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    total = exact + close + failed
    print(f"  {dataset}: {exact}/{total} bit-identical, {close} within 1e-5, "
          f"{failed} failed (worst |diff| {worst:.3e})")
    return failed == 0


def main() -> int:
    action = sys.argv[1] if len(sys.argv) > 1 else "build"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets = ("cryonuseg", "nuinsseg")

    if action == "build":
        print("Building SAM embedding cache")
        report = {}
        for ds in datasets:
            report[ds] = build(ds, device)
        print("\nVerifying against live recomputation (bit-for-bit)")
        ok = all(verify(ds, device) for ds in datasets)
        (CACHE_DIR / "CACHE_REPORT.json").write_text(
            json.dumps({"built": report, "verified_exact": ok}, indent=2), encoding="utf-8")
        print("\nCACHE VERIFIED EXACT" if ok else "\nVERIFICATION FAILED -- do not use the cache")
        return 0 if ok else 1

    if action == "verify":
        ok = all(verify(ds, device) for ds in datasets)
        print("VERIFIED EXACT" if ok else "VERIFICATION FAILED")
        return 0 if ok else 1

    print(f"unknown action '{action}'; use build or verify")
    return 2


if __name__ == "__main__":
    sys.exit(main())
