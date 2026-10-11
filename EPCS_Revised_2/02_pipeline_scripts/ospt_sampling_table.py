"""Reviewer 1 comment 3: per-organ sample counts before and after OSPT, directly verifiable.

OSPT = torch WeightedRandomSampler, weight_i = 1/N_organ(i), with replacement, num_samples = N_train.
Reported per organ (training partition of the revision-3 split):
  raw_images        images of that organ in the training split (what plain shuffling would show per epoch)
  raw_share         raw_images / N_train
  expected_draws    N_train / K  (exact expectation under OSPT; every organ has draw probability 1/K)
  ospt_mean_draws   mean draws per epoch measured by simulating the sampler for N_EPOCHS epochs
  ospt_sd_draws     SD of draws per epoch (sampling noise: balance holds in expectation, not as a quota)
  draws_per_image   ospt_mean_draws / raw_images (>1 means minority organs are oversampled)
  frac_images_unseen  mean fraction of the organ's images never drawn in an epoch
Writes 04_results/analysis/ospt_per_organ_sampling_<dataset>.csv. Pure numpy, seconds, no GPU.
"""
from pathlib import Path

import numpy as np
import pandas as pd

R3 = Path(__file__).resolve().parents[1]
SPLITS = R3 / "04_results" / "splits"
OUT = R3 / "04_results" / "analysis"
N_EPOCHS = 2000
rng = np.random.default_rng(20261007)


def table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    train = df[df["split"] == "train"].reset_index(drop=True)
    organs = sorted(train["organ"].unique())
    k, n = len(organs), len(train)
    counts = train["organ"].value_counts().reindex(organs)
    weights = train["organ"].map(1.0 / counts).to_numpy()
    p = weights / weights.sum()
    org_idx = train["organ"].map({o: i for i, o in enumerate(organs)}).to_numpy()
    draws = np.zeros((N_EPOCHS, k))
    unseen = np.zeros((N_EPOCHS, k))
    for e in range(N_EPOCHS):
        picks = rng.choice(n, size=n, replace=True, p=p)
        draws[e] = np.bincount(org_idx[picks], minlength=k)
        seen = np.zeros(n, dtype=bool)
        seen[picks] = True
        for j in range(k):
            m = org_idx == j
            unseen[e, j] = 1.0 - seen[m].mean()
    rows = []
    for j, o in enumerate(organs):
        rows.append({"organ": o, "raw_images": int(counts[o]), "raw_share": counts[o] / n,
                     "expected_draws": n / k, "ospt_mean_draws": draws[:, j].mean(),
                     "ospt_sd_draws": draws[:, j].std(ddof=1),
                     "draws_per_image": draws[:, j].mean() / counts[o],
                     "frac_images_unseen": unseen[:, j].mean()})
    out = pd.DataFrame(rows)
    out.insert(0, "dataset", label)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nu = pd.read_csv(SPLITS / "nuinsseg_R3_stratified_70_15_15.csv")
    cols = {c.lower(): c for c in nu.columns}
    nu = nu.rename(columns={cols.get("partition", cols.get("split", "split")): "split"})
    t = table(nu, "nuinsseg")
    t.to_csv(OUT / "ospt_per_organ_sampling_nuinsseg.csv", index=False)
    print(f"NuInsSeg: {len(t)} organs, N_train={int(t.raw_images.sum())}, simulated {N_EPOCHS} epochs")
    print(f"  raw images per organ: min {t.raw_images.min()}, max {t.raw_images.max()} (ratio {t.raw_images.max()/t.raw_images.min():.1f}x)")
    print(f"  OSPT mean draws per organ: min {t.ospt_mean_draws.min():.2f}, max {t.ospt_mean_draws.max():.2f}, expected {t.expected_draws.iloc[0]:.2f}")
    print(f"  max |mean - expected| = {(t.ospt_mean_draws - t.expected_draws).abs().max():.3f}; mean per-epoch SD {t.ospt_sd_draws.mean():.2f}")
    print(f"  mean fraction of images unseen per epoch: {t.frac_images_unseen.mean():.3f}")
    print(t.round(3).head(4).to_string(index=False))


if __name__ == "__main__":
    main()
