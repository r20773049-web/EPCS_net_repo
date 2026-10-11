"""
Aggregate every revision-3 run and compute the statistics the reviewers asked
for but that the previous submission could not provide.

What this answers
-----------------
Reviewer 1          : matched-seed comparison between EPCS-Net and the baselines
Reviewer 2 comment 1: whether superiority over the baselines is statistically
                      established, stated either way
Reviewer 3 major 3  : mean and SD or confidence intervals for EVERY model, with
                      paired comparisons on identical splits
Reviewer 4 comment 1: all scores come from the untouched test split

Design decisions, stated explicitly because they affect the conclusions
----------------------------------------------------------------------
* The pairing unit is (seed) on NuInsSeg and (fold, seed) on CryoNuSeg. Both
  models in a pair saw byte-identical data, so the pairing is exact.
* Two tests are reported for every comparison: a paired t-test, and a Wilcoxon
  signed-rank test which does not assume normality. With five pairs the
  Wilcoxon has a floor of p = 0.0625, so it CANNOT reach 0.05 on NuInsSeg no
  matter how large the effect. That is a property of the sample size and is
  reported rather than hidden by quoting only the t-test.
* Holm-Bonferroni correction is applied across the comparisons within each
  dataset, since EPCS-Net is compared against several baselines at once.
* A bootstrap confidence interval over per-image test Dice differences is also
  reported. It has far more resamples available than the seed-level test and so
  is the more informative interval, but it measures a different thing: variation
  across images, not across training runs. Both are given.

Usage:
    python stats_r3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R3 = Path(__file__).resolve().parents[2]
RESULTS = R3 / "04_results"
OUT = RESULTS / "analysis"

sys.path.insert(0, str(R3 / "03_code" / "shared"))
from models_r3 import MODELS                                   # noqa: E402

METRICS = ("dice", "iou", "accuracy", "precision", "recall", "specificity")
PROPOSED = "epcs_net"
N_BOOTSTRAP = 10_000
RNG = np.random.default_rng(20261005)


# --------------------------------------------------------------------------
def collect_runs() -> pd.DataFrame:
    rows = []
    for summary_path in sorted(RESULTS.rglob("summary.json")):
        if any(part.startswith("_archive") for part in summary_path.parts):
            continue          # audit S3: archived old-GPU runs must never be counted
        try:
            s = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "test_mean" not in s:
            continue
        row = {
            "run_id": s["run_id"], "model": s["model"], "label": s.get("model_label", s["model"]),
            "group": s.get("group", ""), "dataset": s["dataset"], "fold": s.get("fold"),
            "seed": s["seed"], "best_epoch": s["best_epoch"],
            "best_validation_dice": s["best_validation_dice"],
            "n_train": s.get("n_train"), "n_validation": s.get("n_validation"),
            "n_test": s.get("n_test"),
            "params_total": s.get("parameters_total"),
            "params_trainable": s.get("parameters_trainable"),
            "wall_seconds": s.get("wall_seconds"),
            "peak_vram_mib": s.get("cuda_peak_allocated_mib"),
            "sam_cached": s.get("sam_embeddings_cached"),
            "dir": str(summary_path.parent),
        }
        for m in METRICS:
            row[f"test_{m}"] = s["test_mean"].get(m)
            row[f"test_{m}_sd_images"] = s.get("test_sd_across_images", {}).get(m)
            row[f"val_{m}"] = s.get("validation_mean", {}).get(m)
        rows.append(row)
    if not rows:
        raise SystemExit("No completed runs found. Run the sweep first.")
    df = pd.DataFrame(rows)
    dup = df[df.duplicated("run_id", keep=False)]
    assert dup.empty, f"duplicate run_id values (double counting): {sorted(dup.run_id.unique())[:5]}"
    return df


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD across seeds, and across folds where folds exist."""
    out = []
    for (dataset, model), g in runs.groupby(["dataset", "model"]):
        rec = {"dataset": dataset, "model": model,
               "label": g["label"].iloc[0], "group": g["group"].iloc[0],
               "n_runs": len(g), "n_seeds": g["seed"].nunique(),
               "n_folds": int(g["fold"].nunique()) if g["fold"].notna().any() else 0,
               "params_total": int(g["params_total"].iloc[0]),
               "params_trainable": int(g["params_trainable"].iloc[0]),
               "mean_best_epoch": float(g["best_epoch"].mean()),
               "sd_best_epoch": float(g["best_epoch"].std(ddof=1)) if len(g) > 1 else 0.0}
        for m in METRICS:
            v = g[f"test_{m}"].astype(float)
            rec[f"test_{m}_mean"] = float(v.mean())
            rec[f"test_{m}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            if len(v) > 1:
                ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(),
                                      scale=stats.sem(v)) if v.std(ddof=1) > 0 else (v.mean(), v.mean())
                rec[f"test_{m}_ci_low"], rec[f"test_{m}_ci_high"] = float(ci[0]), float(ci[1])
            rec[f"val_{m}_mean"] = float(g[f"val_{m}"].astype(float).mean())
        out.append(rec)
    return pd.DataFrame(out).sort_values(["dataset", "test_dice_mean"], ascending=[True, False])


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    pvals = [1.0 if (p is None or p != p) else p for p in pvals]   # audit S6: NaN is never significant
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def paired_tests(runs: pd.DataFrame, metric: str = "dice") -> pd.DataFrame:
    col = f"test_{metric}"
    records = []
    for dataset, g in runs.groupby("dataset"):
        key = ["seed"] if g["fold"].isna().all() else ["fold", "seed"]
        prop = g[g.model == PROPOSED].set_index(key)[col].sort_index()
        if prop.empty:
            continue
        for model, gm in g.groupby("model"):
            if model == PROPOSED:
                continue
            family = "ablation" if str(gm["group"].iloc[0]).startswith("ablation") else "comparator"
            base = gm.set_index(key)[col].sort_index()
            common = prop.index.intersection(base.index)
            if len(common) < 2:
                continue
            a = prop.loc[common].to_numpy(float)
            b = base.loc[common].to_numpy(float)
            d = a - b

            t_stat, t_p = stats.ttest_rel(a, b)
            try:
                w_stat, w_p = stats.wilcoxon(a, b)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            sd = d.std(ddof=1)
            cohen = float(d.mean() / sd) if sd > 0 else np.nan
            ci = (stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
                  if sd > 0 else (d.mean(), d.mean()))

            records.append({
                "dataset": dataset, "family": family, "comparison": f"{PROPOSED} vs {model}",
                "baseline": model, "baseline_label": MODELS.get(model, {}).get("label", model),
                "metric": metric, "n_pairs": len(d),
                "proposed_mean": float(a.mean()), "baseline_mean": float(b.mean()),
                "mean_difference": float(d.mean()), "sd_difference": float(sd),
                "ci95_low": float(ci[0]), "ci95_high": float(ci[1]),
                "cohens_d_paired": cohen,
                "t_statistic": float(t_stat), "t_p_value": float(t_p),
                "wilcoxon_statistic": float(w_stat) if w_stat == w_stat else None,
                "wilcoxon_p_value": float(w_p) if w_p == w_p else None,
                "wilcoxon_p_floor": float(2 ** -(len(d) - 1)) if len(d) > 1 else None,
            })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    parts = []
    for (dataset, family), g in df.groupby(["dataset", "family"]):   # audit S4: Holm within family
        g = g.copy()
        g["t_p_holm"] = holm(g["t_p_value"].tolist())
        wp = g["wilcoxon_p_value"].fillna(1.0).tolist()
        g["wilcoxon_p_holm"] = holm(wp)
        g["significant_t_holm_0.05"] = g["t_p_holm"] < 0.05
        g["significant_wilcoxon_holm_0.05"] = g["wilcoxon_p_holm"] < 0.05
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def bootstrap_per_image(runs: pd.DataFrame, metric: str = "dice") -> pd.DataFrame:
    """Paired bootstrap over per-image test scores, pooled across matched runs."""
    records = []
    for dataset, g in runs.groupby("dataset"):
        key = ["seed"] if g["fold"].isna().all() else ["fold", "seed"]

        def per_image(model):
            frames = []
            for _, r in g[g.model == model].iterrows():
                p = Path(r["dir"]) / "per_image_test.csv"
                if not p.is_file():
                    continue
                d = pd.read_csv(p)[["filename", metric]].copy()
                d["unit"] = "|".join(str(r[k]) for k in key)
                frames.append(d)
            return pd.concat(frames) if frames else None

        prop = per_image(PROPOSED)
        if prop is None:
            continue
        for model in g["model"].unique():
            if model == PROPOSED:
                continue
            base = per_image(model)
            if base is None:
                continue
            # audit S5: seeds share the same test images, so average over matched runs per image
            # first and bootstrap images, instead of treating image x seed rows as independent.
            prop_i = prop.groupby("filename")[metric].mean().reset_index()
            base_i = base.groupby("filename")[metric].mean().reset_index()
            merged = prop_i.merge(base_i, on="filename", suffixes=("_p", "_b"))
            if merged.empty:
                continue
            d = (merged[f"{metric}_p"] - merged[f"{metric}_b"]).to_numpy(float)
            idx = RNG.integers(0, len(d), size=(N_BOOTSTRAP, len(d)))
            means = d[idx].mean(axis=1)
            lo, hi = np.percentile(means, [2.5, 97.5])
            records.append({
                "dataset": dataset, "comparison": f"{PROPOSED} vs {model}",
                "baseline": model, "metric": metric, "n_paired_images": len(d),
                "mean_difference": float(d.mean()),
                "boot_ci95_low": float(lo), "boot_ci95_high": float(hi),
                "excludes_zero": bool(lo > 0 or hi < 0),
                "n_bootstrap": N_BOOTSTRAP,
                "win_rate": float((d > 0).mean()),
            })
    return pd.DataFrame(records)


def per_organ_breakdown(runs: pd.DataFrame) -> pd.DataFrame:
    """Test Dice per organ, which is what a multi-tissue claim actually needs."""
    frames = []
    for _, r in runs.iterrows():
        p = Path(r["dir"]) / "per_image_test.csv"
        if not p.is_file():
            continue
        d = pd.read_csv(p)
        if "organ" not in d.columns:
            continue
        d = d.assign(model=r["model"], dataset=r["dataset"], seed=r["seed"], fold=r["fold"])
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    allp = pd.concat(frames)
    return (allp.groupby(["dataset", "model", "organ"])["dice"]
            .agg(mean="mean", sd="std", n="size").reset_index()
            .sort_values(["dataset", "model", "organ"]))


def fold_averaged_tests(runs: pd.DataFrame, metric: str = "dice") -> pd.DataFrame:
    """Conservative analysis for fold-structured data (audit S5 / FINDINGS_LOG E.3).

    Seeds within a fold share the same test images, so (fold, seed) pairs are not independent.
    Here each model is first averaged over seeds WITHIN each fold, and the paired test runs over
    folds only (n = number of folds, df = n-1). Power is low by construction; it is reported
    next to the 15-pair test, not instead of it.
    """
    col = f"test_{metric}"
    rows = []
    for dataset, g in runs.groupby("dataset"):
        if g["fold"].isna().all():
            continue
        per = g.groupby(["model", "fold"])[col].mean().unstack("fold")
        if PROPOSED not in per.index:
            continue
        a = per.loc[PROPOSED]
        for model in per.index:
            if model == PROPOSED:
                continue
            b = per.loc[model]
            ok = a.notna() & b.notna()
            d = (a[ok] - b[ok]).to_numpy(float)
            if len(d) < 2:
                continue
            sd = d.std(ddof=1)
            t_stat, t_p = stats.ttest_rel(a[ok], b[ok]) if sd > 0 else (np.nan, np.nan)
            rows.append({"dataset": dataset, "baseline": model, "n_folds": len(d),
                         "mean_difference": float(d.mean()),
                         "per_fold_differences": "; ".join(f"{x:+.4f}" for x in d),
                         "folds_proposed_wins": int((d > 0).sum()),
                         "t_p_value_folds": float(t_p) if t_p == t_p else None,
                         "note": "n=%d folds, df=%d: low power; compare with the 15-pair test" % (len(d), len(d) - 1)})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = collect_runs()
    runs.to_csv(OUT / "all_runs.csv", index=False)
    print(f"Collected {len(runs)} completed runs "
          f"({runs['dataset'].nunique()} datasets, {runs['model'].nunique()} models)")

    agg = aggregate(runs)
    agg.to_csv(OUT / "aggregated_by_model.csv", index=False)
    print(f"  wrote aggregated_by_model.csv  ({len(agg)} rows)")

    tests = paired_tests(runs)
    if not tests.empty:
        tests.to_csv(OUT / "paired_significance_tests.csv", index=False)
        print(f"  wrote paired_significance_tests.csv  ({len(tests)} comparisons)")

    fa = fold_averaged_tests(runs)
    if not fa.empty:
        fa.to_csv(OUT / "fold_averaged_tests.csv", index=False)
        print(f"  wrote fold_averaged_tests.csv  ({len(fa)} comparisons)")

    boot = bootstrap_per_image(runs)
    if not boot.empty:
        boot.to_csv(OUT / "bootstrap_per_image.csv", index=False)
        print(f"  wrote bootstrap_per_image.csv  ({len(boot)} comparisons)")

    organ = per_organ_breakdown(runs)
    if not organ.empty:
        organ.to_csv(OUT / "per_organ_test_dice.csv", index=False)
        print(f"  wrote per_organ_test_dice.csv  ({len(organ)} rows)")

    # Headline summary, printed so it is visible without opening a file
    print("\n" + "=" * 92)
    print("TEST-SPLIT DICE, mean +/- SD across matched runs")
    print("=" * 92)
    for dataset, g in agg.groupby("dataset"):
        print(f"\n{dataset}")
        print(f"  {'model':<36}{'runs':>5}{'test Dice':>12}{'SD':>9}{'95% CI':>20}")
        for _, r in g.iterrows():
            lo = r.get("test_dice_ci_low")
            hi = r.get("test_dice_ci_high")
            ci = f"[{lo:.4f}, {hi:.4f}]" if pd.notna(lo) else "n/a"
            print(f"  {r['label'][:35]:<36}{r['n_runs']:>5}{r['test_dice_mean']:>12.4f}"
                  f"{r['test_dice_sd']:>9.4f}{ci:>20}")

    if not tests.empty:
        print("\n" + "=" * 92)
        print("PAIRED COMPARISONS AGAINST EPCS-NET (test Dice, Holm-corrected)")
        print("=" * 92)
        for dataset, g in tests.groupby("dataset"):
            print(f"\n{dataset}")
            print(f"  {'baseline':<24}{'n':>3}{'diff':>9}{'95% CI':>20}"
                  f"{'t p(Holm)':>12}{'W p(Holm)':>12}{'sig':>6}")
            for _, r in g.sort_values("mean_difference", ascending=False).iterrows():
                sig = "yes" if r["significant_t_holm_0.05"] else "no"
                print(f"  {r['baseline']:<24}{r['n_pairs']:>3}{r['mean_difference']:>9.4f}"
                      f"  [{r['ci95_low']:>7.4f},{r['ci95_high']:>7.4f}]"
                      f"{r['t_p_holm']:>12.4f}{r['wilcoxon_p_holm']:>12.4f}{sig:>6}")
            floor = g["wilcoxon_p_floor"].dropna()
            if not floor.empty:
                print(f"  NOTE: with {int(g['n_pairs'].iloc[0])} pairs the Wilcoxon p-value "
                      f"cannot go below {floor.iloc[0]:.4f}, so it cannot reach 0.05.")

    json.dump({"n_runs": int(len(runs)),
               "datasets": sorted(runs["dataset"].unique().tolist()),
               "models": sorted(runs["model"].unique().tolist())},
              (OUT / "stats_manifest.json").open("w", encoding="utf-8"), indent=2)
    print(f"\nAll outputs in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
