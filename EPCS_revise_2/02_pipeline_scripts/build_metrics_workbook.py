"""Rerunnable: builds 04_results/EPCS_R3_Full_Metrics.xlsx from summary.json / per_image_test.csv /
analysis csvs / medsam_r3/summary_all.json. READ-ONLY on results. Cheap (pandas/openpyxl/scipy only)."""
import json, datetime
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

R = Path(__file__).resolve().parents[1] / "04_results"
OUT = R / "EPCS_R3_Full_Metrics.xlsx"
M = ["accuracy", "precision", "recall", "specificity", "dice", "iou"]
SKIP_DIRS = {"analysis", "paper_tables", "splits", "logs", "sam_cache", "medsam_r3"}


def ok(p):
    rel = p.relative_to(R).parts
    return not any(s.startswith("_") for s in rel) and rel[0] not in SKIP_DIRS


rows, missing = [], []
for f in sorted(R.rglob("summary.json")):
    if not ok(f):
        continue
    d = json.load(open(f))
    rel = f.parent.relative_to(R)
    for blk in ("test_mean", "validation_mean"):
        miss = [m for m in M if m not in d.get(blk, {})]
        if miss:
            missing.append((str(rel), blk, miss))
    r = dict(run_dir=str(rel).replace("\\", "/"), group=d.get("group"), model=d["model"],
             model_label=d.get("model_label"), dataset=d["dataset"], fold=d.get("fold"),
             seed=d.get("seed"), best_epoch=d.get("best_epoch"), n_train=d.get("n_train"),
             n_validation=d.get("n_validation"), n_test=d.get("n_test"))
    for m in M:
        r["test_" + m] = d.get("test_mean", {}).get(m, np.nan)
    for m in M:
        r["test_sd_img_" + m] = d.get("test_sd_across_images", {}).get(m, np.nan)
    for m in M:
        r["val_" + m] = d.get("validation_mean", {}).get(m, np.nan)
    r.update(params_total=d.get("parameters_total"), params_trainable=d.get("parameters_trainable"),
             wall_seconds=d.get("wall_seconds"), peak_vram_mib=d.get("cuda_peak_allocated_mib"))
    rows.append(r)
perrun = pd.DataFrame(rows)


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan
    sd = x.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return sd, x.mean() - h, x.mean() + h


def bymodel(df):
    out = []
    for (grp, model), g in df.groupby(["group", "model"], sort=False):
        o = dict(group=grp, model=model, model_label=g.model_label.iloc[0], n_runs=len(g),
                 n_seeds=g.seed.nunique(), n_folds=g.fold.nunique())
        for m in M:
            sd, lo, hi = ci(g["test_" + m])
            o.update({f"test_{m}_mean": g["test_" + m].mean(), f"test_{m}_sd": sd,
                      f"test_{m}_ci95_low": lo, f"test_{m}_ci95_high": hi})
        for m in M:
            o[f"val_{m}_mean"] = g["val_" + m].mean()
        o.update(best_epoch_mean=g.best_epoch.mean(), best_epoch_sd=g.best_epoch.std(ddof=1),
                 params_total=g.params_total.iloc[0], params_trainable=g.params_trainable.iloc[0],
                 wall_seconds_mean=g.wall_seconds.mean(), peak_vram_mib_max=g.peak_vram_mib.max())
        out.append(o)
    return pd.DataFrame(out)


org_rows = []
for r in rows:
    p = R / r["run_dir"] / "per_image_test.csv"
    if not p.exists():
        continue
    t = pd.read_csv(p)
    col = next((c for c in ("organ", "tissue") if c in t.columns), None)
    if col is None:
        continue
    for organ, g in t.groupby(col):
        o = dict(dataset=r["dataset"], group=r["group"], model=r["model"], fold=r["fold"],
                 seed=r["seed"], organ=organ, n_images=len(g))
        o.update({m: g[m].mean() for m in M})
        org_rows.append(o)
org = pd.DataFrame(org_rows)
if len(org):
    agg = {"n_runs": ("seed", "size"), "n_images_per_run_mean": ("n_images", "mean")}
    for m in M:
        agg[f"{m}_mean"] = (m, "mean")
    for m in M:
        agg[f"{m}_sd_across_runs"] = (m, lambda x: x.std(ddof=1))
    org_s = org.groupby(["dataset", "group", "model", "organ"], sort=False).agg(**agg).reset_index()
else:
    org_s = pd.DataFrame()

today = datetime.date.today().isoformat()
with pd.ExcelWriter(OUT, engine="openpyxl") as W:
    readme = [
        ["EPCS-Net Round 3 full metrics workbook", f"Built {today} by 06_scripts/build_metrics_workbook.py (rerunnable; read-only on results)."],
        ["Metrics", "All six: accuracy, precision, recall, specificity, dice, iou. Numbers shown to 4 decimals."],
        ["Split", "test_* = held-out TEST images of the run's split (headline numbers). val_* = VALIDATION split, which is also the checkpoint-selection set (best epoch chosen on validation Dice), so val_* is optimistic and must not be reported as independent."],
        ["Aggregation", "Step 1 (inside a run, from summary.json): per-image metric, then mean over that run's test images. Step 2 (across runs): mean, sample SD (ddof=1) and 95% t-CI (df=n_runs-1) over matched runs (fold x seed) of a model. Empty SD/CI means n_runs=1 (e.g. single-seed ablations). The CI treats runs as independent, which is only approximate: seeds within a fold share the same test images."],
        ["ByModel_<dataset>", "One sheet per dataset: per model n runs/seeds/folds, test mean/SD/CI95 for all six metrics, validation means, best epoch mean+SD, params, mean wall seconds, max peak VRAM (MiB). Includes baselines, reference-21 and ablation models (see 'group')."],
        ["PerRun", "Every completed run (summary.json present, no underscore path segment): all test and validation metrics plus SD across test images."],
        ["PerFold_cryonuseg", "Mean over seeds within each fold, per model, all six test metrics."],
        ["PerOrgan", "From per_image_test.csv (organ column): per run, then summarized per model and organ (mean and SD across runs), all six metrics. CryoNuSeg has about one image per organ per fold, so SDs mostly reflect seeds."],
        ["Paired", "Copies of analysis csvs with section labels: (A) paired_significance_tests = paired t and Wilcoxon over matched (fold,seed) runs, Holm-corrected. WARNING: seeds within a fold share identical test images, so the pairs are not independent and p-values are anti-conservative. (B) fold_averaged_tests = seeds averaged within fold then tested over folds (n=3, df=2, low power). (C) bootstrap_per_image = per-image paired bootstrap CI (images as resampling units)."],
        ["MedSAM", "zero_shot_wholebox = round-2 protocol sanity check only (task mismatch). oracle_boxes = ORACLE UPPER BOUND using ground-truth-derived boxes; never a competitor to EPCS-Net. Dry-run entries excluded."],
        ["Excluded", "Any run dir with a path segment starting with underscore (_cpu_smoketests, _archive_*, _dry_run)."],
    ]
    pd.DataFrame(readme, columns=["Item", "Description"]).to_excel(W, sheet_name="Readme", index=False)
    for ds, g in perrun.groupby("dataset"):
        bymodel(g).to_excel(W, sheet_name=f"ByModel_{ds}"[:31], index=False)
    perrun.to_excel(W, sheet_name="PerRun", index=False)
    c = perrun[perrun.dataset == "cryonuseg"]
    if len(c):
        gb = c.groupby(["group", "model", "fold"], sort=False)
        pf = gb[["test_" + m for m in M]].mean().reset_index()
        pf["n_seeds"] = gb.seed.nunique().values
        pf.to_excel(W, sheet_name="PerFold_cryonuseg", index=False)
    (org_s if len(org_s) else pd.DataFrame({"note": ["no organ column found"]})).to_excel(W, sheet_name="PerOrgan", index=False)
    A = R / "analysis"
    r0 = 0
    sec = [("(A) paired_significance_tests.csv: paired t + Wilcoxon over matched (fold,seed) runs, Holm-corrected. WARNING: seeds within a fold share test images; pairs are not independent.", "paired_significance_tests.csv"),
           ("(B) fold_averaged_tests.csv: seeds averaged within fold, then paired t over folds (n=3, df=2, low power).", "fold_averaged_tests.csv"),
           ("(C) bootstrap_per_image.csv: per-image paired bootstrap (images as resampling units).", "bootstrap_per_image.csv")]
    for lab, fn in sec:
        pd.DataFrame({"": [lab]}).to_excel(W, sheet_name="Paired", startrow=r0, index=False, header=False)
        if (A / fn).exists():
            df = pd.read_csv(A / fn)
            df.to_excel(W, sheet_name="Paired", startrow=r0 + 1, index=False)
            r0 += len(df) + 4
        else:
            r0 += 3
    ms = []
    mp = R / "medsam_r3" / "summary_all.json"
    if mp.exists():
        for e in json.load(open(mp)):
            if e.get("dry_run"):
                continue
            lab = ("SANITY CHECK (zero-shot whole-image box; task mismatch)" if e["mode"] == "zero_shot_wholebox"
                   else "ORACLE UPPER BOUND (GT-derived boxes) - not a competitor")
            o = dict(mode=e["mode"], interpretation=lab, dataset=e["dataset"],
                     n_test_images=e["n_test_images"], split=e.get("split_used"))
            o.update({f"{m}_mean": e["metric_mean"].get(m) for m in M})
            o.update({f"{m}_sd": e["metric_sample_sd"].get(m) for m in M})
            ms.append(o)
    pd.DataFrame(ms).to_excel(W, sheet_name="MedSAM", index=False)
    for ws in W.book.worksheets:
        for row in ws.iter_rows():
            for c_ in row:
                if isinstance(c_.value, float):
                    c_.number_format = "0.0000"
        for i, col in enumerate(ws.columns):
            ws.column_dimensions[col[0].column_letter].width = 110 if (ws.title == "Readme" and i == 1) else 16
    W.book.move_sheet("Readme", offset=-(len(W.book.worksheets) - 1))

print("runs:", len(perrun), perrun.groupby("dataset").size().to_dict())
print("summaries missing metrics:", missing if missing else "none")
