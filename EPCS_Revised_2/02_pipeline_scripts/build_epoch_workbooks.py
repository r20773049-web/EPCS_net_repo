"""Build epoch-history and best-epoch workbooks. Read-only on results; rerunnable."""
import os, json, glob
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "04_results"))
OUT = os.path.join(RES, "epoch_workbooks")
SOURCES = ["cryonuseg", "nuinsseg", "ablation", "reference21"]
MET = ["accuracy", "precision", "recall", "specificity", "dice", "iou"]

def find_runs():
    runs = []
    for src in SOURCES:
        base = os.path.join(RES, src)
        for p in glob.glob(os.path.join(base, "**", "summary.json"), recursive=True):
            rel = os.path.relpath(p, RES).replace("\\", "/").split("/")
            if any(s.startswith("_") for s in rel): continue
            runs.append(p)
    return sorted(runs)

def sheetname(m, used):
    n = m[:31]; i = 1
    while n.lower() in used:
        suf = f"~{i}"; n = m[:31 - len(suf)] + suf; i += 1
    used.add(n.lower()); return n

def main():
    os.makedirs(OUT, exist_ok=True)
    per_ds = {}; issues = []; cols_seen = set()
    for p in find_runs():
        s = json.load(open(p)); d = os.path.dirname(p)
        eh = os.path.join(d, "epoch_history.csv")
        if not os.path.exists(eh):
            issues.append(f"MISSING epoch_history: {d}"); continue
        h = pd.read_csv(eh); cols_seen.update(h.columns)
        if len(h) != s.get("epochs"):
            issues.append(f"TRUNCATED/!= epochs: {s['run_id']} rows={len(h)} epochs={s.get('epochs')}")
        be = s["best_epoch"]
        h.insert(0, "seed", s["seed"]); h.insert(0, "fold", s["fold"] if s.get("fold") is not None else 0)
        h.insert(0, "dataset", s["dataset"]); h.insert(0, "model", s["model"])
        h["is_best_epoch"] = h["epoch"] == be
        if h["is_best_epoch"].sum() != 1: issues.append(f"best epoch not found: {s['run_id']}")
        row = h[h["is_best_epoch"]]
        r = dict(model=s["model"], model_label=s.get("model_label"), group=s.get("group"),
                 dataset=s["dataset"], fold=(s["fold"] if s.get("fold") is not None else 0), seed=s["seed"], best_epoch=be,
                 epochs_run=len(h), peaked_early_1to3=be <= 3, peaked_last_epoch=be == h["epoch"].max(),
                 summary_best_validation_dice=s.get("best_validation_dice"))
        for c in h.columns:
            if c.startswith("validation_"):
                r["val_hist_" + c[11:]] = row[c].iloc[0] if len(row) else np.nan
        for m in MET:
            r["test_" + m] = s["test_mean"][m]
        r["params_total"] = s.get("parameters_total"); r["params_trainable"] = s.get("parameters_trainable")
        r["wall_seconds"] = s.get("wall_seconds")
        for k in ("validation_mean", "best_validation_mean"):
            if isinstance(s.get(k), dict):
                for m, v in s[k].items(): r[f"val_summary_{m}"] = v
        D = per_ds.setdefault(s["dataset"], {"hist": {}, "best": []})
        D["hist"].setdefault(s["model"], []).append(h); D["best"].append(r)

    for ds, D in per_ds.items():
        used = set()
        with pd.ExcelWriter(os.path.join(OUT, f"EPCS_R3_EpochHistories_{ds}.xlsx")) as w:
            for m in sorted(D["hist"]):
                df = pd.concat(D["hist"][m]).sort_values(["fold", "seed", "epoch"])
                df.to_excel(w, sheet_name=sheetname(m, used), index=False)
        B = pd.DataFrame(D["best"]).sort_values(["model", "fold", "seed"])
        num = ["best_epoch"] + [c for c in B.columns if c.startswith(("val_hist_", "test_"))] + ["params_total", "wall_seconds"]
        rows = []
        for m, g in B.groupby("model"):
            r = {"model": m, "n_runs": len(g)}
            for c in num:
                v = pd.to_numeric(g[c], errors="coerce")
                r.update({f"{c}_mean": v.mean(), f"{c}_sd": v.std(ddof=1), f"{c}_min": v.min(), f"{c}_max": v.max()})
            notes = []
            e = g[g.peaked_early_1to3]; l = g[g.peaked_last_epoch]
            if len(e): notes.append("peaked at epoch 1-3 (instability): " + ", ".join(f"f{a}s{b}" for a, b in zip(e.fold, e.seed)))
            if len(l): notes.append("peaked at last epoch (non-convergence): " + ", ".join(f"f{a}s{b}" for a, b in zip(l.fold, l.seed)))
            r["note"] = "; ".join(notes); rows.append(r)
        S = pd.DataFrame(rows)
        ps = []
        for (m, f), g in B.groupby(["model", "fold"]):
            for val, tag in (("test_dice", "test_dice"), ("best_epoch", "best_epoch")):
                pv = g.pivot_table(index="model", columns="seed", values=val)
                pv.insert(0, "metric", tag); pv.insert(1, "fold", f); ps.append(pv.reset_index())
        P = pd.concat(ps).sort_values(["metric", "fold", "model"], ascending=[False, True, True])
        F = B.groupby(["model", "fold"])[["test_" + m for m in MET]].mean().reset_index()
        readme = pd.DataFrame({"item": [
            "Selection rule", "BestEpoch_PerRun", "val_hist_*", "val_summary_*", "test_*", "BestEpoch_Summary",
            "PerSeed", "PerFold", "Epoch histories"], "description": [
            "Checkpoint chosen on validation Dice only (best_epoch = argmax validation dice); the test set is never used for selection.",
            "One row per run (model, fold, seed): best_epoch, validation metrics at best epoch from epoch_history.csv (val_hist_*), summary.json best validation dice, TEST metrics (summary.json test_mean, mean over test images), params, wall seconds.",
            "Validation metrics of the best-epoch row of epoch_history.csv.",
            "Validation values stored in summary.json, if present.",
            "Test metrics at the selected checkpoint (accuracy, precision, recall, specificity, dice, iou).",
            "Per model over runs: mean/SD(ddof=1)/min/max of best epoch and each metric; note flags runs peaking at epoch 1-3 (instability) or at the last epoch (non-convergence).",
            "Model x seed matrix of test Dice and of best epoch, per fold.",
            "Model x fold mean over seeds of the six test metrics.",
            "Separate workbook; one sheet per model, every epoch of every run, columns model, dataset, fold, seed, epoch, all per-epoch values, is_best_epoch."]})
        with pd.ExcelWriter(os.path.join(OUT, f"EPCS_R3_BestEpochs_{ds}.xlsx")) as w:
            readme.to_excel(w, sheet_name="Readme", index=False)
            B.to_excel(w, sheet_name="BestEpoch_PerRun", index=False)
            S.to_excel(w, sheet_name="BestEpoch_Summary", index=False)
            P.to_excel(w, sheet_name="PerSeed", index=False)
            F.to_excel(w, sheet_name="PerFold", index=False)
        print(ds, "runs", len(B), "models", len(D["hist"]))
    print("columns:", sorted(cols_seen)); print("issues:", issues or "none")

if __name__ == "__main__":
    main()
