"""Regenerate revision-3 result figures from test-split data (rerunnable, CPU only, read-only on results).
Outputs: 05_manuscript/figures_r3/*.png (300 dpi) + *.svg
Old-name mapping:
  r3_headline_comparison  <- fig12_headline_comparison
  r3_seed_fold_stability  <- fig10_seed_stability
  r3_training_curves      <- fig11_training_curves
  r3_qual_cryonuseg       <- fig5_cryo_results ; r3_qual_nuinsseg <- fig6_nuin_results
  r3_ablation_5seed       (new ablation chart) ; r3_per_organ_{cryonuseg,nuinsseg} (new)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
import json, traceback
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "04_results"
OUT = ROOT / "05_manuscript" / "figures_r3"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
# Okabe-Ito, fixed per model across all figures
COL = {"epcs_net": "#D55E00", "unet": "#0072B2", "unetpp": "#56B4E9", "unet3plus": "#009E73",
       "eunet3plus": "#CC79A7", "sam_unet": "#E69F00", "medsam_zero": "#7F7F7F", "medsam_oracle": "#222222"}
NAME = {"epcs_net": "EPCS-Net", "unet": "U-Net", "unetpp": "U-Net++", "unet3plus": "U-Net3+",
        "eunet3plus": "eU-Net3+", "sam_unet": "SAM-guided U-Net", "medsam_zero": "MedSAM zero-shot",
        "medsam_oracle": "MedSAM (GT boxes)"}
MODELS = ["epcs_net", "unet", "unetpp", "unet3plus", "eunet3plus", "sam_unet"]
DSET = {"cryonuseg": "CryoNuSeg", "nuinsseg": "NuInsSeg"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def runs(group_dir):
    """completed run dirs (summary.json + per_image_test.csv); skip segments starting with '_'."""
    if not group_dir.is_dir():
        return []
    return [d for d in sorted(group_dir.iterdir())
            if d.is_dir() and not d.name.startswith("_")
            and (d / "summary.json").exists() and (d / "per_image_test.csv").exists()]


def run_info(d):
    n = d.name
    fold = int(n.split("_")[0][4:]) if n.startswith("fold") else 0
    return fold, int(n.split("seed")[-1])


def load_runs(ds, model, base=None):
    rows = []
    for d in runs(base or (RES / ds / ds / model)):
        pi = pd.read_csv(d / "per_image_test.csv")
        fold, seed = run_info(d)
        rows.append(dict(dir=d, fold=fold, seed=seed, dice=pi["dice"].mean(), pi=pi))
    return rows


def medsam(mode, ds):
    p = RES / "medsam_r3" / "summary_all.json"
    if not p.exists():
        return []
    return [s["metric_mean"]["dice"] for s in json.load(open(p))
            if not s.get("dry_run") and s["mode"] == mode and s["dataset"].startswith(ds)]


def msd(v):
    v = np.asarray(v, float)
    if not len(v):
        return np.nan, np.nan
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else np.nan)


def skip(name, why):
    print(f"SKIP {name}: {why}")


def pending(ax, title, msg="PENDING\nruns not yet complete"):
    ax.text(.5, .5, msg, ha="center", va="center", fontsize=13, color="#555", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
    for s in ax.spines.values():
        s.set_linestyle("--"); s.set_visible(True)


def fig_headline():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, ds in zip(axes, ["cryonuseg", "nuinsseg"]):
        data = {m: [r["dice"] for r in load_runs(ds, m)] for m in MODELS}
        data = {m: v for m, v in data.items() if v}
        if not data:
            pending(ax, DSET[ds], "PENDING\nNuInsSeg runs not yet complete")
            continue
        rank = sorted(data, key=lambda m: -np.mean(data[m]))
        rows = [(m, data[m]) for m in rank]
        extra = [(k, v) for k, v in [("medsam_zero", medsam("zero_shot_wholebox", ds)),
                                     ("medsam_oracle", medsam("oracle_boxes", ds))] if v]
        n = len(rows) + len(extra)
        ys = list(range(n))[::-1]
        lo = 1.0
        for (m, v), y in zip(rows + extra, ys):
            c = COL[m]; mu, sd = msd(v); lo = min(lo, min(v))
            ax.scatter(v, [y] * len(v), s=18, color=c, alpha=.45, zorder=2)
            ax.errorbar(mu, y, xerr=None if np.isnan(sd) else sd, fmt="o", ms=8, color=c, capsize=4, lw=2, zorder=3)
            lab = f"{mu:.3f}" + (f" ± {sd:.3f}" if not np.isnan(sd) else "") + f" (n={len(v)})"
            ax.annotate(lab, (max(max(v), mu + (0 if np.isnan(sd) else sd)), y), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9.5)
        ax.set_yticks(ys); ax.set_yticklabels([NAME[m] for m, _ in rows + extra])
        for t, (m, _) in zip(ax.get_yticklabels(), rows + extra):
            if m == "epcs_net":
                t.set_fontweight("bold")
        if extra:
            yd = len(extra) - .5
            ax.axhline(yd, color="#888", lw=1, ls="--")
            ax.text(0.0, yd + .08, "reference only, not ranked", transform=ax.get_yaxis_transform(),
                    va="bottom", fontsize=8.5, color="#555")
        ax.set_xlabel("Test Dice (mean ± SD over runs; dots = runs)")
        ax.set_title(DSET[ds]); ax.set_xlim(max(0, lo - .06), 1.12)
        ax.xaxis.grid(True, color="#e5e5e5"); ax.set_axisbelow(True)
    fig.tight_layout(); save(fig, "r3_headline_comparison")


def fig_stability():
    cr = load_runs("cryonuseg", "epcs_net"); nu = load_runs("nuinsseg", "epcs_net")
    if not cr and not nu:
        return skip("stability", "no EPCS-Net runs")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    seeds = sorted({r["seed"] for r in cr + nu})
    sc = dict(zip(seeds, ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#222222"]))
    ax = axes[0]
    if cr:
        folds = sorted({r["fold"] for r in cr})
        for f in folds:
            rs = sorted([r for r in cr if r["fold"] == f], key=lambda r: r["seed"])
            xs = f + np.linspace(-.2, .2, len(rs)) if len(rs) > 1 else [f]
            for x, r in zip(xs, rs):
                ax.scatter(x, r["dice"], s=70, color=sc[r["seed"]], edgecolor="k", zorder=3, label=f"seed {r['seed']}")
            mu, sd = msd([r["dice"] for r in rs])
            ax.hlines(mu, f - .32, f + .32, color="k", lw=2)
            ax.text(f + .34, mu, f"{mu:.3f}" + (f"\n±{sd:.3f}" if not np.isnan(sd) else ""), va="center", fontsize=9)
        ax.set_xticks(folds); ax.set_xticklabels([f"fold {f}" for f in folds]); ax.set_xlim(.5, max(folds) + .95)
        h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h))
        ax.legend(u.values(), u.keys(), fontsize=8.5, ncol=3, loc="upper left", frameon=False)
    ax.set_title("CryoNuSeg: 3 folds x 5 seeds, test (bar = fold mean)"); ax.set_ylabel("EPCS-Net test Dice")
    ax = axes[1]
    if nu:
        nu = sorted(nu, key=lambda r: r["seed"])
        for i, r in enumerate(nu):
            ax.scatter(i, r["dice"], s=70, color=sc[r["seed"]], edgecolor="k", zorder=3)
            ax.annotate(f"{r['dice']:.3f}", (i, r["dice"]), xytext=(8, 0), textcoords="offset points", fontsize=9, va="center")
        ax.axhline(np.mean([r["dice"] for r in nu]), color="k", lw=1.5)
        ax.set_xticks(range(len(nu))); ax.set_xticklabels([f"seed {r['seed']}" for r in nu]); ax.set_xlim(-.5, len(nu) - .5 + .4)
        ax.set_title(f"NuInsSeg: {len(nu)} seeds, one fixed test split")
    else:
        pending(ax, "NuInsSeg", "PENDING\nNuInsSeg runs not yet complete")
    fig.tight_layout(); save(fig, "r3_seed_fold_stability")


def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9)); any_ = False
    for ax, ds in zip(axes, ["cryonuseg", "nuinsseg"]):
        drawn = False
        for m in MODELS:
            curves, best = [], []
            for d in runs(RES / ds / ds / m):
                curves.append(pd.read_csv(d / "epoch_history.csv")["validation_dice"].values)
                best.append(json.load(open(d / "summary.json"))["best_epoch"])
            if not curves:
                continue
            drawn = any_ = True
            L = max(map(len, curves)); A = np.full((len(curves), L), np.nan)
            for i, c in enumerate(curves):
                A[i, :len(c)] = c
            mu = np.nanmean(A, 0); ep = np.arange(1, L + 1)
            ax.plot(ep, mu, color=COL[m], lw=2.8 if m == "epcs_net" else 1.5, zorder=3 if m == "epcs_net" else 2)
            if len(curves) > 1:
                sd = np.nanstd(A, 0, ddof=1)
                ax.fill_between(ep, mu - sd, mu + sd, color=COL[m], alpha=.10, lw=0)
            be = min(max(int(round(np.mean(best))), 1), L)
            ax.scatter(be, mu[be - 1], marker="*", s=140, color=COL[m], edgecolor="k", zorder=4)
        if drawn:
            ax.set_xlim(1, ax.get_xlim()[1] * 1.0)
            ax.set_title(f"{DSET[ds]} (n = {len(curves)} runs per model)")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Dice"); ax.set_ylim(0, 1)
            ax.yaxis.grid(True, color="#e5e5e5")
        else:
            pending(ax, DSET[ds])
    if not any_:
        plt.close(fig); return skip("curves", "no epoch histories")
    # direct labels at right end, de-overlapped, on the first populated panel(s)
    for ax, ds in zip(axes, ["cryonuseg", "nuinsseg"]):
        ends = []
        for m in MODELS:
            cs = [pd.read_csv(d / "epoch_history.csv")["validation_dice"].values for d in runs(RES / ds / ds / m)]
            if cs:
                L = max(map(len, cs)); A = np.full((len(cs), L), np.nan)
                for i, c in enumerate(cs):
                    A[i, :len(c)] = c
                ends.append([m, L, np.nanmean(A, 0)[-1]])
        ends.sort(key=lambda e: e[2])
        ys_ = [e[2] for e in ends]
        for i in range(1, len(ys_)):
            ys_[i] = max(ys_[i], ys_[i - 1] + .05)
        for e, yv in zip(ends, ys_):
            ax.annotate(NAME[e[0]], (e[1], e[2]), xytext=(e[1] + 1.0, yv), textcoords="data",
                        color=COL[e[0]], va="center", fontsize=9.5, fontweight="bold" if e[0] == "epcs_net" else None,
                        annotation_clip=False)
    fig.suptitle("Validation Dice per epoch: mean ± SD over runs; star = mean best epoch", fontsize=10.5, y=1.0)
    fig.tight_layout(); fig.subplots_adjust(right=.9); save(fig, "r3_training_curves")


def fig_qual(ds):
    runs_ = load_runs(ds, "epcs_net")
    if not runs_:
        return skip(f"qual_{ds}", "no EPCS-Net runs")
    r = next((x for x in runs_ if x["seed"] == 42 and x["fold"] in (0, 1)), runs_[0])
    tp = r["dir"] / "test_predictions"
    if not tp.is_dir():
        return skip(f"qual_{ds}", "no test_predictions")
    sp = RES / "splits" / (f"cryonuseg_R3_fold{r['fold']}.csv" if ds == "cryonuseg" else "nuinsseg_R3_stratified_70_15_15.csv")
    split = pd.read_csv(sp).set_index("filename")
    pi = r["pi"].sort_values("dice").reset_index(drop=True)
    pick = [("High", pi.iloc[-1]), ("Median", pi.iloc[len(pi) // 2]), ("Low", pi.iloc[0])]
    fig, axes = plt.subplots(3, 4, figsize=(11.5, 8.8))
    for i, (tag, row) in enumerate(pick):
        fn = row["filename"]; s = split.loc[fn]
        pp = tp / (os.path.splitext(fn)[0] + ".png")
        try:
            im = np.array(Image.open(s["image_path"]).convert("RGB").resize((256, 256), Image.BILINEAR))
            gt = np.array(Image.open(s["mask_path"]).convert("L").resize((256, 256), Image.NEAREST)) > 127
            pr = np.array(Image.open(pp).convert("L")) > 127
        except Exception as e:
            plt.close(fig); return skip(f"qual_{ds}", f"missing file ({e})")
        ov = im.astype(float)
        for m_, c in [(pr & gt, (0, 158, 115)), (pr & ~gt, (213, 94, 0)), (~pr & gt, (0, 114, 178))]:
            ov[m_] = .4 * ov[m_] + .6 * np.array(c)
        for ax, a, t in zip(axes[i], [im, gt, pr, ov.astype(np.uint8)],
                            ["Image", "Ground truth", f"EPCS-Net, Dice {row['dice']:.3f}", "Overlay (TP/FP/FN)"]):
            ax.imshow(a, cmap="gray"); ax.axis("off")
            if i == 0 and ax is not axes[i][2]:
                ax.set_title(t, fontsize=10.5)
        axes[i][2].set_title(f"{'EPCS-Net, ' if i == 0 else ''}Dice {row['dice']:.3f}", fontsize=10.5)
        axes[i][0].text(-.05, .5, f"{tag}: {row['organ']}", transform=axes[i][0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=10)
    fig.legend(handles=[Patch(color="#009E73", label="true positive"), Patch(color="#D55E00", label="false positive"),
                        Patch(color="#0072B2", label="false negative")], loc="lower center", ncol=3, frameon=False, fontsize=10)
    ft = f"fold {r['fold']}, " if r["fold"] else ""
    fig.suptitle(f"{DSET[ds]} test split, EPCS-Net ({ft}seed {r['seed']}); cases picked by per-image Dice", fontsize=11)
    fig.tight_layout(rect=(.03, .04, 1, .97)); save(fig, f"r3_qual_{ds}")


def fig_ablation():
    base = RES / "ablation" / "cryonuseg"
    order = ["abl0_nosam_epcsbackbone", "abl1_nosam", "abl2_bottleneck", "abl3_pspi", "abl4_bdsg", "abl5_dksr",
             "abl6_ftas", "ablA_acfb_nopspi", "ablB_random_embed", "ablC_pspi_concat", "ablD_gn_encoder"]
    lab = {}
    try:
        a = pd.read_csv(RES / "analysis" / "all_runs.csv"); lab = dict(a[["model", "label"]].drop_duplicates().values)
    except Exception:
        pass
    rows = []
    for m in order:
        rs = runs(base / m)
        if not rs:
            continue
        v = [pd.read_csv(d / "per_image_test.csv")["dice"].mean() for d in rs]
        loss = json.load(open(rs[0] / "summary.json")).get("loss", "")
        rows.append((m, v, "FTAS" if "focal" in loss.lower() or "FTAS" in loss else loss.split(" (")[0]))
    if not rows:
        return skip("ablation", "no completed ablation runs")
    fig, ax = plt.subplots(figsize=(10.5, 6)); n = len(rows)
    losses = sorted({r[2] for r in rows}); mk = dict(zip(losses, ["o", "s", "^", "D"]))
    for i, (m, v, loss) in enumerate(rows):
        y = n - 1 - i; mu, sd = msd(v); c = "#D55E00" if m == "abl6_ftas" else "#0072B2"
        ax.scatter(v, [y] * len(v), s=20, color=c, alpha=.45, marker=mk[loss])
        ax.errorbar(mu, y, xerr=None if np.isnan(sd) else sd, fmt=mk[loss], ms=9, color=c, capsize=4, lw=2, mec="k")
        ax.annotate(f"{mu:.3f}" + (f" ± {sd:.3f}" if not np.isnan(sd) else "") + f" (n={len(v)})",
                    (max(max(v), mu + (0 if np.isnan(sd) else sd)), y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9.5)
    ax.set_yticks(range(n)); ax.set_yticklabels([lab.get(m, m) for m, _, _ in rows][::-1])
    k = sum(1 for m, _, _ in rows if m.startswith("ablA") or m.startswith("ablB") or m.startswith("ablC") or m.startswith("ablD"))
    if k:
        ax.axhline(k - .5, color="#888", ls="--", lw=1)
        ax.text(0.0, k - .5 + .08, "controls A-D", transform=ax.get_yaxis_transform(), va="bottom", fontsize=8.5, color="#555")
    ax.set_xlabel("CryoNuSeg fold 1 test Dice (mean ± SD over seeds; dots = seeds)")
    ax.legend([Line2D([], [], marker=mk[l], ls="", color="#444", mec="k") for l in losses], [f"loss: {l}" for l in losses],
              loc="lower left", frameon=False, fontsize=9)
    ax.xaxis.grid(True, color="#e5e5e5"); ax.set_axisbelow(True); ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + .1)
    fig.tight_layout(); save(fig, "r3_ablation_5seed")


def fig_organ(ds):
    per = {}
    for m in ["epcs_net", "unet", "sam_unet"]:
        rs = load_runs(ds, m)
        if rs:
            per[m] = pd.concat([r["pi"].groupby("organ")["dice"].mean().rename(i) for i, r in enumerate(rs)], axis=1)
    if "epcs_net" not in per:
        return skip(f"organ_{ds}", "no EPCS-Net runs")
    organs = per["epcs_net"].mean(axis=1).sort_values().index.tolist(); n = len(organs)
    fig, ax = plt.subplots(figsize=(8, max(4.2, .34 * n + 1.5)))
    off = {"epcs_net": 0.0, "unet": .24, "sam_unet": -.24}
    for m, df in per.items():
        mu = df.mean(axis=1).reindex(organs).values
        sd = df.std(axis=1, ddof=1).reindex(organs).values if df.shape[1] > 1 else None
        ax.errorbar(mu, np.arange(n) + off[m], xerr=sd, fmt="o", ms=6 if m == "epcs_net" else 4.5, color=COL[m], capsize=2.5,
                    lw=1.4, label=f"{NAME[m]} (n={df.shape[1]})", zorder=3 if m == "epcs_net" else 2)
    ax.set_yticks(range(n)); ax.set_yticklabels(organs); ax.set_ylim(-.7, n - .3)
    ax.set_xlabel("Test Dice per organ (mean ± SD over runs)"); ax.xaxis.grid(True, color="#e5e5e5")
    ax.legend(frameon=False, fontsize=9, loc="lower right"); ax.set_title(DSET[ds])
    fig.tight_layout(); save(fig, f"r3_per_organ_{ds}")


if __name__ == "__main__":
    jobs = [("headline", fig_headline), ("stability", fig_stability), ("curves", fig_curves),
            ("qual_cryo", lambda: fig_qual("cryonuseg")), ("qual_nuin", lambda: fig_qual("nuinsseg")),
            ("ablation", fig_ablation), ("organ_cryo", lambda: fig_organ("cryonuseg")),
            ("organ_nuin", lambda: fig_organ("nuinsseg"))]
    for n, f in jobs:
        try:
            f()
        except Exception as e:
            print(f"FAILED {n}: {e}"); traceback.print_exc(limit=3)
