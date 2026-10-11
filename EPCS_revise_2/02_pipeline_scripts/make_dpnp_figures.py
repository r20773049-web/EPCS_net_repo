"""R3 exploratory-phenotyping figures (NuInsSeg test predictions, tag r3_nuinsseg).
Reads only outputs of 03_code/analysis/dpnp_r3.py. PC1-3 scores are read from the
components CSV; HDBSCAN labels are not stored by dpnp_r3.py, so panel labels are
recomputed deterministically (StandardScaler + PCA(40, rs=0) + HDBSCAN) from the
saved features and CHECKED against the report (n_clusters, outlier_pct, sizes).
Bootstrap per-replicate ARIs are not stored: only the summary is drawn."""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "04_results" / "analysis"
FIG = ROOT / "05_manuscript" / "figures_r3"
TAG = sys.argv[1] if len(sys.argv) > 1 else "r3_nuinsseg"
SUF = "_cryonuseg" if TAG == "r3_cryonuseg" else ""
OI = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]  # Okabe-Ito
GREY = "#9a9a9a"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})


def f(prefix, ext):
    return AN / f"dpnp_{prefix}_{TAG}.{ext}"


def need(*paths):
    miss = [p.name for p in paths if not p.is_file()]
    if miss:
        print(f"SKIP: missing inputs {miss}")
        return False
    return True


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_pca():
    comp, mass, corr, rep = f("components", "csv"), f("pca_loading_mass", "csv"), f("pca_feature_correlations", "csv"), f("report", "json")
    if not need(comp, mass, corr, rep):
        return
    d = pd.read_csv(comp, usecols=["PC1", "PC2", "PC3", "area"])
    m = pd.read_csv(mass)
    c = pd.read_csv(corr)
    r = json.loads(rep.read_text())
    ev = r["pca"]["explained_variance_pct"]
    rho = c[(c.component == "PC1") & (c.feature == "area")].spearman_rho.iloc[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    hb = a1.hexbin(d.PC1, d.PC2, gridsize=55, bins="log", cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=a1, label="components per bin (log10)", pad=0.01)
    a1.set_xlabel(f"PC1 ({ev[0]:.1f}% variance)")
    a1.set_ylabel(f"PC2 ({ev[1]:.1f}% variance)")
    a1.set_title("A  PC1 vs PC2", loc="left", fontweight="bold")
    a1.text(*((0.02, 0.97) if SUF else (0.98, 0.97)), f"PC1 vs area: rho = {rho:+.2f}", transform=a1.transAxes, ha="left" if SUF else "right", va="top",
            fontsize=10.5, bbox=dict(fc="white", ec="0.7", alpha=0.9))
    x = np.arange(len(m))
    w = 0.38
    b1 = a2.bar(x - w / 2, m.loading_mass_interpretable_pct, w, color=OI[1], label="7 interpretable")
    b2 = a2.bar(x + w / 2, m.loading_mass_embedding_pct, w, color=OI[0], label="512 embedding")
    for b in list(b1) + list(b2):
        a2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.1f}", ha="center", fontsize=10)
    a2.set_xticks(x)
    a2.set_xticklabels(m.component)
    a2.set_ylim(0, 135)
    a2.set_ylabel("loading mass (%)")
    a2.set_title("B  Loading mass", loc="left", fontweight="bold")
    a2.legend(fontsize=10, loc="upper center", frameon=False)
    fig.tight_layout()
    save(fig, "r3_dpnp_pca" + SUF)


def recompute_labels(rep):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import HDBSCAN
    df = pd.read_csv(f("components", "csv"))
    cols = rep["feature_vector"]["interpretable"] + [f"resnet18_{i}" for i in range(rep["feature_vector"]["embedding_dims"])]
    X = StandardScaler().fit_transform(df[cols].to_numpy(float))
    sc = PCA(n_components=rep["pca"]["n_components"], random_state=0).fit_transform(X)
    out = {}
    for k in ("clustering_primary_prespecified", "clustering_sensitivity_fixed5"):
        s = rep[k]
        lab = HDBSCAN(min_cluster_size=s["min_cluster_size"], min_samples=s["min_samples"], metric=s["metric"]).fit_predict(sc)
        sizes = {int(c): int((lab == c).sum()) for c in np.unique(lab) if c != -1}
        ok = (len(sizes) == s["n_clusters"] and abs((lab == -1).mean() * 100 - s["outlier_pct"]) < 1e-6
              and sizes == {int(a): b for a, b in s["cluster_sizes"].items()})
        out[k] = (lab, ok)
        print(f"recomputed {k}: matches report = {ok}")
    return out


def fig_hdbscan():
    comp, rep = f("components", "csv"), f("report", "json")
    if not need(comp, rep):
        return
    r = json.loads(rep.read_text())
    d = pd.read_csv(comp, usecols=["PC1", "PC2"])
    try:
        lab = recompute_labels(r)
    except Exception as e:
        print(f"SKIP hdbscan: could not recompute labels ({e})")
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    ev = r["pca"]["explained_variance_pct"]
    for ax, key, title in ((axs[0], "clustering_primary_prespecified", "A  Prespecified rule"),
                           (axs[1], "clustering_sensitivity_fixed5", "B  min_cluster_size = 5")):
        s = r[key]
        l, ok = lab[key]
        if not ok:
            print(f"WARNING: recomputed labels for {key} do not match the report")
        noise = l == -1
        ax.scatter(d.PC1[noise], d.PC2[noise], s=4, c=GREY, alpha=0.5, lw=0, label="noise", rasterized=True)
        for i, cl in enumerate(sorted(set(l[~noise]))):
            mk = l == cl
            small = mk.sum() <= 50
            ax.scatter(d.PC1[mk], d.PC2[mk], s=40 if small else 7, c=OI[i % len(OI)],
                       marker="D" if small else "o", edgecolor="k" if small else "none", lw=0.5,
                       label=f"cluster {cl} (n={mk.sum()})", rasterized=True)
        sil = s["silhouette_geometric_separation"]
        txt = (f"clusters: {s['n_clusters']}\nnoise: {s['outlier_pct']:.1f}%\n" + (f"silhouette = {sil:.2f}" if sil is not None else "silhouette: n/a"))
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=11.5, va="bottom",
                bbox=dict(fc="white", ec="0.7", alpha=0.92))
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
        ax.legend(fontsize=10, loc="upper right", frameon=True)
    axs[0].set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    fig.tight_layout()
    save(fig, "r3_dpnp_hdbscan" + SUF)


def fig_stability():
    rep, conf = f("report", "json"), f("cluster_confounds", "csv")
    if not need(rep, conf):
        return
    r = json.loads(rep.read_text())
    s = r.get("sensitivity_stability")
    c = pd.read_csv(conf)
    if not s:
        print("SKIP stability: no stability block")
        return
    lab = {"organ": "organ", "species": "species", "filename": "source image"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    a1.axhline(0, color="0.7", lw=0.8)
    a1.errorbar([0], [s["ari_mean"]], yerr=[s["ari_sd"]], fmt="o", color=OI[0], capsize=7, ms=9, lw=2)
    a1.text(0.12, s["ari_mean"], f"mean = {s['ari_mean']:.2f}\nSD = {s['ari_sd']:.2f}", va="center", fontsize=12)
    a1.set_xlim(-0.5, 1.0)
    a1.set_xticks([])
    a1.set_ylim(-0.15, 1.1)
    a1.set_ylabel("adjusted Rand index")
    a1.set_title("A  Bootstrap stability", loc="left", fontweight="bold")
    names = [lab.get(k, k) for k in c["confound"]]
    cols = [OI[0]] * len(c)
    y = np.arange(len(c))
    a2.barh(y, c.cramers_v, color=cols)
    for i, row in c.iterrows():
        a2.text(row.cramers_v + 0.008, i, f"V = {row.cramers_v:.2f}", va="center", fontsize=11.5)
    a2.set_yticks(y)
    a2.set_yticklabels(names)
    a2.invert_yaxis()
    a2.set_xlim(0, 0.8)
    a2.set_xlabel("Cramér's V")
    a2.set_title("B  Cluster vs confounds", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "r3_dpnp_stability_confounds" + SUF)


if __name__ == "__main__":
    fig_pca()
    fig_hdbscan()
    fig_stability()
    print("tag:", TAG)
