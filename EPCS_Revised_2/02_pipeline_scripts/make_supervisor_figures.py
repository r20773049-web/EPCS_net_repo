r"""Supervisor-style figures (CPU only, real data only).
  r3_qual_cryonuseg.png / r3_qual_nuinsseg.png : old-paper Fig. 5/6 style qualitative panels
  r3_gradcam_table_{nuinsseg,cryonuseg}.png    : Grad-CAM table figures, cells CROPPED from the saved composites
All outputs: PNG, RGB, white, 300 dpi, exactly 2000 px wide."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
try:
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:
        pass
import json, textwrap
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "04_results"
AN = RES / "analysis"
OUT = ROOT / "05_manuscript" / "figures_r3"
ORIG = ROOT / "05_manuscript" / "_full_resolution_originals"
OUT.mkdir(parents=True, exist_ok=True)
W = 2000 / 300.0
plt.rcParams.update({"font.family": "DejaVu Sans"})


def canvas(h):
    fig = plt.figure(figsize=(W, h), dpi=300, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(h, 0); ax.axis("off")
    return fig, ax


def put(ax, arr, x, y, w, h, gray=False):
    ax.imshow(arr, extent=(x, x + w, y + h, y), cmap="gray" if gray else None, interpolation="lanczos", zorder=2)
    ax.set_xlim(0, W); ax.set_ylim(ax.figure.get_figheight(), 0)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="black", lw=0.8, zorder=3))


def save(fig, name):
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)
    im = Image.open(p).convert("RGB")
    im.save(p, dpi=(300, 300))
    print("wrote", name, im.size)


# ---------------------------------------------------------------- qualitative
def load_run(ds):
    base = RES / ds / ds / "epcs_net"
    rs = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and not d.name.startswith("_") and (d / "per_image_test.csv").exists():
            fold = int(d.name.split("_")[0][4:]) if d.name.startswith("fold") else 0
            rs.append(dict(dir=d, fold=fold, seed=int(d.name.split("seed")[-1]), pi=pd.read_csv(d / "per_image_test.csv")))
    return next(x for x in rs if x["seed"] == 42 and x["fold"] in (0, 1))


def fig_qual(ds):
    r = load_run(ds)
    tp = r["dir"] / "test_predictions"
    sp = RES / "splits" / (f"cryonuseg_R3_fold{r['fold']}.csv" if ds == "cryonuseg" else "nuinsseg_R3_stratified_70_15_15.csv")
    split = pd.read_csv(sp).set_index("filename")
    pi = r["pi"].sort_values("dice").reset_index(drop=True)
    pick = [("High", pi.iloc[-1]), ("Median", pi.iloc[len(pi) // 2]), ("Low", pi.iloc[0])]
    H = 6.55
    fig, ax = canvas(H)
    ax.add_patch(Rectangle((0.03, 0.03), W - 0.06, H - 0.06, fill=False, ec="black", lw=1.4))
    cell, gap, x0 = 1.42, 0.07, 0.52
    xs = [x0 + i * (cell + gap) for i in range(4)]
    heads = ["Input (H&E)", "Ground truth", "Prediction", "Overlay (TP / FP / FN)"]
    for x, t in zip(xs, heads):
        ax.text(x + cell / 2, 0.30, t, ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.plot([0.03, W - 0.03], [0.52, 0.52], color="black", lw=0.8)
    for i, (tag, row) in enumerate(pick):
        fn = row["filename"]; s = split.loc[fn]
        im = np.array(Image.open(s["image_path"]).convert("RGB").resize((256, 256), Image.BILINEAR))
        gt = np.array(Image.open(s["mask_path"]).convert("L").resize((256, 256), Image.NEAREST)) > 127
        pr = np.array(Image.open(tp / (os.path.splitext(fn)[0] + ".png")).convert("L")) > 127
        ov = im.astype(float)
        for m_, c in [(pr & gt, (0, 158, 115)), (pr & ~gt, (213, 94, 0)), (~pr & gt, (0, 114, 178))]:
            ov[m_] = .4 * ov[m_] + .6 * np.array(c)
        y = 0.95 + i * 1.78
        for j, (a, g) in enumerate([(im, False), (gt.astype(float), True), (pr.astype(float), True), (ov.astype(np.uint8), False)]):
            put(ax, a, xs[j], y, cell, cell, gray=g)
        ax.text(xs[2] + cell / 2, y - 0.1, f"Dice = {row['dice']:.3f}", ha="center", va="center", fontsize=9)
        ax.text(0.28, y + cell / 2, tag, rotation=90, ha="center", va="center", fontsize=10, fontweight="bold")
    ly = H - 0.30
    items = [("#009E73", "True positive (TP)"), ("#D55E00", "False positive (FP)"), ("#0072B2", "False negative (FN)")]
    x = 0.55
    for c, t in items:
        ax.add_patch(Rectangle((x, ly - 0.08), 0.16, 0.16, fc=c, ec="none"))
        ax.text(x + 0.22, ly, t, va="center", fontsize=8.8)
        x += 2.0
    save(fig, f"r3_qual_{ds}")


# ---------------------------------------------------------------- Grad-CAM table
def cells_from_composite(path):
    im = np.array(Image.open(path).convert("RGB"))
    nw = (im.min(2) < 250)
    cols = np.where(nw[300:].mean(0) > 0.3)[0]
    x0, x1 = cols.min(), cols.max() + 1
    rows = nw[:, x0:x1].mean(1) > 0.5
    d = np.diff(rows.astype(int)); st = np.where(d == 1)[0] + 1; en = np.where(d == -1)[0] + 1
    assert len(st) == 7 and len(en) == 7, (st, en)
    cw = (x1 - x0) / 4.0
    out = []
    for a, b in zip(st, en):
        row = []
        for j in range(4):
            xa, xb = int(round(x0 + j * cw)) + 3, int(round(x0 + (j + 1) * cw)) - 3
            row.append(Image.fromarray(im[a + 3:b - 3, xa:xb]))
        out.append(row)
    return out


def ell(lines, n):
    if len(lines) <= n:
        return lines
    return lines[:n - 1] + [lines[n - 1][:17].rstrip() + "..."]


def fig_table(ds, tag):
    comp = ORIG / f"fig13_gradcam_best_and_failure_cases_{ds}.png"
    rep = json.load(open(AN / f"gradcam_report_{tag}.json"))
    rows = [("Highest overlap", c) for c in rep["best_cases"]] + [("Lowest overlap", c) for c in rep["worst_cases"]]
    assert len(rows) == 7
    csv = pd.read_csv(AN / f"gradcam_per_image_{tag}.csv").set_index("filename")
    for _, c in rows:   # cross-check json vs csv
        assert abs(csv.loc[c["filename"], "gradcam_iou_overlap"] - c["gradcam_iou_overlap"]) < 1e-9
    cells = cells_from_composite(comp)
    th = 0.9; pad = 0.04
    hh, bh = 0.52, 0.24
    rh = th + 2 * pad
    H = hh + 2 * bh + 7 * rh + 0.05
    fig, ax = canvas(H)
    cx = [0.0, 1.38]  # case col start/end
    xs = [1.38 + i * (th + 0.02) for i in range(4)]
    nx = [xs[3] + th + 0.04]
    nw_ = (W - 0.02 - nx[0]) / 3
    ncols = [nx[0] + k * nw_ for k in range(3)]
    ax.add_patch(Rectangle((0.01, 0.01), W - 0.02, H - 0.02, fill=False, ec="black", lw=1.2))
    ax.add_patch(Rectangle((0.01, 0.01), W - 0.02, hh, fc="#e8e8e8", ec="black", lw=0.8))
    ax.text(0.08, hh / 2, "Group / Case", fontsize=8, fontweight="bold", va="center")
    for x, t in zip(xs, ["Input\n(H&E)", "Ground\ntruth", "Prediction", "Grad-CAM\noverlay"]):
        ax.text(x + th / 2, hh / 2, t, fontsize=8, fontweight="bold", ha="center", va="center")
    for x, t in zip(ncols, ["CAM\nIoU", "Dice", "GT fg\n(%)"]):
        ax.text(x + nw_ / 2, hh / 2, t, fontsize=8, fontweight="bold", ha="center", va="center")
    y = hh
    last = None
    n = 0
    for grp, c in rows:
        if grp != last:
            ax.add_patch(Rectangle((0.01, y), W - 0.02, bh, fc="#f4f4f4", ec="black", lw=0.8))
            ax.text(0.08, y + bh / 2, grp, fontsize=8, fontweight="bold", va="center")
            y += bh; last = grp
        else:
            ax.plot([0.01, W - 0.01], [y, y], color="#bbbbbb", lw=0.5)
        fn = c["filename"]
        organ = c["organ"]
        stem = os.path.splitext(fn.split("__")[-1])[0]
        l1 = ell(textwrap.wrap(organ, 19, break_long_words=False), 3)
        l2 = ell(textwrap.wrap(stem, 19, break_long_words=False), 2)
        ytxt = y + rh / 2
        txt1 = "\n".join(l1); txt2 = "\n".join(l2)
        ax.text(0.08, ytxt - 0.02 * len(l2) - 0.0, txt1, fontsize=7.0, fontweight="bold", va="bottom", linespacing=1.15)
        ax.text(0.08, ytxt + 0.04, txt2, fontsize=7.0, va="top", color="#333", linespacing=1.15)
        for j in range(4):
            im = cells[n][j].resize((300, 300), Image.LANCZOS)
            put(ax, np.array(im), xs[j], y + pad, th, th)
        for x, v in zip(ncols, [f"{c['gradcam_iou_overlap']:.3f}", f"{c['dice']:.3f}", f"{c['gt_foreground_fraction'] * 100:.1f}"]):
            ax.text(x + nw_ / 2, ytxt, v, fontsize=8.5, ha="center", va="center")
        y += rh; n += 1
    ax.plot([0.01, W - 0.01], [hh + bh + 3 * rh, hh + bh + 3 * rh], color="black", lw=0.6)
    for x in [1.35, xs[3] + th + 0.02] + [ncols[1], ncols[2]]:
        ax.plot([x, x], [0.01, H - 0.01], color="#999999", lw=0.4, zorder=1)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    print(ds, "table height in at 6.5in width:", H * 6.5 / W)
    save(fig, f"r3_gradcam_table_{ds}")


if __name__ == "__main__":
    fig_qual("cryonuseg"); fig_qual("nuinsseg")
    fig_table("nuinsseg", "epcs_net__nuinsseg__seed2024")
    fig_table("cryonuseg", "epcs_net__cryonuseg__fold3__seed2024")
