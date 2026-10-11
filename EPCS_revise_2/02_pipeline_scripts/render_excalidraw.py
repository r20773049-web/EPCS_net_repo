"""Render the Excalidraw figure sources to PNG (300 dpi, white) and SVG with matplotlib.

Supported: rectangles (rounded, Excalidraw adaptive radius), ellipses (badges, junction dots),
text (free and bound to a container, bold via customData.bold), elbow arrows with solid
triangular heads.  Usage: python render_excalidraw.py   (run from anywhere)"""
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Polygon
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "01_excalidraw_figures"
OUT = ROOT / "05_manuscript" / "figures"
MAP = {
    "Figure_01_EPCS_Net_Methodology_Pipeline": "fig1_pipeline",
    "Figure_02_PSPI_Pyramid_Scale_Prior_Injection": "fig2_pspi",
    "Figure_03_ACFB_Adaptive_Crossstream_Fusion_Block": "fig3_acfb",
    "Figure_04_FSFD_Full_Scale_Fused_Decoder": "fig4_fsfd",
}
PAD = 14
PX = 0.75           # pt per px (96 px/in)
HEAD_L, HEAD_HW = 13.0, 5.8
for f in ("arial.ttf", "arialbd.ttf", "seguisym.ttf"):
    p = Path("C:/Windows/Fonts") / f
    if p.exists():
        fm.fontManager.addfont(str(p))
FAMILY = ["Arial", "Segoe UI Symbol", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "path"
plt.rcParams["font.family"] = FAMILY


def render(path, name):
    els = [e for e in json.load(open(path, encoding="utf8"))["elements"] if not e.get("isDeleted")]
    xs, ys = [], []
    for e in els:
        if e["type"] in ("arrow", "line"):
            for px, py in e["points"]:
                xs.append(e["x"] + px); ys.append(e["y"] + py)
        else:
            xs += [e["x"], e["x"] + e["width"]]; ys += [e["y"], e["y"] + e["height"]]
    x0, x1, y0, y1 = min(xs) - PAD, max(xs) + PAD, min(ys) - PAD, max(ys) + PAD
    W, H = x1 - x0, y1 - y0
    fig = plt.figure(figsize=(W / 96, H / 96), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
    byid = {e["id"]: e for e in els}
    z = 1
    for e in els:
        t = e["type"]
        lw = e.get("strokeWidth", 2) * PX
        z += 1
        if t == "rectangle":
            rd = e.get("roundness")
            r = min(0.25 * min(e["width"], e["height"]), (rd or {}).get("value", 32)) if rd else 0
            bg = e["backgroundColor"]
            ax.add_patch(FancyBboxPatch((e["x"] + r, e["y"] + r), e["width"] - 2 * r, e["height"] - 2 * r,
                         boxstyle=f"round,pad={r},rounding_size={r}" if r else "square,pad=0",
                         fc="none" if bg == "transparent" else bg, ec=e["strokeColor"], lw=lw,
                         ls="--" if e["strokeStyle"] == "dashed" else "-", zorder=z, joinstyle="round"))
        elif t == "ellipse":
            bg = e["backgroundColor"]
            ax.add_patch(Ellipse((e["x"] + e["width"] / 2, e["y"] + e["height"] / 2), e["width"], e["height"],
                                 fc="none" if bg == "transparent" else bg, ec=e["strokeColor"], lw=lw, zorder=z))
        elif t in ("arrow", "line"):
            pts = [(e["x"] + a, e["y"] + b) for a, b in e["points"]]
            col = e["strokeColor"]
            head = e.get("endArrowhead")
            if head:   # shorten the shaft so the (filled) head tip lands exactly on the target
                (px_, py_), (qx, qy) = pts[-1], pts[-2]
                ang = math.atan2(py_ - qy, px_ - qx)
                base = (px_ - (HEAD_L - 1.5) * math.cos(ang), py_ - (HEAD_L - 1.5) * math.sin(ang))
                shaft = pts[:-1] + [base]
            else:
                shaft = pts
            ax.plot(*zip(*shaft), color=col, lw=lw, zorder=z + 20, solid_capstyle="butt", solid_joinstyle="miter")
            if head:
                (px_, py_), (qx, qy) = pts[-1], pts[-2]
                ang = math.atan2(py_ - qy, px_ - qx)
                ux, uy = math.cos(ang), math.sin(ang)
                nx, ny = -uy, ux
                bx, by = px_ - HEAD_L * ux, py_ - HEAD_L * uy
                ax.add_patch(Polygon([(px_, py_), (bx + HEAD_HW * nx, by + HEAD_HW * ny), (bx - HEAD_HW * nx, by - HEAD_HW * ny)],
                                     closed=True, fc=col, ec=col, lw=0.5, zorder=z + 20, joinstyle="miter"))
        elif t == "text":
            c = byid.get(e.get("containerId"))
            txt = e["text"].split("\n")
            fs = e["fontSize"]
            bold = (e.get("customData") or {}).get("bold", False)
            lh = fs * e.get("lineHeight", 1.25)
            if c:
                cx, ytop = c["x"] + c["width"] / 2, c["y"] + c["height"] / 2 - lh * len(txt) / 2
                ha, xx = "center", cx
            else:
                ha = e.get("textAlign", "left")
                xx = e["x"] if ha == "left" else e["x"] + e["width"] / 2 if ha == "center" else e["x"] + e["width"]
                ytop = e["y"]
            for i, line in enumerate(txt):
                yc = ytop + lh * (i + 0.5)
                ax.text(xx, yc + 0.36 * fs, line, ha=ha, va="baseline", fontsize=fs * PX, family=FAMILY,
                        fontweight="bold" if bold else "normal", color=e["strokeColor"], zorder=100)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(OUT / f"{name}.svg", facecolor="white")
    plt.close(fig)
    print(name, int(W), int(H), "ratio %.2f" % (W / H))


if __name__ == "__main__":
    for stem, name in MAP.items():
        render(SRC / f"{stem}.excalidraw", name)
