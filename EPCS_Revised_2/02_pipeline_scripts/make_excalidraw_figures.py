"""
Generate the four EPCS-Net schematic figures as Excalidraw sources (v3 design).

Design rules (see 05_manuscript/figures/FIGURE_REDESIGN_NOTES.md)
  * every arrow is an ELBOW arrow made of horizontal / vertical segments only,
    bound to its boxes (startBinding / endBinding) so boxes can be moved in Excalidraw;
  * arrows never cross each other and never pass through a box or a label
    (checked by lint() at generation time; the script aborts on any violation);
  * fan-out / fan-in points get an automatic filled junction dot;
  * one palette for all figures, lane panels with a bold lane title,
    Helvetica (fontFamily 2), roughness 0, solid triangular arrowheads.

Content facts were taken from the implementation (METHODS_CODE_FIDELITY.md) and the
FSFD subsection of EPCS_revise3.tex.

Usage:  python make_excalidraw_figures.py [output_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import ImageFont

# ---------------------------------------------------------------- palette
INK = "#1e1e1e"
K = {  # fill, border, lane tint
    "sam": ("#ffc9c9", "#e03131", "#fff5f5"),
    "loc": ("#a5d8ff", "#1971c2", "#edf6ff"),
    "acfb": ("#ffd8a8", "#e8590c", "#fff4e6"),
    "fus": ("#b2f2bb", "#2f9e44", "#ebfbee"),
    "dec": ("#d0bfff", "#6741d9", "#f5f2ff"),
    "loss": ("#fff3bf", "#f08c00", "#fffbe6"),
    "neu": ("#f1f3f5", "#495057", "#f8f9fa"),
}
FONT = 2
LH = 1.25
NOW = 1760000000000
ARROW_W = 2
BOX_W = 2
HEAD_L = 13

_FONTS = {}


def _font(size, bold):
    key = (size, bold)
    if key not in _FONTS:
        f = "C:/Windows/Fonts/" + ("arialbd.ttf" if bold else "arial.ttf")
        _FONTS[key] = ImageFont.truetype(f, int(size))
    return _FONTS[key]


def text_size(text, size, bold=False):
    f = _font(size, bold)
    w = max(f.getlength(l) for l in text.split("\n"))
    return w, len(text.split("\n")) * size * LH


class Box:
    def __init__(self, id, x, y, w, h):
        self.id, self.x, self.y, self.w, self.h = id, x, y, w, h

    def p(self, side, f=0.5):
        if side == "t":
            return (self.x + self.w * f, self.y)
        if side == "b":
            return (self.x + self.w * f, self.y + self.h)
        if side == "l":
            return (self.x, self.y + self.h * f)
        return (self.x + self.w, self.y + self.h * f)

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2

    @property
    def r(self):
        return self.x + self.w

    @property
    def b(self):
        return self.y + self.h


class Card:
    """Header strip (bold title) + body, two rectangles that behave as one block."""

    def __init__(self, head, body):
        self.head, self.body = head, body
        self.x, self.y, self.w = head.x, head.y, head.w
        self.h = body.b - head.y

    cx = property(lambda s: s.x + s.w / 2)
    cy = property(lambda s: s.y + s.h / 2)
    r = property(lambda s: s.x + s.w)
    b = property(lambda s: s.y + s.h)


class Fig:
    def __init__(self, tag):
        self.tag = tag
        self.n = 0
        self.items = []          # (z, seq, element)
        self.byid = {}
        self.arrows = []         # dicts with pts, src, dst
        self.lanes = []
        self.free_texts = []     # (x, y, w, h, text)
        self.solids = []         # Box objects (non-lane rectangles)
        self.issues = []

    # ---- ids / raw elements
    def nid(self, p):
        self.n += 1
        return f"{self.tag}_{p}{self.n:03d}"

    def _base(self, id, typ, x, y, w, h):
        return {
            "id": id, "type": typ, "x": float(x), "y": float(y), "width": float(w), "height": float(h),
            "angle": 0, "strokeColor": INK, "backgroundColor": "transparent", "fillStyle": "solid",
            "strokeWidth": BOX_W, "strokeStyle": "solid", "roughness": 0, "opacity": 100,
            "groupIds": [], "frameId": None, "roundness": None, "seed": 1000 + self.n,
            "version": 1, "versionNonce": 7000 + self.n, "isDeleted": False, "boundElements": [],
            "updated": NOW, "link": None, "locked": False,
        }

    def add(self, el, z):
        self.items.append((z, len(self.items), el))
        self.byid[el["id"]] = el

    # ---- text
    def _text(self, text, size, bold, color, cont=None, x=0, y=0, align="center", z=1.0):
        w, h = text_size(text, size, bold)
        tid = self.nid("t")
        if cont is not None:
            x = cont.x + (cont.w - w) / 2
            y = cont.y + (cont.h - h) / 2
        el = self._base(tid, "text", x, y, w, h)
        el.update({
            "text": text, "originalText": text, "fontSize": size, "fontFamily": FONT,
            "textAlign": align, "verticalAlign": "middle" if cont is not None else "top",
            "containerId": cont.id if cont is not None else None, "lineHeight": LH,
            "baseline": int(size * 0.9), "autoResize": True, "strokeColor": color,
            "customData": {"bold": bool(bold)},
        })
        self.add(el, z)
        return el, w, h

    def label(self, x, y, text, size=18, bold=False, color=INK, align="left", w_fixed=None):
        """Free label. x is the left edge (align left), centre (center) or right edge (right)."""
        tw, th = text_size(text, size, bold)
        if align == "center":
            x0 = x - tw / 2
        elif align == "right":
            x0 = x - tw
        else:
            x0 = x
        el, w, h = self._text(text, size, bold, color, None, x0, y, align, z=3.5)
        self.free_texts.append((x0, y, tw, th, text))
        return Box(el["id"], x0, y, tw, th)

    # ---- shapes
    def lane(self, x, y, w, h, kind, title=None, title_size=20, title_color=None, title_dx=14):
        fill, edge, tint = K[kind]
        lid = self.nid("lane")
        el = self._base(lid, "rectangle", x, y, w, h)
        el.update({"backgroundColor": tint, "strokeColor": fill if kind != "neu" else "#ced4da",
                   "strokeWidth": 1, "roundness": {"type": 3, "value": 16}})
        self.add(el, 0)
        b = Box(lid, x, y, w, h)
        self.lanes.append(b)
        if title:
            self.label(x + title_dx, y + 8, title, size=title_size, bold=True, color=title_color or edge)
        return b

    def box(self, x, y, w, h, kind, text, size=20, bold=False, fill=None, edge=None, z=1.0, radius=12, tcolor=INK):
        f, e, _ = K[kind]
        rid = self.nid("r")
        el = self._base(rid, "rectangle", x, y, w, h)
        el.update({"backgroundColor": fill or f, "strokeColor": edge or e,
                   "roundness": {"type": 3, "value": radius}})
        self.add(el, z)
        b = Box(rid, x, y, w, h)
        if text:
            t, tw, th = self._text(text, size, bold, tcolor, b, z=z + 0.01)
            el["boundElements"].append({"type": "text", "id": t["id"]})
            if tw > w - 14 or th > h - 8:
                self.issues.append(f"TEXT DOES NOT FIT box '{text[:30]}' text {tw:.0f}x{th:.0f} box {w:.0f}x{h:.0f}")
        self.solids.append(b)
        return b

    def card(self, x, y, w, kind, title, body, h=None, tsize=20, bsize=18, hdr=38, body_fill="#ffffff"):
        n = len(body.split("\n"))
        bh = n * bsize * LH + 20
        if h is not None:
            bh = h - hdr
        head = self.box(x, y, w, hdr, kind, title, size=tsize, bold=True, radius=12)
        bod = self.box(x, y + hdr, w, bh, kind, body, size=bsize, fill=body_fill, radius=12)
        # square off the joint: thin overlay hides the double border and the inner round corners
        c = Card(head, bod)
        return c

    def badge(self, cx, cy, num, kind="acfb", d=40, size=22):
        f, e, _ = K[kind]
        bid = self.nid("b")
        el = self._base(bid, "ellipse", cx - d / 2, cy - d / 2, d, d)
        el.update({"backgroundColor": e, "strokeColor": e})
        self.add(el, 4)
        b = Box(bid, cx - d / 2, cy - d / 2, d, d)
        t, tw, th = self._text(str(num), size, True, "#ffffff", b, z=4.01)
        el["boundElements"].append({"type": "text", "id": t["id"]})
        return b

    # ---- arrows
    def _res(self, obj, side):
        if isinstance(obj, Card):
            if side == "t":
                return obj.head, obj.head.p("t")
            if side == "b":
                return obj.body, obj.body.p("b")
            r = obj.head if obj.cy < obj.head.b else obj.body
            return r, ((obj.x if side == "l" else obj.r), obj.cy)
        return obj, obj.p(side)

    def link(self, src, ss, dst, ds, way=None, ym=None, xm=None, sfrac=0.5, dfrac=0.5):
        sb, p0 = self._res(src, ss) if not (ss in "lr" and sfrac != 0.5) else (src, src.p(ss, sfrac))
        db, p1 = self._res(dst, ds) if not (ds in "lr" and dfrac != 0.5) else (dst, dst.p(ds, dfrac))
        if way is not None:
            pts = [p0] + list(way) + [p1]
        else:
            x0, y0 = p0
            x1, y1 = p1
            if ss in "bt" and ds in "bt":
                if abs(x0 - x1) < 0.5:
                    pts = [p0, p1]
                else:
                    ym = ym if ym is not None else (y0 + y1) / 2
                    pts = [p0, (x0, ym), (x1, ym), p1]
            elif ss in "lr" and ds in "lr":
                if abs(y0 - y1) < 0.5:
                    pts = [p0, p1]
                else:
                    xm = xm if xm is not None else (x0 + x1) / 2
                    pts = [p0, (xm, y0), (xm, y1), p1]
            elif ss in "bt" and ds in "lr":
                pts = [p0, (x0, y1), p1]
            else:
                pts = [p0, (x1, y0), p1]
        # clean: drop duplicate points
        clean = [pts[0]]
        for p in pts[1:]:
            if abs(p[0] - clean[-1][0]) > 0.01 or abs(p[1] - clean[-1][1]) > 0.01:
                clean.append(p)
        pts = clean
        for a, b in zip(pts, pts[1:]):
            if abs(a[0] - b[0]) > 0.01 and abs(a[1] - b[1]) > 0.01:
                self.issues.append(f"DIAGONAL segment {a}->{b}")
        aid = self.nid("a")
        x0, y0 = pts[0]
        rel = [[float(px - x0), float(py - y0)] for px, py in pts]
        xs = [r[0] for r in rel]
        ys = [r[1] for r in rel]
        el = self._base(aid, "arrow", x0, y0, max(xs) - min(xs), max(ys) - min(ys))

        def fp(box, p):
            return [round((p[0] - box.x) / box.w, 4), round((p[1] - box.y) / box.h, 4)]

        el.update({
            "strokeWidth": ARROW_W, "points": rel, "lastCommittedPoint": None,
            "startBinding": {"elementId": sb.id, "focus": 0, "gap": 0, "fixedPoint": fp(sb, p0)},
            "endBinding": {"elementId": db.id, "focus": 0, "gap": 0, "fixedPoint": fp(db, pts[-1])},
            "startArrowhead": None, "endArrowhead": "triangle", "elbowed": True,
            "fixedSegments": None, "startIsSpecial": None, "endIsSpecial": None,
        })
        self.add(el, 2)
        for bx in (sb, db):
            self.byid[bx.id]["boundElements"].append({"type": "arrow", "id": aid})
        self.arrows.append({"id": aid, "pts": pts, "src": sb.id, "dst": db.id})
        return el

    # ---- junction dots + lint
    def dots(self):
        # incident directions at each point
        inc = {}
        for a in self.arrows:
            pts = a["pts"]
            for i, p in enumerate(pts):
                key = (round(p[0], 1), round(p[1], 1))
                dirs = inc.setdefault(key, set())
                for j in (i - 1, i + 1):
                    if 0 <= j < len(pts):
                        q = pts[j]
                        dx, dy = q[0] - p[0], q[1] - p[1]
                        dirs.add(("h", 1 if dx > 0 else -1) if abs(dx) > abs(dy) else ("v", 1 if dy > 0 else -1))
            # interior pass-through points of other arrows
        for a in self.arrows:
            for p, q in zip(a["pts"], a["pts"][1:]):
                for key, dirs in list(inc.items()):
                    kx, ky = key
                    if abs(p[0] - q[0]) < 0.01 and abs(kx - p[0]) < 0.6 and min(p[1], q[1]) + 0.6 < ky < max(p[1], q[1]) - 0.6:
                        dirs.add(("v", 1)); dirs.add(("v", -1))
                    if abs(p[1] - q[1]) < 0.01 and abs(ky - p[1]) < 0.6 and min(p[0], q[0]) + 0.6 < kx < max(p[0], q[0]) - 0.6:
                        dirs.add(("h", 1)); dirs.add(("h", -1))
        for (kx, ky), dirs in inc.items():
            if len(dirs) >= 3:
                did = self.nid("j")
                el = self._base(did, "ellipse", kx - 5.5, ky - 5.5, 11, 11)
                el.update({"backgroundColor": INK, "strokeColor": INK, "strokeWidth": 1})
                self.add(el, 3)

    def lint(self):
        iss = self.issues
        segs = []
        for a in self.arrows:
            for p, q in zip(a["pts"], a["pts"][1:]):
                segs.append((a, p, q))

        def hits(p, q, r, m=2.0):  # segment vs rect interior
            x0, x1 = sorted((p[0], q[0]))
            y0, y1 = sorted((p[1], q[1]))
            return x1 > r[0] + m and x0 < r[0] + r[2] - m and y1 > r[1] + m and y0 < r[1] + r[3] - m

        for a, p, q in segs:
            for b in self.solids:
                if b.id in (a["src"], a["dst"]):
                    # still forbid running through its own body except at the ends
                    pass
                if hits(p, q, (b.x, b.y, b.w, b.h)):
                    iss.append(f"ARROW {a['id']} passes through box {b.id}")
            for (x, y, w, h, t) in self.free_texts:
                if hits(p, q, (x - 2, y - 1, w + 4, h + 2), m=0):
                    iss.append(f"ARROW {a['id']} touches text '{t[:25]}'")
        # crossings between different arrows (strict perpendicular interior intersections)
        for i, (a1, p1, q1) in enumerate(segs):
            for (a2, p2, q2) in segs[i + 1:]:
                if a1 is a2:
                    continue
                h1 = abs(p1[1] - q1[1]) < 0.01
                h2 = abs(p2[1] - q2[1]) < 0.01
                if h1 == h2:
                    continue
                hs, vs = ((p1, q1), (p2, q2)) if h1 else ((p2, q2), (p1, q1))
                y = hs[0][1]
                x = vs[0][0]
                hx0, hx1 = sorted((hs[0][0], hs[1][0]))
                vy0, vy1 = sorted((vs[0][1], vs[1][1]))
                if hx0 + 0.6 < x < hx1 - 0.6 and vy0 + 0.6 < y < vy1 - 0.6:
                    iss.append(f"CROSSING arrows {a1['id']} x {a2['id']} at ({x:.0f},{y:.0f})")
        # solid box overlaps
        for i, b in enumerate(self.solids):
            for c in self.solids[i + 1:]:
                if b.x < c.r - 1 and c.x < b.r - 1 and b.y < c.b - 1 and c.y < b.b - 1:
                    iss.append(f"OVERLAP boxes {b.id} {c.id}")
        # free text overlapping solids
        for (x, y, w, h, t) in self.free_texts:
            for b in self.solids:
                if x < b.r - 1 and b.x < x + w - 1 and y < b.b - 1 and b.y < y + h - 1:
                    iss.append(f"LABEL '{t[:25]}' overlaps box {b.id}")
        return iss

    # ---- output
    def bounds(self):
        xs, ys = [], []
        for _, _, e in self.items:
            if e["type"] == "arrow":
                for px, py in e["points"]:
                    xs.append(e["x"] + px); ys.append(e["y"] + py)
            else:
                xs += [e["x"], e["x"] + e["width"]]; ys += [e["y"], e["y"] + e["height"]]
        return min(xs), min(ys), max(xs), max(ys)

    def write(self, path: Path, label: str):
        self.dots()
        iss = self.lint()
        els = [e for _, _, e in sorted(self.items, key=lambda t: (t[0], t[1]))]
        ids = [e["id"] for e in els]
        assert len(ids) == len(set(ids)), "duplicate ids"
        # normalise origin so the drawing starts at (0, 0)
        x0, y0, x1, y1 = self.bounds()
        for e in els:
            e["x"] = round(e["x"] - x0, 2)
            e["y"] = round(e["y"] - y0, 2)
        doc = {"type": "excalidraw", "version": 2,
               "source": "EPCS-Net revision 3 -- make_excalidraw_figures.py (v3 design)",
               "elements": els,
               "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"}, "files": {}}
        path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print(f"  wrote {path.name} ({len(els)} elements, {x1 - x0:.0f} x {y1 - y0:.0f} px, ratio {(x1 - x0) / (y1 - y0):.2f}) {label}")
        for i in iss:
            print("    LINT:", i)
        return iss


# ===========================================================================
# Figure 3 -- ACFB
# ===========================================================================
def figure_03(out):
    f = Fig("f3")
    # input lanes
    ly, lh = 0, 118
    lw = 350
    sam_l = f.lane(0, ly, lw, lh, "sam", "SAM branch")
    loc_l = f.lane(lw + 20, ly, lw, lh, "loc", "Local branch")
    b_s = f.box(18, ly + 42, lw - 36, 64, "sam", "SAM prior F_S\n256 ch, H_k x W_k".replace("x", "×"), size=21)
    b_x = f.box(lw + 20 + 18, ly + 42, lw - 36, 64, "loc", "Local features X_L\nC_k ch, H_k × W_k", size=21)
    W = 2 * lw + 20
    cx = W / 2
    # ACFB lane with cards
    ay = ly + lh + 52
    cw = 600
    cxl = (W - cw) / 2
    steps = [
        ("Channel projection",
         "1×1 conv brings both streams to C_k channels\nX' = W_x X_L ,   S' = W_s F_S"),
        ("Bidirectional dual-stream gating (BDSG)",
         "g_s = σ(W_gs S') ,   g_x = σ(W_gx X')\nX_cross = X' ⊙ g_s + S' ⊙ g_x\neach stream gates the other"),
        ("Dual-kernel refinement (DKSR)",
         "X_ms = Conv3×3(X_cross) + Conv5×5(X_cross)\n3×3: boundary detail,  5×5: wider context"),
        ("Squeeze-and-excitation attention",
         "s = σ(Conv(h→C_k)(ReLU(Conv(C_k→h)(GAP(X_ms)))))\nh = max(C_k / 4, 8)\nX_att = X_ms ⊙ s"),
        ("Residual + GroupNorm(8)",
         "Φ_k = GroupNorm(W_out X_att + X_ms)\nresidual on the post-DKSR tensor X_ms"),
    ]
    lane_top = ay
    y = ay + 46
    cards = []
    for i, (t, b) in enumerate(steps):
        c = f.card(cxl, y, cw, "acfb", t, b, tsize=21, bsize=20)
        cards.append(c)
        y = c.b + 32
    lane_bot = y - 32 + 18
    acfb_l = f.lane(0, lane_top, W, lane_bot - lane_top, "acfb", "ACFB (one per scale k)")
    # lanes were added with z=0, so they sit behind everything
    for i, c in enumerate(cards):
        f.badge(c.x, c.head.cy, i + 1)
    # output
    oy = lane_bot + 44
    b_o = f.box(cxl + 60, oy, cw - 120, 62, "fus", "Fused output Φ_k\nC_k ch, H_k × W_k", size=21, bold=True)
    # arrows
    ym = ly + lh + 26
    f.link(b_s, "b", cards[0], "t", ym=ym)
    f.link(b_x, "b", cards[0], "t", ym=ym)
    for a, b in zip(cards, cards[1:]):
        f.link(a, "b", b, "t")
    f.link(cards[-1], "b", b_o, "t")
    f.label(cx, b_o.b + 16, "GroupNorm(8) is used only inside ACFB;\nthe encoder DoubleConv uses BatchNorm2d.",
            size=18, color="#495057", align="center")
    return f.write(out / "Figure_03_ACFB_Adaptive_Crossstream_Fusion_Block.excalidraw", "ACFB")


# ===========================================================================
# Figure 2 -- PSPI
# ===========================================================================
def figure_02(out):
    f = Fig("f2")
    rows = [("Scale 1: 256×256", "64", "256×256"), ("Scale 2: 128×128", "128", ""),
            ("Scale 3: 64×64", "256", ""), ("Scale 4: 32×32", "512", "")]
    lane_h, gap = 186, 16
    bh = 62
    rx = 292            # row lane x
    c2 = rx + 12
    c2w = 192
    ax = c2 + c2w + 46
    aw = 100
    px = ax + aw + 42
    pw = 104
    lane_r = px + pw + 12
    bus_x = lane_r + 34
    fx = bus_x + 30
    fw = 140
    total_h = 4 * lane_h + 3 * gap
    # SAM lane (left)
    sam_lane = f.lane(0, 0, 262, total_h, "sam", "SAM branch\n(frozen)")
    mid = total_h / 2
    sam_box = f.box(20, mid - 80, 222, 122, "sam", "Frozen SAM ViT-B\nimage encoder\n[B, 256, 64, 64]\ninput 256 → 1024", size=20, bold=True)
    f.label(131, mid + 54, "PSPI: bilinear resize\n(align_corners=False)", size=18, color="#495057", align="center")
    fsfd = f.box(fx, mid - 100, fw, 200, "dec", "FSFD\nFull-Scale\nFused\nDecoder", size=21, bold=True)
    yF = fsfd.cy
    lane_objs = []
    for i, (title, ch, _) in enumerate(rows):
        y0 = i * (lane_h + gap)
        f.lane(rx, y0, lane_r - rx, lane_h, "neu", title, title_color="#343a40")
        yr = y0 + 42
        red = f.box(c2, yr, c2w, bh, "sam", "SAM prior\n256 ch", size=20)
        blue = f.box(c2, yr + bh + 10, c2w, bh, "loc", f"Local encoder x{i + 1}\n{ch} ch", size=20)
        yc = (red.cy + blue.cy) / 2
        ac = f.box(ax, yc - 36, aw, 72, "acfb", f"ACFB {i + 1}", size=21, bold=True)
        ph = f.box(px, yc - 36, pw, 72, "fus", f"Φ{i + 1}\n{ch} ch", size=21)
        xm = c2 + c2w + 22
        f.link(red, "r", ac, "l", xm=xm)
        f.link(blue, "r", ac, "l", xm=xm)
        f.link(ac, "r", ph, "l")
        f.link(sam_box, "r", red, "l", xm=262 + (rx - 262) / 2)
        f.link(ph, "r", fsfd, "l", xm=bus_x)
    return f.write(out / "Figure_02_PSPI_Pyramid_Scale_Prior_Injection.excalidraw", "PSPI")


# ===========================================================================
# Figure 4 -- FSFD (matrix: rows = decoder levels, columns = encoder scales)
# ===========================================================================
def figure_04(out):
    f = Fig("f4")
    cw, ch, gap = 122, 84, 40
    x_lane = 0
    c = [68 + k * (cw + gap) for k in range(4)]          # cell x for columns 1..4
    lane_r = c[3] + cw + 20
    dc_x = lane_r + 44
    dc_w = 176
    hdr_y = 0
    jog = hdr_y + ch + 22
    # header boxes (encoder fused maps)
    hdr_txt = [("Φ1", "64 ch", "256×256"), ("Φ2", "128 ch", "128×128"), ("Φ3", "256 ch", "64×64"), ("Φ4", "512 ch", "32×32")]
    heads = []
    for k in range(4):
        heads.append(f.box(c[k], hdr_y, cw, ch, "fus", "\n".join(hdr_txt[k]), size=20, bold=True))
    lane_y = jog + 30
    # row definitions: (name, res, cells[(col, text)], in->out, caption)
    rows = [
        ("d4", "32×32", [(3, "Φ4\n512 ch", "fus")], "512 → 512", None),
        ("d3", "64×64", [(0, "Pool4(Φ1)\n64 ch", "fus"), (1, "Pool2(Φ2)\n128 ch", "fus"), (2, "Φ3\n256 ch", "fus"), (3, "Up(d4)\n512 ch", "dec")],
         "960 → 256", "64 + 128 + 256 + 512 = 960 ch"),
        ("d2", "128×128", [(0, "Pool2(Φ1)\n64 ch", "fus"), (1, "Φ2\n128 ch", "fus"), (3, "Up(d3)\n256 ch", "dec")],
         "448 → 128", "64 + 128 + 256 = 448 ch"),
        ("d1", "256×256", [(0, "Φ1\n64 ch", "fus"), (3, "Up(d2)\n128 ch", "dec")],
         "192 → 64", "64 + 128 = 192 ch"),
    ]
    cells = {}
    ups = set()
    dcs = []
    lanes = []
    y = lane_y
    for ri, (nm, res, cl, io, cap) in enumerate(rows):
        lh = 12 + ch + (10 + 22 + 8 if cap else 12)
        lx = x_lane if ri > 0 else c[3] - 40
        lane = f.lane(lx, y, lane_r - lx, lh, "dec")
        lanes.append((lane, y + 12 + ch / 2))
        for col, txt, kind in cl:
            cells[(ri, col)] = f.box(c[col], y + 12, cw, ch, kind, txt, size=20)
            if kind == "dec":
                ups.add((ri, col))
        dc = f.box(dc_x, y + 12, dc_w, ch, "dec", f"{nm} · {res}\nDoubleConv\n{io} ch", size=20, bold=True)
        dcs.append(dc)
        if cap:
            f.label(c[3] + cw, y + 12 + ch + 8, cap, size=18, color="#495057", align="right")
        y += lh + 40
    # arrows: headers -> cells (rail left of each cell)
    for k in range(4):
        for ri in range(4):
            if (ri, k) in cells and (ri, k) not in ups:
                cell = cells[(ri, k)]
                rail = c[k] - 26
                f.link(heads[k], "b", cell, "l", way=[(heads[k].cx, jog), (rail, jog), (rail, cell.cy)])
    # lane -> DoubleConv (lane acts as concatenation group)
    for ri, (lane, yc) in enumerate(lanes):
        # bind the arrow to the last cell of the row (right edge) for d4, the lane otherwise
        if ri == 0:
            f.link(cells[(0, 3)], "r", dcs[0], "l")
        else:
            f.link(lane, "r", dcs[ri], "l", sfrac=(yc - lane.y) / lane.h)
    # decoder chain: DoubleConv -> Up cell of the next row
    for ri in range(3):
        up = cells[(ri + 1, 3)]
        ymid = lanes[ri][0].b + 20
        f.link(dcs[ri], "b", up, "t", way=[(dcs[ri].cx, ymid), (up.cx, ymid)])
    head = f.box(dc_x - 12, dcs[3].b + 44, dc_w + 24, ch, "dec", "1×1 conv head\nsigmoid, thr. 0.5\n1 × 256 × 256", size=20, bold=True)
    f.link(dcs[3], "b", head, "t")
    f.label(0, head.y + 8, "Pool_s: max-pool, stride s\nUp: bilinear (align_corners=False)\nCat: channel-wise concatenation",
            size=18, color="#495057")
    return f.write(out / "Figure_04_FSFD_Full_Scale_Fused_Decoder.excalidraw", "FSFD")


# ===========================================================================
# Figure 1 -- pipeline
# ===========================================================================
def figure_01(out):
    f = Fig("f1")
    CW, G = 400, 40
    A, B, C = 0, CW + G, 2 * (CW + G)
    W = 3 * CW + 2 * G
    pad, iw = 18, CW - 36
    ts, bs = 19, 18
    gap = 26
    # ---- R1 data lane (cols A..B)
    y1 = 0
    yc = y1 + 44
    ospt = f.card(A + pad, yc, iw, "neu", "OSPT data loader",
                  "WeightedRandomSampler\nweight(i) = 1 / count(organ(i))\ndrawn with replacement", h=130, tsize=ts, bsize=bs)
    pre = f.card(B + pad, yc, iw, "neu", "Preprocessing, augmentation",
                 "resize to 256×256 (bilinear),\nscale to [0, 1]; only augmentation:\nhorizontal flip, p = 0.5 (training)", h=130, tsize=ts, bsize=bs)
    r1b = ospt.b + 18
    f.lane(A, y1, 2 * CW + G, r1b - y1, "neu", "Data")
    # ---- R2 branches + fusion
    y2 = r1b + 62
    yc = y2 + 44
    sam_enc = f.card(A + pad, yc, iw, "sam", "Frozen SAM ViT-B encoder",
                     "256×256 patch upsampled to\n1024×1024, official SAM preprocess\nembedding [B, 256, 64, 64]\nimage encoder only, no gradient", h=150, tsize=ts, bsize=bs)
    loc_enc = f.card(C + pad, yc, iw, "loc", "Local encoder (EPCS-Net)",
                     "4 DoubleConv stages, base 64\n(64 / 128 / 256 / 512), MaxPool\nDoubleConv = [Conv3×3, BN, ReLU] ×2", h=150, tsize=ts, bsize=bs)
    yc2 = sam_enc.b + gap
    pspi = f.card(A + pad, yc2, iw, "sam", "PSPI",
                  "Pyramid-Scale Prior Injection\nbilinear (align_corners=False)\nto 256 / 128 / 64 / 32", h=120, tsize=ts, bsize=bs)
    scales = f.card(C + pad, yc2, iw, "loc", "Four encoder scales",
                    "x1: 64 ch, 256×256   x2: 128 ch, 128×128\nx3: 256 ch, 64×64     x4: 512 ch, 32×32", h=120, tsize=ts, bsize=bs - 1)
    r2b = pspi.b + 18
    f.lane(A, y2, CW, r2b - y2, "sam", "SAM branch")
    f.lane(C, y2, CW, r2b - y2, "loc", "Local branch")
    acfb = f.card(B + pad, yc, iw, "acfb", "ACFB (one per scale, ×4)",
                  "Adaptive Cross-stream Fusion Block\n1 projection (1×1), 2 gating (BDSG)\n3 DKSR (3×3 + 5×5), 4 SE attention\n5 residual + GroupNorm(8)", h=150, tsize=ts, bsize=bs - 1)
    fused = f.box(B + pad, acfb.b + 40, iw, 92, "fus", "Fused maps\nΦ1: 64 ch    Φ2: 128 ch\nΦ3: 256 ch    Φ4: 512 ch", size=19, bold=True)
    f.lane(B, y2, CW, r2b - y2, "acfb", "Fusion")
    # ---- R3 decoder + training
    y3 = r2b + 56
    yc = y3 + 44
    fsfd = f.card(B + pad, yc, iw, "dec", "FSFD", "Full-Scale Fused Decoder\nevery encoder scale feeds every\ndecoder level; widths 512/256/128/64\ninput channels 512 (d4), 960 (d3),\n448 (d2), 192 (d1)", tsize=ts, bsize=bs - 1)
    head = f.box(B + pad, fsfd.b + gap + 6, iw, 76, "dec", "1×1 conv head → sigmoid\nthreshold 0.5 → binary mask", size=19, bold=True)
    f.lane(B, y3, CW, head.b + 18 - y3, "dec", "Decoder")
    loss = f.card(C + pad, yc, iw, "loss", "FTAS loss", "0.3 × BCE + 0.7 × focal-Tversky\nα = 0.7, β = 0.3, γ = 0.75", tsize=ts, bsize=bs)
    train = f.card(C + pad, loss.b + gap, iw, "loss", "Training protocol",
                   "AdamW, lr 3e-4, weight decay 1e-4\nbatch size 1, 50 epochs\nno scheduler, no gradient clipping\nsingle precision", tsize=ts, bsize=bs)
    ev = f.card(C + pad, train.b + gap, iw, "loss", "Evaluation protocol",
                "organ-stratified split; checkpoint on\nvalidation Dice only; test split used\nonce; five matched seeds, paired tests", tsize=ts, bsize=bs - 1)
    r3b = ev.b + 18
    f.lane(C, y3, CW, r3b - y3, "loss", "Training and evaluation")
    # ---- R4 post hoc
    y4 = r3b + 56
    dpnp = f.box(C + pad, y4 + 44, iw, 84, "neu", "DPNP: Deep Phenotypic\nNuclear Profiling\n(post hoc, exploratory)", size=19, bold=True)
    ly = dpnp.b + 44
    names = [("Grad-CAM", "attention vs. nuclei"), ("PCA", "connected-region features"), ("HDBSCAN", "density clustering of\nconnected regions")]
    leaves = [f.box(c0 + pad, ly, iw, 84, "neu", f"{t}\n{d}", size=18) for c0, (t, d) in zip((A, B, C), names)]
    f.lane(A, y4, W, leaves[0].b + 18 - y4, "neu", "Post hoc analysis (inference only)")
    # ---- arrows
    f.link(ospt, "r", pre, "l")
    ym = r1b + 31
    f.link(pre, "b", sam_enc, "t", ym=ym)
    f.link(pre, "b", loc_enc, "t", ym=ym)
    f.link(sam_enc, "b", pspi, "t")
    f.link(loc_enc, "b", scales, "t")
    f.link(pspi, "r", acfb, "l", xm=A + CW + G / 2)
    f.link(scales, "l", acfb, "r", xm=C - G / 2)
    f.link(acfb, "b", fused, "t")
    f.link(fused, "b", fsfd, "t")
    f.link(fsfd, "b", head, "t")
    f.link(head, "r", loss, "l", xm=C - G / 2)
    f.link(loss, "b", train, "t")
    f.link(train, "b", ev, "t")
    f.link(ev, "b", dpnp, "t")
    for lf in leaves:
        f.link(dpnp, "b", lf, "t", ym=dpnp.b + 22)
    return f.write(out / "Figure_01_EPCS_Net_Methodology_Pipeline.excalidraw", "pipeline")




def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "01_excalidraw_figures"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Writing Excalidraw sources to {out}")
    bad = 0
    for fn in (figure_01, figure_02, figure_03, figure_04):
        bad += len(fn(out))
    print("LINT ISSUES:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
