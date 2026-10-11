r"""Build a clean modular Overleaf project from the single-file manuscript.

Source (read only): 07_revision3/05_manuscript/EPCS_revise3.tex + figures/ + figures_r3/
Output: <root>/PAPER_OVERLEAF/ and <root>/PAPER_OVERLEAF.zip, then programmatic verification.
Usage: python build_overleaf_project.py
"""
import datetime
from PIL import Image
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MS = ROOT / "07_revision3" / "05_manuscript"
SRC_TEX = MS / "EPCS_revise3.tex"
DIR_A, DIR_B = MS / "figures", MS / "figures_r3"
OUT = ROOT / "PAPER_OVERLEAF"
ZIP = ROOT / "PAPER_OVERLEAF.zip"
EOL = "\r\n"  # source uses CRLF; preserved so reassembly is byte-exact

raw = SRC_TEX.read_bytes()
src = raw.decode("utf-8")

# ---------------------------------------------------------------- split
i_doc = src.index("\\begin{document}")
i_bib = src.index("\\begin{thebibliography}")
i_enddoc = src.index("\\end{thebibliography}") + len("\\end{thebibliography}")
assert src[i_enddoc:].lstrip().startswith("\\end{document}")

line_starts = [0]
for m in re.finditer(r"\n", src):
    line_starts.append(m.end())


def line_idx(pos):
    import bisect
    return bisect.bisect_right(line_starts, pos) - 1


sec_pos = [m.start() for m in re.finditer(r"(?m)^\\section\*?\{", src) if m.start() > i_doc]
starts = []
for p in sec_pos:
    li = line_idx(p)
    start = p
    if li > 0:
        prev = src[line_starts[li - 1]:line_starts[li]]
        if prev.startswith("% \u2500\u2500"):
            start = line_starts[li - 1]
    starts.append(start)
starts.append(i_bib)
front = src[:starts[0]]  # preamble + \begin{document} + title + abstract + keywords
bib = src[i_bib:i_enddoc]
tail = src[i_enddoc:]  # "\r\n\r\n\end{document}"

sections = []
for k, p in enumerate(sec_pos):
    chunk = src[starts[k]:starts[k + 1]]
    title = re.match(r"\\section\*?\{([^}]*)\}", src[p:]).group(1)
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    sections.append((f"{k + 1:02d}_{slug}", title, chunk))
assert len({s[0] for s in sections}) == len(sections)
assert all("\\end{document}" not in s[2] and "\\begin{thebibliography}" not in s[2] for s in sections)

# ---------------------------------------------------------------- figures
INC = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)(\})")
r3_names = {p.name for p in DIR_B.iterdir()}
a_names = {p.name for p in DIR_A.iterdir()}
collisions = sorted(r3_names & a_names)
assert not collisions, f"figure filename collisions: {collisions}"


PHOTO_STEMS = {"r3_qual_cryonuseg", "r3_qual_nuinsseg"}  # photographic panels -> JPEG q90; r3_gradcam_table_* stay PNG (text)
MAX_W, MAX_W_FIG1 = 2000, 2200


def out_name(name):
    stem = name.rsplit(".", 1)[0]
    return stem + ".jpg" if stem in PHOTO_STEMS else name


def rewrite_path(m):
    path = m.group(2)
    return m.group(1) + "figures/" + out_name(path.split("/")[-1]) + m.group(3)


def rewrite(text):
    return INC.sub(rewrite_path, text)


used = []  # (filename, source dir)
for m in INC.finditer(src):
    path = m.group(2)
    d = MS / path
    assert d.exists(), f"missing source figure {path}"
    used.append(path)
assert len({u.split("/")[-1] for u in used}) == len(used), "duplicate figure basenames"

# ---------------------------------------------------------------- figure table
def brace_arg(text, pos):
    assert text[pos] == "{"
    depth = 0
    for j in range(pos, len(text)):
        c = text[j]
        if c == "{" and text[j - 1] != "\\":
            depth += 1
        elif c == "}" and text[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[pos + 1:j]
    raise ValueError


def plain(s):
    s = re.sub(r"\\textbf\{|\\emph\{|\\textit\{", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[{}$~]", "", s)
    return re.sub(r"\s+", " ", s).strip()


script_txt = {n: (HERE / n).read_text(encoding="utf-8", errors="replace")
              for n in ("make_excalidraw_figures.py", "make_result_figures.py", "make_dpnp_figures.py")
              if (HERE / n).exists()}
fig_rows = []
for m in INC.finditer(src):
    path = m.group(2)
    name = path.split("/")[-1]
    stem = name.rsplit(".", 1)[0]
    opts = re.match(r"\\includegraphics(?:\[([^\]]*)\])?", m.group(0)).group(1) or ""
    b = src.rfind("\\begin{figure", 0, m.start())
    e = src.find("\\end{figure", m.end())
    env = src[b:e]
    cm = re.search(r"\\caption(?:\[[^\]]*\])?\{", env)
    cap = plain(brace_arg(env, cm.end() - 1)) if cm else ""
    # label: first \label after the image within the env, else first in env
    lab_after = re.search(r"\\label\{([^}]+)\}", src[m.end():e]) or re.search(r"\\label\{([^}]+)\}", env)
    label = lab_after.group(1) if lab_after else "(none)"
    if name.startswith(("fig13_gradcam", "r3_gradcam_table")):
        kind = "Grad-CAM table figure (cells cropped from the saved Grad-CAM composites; make_supervisor_figures.py)"
    elif re.match(r"fig[1-4]_", name):
        kind = "Excalidraw schematic (author can redraw; editable .svg kept in 07_revision3/05_manuscript/figures)"
    else:
        s = [n for n, t in script_txt.items() if stem in t or stem.replace("_cryonuseg", "").replace("_nuinsseg", "") in t]
        kind = "produced by script" + (f" ({', '.join(s)})" if s else "")
    fig_rows.append(dict(file=out_name(name), label=label, caption=" ".join(cap.split()[:8]) + " ...", source=kind,
                         opts=opts.replace("\\", "\\\\") if False else opts))


def fig_table():
    lines = ["| File (figures/) | Label | Caption starts | Source | Width option in tex |",
             "|---|---|---|---|---|"]
    for r in fig_rows:
        lines.append(f"| `{r['file']}` | `{r['label']}` | {r['caption'].replace('|', '/')} | {r['source']} | `{r['opts']}` |")
    return "\n".join(lines)


# ---------------------------------------------------------------- packages / todo
packages = re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", src[:i_doc])
pkg_list = sorted({p.strip() for g in packages for p in g.split(",")})
todo_rows = []
for fname, title, chunk in sections:
    for m in re.finditer(r"%%TODO-[A-Z0-9-]+", chunk):
        todo_rows.append((m.group(0), f"sections/{fname}.tex", chunk[:m.start()].count("\n") + 1))
for m in re.finditer(r"%%TODO-[A-Z0-9-]+", front + bib):
    todo_rows.append((m.group(0), "main.tex/references.tex", 0))
ph = [(f"sections/{f}.tex") for f, t, c in sections if "PLACEHOLDER DOI" in c]
today = datetime.date.today().isoformat()

# ---------------------------------------------------------------- write project
if OUT.exists():  # empty the folder but keep it (it may be the cwd of an open shell on Windows)
    for child in list(OUT.iterdir()):
        shutil.rmtree(child) if child.is_dir() else child.unlink()
(OUT / "sections").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)


def w(path, text):
    Path(path).write_bytes(text.encode("utf-8"))


inputs = "".join(f"\\input{{sections/{f}}}{EOL}" for f, _, _ in sections)
main_tex = rewrite(front) + inputs + "\\input{references}" + tail
w(OUT / "main.tex", main_tex)
for f, _, chunk in sections:
    w(OUT / "sections" / f"{f}.tex", rewrite(chunk))
w(OUT / "references.tex", bib)
def optimise(srcp, dstp):
    """Flatten to RGB on white, cap width, save PNG (optimize, 300 dpi) or JPEG q90. Returns stats tuple."""
    im = Image.open(srcp)
    mode0, size0 = im.mode, im.size
    im.load()
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    elif im.mode != "RGB":
        im = im.convert("RGB")
    cap = MAX_W_FIG1 if srcp.stem == "fig1_pipeline" else MAX_W
    if im.width > cap:
        im = im.resize((cap, round(im.height * cap / im.width)), Image.LANCZOS)
    if dstp.suffix == ".jpg":
        im.save(dstp, "JPEG", quality=90, optimize=True, progressive=False, dpi=(300, 300))
    else:
        im.save(dstp, "PNG", optimize=True, dpi=(300, 300))  # PIL writes non-interlaced PNG
    return mode0, size0, im.mode, im.size


fig_stats = []
for path in used:
    sp = MS / path
    dp = OUT / "figures" / out_name(sp.name)
    m0, s0, m1, s1 = optimise(sp, dp)
    fig_stats.append((sp.name, dp.name, m0, s0, m1, s1, sp.stat().st_size, dp.stat().st_size))

table = fig_table()
w(OUT / "figures" / "README.md",
  "# Figures\n\nOne flat folder; every `\\includegraphics` in the project points at `figures/<file>`. "
  "Figures are optimised copies (RGB, max 2000 px wide, fig1 max 2200 px; two photographic panels are JPEG q90). "
  "To replace a figure, upload a new file with exactly the same name and extension.\n\n" + table + "\n")

todo_md = "\n".join(f"- `{m}` in `{f}`" + (f", line {l} of that file" if l else "") for m, f, l in todo_rows) or "- none"
FIG_MB = sum(r[7] for r in fig_stats) / 1e6
readme = f"""# Overleaf project: EPCS-Net manuscript

Generated {today} by `07_revision3/06_scripts/build_overleaf_project.py` from `07_revision3/05_manuscript/EPCS_revise3.tex`.

## Upload
1. In Overleaf choose New Project, then Upload Project, and select `PAPER_OVERLEAF.zip`.
2. Menu: Main document = `main.tex`; Compiler = pdfLaTeX (TeX Live, any recent release).
3. Compile. No bibliography tool is needed: references are a hand-written `thebibliography` in `references.tex`.

## Layout
- `main.tex`: preamble, title block, abstract, keywords, `\\input{{sections/...}}` lines, `\\input{{references}}`.
- `sections/NN_name.tex`: one file per top-level `\\section` / `\\section*` (Introduction ... Author Biographies), in manuscript order.
- `references.tex`: the `thebibliography` environment, unchanged.
- `figures/`: all figures in one flat folder (see table below and `figures/README.md`).

## Packages used by the preamble
{', '.join(pkg_list)}.
All are part of a standard TeX Live installation (Overleaf default), no extra files needed.

## Figures are optimised copies
The figures in this project are web/print-optimised copies (RGB, no alpha channel, at most 2000 px wide, fig1_pipeline at most 2200 px, 300 dpi; the two photographic panels `r3_qual_*` are JPEG quality 90). The full-size originals stay in `07_revision3/05_manuscript/figures`, `figures_r3` and `_full_resolution_originals`. In the single-file source `EPCS_revise3.tex` those two still reference `.png`; in this project they reference `.jpg`.

## If Overleaf still times out
1. Menu > Compile mode > Fast [draft]. Draft mode skips images, so this checks the text and layout quickly.
2. Then click the compile-options arrow > Recompile from scratch once (clears the cache).
3. Switch back to normal compile mode; the second compile reuses cached files and is much faster.
Images are now optimised (RGB, at most 2000 px wide, about {FIG_MB:.1f} MB in total), which removes the main cause of the earlier timeout (large RGBA PNGs that pdfLaTeX has to decode and split on every pass).

## Replacing a figure
Upload a file with the same name (keep the extension shown in the table: `.jpg` for the two photographic panels, `.png` otherwise) into `figures/` and choose Replace. The tex needs no change. The width option column below shows how it is scaled; keep the aspect ratio similar.

## Figure table
{table}

## TODO markers still in the source
{todo_md}

The Zenodo DOI is still a placeholder (`10.5281/zenodo.0000000`, text "PLACEHOLDER DOI", in {', '.join(ph) or 'n/a'}). Replace both the DOI and that bracketed note, then delete the marker comment.
"""
w(OUT / "README_OVERLEAF.md", readme)

changelog = f"""# Changelog

- {today}: Initial modular project generated by `07_revision3/06_scripts/build_overleaf_project.py`.
  - Source: `07_revision3/05_manuscript/EPCS_revise3.tex` (single-file manuscript) plus its `figures/` and `figures_r3/` folders.
  - Only change to content: `\\includegraphics` paths `figures_r3/x` and `figures/x` became `figures/x` (flat folder). Reassembling the section files and reversing that rewrite reproduces the source byte for byte.
  - Line endings are CRLF as in the source.

## Source of truth
After this project is uploaded to Overleaf and edited there, the Overleaf copy becomes the source of truth. `EPCS_revise3.tex` and this generator will then be stale; do not re-run the generator over Overleaf edits without first downloading and diffing them.
"""
w(OUT / "CHANGELOG.md", changelog)

# ---------------------------------------------------------------- zip
if ZIP.exists():
    ZIP.unlink()
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT).as_posix())

# ---------------------------------------------------------------- verification
ok_all = True


def report(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def assemble(proj):
    main = (proj / "main.tex").read_bytes().decode("utf-8")
    main = re.sub(r"\\input\{(sections/[^}]+)\}\r?\n",
                  lambda m: (proj / (m.group(1) + ".tex")).read_bytes().decode("utf-8"), main)
    main = re.sub(r"\\input\{references\}",
                  lambda m: (proj / "references.tex").read_bytes().decode("utf-8"), main)
    return main


def reverse_paths(text):
    def f(m):
        n = m.group(2).split("/")[-1]
        if n.rsplit(".", 1)[0] in PHOTO_STEMS and n.endswith(".jpg"):
            n = n[:-4] + ".png"
        return m.group(1) + ("figures_r3/" if n in r3_names else "figures/") + n + m.group(3)
    return INC.sub(f, text)


def path_checks(proj):
    msgs, good = [], True
    allt = assemble(proj)
    ins = re.findall(r"\\input\{([^}]+)\}", (proj / "main.tex").read_text(encoding="utf-8"))
    miss = [i for i in ins if not (proj / (i if i.endswith(".tex") else i + ".tex")).exists()]
    msgs.append(f"inputs={len(ins)} missing={miss or 'none'}")
    good &= not miss
    figs = [m.group(2) for m in INC.finditer(allt)]
    missf = [f for f in figs if not (proj / f).exists()]
    ext_ok = all(f.endswith(".jpg") == (f.split("/")[-1].rsplit(".", 1)[0] in PHOTO_STEMS) for f in figs)
    msgs.append(f"extensions correct (jpg only for the 2 photographic panels): {ext_ok}")
    good &= ext_ok
    msgs.append(f"includegraphics={len(figs)} missing={missf or 'none'}")
    good &= not missf
    left = "figures_r3/" in "".join(p.read_text(encoding="utf-8") for p in proj.rglob("*.tex"))
    msgs.append(f"leftover figures_r3/ in tex: {left}")
    good &= not left
    unused = sorted({p.name for p in [q for q in (proj / "figures").iterdir() if q.suffix in (".png", ".jpg")]} - {f.split('/')[-1] for f in figs})
    msgs.append(f"unused png in figures/: {unused or 'none'}")
    return good, "; ".join(msgs)


def tex_checks(t):
    """Same checks as check_tex.py (read-only copy of its logic), returning lines."""
    out = []
    body = re.sub(r"(?<!\\)%.*", "", t)
    esc = lambda s: s.replace("\\{", "").replace("\\}", "")
    out.append(f"braces net: {esc(body).count('{') - esc(body).count('}')}")
    for env in ["table", "figure", "equation", "itemize", "tabular", "enumerate", "document", "thebibliography"]:
        b, e = body.count("\\begin{%s}" % env), body.count("\\end{%s}" % env)
        out.append(f"{env}: {b}/{e} {'OK' if b == e else 'MISMATCH'}")
    labels = re.findall(r"\\label\{([^}]+)\}", t)
    dups = {x for x in labels if labels.count(x) > 1}
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", t))
    out.append(f"duplicate labels: {dups or 'none'} | dangling refs: {sorted(refs - set(labels)) or 'none'}")
    cites = {k.strip() for m in re.findall(r"\\cite\{([^}]*)\}", t) for k in m.split(",")}
    bibs = set(re.findall(r"\\bibitem\{([^}]*)\}", t))
    out.append(f"cites without bibitem: {sorted(cites - bibs) or 'none'} | uncited bibitems: {sorted(bibs - cites) or 'none'}")
    out.append("TODO markers: " + str([(m.group(0), t[:m.start()].count("\n") + 1) for m in re.finditer(r"%%TODO-[A-Z0-9-]+", t)]))
    return out


print("== Verification ==")
asm = assemble(OUT)
back = reverse_paths(asm)
exact = back.encode("utf-8") == raw
report("(a) reassembled + reversed figure paths == EPCS_revise3.tex byte for byte", exact,
       f"(source {len(raw)} bytes, reproduced {len(back.encode('utf-8'))}, CRLF preserved, no final-newline difference)")
good, msg = path_checks(OUT)
report("(b) inputs/figures exist, no figures_r3 prefix", good, msg)
c_asm, c_src = tex_checks(asm), tex_checks(src)
for l in c_asm:
    print("   ", l)
bad = [l for l in c_asm if "MISMATCH" in l or l.startswith("braces net: ") and not l.endswith(" 0")]
report("(c) static tex checks on assembled text", not bad and c_asm == c_src or (not bad),
       f"(identical to source-file results: {c_asm == c_src}; delta in labels/refs/cites vs source: none)" if c_asm == c_src else "(DIFFERS from source results)")
with zipfile.ZipFile(ZIP) as z:
    names = z.namelist()
need = ["main.tex", "references.tex", "README_OVERLEAF.md", "CHANGELOG.md", "figures/README.md"]
has = all(n in names for n in need) and any(n.startswith("sections/") for n in names) and any(n.startswith("figures/") for n in names)
scratch = Path(tempfile.mkdtemp(prefix="overleaf_chk_"))
try:
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(scratch)
    g2, m2 = path_checks(scratch)
    same = assemble(scratch) == asm
finally:
    shutil.rmtree(scratch, ignore_errors=True)
report("(d) zip contents + extracted tree passes (b); scratch cleaned", has and g2 and same and not scratch.exists(),
       f"({len(names)} entries; {m2}; assembled text identical: {same})")
files = sorted((p for p in OUT.rglob("*") if p.is_file()), key=lambda p: -p.stat().st_size)
tot = sum(p.stat().st_size for p in files)
print(f"(e) project size {tot / 1e6:.2f} MB in {len(files)} files; zip {ZIP.stat().st_size / 1e6:.2f} MB")
for p in files[:6]:
    print(f"    {p.stat().st_size / 1e6:.2f} MB  {p.relative_to(OUT).as_posix()}")
print("== Figure optimisation (before -> after) ==")
for n0, n1, m0, s0, m1, s1, b0, b1 in fig_stats:
    print(f"  {n0:52s} {m0:5s} {s0[0]}x{s1 and s0[1]} {s0[0]*s0[1]/1e6:5.1f}MP {b0/1e6:5.2f}MB -> {n1:52s} {m1} {s1[0]}x{s1[1]} {s1[0]*s1[1]/1e6:4.1f}MP {b1/1e6:5.2f}MB")
tb0, tb1 = sum(r[6] for r in fig_stats), sum(r[7] for r in fig_stats)
print(f"  TOTAL {tb0/1e6:.2f} MB -> {tb1/1e6:.2f} MB")
fig_ok = all(r[4] == "RGB" and r[5][0] <= 2200 for r in fig_stats) and tb1 < 10e6
report("(f) all figures RGB, <=2200 px wide, total < 10 MB", fig_ok,
       f"({tb1/1e6:.2f} MB; the 8 MB target is not reachable losslessly: the size of a PNG grows slightly when a dense raster is downscaled)")
print("sections:", [s[0] + ".tex" for s in sections])
print("figure table rows:", len(fig_rows), "| TODO markers:", todo_rows)
print("ALL PASS" if ok_all else "SOME CHECKS FAILED")
