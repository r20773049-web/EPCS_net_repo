"""Assemble package/EPCS_revise3_overleaf[_DRAFT|_FINAL].zip (clean tex, highlighted tex, figures, references note).
Zip is named _FINAL only if zero %%TODO- markers remain and no referenced figure is missing; otherwise _DRAFT.
Figures are copied only from where the tex expects them (relative to the tex dir). Stale copies in
OLD_ROUNDS_1_2/06_manuscript_revision are NOT used (old validation figures); they are listed as 'stale version exists'.
Usage: python build_overleaf_package.py"""
import re, sys, zipfile, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_highlighted_tex as bh

ROOT = bh.ROOT
TEX = bh.NEW
PKG = TEX.parent / "package"
OLD_FIG_DIR = ROOT / "OLD_ROUNDS_1_2" / "06_manuscript_revision"


def main():
    PKG.mkdir(parents=True, exist_ok=True)
    if bh.main() != 0:
        print("highlight verification FAILED; aborting package")
        return 1
    txt = TEX.read_text(encoding="utf-8")
    lines = txt.splitlines()

    # TODO markers
    todos = [(i + 1, l.strip()) for i, l in enumerate(lines) if "%%TODO-" in l]
    print("\n== TODO markers: %d ==" % len(todos))
    for n, l in todos:
        print("  line %d: %s" % (n, l[:110]))
    kinds = {}
    for n, l in todos:
        k = re.search(r"%%(TODO-[A-Z0-9-]+)", l).group(1)
        kinds[k] = kinds.get(k, 0) + 1
    print("  by kind:", kinds)

    # figures
    nocomment = re.sub(r"(?m)(?<!\\)%.*$", "", txt)
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", nocomment)
    present, missing = [], []
    for f in figs:
        p = TEX.parent / f
        if p.exists():
            present.append((f, p))
        else:
            stale = OLD_FIG_DIR / Path(f).name
            missing.append((f, stale.exists()))
    print("\n== Figures: %d referenced, %d present, %d missing ==" % (len(figs), len(present), len(missing)))
    for f, stale in missing:
        print("  MISSING %s%s" % (f, "  (stale old-validation version exists in OLD_ROUNDS_1_2/06_manuscript_revision, not copied)" if stale else ""))

    # references note
    bibs = re.findall(r"\\bibitem\{([^}]+)\}", txt)
    old = bh.OLD.read_text(encoding="utf-8")
    oldb = set(re.findall(r"\\bibitem\{([^}]+)\}", old))
    newb = [b for b in bibs if b not in oldb]
    note = ["References note (EPCS_revise3)",
            "",
            "The manuscript uses an inline thebibliography environment (no .bib needed to compile).",
            "Total bibitems: %d; new relative to revise2: %d" % (len(bibs), len(newb)),
            "New keys: " + (", ".join(newb) or "none"),
            "new_references_R3.bib in the manuscript folder is a working list for the added entries only.",
            "Compile with pdflatex twice (hyperref/labels).",
            "Open TODO markers at build time: %d" % len(todos)]
    note_path = PKG / "REFERENCES_NOTE.txt"
    note_path.write_text("\n".join(note) + "\n", encoding="utf-8")

    # \input files (only included if the tex inputs them)
    inputs = []
    for f in re.findall(r"\\(?:input|include)\{([^}]+)\}", nocomment):
        fp = TEX.parent / (f if Path(f).suffix else f + ".tex")
        inputs.append((fp.relative_to(TEX.parent).as_posix(), fp))
    miss_in = [f for f, p in inputs if not p.exists()]
    inputs = [(f, p) for f, p in inputs if p.exists()]
    print("input files: %d (missing %d)" % (len(inputs) + len(miss_in), len(miss_in)))
    missing += [(f, False) for f in miss_in]

    final = not todos and not missing
    for old_zip in PKG.glob("EPCS_revise3_overleaf*.zip"):
        old_zip.unlink()
    zpath = PKG / ("EPCS_revise3_overleaf_%s.zip" % ("FINAL" if final else "DRAFT"))
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(TEX, "EPCS_revise3.tex")
        z.write(bh.OUT, "EPCS_revise3_HIGHLIGHTED.tex")
        for f, p in present:
            z.write(p, f)
        for f, p in inputs:
            z.write(p, f)
        z.write(note_path, "REFERENCES_NOTE.txt")
        z.writestr("TODO_REPORT.txt", "\n".join("line %d: %s" % t for t in todos) or "no markers")
        if missing:
            z.writestr("MISSING_FIGURES.txt", "\n".join(f for f, _ in missing))
    import tempfile
    pat = r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}"
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(td)
        bad = []
        for t in ("EPCS_revise3.tex", "EPCS_revise3_HIGHLIGHTED.tex"):
            hx = re.sub(r"(?m)(?<!\\)%.*$", "", (Path(td) / t).read_text(encoding="utf-8"))
            bad += [t + ":" + f for f in re.findall(pat, hx) if not (Path(td) / f).exists()]
    print("flat-unzip check: %d includegraphics paths unresolved %s" % (len(bad), bad))
    print("\nwrote", zpath, "(%d bytes)" % zpath.stat().st_size)
    print("status:", "FINAL" if final else "DRAFT (markers remain or figures missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
