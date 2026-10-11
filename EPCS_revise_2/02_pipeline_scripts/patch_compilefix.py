"""Layout-only compile fixes for EPCS_revise3.tex (no wording / number changes).
Backs up first, then applies assertion-guarded exact-match edits. Idempotent guard: refuses if already applied.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "07_revision3" / "05_manuscript" / "EPCS_revise3.tex"
BAK = TEX.parent / "_backups" / "EPCS_revise3_before_compilefix.bak"

raw = TEX.read_bytes()
if not BAK.exists():
    BAK.parent.mkdir(exist_ok=True)
    shutil.copy2(TEX, BAK)
s = raw.decode("utf-8")
assert "\\emergencystretch" not in s, "already patched"


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:70])
    s = s.replace(old, new)


# 1. preamble: allow TeX a third line-breaking pass (does not touch other settings)
rep("\\setlength{\\parskip}{6pt}\r\n", "\\setlength{\\parskip}{6pt}\r\n\\emergencystretch=3em\r\n")

# 2. underfull hbox at main.tex line 83: "\\[4pt]" at end of a paragraph leaves an underfull empty last line
rep("\\bfseries\\color{wordblue} Abstract}\\\\[4pt]\r\n",
    "\\bfseries\\color{wordblue} Abstract}\\par\\nobreak\\vspace{4pt}\r\n")

# 3. the two identical efficiency tabulars (5 columns, long headers): scale to text width
for lab in ("tab:cryo_efficiency", "tab:nu_efficiency"):
    i = s.index("\\label{%s}\r\n" % lab)
    j = s.index("\\begin{tabular}{lcccc}\r\n", i)
    assert j - i < 60 + len(lab)
    k = s.index("\\end{tabular}\r\n", j)
    s = (s[:j] + "\\resizebox{\\textwidth}{!}{%\r\n" + s[j:k] + "\\end{tabular}%\r\n}\r\n" + s[k + len("\\end{tabular}\r\n"):])

# 4. overfull text paragraphs: local \sloppy (group closed by \par), wording untouched
for start in (
    "The CryoNuSeg dataset \\cite{r32} comprises 30 H\\&E-stained",
    "Patch embedding at patch size 16 therefore produces a token sequence",
    "We replaced ACFB inside the EPCS-Net backbone by the X-Gated Fusion Block",
):
    assert s.count(start) == 1, start
    i = s.index(start)
    e = s.index("\r\n", i)
    s = s[:i] + "{\\sloppy " + s[i:e] + "\\par}" + s[e:]
start = "  \\item \\textbf{Reference-21 fusion, bottleneck and four-scale:} The X-Gated"
assert s.count(start) == 1
i = s.index(start)
e = s.index("\r\n", i)
s = s[:i] + "  \\item {\\sloppy " + s[i + len("  \\item "):e] + "\\par}" + s[e:]

TEX.write_bytes(s.encode("utf-8"))
print("patched", TEX, len(raw), "->", len(s.encode("utf-8")))
