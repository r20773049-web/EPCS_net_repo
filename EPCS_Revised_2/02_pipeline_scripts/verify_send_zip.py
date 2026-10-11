"""Independent verification of SEND_TO_MAM/EPCS_Paper_Draft_Overleaf.zip:
figure modes/sizes, every \\input and \\includegraphics resolves, and the text is unchanged except layout wrappers
(diff of the reassembled tex against the pre-compilefix backup, ignoring layout-only constructs)."""
import io
import re
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
z = zipfile.ZipFile(ROOT / "SEND_TO_MAM" / "EPCS_Paper_Draft_Overleaf.zip")
names = z.namelist()
print("zip test:", z.testzip(), "| files:", len(names), "| MB:", round((ROOT / "SEND_TO_MAM" / "EPCS_Paper_Draft_Overleaf.zip").stat().st_size / 1e6, 2))

tot_mp = 0.0
bad = []
for n in names:
    if n.startswith("figures/") and n.lower().endswith((".png", ".jpg")):
        im = Image.open(io.BytesIO(z.read(n)))
        mp = im.width * im.height / 1e6
        tot_mp += mp
        if im.mode != "RGB" or im.width > 2200 or im.info.get("interlace"):
            bad.append((n, im.mode, im.size))
print("figures decoded total MP:", round(tot_mp, 1), "| non-RGB / too wide / interlaced:", bad or "none")

main = z.read("main.tex").decode("utf-8")
sections = {n: z.read(n).decode("utf-8") for n in names if n.startswith("sections/")}
inputs = re.findall(r"\\input\{([^}]+)\}", main)
missing_in = [i for i in inputs if (i if i.endswith(".tex") else i + ".tex") not in names]
allsrc = main + "\n".join(sections.values())
inc = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", allsrc)
print("inputs:", len(inputs), "missing:", missing_in or "none", "| includegraphics:", len(inc), "missing:", [i for i in inc if i not in names] or "none")

# content check: strip layout-only constructs from old and new and compare words
old = (ROOT / "07_revision3" / "05_manuscript" / "_backups" / "EPCS_revise3_before_compilefix.bak").read_text(encoding="utf-8")
new = (ROOT / "07_revision3" / "05_manuscript" / "EPCS_revise3.tex").read_text(encoding="utf-8")


def norm(t):
    t = re.sub(r"\\sloppy|\\emergencystretch=3em|\\par\\nobreak\\vspace\{4pt\}|\\\\\[4pt\]|\\resizebox\{\\textwidth\}\{!\}\{%?|\\par", "", t)
    t = re.sub(r"[{}%\s]+", " ", t)
    return t.strip()


a, b = norm(old).split(), norm(new).split()
import difflib
sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
changes = [(tag, " ".join(a[i1:i2])[:60], " ".join(b[j1:j2])[:60]) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
print("word-level differences old vs new after removing layout constructs:", len(changes))
for c in changes[:10]:
    print("  ", c)
