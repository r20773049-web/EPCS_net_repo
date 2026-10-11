"""Unzip SEND_TO_MAM/EPCS_Paper_Draft_Overleaf.zip into a scratch folder and check that every \\input and every
\\includegraphics resolves; also compares it with PAPER_OVERLEAF.zip and estimates tabular widths."""
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A, B = ROOT / "SEND_TO_MAM" / "EPCS_Paper_Draft_Overleaf.zip", ROOT / "PAPER_OVERLEAF.zip"
print("zips byte-identical:", A.read_bytes() == B.read_bytes())
tmp = Path(tempfile.mkdtemp(prefix="ovl_chk_"))
bad = 0
try:
    with zipfile.ZipFile(A) as z:
        z.extractall(tmp)
    texs = [p for p in tmp.rglob("*.tex")]
    for p in texs:
        t = re.sub(r"(?<!\\)%.*", "", p.read_text(encoding="utf-8"))
        for m in re.findall(r"\\input\{([^}]+)\}", t):
            f = tmp / (m if m.endswith(".tex") else m + ".tex")
            if not f.exists():
                bad += 1
                print("MISSING input", m)
        for m in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t):
            if not (tmp / m).exists():
                bad += 1
                print("MISSING figure", m)
    n_in = sum(len(re.findall(r"\\input\{", p.read_text(encoding="utf-8"))) for p in texs)
    n_fig = sum(len(re.findall(r"\\includegraphics", p.read_text(encoding="utf-8"))) for p in texs)
    print(f"tex files {len(texs)}, \\input {n_in}, \\includegraphics {n_fig}, unresolved {bad}")

    # tabular width estimate (character-based, 12pt Times ~ 5.2pt/char, text width 7in = 504pt, \tabcolsep 6pt)
    full = "\n".join(p.read_text(encoding="utf-8") for p in sorted(tmp.rglob("*.tex")))
    pos = 0
    for m in re.finditer(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", full, re.S):
        spec, body = m.group(1), m.group(2)
        ncol = len(re.findall(r"[lcrp]", re.sub(r"\{[^}]*\}", "", spec)))
        rows = [r for r in body.split("\\\\") if r.strip() and not r.strip().startswith(("\\toprule", "\\bottomrule"))]
        widths = [0.0] * ncol
        for r in rows:
            cells = r.split("&")
            for i, c in enumerate(cells[:ncol]):
                c = re.sub(r"\\[a-zA-Z]+|[{}$\\]", "", c)
                widths[i] = max(widths[i], len(c.strip()) * 5.2)
        est = sum(widths) + 12 * ncol
        before = full[max(0, m.start() - 40):m.start()]
        wrapped = "resizebox" in before
        flag = "OK(resizebox)" if wrapped else ("OVER" if est > 504 else "fits")
        if flag == "OVER" or wrapped:
            pass
        print(f"tabular cols={ncol} est={est:6.0f}pt {flag}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print("scratch removed:", not tmp.exists())
sys.exit(1 if bad else 0)
