"""Split the 8-column result tables: tab:cryo/tab:nu keep Model, Loss, Dice, IoU; tab:*_allmetrics keep Accuracy, Precision, Recall, Specificity.
Removes resizebox so the text is set at a normal size. Assertion-guarded; run once."""
import re
from pathlib import Path
P = Path(r"<PROJECT_ROOT>\05_manuscript\EPCS_revise3.tex")
raw = P.read_bytes().decode("utf-8")
crlf = "\r\n" in raw
L = raw.replace("\r\n", "\n").split("\n")


def find(label):
    i = next(k for k, l in enumerate(L) if f"\\label{{{label}}}" in l)
    s = i
    while "\\begin{tabular}" not in L[s]:
        s += 1
    e = s
    while "\\end{tabular}" not in L[e]:
        e += 1
    return i, s, e


def trim_rows(s, e, keep, trailing):
    for k in range(s + 2, e):
        l = L[k]
        if "&" not in l:
            continue
        body = l.rstrip()
        end = ""
        m = re.search(r"\\\\.*$", body)
        if m:
            end = m.group(0)
            body = body[: m.start()]
        cells = body.split("&")
        assert len(cells) == trailing, (k, len(cells), l[:80])
        L[k] = "&".join(cells[i] for i in keep).rstrip() + " " + end


def rework(label, nin, keep, spec_new):
    i, s, e = find(label)
    # drop resizebox opening/closing
    j = s - 1
    while "\\resizebox" not in L[j]:
        j -= 1
    assert i < j < s
    L[j] = "\\small"
    tab = L[s]
    assert tab.startswith("\\begin{tabular}")
    L[s] = "\\begin{tabular}{" + spec_new + "}"
    assert L[e].strip() == "\\end{tabular}%" and L[e + 1].strip() == "}", L[e : e + 2]
    L[e] = "\\end{tabular}"
    L[e + 1] = ""
    trim_rows(s, e, keep, nin)
    # also unify tabcolsep
    return i


for lab in ("tab:cryo", "tab:nu"):
    rework(lab, 8, [0, 1, 2, 3], "llcc")
for lab in ("tab:cryo_allmetrics", "tab:nu_allmetrics"):
    rework(lab, 7, [0, 1, 2, 3, 4], "lcccc")
out = "\n".join(L)
if crlf:
    out = out.replace("\n", "\r\n")
P.write_bytes(out.encode("utf-8"))
print("ok")
