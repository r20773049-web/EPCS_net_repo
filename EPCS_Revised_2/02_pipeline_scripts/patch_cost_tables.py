from pathlib import Path
P = Path(r"<PROJECT_ROOT>\05_manuscript\EPCS_revise3.tex")
raw = P.read_bytes().decode("utf-8")
crlf = "\r\n" in raw
L = raw.replace("\r\n", "\n").split("\n")
for lab in ("tab:cryo_efficiency", "tab:nu_efficiency"):
    i = next(k for k, l in enumerate(L) if f"\\label{{{lab}}}" in l)
    j = i
    while "\\resizebox" not in L[j]:
        j += 1
    assert j - i < 6, (lab, j - i)
    e = j
    while "\\end{tabular}" not in L[e]:
        e += 1
    assert L[e].strip() == "\\end{tabular}%" and L[e + 1].strip() == "}", L[e:e + 2]
    L[j] = "\\small\\setlength{\\tabcolsep}{6pt}"
    L[e] = "\\end{tabular}"
    L[e + 1] = ""
out = "\n".join(L)
P.write_bytes((out.replace("\n", "\r\n") if crlf else out).encode("utf-8"))
print("ok")
