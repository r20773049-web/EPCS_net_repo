"""Static checks on EPCS_revise3.tex (no LaTeX toolchain on this machine). Usage: python check_tex.py"""
import re
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "05_manuscript" / "EPCS_revise3.tex"
t = p.read_text(encoding="utf-8")
body = re.sub(r"(?<!\\)%.*", "", t)
esc = lambda s: s.replace("\\{", "").replace("\\}", "")
print("braces net:", esc(body).count("{") - esc(body).count("}"))
for env in ["table", "figure", "equation", "itemize", "tabular", "enumerate", "document", "thebibliography"]:
    b, e = body.count("\\begin{%s}" % env), body.count("\\end{%s}" % env)
    print(f"  {env}: {b}/{e}", "OK" if b == e else "MISMATCH")
labels = re.findall(r"\\label\{([^}]+)\}", t)
dups = {x for x in labels if labels.count(x) > 1}
refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", t))
print("duplicate labels:", dups or "none", "| dangling refs:", sorted(refs - set(labels)) or "none")
cites = {k.strip() for m in re.findall(r"\\cite\{([^}]*)\}", t) for k in m.split(",")}
bibs = set(re.findall(r"\\bibitem\{([^}]*)\}", t))
print("cites without bibitem:", sorted(cites - bibs) or "none", "| uncited bibitems:", sorted(bibs - cites) or "none")
print("repeated words:", sorted(set(re.findall(r"\b(\w{4,}) \1\b", body)))[:10] or "none")
print("TODO markers:", [(m.group(0), t[:m.start()].count("\n") + 1) for m in re.finditer(r"%%TODO-[A-Z0-9-]+", t)])
print("stale phrases:", [s for s in ["earns its place", "property of the architecture", "16×16", "16$\\times$16", "undetected cancerous"] if s in t] or "none")
