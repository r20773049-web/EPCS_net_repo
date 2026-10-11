"""Checks RESPONSE_DRAFT_R3.md: every cited tex label exists, placeholder count, 4-dp numbers quoted."""
import re
from pathlib import Path
m = Path(__file__).resolve().parents[1] / "05_manuscript"
d = (m / "RESPONSE_DRAFT_R3.md").read_text(encoding="utf-8")
t = (m / "EPCS_revise3.tex").read_text(encoding="utf-8")
labs = set(re.findall(r"\\label\{([^}]+)\}", t))
used = set(re.findall(r"((?:sec|tab|fig|eq):[A-Za-z0-9_]+)", d))
print("labels cited:", len(used), "missing:", sorted(u for u in used if u not in labs))
print("placeholders:", len(re.findall(r"\[\[RESULT", d)))
print("numbers quoted:", sorted(set(re.findall(r"\b\d\.\d{3,4}\b", d))))
