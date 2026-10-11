"""Make the Figure Provenance sentence accurate for the current schematics (composed from explicit coordinates by a script,
stored as editable Excalidraw files). If the author later redraws them by hand, change the sentence back."""
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "05_manuscript" / "EPCS_revise3.tex"
t = p.read_bytes().decode("utf-8")
old = r"are schematics drawn in Excalidraw. All other figures"
new = (r"are vector schematics composed from explicit box and arrow coordinates by a script in the code archive, "
       r"stored as editable Excalidraw documents and rendered to image files. All other figures")
assert t.count(old) == 1, t.count(old)
p.write_bytes(t.replace(old, new).encode("utf-8"))
print("ok")
