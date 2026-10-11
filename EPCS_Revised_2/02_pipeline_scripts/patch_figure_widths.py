"""Set the page widths of the four redesigned schematic figures so that text prints at about 6.5-7 pt (agent's size analysis:
printed pt = font px x width(pt) / canvas px). Assertion-guarded exact-match replacement on EPCS_revise3.tex."""
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "05_manuscript" / "EPCS_revise3.tex"
t = p.read_bytes().decode("utf-8")


def rep(old, new):
    global t
    assert t.count(old) == 1, (t.count(old), old[:80])
    t = t.replace(old, new)


# Figure 1 keeps full width; it is no longer height-limited
rep(r"\includegraphics[width=0.997\linewidth,height=0.82\textheight,keepaspectratio]{figures/fig1_pipeline.png}",
    r"\includegraphics[width=0.997\linewidth,height=0.82\textheight,keepaspectratio]{figures/fig1_pipeline.png}")
rep(r"\includegraphics[width=0.460\linewidth]{figures/fig2_pspi.png}",
    r"\includegraphics[width=0.65\linewidth]{figures/fig2_pspi.png}")
rep(r"\includegraphics[width=0.486\linewidth,height=0.55\textheight,keepaspectratio]{figures/fig3_acfb.png}",
    r"\includegraphics[width=0.62\linewidth,height=0.72\textheight,keepaspectratio]{figures/fig3_acfb.png}")
rep(r"\includegraphics[width=0.545\linewidth]{figures/fig4_fsfd.png}",
    r"\includegraphics[width=0.70\linewidth]{figures/fig4_fsfd.png}")
p.write_bytes(t.encode("utf-8"))
print("ok")
