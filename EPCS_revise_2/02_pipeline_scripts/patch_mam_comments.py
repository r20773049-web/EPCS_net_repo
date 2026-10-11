"""Supervisor (mam) minor changes, round 2. Assertion-guarded, run once."""
import re
from pathlib import Path
P = Path(r"<PROJECT_ROOT>\05_manuscript\EPCS_revise3.tex")
raw = P.read_bytes().decode("utf-8")
crlf = "\r\n" in raw
t = raw.replace("\r\n", "\n")


def rep(a, b, n=1):
    global t
    assert t.count(a) == n, (t.count(a), a[:80])
    t = t.replace(a, b)


# Sec 4.2 table pointer
rep("reported alongside Dice and IoU in Tables~\\ref{tab:cryo} and \\ref{tab:nu}, so that",
    "reported in Tables~\\ref{tab:cryo_allmetrics} and \\ref{tab:nu_allmetrics} (Dice and IoU in Tables~\\ref{tab:cryo} and \\ref{tab:nu}), so that")
# eUnet3+ -> local-encoder
rep("it receives local eUnet3+ features $X_L^k", "it receives local-encoder (DoubleConv) features $X_L^k")
rep("and eUnet3+ features are symmetrically projected", "and local-encoder features are symmetrically projected")
# PSPI overclaim
rep("The PSPI strategy ensures that the benefits of foundation model pretraining permeate the entire encoder hierarchy rather than being restricted to a single bottleneck fusion point.",
    "This is designed to let SAM context reach every encoder scale; the ablation (Section~\\ref{sec:ablation_sec}) is not conclusive on its benefit.")
rep("PSPI is the first key architectural contribution of EPCS-Net.", "PSPI is the first architectural component of EPCS-Net.")
# 5x5 branch
rep("Adding the two branches together gives a richer representation than either alone, at no extra multiplicative cost.",
    "The $5{\\times}5$ branch adds computation, and its gain was not significant in the ablation.")
# decoder
rep("ensuring that global SAM contextual priors propagate through the entire reconstruction pathway", "so that SAM-enriched features reach every decoder node")
# intro
rep("demonstrating exceptional performance in medical imaging applications", "demonstrating promising results in medical imaging applications")
rep("Secondly, interpretability tools are often absent, despite the necessity for transparency in AI decision-making within therapeutic contexts \\cite{r22}. Clinicians are unprepared to integrate black-box predictions into diagnostic processes where errors directly impact patient outcomes \\cite{r23}.",
    "Secondly, interpretability tools are often absent, although transparency is important for medical AI \\cite{r22,r23}.")
rep("A novel multi-scale SAM integration strategy", "A multi-scale SAM integration strategy")
rep("A novel fusion module employing bidirectional dual-stream gating",
    "A fusion module, modified from the X-Gated Fusion Block of \\cite{r21}, employing bidirectional dual-stream gating")
# captions
rep("Six of the eight trained models are shown; the MedSAM rows below the dashed line are not ranked (Table~\\ref{tab:medsam}).",
    "The two Reference-21 variants are left out here (values in Tables~\\ref{tab:cryo}, \\ref{tab:cryo_perfold} and \\ref{tab:nu}); the MedSAM rows below the dashed line are not ranked (Table~\\ref{tab:medsam}).")
rep("for six models; stars mark the mean best epoch.",
    "for six models (the two Reference-21 variants are left out; see Tables~\\ref{tab:cryo} and \\ref{tab:nu}); stars mark the mean best epoch.")
# author biographies
i = t.index("% ── Author Biographies")
j = t.index("\\begin{thebibliography}")
t = t[:i] + t[j:]
assert "Author Biographies" not in t and "principal architect" not in t
out = t.replace("\n", "\r\n") if crlf else t
P.write_bytes(out.encode("utf-8"))
print("ok")
