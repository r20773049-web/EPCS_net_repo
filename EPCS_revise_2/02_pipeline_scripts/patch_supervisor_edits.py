"""Apply the supervisor's 23 requests (2026-10-11) to EPCS_revise3.tex.

Assertion-guarded: every replacement must match exactly once; nothing is written unless all assertions pass.
CRLF line endings are preserved (file is normalised to LF in memory and converted back on write).
Text-only processing, no model code, no GPU.  Run once on the pre-edit file (a backup is kept in
05_manuscript/_backups/EPCS_revise3_before_supervisor_edits.bak).  Prints a change log to stdout.
"""
import ctypes
import re
import sys
from pathlib import Path

try:  # BELOW_NORMAL priority (shared machine)
    k32 = ctypes.windll.kernel32
    k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)
except Exception:
    pass

P = Path(__file__).resolve().parents[1] / "05_manuscript" / "EPCS_revise3.tex"
raw = P.read_bytes().decode("utf-8")
assert raw.count("\r\n") == raw.count("\n") and raw.count("\r\n") > 1000, "file is not pure CRLF"
t = raw.replace("\r\n", "\n")
LOG = []


def log(req, msg):
    LOG.append((req, msg))


def lineno(idx):
    return t.count("\n", 0, idx) + 1


def rep(old, new, req=None):
    global t
    assert t.count(old) == 1, (t.count(old), old[:90])
    if req:
        log(req, "line %d: %s ... -> %s ..." % (lineno(t.index(old)), old[:70].replace("\n", " "), new[:70].replace("\n", " ")))
    t = t.replace(old, new)


def between(start, end, new, req=None):
    """Replace text from `start` (unique) to the first `end` after it (inclusive)."""
    global t
    assert t.count(start) == 1, (t.count(start), start[:90])
    s = t.index(start)
    e = t.index(end, s)
    assert e + len(end) - s < 12000, "span suspiciously long"
    if req:
        log(req, "line %d-%d: replaced %d chars (%s ...)" % (lineno(s), lineno(e), e + len(end) - s, start[:60]))
    t = t[:s] + new + t[e + len(end):]


def match_brace(s, i):
    """s[i] == '{'; return index of matching '}' (ignores escaped braces)."""
    assert s[i] == "{"
    d = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            d += 1
        elif c == "}":
            d -= 1
            if d == 0:
                return j
        j += 1
    raise AssertionError("unbalanced")


def setcap(label, new, req=None):
    global t
    key = "\\label{%s}" % label
    assert t.count(key) == 1, (t.count(key), label)
    li = t.index(key)
    ci = t.rfind("\\caption{", 0, li)
    assert ci != -1
    ob = ci + len("\\caption")
    cb = match_brace(t, ob)
    assert t[cb + 1:li].strip() == "", (label, t[cb + 1:li][:50])
    old = t[ob + 1:cb]
    log(req or "11", "%s (line %d) caption: BEFORE: %s || AFTER: %s" % (label, lineno(ci), old, new))
    t = t[:ob + 1] + new + t[cb:]


def klass(first):
    f = first.lower()
    for k, key in ((7, "epcs-net"), (4, "sam-guided"), (5, "bottleneck"), (6, "four scale"), (3, "eu-net3+"),
                   (1, "u-net++"), (2, "u-net3+"), (0, "u-net")):
        if key in f:
            return k
    raise AssertionError(first)


def reorder(label, n):
    global t
    li = t.index("\\label{%s}" % label)
    bi = t.index("\\midrule\n", li) + len("\\midrule\n")
    ei = t.index("\\bottomrule", bi)
    body = t[bi:ei]
    assert "\\midrule" not in body
    rows = [r for r in body.split("\n") if r.strip()]
    assert len(rows) == n and all(r.rstrip().endswith("\\\\") for r in rows), (label, len(rows))
    ks = [klass(r.split("&")[0]) for r in rows]
    assert len(set(ks)) == n, (label, ks)
    new = [r for _, r in sorted(zip(ks, rows))]
    t = t[:bi] + "\n".join(new) + "\n" + t[ei:]
    log("15", "%s (line %d): rows reordered, old key order %s -> sorted (U-Net, U-Net++, U-Net3+, eU-Net3+, SAM-guided, Ref-21 bottleneck, Ref-21 four scales%s)" % (label, lineno(li), ks, ", EPCS-Net last" if n == 8 else ""))


# ---------------------------------------------------------------- preamble / spacing (general)
rep("\\usepackage{array}\n", "\\usepackage{array}\n\\usepackage{colortbl}\n", "12")
rep("\\usepackage{float}\n", "\\usepackage{float}\n\\usepackage{placeins}\n", "spacing")
rep("\\emergencystretch=3em\n",
    "\\emergencystretch=3em\n"
    "\\raggedbottom\n"
    "\\renewcommand{\\topfraction}{0.9}\n"
    "\\renewcommand{\\bottomfraction}{0.8}\n"
    "\\renewcommand{\\textfraction}{0.07}\n"
    "\\renewcommand{\\floatpagefraction}{0.8}\n"
    "\\setcounter{topnumber}{3}\n"
    "\\setcounter{bottomnumber}{2}\n"
    "\\setcounter{totalnumber}{5}\n"
    "\\setlength{\\floatsep}{10pt plus 2pt minus 2pt}\n"
    "\\setlength{\\textfloatsep}{12pt plus 2pt minus 4pt}\n"
    "\\setlength{\\intextsep}{10pt plus 2pt minus 2pt}\n"
    "\\captionsetup{font=small,labelfont=bf,skip=4pt}\n"
    "\\definecolor{hlcell}{HTML}{DCE9F7}\n", "spacing")

nH = t.count("\\begin{table}[H]") + t.count("\\begin{figure}[H]")
t = t.replace("\\begin{table}[H]", "\\begin{table}[!htbp]").replace("\\begin{figure}[H]", "\\begin{figure}[!htbp]")
assert "[H]" not in t
log("spacing", "%d forced [H] placements replaced by [!htbp]" % nH)

# FloatBarrier before every \section{...} except the first, before the starred back-matter and at two logical joints
secs = [m.start() for m in re.finditer(r"(?m)^\\section\{", t)]
assert len(secs) == 6, len(secs)
for s in reversed(secs[1:]):
    t = t[:s] + "\\FloatBarrier\n" + t[s:]
log("spacing", "\\FloatBarrier inserted before the %d section headings after the Introduction" % (len(secs) - 1))
rep("\\subsection{Results --- NuInsSeg}\n", "\\FloatBarrier\n\\subsection{Results --- NuInsSeg}\n", "spacing")
rep("\\subsection{Explainability Results}\n", "\\FloatBarrier\n\\subsection{Explainability Results}\n", "spacing")

# ---------------------------------------------------------------- 1 contributions
rep("The pipeline has not been validated against pathologist annotations or clinical labels.\n\\end{itemize}",
    "The pipeline has not been validated against pathologist annotations or clinical labels.\n"
    "  \\item \\textbf{Held-out test protocol with matched seeds:} Organ- and tissue-stratified splits (459/103/103 images on NuInsSeg; three tissue-stratified folds on CryoNuSeg), checkpoint selection on validation Dice only, a single test evaluation per run, and the same five seeds for every model (15 matched runs per model on CryoNuSeg, five on NuInsSeg), analysed with paired tests (Section~\\ref{sec:protocol}).\n"
    "  \\item \\textbf{Controlled comparison with the closest published SAM-fusion operator:} The X-Gated Fusion Block of \\cite{r21}, re-implemented and verified against the released code, evaluated at the bottleneck and at four scales inside the same backbone, loss, splits and seeds (Section~\\ref{sec:ref21_results}).\n"
    "  \\item \\textbf{Five-seed component ablation with controls:} A one-fold ablation over eleven variants that includes controls for the SAM embedding (random frozen embedding), for PSPI, for ACFB (plain concatenation) and for the normalisation layer, reported with confidence intervals (Section~\\ref{sec:ablation_sec}).\n"
    "  \\item \\textbf{Exploratory phenotyping with stability, confounder and instance-merge analysis:} The Grad-CAM, PCA and HDBSCAN analysis is accompanied by bootstrap stability, source-image association and a measurement of how often connected components merge several nuclei, so that its limits are reported together with its outputs (Section~\\ref{sec:explain}).\n"
    "  \\item \\textbf{Reproducibility:} Parameter counts, idle-GPU latency and peak memory for every model, and released code and checkpoints (Section~\\ref{sec:impl}; Code Availability).\n"
    "\\end{itemize}", "1")

# ---------------------------------------------------------------- abstract (9-10)
rep("MedSAM served only as a sanity check and an oracle-box upper bound.",
    "MedSAM, evaluated frozen, is reported separately from the trained models.", "9-10")

# ---------------------------------------------------------------- OSPT text + Table 2 (3, 4, 21)
rep("Table~\\ref{tab:ospt} quantifies this on the NuInsSeg training partition, and Supplementary Table~\\ref{tab:ospt_supp} lists all 31 organs.",
    "Table~\\ref{tab:ospt} quantifies this on the NuInsSeg training partition.", "21")
setcap("tab:ospt", r"OSPT sampling on the NuInsSeg training partition ($N_{\text{train}} = 459$, $K = 31$ organs), from a simulation of 2{,}000 epochs of the weighted sampler. Exact expectation: 14.81 draws per organ per epoch.", "3")
li = t.index("\\label{tab:ospt}")
bi = t.index("\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lccccc}", li)
ei = t.index("\\end{tabular}%\n}\n", bi) + len("\\end{tabular}%\n}\n")
OSPT_TAB = r"""\small
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.0cm}>{\centering\arraybackslash}p{1.3cm}>{\centering\arraybackslash}p{2.4cm}>{\centering\arraybackslash}p{2.7cm}>{\centering\arraybackslash}p{2.4cm}>{\centering\arraybackslash}p{2.6cm}@{}}
\toprule
\textbf{Organ group} & \textbf{Organs} & \textbf{Raw images per organ} & \textbf{Mean OSPT draws per organ-epoch} & \textbf{Draws per image per epoch} & \textbf{Images undrawn in an epoch} \\
\midrule
Small organs (4--8 images)    & 19 & 4--8   & 14.79 & 2.20 & 12.4\% \\
Large organs (20--33 images)  & 12 & 20--33 & 14.83 & 0.56 & 57.3\% \\
\midrule
All organs                    & 31 & 4--33 (median 8) & 14.62--14.97 & 0.45--3.69 & 29.8\% (organ mean) \\
\bottomrule
\end{tabular}
"""
old_tab = t[bi:ei]
assert "Small organs (4--8 images)" in old_tab and "29.8\\% (organ mean)" in old_tab
t = t[:bi] + OSPT_TAB + t[ei:]
log("4", "tab:ospt: resizebox removed; \\small + 6 fixed-width p-columns (total 15.4 cm + gaps, below 17.2 cm text width), wrapped headers, no scaling; pointer to Supplementary Table S1 removed from caption and from the text before the table")

# ---------------------------------------------------------------- 5 Figure 2
rep("\\includegraphics[width=0.65\\linewidth]{figures/fig2_pspi.png}", "\\includegraphics[width=0.92\\linewidth]{figures/fig2_pspi.png}", "5")

# ---------------------------------------------------------------- 23 Step 5 of ACFB (+ keep BN/GN clarification)
rep("written in Equation~(\\ref{eq:acfb_out}). Figure~\\ref{fig:acfb} shows the full ACFB structure:",
    "written in Equation~(\\ref{eq:acfb_out}). Group Normalisation (8 groups) is used only inside ACFB. Figure~\\ref{fig:acfb} shows the full ACFB structure:", "23")
between("\nGroup Normalisation with 8 groups is used at this point, inside the fusion block",
        "is the form used in the released code and for every number reported here.\n",
        "", "23")
rep("and the consequences of batch size 1 for the encoder's BN are discussed in Section~\\ref{sec:acfb_sec}.",
    "and Batch Normalisation at batch size 1 is examined by control D of the ablation (Section~\\ref{sec:ablation_sec}).", "23")
rep("a point examined by a single-fold control in Section~\\ref{sec:acfb_sec}.",
    "a point examined by a single-fold control in Section~\\ref{sec:ablation_sec}.", "23")
rep("(Group Normalisation in the encoder blocks; motivation in Section~\\ref{sec:acfb_sec})",
    "(Group Normalisation in the DoubleConv blocks, a check on Batch Normalisation at batch size 1)", "23")

# ---------------------------------------------------------------- 6 Figure Provenance
between("\\subsection{Figure Provenance}", "are the released dataset patches.\n", "", "6")
assert "figprov" not in t

# ---------------------------------------------------------------- 9-10 MedSAM
between("\\subsection{MedSAM: Sanity Check and Oracle-Box Upper Bound}", "is not included in the ranked comparison of Table~\\ref{tab:cryo}.",
        "\\subsection{MedSAM Evaluation}\n\\label{sec:medsam}\n\n"
        "MedSAM \\cite{r19} segments the object inside a supplied bounding box. We ran the official checkpoint frozen, with no fine-tuning and no threshold tuning, on the same test partitions as the trained models, under two box protocols. In the first, one box covers the whole image and the mask with the highest predicted IoU of three is kept. In the second, the prompt is one box per ground-truth connected component (on average 166.8, 184.3 and 159.2 boxes per image on CryoNuSeg folds 1 to 3 and 43.5 on NuInsSeg), so this protocol uses ground-truth localisation that is unavailable at test time and its numbers are not comparable with those of the trained models. Table~\\ref{tab:medsam} gives both results; MedSAM is not included in the ranked comparison of Table~\\ref{tab:cryo}.", "9-10")
li = t.index("\\label{tab:medsam}")
ei = t.index("\\end{table}", li)
MEDSAM_TAB = r"""\label{tab:medsam}
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lccccc}
\toprule
 & & \multicolumn{2}{c}{\textbf{Whole-image box}} & \multicolumn{2}{c}{\textbf{One box per ground-truth component}} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
\textbf{Test set} & \textbf{Images} & \textbf{Dice} & \textbf{IoU} & \textbf{Dice} & \textbf{IoU} \\
\midrule
CryoNuSeg fold 1 & 10  & 0.1994 $\pm$ 0.1162 & 0.1151 & 0.7130 $\pm$ 0.1247 & 0.5652 \\
CryoNuSeg fold 2 & 10  & 0.1505 $\pm$ 0.1040 & 0.0845 & 0.6740 $\pm$ 0.0916 & 0.5144 \\
CryoNuSeg fold 3 & 10  & 0.2737 $\pm$ 0.1535 & 0.1673 & 0.7153 $\pm$ 0.0714 & 0.5609 \\
NuInsSeg         & 103 & 0.1475 $\pm$ 0.1192 & 0.0845 & 0.7484 $\pm$ 0.1041 & 0.6072 \\
\bottomrule
\end{tabular}
"""
old = t[li:ei]
for needle in ("0.1994 $\\pm$ 0.1162 & 0.1151 & 0.7130 $\\pm$ 0.1247 & 0.5652", "0.2737 $\\pm$ 0.1535 & 0.1673 & 0.7153 $\\pm$ 0.0714 & 0.5609",
               "0.1475 $\\pm$ 0.1192 & 0.0845 & 0.7484 $\\pm$ 0.1041 & 0.6072", "0.1505 $\\pm$ 0.1040 & 0.0845 & 0.6740 $\\pm$ 0.0916 & 0.5144"):
    assert needle in old, needle
t = t[:li] + MEDSAM_TAB + t[ei:]
setcap("tab:medsam", r"MedSAM, frozen (no training, hence no seeds), on the test partitions. Box protocols: one whole-image box, or one box per ground-truth connected component (ground-truth localisation). Dice is mean $\pm$ SD over test images.", "9-11")
between("The whole-box setting scores Dice 0.1505 to 0.2737", "because the trained models receive no localisation at all.",
        "With the whole-image box MedSAM scores Dice 0.1505 to 0.2737 on the CryoNuSeg folds and 0.1475 on NuInsSeg, as expected when a single box has to cover dozens of small nuclei. With one box per ground-truth component it reaches Dice 0.6740 to 0.7153 on the CryoNuSeg folds and 0.7484 on NuInsSeg. These values show what a prompted model reaches when the nuclei are already located; we do not compare them with the trained models, which receive no localisation at all.", "9-10")
rep("\\item \\textbf{MedSAM \\cite{r19} (not a competitor):} The official MedSAM checkpoint evaluated frozen and zero-shot, with a whole-image box as a sanity check and with ground-truth-derived boxes as an oracle upper bound (Section~\\ref{sec:medsam}). It is not part of the ranked comparison.",
    "\\item \\textbf{MedSAM \\cite{r19}:} The official MedSAM checkpoint evaluated frozen and zero-shot, with a whole-image box and with one box per ground-truth connected component (Section~\\ref{sec:medsam}). It is reported separately from the ranked comparison.", "9-10")
rep("MedSAM is deliberately not ranked: it is an externally trained promptable model whose whole-image-box result is a sanity check and whose oracle-box result uses ground-truth localisation, so neither is a like-for-like competitor (Section~\\ref{sec:medsam}).",
    "MedSAM is evaluated separately and not ranked: it is an externally trained promptable model, and one of its two box protocols uses ground-truth localisation (Section~\\ref{sec:medsam}).", "9-10")
rep("MedSAM is a sanity check and an oracle-box upper bound, not a competitor (Section~\\ref{sec:medsam}).",
    "MedSAM is evaluated frozen and separately from the trained models, and one of its two box protocols uses ground-truth localisation (Section~\\ref{sec:medsam}).", "9-10")
rep("\\item \\textbf{MedSAM oracle-box result.} The MedSAM result prompted with boxes derived from ground-truth masks is an upper bound, because it uses ground-truth localisation that is unavailable at test time. It is not a competing method.",
    "\\item \\textbf{MedSAM with ground-truth boxes.} The MedSAM result prompted with one box per ground-truth connected component uses ground-truth localisation that is unavailable at test time, so it is not comparable with the trained models.", "9-10")

rep("scoring and the MedSAM oracle result are added", "scoring and the MedSAM ground-truth-box result are added", "9-10")

# ---------------------------------------------------------------- 7-8 Table 6
setcap("tab:seeds", r"EPCS-Net test Dice of every run, each at its best-validation-Dice checkpoint. CryoNuSeg: three folds $\times$ five seeds, each cell the mean over the fold's 10 test images. NuInsSeg: five seeds, 103 test images. Summary rows: mean $\pm$ SD over runs.", "7")
rep("NuInsSeg (103 test images) & 0.6719", "NuInsSeg (fixed split) & 0.6719", "8")

# ---------------------------------------------------------------- 13 remove Note paragraphs
t, nn = re.subn(r"\\par\\smallskip\n\{\\footnotesize\\noindent\\textit\{Note\.\}[^\n]*\}\n(?=\\end\{table\})", "", t)
assert nn == 3, nn
assert "textit{Note" not in t
log("13", "3 'Note.' paragraphs removed (tab:seeds, tab:cryo, tab:nu); definitions moved into the short captions")

# ---------------------------------------------------------------- 15 row order
reorder("tab:cryo", 8)
reorder("tab:nu", 8)
reorder("tab:paired", 7)
reorder("tab:nu_paired", 7)
reorder("tab:cryo_perfold", 8)
reorder("tab:cryo_allmetrics", 8)
reorder("tab:nu_allmetrics", 8)

# ---------------------------------------------------------------- 12 highlight in tab:cryo_perfold
rep("EPCS-Net (proposed) & 0.7827 & 0.7851 & 0.8021 & 0.7899 \\\\",
    "EPCS-Net (proposed) & 0.7827 & 0.7851 & 0.8021 & \\cellcolor{hlcell}\\textbf{0.7899} \\\\", "12")

# ---------------------------------------------------------------- 16 all-metrics tables: readable, aligned
for lab in ("tab:cryo_allmetrics", "tab:nu_allmetrics"):
    li = t.index("\\label{%s}" % lab)
    ei = t.index("\\end{table}", li)
    seg = t[li:ei]
    assert seg.count("\\resizebox{\\textwidth}{!}{%") == 1
    seg = seg.replace("\\resizebox{\\textwidth}{!}{%", "\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\textwidth}{!}{%")
    assert seg.count("Ref-21 X-Gated fusion, bottleneck") == 1 and seg.count("Ref-21 X-Gated fusion, four scales") == 1
    seg = seg.replace("Ref-21 X-Gated fusion, bottleneck", "Ref-21 bottleneck").replace("Ref-21 X-Gated fusion, four scales", "Ref-21 four scales")
    t = t[:li] + seg + t[ei:]
log("16", "tab:cryo_allmetrics (and tab:nu_allmetrics): \\footnotesize (10 pt) + tabcolsep 3pt inside resizebox to text width; shortened Ref-21 labels; estimated final size about 9 pt")

# ---------------------------------------------------------------- 17 Figure 11 same options as Figure 6
rep("\\includegraphics[width=0.974\\linewidth,height=0.82\\textheight,keepaspectratio]{figures_r3/r3_qual_nuinsseg.png}",
    "\\includegraphics[width=1.0\\linewidth,height=0.82\\textheight,keepaspectratio]{figures_r3/r3_qual_nuinsseg.png}", "17")

# ---------------------------------------------------------------- 18 Grad-CAM figures
rep("\\includegraphics[width=0.62\\linewidth,height=0.85\\textheight,keepaspectratio]{figures/fig13_gradcam_best_and_failure_cases_nuinsseg.png}",
    "\\includegraphics[width=0.97\\linewidth,height=0.80\\textheight,keepaspectratio]{figures_r3/r3_gradcam_table_nuinsseg.png}", "18")
rep("\\includegraphics[width=0.62\\linewidth,height=0.85\\textheight,keepaspectratio]{figures/fig13_gradcam_best_and_failure_cases_cryonuseg.png}",
    "\\includegraphics[width=0.97\\linewidth,height=0.80\\textheight,keepaspectratio]{figures_r3/r3_gradcam_table_cryonuseg.png}", "18")

# ---------------------------------------------------------------- 20 explainability prose
between("The DPNP pipeline runs identically on both datasets after segmentation", "folding them into one combined figure.",
        "The DPNP pipeline runs identically on both datasets after segmentation (same code, thresholds and parameters), yet its numbers differ visibly, so we report them separately.", "20")
between("We threshold the resulting CAM at 0.5", "not as independent proof of interpretability.",
        "We threshold the CAM at 0.5 and compute its IoU against the ground-truth nuclei mask, on the held-out test images of one EPCS-Net run per dataset (CryoNuSeg: fold 3, seed 2024, 10 images; NuInsSeg: seed 2024, 103 images). Mean overlap is 0.2264 on CryoNuSeg (sample SD 0.0821, median 0.2087, range 0.1550 to 0.4256) and 0.3155 on NuInsSeg (SD 0.1726, median 0.3074, range 0.0165 to 0.7003). Each value comes from a single run, so there is no seed-to-seed estimate; we present the overlap as a plausibility check on model behaviour, not as proof of interpretability.", "20")
between("Figures~\\ref{fig:gradcam} and \\ref{fig:gradcam_cryo} show the three highest-overlap", "the panels are an inspection aid and not an explanation.",
        "Figures~\\ref{fig:gradcam} and \\ref{fig:gradcam_cryo} show the three highest-overlap and four lowest-overlap test images per dataset. CAM overlap measures localisation, not segmentation quality. On CryoNuSeg the four lowest-overlap images (overlap 0.155 to 0.182) still reach Dice 0.691 to 0.836 (mean 0.768, against 0.828 for the three highest-overlap images). On NuInsSeg the four lowest-overlap images (overlap 0.016 to 0.027; three mouse fat images and one mouse heart image) have Dice 0.100 to 0.341 (mean 0.228, against 0.716), so poor localisation and poor segmentation coincide there. We did not investigate what draws attention away from nuclei; the panels are an inspection aid, not an explanation.", "20")
between("Following segmentation, we extract connected components", "is presumably affected in the same way.",
        "Following segmentation, connected components of the binary mask are extracted as a post-hoc step to obtain regions for feature extraction; this is not evaluated as instance segmentation. Touching nuclei merge into one component, so the outputs are exploratory connected-region features, not nucleus features. Against the NuInsSeg ground-truth instances (103 test images, seed-2024 run), 5,207 nuclei correspond to 3,614 components (count ratio 0.694): on average per image 18.0\\% of components merge two or more nuclei (22.3\\% pooled, 805 of 3,614), 62.7\\% match exactly one nucleus and 5.8\\% of nuclei are split. Merging inflates the features: the mean component area is 1.220 times the mean ground-truth nucleus area ($p = 9.9\\times10^{-5}$, Mann--Whitney). Merging could not be measured on CryoNuSeg, which ships binary masks only; the same step yields 4,973 components from its 30 images and is presumably affected in the same way.", "20")
between("We do not assign descriptive names to the components.", "the CryoNuSeg components were examined only for PC1).",
        "We do not assign names to the components: about 99\\% of each component's loading mass falls on the 512 ResNet18 embedding dimensions (98.8\\%, 99.2\\% and 99.0\\% for PC1 to PC3 on NuInsSeg; 99.2\\%, 99.3\\% and 99.2\\% on CryoNuSeg). The only interpretation we retain is that PC1 tracks region area (Spearman $\\rho = 0.865$ on NuInsSeg and $0.805$ on CryoNuSeg). PC2 and PC3 have only moderate correlates (NuInsSeg run: area $0.541$ and $a^*$ $0.454$ for PC2; intensity variance $0.503$ and $a^*$ $-0.350$ for PC3), which does not support naming them.", "20")
between("We identify nuclear phenotype clusters using HDBSCAN", "We did not establish why.",
        "We run HDBSCAN on the PCA-reduced feature space (up to 40 dimensions) under the prespecified rule \\texttt{min\\_cluster\\_size} $= \\max(5, N/10)$ for $N$ instances, with \\texttt{min\\_samples} of 5 and Euclidean distance; instances labelled $-1$ are outliers. The minimum cluster size is 361 on the NuInsSeg test predictions (3,614 components, 103 images, EPCS-Net seed-2024 run) and 497 on the CryoNuSeg test predictions (4,973 components, 30 images); on both, HDBSCAN finds zero clusters and labels every component an outlier (100\\%). We did not establish why.", "20")
between("Zero clusters could mean either that the feature space is uninformative", "and are exploratory throughout.",
        "Zero clusters could mean an uninformative feature space or a rule that is too strict for these pool sizes, so we ran an exploratory, post hoc sensitivity check with a fixed \\texttt{min\\_cluster\\_size} of 5; the prespecified rule remains the primary one and gives 0 clusters. The check yields two clusters on each dataset (NuInsSeg: 1,947 and 5 components, 46.0\\% outliers, silhouette 0.321; CryoNuSeg: 1,534 and 5 components, 69.1\\% outliers, silhouette 0.251). We do not read these as real structure, because they are neither stable nor free of confounders; a silhouette score measures geometric separation only and is not biological validation. Under 25 bootstrap subsamples of 80\\% of the components the adjusted Rand index is $0.539 \\pm 0.310$ on NuInsSeg (2 to 14 clusters; Figure~\\ref{fig:stability_nu}) and $0.412 \\pm 0.374$ on CryoNuSeg (2 to 4 clusters; Figure~\\ref{fig:stability_cryo}). Cluster assignment is associated with the source image (NuInsSeg: Cram\\'er's $V = 0.265$, $p = 0.0093$; CryoNuSeg: $V = 0.185$, $p = 0.0048$), whereas no association with organ ($p = 0.98$), species ($p = 0.21$) or tissue type ($p = 0.28$) was detected; stain variation was not assessed, so a stain effect is not excluded. The prespecified rule finds nothing, and the structure found by the sensitivity check is a two-cluster split with one five-member cluster that is unstable under resampling and tied to the source image, so the clustering supports no phenotypic claim. Both analyses use EPCS-Net seed-2024 predictions and are exploratory throughout.", "20")
between("Components labelled $-1$ lie outside the dense regions", "would be needed first (Section~\\ref{sec:future}).",
        "Components labelled $-1$ lie outside the dense regions of the embedding; this is a statistical property of the feature space and carries no diagnostic meaning. Figure~\\ref{fig:hdbscan} plots the sensitivity-check assignment in PC1--PC2 space for the NuInsSeg test predictions (Figure~\\ref{fig:hdbscan_cryo} for CryoNuSeg), for transparency only. None of this has been checked against pathologist annotations or clinical labels, which we regard as a prerequisite for any diagnostic weight. Spatial statistics such as SpatialQPFs \\cite{r72} and entropy measures \\cite{r73} presuppose reliable single-nucleus identities, which the connected-component step does not provide (Section~\\ref{sec:future}).", "20")

# ---------------------------------------------------------------- 21 / 22 Supplementary and Acknowledgement
s = t.index("\\section*{Supplementary Information}")
e = t.index("\\section*{Authors' Contribution and CRediT}")
removed = t[s:e]
assert "\\begin{longtable}" in removed and "\\end{longtable}" in removed and "\\section*{Acknowledgement}" in removed
assert removed.count("\\section*") == 2 and "\\label{tab:ospt_supp}" in removed
log("21", "Supplementary Information section + Table S1 (longtable, label tab:ospt_supp) deleted: old lines %d-%d" % (lineno(s), lineno(t.index("\\section*{Acknowledgement}"))-1))
log("22", "Acknowledgement section deleted: old lines %d-%d" % (lineno(t.index("\\section*{Acknowledgement}")), lineno(e)-1))
t = t[:s] + "% -- Authors' Contribution --\n" + t[e:]
assert "ospt_supp" not in t and "Acknowledg" not in t and "Supplementary" not in t

# ---------------------------------------------------------------- 11, 14, 19 captions
C = {
 "tab:datasets": r"Datasets and partitions. CryoNuSeg: three-fold cross-validation (sizes per fold); NuInsSeg: one organ-stratified split.",
 "fig:pipeline": r"EPCS-Net pipeline: frozen SAM branch and DoubleConv local encoder joined by PSPI and ACFB, full-scale decoder, FTAS loss and the DPNP analysis.",
 "tab:encoder": r"Local encoder stages: output size and parameters (total 4{,}687{,}296).",
 "fig:pspi": r"Pyramid-Scale Prior Injection: the SAM embedding is resized to all four encoder scales and fused by ACFB at each.",
 "fig:acfb": r"Adaptive Cross-stream Fusion Block: projection, bidirectional gating (BDSG), dual-kernel refinement (DKSR), squeeze-and-excitation, and residual (post-DKSR tensor) with Group Normalisation.",
 "tab:acfb": r"ACFB hyperparameters and parameters per encoder scale (total 13{,}659{,}760).",
 "fig:fsfd": r"Full-Scale Fused Decoder with multi-scale skip connections (960 channels at decoder scale 3).",
 "tab:hyperparams": r"Training hyperparameters, identical for both datasets.",
 "fig:seedstability": r"Test Dice of each EPCS-Net run. Left: CryoNuSeg, three folds $\times$ five seeds (15 runs; bars: fold means). Right: NuInsSeg, five seeds on one split.",
 "tab:cryo": r"Test performance on CryoNuSeg, mean over 15 matched runs per model (three folds $\times$ five seeds); Dice and IoU as mean $\pm$ SD over runs. Loss: training loss.",
 "tab:cryo_efficiency": r"Computational cost: total and trainable parameters, latency and peak GPU memory of one $1{\times}3{\times}256{\times}256$ FP32 forward pass on an idle GPU (20 warm-up, 100 timed passes; Section~\ref{sec:impl}). Same for both datasets.",
 "tab:paired": r"CryoNuSeg test: EPCS-Net minus each comparator in Dice. Columns 2--5: 15 matched runs (Holm-corrected $p$; pairs share test images). Last two: fold-level test over three folds (uncorrected $t$-test, $df=2$).",
 "tab:cryo_perfold": r"CryoNuSeg test Dice per fold (10 test images; mean over five seeds). Last column: mean over all 15 runs.",
 "tab:cryo_allmetrics": r"All six pixel-level measures on CryoNuSeg: mean $\pm$ SD over the 15 runs of each model.",
 "fig:cryo_qual": r"Qualitative results on CryoNuSeg (fold 1, seed 42): highest, median and lowest per-image Dice among the 10 test images. Overlay: green true positive, orange false positive, blue false negative.",
 "tab:nu": r"Test performance on NuInsSeg (103 test images), mean over five seeds per model; Dice and IoU as mean $\pm$ SD over seeds. Loss: training loss.",
 "tab:nu_allmetrics": r"All six pixel-level measures on NuInsSeg: mean $\pm$ SD over the five seeds of each model.",
 "tab:nu_efficiency": r"Computational cost on an idle GPU; identical to Table~\ref{tab:cryo_efficiency} because it does not depend on the dataset.",
 "fig:headline": r"Test Dice per model, mean $\pm$ SD over runs (large markers) and individual runs (dots): CryoNuSeg, 15 matched runs; NuInsSeg, five seeds. Six of the eight trained models are shown; the MedSAM rows below the dashed line are not ranked (Table~\ref{tab:medsam}).",
 "tab:nu_paired": r"NuInsSeg test (103 images): EPCS-Net minus each comparator in Dice. Columns 2--5: five matched seeds (Holm-corrected $p$). Last two: image-level paired bootstrap over the 103 test images.",
 "fig:organ_cryo": r"CryoNuSeg test Dice per tissue, mean $\pm$ SD over runs (EPCS-Net 15 runs; SAM-guided baseline; U-Net); descriptive only.",
 "fig:organ_nu": r"NuInsSeg test Dice per organ, mean $\pm$ SD over five seeds (EPCS-Net, SAM-guided baseline, U-Net); descriptive only.",
 "fig:trainingcurves": r"Validation Dice per epoch, mean $\pm$ SD over runs (CryoNuSeg 15 runs, NuInsSeg five seeds) for six models; stars mark the mean best epoch. Validation, not test, Dice.",
 "fig:nu_qual": r"Qualitative results on NuInsSeg (seed 42): highest, median and lowest per-image Dice among the 103 test images. Overlay: green true positive, orange false positive, blue false negative.",
 "fig:gradcam": r"Grad-CAM on NuInsSeg test images (EPCS-Net, seed 2024): three highest- and four lowest-overlap cases.",
 "fig:gradcam_cryo": r"Grad-CAM on CryoNuSeg test images (fold 3, seed 2024): three highest- and four lowest-overlap cases.",
 "tab:pca": r"PCA explained variance of the phenotyping features (EPCS-Net seed-2024 test predictions; NuInsSeg 3,614 components, CryoNuSeg 4,973).",
 "fig:pca": r"PCA colour maps of the first three components on NuInsSeg test predictions (EPCS-Net seed 2024); no verified interpretation beyond PC1 tracking area.",
 "fig:pca_cryo": r"As Figure~\ref{fig:pca}, for the CryoNuSeg test predictions (30 images).",
 "fig:hdbscan": r"HDBSCAN clusters (fixed \texttt{min\_cluster\_size}=5 sensitivity check) in PC1--PC2 space, NuInsSeg test predictions; grey: outliers. Transparency only, not evidence of phenotypes.",
 "fig:hdbscan_cryo": r"As Figure~\ref{fig:hdbscan}, for the CryoNuSeg test predictions (30 images).",
 "fig:stability_nu": r"Stability and confounder analysis of the sensitivity-check clusters, NuInsSeg test predictions.",
 "tab:ablation": r"Five-seed ablation on CryoNuSeg fold 1 (10 test images): test Dice and IoU, mean $\pm$ SD over five seeds. One-fold result, not a causal attribution.",
 "fig:ablation": r"Five-seed ablation on CryoNuSeg fold 1: test Dice per variant, mean $\pm$ SD over seeds (dots: seeds; marker shape: loss).",
}
for lab, txt in C.items():
    setcap(lab, txt, "14" if lab in ("tab:cryo_efficiency", "tab:nu_efficiency") else ("19" if lab.startswith("fig:") else "11"))
# fig:stability_cryo is already one line; leave as is
assert "\\label{tab:cryo_efficiency}" in t

# ---------------------------------------------------------------- final sanity
for bad in ("sanity check", "oracle", "not a competitor", "Oracle", "upper bound", "Sanity", "[H]", "Provenance", "Supplementary", "Acknowledg"):
    assert bad not in t, (bad, t[max(0, t.index(bad) - 150):t.index(bad) + 150] if bad in t else "")
assert t.count("\\begin{table}") == t.count("\\end{table}") and t.count("\\begin{figure}") == t.count("\\end{figure}")

out = t.replace("\n", "\r\n")
P.write_bytes(out.encode("utf-8"))
print("OK: wrote", P, "(%d -> %d lines)" % (raw.count("\n"), t.count("\n")))
for req, msg in LOG:
    print("[%s] %s" % (req, msg))
