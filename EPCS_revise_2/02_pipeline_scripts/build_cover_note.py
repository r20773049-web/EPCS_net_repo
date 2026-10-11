"""Build SEND_TO_MAM/Draft_Cover_Note.pdf + .md. Numbers are read from the csv files.
Run: python build_cover_note.py (CPU only)"""
import os, zipfile
import pandas as pd
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
AN = os.path.join(ROOT, "07_revision3", "04_results", "analysis")
OUT = os.path.join(ROOT, "SEND_TO_MAM")
os.makedirs(OUT, exist_ok=True)
SUBID = "75a92866-8513-48b6-9941-b16b3a741de2"
TITLE = ("EPCS-Net: SAM-Guided Multi-Scale Semantic Nuclei Segmentation and Exploratory "
         "Phenotypic Profiling for Pan-Cancer Analysis")

agg = pd.read_csv(os.path.join(AN, "aggregated_by_model.csv"))
pt = pd.read_csv(os.path.join(AN, "paired_significance_tests.csv"))
ft = pd.read_csv(os.path.join(AN, "fold_averaged_tests.csv"))
order = [("epcs_net", "EPCS-Net"), ("sam_unet", "SAM-guided baseline"),
         ("ref21_bottleneck", "Reference-21, bottleneck"), ("ref21_pyramid", "Reference-21, four scales"),
         ("eunet3plus", "eU-Net3+"), ("unet", "U-Net"), ("unet3plus", "U-Net3+"), ("unetpp", "U-Net++")]


def cell(ds, m):
    r = agg[(agg.dataset == ds) & (agg.model == m)].iloc[0]
    return "%.3f ± %.3f" % (r.test_dice_mean, r.test_dice_sd), int(r.n_runs)


rows = [["Model", "CryoNuSeg (15 runs)", "NuInsSeg (5 runs)"]]
for m, n in order:
    c, nc = cell("cryonuseg", m)
    u, nu = cell("nuinsseg", m)
    assert nc == 15 and nu == 5
    rows.append([n, c, u])


def prow(ds):
    return pt[(pt.dataset == ds) & (pt.family == "comparator") & (pt.comparison == "epcs_net vs sam_unet")
              & (pt.metric == "dice")].iloc[0]


def stat(ds):
    r = prow(ds)
    return "%+.3f (95%% CI %+.3f to %+.3f), Holm p = %.3f, n = %d pairs" % (
        r.mean_difference, r.ci95_low, r.ci95_high, r.t_p_holm, r.n_pairs)


sc, sn = stat("cryonuseg"), stat("nuinsseg")
fr = ft[(ft.dataset == "cryonuseg") & (ft.baseline == "sam_unet")].iloc[0]
folds = "ahead in %d of %d folds (fold-averaged difference %+.3f)" % (
    fr.folds_proposed_wins, fr.n_folds, fr.mean_difference)
stat_lines = ["CryoNuSeg: " + sc + "; " + folds + ".", "NuInsSeg: " + sn + "."]

z = zipfile.ZipFile(os.path.join(ROOT, "PAPER_OVERLEAF.zip"))
names = [n for n in z.namelist() if not n.endswith("/")]
figs = [n for n in names if n.startswith("figures/") and n.endswith(".png")]
secs = [n for n in names if n.startswith("sections/")]
files_txt = [
    "Top level: main.tex, references.tex, README_OVERLEAF.md, CHANGELOG.md",
    "sections/ (%d files): 01 introduction to 06 conclusion, 07 supplementary information, then back matter 08 to 17 "
    "(acknowledgement, contributions, funding, declarations, data and code availability, biographies)" % len(secs),
    "figures/ (%d PNG files plus README.md): pipeline and module diagrams (fig1 to fig4), headline comparison, "
    "ablation, per-organ, qualitative, training curves, seed/fold stability, phenotyping (PCA, HDBSCAN, confounds), "
    "Grad-CAM best and failure cases" % len(figs),
    "Total: %d files. The response letter is a separate PDF and is not in the zip." % len(names)]

intro = ["Paper: " + TITLE,
         "Submission ID: " + SUBID + ". Round 3: minor revision requested; four reviewers."]
what = ["This is a DRAFT of the revised manuscript as an Overleaf project: main.tex, the section files, "
        "references.tex and the figures. The point-by-point response letter is sent separately.",
        "To open: Overleaf > New Project > Upload Project > select PAPER_OVERLEAF.zip. Set the main document to "
        "main.tex and the compiler to pdfLaTeX."]
done = [
    "Model selection and early stopping now use the validation split; all reported numbers come from a held-out test split.",
    "Every model was trained with five matched seeds. On CryoNuSeg, three tissue-stratified folds were added (15 runs per model).",
    "The ablation was repeated with five seeds and includes the requested controls (A to D).",
    "The reference-21 benchmark was added in two variants (bottleneck and four scales).",
    "MedSAM was added as a sanity check, together with an oracle-box analysis.",
    "Inference latency was re-measured on an idle GPU.",
    "Grad-CAM panels showing failure cases were added.",
    "Phenotyping is now described as exploratory, with stability and confound checks.",
    "The methods text and the reference list were corrected against the code and publisher records.",
    "Figures were redrawn (final redraws are still open, see section 5)."]
cav = [
    "The lead over the SAM-guided baseline is small on CryoNuSeg (%+.3f Dice). On NuInsSeg it is not significant in "
    "the seed-level test (p = %.3f); it is supported by the image-level bootstrap." % (
        prow("cryonuseg").mean_difference, prow("nuinsseg").t_p_holm),
    "The ablation is non-monotonic (some added components do not raise Dice) and was run on one fold only.",
    "Baselines differ in architecture and loss, so the comparison does not isolate a single component.",
    "Phenotyping is exploratory and the clusters are unstable across settings.",
    "The test split has indirect exposure.",
    "No whole-slide or clinical claim is made."]
openi = [
    "Zenodo DOI and release tag are still to be added.",
    "Overleaf compile check: the project has never been compiled outside Overleaf, so first-compile issues are possible.",
    "Title decision: 'Pan-Cancer' is still in the title.",
    "Final figure redraws."]

chk = []


def v(label, ok):
    chk.append(ok)
    print("VERIFY %-60s %s" % (label, "OK" if ok else "FAIL"))


for m, n in order:
    for ds in ("cryonuseg", "nuinsseg"):
        r = agg[(agg.dataset == ds) & (agg.model == m)].iloc[0]
        v("%s %s %s" % (ds, m, cell(ds, m)[0]), cell(ds, m)[0] == "%.3f ± %.3f" % (r.test_dice_mean, r.test_dice_sd))
v("CryoNuSeg diff vs SAM-guided = +0.012", sc.startswith("+0.012"))
v("NuInsSeg Holm p = 0.070", "p = 0.070" in sn)
v("CryoNuSeg ahead in 3 of 3 folds", fr.folds_proposed_wins == 3 and fr.n_folds == 3)
print("VERIFY summary:", "ALL OK" if all(chk) else "FAILURES")

# ---- markdown
md = ["# Draft Cover Note", "", "**" + intro[0] + "**", "", intro[1], "", "## 1. What this draft contains and how to open it"]
md += ["- " + x for x in what] + ["", "## 2. Done in response to the reviewers"] + ["- " + x for x in done]
md += ["", "## 3. Key results: test Dice, mean ± SD (from the analysis csv files)", "",
       "| " + " | ".join(rows[0]) + " |", "|---|---|---|"] + ["| " + " | ".join(r) + " |" for r in rows[1:]]
md += ["", "EPCS-Net vs the SAM-guided baseline (paired, Dice):", ""] + ["- " + x for x in stat_lines]
md += ["", "## 4. Caveats"] + ["- " + x for x in cav] + ["", "## 5. Still open for the author"] + ["- " + x for x in openi]
md += ["", "## 6. Files in the zip (PAPER_OVERLEAF.zip)"] + ["- " + x for x in files_txt]
open(os.path.join(OUT, "Draft_Cover_Note.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

# ---- pdf
F = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("Cal", F + "calibri.ttf"))
pdfmetrics.registerFont(TTFont("CalB", F + "calibrib.ttf"))
pdfmetrics.registerFont(TTFont("CalI", F + "calibrii.ttf"))
pdfmetrics.registerFontFamily("Cal", normal="Cal", bold="CalB", italic="CalI", boldItalic="CalB")
NAVY = colors.HexColor("#1F3864")
base = ParagraphStyle("b", fontName="Cal", fontSize=9.8, leading=12.4)
H = ParagraphStyle("h", parent=base, fontName="CalB", fontSize=11, leading=13.5, textColor=NAVY, spaceBefore=7, spaceAfter=2)
T = ParagraphStyle("t", parent=base, fontName="CalB", fontSize=15, leading=18, textColor=NAVY)
B = ParagraphStyle("bl", parent=base, leftIndent=11, bulletIndent=2, spaceAfter=1)


def bl(items):
    return [Paragraph(escape(x), B, bulletText="•") for x in items]


st = [Paragraph("Draft Cover Note", T), Spacer(1, 3),
      Paragraph("<b>%s</b>" % escape(intro[0]), base), Spacer(1, 2), Paragraph(escape(intro[1]), base),
      Paragraph("1. What this draft contains and how to open it", H)] + bl(what)
st += [Paragraph("2. Done in response to the reviewers", H)] + bl(done)
st += [Paragraph("3. Key results: test Dice, mean ± SD (from the analysis csv files)", H)]
cs = ParagraphStyle("c", parent=base, fontSize=9, leading=11)
tb = Table([[Paragraph(escape(c), ParagraphStyle("ch", parent=cs, fontName="CalB") if i == 0 else cs) for c in r]
            for i, r in enumerate(rows)], colWidths=[62 * mm, 55 * mm, 55 * mm])
tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F2")),
                        ("GRID", (0, 0), (-1, -1), .4, colors.grey),
                        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F3F7FB")),
                        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
st += [tb, Spacer(1, 3), Paragraph("EPCS-Net vs the SAM-guided baseline (paired, Dice):", base)] + bl(stat_lines)
st += [Paragraph("4. Caveats", H)] + bl(cav) + [Paragraph("5. Still open for the author", H)] + bl(openi)
st += [Paragraph("6. Files in the zip (PAPER_OVERLEAF.zip)", H)] + bl(files_txt)
SimpleDocTemplate(os.path.join(OUT, "Draft_Cover_Note.pdf"), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                  topMargin=14 * mm, bottomMargin=14 * mm, title="Draft Cover Note").build(st)
print("written", OUT)
