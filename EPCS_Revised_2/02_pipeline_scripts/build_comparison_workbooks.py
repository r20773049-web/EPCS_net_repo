"""
Build the revision-3 comparison workbooks.

Outputs (into 07_revision3/02_comparison_workbooks/):
  1. EPCS_R3_Results_Comparison.xlsx  -- every previously reported number,
     side by side with what the reviewers now demand and an empty matrix for
     the new runs to fill.
  2. EPCS_R3_Task_Schedule.xlsx       -- the reviewer requirement matrix, the
     prioritised schedule, the figure plan, and the citation list.

Every "previous" number is copied from OLD_ROUNDS_1_2/EPCS_Master_Results_ForPaper.csv or
from the per-run summary JSON files, not retyped from the manuscript.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[2]          # ...\Desktop\EPCS
OUT = Path(__file__).resolve().parents[1] / "02_comparison_workbooks"

HDR_FILL = PatternFill("solid", fgColor="1F4E79")
HDR_FONT = Font(bold=True, color="FFFFFF", size=11)
SEC_FILL = PatternFill("solid", fgColor="D9E2F3")
TODO_FILL = PatternFill("solid", fgColor="FFF2CC")
BAD_FILL = PatternFill("solid", fgColor="FCE4E4")
OK_FILL = PatternFill("solid", fgColor="E2EFDA")

P_FILL = {
    "P0": PatternFill("solid", fgColor="F4B6B6"),
    "P1": PatternFill("solid", fgColor="FBD5A5"),
    "P2": PatternFill("solid", fgColor="FFE89A"),
    "P3": PatternFill("solid", fgColor="D9EAD3"),
}


def sheet(wb, name, rows, widths=None, freeze="A2", wrap_cols=(), fill_rules=None,
          title_note=None):
    ws = wb.create_sheet(name[:31])
    start = 1
    if title_note:
        ws.cell(1, 1, title_note).font = Font(italic=True, size=10, color="555555")
        ws.cell(1, 1).alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=1, start_column=1, end_row=1,
                       end_column=max(2, len(rows[0]) if rows else 2))
        ws.row_dimensions[1].height = 30
        start = 2

    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            cell = ws.cell(start + r_i, c_i + 1, val)
            if r_i == 0:
                cell.fill, cell.font = HDR_FILL, HDR_FONT
                cell.alignment = Alignment(wrap_text=True, vertical="center",
                                           horizontal="center")
            else:
                if c_i in wrap_cols:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
                else:
                    cell.alignment = Alignment(vertical="top")
                if fill_rules:
                    f = fill_rules(row, c_i, val)
                    if f:
                        cell.fill = f
    if widths:
        for i, w in enumerate(widths):
            ws.column_dimensions[get_column_letter(i + 1)].width = w
    ws.freeze_panes = freeze if start == 1 else f"A{start + 1}"
    if len(rows) > 1:
        ws.auto_filter.ref = (
            f"A{start}:{get_column_letter(len(rows[0]))}{start + len(rows) - 1}")
    ws.row_dimensions[start].height = 34
    return ws


# ---------------------------------------------------------------- source data
PARAMS = {       # model_id -> (total, trainable)
    "U-Net": (7_849_601, 7_849_601),
    "U-Net++": (2_291_969, 2_291_969),
    "U-Net3+": (6_684_289, 6_684_289),
    "eU-Net3+ (no SAM)": (11_767_265, 11_767_265),
    "SAM-guided baseline": (102_243_889, 8_508_417),
    "EPCS-Net (proposed)": (120_417_697, 26_682_225),
}

RUN_A = [
    # dataset, model, best epoch, acc, prec, rec, spec, dice, diceSD, iou, iouSD, ms, vram
    ("CryoNuSeg", "U-Net", 33, .8656, .8369, .7650, .9383, .7768, .1308, .6492, .1593, 13.5047, 330.87),
    ("CryoNuSeg", "U-Net++", 18, .8328, .7007, .8673, .8411, .7479, .0983, .6058, .1291, 29.2788, 308.53),
    ("CryoNuSeg", "U-Net3+", 13, .8629, .8002, .7869, .9228, .7737, .1047, .6407, .1376, 87.2742, 1406.50),
    ("CryoNuSeg", "eU-Net3+ (no SAM)", 9, .8418, .7120, .8944, .7815, .7909, .0626, .6579, .0857, 12.5442, 466.64),
    ("CryoNuSeg", "SAM-guided baseline", 14, .8792, .8161, .8374, .9158, .8126, .0743, .6897, .1034, 388.3468, 2952.76),
    ("CryoNuSeg", "EPCS-Net (proposed)", 21, .9023, .8014, .9049, .8976, .8471, .0503, .7374, .0742, 412.9578, 3305.65),
    ("NuInsSeg", "U-Net", 50, .9147, .8291, .5625, .9772, .6011, .2249, .4643, .2193, 5.1036, 358.76),
    ("NuInsSeg", "U-Net++", 37, .9124, .8040, .5625, .9729, .5872, .2353, .4520, .2219, 9.1532, 317.70),
    ("NuInsSeg", "U-Net3+", 47, .9142, .8119, .5444, .9769, .5798, .2437, .4470, .2289, 33.5746, 1431.15),
    ("NuInsSeg", "eU-Net3+ (no SAM)", 26, .9103, .7745, .6165, .9630, .6099, .2159, .4707, .2097, 5.3204, 509.73),
    ("NuInsSeg", "SAM-guided baseline", 22, .9080, .8195, .5332, .9780, .5489, .2859, .4281, .2567, 148.1958, 2991.34),
    ("NuInsSeg", "EPCS-Net (proposed)", 35, .9308, .7209, .8187, .9513, .7366, .1375, .5990, .1500, 160.0949, 3410.33),
]

SEEDS = [
    ("CryoNuSeg", 42, 21, .9023, .8014, .9049, .8976, .8471, .7374, ""),
    ("CryoNuSeg", 123, 42, .9046, .8386, .8637, .9255, .8480, .7381, ""),
    ("CryoNuSeg", 456, 16, .8986, .8394, .8521, .9284, .8397, .7264, ""),
    ("CryoNuSeg", 789, 38, .9037, .8125, .8914, .8993, .8488, .7394, ""),
    ("CryoNuSeg", 2024, 20, .9072, .8374, .8609, .9235, .8467, .7365, ""),
    ("NuInsSeg", 42, 35, .9308, .7209, .8187, .9513, .7366, .5990, "best of the five; quoted as the illustrative run elsewhere"),
    ("NuInsSeg", 123, 10, .9161, .6995, .7587, .9429, .6703, .5281, ""),
    ("NuInsSeg", 456, 9, .9137, .7089, .7601, .9415, .6792, .5355, ""),
    ("NuInsSeg", 789, 7, .9256, .7217, .7950, .9488, .7259, .5851, ""),
    ("NuInsSeg", 2024, 1, .9193, .7670, .6892, .9593, .6666, .5215, "peaked at epoch 1, never improved - possible instability"),
]

ABLATION = [
    ("1", "eU-Net3+ only (no SAM, no ACFB)", .7173, .5592, "", 1, "single run, earlier training pass"),
    ("2", "+ SAM bottleneck fusion (single scale)", .7659, .6266, "+0.0486", 1, "single run, earlier training pass"),
    ("3", "+ PSPI (pyramid 4-scale SAM fusion)", .7990, .6614, "+0.0331", 1, "single run, earlier training pass"),
    ("4", "+ ACFB bidirectional gating (BDSG)", .8090, .6751, "+0.0100", 1, "single run, earlier training pass"),
    ("5", "+ Dual-kernel refinement (DKSR)", .8120, .6840, "+0.0030", 1, "single run; increment smaller than the 5-seed SD on NuInsSeg"),
    ("6", "+ FTAS loss (full EPCS-Net)", .8371, .7220, "+0.0251", 1, "does NOT match the current Table 5 EPCS-Net Dice of 0.8471"),
]

ABLATION_MISSING = [
    ("A", "ACFB without PSPI (bottleneck ACFB only)", "separates the fusion operator from multi-scale injection", "Reviewer 1, Reviewer 3"),
    ("B", "Random frozen embedding instead of SAM", "tests whether gains come from SAM semantics or merely an extra input stream", "Reviewer 3"),
    ("C", "PSPI with plain concatenation instead of ACFB", "isolates PSPI from ACFB in the other direction", "Reviewer 1"),
    ("D", "GroupNorm encoder instead of BatchNorm encoder", "answers the batch-size-1 BatchNorm objection with evidence", "Reviewer 1"),
    ("E", "Swain et al. X-Gated Fusion at the bottleneck", "the published reference-21 fusion, reimplemented in our codebase", "Reviewer 4"),
    ("F", "Swain et al. X-Gated Fusion at all four scales", "isolates the claim: is the gain multi-scale injection or the fusion operator?", "Reviewer 4"),
]

MEDSAM = [
    ("CryoNuSeg", .6389, .4893, .2135, .8482, .2226, .2033, .1387, .1428, 476.0099, 2772.71),
    ("NuInsSeg", .7247, .2398, .2436, .8166, .1670, .1186, .0961, .0778, 413.6538, 2772.71),
]

GRADCAM = [
    ("CryoNuSeg", .3113, .1141, .1435, .4548, 0.5, 6),
    ("NuInsSeg", .4701, .1561, .0933, .8177, 0.5, 133),
]

HDBSCAN = [
    ("CryoNuSeg", "PRESPECIFIED: max(5, N/10)", 925, 92, 5, "euclidean", 0, 100.0, "n/a", "every component labelled noise"),
    ("CryoNuSeg", "SENSITIVITY: fixed 5", 925, 5, 5, "euclidean", 2, 31.7838, 0.4088, "this is what Figure 9 currently shows"),
    ("NuInsSeg", "PRESPECIFIED: max(5, N/10)", 4856, 485, 5, "euclidean", 0, 100.0, "n/a", "every component labelled noise"),
    ("NuInsSeg", "SENSITIVITY: fixed 5", 4856, 5, 5, "euclidean", 2, 6.7339, 0.5240, "this is what Figure 9 currently shows"),
]

PCA = [
    ("CryoNuSeg", 22.0939, 11.2364, 6.9783, 40.3086),
    ("NuInsSeg", 18.5676, 11.5913, 9.1074, 39.2662),
]

OSPT_ABL = [
    ("NuInsSeg", "OSPT enabled (organ-balanced)", 35, .9308, .7209, .8187, .9513, .7366, .5990),
    ("NuInsSeg", "OSPT disabled (flat sampling)", 45, .9295, .7197, .8221, .9490, .7359, .6010),
]

LATENCY_AUDIT = [
    ("CryoNuSeg", "EPCS-Net", "Run A", 6, 412.96, 3305.65),
    ("CryoNuSeg", "EPCS-Net", "Run B seed 42", 6, 424.48, 3305.65),
    ("CryoNuSeg", "EPCS-Net", "Run B seed 2024", 6, 440.04, 3297.31),
    ("NuInsSeg", "EPCS-Net", "Run A (OSPT on)", 133, 160.09, 3410.33),
    ("NuInsSeg", "EPCS-Net", "Run A OSPT ablation (flat)", 133, 419.35, 3305.65),
    ("NuInsSeg", "EPCS-Net", "Run B seed 42", 133, 160.40, 3412.14),
]

MODELS = ["U-Net", "U-Net++", "U-Net3+", "eU-Net3+ (no SAM)",
          "SAM-guided baseline (ours, bottleneck)",
          "Swain et al. 2024 [21] (official code)",
          "EPCS-Net (proposed)"]
SEED_LIST = [42, 123, 456, 789, 2024]


# ------------------------------------------------------- NEW-result ingestion
R3_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = R3_ROOT / "04_results" / "analysis"
MEDSAM_SUMMARY = R3_ROOT / "04_results" / "medsam_r3" / "summary_all.json"
METRICS6 = ["accuracy", "precision", "recall", "specificity", "dice", "iou"]
PREV_ID = {"U-Net": "unet", "U-Net++": "unetpp", "U-Net3+": "unet3plus",
           "eU-Net3+ (no SAM)": "eunet3plus", "SAM-guided baseline": "sam_unet",
           "EPCS-Net (proposed)": "epcs_net"}


def _csv(name):
    p = ANALYSIS / name
    return pd.read_csv(p) if p.is_file() else None


def _num(v, dp=4):
    return round(float(v), dp) if pd.notna(v) else ""


def load_job_matrix():
    """The real job matrix from run_sweep.build_matrix(), with status from summary.json."""
    import importlib.util
    sweep = R3_ROOT / "03_code" / "runners" / "run_sweep.py"
    spec = importlib.util.spec_from_file_location("run_sweep_ro", sweep)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # defines functions only; main() not called
    out = []
    for j in mod.build_matrix():
        d = mod.run_dir(j["dataset"], j["model"], j["seed"], j["fold"])
        sp = d / "summary.json"
        rec = dict(j, dir=d, summary=None, status="COMPLETE" if sp.is_file() else "PENDING",
                   label=mod.MODELS[j["model"]].get("label", j["model"]))
        if sp.is_file():
            try:
                rec["summary"] = json.loads(sp.read_text(encoding="utf-8"))
            except Exception:
                rec["status"] = "UNREADABLE"
        out.append(rec)
    return out


def medsam_rows():
    if not MEDSAM_SUMMARY.is_file():
        return []
    return [r for r in json.loads(MEDSAM_SUMMARY.read_text(encoding="utf-8"))
            if not r.get("dry_run") and "metric_mean" in r]


# ------------------------------------------------------------------ workbook 1
def build_results(path: Path):
    wb = Workbook()
    wb.remove(wb.active)

    sheet(wb, "00_README", [
        ["Sheet", "What it holds", "Status"],
        ["01_PREV_Headline", "Run A: all six models, single seed, both datasets. The current Tables 5 and 7.", "PREVIOUS WORK"],
        ["02_PREV_EPCSNet_5seeds", "Run B: EPCS-Net across five seeds. The current Table 9.", "PREVIOUS WORK"],
        ["03_PREV_MedSAM", "Frozen zero-shot MedSAM, whole-image box prompt.", "PREVIOUS WORK"],
        ["04_PREV_Ablation", "The current Table 13 chain, plus the configurations reviewers say are missing.", "PREVIOUS WORK + GAPS"],
        ["05_PREV_DPNP", "Grad-CAM overlap, HDBSCAN parameters, PCA explained variance.", "PREVIOUS WORK"],
        ["06_PREV_Latency_Params", "Latency and parameter audit. Shows the unexplained latency spread.", "PREVIOUS WORK + PROBLEM"],
        ["07_GAP_Demanded", "Every reviewer demand about results, what exists now, what must change.", "THE GAP LIST"],
        ["08_NEW_Run_Matrix", "The real 210-job matrix from run_sweep.py with status from summary.json and the TEST scores of completed runs.", "AUTO-FILLED (regenerate as sweep runs)"],
        ["08b_NEW_vs_PREV_Headline", "New test-split aggregates (mean, SD, 95% CI) next to the PREVIOUS validation numbers, plus MedSAM sanity/oracle rows below a rule.", "AUTO-FILLED"],
        ["09_NEW_Significance", "Paired tests (analysis/paired_significance_tests.csv) and fold-averaged tests (fold_averaged_tests.csv).", "AUTO-FILLED"],
        ["10_OSPT_Organ_Balance", "Per-organ counts before and after OSPT. Answers Reviewer 1 comment 3.", "COMPUTED"],
        ["11_CryoNuSeg_Tissue_Audit", "Actual tissue types vs what the manuscript claims. Contains a factual error to fix.", "COMPUTED - ERROR FOUND"],
    ], widths=[26, 86, 24], wrap_cols=(1,),
        title_note="EPCS-Net revision 3. Previous results and the results the reviewers now demand. "
                   "Sheets prefixed PREV are the already-published numbers; sheets prefixed NEW are empty templates for this round.")

    # 01 headline
    rows = [["Dataset", "Model", "Best epoch", "Accuracy", "Precision", "Recall",
             "Specificity", "Dice", "Dice SD (images)", "IoU", "IoU SD (images)",
             "Params TOTAL", "Params TRAINABLE", "ms / image", "Peak VRAM (MiB)",
             "Seeds run", "Selection / scoring set"]]
    for d, m, ep, a, p, r, s, di, dsd, io, isd, ms, vr in RUN_A:
        tot, tr = PARAMS[m]
        rows.append([d, m, ep, a, p, r, s, di, dsd, io, isd, tot, tr, ms, vr,
                     5 if m.startswith("EPCS") else 1,
                     "validation set used for BOTH checkpoint selection and reporting"])
    sheet(wb, "01_PREV_Headline", rows,
          widths=[12, 24, 10, 10, 10, 10, 11, 9, 13, 9, 13, 14, 16, 11, 14, 10, 46],
          wrap_cols=(16,),
          title_note="Run A. Reviewers 1, 2, 3 and 4 all object to the last two columns: "
                     "single-seed baselines, and selection on the same set that is reported.",
          fill_rules=lambda row, c, v: BAD_FILL if c in (15, 16) and row[1] != "Model" else None)

    # 02 seeds
    rows = [["Dataset", "Seed", "Best epoch", "Accuracy", "Precision", "Recall",
             "Specificity", "Dice", "IoU", "Note"]]
    rows += [list(r) for r in SEEDS]
    rows.append(["CryoNuSeg", "MEAN of 5", 27.4, .9033, .8259, .8746, .9149, .8460, .7356, "SD(Dice) = 0.0037, range 0.8397-0.8488"])
    rows.append(["NuInsSeg", "MEAN of 5", 12.4, .9211, .7236, .7643, .9488, .6957, .5539, "SD(Dice) = 0.0330, range 0.6666-0.7366"])
    sheet(wb, "02_PREV_EPCSNet_5seeds", rows,
          widths=[12, 12, 11, 10, 10, 10, 11, 9, 9, 60], wrap_cols=(9,),
          title_note="EPCS-Net only. No baseline has multi-seed numbers, which is why no paired test is possible today.")

    # 03 MedSAM
    rows = [["Dataset", "Accuracy", "Precision", "Recall", "Specificity", "Dice",
             "Dice SD", "IoU", "IoU SD", "ms / image", "Peak VRAM (MiB)", "Protocol"]]
    for r in MEDSAM:
        rows.append(list(r) + ["frozen, zero-shot, ONE whole-image box prompt, no fine-tuning"])
    sheet(wb, "03_PREV_MedSAM", rows,
          widths=[12, 10, 10, 10, 11, 9, 9, 9, 9, 11, 14, 56], wrap_cols=(11,),
          title_note="Reviewer 1: a whole-image box cannot specify many separate nuclei, so this is not MedSAM's intended regime. "
                     "It must move out of the competitive comparison, and a properly prompted configuration must be added.")

    # 04 ablation
    rows = [["Row", "Variant", "Dice", "IoU", "Dice gain", "Seeds run", "Problem"]]
    rows += [list(r) for r in ABLATION]
    rows.append(["", "", "", "", "", "", ""])
    rows.append(["MISSING CONFIGURATIONS THE REVIEWERS ASK FOR", "", "", "", "", "", ""])
    rows.append(["ID", "Variant to add", "Dice", "IoU", "Why it is needed", "Seeds to run", "Requested by"])
    for i, v, why, who in ABLATION_MISSING:
        rows.append([i, v, "TO RUN", "TO RUN", why, 5, who])
    sheet(wb, "04_PREV_Ablation", rows,
          widths=[10, 42, 10, 10, 44, 12, 52], wrap_cols=(1, 4, 6),
          title_note="Current Table 13. Every row is a single run from a separate, earlier training pass, and its final row (0.8371) "
                     "does not match the current EPCS-Net result (0.8471). The DKSR increment (+0.0030) is smaller than the "
                     "five-seed standard deviation on NuInsSeg (0.0330).",
          fill_rules=lambda row, c, v: TODO_FILL if v == "TO RUN" else None)

    # 05 DPNP
    ws = wb.create_sheet("05_PREV_DPNP")
    r = 1

    def block(header, data, widths):
        nonlocal r
        ws.cell(r, 1, header).font = Font(bold=True, size=12)
        ws.cell(r, 1).fill = SEC_FILL
        r += 1
        for row in [header_row for header_row in [data[0]]] + data[1:]:
            for c, val in enumerate(row):
                cell = ws.cell(r, c + 1, val)
                if row is data[0]:
                    cell.fill, cell.font = HDR_FILL, HDR_FONT
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            r += 1
        r += 2
        for i, w in enumerate(widths):
            cur = ws.column_dimensions[get_column_letter(i + 1)].width or 0
            ws.column_dimensions[get_column_letter(i + 1)].width = max(cur, w)

    block("GRAD-CAM OVERLAP WITH GROUND TRUTH (IoU)",
          [["Dataset", "Mean IoU", "SD", "Min", "Max", "Threshold", "N images"]] +
          [list(x) for x in GRADCAM], [16, 12, 10, 10, 10, 12, 12])
    block("HDBSCAN CLUSTERING",
          [["Dataset", "Configuration", "N components", "min_cluster_size", "min_samples",
            "Metric", "Clusters", "Outlier %", "Silhouette", "Note"]] +
          [list(x) for x in HDBSCAN], [16, 30, 14, 18, 14, 12, 10, 12, 12, 44])
    block("PCA EXPLAINED VARIANCE (%)",
          [["Dataset", "PC1", "PC2", "PC3", "PC1+PC2+PC3"]] +
          [list(x) for x in PCA], [16, 12, 10, 10, 14])
    block("OSPT ABLATION (NuInsSeg)",
          [["Dataset", "Variant", "Best epoch", "Accuracy", "Precision", "Recall",
            "Specificity", "Dice", "IoU"]] + [list(x) for x in OSPT_ABL],
          [16, 30, 12, 10, 10, 10, 12, 10, 10])
    ws.cell(r, 1, "Open problems on this sheet:").font = Font(bold=True)
    for i, t in enumerate([
        "The prespecified HDBSCAN rule finds ZERO clusters. Figure 9 shows the sensitivity check instead. The manuscript must make the prespecified rule primary.",
        "Feature vector = 7 interpretable morphological terms + 512 ResNet18 embedding dimensions. The PC labels 'shape/size', 'staining intensity', 'texture/chromatin' are not yet supported by any loading analysis.",
        "Connected components can merge touching nuclei, so these are region features, not per-nucleus measurements. NuInsSeg ships ground-truth instance masks, so this can be quantified.",
        "OSPT delta is 0.0007 Dice. OSPT was never applied to CryoNuSeg at all (that split has no organ column).",
    ], start=1):
        ws.cell(r + i, 1, f"{i}. {t}").alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=r + i, start_column=1, end_row=r + i, end_column=10)
        ws.row_dimensions[r + i].height = 30

    # 06 latency
    rows = [["Dataset", "Model", "Run", "N val images", "ms / image", "Peak VRAM (MiB)", "Comment"]]
    for d, m, run, n, ms, vr in LATENCY_AUDIT:
        rows.append([d, m, run, n, ms, vr, ""])
    rows.append(["", "", "", "", "", "", ""])
    rows.append(["PROBLEM", "The same model on the same 133 NuInsSeg images reports 160.09 and 419.35 ms/image in two different runs.",
                 "", "", "", "", ""])
    rows.append(["CAUSE", "Timing is CUDA-synchronised around the forward pass only, with no warm-up iterations, and runs were executed while other GPU jobs were active.",
                 "", "", "", "", ""])
    rows.append(["FIX", "Re-measure every model on an idle GPU: fixed warm-up iterations, explicit synchronisation, data loading excluded, batch size 1, report GPU model and driver. Inference only, no retraining needed.",
                 "", "", "", "", ""])
    sheet(wb, "06_PREV_Latency_Params", rows,
          widths=[14, 16, 28, 14, 12, 16, 20], wrap_cols=(1,),
          title_note="Reviewer 3 minor 2 and Reviewer 4 comment 4 both ask why latency differs so much. This sheet shows it is not a dataset effect.")

    # 07 gap list
    rows = [["ID", "Reviewer", "What is demanded", "What exists today", "What must change", "Type", "Priority"]]
    gaps = [
        ("G1", "R3 major 1, R4 comment 1", "Report on a set that was never used to choose the checkpoint",
         "All scores are validation scores; the best-validation-Dice checkpoint is then scored on that same validation set",
         "New three-way split. Select on validation, report on an untouched test split", "EXPERIMENT", "P0"),
        ("G2", "R1, R2 comment 1, R3 major 3", "Baselines under the same five seeds, with paired tests and intervals",
         "EPCS-Net has 5 seeds; every baseline is a single run at an unfixed seed",
         "Retrain all baselines across the same five seeds, then paired tests on identical splits", "EXPERIMENT", "P0"),
        ("G3", "R2 comment 2", "CryoNuSeg validation must be tissue-stratified and cover all ten tissues",
         "Random 24/6 split; validation covers only 5 of 10 tissues; no organ column exists for CryoNuSeg",
         "Tissue-stratified splits, ideally three-fold so every tissue appears in test", "EXPERIMENT", "P0"),
        ("G4", "R1 comment 2, R3 major 4", "Ablation rerun under the current protocol, multi-seed, with missing controls",
         "Six single runs from an earlier training pass; final row does not match the current result",
         "Rerun all rows x 5 seeds under the current code, add configurations A-F on sheet 04", "EXPERIMENT", "P1"),
        ("G5", "R4 comment 2", "Benchmark the published reference-21 architecture directly",
         "Only a simplified bottleneck-concatenation stand-in",
         "Run the official MIT-licensed code; also port its X-Gated Fusion into our codebase for a controlled test", "EXPERIMENT", "P0"),
        ("G6", "R1 comment 4", "MedSAM evaluated in its intended regime, or removed from the competitive comparison",
         "One whole-image box prompt, frozen, zero-shot",
         "Move to a separate zero-shot sanity subsection AND add a per-nucleus prompted configuration", "EXPERIMENT", "P1"),
        ("G7", "R3 major 2, R4 comment 4", "Explain how the official SAM checkpoint accepts the stated input",
         "Methods claim a 256 input and a 16x16 token grid",
         "TEXT FIX: the code upsamples 256 to 1024 and uses the official preprocess, so the grid is 64x64", "TEXT", "P0"),
        ("G8", "R1 comment 1", "Resolve BatchNorm versus GroupNorm",
         "Encoder DoubleConv uses BatchNorm2d; only ACFB step 5 uses GroupNorm(8); the text reads as a global claim",
         "TEXT FIX plus ablation D: run a GroupNorm-encoder variant so the choice is evidenced", "TEXT + EXPERIMENT", "P1"),
        ("G9", "R1 comment 3", "Specify the OSPT sampler and give per-organ counts",
         "Sampler is inverse-organ-frequency weighted with replacement; never stated in the paper; never applied to CryoNuSeg",
         "TEXT FIX plus the table on sheet 10", "TEXT", "P1"),
        ("G10", "R2 comment 3", "Low-overlap Grad-CAM failure cases",
         "Only representative good cases are shown; per-sample overlap ranges from 0.09 to 0.82 on NuInsSeg",
         "New figure panel built from the lowest-overlap cases already in the per-sample file", "FIGURE", "P1"),
        ("G11", "R3 major 5, R4 comment 3", "Make the prespecified clustering primary; validate the phenotyping",
         "Prespecified rule yields zero clusters; Figure 9 shows the sensitivity check; PC labels unsupported",
         "Restructure the section, add loading analysis, cluster stability, and organ/species/stain confound checks", "EXPERIMENT + TEXT", "P1"),
        ("G12", "R4 comment 3", "Compare predicted components with ground-truth instances",
         "Components are assumed to be nuclei",
         "Quantify merging against NuInsSeg label masks, report merge rate and feature bias", "EXPERIMENT", "P1"),
        ("G13", "R3 minor 2, R4 comment 4", "Proper latency reporting",
         "Uncontrolled timing, no warm-up, contradictory numbers for the same model",
         "Re-measure on an idle GPU with warm-up; report hardware, boundaries, synchronisation", "EXPERIMENT", "P1"),
        ("G14", "R4 comment 4", "Report total parameters alongside trainable",
         "Only trainable is in the tables, although total is already recorded in every run summary",
         "TEXT FIX: add the total column (already on sheet 01)", "TEXT", "P2"),
        ("G15", "R3 minor 1", "Stop converting pixel recall into missed nuclei",
         "False-negative rate is presented in clinical terms",
         "TEXT FIX: describe it as a pixel-level quantity only", "TEXT", "P2"),
        ("G16", "R2 comment 4", "Do not imply demonstrated whole-slide feasibility",
         "Mostly handled, but must stay consistent after edits",
         "TEXT CHECK across abstract, discussion and future work", "TEXT", "P2"),
        ("G17", "R2 comment 1", "State plainly that superiority is not statistically established",
         "Stated in the discussion",
         "Keep, and update once paired tests exist", "TEXT", "P2"),
        ("G18", "R3 minor 4, R4 comments 2 and 3", "Add the named citations",
         "None of the nine are cited",
         "Add and discuss; see the citation sheet in the schedule workbook", "TEXT", "P2"),
        ("G19", "R3 minor 3, R4 comment 4", "Editorial consistency and a real code DOI",
         "Code Availability still carries a placeholder DOI",
         "Publish the release, mint the Zenodo DOI, reconcile abstract, tables and archive", "ADMIN", "P0"),
        ("G20", "Found in this audit", "CryoNuSeg tissue list in the manuscript is factually wrong",
         "Five of the ten named tissues do not exist in the dataset",
         "TEXT FIX: see sheet 11", "TEXT", "P0"),
    ]
    rows += [list(g) for g in gaps]
    sheet(wb, "07_GAP_Demanded", rows,
          widths=[7, 24, 44, 56, 60, 20, 10], wrap_cols=(2, 3, 4),
          title_note="The complete results gap list. P0 blocks resubmission.",
          fill_rules=lambda row, c, v: P_FILL.get(row[6]) if c == 6 and row[0] != "ID" else None)

    # 08 run matrix (real matrix from run_sweep.py)
    jobs = load_job_matrix()
    rows = [["Run ID", "Block", "Dataset", "Fold", "Model", "Seed", "Status",
             "Best epoch (val)", "TEST Dice", "TEST IoU", "TEST Accuracy",
             "TEST Precision", "TEST Recall", "TEST Specificity", "Params total",
             "Params trainable", "Run directory"]]
    for j in jobs:
        sm = j["summary"] or {}
        tm = sm.get("test_mean", {})
        rows.append([j["run_id"], j["block"], j["dataset"], j["fold"] or "-", j["label"],
                     j["seed"], j["status"], sm.get("best_epoch", ""),
                     _num(tm.get("dice")), _num(tm.get("iou")), _num(tm.get("accuracy")),
                     _num(tm.get("precision")), _num(tm.get("recall")),
                     _num(tm.get("specificity")),
                     sm.get("parameters_total", ""), sm.get("parameters_trainable", ""),
                     str(j["dir"])])
    n_done = sum(j["status"] == "COMPLETE" for j in jobs)
    sheet(wb, "08_NEW_Run_Matrix", rows,
          widths=[40, 20, 12, 7, 34, 8, 11, 15, 11, 11, 13, 14, 12, 14, 14, 16, 50],
          title_note=f"Real job matrix from run_sweep.build_matrix(): {len(jobs)} jobs, {n_done} complete, "
                     f"{len(jobs) - n_done} pending (status = existence of summary.json). Scores are TEST split.",
          fill_rules=lambda row, c, v: (TODO_FILL if v == "PENDING" else OK_FILL) if c == 6 else None)

    # 08b new vs previous headline
    agg = _csv("aggregated_by_model.csv")
    prev = {(d.lower(), PREV_ID[m]): (dice, iou) for d, m, _e, _a, _p, _r, _s, dice, _ds, iou, *_ in RUN_A
            if m in PREV_ID}
    rows = [["Dataset", "Model", "Group", "NEW runs", "NEW Accuracy", "NEW Precision", "NEW Recall",
             "NEW Specificity", "NEW Dice mean", "NEW Dice SD", "NEW Dice 95% CI low", "NEW Dice 95% CI high",
             "NEW IoU mean", "NEW IoU SD", "PREV Dice (validation, old)", "PREV IoU (validation, old)",
             "Dice change (NEW test - PREV val)", "Params total", "Params trainable"]]
    if agg is not None:
        agg = agg.sort_values(["dataset", "group", "test_dice_mean"], ascending=[True, True, False])
        for _, r in agg.iterrows():
            pd_, pi_ = prev.get((r["dataset"], r["model"]), ("", ""))
            n1 = int(r["n_runs"]) < 2
            rows.append([r["dataset"], r["label"], r["group"], int(r["n_runs"]),
                         _num(r["test_accuracy_mean"]), _num(r["test_precision_mean"]),
                         _num(r["test_recall_mean"]), _num(r["test_specificity_mean"]),
                         _num(r["test_dice_mean"]), "" if n1 else _num(r["test_dice_sd"]),
                         _num(r.get("test_dice_ci_low")), _num(r.get("test_dice_ci_high")),
                         _num(r["test_iou_mean"]), "" if n1 else _num(r["test_iou_sd"]),
                         pd_, pi_, _num(r["test_dice_mean"] - pd_) if pd_ != "" else "",
                         int(r["params_total"]) if pd.notna(r["params_total"]) else "",
                         int(r["params_trainable"]) if pd.notna(r["params_trainable"]) else ""])
    ms = medsam_rows()
    if ms:
        rows.append(["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])
        rows.append(["EXTERNAL REFERENCE ROWS - NOT PART OF THE RANKING", "", "", "", "", "", "", "",
                     "", "", "", "", "", "", "", "", "", "", ""])
        for r in ms:
            tag = ("MedSAM zero-shot sanity check" if r["mode"] == "zero_shot_wholebox"
                   else "MedSAM ORACLE upper bound (GT boxes)")
            m_, sd_ = r["metric_mean"], r["metric_sample_sd"]
            rows.append([r["dataset"], tag, "external", f"{r['n_test_images']} test images",
                         _num(m_["accuracy"]), _num(m_["precision"]), _num(m_["recall"]),
                         _num(m_["specificity"]), _num(m_["dice"]), _num(sd_["dice"]), "", "",
                         _num(m_["iou"]), _num(sd_["iou"]), "", "", "", "", ""])
    sheet(wb, "08b_NEW_vs_PREV_Headline", rows,
          widths=[14, 44, 16, 14] + [12] * 8 + [12, 12, 16, 16, 18, 14, 14],
          title_note="NEW = untouched TEST split, matched seeds (and folds for CryoNuSeg); PREV = old VALIDATION numbers from Run A (single run). "
                     "Not like-for-like: the change column reflects a protocol change. Rows with 1 run have no SD or CI. "
                     "MedSAM SD is across test images, per fold or split.",
          fill_rules=lambda row, c, v: (SEC_FILL if str(row[0]).startswith("EXTERNAL") else
                                        TODO_FILL if row[2] == "external" else None))

    # 09 significance (from analysis CSVs)
    tests = _csv("paired_significance_tests.csv")
    folds = _csv("fold_averaged_tests.csv")
    rows = [["Dataset", "Family", "Comparison", "Metric", "n pairs", "EPCS-Net mean", "Baseline mean",
             "Mean difference", "95% CI low", "95% CI high", "Cohen d (paired)", "t p value",
             "t p (Holm)", "Wilcoxon p", "Wilcoxon p (Holm)", "Wilcoxon p floor",
             "Significant (t, Holm, 0.05)"]]
    if tests is not None and not tests.empty:
        for _, r in tests.sort_values(["dataset", "family", "mean_difference"],
                                      ascending=[True, True, False]).iterrows():
            rows.append([r["dataset"], r["family"], f"EPCS-Net vs {r['baseline_label']}", r["metric"],
                         int(r["n_pairs"]), _num(r["proposed_mean"]), _num(r["baseline_mean"]),
                         _num(r["mean_difference"]), _num(r["ci95_low"]), _num(r["ci95_high"]),
                         _num(r["cohens_d_paired"], 3), _num(r["t_p_value"], 6), _num(r["t_p_holm"], 6),
                         _num(r["wilcoxon_p_value"], 6), _num(r["wilcoxon_p_holm"], 6),
                         _num(r["wilcoxon_p_floor"], 6),
                         "yes" if r["significant_t_holm_0.05"] else "no"])
    else:
        rows.append(["", "", "no paired tests yet", "", "", "", "", "", "", "", "", "", "", "", "", "", "PENDING"])
    rows.append([""] * 17)
    rows.append(["FOLD-AVERAGED TESTS (CryoNuSeg, n = folds, low power)", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])
    rows.append(["Dataset", "Baseline", "n folds", "Mean difference", "Per-fold differences",
                 "Folds EPCS-Net wins", "t p (folds)", "Note"] + [""] * 9)
    if folds is not None:
        for _, r in folds.iterrows():
            rows.append([r["dataset"], r["baseline"], int(r["n_folds"]), _num(r["mean_difference"]),
                         r["per_fold_differences"], int(r["folds_proposed_wins"]),
                         _num(r["t_p_value_folds"], 6), r["note"]] + [""] * 9)
    n_fold_hdr = len(rows) - (len(folds) if folds is not None else 0)
    ws9 = sheet(wb, "09_NEW_Significance", rows,
                widths=[14, 14, 44, 9, 9, 14, 14, 15, 11, 11, 14, 11, 11, 12, 15, 14, 18],
                title_note="Paired across matched seeds (x folds on CryoNuSeg) on identical splits, Holm-corrected within dataset; "
                           "source: 04_results/analysis/paired_significance_tests.csv and fold_averaged_tests.csv. "
                           "Fold-averaged tests are the conservative check because runs within a fold share a split.",
                fill_rules=lambda row, c, v: (OK_FILL if v == "yes" else BAD_FILL if v == "no" else None) if c == 16 else None)
    ws9.auto_filter.ref = None

    # 10 OSPT organ balance
    split_csv = ROOT / "OLD_ROUNDS_1_2" / "02_code_RunA" / "NuInsSeg_RunA" / "splits" / "split_nuinsseg_80_20_seed42_RunA.csv"
    rows = [["Organ", "Total patches", "Train patches", "Validation patches",
             "Share of train without OSPT", "Expected draws per epoch WITH OSPT",
             "Share of train with OSPT", "Oversampling factor",
             "Flag on the CURRENT random split"]]
    n_zero = 0
    if split_csv.is_file():
        df = pd.read_csv(split_csv)
        tr = df[df["split"] == "train"]
        va = df[df["split"] == "validation"]
        n_tr, k = len(tr), tr["organ"].nunique()
        for organ in sorted(df["organ"].unique()):
            c_all = int((df["organ"] == organ).sum())
            c_tr = int((tr["organ"] == organ).sum())
            c_va = int((va["organ"] == organ).sum())
            flat = c_tr / n_tr if n_tr else 0
            exp = n_tr / k if k else 0
            bal = 1 / k if k else 0
            if c_va == 0:
                flag = "NOT EVALUATED - zero validation patches"
                n_zero += 1
            elif c_va <= 2:
                flag = "thin - 2 or fewer validation patches"
            else:
                flag = ""
            rows.append([organ, c_all, c_tr, c_va, round(flat, 5),
                         round(exp, 2), round(bal, 5),
                         round(bal / flat, 3) if flat else "", flag])
        rows.append(["TOTAL", len(df), n_tr, len(va), 1.0, n_tr, 1.0, "",
                     f"{n_zero} of {k} organs never appear in validation"])
    sheet(wb, "10_OSPT_Organ_Balance", rows,
          widths=[38, 14, 14, 18, 26, 30, 24, 18, 40], wrap_cols=(8,),
          fill_rules=lambda row, c, v: (BAD_FILL if isinstance(v, str) and v.startswith("NOT EVALUATED")
                                        else (TODO_FILL if isinstance(v, str) and v.startswith("thin") else None)) if c == 8 else None,
          title_note="Answers Reviewer 1 comment 3 exactly. The sampler weight is 1/count(organ) with replacement, so the sum of "
                     "weights equals the number of organs K and every organ has draw probability exactly 1/K. Minority organs are "
                     "oversampled by the factor shown. TWO FURTHER PROBLEMS: CryoNuSeg has no organ column in its split manifest, "
                     "so OSPT was never applied to CryoNuSeg at all; and the current random NuInsSeg validation split leaves two "
                     "organs with zero validation patches, so the multi-tissue evaluation does not actually cover every organ.")

    # 11 CryoNuSeg tissue audit
    rows = [["Tissue in the actual dataset", "Patches", "In train", "In validation",
             "Named in the manuscript?", "Manuscript also names (but does not exist)"]]
    actual = {
        "Human_AdrenalGland": (3, 2, 1, "YES"),
        "Human_Larynx": (3, 2, 1, "YES"),
        "Human_LymphNodes": (3, 1, 2, "YES"),
        "Human_Mediastinum": (3, 3, 0, "NO - missing from the manuscript list"),
        "Human_Pancreas": (3, 3, 0, "NO - missing from the manuscript list"),
        "Human_Pleura": (3, 3, 0, "NO - missing from the manuscript list"),
        "Human_Skin": (3, 2, 1, "NO - missing from the manuscript list"),
        "Human_Testes": (3, 2, 1, "YES"),
        "Human_Thymus": (3, 3, 0, "NO - missing from the manuscript list"),
        "Human_ThyroidGland": (3, 3, 0, "YES"),
    }
    phantom = ["medullary thyroid", "mouth floor", "tonsil", "brain", "colon", "", "", "", "", ""]
    for i, (k, (tot, trn, val, named)) in enumerate(actual.items()):
        rows.append([k, tot, trn, val, named, phantom[i]])
    rows.append(["", "", "", "", "", ""])
    rows.append(["SUMMARY", 30, 24, 6,
                 "Only 5 of the 10 tissues named in Section 3.1 actually exist in CryoNuSeg.",
                 "The other 5 names in the manuscript are not in this dataset."])
    rows.append(["VALIDATION COVERAGE", "", "", 6,
                 "The 6 validation patches span only 5 distinct tissues (LymphNodes appears twice).",
                 "This is exactly Reviewer 2 comment 2."])
    sheet(wb, "11_CryoNuSeg_Tissue_Audit", rows,
          widths=[32, 11, 11, 15, 62, 48], wrap_cols=(4, 5),
          title_note="FACTUAL ERROR FOUND. Section 3.1 lists ten tissues; five of them are not in the dataset. "
                     "Computed directly from the image filenames and the split manifest.",
          fill_rules=lambda row, c, v: BAD_FILL if isinstance(v, str) and v.startswith("NO -") else None)

    wb.save(path)
    print(f"  wrote {path.name}  ({len(wb.sheetnames)} sheets)")


# ------------------------------------------------------------------ workbook 2
def build_schedule(path: Path):
    wb = Workbook()
    wb.remove(wb.active)

    reqs = [
        # ID, source, requirement, response type, priority, phase, GPU h, depends, deliverable
        ("E1", "Editor 1", "Remove AI-generated images", "Replace all four schematic figures with Excalidraw sources generated by a committed script; declare provenance for every figure", "P0", "1", 0, "-", "Figures 1-4 re-exported; provenance paragraph in Methods"),
        ("E2", "Editor 2", "Move Dataset and Data Preprocessing into Methodology", "Structural edit: fold Section 3 into Section 4", "P0", "1", 0, "-", "Renumbered manuscript"),
        ("E3", "Editor 3", "Add a Limitations of the Proposed Work subsection", "New subsection collecting the limitations now scattered across Discussion", "P0", "1", 0, "-", "New Section 6.x"),
        ("E4", "Editor 4", "Move Future Work to the end of the Conclusion", "Structural edit: demote Section 8 into the Conclusion", "P0", "1", 0, "-", "Renumbered manuscript"),
        ("E5", "Editor / journal policy", "Deposit code in a DOI-assigning repository", "Publish the GitHub release, enable the Zenodo hook, mint the DOI, replace the placeholder", "P0", "1", 0, "-", "Real DOI in Code Availability"),
        ("X1", "Audit finding", "CryoNuSeg tissue list is factually wrong", "Correct the ten tissue names in the dataset section", "P0", "1", 0, "-", "Corrected Section 3.1"),
        ("X2", "Audit finding", "SAM token grid stated as 16x16", "Correct to: patch upsampled 256 to 1024, official preprocess, 64x64 grid, 256 channels", "P0", "1", 0, "-", "Corrected Methods + Figure 1 + equation for NCAP"),
        ("X3", "Audit finding", "OSPT never applied to CryoNuSeg", "Disclose that OSPT applies to NuInsSeg only, because CryoNuSeg has no organ grouping in the split", "P0", "1", 0, "-", "Disclosure in Methods and Limitations"),

        ("R3.1", "Reviewer 3 major 1 / Reviewer 4 comment 1", "Checkpoint selected and reported on the same set", "Build organ-stratified train/validation/test splits; select on validation, report on test", "P0", "2", 0, "-", "New split manifests"),
        ("R3.3", "Reviewer 3 major 3 / Reviewer 1 / Reviewer 2 comment 1", "Baselines are single runs at unfixed seeds", "Retrain every model across the same five seeds on the new splits", "P0", "3", 45, "R3.1", "Run matrix sheet 08 filled"),
        ("R2.2", "Reviewer 2 comment 2", "CryoNuSeg split is not tissue-stratified", "Three-fold tissue-stratified cross-validation so all ten tissues appear in test", "P0", "2-3", 0, "R3.1", "Fold manifests + results"),
        ("R4.2", "Reviewer 4 comment 2", "Benchmark the published reference-21 architecture", "Run the official MIT-licensed code on identical splits; additionally port its X-Gated Fusion into our codebase at bottleneck and at four scales", "P0", "3", 10, "R3.1", "Two new baseline rows + two ablation rows"),
        ("STAT", "Reviewers 1, 2, 3", "No paired significance testing or intervals", "Paired t-test and Wilcoxon across matched seeds, Holm correction, bootstrap CIs on per-image test Dice", "P0", "4", 0, "R3.3", "Significance sheet 09 filled"),

        ("R1.2", "Reviewer 1 comment 2 / Reviewer 3 major 4", "Ablation is single-run and from an earlier pass", "Rerun all six rows under the current code x 5 seeds, plus configurations A-F", "P1", "3", 12, "R3.1", "New ablation table with intervals"),
        ("R1.1", "Reviewer 1 comment 1", "BatchNorm versus GroupNorm contradiction", "Clarify that GroupNorm is used only inside ACFB; run a GroupNorm-encoder variant as evidence", "P1", "3", 2, "R1.2", "Corrected text + ablation row D"),
        ("R1.4", "Reviewer 1 comment 4", "MedSAM evaluated outside its intended regime", "Move to a separate zero-shot sanity subsection; add a per-nucleus prompted configuration", "P1", "3", 1, "-", "Reframed subsection + new row"),
        ("R1.3", "Reviewer 1 comment 3", "OSPT sampler unspecified", "State the sampler exactly; publish per-organ counts before and after", "P1", "1", 0, "-", "New table + corrected Methods"),
        ("R2.3", "Reviewer 2 comment 3", "No low-overlap Grad-CAM failure cases", "New panel built from the lowest-overlap samples already recorded", "P1", "4", 0.5, "-", "New figure panel"),
        ("R3.5", "Reviewer 3 major 5 / Reviewer 4 comment 3", "Clustering: prespecified rule finds nothing, figure shows the sensitivity check", "Make the prespecified rule primary; present the alternative explicitly as a sensitivity check; add stability and confound analysis", "P1", "4", 2, "-", "Rewritten subsection + new analysis"),
        ("R4.3", "Reviewer 4 comment 3", "Components are not per-nucleus measurements", "Quantify merging against NuInsSeg ground-truth instance masks; compare feature distributions; PCA loading analysis", "P1", "4", 2, "-", "New table + relabelled components"),
        ("R3.M2", "Reviewer 3 minor 2 / Reviewer 4 comment 4", "Latency reporting is uncontrolled", "Re-measure on an idle GPU with warm-up and explicit synchronisation; report hardware and boundaries", "P1", "4", 1, "R3.3", "New latency table"),

        ("R3.M1", "Reviewer 3 minor 1", "Pixel recall translated into missed nuclei", "Describe false-negative rate as a pixel-level quantity only", "P2", "5", 0, "-", "Corrected text"),
        ("R2.4", "Reviewer 2 comment 4", "Whole-slide feasibility must not be implied", "Consistency pass across abstract, discussion and future work", "P2", "5", 0, "-", "Corrected text"),
        ("R2.1", "Reviewer 2 comment 1", "State that superiority is not statistically established", "Update once paired tests exist; keep an explicit statement either way", "P2", "5", 0, "STAT", "Corrected text"),
        ("R4.4a", "Reviewer 4 comment 4", "Report total parameters alongside trainable", "Add the total column to both cost tables", "P2", "5", 0, "-", "Updated Tables 6 and 8"),
        ("R3.M4", "Reviewer 3 minor 4", "Cite the named omics methods", "Add six citations with one or two sentences of context", "P2", "5", 0, "-", "Updated Related Work"),
        ("R4.2b", "Reviewer 4 comment 2", "Discuss SlideMamba", "Add to the fusion discussion in Related Work", "P2", "5", 0, "-", "Updated Related Work"),
        ("R4.3b", "Reviewer 4 comment 3", "Discuss SpatialQPFs and entropy measures", "Add to Section 5.6.2 and to Future Work", "P2", "5", 0, "-", "Updated text"),
        ("R3.M3", "Reviewer 3 minor 3", "Editorial consistency", "Check every cross-reference; state which tables are five-seed means; reconcile abstract, tables and archive", "P2", "6", 0, "all", "Final consistency pass"),
        ("DOC", "Submission requirement", "Point-by-point response letter", "One response per numbered comment, cross-referenced to the exact section and table", "P1", "6", 0, "all", "Response PDF"),
        ("HL", "Submission requirement", "Highlighted-changes manuscript", "Regenerate the highlighted diff from the final clean tex", "P2", "6", 0, "all", "Highlighted tex + PDF"),
    ]
    rows = [["ID", "Raised by", "The objection", "How we answer it", "Priority",
             "Phase", "GPU hours", "Depends on", "Deliverable", "Status"]]
    rows += [list(r) + ["NOT STARTED"] for r in reqs]
    sheet(wb, "01_Reviewer_Requirements", rows,
          widths=[8, 34, 46, 72, 9, 8, 11, 12, 44, 14], wrap_cols=(2, 3, 8),
          title_note="Master matrix. P0 blocks resubmission; P1 is demanded but survivable with a strong justification; "
                     "P2 is cheap and should simply be done.",
          fill_rules=lambda row, c, v: P_FILL.get(row[4]) if c == 4 and row[0] != "ID" else None)

    phases = [
        ("PHASE 1", "Text and admin fixes that need no GPU", "Days 1-2", 0,
         "E1-E5, X1-X3, R1.3. Everything that is a known-correct correction. Doing these first means the manuscript is already better even if compute slips.",
         "Corrected Methods, new figures, real Zenodo DOI, restructured sections"),
        ("PHASE 2", "Split design and harness", "Day 2", 0,
         "Build organ-stratified NuInsSeg 70/15/15 and tissue-stratified CryoNuSeg three-fold manifests. Extend the runner to take a test split and to log test metrics separately. Preflight every model.",
         "Split manifests + updated runner, verified on one short run"),
        ("PHASE 3", "The main training sweep", "Days 3-6, mostly unattended", 67,
         "NuInsSeg 7 models x 5 seeds. CryoNuSeg 7 models x 3 folds x 5 seeds. Ablation rows 1-6 plus A-F x 5 seeds. Prompted MedSAM. Reference-21 official code.",
         "Sheet 08 fully populated"),
        ("PHASE 4", "Analysis on the new results", "Day 7", 6,
         "Paired tests and bootstrap intervals. Latency re-measurement on an idle GPU. Grad-CAM failure panel. Clustering restructure, stability and confound checks. Component versus ground-truth instance comparison. PCA loadings.",
         "Sheets 09 and 06 filled, new figures and tables"),
        ("PHASE 5", "Manuscript rewrite", "Days 8-9", 0,
         "Rewrite every results section against the new numbers. Add the Limitations subsection. Fold Future Work into the Conclusion. Add the nine citations. Soften or strengthen claims to match what the statistics now support.",
         "New clean tex"),
        ("PHASE 6", "Response letter and submission package", "Day 10", 0,
         "Point-by-point response. Highlighted-changes version. Final consistency audit. Compile check on Overleaf.",
         "Submission-ready package"),
    ]
    rows = [["Phase", "What happens", "When", "GPU hours", "Contents", "Exit criterion"]]
    rows += [list(p) for p in phases]
    rows.append(["", "", "TOTAL", 73, "", ""])
    sheet(wb, "02_Schedule_By_Priority", rows,
          widths=[11, 42, 24, 12, 96, 46], wrap_cols=(1, 4, 5),
          title_note="Ordered so that nothing cheap waits on anything expensive, and so the GPU is never idle while text is being written.")

    figs = [
        (1, "EPCS-Net methodology pipeline", "EPCS_methodology.png", "REPLACE",
         "Figure_01_EPCS_Net_Methodology_Pipeline.excalidraw",
         "Redrawn in Excalidraw. Corrects the SAM geometry and adds the new three-way evaluation protocol."),
        (2, "Pyramid-Scale Prior Injection", "fig2_pspi_pyramid.png", "REPLACE",
         "Figure_02_PSPI_Pyramid_Scale_Prior_Injection.excalidraw",
         "Redrawn. States the 64x64 embedding explicitly."),
        (3, "Adaptive Cross-stream Fusion Block", "fig3_ACFB.png", "REPLACE",
         "Figure_03_ACFB_Adaptive_Crossstream_Fusion_Block.excalidraw",
         "Redrawn. Notes that GroupNorm is local to this block and that the residual is taken post-DKSR."),
        (4, "Full-Scale Fused Decoder", "fig4_fsfd.png", "REPLACE",
         "Figure_04_FSFD_Full_Scale_Fused_Decoder.excalidraw",
         "Redrawn. Channel arithmetic verified against the code."),
        (5, "CryoNuSeg qualitative results", "fig5_cryo_results.png", "REGENERATE",
         "n/a - generated from saved predictions",
         "Must be rebuilt from the new test-split predictions."),
        (6, "NuInsSeg qualitative results", "fig6_nuin_results.png", "REGENERATE",
         "n/a - generated from saved predictions",
         "Must be rebuilt from the new test-split predictions."),
        (7, "Grad-CAM overlays", "fig7_gradcam.png", "EXTEND",
         "n/a - generated from model outputs",
         "Add the low-overlap failure panel Reviewer 2 asked for twice."),
        (8, "PCA nuclear phenotyping", "fig8_pca.png", "REVISE",
         "n/a - generated from model outputs",
         "Component labels must follow the loading analysis, not precede it."),
        (9, "HDBSCAN clustering", "fig9_hdbscan.png", "REVISE",
         "n/a - generated from model outputs",
         "Must show the prespecified rule as primary and label the fixed-5 run as a sensitivity check."),
        (10, "Five-seed stability", "fig10_seed_stability.png", "REGENERATE",
         "n/a - replot of table values",
         "Extend to all models now that baselines have seeds."),
        (11, "Training curves", "fig11_training_curves.png", "REGENERATE",
         "n/a - replot of table values",
         "Rebuild from the new runs."),
        (12, "Headline comparison", "fig12_headline_comparison.png", "REGENERATE",
         "n/a - replot of table values",
         "Rebuild on test scores with error bars across seeds."),
        (13, "NEW: Grad-CAM failure cases", "-", "NEW",
         "n/a - generated from model outputs",
         "Lowest-overlap samples, both datasets. Answers Reviewer 2 comment 3."),
        (14, "NEW: component versus ground-truth instance", "-", "NEW",
         "n/a - generated from model outputs",
         "Shows how often connected components merge touching nuclei. Answers Reviewer 4 comment 3."),
    ]
    rows = [["Figure", "Title", "Current file", "Action", "Excalidraw source", "Why"]]
    rows += [list(f) for f in figs]
    sheet(wb, "03_Figure_Plan", rows,
          widths=[9, 42, 32, 14, 56, 72], wrap_cols=(5,),
          title_note="Only the four schematic figures become Excalidraw documents. The rest are plots of real model output and are "
                     "regenerated by analysis scripts, which is itself the answer to the editor's AI-image concern.",
          fill_rules=lambda row, c, v: (TODO_FILL if v in ("REPLACE", "NEW") else
                                        (OK_FILL if v in ("REGENERATE", "REVISE", "EXTEND") else None)) if c == 3 and row[0] != "Figure" else None)

    cites = [
        ("mclSTExp", "Reviewer 3 minor 4", "Related Work, omics paragraph", "Spatial transcriptomics representation learning"),
        ("SpaBatch", "Reviewer 3 minor 4", "Related Work, omics paragraph", "Batch effect correction for spatial omics"),
        ("SpaCross", "Reviewer 3 minor 4", "Related Work, omics paragraph", "Cross-sample spatial domain analysis"),
        ("TSCCA", "Reviewer 3 minor 4", "Related Work, omics paragraph", "Sparse canonical correlation for multi-omics"),
        ("scstGCN", "Reviewer 3 minor 4", "Related Work, omics paragraph", "Graph convolution for spatial transcriptomics"),
        ("SpaICL", "Reviewer 3 minor 4", "Related Work, omics paragraph", "In-context learning for spatial omics"),
        ("SlideMamba (Sci Rep 16, 4253, 2026)", "Reviewer 4 comment 2", "Related Work, fusion paragraph",
         "Entropy-guided adaptive fusion of GNN and Mamba. Directly comparable in spirit to ACFB's adaptive weighting, so it belongs beside our fusion discussion."),
        ("SpatialQPFs (Sci Rep 14, 29585, 2024)", "Reviewer 4 comment 3", "Section 5.6.2 and Future Work",
         "Interpretable spatial statistical features. Complements DPNP by describing cell-to-cell spatial relationships rather than single-region morphology."),
        ("Entropy measures (iScience 28(6), 112765, 2025)", "Reviewer 4 comment 3", "Section 5.6.2 and Future Work",
         "Entropy measures for tissue complexity and spatial heterogeneity."),
    ]
    rows = [["Reference", "Requested by", "Where it goes", "What to say about it", "Status"]]
    rows += [list(c) + ["NOT ADDED"] for c in cites]
    sheet(wb, "04_Citations_To_Add", rows,
          widths=[42, 24, 36, 86, 14], wrap_cols=(3,),
          title_note="Nine citations named by reviewers. The last three need real discussion, not a bare citation, because each "
                     "reviewer explains why the work is relevant.")

    edits = [
        ("Abstract", "Update every headline number to the test-split five-seed result; state the split design; state that significance was tested", "P0"),
        ("Section 3 -> Section 4", "Fold Datasets and Preprocessing into Methodology per the editor", "P0"),
        ("Section 3.1", "Correct the CryoNuSeg tissue list; five named tissues are not in the dataset", "P0"),
        ("Section 3.1 / 3.2", "Describe the new three-way and cross-validated split design, with per-tissue and per-organ counts", "P0"),
        ("Section 4.2 (OSPT)", "State the sampler exactly; add per-organ counts; disclose that OSPT is NuInsSeg-only", "P0"),
        ("Section 4.5 (SAM)", "Correct the input geometry: 256 upsampled to 1024, official preprocess, 64x64 token grid", "P0"),
        ("Equation for NCAP", "Dimension is B x 256 x 64 x 64, not B x 256 x H/8 x W/8", "P1"),
        ("Section 4.4 / 4.6.5", "Clarify that GroupNorm is used inside ACFB only; the encoder uses BatchNorm; cite the new ablation", "P1"),
        ("Section 4.6 (ACFB)", "Reconcile equation 14 with the code, which takes the residual on the post-DKSR tensor", "P1"),
        ("Section 4.10", "Describe checkpoint selection on validation and reporting on test, explicitly", "P0"),
        ("Section 5.x (all results)", "Rewrite against test-split numbers with intervals; add the reference-21 baseline row", "P0"),
        ("Section 5.x (cost)", "Add total parameters; replace the latency table with the controlled re-measurement", "P1"),
        ("Section 5.6.x (ablation)", "Replace with the multi-seed rerun; remove 'every component earns its place' unless intervals support it", "P0"),
        ("Section 5.6.x (MedSAM)", "Move out of the competitive comparison into a labelled zero-shot sanity check; add the prompted configuration", "P1"),
        ("Section 5.6.x (Grad-CAM)", "Add the failure-case panel and discuss it", "P1"),
        ("Section 5.6.x (DPNP)", "Call the outputs exploratory connected-region features; add the instance comparison; support or drop the PC labels", "P1"),
        ("Section 5.6.x (HDBSCAN)", "Prespecified rule primary, fixed-5 explicitly a sensitivity check; silhouette is not biological validation", "P0"),
        ("Section 6 (Discussion)", "Rewrite the margin discussion around the paired tests", "P0"),
        ("Section 6.x (NEW Limitations)", "New subsection per the editor, collecting every limitation in one place", "P0"),
        ("Section 7 (Conclusion)", "Update numbers; append Future Work here and delete Section 8", "P0"),
        ("Related Work", "Add the nine citations; state precisely how EPCS-Net differs from reference 21", "P1"),
        ("Code Availability", "Replace the placeholder DOI with the real Zenodo DOI for the tagged release", "P0"),
        ("All figures", "Re-export from Excalidraw and the analysis scripts; add a provenance statement", "P0"),
        ("Whole document", "Cross-reference audit; confirm which tables are five-seed means; remove the pixel-to-nuclei translation", "P2"),
    ]
    rows = [["Location", "Edit required", "Priority", "Done?"]]
    rows += [list(e) + ["NO"] for e in edits]
    sheet(wb, "05_Manuscript_Edit_Checklist", rows,
          widths=[34, 104, 10, 9], wrap_cols=(1,),
          title_note="Every manuscript location that must change, in document order.",
          fill_rules=lambda row, c, v: P_FILL.get(row[2]) if c == 2 and row[0] != "Location" else None)

    sheet(wb, "00_README", [
        ["Sheet", "Purpose"],
        ["01_Reviewer_Requirements", "Every objection from the editor and all four reviewers, with our answer, priority, phase and GPU cost."],
        ["02_Schedule_By_Priority", "Six phases, ordered so cheap work never waits on expensive work."],
        ["03_Figure_Plan", "What happens to each of the twelve current figures, plus two new ones."],
        ["04_Citations_To_Add", "The nine references reviewers named, and what to say about each."],
        ["05_Manuscript_Edit_Checklist", "Every edit, in document order, with a done column."],
    ], widths=[34, 104], wrap_cols=(1,),
        title_note="EPCS-Net revision 3 task schedule.")
    wb.move_sheet("00_README", offset=-5)

    wb.save(path)
    print(f"  wrote {path.name}  ({len(wb.sheetnames)} sheets)")


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    out.mkdir(parents=True, exist_ok=True)
    print(f"Writing workbooks to {out}")
    build_results(out / "EPCS_R3_Results_Comparison.xlsx")
    build_schedule(out / "EPCS_R3_Task_Schedule.xlsx")
    print("Done.")


if __name__ == "__main__":
    main()
