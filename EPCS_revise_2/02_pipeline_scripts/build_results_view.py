"""Builds the clean, separated, human-facing view of the round-3 deliverables:

    <project root>/NEW_RESULTS_R3/
        00_READ_ME_FIRST.md        what is here, final vs provisional, comment-to-file map
        01_master_and_metrics/     master workbook, full metrics, epoch workbooks, paper tables, comparison workbooks
        02_reviewer_comment_tracking/  coverage workbook, fulfilment audit, verification report, comments (summary + verbatim), alignment check
        03_manuscript_and_response/    tex, highlighted tex, Overleaf zip, response PDF/docx
        04_new_figures/                figures 1-4 and result figures
        05_analysis_outputs/           statistics, significance, OSPT, MedSAM, DPNP/Grad-CAM outputs (files < 5 MB)
        06_plan_handoff_tracker/       handoff, tracker, queue, findings, audit

Files are COPIES (nothing under 07_revision3 is moved or renamed, because scripts and the running chain use those
paths). The view is rebuilt from scratch on every run, so it never goes stale: rerun it after every evaluation.
Run:  python 07_revision3/06_scripts/build_results_view.py
"""
import shutil
from datetime import datetime
from pathlib import Path

R3 = Path(__file__).resolve().parents[1]
ROOT = R3.parent
OUT = ROOT / "NEW_RESULTS_R3"
MAX_BYTES = 5 * 1024 * 1024

SECTIONS = {
    "01_master_and_metrics": [
        (R3 / "EPCS_R3_MASTER_Results.xlsx", None),
        (R3 / "04_results" / "EPCS_R3_Full_Metrics.xlsx", None),
        (R3 / "04_results" / "epoch_workbooks", "*.xlsx"),
        (R3 / "04_results" / "paper_tables", "*.xlsx"),
        (R3 / "02_comparison_workbooks", "*.xlsx"),
        (R3 / "04_results" / "RESULTS_INDEX.md", None),
        (R3 / "04_results" / "latency", "*.csv"),
        (R3 / "04_results" / "latency", "*.json"),
    ],
    "02_reviewer_comment_tracking": [
        (R3 / "00_plan" / "EPCS_R3_Reviewer_Comment_Coverage.xlsx", None),
        (R3 / "00_plan" / "FULFILMENT_AUDIT_R3.md", None),
        (R3 / "00_plan" / "VERIFICATION_REPORT_R3.md", None),
        (R3 / "00_plan" / "ALIGNMENT_CHECK.md", None),
        (R3 / "00_plan" / "CHAIN_OUTPUT_VERIFICATION.md", None),
        (R3 / "00_plan" / "REVIEWER_COMMENTS_ROUND3.md", None),
        (R3 / "05_manuscript" / "response" / "REVIEWER_COMMENTS_VERBATIM.md", None),
    ],
    "03_manuscript_and_response": [
        (R3 / "05_manuscript" / "EPCS_revise3.tex", None),
        (R3 / "05_manuscript" / "package", "*"),
        (R3 / "05_manuscript" / "response", "*.pdf"),
        (R3 / "05_manuscript" / "response", "*.docx"),
        (R3 / "05_manuscript" / "RESPONSE_DRAFT_R3.md", None),
        (R3 / "05_manuscript" / "new_references_R3.bib", None),
    ],
    "04_new_figures": [
        (R3 / "05_manuscript" / "figures", "*.png"),
        (R3 / "05_manuscript" / "figures", "*.svg"),
        (R3 / "05_manuscript" / "figures_r3", "*.png"),
        (R3 / "05_manuscript" / "figures_r3", "*.svg"),
    ],
    "05_analysis_outputs": [
        (R3 / "04_results" / "analysis", "*.csv"),
        (R3 / "04_results" / "analysis", "*.json"),
        (R3 / "04_results" / "analysis", "dpnp_*"),
        (R3 / "04_results" / "medsam_r3" / "summary_all.json", None),
    ],
    "06_plan_handoff_tracker": [
        (ROOT / "HANDOFF.md", None),
        (ROOT / "README.md", None),
        (R3 / "00_plan" / "TASK_TRACKER.md", None),
        (R3 / "00_plan" / "EPCS_R3_Task_Tracker.xlsx", None),
        (R3 / "00_plan" / "QUEUE.md", None),
        (R3 / "00_plan" / "FINDINGS_LOG.md", None),
        (R3 / "00_plan" / "CODE_REVIEW_R3.md", None),
        (R3 / "00_plan" / "IMPLEMENTATION_PLAN.md", None),
    ],
}


def collect(src: Path, pattern):
    if src.is_file():
        return [src]
    if src.is_dir():
        return sorted(p for p in src.glob(pattern or "*") if p.is_file())
    return []


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    listing, skipped, seen = {}, [], set()
    for sec, items in SECTIONS.items():
        d = OUT / sec
        d.mkdir()
        listing[sec] = []
        for src, pat in items:
            for f in collect(src, pat):
                if f in seen or "validation_runA" in f.name:
                    continue
                seen.add(f)
                if f.stat().st_size > MAX_BYTES:
                    skipped.append((f.name, f.stat().st_size // 1024))
                    continue
                if any(part.startswith("_") for part in f.parts[len(R3.parts):]):
                    continue            # _dry_run, _cpu_smoketests, _archive...: never cited
                dest = d / f.name
                if dest.exists():
                    dest = d / f"{f.parent.name}__{f.name}"
                shutil.copy2(f, dest)
                listing[sec].append(dest.name)

    lines = [f"# NEW RESULTS, ROUND 3 (view rebuilt {datetime.now():%Y-%m-%d %H:%M})", "",
             "Everything in this folder is a COPY of the newest round-3 deliverables. Rounds 1-2 are untouched in the "
             "`OLD_ROUNDS_1_2/` at the project root (`02_code_*`, `03_results_*`, `06_manuscript_revision`, old logs). "
             "Source of truth for every file is under `07_revision3/`. Rebuild with "
             "`python 07_revision3/06_scripts/build_results_view.py`.", "",
             "## Final or provisional?",
             "* **All experiments are FINAL**: CryoNuSeg (8 models x 5 seeds x 3 folds, ablation, MedSAM), NuInsSeg (5 seeds, fixed test split), latency (idle GPU), Grad-CAM panels and test-split phenotyping (both datasets).",
             "* **Still pending**: the Zenodo DOI (`%%TODO-ZENODO-DOI`) and a full LaTeX compile check (e.g. on Overleaf).",
             "* The Overleaf zip carries _DRAFT in its name only while a %%TODO marker remains.",
             "* `02_reviewer_comment_tracking/CHAIN_OUTPUT_VERIFICATION.md` records verification of the post-sweep chain outputs.", "",
             "## Where to look",
             "| You want | Open |", "|---|---|",
             "| every number, per run / seed / fold / average, with commentary vs previous rounds | `01_master_and_metrics/EPCS_R3_MASTER_Results.xlsx` |",
             "| all six metrics (accuracy, precision, recall, specificity, Dice, IoU), mean/SD/CI | `01_master_and_metrics/EPCS_R3_Full_Metrics.xlsx` |",
             "| per-epoch values and best epochs for every run | `01_master_and_metrics/EPCS_R3_EpochHistories_*.xlsx`, `EPCS_R3_BestEpochs_*.xlsx` |",
             "| is each reviewer comment fulfilled? | `02_reviewer_comment_tracking/EPCS_R3_Reviewer_Comment_Coverage.xlsx`, `FULFILMENT_AUDIT_R3.md` |",
             "| are we doing the right things? | `02_reviewer_comment_tracking/ALIGNMENT_CHECK.md` |",
             "| what is still to do, in order | `06_plan_handoff_tracker/QUEUE.md`, `TASK_TRACKER.md` |",
             "| new manuscript, marked-up manuscript, response letter | `03_manuscript_and_response/` |",
             "| new figures | `04_new_figures/` |",
             "| statistics, OSPT table, MedSAM, other analysis csvs | `05_analysis_outputs/` |", "",
             "## Comment-to-evidence map",
             "| Comment | Evidence file(s) |", "|---|---|",
             "| R3-M1, R4-1 (held-out test) | master workbook `Master_PerRun` (test_* columns), tex section Evaluation Protocol |",
             "| R1-1, R2-1, R3-M3 (matched seeds, significance) | `paired_significance_tests.csv`, `fold_averaged_tests.csv`, `bootstrap_per_image.csv` |",
             "| R1-2, R3-M4 (ablation) | master workbook `Ablation`, `figures_r3/r3_ablation_5seed` |",
             "| R4-2 (reference 21) | master workbook (models ref21_*), tex subsection on reference 21 |",
             "| R1-4 (MedSAM) | `medsam_r3_summary_all.json`, master workbook `MedSAM` |",
             "| R1-3 (OSPT) | `ospt_per_organ_sampling_nuinsseg.csv`, tex table OSPT |",
             "| R2-3 (Grad-CAM failure panel) | `04_new_figures/fig13_gradcam_*_{nuinsseg,cryonuseg}.png` |",
             "| R3-m2, R4-4 (latency) | `01_master_and_metrics/latency_r3.csv`, `latency_r3.json` |",
             "| R3-M5, R4-3 (phenotyping) | `dpnp_*`, `instance_vs_component_*`, `feature_bias_*` csvs (`*_r3_*` = test split, both datasets), figures `04_new_figures/r3_dpnp_*` |",
             "| Editor (response PDF) | `03_manuscript_and_response/EPCS_R3_Response_to_Reviewers_DRAFT.pdf` |", "",
             "## Never cite",
             "`_cpu_smoketests`, `medsam_r3/_dry_run`, `_archive_oldGPU_RTX4500Ada_partial`, `_chain_dryrun` (excluded from this view), "
             "and any file named `*validation_runA*` (old round-1/2 predictions).", "",
             "## Contents"]
    for sec, names in listing.items():
        lines += ["", f"### {sec}"] + [f"* {n}" for n in names]
    if skipped:
        lines += ["", "### Not copied (over 5 MB; see 07_revision3/04_results/analysis)"] + [f"* {n} ({kb} KB)" for n, kb in skipped]
    (OUT / "00_READ_ME_FIRST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("view built:", OUT, "|", sum(len(v) for v in listing.values()), "files; skipped", len(skipped))


if __name__ == "__main__":
    main()
