"""
Build the paper-ready tables for the revision.

Produces, in 04_results/paper_tables/:
  EPCS_R3_Paper_Tables.xlsx   one sheet per manuscript table, formatted so a
                              human can read and judge it directly
  latex/*.tex                 the same tables as LaTeX, ready to paste into the
                              manuscript with no retyping
  OLD_VS_NEW.xlsx             every previously published headline number beside
                              its replacement, which is what the response letter
                              needs

Design rules
------------
* Every cell traces to a file under 04_results. Nothing is typed by hand.
* Where a number replaces a previously published one, BOTH are shown. The point
  of this revision is that the numbers changed and we say so.
* Scores are test-split scores. Validation scores appear only where explicitly
  labelled, because mixing the two is the error the reviewers objected to.
* Tables regenerate safely while the sweep is still running; incomplete blocks
  are written with a clear "pending" marker rather than silently omitted.

Usage:
    python build_paper_tables.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

R3 = Path(__file__).resolve().parents[2]
RESULTS = R3 / "04_results"
ANALYSIS = RESULTS / "analysis"
OUT = RESULTS / "paper_tables"
TEX = OUT / "latex"

HDR = PatternFill("solid", fgColor="1F4E79")
HDRF = Font(bold=True, color="FFFFFF", size=11)
SUB = PatternFill("solid", fgColor="DDEBF7")
BEST = PatternFill("solid", fgColor="C6EFCE")
OLDF = PatternFill("solid", fgColor="F2F2F2")
PEND = PatternFill("solid", fgColor="FFF2CC")
THIN = Side(style="thin", color="999999")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

# Previously published headline numbers, for the side-by-side comparison.
# Source: OLD_ROUNDS_1_2/EPCS_Master_Results_ForPaper.csv (validation split, single seed except
# EPCS-Net which was five seeds).
PUBLISHED = {
    ("cryonuseg", "unet"): 0.7768, ("cryonuseg", "unetpp"): 0.7479,
    ("cryonuseg", "unet3plus"): 0.7737, ("cryonuseg", "eunet3plus"): 0.7909,
    ("cryonuseg", "sam_unet"): 0.8126, ("cryonuseg", "epcs_net"): 0.8460,
    ("nuinsseg", "unet"): 0.6011, ("nuinsseg", "unetpp"): 0.5872,
    ("nuinsseg", "unet3plus"): 0.5798, ("nuinsseg", "eunet3plus"): 0.6099,
    ("nuinsseg", "sam_unet"): 0.5489, ("nuinsseg", "epcs_net"): 0.6957,
}
PUBLISHED_ABLATION = [
    ("abl1_nosam", "eU-Net3+ only (no SAM, no ACFB)", 0.7173, 0.5592),
    ("abl2_bottleneck", "+ SAM bottleneck fusion", 0.7659, 0.6266),
    ("abl3_pspi", "+ PSPI (four-scale)", 0.7990, 0.6614),
    ("abl4_bdsg", "+ ACFB bidirectional gating", 0.8090, 0.6751),
    ("abl5_dksr", "+ dual-kernel refinement", 0.8120, 0.6840),
    ("abl6_ftas", "+ FTAS loss (full EPCS-Net)", 0.8371, 0.7220),
]
ORDER = ["unet", "unetpp", "unet3plus", "eunet3plus", "sam_unet",
         "ref21_bottleneck", "ref21_pyramid", "epcs_net"]
METRICS = ["accuracy", "precision", "recall", "specificity", "dice", "iou"]


def sheet(wb, name, rows, widths=None, note=None, highlight_best_col=None,
          fills=None):
    ws = wb.create_sheet(name[:31])
    r0 = 1
    if note:
        c = ws.cell(1, 1, note)
        c.font = Font(italic=True, size=9, color="444444")
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=1, start_column=1, end_row=1,
                       end_column=max(2, len(rows[0]) if rows else 2))
        ws.row_dimensions[1].height = 42
        r0 = 2
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = ws.cell(r0 + i, j + 1, val)
            cell.border = BOX
            if i == 0:
                cell.fill, cell.font = HDR, HDRF
                cell.alignment = Alignment(wrap_text=True, horizontal="center",
                                           vertical="center")
            else:
                cell.alignment = Alignment(vertical="center",
                                           horizontal="left" if j == 0 else "center")
                if fills:
                    f = fills(row, j, val)
                    if f:
                        cell.fill = f
    if highlight_best_col is not None and len(rows) > 1:
        col = highlight_best_col
        vals = [(i, r[col]) for i, r in enumerate(rows[1:], start=1)
                if isinstance(r[col], (int, float))]
        if vals:
            bi = max(vals, key=lambda t: t[1])[0]
            for j in range(len(rows[0])):
                ws.cell(r0 + bi, j + 1).fill = BEST
    if widths:
        for i, w in enumerate(widths):
            ws.column_dimensions[get_column_letter(i + 1)].width = w
    ws.freeze_panes = f"A{r0 + 1}"
    ws.row_dimensions[r0].height = 34
    return ws


def latex_table(path: Path, caption: str, label: str, header: list[str],
                body: list[list], note: str | None = None, align: str | None = None):
    TEX.mkdir(parents=True, exist_ok=True)
    cols = align or ("l" + "c" * (len(header) - 1))

    def esc(t):
        return (str(t).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")
                .replace("#", r"\#").replace("±", r"$\pm$"))

    def fmt(v):
        if isinstance(v, float):
            return "n/a" if v != v else f"{v:.4f}"
        t = esc(v)
        return "n/a" if t.lower() == "nan" else t

    lines = [r"\begin{table}[H]", r"\centering",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\resizebox{\textwidth}{!}{%",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(rf"\textbf{{{esc(h)}}}" for h in header) + r" \\", r"\midrule"]
    for row in body:
        if row == RULE:
            lines.append(r"\midrule")
            continue
        assert len(row) == len(header), (path.name, row)
        lines.append(" & ".join(fmt(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    if note:
        lines.append(rf"\par\vspace{{2pt}}\footnotesize{{{esc(note)}}}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines), encoding="utf-8")


RULE = "__RULE__"


def load_medsam() -> dict:
    """Aggregate MedSAM summaries: {(dataset, mode): dict(metric -> (mean, sd), n)}."""
    p = RESULTS / "medsam_r3" / "summary_all.json"
    if not p.is_file():
        return {}
    out = {}
    for r in json.loads(p.read_text(encoding="utf-8")):
        if r.get("dry_run") or "metric_mean" not in r:
            continue
        ds = "cryonuseg" if str(r["dataset"]).startswith("cryonuseg") else r["dataset"]
        out.setdefault((ds, r["mode"]), []).append(r)
    res = {}
    for k, rs in out.items():
        n = sum(r["n_test_images"] for r in rs)
        d = {"n": n, "parts": len(rs)}
        for m in METRICS:
            mu = sum(r["n_test_images"] * r["metric_mean"][m] for r in rs) / n
            var = sum((r["n_test_images"] - 1) * r["metric_sample_sd"][m] ** 2 +
                      r["n_test_images"] * (r["metric_mean"][m] - mu) ** 2 for r in rs) / max(n - 1, 1)
            d[m] = (mu, var ** 0.5)
        res[k] = d
    return res


def pmci(mean, sd, lo, hi, n_runs) -> str:
    if pd.isna(mean):
        return "pending"
    if n_runs < 2 or pd.isna(sd):
        return f"{mean:.4f} (n=1)"
    s = f"{mean:.4f} ± {sd:.4f}"
    if pd.notna(lo) and pd.notna(hi):
        s += f" [{lo:.4f}, {hi:.4f}]"
    return s


def load(name: str) -> pd.DataFrame | None:
    p = ANALYSIS / name
    return pd.read_csv(p) if p.is_file() else None


def pm(mean, sd, dp=4) -> str:
    if pd.isna(mean):
        return "pending"
    return f"{mean:.{dp}f} ± {sd:.{dp}f}" if pd.notna(sd) else f"{mean:.{dp}f}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    TEX.mkdir(parents=True, exist_ok=True)
    agg = load("aggregated_by_model.csv")
    tests = load("paired_significance_tests.csv")
    boot = load("bootstrap_per_image.csv")
    organ = load("per_organ_test_dice.csv")
    runs = load("all_runs.csv")

    if agg is None:
        print("No aggregated results yet. Run analysis/stats_r3.py first "
              "(it needs at least one completed run).")
        return 1

    wb = Workbook()
    wb.remove(wb.active)
    built = []

    # ---------- per-dataset quantitative tables ----------
    for ds, tname in (("cryonuseg", "CryoNuSeg"), ("nuinsseg", "NuInsSeg")):
        g = agg[(agg.dataset == ds) & (agg.group != "ablation") &
                (agg.group != "ablation_control")]
        if g.empty:
            continue
        g = g.set_index("model").reindex([m for m in ORDER if m in set(g.model)])
        header = ["Model", "Runs", "Accuracy", "Precision", "Recall",
                  "Specificity", "Dice (test)", "IoU (test)",
                  "Params total (M)", "Params trainable (M)",
                  "Published Dice (val)", "Change"]
        body = []
        for m, r in g.iterrows():
            old = PUBLISHED.get((ds, m))
            n_r = int(r["n_runs"])
            cells = [pmci(r[f"test_{k}_mean"], r[f"test_{k}_sd"],
                          r.get(f"test_{k}_ci_low"), r.get(f"test_{k}_ci_high"), n_r)
                     for k in METRICS]
            pt, ptr = r["params_total"], r["params_trainable"]
            body.append([r["label"], n_r] + cells +
                        [f"{pt/1e6:.2f}" if pd.notna(pt) else "n/a",
                         f"{ptr/1e6:.2f}" if pd.notna(ptr) else "n/a",
                         f"{old:.4f}" if old else "n/a",
                         f"{r['test_dice_mean']-old:+.4f}" if old else "n/a"])
        ms = load_medsam()
        ms_rows = []
        for mode, lab in (("zero_shot_wholebox", "MedSAM zero-shot, whole-image box (sanity check)"),
                          ("oracle_boxes", "MedSAM with ground-truth component boxes (ORACLE upper bound, not a competitor)")):
            d = ms.get((ds, mode))
            if d:
                ms_rows.append([lab, f"{d['n']} img"] +
                               [f"{d[k][0]:.4f} ± {d[k][1]:.4f}" for k in METRICS] +
                               ["n/a (frozen)", "0.00", "n/a", "n/a"])
        if ms_rows:
            body = body + [RULE] + ms_rows
        note = (f"{tname}: every score is on the untouched TEST split; the checkpoint "
                f"was selected on validation only. Mean ± SD across matched seeds"
                + (" and folds" if ds == "cryonuseg" else "") +
                ". The 'Published' column is the previously reported VALIDATION Dice "
                "and is shown for transparency, not as a like-for-like comparison.")
        xl_rows = [["Not part of the ranking: external reference rows"] +
                   [""] * (len(header) - 1) if row == RULE else row for row in body]
        ws_q = sheet(wb, f"T_{tname}_quantitative", [header] + xl_rows,
                     widths=[44, 8] + [34] * 6 + [16, 18, 15, 10], note=note)
        for i, row in enumerate(xl_rows):
            if str(row[0]).startswith("Not part of the ranking"):
                for j in range(len(header)):
                    ws_q.cell(3 + i, j + 1).fill = SUB
                    ws_q.cell(3 + i, j + 1).font = Font(bold=True, italic=True)
        latex_table(TEX / f"table_{ds}_quantitative.tex",
                    f"Segmentation performance on the {tname} test split. "
                    f"Mean $\\pm$ SD across matched seeds"
                    + (" and cross-validation folds" if ds == "cryonuseg" else "") +
                    ", with 95\% CI in brackets (t-interval over runs). Rows below the rule "
                    "are MedSAM reference points (zero-shot sanity check and oracle-box upper "
                    "bound), excluded from the ranking; their SD is across test images. "
                    "Checkpoints were selected on validation only.",
                    f"tab:{ds}_r3",
                    header[:-2], [b if b == RULE else b[:-2] for b in body],
                    note="Scores are test-split scores; no model saw the test split "
                         "during training or checkpoint selection.")
        built.append(f"{tname} quantitative")

    # ---------- computational cost ----------
    lat = None
    lp = RESULTS / "latency" / "latency_r3.csv"
    if lp.is_file():
        lat = pd.read_csv(lp).set_index("model")
    for ds, tname in (("cryonuseg", "CryoNuSeg"), ("nuinsseg", "NuInsSeg")):
        g = agg[(agg.dataset == ds) & (~agg.group.isin(["ablation", "ablation_control"]))]
        if g.empty:
            continue
        g = g.set_index("model").reindex([m for m in ORDER if m in set(g.model)])
        header = ["Model", "Params total (M)", "Params trainable (M)",
                  "Frozen (M)", "ms / image", "Median ms", "Images / s",
                  "Peak VRAM (MiB)"]
        body = []
        for m, r in g.iterrows():
            tot, tr = r["params_total"] / 1e6, r["params_trainable"] / 1e6
            if lat is not None and m in lat.index:
                lr = lat.loc[m]
                body.append([r["label"], round(tot, 4), round(tr, 4), round(tot - tr, 4),
                             round(float(lr["ms_mean"]), 2), round(float(lr["ms_median"]), 2),
                             round(float(lr["images_per_second"]), 1),
                             round(float(lr["peak_vram_mib"]), 1)])
            else:
                body.append([r["label"], round(tot, 4), round(tr, 4), round(tot - tr, 4),
                             "pending", "pending", "pending", "pending"])
        note = ("Latency measured separately on an IDLE GPU with 20 warm-up iterations "
                "discarded and 100 timed iterations, CUDA-synchronised either side of the "
                "forward pass, data loading excluded, batch size 1, FP32. The previously "
                "published latency figures were taken during training with no warm-up and "
                "with concurrent GPU jobs, which is why they were not comparable.")
        sheet(wb, f"T_{tname}_cost", [header] + body,
              widths=[34, 18, 20, 13, 13, 13, 13, 17], note=note,
              fills=lambda row, j, v: PEND if v == "pending" else None)
        latex_table(TEX / f"table_{ds}_cost.tex",
                    f"Computational cost on {tname}: parameter counts and controlled "
                    f"inference latency.", f"tab:{ds}_cost_r3", header, body,
                    note="Latency measured on an idle GPU with warm-up; see Methods.")
        built.append(f"{tname} cost")

    # ---------- per-run detail ----------
    if runs is not None and not runs.empty:
        header = ["Dataset", "Model", "Fold", "Seed", "Best epoch",
                  "Validation Dice", "TEST Dice", "TEST IoU"]
        body = [[r["dataset"], r["label"], r["fold"] if pd.notna(r["fold"]) else "-",
                 r["seed"], r["best_epoch"], round(r["best_validation_dice"], 4),
                 round(r["test_dice"], 4), round(r["test_iou"], 4)]
                for _, r in runs.sort_values(
                    ["dataset", "model", "fold", "seed"]).iterrows()]
        sheet(wb, "T_per_run_detail", [header] + body,
              widths=[12, 32, 7, 8, 12, 17, 12, 12],
              note="Every individual run behind the aggregated tables. This is the "
                   "matched-seed evidence the reviewers asked for.")
        built.append("per-run detail")

    # ---------- significance ----------
    if tests is not None and not tests.empty:
        header = ["Dataset", "Comparison", "Pairs", "EPCS-Net", "Baseline",
                  "Difference", "95% CI", "Cohen's d", "t p (Holm)",
                  "Wilcoxon p (Holm)", "Significant"]
        body = []
        for _, r in tests.sort_values(["dataset", "mean_difference"],
                                      ascending=[True, False]).iterrows():
            body.append([r["dataset"], r["baseline_label"], int(r["n_pairs"]),
                         round(r["proposed_mean"], 4), round(r["baseline_mean"], 4),
                         round(r["mean_difference"], 4),
                         f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]",
                         round(r["cohens_d_paired"], 3), round(r["t_p_holm"], 4),
                         round(r["wilcoxon_p_holm"], 4),
                         "yes" if r["significant_t_holm_0.05"] else "no"])
        note = ("Paired across matched seeds on byte-identical splits, Holm-corrected "
                "within each dataset. NOTE: with five pairs the Wilcoxon signed-rank "
                "p-value cannot fall below 0.0625, so it cannot reach 0.05 regardless of "
                "effect size. That is a property of the sample size and is reported "
                "rather than hidden by quoting only the t-test.")
        sheet(wb, "T_significance", [header] + body,
              widths=[12, 32, 7, 12, 12, 12, 20, 11, 12, 18, 12], note=note)
        latex_table(TEX / "table_significance.tex",
                    "Paired comparisons between EPCS-Net and each baseline on matched "
                    "seeds and identical splits, Holm-corrected within dataset.",
                    "tab:significance_r3", header, body,
                    note="With five pairs the Wilcoxon p-value has a floor of 0.0625.")
        built.append("significance")

    if boot is not None and not boot.empty:
        header = ["Dataset", "Baseline", "Paired images", "Mean difference",
                  "Bootstrap 95% CI", "Excludes zero", "Win rate"]
        body = [[r["dataset"], r["baseline"], int(r["n_paired_images"]),
                 round(r["mean_difference"], 4),
                 f"[{r['boot_ci95_low']:.4f}, {r['boot_ci95_high']:.4f}]",
                 "yes" if r["excludes_zero"] else "no",
                 f"{r['win_rate']*100:.1f}%"] for _, r in boot.iterrows()]
        sheet(wb, "T_bootstrap", [header] + body,
              widths=[12, 28, 15, 16, 24, 15, 11],
              note="Paired bootstrap over per-image test scores (10,000 resamples). "
                   "This measures variation across IMAGES, whereas the significance "
                   "table measures variation across TRAINING RUNS. Both are reported "
                   "because they answer different questions.")
        built.append("bootstrap")

    # ---------- ablation ----------
    ab = agg[agg.group.isin(["ablation", "ablation_control"])]
    if not ab.empty:
        header = ["#", "Variant", "Runs", "Dice (test)", "IoU (test)",
                  "Dice 95% CI", "Published single run", "Change"]
        body = []
        pub = {m: (d, i) for m, _, d, i in PUBLISHED_ABLATION}
        for m, r in ab.set_index("model").iterrows():
            od = pub.get(m, (None, None))[0]
            ci = (f"[{r['test_dice_ci_low']:.4f}, {r['test_dice_ci_high']:.4f}]"
                  if pd.notna(r.get("test_dice_ci_low")) else "n/a")
            body.append([r["label"].split(".")[0] if "." in r["label"] else "",
                         r["label"], int(r["n_runs"]),
                         pmci(r["test_dice_mean"], r["test_dice_sd"], None, None, int(r["n_runs"])),
                         pmci(r["test_iou_mean"], r["test_iou_sd"], None, None, int(r["n_runs"])), ci,
                         f"{od:.4f}" if od else "new in this revision",
                         f"{r['test_dice_mean']-od:+.4f}" if od else "n/a"])
        note = ("Every row rerun under the CURRENT implementation and protocol across "
                "five seeds, replacing the previous single-run table taken from a "
                "separate earlier training pass. Rows A to D are the controls the "
                "reviewers asked for and did not exist before. Where a confidence "
                "interval overlaps the row above it, the increment is NOT established.")
        sheet(wb, "T_ablation", [header] + body,
              widths=[5, 40, 7, 18, 18, 22, 20, 11], note=note)
        latex_table(TEX / "table_ablation.tex",
                    "Component ablation on CryoNuSeg, rerun under the current "
                    "implementation across five seeds with 95\\% confidence intervals.",
                    "tab:ablation_r3", header, body,
                    note="Increments whose intervals overlap are not established.")
        built.append("ablation")

    # ---------- per-organ ----------
    if organ is not None and not organ.empty:
        piv = organ[organ.model == "epcs_net"].pivot_table(
            index=["dataset", "organ"], values=["mean", "sd", "n"]).reset_index()
        header = ["Dataset", "Organ / tissue", "Test images", "Mean Dice", "SD"]
        body = [[r["dataset"], r["organ"], int(r["n"]), round(r["mean"], 4),
                 round(r["sd"], 4) if pd.notna(r["sd"]) else "n/a"]
                for _, r in piv.iterrows()]
        sheet(wb, "T_per_organ", [header] + body, widths=[12, 38, 13, 12, 10],
              note="EPCS-Net test Dice broken down by organ. A multi-tissue claim "
                   "needs this, not a single pooled average.")
        built.append("per-organ")

    # ---------- DPNP and instance tables ----------
    for tag, title in (("validation_runA", "NuInsSeg"), ("validation_runA_cryo", "CryoNuSeg")):
        rp = ANALYSIS / f"dpnp_report_{tag}.json"
        if not rp.is_file():
            continue
        d = json.loads(rp.read_text(encoding="utf-8"))
        p, s = d["clustering_primary_prespecified"], d["clustering_sensitivity_fixed5"]
        st = d.get("sensitivity_stability") or {}
        header = ["Configuration", "Role", "min_cluster_size", "Clusters",
                  "Outliers %", "Silhouette", "Stability (ARI)", "Cluster count range"]
        body = [
            ["max(5, N/10)", "PRESPECIFIED, primary", p["min_cluster_size"],
             p["n_clusters"], round(p["outlier_pct"], 2),
             p["silhouette_geometric_separation"] or "n/a", "n/a", "n/a"],
            ["fixed 5", "sensitivity check only", s["min_cluster_size"],
             s["n_clusters"], round(s["outlier_pct"], 2),
             round(s["silhouette_geometric_separation"], 4)
             if s["silhouette_geometric_separation"] else "n/a",
             f"{st.get('ari_mean', float('nan')):.3f} ± {st.get('ari_sd', float('nan')):.3f}"
             if st else "n/a",
             str(st.get("n_clusters_range")) if st else "n/a"],
        ]
        conf = d.get("confounds", [])
        for c in conf:
            body.append([f"confound: {c['confound']}", "association test", "-", "-", "-",
                         "-", f"Cramer's V {c['cramers_v']:.3f}", f"p = {c['p_value']:.3g}"])
        sheet(wb, f"T_DPNP_{title}", [header] + body,
              widths=[22, 24, 18, 10, 12, 13, 20, 22],
              note=("The prespecified rule is the primary analysis and finds no clusters. "
                    "The fixed value is a sensitivity check only. A silhouette score "
                    "measures geometric separation, not biological validity. Cluster "
                    "assignment is tested for association with organ, species and source "
                    "image; a significant association means the structure is a confound."))
        built.append(f"DPNP {title}")

    ip = ANALYSIS / "instance_vs_component_summary_validation_runA.json"
    if ip.is_file():
        d = json.loads(ip.read_text(encoding="utf-8"))
        fb = load("feature_bias_components_vs_instances_validation_runA.csv")
        header = ["Quantity", "Value"]
        body = [
            ["Ground-truth nuclei (instances)", f"{d['total_gt_instances']:,}"],
            ["Predicted connected components", f"{d['total_pred_components']:,}"],
            ["Count ratio (components / nuclei)", round(d["overall_count_ratio"], 3)],
            ["Components merging 2+ nuclei", f"{d['mean_merge_rate']*100:.1f}%"],
            ["Components matching exactly 1 nucleus", f"{d['mean_clean_rate']*100:.1f}%"],
            ["Nuclei split across 2+ components", f"{d['mean_split_rate']*100:.1f}%"],
            ["Mean nuclei per component", round(d["mean_gt_per_component"], 2)],
            ["Most nuclei inside one component", d["max_gt_in_one_component"]],
        ]
        if fb is not None:
            for _, r in fb.iterrows():
                body.append([f"Feature bias, {r['feature']} (predicted / ground truth)",
                             f"{r['ratio_of_means']:.3f}  (p = {r['p_value']:.3g})"])
        sheet(wb, "T_components_vs_instances", [header] + body, widths=[52, 30],
              note=("Measured against NuInsSeg ground-truth instance masks. This is why "
                    "the DPNP outputs are described as exploratory connected-region "
                    "descriptors and not as per-nucleus phenotypes. CryoNuSeg is excluded "
                    "because it ships binary masks only."))
        latex_table(TEX / "table_components_vs_instances.tex",
                    "Correspondence between predicted connected components and "
                    "ground-truth nuclear instances on NuInsSeg.",
                    "tab:components_r3", header, body,
                    note="Merging is measured, not assumed.")
        built.append("components vs instances")

    # ---------- OSPT ----------
    oc = RESULTS / "splits" / "nuinsseg_R3_per_organ_counts.csv"
    if oc.is_file():
        df = pd.read_csv(oc)
        k = len(df)
        n_train = int(df["train"].sum())
        header = ["Organ", "Total", "Train", "Validation", "Test",
                  "Share of train without OSPT", "Expected draws with OSPT",
                  "Oversampling factor"]
        body = []
        for _, r in df.iterrows():
            flat = r["train"] / n_train if n_train else 0
            body.append([r.iloc[0], int(r["total"]), int(r["train"]),
                         int(r["validation"]), int(r["test"]),
                         f"{flat*100:.2f}%", round(n_train / k, 1),
                         round((1 / k) / flat, 2) if flat else "n/a"])
        sheet(wb, "T_OSPT_sampling", [header] + body,
              widths=[38, 9, 9, 13, 9, 26, 24, 20],
              note=(f"OSPT uses a WeightedRandomSampler with weight 1/count(organ) and "
                    f"replacement. The weights sum to the number of organs ({k}), so every "
                    f"organ has draw probability exactly 1/{k} and an expected "
                    f"{n_train/k:.1f} draws per epoch. This answers Reviewer 1 comment 3 "
                    f"analytically rather than by assertion."))
        built.append("OSPT sampling")

    xp = OUT / "EPCS_R3_Paper_Tables.xlsx"
    wb.save(xp)

    # ---------- old vs new ----------
    wb2 = Workbook()
    wb2.remove(wb2.active)
    header = ["Dataset", "Model", "Published Dice (validation, selected set)",
              "Revision 3 Dice (untouched test split)", "Change",
              "Published runs", "Revision 3 runs"]
    body = []
    for (ds, m), old in PUBLISHED.items():
        row = agg[(agg.dataset == ds) & (agg.model == m)]
        if row.empty:
            body.append([ds, m, old, "pending", "pending",
                         5 if m == "epcs_net" else 1, "pending"])
        else:
            r = row.iloc[0]
            body.append([ds, r["label"], old, round(r["test_dice_mean"], 4),
                         round(r["test_dice_mean"] - old, 4),
                         5 if m == "epcs_net" else 1, int(r["n_runs"])])
    sheet(wb2, "Headline_old_vs_new", [header] + body,
          widths=[12, 34, 32, 32, 10, 15, 15],
          note=("The previously published figures were validation scores on the same set "
                "used to select the checkpoint, and every baseline was a single run. The "
                "revision-3 figures are test-split scores with matched seeds. They are not "
                "like-for-like and the change column should be read as a protocol change, "
                "not a performance regression."),
          fills=lambda row, j, v: PEND if v == "pending" else (OLDF if j == 2 else None))
    wb2.save(OUT / "OLD_VS_NEW.xlsx")

    manifest = {"generated": datetime.now().isoformat(timespec="seconds"),
                "tables_built": built,
                "workbook": str(xp), "latex_dir": str(TEX)}
    (OUT / "TABLES_MANIFEST.json").write_text(json.dumps(manifest, indent=2),
                                              encoding="utf-8")
    print(f"Built {len(built)} tables:")
    for b in built:
        print(f"  - {b}")
    print(f"\n  {xp}")
    print(f"  {OUT / 'OLD_VS_NEW.xlsx'}")
    print(f"  {TEX}  ({len(list(TEX.glob('*.tex')))} LaTeX files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
