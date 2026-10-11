"""Add a predicted-vs-ground-truth feature table to Sec. 4.7.2 (values from 04_results/analysis/feature_bias_components_vs_instances_r3.csv)."""
import csv
from pathlib import Path
P = Path(r"<PROJECT_ROOT>\05_manuscript\EPCS_revise3.tex")
C = Path(r"<PROJECT_ROOT>\04_results\analysis\feature_bias_components_vs_instances_r3.csv")
raw = P.read_bytes().decode("utf-8")
crlf = "\r\n" in raw
t = raw.replace("\r\n", "\n")
names = {"area": "Area (px)", "roundness": "Roundness", "mean_intensity": "Mean intensity", "var_intensity": "Intensity variance"}
rows = []
for r in csv.DictReader(open(C)):
    p = float(r["p_value"])
    ps = "%.1e" % p
    m, e = ps.split("e")
    ps = "$%s\\times10^{%d}$" % (m, int(e))
    d = 2 if r["feature"] != "roundness" else 3
    rows.append("%s & %.*f & %.*f & %.2f & %s \\\\" % (names[r["feature"]], d, float(r["predicted_mean"]), d, float(r["gt_mean"]), float(r["ratio_of_means"]), ps))
n_pred, n_gt = rows and (int(float(next(csv.DictReader(open(C)))["predicted_n"])), int(float(next(csv.DictReader(open(C)))["gt_n"])))
tab = (
    "\n\\begin{table}[!htbp]\n\\centering\n\\small\n"
    "\\caption{Predicted components versus ground-truth nuclei on the NuInsSeg test set (EPCS-Net seed 2024; %s components, %s nuclei with a valid feature vector): feature means, ratio of means and Mann--Whitney $p$.}\n"
    "\\label{tab:feature_bias}\n\\begin{tabular}{lcccc}\n\\toprule\n"
    "\\textbf{Feature} & \\textbf{Predicted mean} & \\textbf{Ground-truth mean} & \\textbf{Ratio} & \\textbf{$p$} \\\\\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
) % (format(n_pred, ","), format(n_gt, ","), "\n".join(rows))
anchor = "NuInsSeg  & 19.9\\% & 11.7\\% & 8.1\\% & 39.7\\% \\\\\n\\bottomrule\n\\end{tabular}\n"
i = t.index(anchor) + len(anchor)
j = t.index("\\end{table}", i) + len("\\end{table}")
assert j - i < 40
sent = ("\n\nTable~\\ref{tab:feature_bias} compares four of the features of Equation~(\\ref{eq:fn}) between predicted components and ground-truth nuclei. "
        "Predicted components are larger (area ratio 1.22) and rounder, as expected when touching nuclei merge, and their intensity variance is lower (ratio 0.80); "
        "the colour features $L^*, a^*, b^*$ were not compared, so the table covers four of the seven features.")
t = t[:j] + sent + "\n" + tab + t[j:]
P.write_bytes((t.replace("\n", "\r\n") if crlf else t).encode("utf-8"))
print("ok", rows)
