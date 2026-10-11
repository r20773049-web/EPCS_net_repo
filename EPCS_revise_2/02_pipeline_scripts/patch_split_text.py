from pathlib import Path
P = Path(r"<PROJECT_ROOT>\05_manuscript\EPCS_revise3.tex")
t = P.read_bytes().decode("utf-8")
R = [
 ("reports all six measures (accuracy, precision, recall, specificity, Dice and IoU) as mean $\\pm$ SD over the 15 runs, so that no measure is reported only as a mean.",
  "reports accuracy, precision, recall and specificity as mean $\\pm$ SD over the 15 runs (Dice and IoU are in Table~\\ref{tab:cryo}), so that no measure is reported only as a mean."),
 ("(all six measures with SD across seeds in Table~\\ref{tab:nu_allmetrics})",
  "(accuracy, precision, recall and specificity with SD across seeds in Table~\\ref{tab:nu_allmetrics})"),
]
for a, b in R:
    assert t.count(a) == 1, a
    t = t.replace(a, b)
P.write_bytes(t.encode("utf-8"))
print("ok")
