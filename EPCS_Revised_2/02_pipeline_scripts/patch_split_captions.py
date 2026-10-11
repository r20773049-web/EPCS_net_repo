from pathlib import Path
P = Path(r"<PROJECT_ROOT>\05_manuscript\EPCS_revise3.tex")
t = P.read_bytes().decode("utf-8")
R = [
 ("Dice and IoU as mean $\\pm$ SD over runs. Loss: training loss.}",
  "Dice and IoU as mean $\\pm$ SD over runs; other metrics in Table~\\ref{tab:cryo_allmetrics}. Loss: training loss.}"),
 ("Dice and IoU as mean $\\pm$ SD over seeds. Loss: training loss.}",
  "Dice and IoU as mean $\\pm$ SD over seeds; other metrics in Table~\\ref{tab:nu_allmetrics}. Loss: training loss.}"),
 ("\\caption{All six pixel-level measures on CryoNuSeg: mean $\\pm$ SD over the 15 runs of each model.}",
  "\\caption{Accuracy, precision, recall and specificity on CryoNuSeg: mean $\\pm$ SD over 15 runs (Dice and IoU in Table~\\ref{tab:cryo}).}"),
 ("\\caption{All six pixel-level measures on NuInsSeg: mean $\\pm$ SD over the five seeds of each model.}",
  "\\caption{Accuracy, precision, recall and specificity on NuInsSeg: mean $\\pm$ SD over five seeds (Dice and IoU in Table~\\ref{tab:nu}).}"),
]
for a, b in R:
    assert t.count(a) == 1, a
    t = t.replace(a, b)
P.write_bytes(t.encode("utf-8"))
print("ok")
