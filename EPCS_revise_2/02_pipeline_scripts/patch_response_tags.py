from pathlib import Path
p = Path(r"<PROJECT_ROOT>\05_manuscript\RESPONSE_DRAFT_R3.md")
t = p.read_bytes().decode("utf-8")
R = [
 ("The MedSAM numbers are removed from the competitor comparison and reported in a separate, clearly labelled subsection.",
  "The MedSAM numbers are removed from the comparison tables and reported separately in Section \\ref{sec:medsam}."),
 ("(i) Zero-shot sanity check (whole-image box):", "(i) Zero-shot, whole-image box:"),
 ("This shows only that an unadapted, unprompted model is not a nuclear segmenter; it is not a competitor.",
  "This shows only that an unadapted, unprompted model does not segment nuclei."),
 ("(ii) Oracle-prompted MedSAM, with boxes taken from ground-truth connected components:",
  "(ii) Box-prompted MedSAM, with boxes derived from ground-truth connected components:"),
 ("This is an upper bound that uses ground-truth localisation at test time. It is not available to any method that has to find the nuclei itself, so it is labelled as an oracle and is not presented as a competing method.",
  "These boxes use ground-truth localisation at test time, which no method that must find the nuclei itself has, so this protocol is reported separately and not ranked with the trained models."),
 ("so it is an oracle upper bound and not a fair competitor; the zero-shot result is a sanity check only.",
  "so it uses ground-truth localisation and is reported separately from the trained models; the zero-shot result only shows that the unadapted model does not segment nuclei."),
 ("MedSAM is no longer in the competitor tables.", "MedSAM is no longer in the comparison tables."),
]
for a, b in R:
    assert t.count(a) == 1, a
    t = t.replace(a, b)
p.write_bytes(t.encode("utf-8"))
print("ok")
