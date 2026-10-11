from pathlib import Path
p = Path(r"<PROJECT_ROOT>\05_manuscript\RESPONSE_DRAFT_R3.md")
t = p.read_bytes().decode("utf-8")
a = "**Summary of what changed in the protocol.**"
assert t.count(a) == 1
add = ("**Further wording changes (Editor: remove overclaims).** (1) The local encoder is now called the local-encoder (DoubleConv) branch throughout; \"eU-Net3+\" is kept only for the baseline and ablation row 1 (Reviewer 1). "
       "(2) Statements that the ablation does not support were replaced: the PSPI sentence now says it is designed to let SAM context reach every encoder scale and that the ablation is not conclusive on its benefit; the claim of no extra multiplicative cost for the 5x5 branch now says the branch adds computation and its gain was not significant; the decoder sentence no longer says that priors propagate through the whole pathway. "
       "(3) \"Novel\" was dropped from the PSPI and ACFB contribution items and ACFB is described as modified from the X-Gated Fusion Block of [21]; \"exceptional performance\" became \"promising results\"; the two clinician sentences in the Introduction were cut. "
       "(4) The author biographies were removed. (5) The captions of the ranking and training-curve figures state that the two reference-21 variants are left out and where their values are. (6) Section 4.7.2 now has a table comparing predicted components with ground-truth nuclei for four of the seven features (area, roundness, mean intensity, intensity variance); the three colour features were not compared.\n\n")
t = t.replace(a, add + a)
p.write_bytes(t.encode("utf-8"))
print("ok")
