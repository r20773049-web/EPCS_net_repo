"""Build EPCS_GITHUB_RELEASE/ (copy only; nothing in the working tree is changed or deleted).
Layout (serialised):
  01_code/             shared, runners, analysis, reference21, medsam_r3, patches, tables   (from 07_revision3/03_code)
  02_scripts/          pipeline, figure/table/workbook builders, checkers                    (from 07_revision3/06_scripts)
  03_results/          every result file except model checkpoints and the SAM feature cache
  04_manuscript/       Overleaf project, response letter, tex source
  05_figure_sources/   Excalidraw sources of the four schematics
  06_checkpoints/      manifest (sha256, size) of the checkpoints that go to Zenodo, not into git
  07_data_info/        dataset links and split manifests
  08_previous_rounds_code/  code of rounds 1-2 (no results)
"""
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R3 = ROOT / "07_revision3"
OUT = ROOT / "EPCS_GITHUB_RELEASE"
SKIP_DIRS = {"__pycache__", "_backups", "_history", ".ipynb_checkpoints"}
SKIP_RES = {"sam_cache", "_archive_oldGPU_RTX4500Ada_partial", "_cpu_smoketests", "_chain_dryrun", "_dry_run", "_dpnp_cryonuseg_work"}
SKIP_EXT = {".pt", ".pth", ".pyc", ".npy"}


def copy_tree(src, dst, skip_dirs=SKIP_DIRS, skip_ext=(), skip_name=lambda n: False):
    n = 0
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if any(part in skip_dirs for part in rel.parts) or not p.is_file():
            continue
        if p.suffix.lower() in skip_ext or skip_name(p.name):
            continue
        t = dst / rel
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, t)
        n += 1
    return n


if OUT.exists():
    sys.exit("EPCS_GITHUB_RELEASE already exists; remove it first")
OUT.mkdir()
cnt = {}
cnt["01_code"] = copy_tree(R3 / "03_code", OUT / "01_code")
cnt["02_scripts"] = copy_tree(R3 / "06_scripts", OUT / "02_scripts")
cnt["03_results"] = copy_tree(R3 / "04_results", OUT / "03_results", skip_dirs=SKIP_DIRS | SKIP_RES, skip_ext=SKIP_EXT,
                              skip_name=lambda n: "validation_runA" in n)
for f in ("EPCS_R3_MASTER_Results.xlsx",):
    if (R3 / f).exists():
        shutil.copy2(R3 / f, OUT / "03_results" / f)
man = R3 / "05_manuscript"
cnt["04_manuscript"] = copy_tree(ROOT / "PAPER_OVERLEAF", OUT / "04_manuscript" / "overleaf_project")
(OUT / "04_manuscript").mkdir(exist_ok=True)
shutil.copy2(ROOT / "PAPER_OVERLEAF.zip", OUT / "04_manuscript" / "EPCS_paper_overleaf_project.zip")
shutil.copy2(man / "EPCS_revise3.tex", OUT / "04_manuscript" / "EPCS_revise3_single_file.tex")
resp = man / "response"
if resp.exists():
    cnt["response"] = copy_tree(resp, OUT / "04_manuscript" / "response")
cnt["05_figure_sources"] = copy_tree(ROOT / "EXCALIDRAW_FOR_YOU", OUT / "05_figure_sources")
cnt["07_data_info"] = copy_tree(R3 / "04_results" / "splits", OUT / "07_data_info" / "splits")
old = ROOT / "OLD_ROUNDS_1_2"
for d in ("02_code_RunA", "02_code_RunB", "02_code_DPNP", "02_code_MedSAM"):
    if (old / d).exists():
        cnt["08/" + d] = copy_tree(old / d, OUT / "08_previous_rounds_code" / d, skip_ext=SKIP_EXT)

# checkpoint manifest
rows = []
for p in sorted((R3 / "04_results").rglob("*.pt")):
    rel = p.relative_to(R3 / "04_results")
    if rel.parts[0] in SKIP_RES:
        continue
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    rows.append((rel.as_posix(), p.stat().st_size, h.hexdigest()))
(OUT / "06_checkpoints").mkdir()
with open(OUT / "06_checkpoints" / "CHECKPOINT_MANIFEST.csv", "w", encoding="utf-8") as f:
    f.write("path_relative_to_results,bytes,sha256\n")
    for r in rows:
        f.write("%s,%d,%s\n" % r)
print(cnt)
print("checkpoints listed:", len(rows), "total GB:", round(sum(r[1] for r in rows) / 1e9, 2))
