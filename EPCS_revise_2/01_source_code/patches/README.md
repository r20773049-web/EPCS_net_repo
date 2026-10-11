# S1 patch: EPCS-backbone-without-SAM ablation row (`abl0_nosam_epcsbackbone`)

Fixes audit finding S1: row 1 (eU-Net3+) and rows 2-6 (EPCSNet backbone) differ in backbone, so
step 1->2 conflates "change backbone" with "add SAM". New row 0 = EPCSNet with no SAM at all.

Files here (nothing in `shared/` or `runners/` has been modified):

- `models_r3_nosam_patch.py`  applies 6 anchored edits to `models_r3.py` (idempotent, each anchor must match once, keeps line endings).
- `test_nosam_patch.py`       CPU test; works on a temp copy, never touches the live file.

## Definition (justification is in the patch docstring)

`EPCSNet(sam_source="none")`: no context encoder is built, no fusion blocks (f1..f4) are built, and the
decoder receives `f_i = e_i` (identity) at all four scales. This reuses the identity convention that the
existing `injection="bottleneck"` branch already uses for f1..f3. Encoder, decoder, head are identical
to EPCS-Net (13,022,465 params, all trainable). Registry: `group="ablation"`, `loss="bce_dice"`,
`sam=False` (so no checkpoint check and no SAM cache), label `0. EPCS backbone, no SAM`.

## Apply (only after the GPU sweep has finished, so no new job process imports a half-edited file)

Run from `07_revision3/03_code`:

```
python patches/test_nosam_patch.py                                   # must print ALL PASSED
copy shared\models_r3.py shared\models_r3.py.bak_pre_nosam
python patches/models_r3_nosam_patch.py --target shared/models_r3.py
```

Then add the id to `ABLATION` in `runners/run_sweep.py` (line 51), e.g.:

```
python - <<'PY'
p="runners/run_sweep.py"; s=open(p,encoding="utf-8").read()
old='ABLATION = ["abl1_nosam",'; assert s.count(old)==1
open(p,"w",encoding="utf-8").write(s.replace(old,'ABLATION = ["abl0_nosam_epcsbackbone", "abl1_nosam",'))
PY
```
(or just edit line 51 by hand). Verify: `python runners/run_sweep.py --plan` should list 5 more
`ablation-cryonuseg` jobs pending.

## Run the 5 new seeds (CryoNuSeg fold 1, ~1 min each, CPU/GPU as configured)

Via the sweep (all finished jobs are skipped because their `summary.json` exists):

```
python runners/run_sweep.py --run --only ablation
```

or directly, one job per seed (seeds 42 123 456 789 2024):

```
for s in 42 123 456 789 2024; do python shared/protocol_r3.py --model abl0_nosam_epcsbackbone --dataset cryonuseg --seed $s --fold 1; done
```
(PowerShell: `foreach ($s in 42,123,456,789,2024) { python shared/protocol_r3.py --model abl0_nosam_epcsbackbone --dataset cryonuseg --seed $s --fold 1 }`)

## Every site that enumerates ablation model ids (grep `abl1_nosam` / `ABLATION` in 03_code and 06_scripts)

| File | Line | Needs change? |
|---|---|---|
| `03_code/shared/models_r3.py` | 556 (registry) | YES, done by the patch (adds `abl0_...` before `abl1_nosam`) |
| `03_code/runners/run_sweep.py` | 51 `ABLATION` list | YES, one-line edit above. `build_matrix`, `is_done`, `sort_key` then pick it up automatically |
| `03_code/tables/build_paper_tables.py` | 65-67 `PUBLISHED_ABLATION` | Optional. Holds published (Table 13) Dice/IoU for rows 1-2.. only; row 0 has no published value. Row 321 `pub.get` lookup should tolerate a missing id: check it returns blank, not KeyError, when you rebuild |
| `06_scripts/build_comparison_workbooks.py` | 121 `ABLATION`, 130 `ABLATION_MISSING` | Optional, cosmetic. Hand-curated published/missing-row lists; add row 0 as "new in R3" if wanted |
| `03_code/analysis/stats_r3.py` | 146 | No edit. Uses `group.startswith("ablation")`, so row 0 is included automatically in the ablation Holm family (family size grows by one pair, so Holm thresholds change; re-run stats) |
| `03_code/tables/build_paper_tables.py` | 176, 222, 316 | No edit (group filter) |
| `03_code/shared/protocol_r3.py` | 208 | No edit (group bucket) |
| `03_code/runners/benchmark_r3.py` + `04_results/logs/benchmark_r3*.json` | | No edit needed. `estimate_seconds` returns 0 for ids without a benchmark entry (only affects the time estimate and cheapest-first ordering; the job sorts first among ablation jobs of a seed) |
| `00_plan/CODE_REVIEW_R3.md` | 39, 77 | Documentation only |

`PUBLISHED_PARAMS` and `verify_published_parameter_counts` need no change (new model is not a published architecture).
