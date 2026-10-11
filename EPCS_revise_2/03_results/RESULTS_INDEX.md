# Results index (07_revision3/04_results)

Written 2026-10-07 while the GPU sweep is still running. Status: FINAL = complete, citable; PROVISIONAL = still being written or analysed on partial data, re-check after the sweep. Reviewer IDs refer to `00_plan/REVIEWER_COMMENTS_ROUND3.md` (R1.x = Reviewer 1 item x, R3.M2 = Reviewer 3 Major 2, R3.m1 = Reviewer 3 Minor 1, Ed = editor).

## NEVER CITE

| Folder | Why |
|---|---|
| `_cpu_smoketests/` | CPU smoke tests of the analysis code on tiny inputs; not results. |
| `medsam_r3/_dry_run/` | MedSAM pipeline dry run; not results. |
| `_archive_oldGPU_RTX4500Ada_partial/` | Partial runs from the previous GPU (RTX 4500 Ada); superseded, protocol/hardware mismatch. |

Any path segment starting with `_` is excluded from every table and from `EPCS_R3_Full_Metrics.xlsx`.

## Per-run results

| Path | Produced by | Answers | Status |
|---|---|---|---|
| `cryonuseg/cryonuseg/<model>/fold{1-3}_seed{42,123,456,789,2024}/` (summary.json, per_image_test.csv, per_image_validation.csv, epoch_history.csv, best_model.pt, test_predictions/) models: epcs_net, unet, unetpp, unet3plus, eunet3plus, sam_unet, ref21_bottleneck, ref21_pyramid | `run_sweep.py` + `03_code/shared`, `03_code/runners` | R1.1, R3.M3, R2.1 (matched 5-seed baselines: seeds 42,123,456,789,2024 x 3 folds), R4.1 | PROVISIONAL until sweep ends (15 runs per model complete at time of writing) |
| `nuinsseg/` (appears when the sweep reaches it) | same | same, plus R2.3 (low-overlap cases), R1.3 (per-organ OSPT) | PROVISIONAL, not yet present |
| `reference21/cryonuseg/{ref21_bottleneck,ref21_pyramid}/` | `03_code/reference21` (X-gated fusion re-implementation; see FIDELITY_REPORT.md) under the same sweep | R4.2 | PROVISIONAL (15 runs each complete) |
| `ablation/cryonuseg/{abl1..abl6,ablA..ablD}/` | sweep, ablation variants (ablA no-PSPI, ablB random embedding = R3.M4 controls; ablC, ablD extra) | R1.2, R3.M4, R1.1 (GroupNorm encoder ablD, R1.1 norm) | PROVISIONAL; mostly single seed (42); abl1_nosam still accumulating. Do not claim monotonic or causal component benefit from single seeds |
| `sam_cache/` | SAM embedding cache (CACHE_REPORT.json) | R3.M2, R4.4 (SAM input 256 handling, reproducibility) | FINAL (infrastructure) |
| `splits/` (cryonuseg_R3_fold1-3.csv, nuinsseg_R3_stratified_70_15_15.csv, nuinsseg_R3_per_organ_counts.csv, SPLIT_REPORT.json) | split builder in `03_code/shared` | R1.3 (per-organ counts), R2.2 (CryoNuSeg tissues in validation), R3.M1, R4.1 | FINAL |

## MedSAM

| Path | Produced by | Answers | Status |
|---|---|---|---|
| `medsam_r3/summary_all.json`, `zero_shot_wholebox/`, `oracle_boxes/` | `03_code/medsam_r3/medsam_r3.py` | R1.4 | FINAL for the files present. zero_shot_wholebox = sanity check only; oracle_boxes = ORACLE UPPER BOUND using GT-derived boxes, never a competitor |

## Analysis (`analysis/`)

| File | Produced by | Answers | Status |
|---|---|---|---|
| `all_runs.csv`, `aggregated_by_model.csv`, `stats_manifest.json` | `03_code/analysis/stats_r3.py` | R1.1, R3.M3 | PROVISIONAL (manifest lists cryonuseg only; rerun after sweep) |
| `paired_significance_tests.csv` | stats_r3.py (paired t + Wilcoxon, Holm). Seeds within a fold share test images, so pairs are not independent | R3.M3, R1.1, R2.1 | PROVISIONAL |
| `fold_averaged_tests.csv` | stats_r3.py (seeds averaged per fold, n=3 folds, low power) | R3.M3, R2.1 | PROVISIONAL |
| `bootstrap_per_image.csv` | stats_r3.py (per-image paired bootstrap, single-run level) | R3.M3, R1.2 | PROVISIONAL |
| `per_organ_test_dice.csv` | stats_r3.py | R1.3, R2.2, R2.3 | PROVISIONAL (Dice only; all metrics are in PerOrgan of the workbook) |
| `dpnp_*_validation_runA*.csv/json`, `features_*`, `feature_bias_components_vs_instances_*`, `instance_vs_component_*` | `dpnp_r3.py`, `instances_r3.py` | R3.M5, R4.3 | PROVISIONAL (cryo runs only; validation split features, exploratory, no diagnostic claims) |

## Tables, workbooks, logs

| Path | What | Answers | Status |
|---|---|---|---|
| `EPCS_R3_Full_Metrics.xlsx` | All six metrics, test and validation, per model/run/fold/organ, paired tests, MedSAM. Built by `06_scripts/build_metrics_workbook.py` | R3.m3 (are all table metrics five-seed means), R3.M3, R1.1 | Regenerate after the sweep; PROVISIONAL now |
| `paper_tables/` (EPCS_R3_Paper_Tables.xlsx, OLD_VS_NEW.xlsx, TABLES_MANIFEST.json, latex/*.tex) | `03_code/tables` | R3.m3, R1.2 (old 0.8371 vs new numbers) | PROVISIONAL |
| `logs/benchmark_r3.json`, `benchmark_cached.json` | `03_code/runners/benchmark_r3.py` (latency / cost) | R3.m2, R4.4, R2.4 (WSI time not measured) | FINAL for current GPU; `*_RTX4500Ada.json` = old GPU, do not cite as current |
| `logs/analysis_chain_report.json`, `concurrency_probe.json`, `sweep_*.log`, `medsam_r3_full.log`, `sam_cache_*.log`, `verify_nuinsseg_full.log` | operational logs | none (provenance only) | not for citation; sweep logs PROVISIONAL |
