"""Rebuild 00_plan/EPCS_R3_Reviewer_Comment_Coverage.xlsx (rerunnable, CPU-only, no GPU).
Edit the C (comments) / M (metrics) tables below as work progresses, then rerun.
Paths are relative to the project root (EPCS/); existence is checked at run time.
Status rule: RESOLVED-evidence-final only if no supporting number still depends on the sweep."""
import os
from collections import Counter
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))          # .../EPCS
OUT = os.path.join(ROOT, "07_revision3", "00_plan", "EPCS_R3_Reviewer_Comment_Coverage.xlsx")
R = "07_revision3/"
TEX = R + "05_manuscript/EPCS_revise3.tex"
RESP = R + "05_manuscript/RESPONSE_DRAFT_R3.md"
FIND = R + "00_plan/FINDINGS_LOG.md"
CR = R + "00_plan/CODE_REVIEW_R3.md"
AN = R + "04_results/analysis/"
PEND_NUIN = "pending NuInsSeg sweep block (40 runs; sweep_master.log tail showed 38 left, none in all_runs.csv yet) + final stats rerun"

STATUSES = ["RESOLVED-evidence-final", "PARTIAL-disclosed", "USER-ACTION", "DECLINED-DISCLOSED",
            "RESOLVED-text-pending-final-numbers", "IN PROGRESS", "PENDING"]  # last three legacy, unused after final audit
F, PA, UA, DC, T, IP, PE = STATUSES

# (ID, reviewer, text, type, prio, status, done, evidence[], remains, tasks)
C_OLD = [
("Ed-1","Editor","Remove AI-generated images","admin","P0",F,
 "Figs 1-4 redrawn as Excalidraw sources, rendered to PNG/SVG, inserted in tex; provenance stated.",
 [R+"05_manuscript/figures/fig1_pipeline.png",R+"05_manuscript/figures/fig2_pspi.png",R+"05_manuscript/figures/fig3_acfb.png",R+"05_manuscript/figures/fig4_fsfd.png",R+"06_scripts/render_excalidraw.py"],
 "Figs 5-12 are data plots (not AI images) but still show OLD validation-number plots; regenerate after sweep (M5). RESPONSE_DRAFT Ed-1 text is stale (says export not done). Optional user re-export.","M1, A2"),
("Ed-2","Editor","Move Datasets + Preprocessing into Methodology","text","P0",F,
 "Done in tex (Edit 9).",[TEX,R+"05_manuscript/PHASE1_EDIT_REPORT.md"],"Nothing (verify numbering at compile).","DONE-1"),
("Ed-3","Editor","Add Limitations subsection","text","P0",T,
 "Subsection added and extended with caveats (B1).",[TEX,R+"05_manuscript/PHASE1_EDIT_REPORT.md"],
 "Refresh with test-split numbers, audit items S2,S7-S10, indirect test exposure, latency figures (still quote 412.96/160.09 ms).","DONE-1, F3, M5, N2"),
("Ed-4","Editor","Future Work at end of Conclusion","text","P0",F,
 "Future Work paragraph in Conclusion, extended (SpatialQPFs/entropy).",[TEX],"Nothing.","DONE-1, M4"),
("Ed-5","Editor","(journal policy) Deposit custom code in a DOI-assigning repository (Zenodo); link in Methods / Code Availability","admin","P0",UA,
 "Code still being finalised locally; placeholder DOI still in tex.",[R+"03_code/runners/run_sweep.py"],
 "User pushes GitHub release, mints real Zenodo DOI after sweep+analysis; replace placeholder; same archive cited in Abstract/tables/Code Availability.","Z1"),
("Ed-6","Editor","(assumed item; condensed file lists 4 editor points) Point-by-point response letter + highlighted manuscript - confirm against original letter","admin","P0",IP,
 "Response skeleton with 25 [[RESULT]] placeholders, 46 labels verified.",[RESP,R+"06_scripts/check_response_draft.py"],
 "Fill placeholders after sweep; build PDF and highlighted tex (no LaTeX toolchain on this machine).","A4, M7"),

("R1-1a","Reviewer 1","Sec 4.4 DoubleConv BatchNorm vs Sec 4.6.5 GroupNorm (batch size 1): reconcile","text","P1",IP,
 "Clarified in tex (l.373: GroupNorm only in ACFB step 5). Ablation D (GN encoder) now FINAL 5 seeds: 0.7927+-0.0023 vs full 0.7827 (FINDINGS F.4), not yet in tex (l.957 still says 'still under evaluation', %%TODO-GN-ABLATION%%).",[TEX,FIND,AN+"aggregated_by_model.csv"],
 "Insert ablation D result in Sec acfb/Limitations; delete marker; update RESPONSE R1-1.","DONE-4, E1, E8"),
("R1-1b","Reviewer 1","Repeat strongest baselines (eU-Net3+ no SAM, SAM-guided) with same five seeds (matched seeds)","experiment","P0",IP,
 "CryoNuSeg: 15 matched runs per model (3 folds x 5 seeds) complete; paired tests + fold-averaged tests done.",
 [AN+"aggregated_by_model.csv",AN+"paired_significance_tests.csv",AN+"fold_averaged_tests.csv",R+"04_results/logs/sweep_master.log"],
 PEND_NUIN+". Fold-level significance limited (only eU-Net3+ survives Holm).","E1, E2, N4, N5"),
("R1-2a","Reviewer 1","Table 13 ablation is single run, earlier pass; final 0.8371 != 0.8471: repeat multi-seed under current protocol","experiment","P0",IP,
 "5-seed ablation FINAL (10 variants + abl0, CryoNuSeg fold 1, test, n=5): NOT monotonic, rows 3-6 indistinguishable (FINDINGS F, G). Tex Table 13 (l.916, 0.8371) and l.901/l.922 still the old single-run table; RESPONSE R1-2 still says 're-run in progress'.",
 [R+"04_results/ablation/cryonuseg",AN+"aggregated_by_model.csv",AN+"paired_significance_tests.csv",FIND],
 "Rewrite Table 13 + text on 5-seed mean+-SD; loss confound (BCE+Dice rows 0-5 vs FTAS row 6/A-D) must be stated; rewrite response.","E1, E8, N9"),
("R1-2b","Reviewer 1","DKSR step ~0.003: drop 'every component earns its place' unless supported","text","P1",PE,
 "Tex already says single-run steps 'may lie within run-to-run variation' (l.922) and has no 'earns its place'; but l.922 still states 'no row scores below the one above it'. Evidence to decide now exists: rows 3-6 Holm p=1.0 vs full, row 6 < row 5 (FINDINGS F.1).",[TEX,FIND,AN+"paired_significance_tests.csv"],
 "Delete monotonic sentence; state DKSR/FTAS steps are within noise (can do now, no sweep needed).","M5, N9"),
("R1-2c","Reviewer 1","(implied by audit S1/S2) ablation step 1->2 changes backbone; baselines differ in loss as well","experiment","P1",IP,
 "Patch applied; abl0 (EPCS backbone, no SAM) 5 seeds DONE: 0.7453+-0.0251; full-minus-abl0 +0.0374 CI [-0.003,0.078] Holm p=0.50 (FINDINGS G). Not in tex or response.",
 [CR,R+"03_code/patches/models_r3_nosam_patch.py",R+"04_results/ablation/cryonuseg/abl0_nosam_epcsbackbone",R+"04_results/logs/abl0_seed42.log",AN+"paired_significance_tests.csv",FIND],
 "Add abl0 row + wording ('suggestive, not conclusive', fold 1 only, 10 test images); relabel row 1 as eU-Net3+ backbone; decide optional abl5 on NuInsSeg (F2).","F1, N1, F2, F3"),
("R1-3","Reviewer 1","OSPT 'balanced within every epoch' vs pooled maths: specify actual sampler; per-organ counts before/after","text","P1",IP,
 "Sampler specified in tex (l.220-227, 1/K proof; l.228 'expectation not guarantee'). Simulation evidence NOW EXISTS: 2000 epochs, 459 imgs, 31 organs, expected 14.81 draws/organ, observed 14.62-14.97, per-epoch SD ~3.8, 29.8% imgs undrawn (FINDINGS H). NOT in tex (no table) and l.218 still says 'balanced within every epoch'.",
 [TEX,AN+"ospt_per_organ_sampling_nuinsseg.csv",R+"06_scripts/ospt_sampling_table.py",R+"04_results/splits/nuinsseg_R3_per_organ_counts.csv",FIND],
 "Add before/after table to tex; reword l.218 and intro contribution; fill RESPONSE placeholder #7 (draft still says 'needs a small script').","DONE-3, N6"),
("R1-4","Reviewer 1","MedSAM zero-shot whole-image box is a task mismatch: separate sanity experiment or adapted/prompted MedSAM","experiment","P1",IP,
 "Results FINAL in 04_results/medsam_r3 (zero-shot cryo folds 0.199/0.151/0.274, nuinsseg 0.148; oracle-box cryo ~0.67-0.71). Tex NOT updated: l.605 still describes whole-box zero-shot on validation split as a competitor, Table l.635 still 0.2226 (old validation); no oracle result; RESPONSE R1-4 says 'blocked on weights' (stale).",
 [R+"04_results/medsam_r3/summary_all.json",R+"04_results/logs/medsam_r3_full.log",R+"04_results/medsam_r3/oracle_boxes",R+"04_results/medsam_r3/zero_shot_wholebox"],
 "Move MedSAM to separate sanity subsection with test numbers, add oracle upper bound, drop from competitor ranking; NuInsSeg EPCS comparison number needs sweep.","E7, A3"),

("R2-1","Reviewer 2","Baselines single run: keep explicit statement that superiority over baselines is not statistically established","text","P0",IP,
 "CryoNuSeg paired tests final (15 pairs: all 7 significant; fold-averaged n=3: only eU-Net3+ survives Holm). Tex still OLD: l.936 / l.941 / l.956 say baselines single run and no paired test run (%%TODO-STATS-RESULTS, %%TODO-LIMITATIONS-STATS%%); l.939 says NuInsSeg margin 'comfortably outside noise' from old runs.",
 [AN+"paired_significance_tests.csv",AN+"fold_averaged_tests.csv",FIND],
 "Rewrite statement with both seed-level and fold-level results (can do for CryoNuSeg now); NuInsSeg needs sweep.","E2, N4, N5, F3, M5"),
("R2-2","Reviewer 2","CryoNuSeg validation split not tissue-stratified: keep limitation prominent","text","P1",IP,
 "New tissue-stratified 3-fold splits exist (evidence). Tex does NOT describe them: l.161 still says 80/20, 24 train/6 validation, shuffled; l.951 limitation describes old six-patch validation.",
 [TEX,R+"04_results/splits/cryonuseg_R3_fold1.csv",R+"04_results/splits/SPLIT_REPORT.json"],
 "Rewrite Sec cryonuseg/preproc for 3-fold stratified train/val/test; 5-image validation caveat (S8); check 'pan-cancer' wording (can do now).","B1, F3, M5"),
("R2-3","Reviewer 2","Grad-CAM low-overlap failure-case panel (esp. NuInsSeg) or keep as stated limitation","experiment","P1",PE,
 "Script CPU-tested on one run (3 best + 4 worst legible) in _cpu_smoketests only; no final panel, none in tex (Fig 7 old, l.838); no limitation sentence added either.",
 [R+"03_code/analysis/gradcam_r3.py",R+"04_results/_cpu_smoketests/fig13_gradcam_best_and_failure_cases_cryonuseg.png",AN+"gradcam_report_test.json"],
 "Run on final EPCS-Net checkpoints (CryoNuSeg possible now; NuInsSeg needs sweep); meanwhile add limitation sentence (can do now).","E4, B2"),
("R2-4","Reviewer 2","Whole-slide inference time unmeasured: do not imply WSI feasibility","text","P1",T,
 "Wording added in caveat pass.",[TEX],"Recheck in final abstract/conclusion rewrite.","B1, M5"),

("R3-M1","Reviewer 3","Validation = checkpoint-selection set for all scores","experiment","P0",IP,
 "Train/val/test split; checkpoint selection on validation, scores on test; cache verified exact; CryoNuSeg test block complete. Tex NOT rewritten: abstract l.85, Tables l.567-636, l.619, l.684 still old validation numbers (0.8471, 0.8460+-0.0037).",
 [R+"04_results/splits/SPLIT_REPORT.json",R+"04_results/sam_cache/CACHE_REPORT.json",AN+"aggregated_by_model.csv"],
 PEND_NUIN+"; rewrite abstract/results on test numbers.","E1, S5, M5"),
("R3-M2","Reviewer 3","SAM input 256x256 -> 16x16 grid vs official ViT-B 1024/64x64: explain compatibility","text","P0",F,
 "Found manuscript error (grid is 64x64; patches upsampled to 1024 with official preprocess, checkpoint unmodified); corrected.",[TEX,FIND,R+"05_manuscript/PHASE1_EDIT_REPORT.md"],
 "Nothing.","DONE-2"),
("R3-M3","Reviewer 3","Baselines single run: repeat under same seeds; mean+-SD/CI; paired comparisons on identical splits","experiment","P0",IP,
 "CryoNuSeg: all 8 models x 15 runs; mean, SD, CI, paired t/Wilcoxon/Holm, bootstrap done.",
 [AN+"aggregated_by_model.csv",AN+"paired_significance_tests.csv",AN+"bootstrap_per_image.csv"],PEND_NUIN+".","E1, E2, N5"),
("R3-M4a","Reviewer 3","Table 13 earlier pass: rerun with current protocol, multi-seed","experiment","P0",IP,
 "5-seed ablation final (see R1-2a); tex still old single pass.",[R+"04_results/ablation/cryonuseg",AN+"aggregated_by_model.csv"],"Rewrite table/text (same as R1-2a).","E1, E8, N9"),
("R3-M4b","Reviewer 3","Add missing controls: ACFB without PSPI; random-embedding control","experiment","P0",IP,
 "Controls A-D done, 5 seeds: A ACFB-no-PSPI 0.7545, B random embedding 0.7620 (vs SAM bottleneck 0.7661), C 0.7878, D 0.7927 (FINDINGS F). Not in tex (l.922+ still lists them as future extensions) or response.",
 [CR,R+"04_results/logs/sweep_master.log",AN+"aggregated_by_model.csv",AN+"paired_significance_tests.csv"],
 "Add controls to table; report that random embedding ~ real SAM bottleneck; disclose S10 weak random-embedding control; remove 'extensions' sentence.","E1, F3, N9"),
("R3-M4c","Reviewer 3","Else label preliminary, drop causal language; report provenance of old pass","text","P1",PE,
 "Tex l.901 already labels old table preliminary single-run earlier pass (partial provenance); causal wording not removed everywhere (NCAP para, l.924).",[TEX,FIND],"Replace with 5-seed table; state old pass was superseded and not monotonic (can do now).","M5, N9"),
("R3-M5a","Reviewer 3","HDBSCAN: declare prespecified (max(5,N/10) -> all noise) vs sensitivity (min_cluster_size=5, Fig 12)","text","P1",IP,
 "Pipeline reproduced exactly; prespecified=0 clusters; dpnp_r3 so far run on validation Run-A only. Tex l.885-887 reports the all-noise result but frames min_cluster_size=5 as showing 'real structure'; no explicit prespecified-vs-post-hoc-sensitivity label; contradicts instability finding.",[AN+"dpnp_report_validation_runA.json",R+"03_code/analysis/dpnp_r3.py"],
 "Rerun on test predictions after sweep; update Fig 12 + text.","E5, B2, M5"),
("R3-M5b","Reviewer 3","Silhouette != biological validation; no diagnostic/phenotypic claims","text","P1",IP,
 "Tex l.889 says no pathologist validation (OK) but l.887 calls silhouette 0.41/0.52 'reasonable separation ... real structure', contradicting ARI 0.30/0.37 and confounds in dpnp report.",[AN+"dpnp_report_validation_runA.json",FIND],"Final text in M5 on test-split numbers.","E5, M5"),
("R3-M5c","Reviewer 3","Connected components can merge nuclei: call them exploratory connected-region features","text","P1",IP,
 "Quantified on validation Run-A: 21.1% of components merge >=2 nuclei, area +44%. Tex l.845 only says touching nuclei merge (qualitative); no numbers; term 'exploratory connected-region features' absent.",[AN+"instance_vs_component_summary_validation_runA.json",AN+"feature_bias_components_vs_instances_validation_runA.csv"],
 "Rerun on test predictions; rename terminology throughout.","E6, M5"),
("R3-m1","Reviewer 3","FNR from pixel recall != missed nuclei","text","P2",F,
 "Wording fixed (Edit 8).",[TEX,R+"05_manuscript/PHASE1_EDIT_REPORT.md"],"Nothing.","DONE-5"),
("R3-m2","Reviewer 3","Latency: GPU model, procedure, warm-up, data loading included?; explain CryoNuSeg vs NuInsSeg difference","experiment","P1",IP,
 "Cause evidenced (GPU concurrency 2.77x ~ 419 vs 160 ms); data loading 4.3 ms/img; idle-GPU script (warm-up 20, sync) written. Tex l.535 says models 'profiled under identical conditions' and tables/l.966 still quote 412.96/160.09 ms: CONTRADICTS finding. Latency pending idle GPU.",
 [R+"04_results/logs/concurrency_probe.json",R+"04_results/logs/benchmark_r3.json",R+"03_code/analysis/latency_r3.py",AN+"latency_r3.json"],
 "Run latency_r3.py on idle GPU after sweep; update tex numbers.","S6, E3, N2"),
("R3-m3a","Reviewer 3","Check cross-references","text","P2",PE,"check_tex.py validates braces/envs/cites only.",[R+"06_scripts/check_tex.py"],"Full ref audit after compile.","M6"),
("R3-m3b","Reviewer 3","State whether all Table 9 metrics are five-seed means","text","P2",PE,"Not yet.",[TEX],"Wording after final tables.","M6"),
("R3-m3c","Reviewer 3","Abstract/tables/Code Availability refer to same implementation and archive","admin","P1",UA,"Blocked on DOI.",[TEX],"Depends on Zenodo DOI.","M6, Z1"),
("R3-m4","Reviewer 3","Cite omics methods mclSTExp, SpaBatch, SpaCross, TSCCA, scstGCN, SpaICL","citation","P2",F,
 "6 omics refs added, verified via Crossref.",[R+"05_manuscript/new_references_R3.bib",TEX],"Nothing.","M3"),

("R4-1a","Reviewer 4","Independent evaluation on untouched test set or nested CV","experiment","P0",IP,
 "Held-out test split in all runs; CryoNuSeg 3-fold complete.",[R+"04_results/splits/SPLIT_REPORT.json",AN+"all_runs.csv"],PEND_NUIN+". Disclose indirect test exposure.","E1, F3"),
("R4-1b","Reviewer 4","Loss weights also chosen on validation","text","P1",PE,
 "Protocol selects on validation; loss-weight selection provenance not yet documented.",[R+"03_code/shared/protocol_r3.py"],"State how loss weights were chosen (validation only; earlier overlapping val images) in Methods/Limitations.","F3, M5 (no dedicated task)"),
("R4-2a","Reviewer 4","Benchmark ref [21] directly under common protocol","experiment","P0",IP,
 "Bit-identical port; CryoNuSeg: ref21 bottleneck 0.7637, four-scale 0.7590 vs EPCS 0.7899.",[R+"03_code/reference21/FIDELITY_REPORT.md",R+"04_results/reference21/cryonuseg",AN+"aggregated_by_model.csv"],
 PEND_NUIN+"; results sentence %%TODO-REF21-RESULTS.","E1, M2, M5"),
("R4-2b","Reviewer 4","Discuss SlideMamba (Khan, Sci Rep 16, 4253, 2026)","citation","P2",F,"Added to Related Work; ref verified.",[TEX,R+"05_manuscript/new_references_R3.bib"],"Nothing.","M2, M3"),
("R4-3a","Reviewer 4","Compare features of predicted components vs GT instances","experiment","P1",IP,
 "Done on validation Run-A (8097 GT vs 4995 components, area ratio 1.44).",[AN+"instance_vs_component_summary_validation_runA.json",AN+"feature_bias_components_vs_instances_validation_runA.csv"],
 "Rerun on test predictions (instances_r3.py untested for new paths).","E6"),
("R4-3b","Reviewer 4","Support PCA component labels via loading analysis","experiment","P1",IP,
 "Loading mass: 98.8-99.1% on embedding; PC2/PC3 labels not supportable. Validation Run-A only.",[AN+"dpnp_pca_loading_mass_validation_runA.csv",AN+"dpnp_pca_feature_correlations_validation_runA.csv"],
 "Rerun on test predictions; drop labels in tex.","E5, M5"),
("R4-3c","Reviewer 4","Stability + organ/species/stain/image-effect checks for sensitivity clusters","experiment","P1",IP,
 "Bootstrap ARI 0.30/0.37, Cramer V organ 0.26, image 0.29/0.22. Validation Run-A only.",[AN+"dpnp_report_validation_runA.json",AN+"dpnp_cluster_confounds_validation_runA.csv"],
 "Rerun on test predictions. Note: stain effect not separately modelled (image ID only).","E5"),
("R4-3d","Reviewer 4","Discuss SpatialQPFs and entropy measures in Sec 5.6.2 / Future Work; what instance separation/cell classification is needed","citation","P2",F,
 "Future Work extended; refs added.",[TEX,R+"05_manuscript/new_references_R3.bib"],"Nothing.","M3, M4"),
("R4-4a","Reviewer 4","Explain SAM 256 input (positional embeddings, preprocessing, tensor shapes)","text","P1",F,"Corrected and explained (1024/64x64).",[TEX,FIND],"Nothing.","DONE-2"),
("R4-4b","Reviewer 4","Fixed code release for every reported result; replace placeholder Zenodo DOI","admin","P0",UA,"Placeholder DOI remains.",[TEX],"User release + DOI, last step.","Z1"),
("R4-4c","Reviewer 4","Explain dataset-dependent latency (hardware, timing boundaries, warm-up, sync)","experiment","P1",IP,"See R3-m2.",[R+"04_results/logs/concurrency_probe.json",AN+"latency_r3.json"],"Idle-GPU run + tex update.","E3, N2"),
("R4-4d","Reviewer 4","Report total + trainable parameters","text","P1",IP,
 "Wording added; count_params in latency_r3 prepared; numeric table not yet produced.",[R+"03_code/shared/models_r3.py",AN+"latency_r3.csv"],"Run latency_r3.py (params total/trainable) and verify tex numbers.","DONE-6, E3"),
]

# ---- FINAL AUDIT (FULFILMENT_AUDIT_FINAL.md): status/done/remains overrides keyed by ID. ----
# id: (status, done = tex location + response agreement, extra evidence paths, remains)
X = {
"Ed-1": (F, "tex l.559-561 sec:figprov: 'No figure was generated by an AI image model'. Response Ed-1 agrees.", [], "Visual check of figures in compiled PDF (user)."),
"Ed-2": (F, "tex l.148 sec:method > l.154 sec:datasets (CryoNuSeg l.159, NuInsSeg l.164, preprocessing l.185). Response agrees.", [], "None."),
"Ed-3": (F, "tex l.1166 sec:limitations, 18 items incl. 'Indirect test exposure', 'Split independence'. Response Ed-3 item (v) says stain is listed; tex has no stain item (response error).", [], "Fix response Ed-3 (v) or add stain to tex Limitations."),
"Ed-4": (F, "tex l.1204 sec:future inside Conclusion (l.1193); SpatialQPFs/entropy at l.1206. Response agrees.", [], "None."),
"Ed-5": (UA, "tex l.1327: placeholder DOI 10.5281/zenodo.0000000 with %%TODO-ZENODO-DOI. Response Ed-5 says 'Not yet done' (consistent).", [], "User: GitHub release + Zenodo DOI; replace placeholder."),
"Ed-6": (UA, "Response draft exists (171 lines, 44 comments answered) but has 2 [[RESULT]] markers (Ed-5, R3-m3), a dangling eq:ncap ref, corrupted 'ef{' refs in R2-3; highlighted manuscript not produced.", [], "User: fix response typos, produce highlighted tex/PDF, final upload."),
"R1-1a": (F, "tex l.299 and l.409-411: GroupNorm only in ACFB step 5; control D 0.7927 vs 0.7827 'not significant'. Response R1-1 agrees (says 'throughout the encoder', tex 'all encoder and decoder DoubleConv').", [], "None."),
"R1-1b": (F, "tex tab:cryo l.685, tab:nu l.836, tab:paired l.739, tab:nu_paired l.922; all 8 models x 5 seeds. Response R2-1 numbers match csv.", [], "None."),
"R1-2a": (F, "tex tab:ablation l.1111; l.1141 'Row 6 (0.7827) scores below row 5'; no 0.8371 in tex. Response says Holm p=0.021 for row1-vs-full; tex and csv give 0.023 (0.0229).", [], "Fix response p 0.021 -> 0.023."),
"R1-2b": (F, "tex l.1141: DKSR step 'not supported by the data'; 'every component earns its place' absent. Response agrees.", [], "None."),
"R1-2c": (F, "tex l.1117 row 0 (0.7453) and l.1143 (+0.0374, CI -0.0030 to +0.0778, Holm 0.50, 'crosses zero'). Response R3-M4 compares random embed to SAM bottleneck row 2 ('recovers 0.017 of 0.037'); tex l.1143 explicitly declines that comparison.", [], "Align response R3-M4 with tex (drop row-2 vs B comparison)."),
"R1-3": (F, "tex l.236, l.247 'balanced in expectation, not within every training epoch'; tab:ospt l.252 before/after; supp table l.1216. Response R1-3 agrees.", [], "None."),
"R1-4": (F, "tex sec:medsam l.648, tab:medsam l.655, l.670 oracle 0.6740-0.7153; MedSAM excluded from ranking l.642. Response R1-4 agrees (zero-shot 0.199/0.151/0.274, 0.148).", [], "None."),
"R2-1": (F, "tex l.732-734 sec:res_cryo_stats; limitations 'Statistical evidence is limited'; NuInsSeg Holm 0.070 disclosed in abstract l.85. Response R2-1 matches csv (0.0088, 0.0701).", [], "None."),
"R2-2": (PA, "Stratified 3-fold l.161; small-partition limitation (sec:limitations); caveat l.903. Title (l.50), keywords and 'Organ-Stratified Pan-Cancer Training' still say Pan-Cancer; no leave-tissue-out result exists.", [], "User: title/keyword decision on 'Pan-Cancer'."),
"R2-3": (F, "tex fig:gradcam l.1008, fig:gradcam_cryo l.1015, text l.1000-1002 (NuInsSeg mean 0.3155, worst cases Dice 0.228). Response R2-3 numbers agree but contains corrupted 'ef{fig:gradcam}' and a stray duplicate sentence.", [R+"04_results/analysis/gradcam_report_epcs_net__nuinsseg__seed2024.json"], "Clean response R2-3 text."),
"R2-4": (DC, "Not measured; tex l.1186 'Whole-slide feasibility is therefore not demonstrated'; l.575. Response R2-4 and 'cannot satisfy' list item 1 agree.", [], "Declined with reason (needs stitching pipeline)."),
"R3-M1": (F, "tex abstract l.85 and sec:protocol l.199: validation selects checkpoint, scores on test. Response agrees. No 0.8471 in tex.", [], "None."),
"R3-M2": (F, "tex l.274-276: upsample to 1024, 64x64 grid, 4096 tokens, official checkpoint unmodified. Response agrees but cites eq:ncap, which is not a label in the tex.", [], "Fix response eq:ncap reference."),
"R3-M3": (F, "tex tab:cryo l.685, tab:paired l.739; mean, SD, CI, Holm, bootstrap. Response agrees.", [], "None."),
"R3-M4a": (F, "tex l.1111 tab:ablation (5-seed, test split; no old rows). Response agrees.", [], "None."),
"R3-M4b": (F, "tex l.1111-1143: controls A-D, B random embed 0.7620 (-0.0207, CI -0.0005 to +0.0419). Response R3-M4 agrees on numbers; interpretive claim differs (see R1-2c).", [], "Align response wording."),
"R3-M4c": (F, "tex l.1141 'most steps are indistinguishable from noise'; abstract l.85 'preliminary one-fold ablation is non-monotonic'. Response agrees.", [], "None."),
"R3-M5a": (F, "tex l.1068-1070: primary prespecified rule max(5,N/10) gives 0 clusters; min_cluster_size=5 'post hoc setting'. Response R3-M5 numbers agree.", [R+"04_results/analysis/dpnp_report_r3_nuinsseg.json",R+"04_results/analysis/dpnp_report_r3_cryonuseg.json"], "None."),
"R3-M5b": (F, "tex l.1070 'We do not read these clusters as real structure'; no diagnostic claim (l.113, l.1188). tex has no explicit sentence that silhouette is not biological validation; response R3-M5 says 'the text says so'.", [R+"04_results/analysis/dpnp_report_r3_nuinsseg.json"], "Add one sentence to tex or soften response."),
"R3-M5c": (F, "tex l.1020 'exploratory connected-region features'; 18.0% (22.3% pooled) merge >=2 nuclei. Response R4-3a agrees.", [R+"04_results/analysis/instance_vs_component_summary_r3.json",R+"04_results/analysis/feature_bias_components_vs_instances_r3.csv"], "None."),
"R3-m1": (F, "tex l.594 metrics; limitation 'Recall is not a nucleus count'. Response agrees.", [], "None."),
"R3-m2": (F, "tex l.574-575: RTX PRO 4000 Blackwell, 20 warm-up, 100 timed, sync, 151.71 ms EPCS-Net; tab:cryo_efficiency l.710. Values match latency_r3.csv. Response agrees; old 412.96 absent from tex.", [R+"04_results/latency/latency_r3.csv",R+"04_results/latency/latency_r3.json"], "None."),
"R3-m3a": (UA, "No LaTeX compile has been run on this machine; response R3-m3 itself carries a [[RESULT]] marker. Labels cited by response: 55, 1 missing (eq:ncap).", [], "User: Overleaf compile + cross-reference/visual check."),
"R3-m3b": (F, "tex l.601 tab:seeds caption states mean+-SD over runs; l.757, l.854 'mean over five seeds'. Response agrees.", [], "None."),
"R3-m3c": (UA, "Abstract/Code Availability (l.1327) depend on the real DOI.", [], "User: Zenodo DOI, then recheck."),
"R3-m4": (F, "tex l.140 cites mclSTExp, scstGCN, SpaBatch, SpaCross, TSCCA, SpaICL (also l.1206). Response agrees; CITATION_AUDIT verified.", [R+"00_plan/CITATION_AUDIT.md"], "None."),
"R4-1a": (PA, "Held-out test split used (l.199); sec:limitations 'Indirect test exposure': design fixed on earlier validation sets overlapping test images. Response R4-1 agrees. Not an untouched set or nested CV.", [R+"00_plan/CODE_REVIEW_R3.md"], "Limited by design; disclosed. Nested CV not done."),
"R4-1b": (F, "tex l.526 FTAS 0.3:0.7 'determined empirically through validation performance monitoring' on earlier validation sets overlapping test. Response R4-1 agrees.", [], "None."),
"R4-2a": (DC, "tex sec:ref21_results l.803: verified port (bit-identical), not authors' trained model; 0.7637 / 0.7590 (CryoNuSeg). Response R4-2 agrees. Official pipeline unrunnable.", [], "Declined as-published benchmark; port disclosed."),
"R4-2b": (F, "tex l.140 SlideMamba cite r65. Response agrees.", [], "None."),
"R4-3a": (F, "tex l.1020: 5,207 GT nuclei vs 3,614 components, ratio 0.694, area 1.220. NuInsSeg only; CryoNuSeg has no instance GT (disclosed). Response R4-3 agrees.", [R+"04_results/analysis/instance_vs_component_summary_r3.json"], "None."),
"R4-3b": (F, "tex l.1049: loading mass 98.8/99.2/99.0%; labels dropped; PC1~area rho 0.865. Response R4-3(b) agrees.", [R+"04_results/analysis/dpnp_pca_loading_mass_r3_nuinsseg.csv",R+"04_results/analysis/dpnp_pca_loading_mass_r3_cryonuseg.csv"], "None."),
"R4-3c": (F, "tex phenotyping section: bootstrap ARI 0.539 (NuInsSeg) / 0.412 (CryoNuSeg), organ/species/tissue/source-image Cramer V; stain explicitly NOT assessed and listed in Limitations (Phenotyping confounders) after the final audit. Response R3-M5/Ed-3 now agree.", [R+"04_results/analysis/dpnp_cluster_confounds_r3_nuinsseg.csv",R+"04_results/analysis/dpnp_cluster_confounds_r3_cryonuseg.csv"], "None (a stain-colour confound check was not run; stated as a limitation)."),
"R4-3d": (F, "tex l.1073 and l.1206 (SpatialQPFs r72, entropy r73; instance separation needed). Response agrees.", [], "None."),
"R4-4a": (F, "tex l.274-276 (see R3-M2). Response agrees.", [], "None."),
"R4-4b": (UA, "tex l.1327 placeholder DOI. Response R4-4b: 'Not yet done'.", [], "User: tagged release + Zenodo DOI."),
"R4-4c": (F, "tex l.574-575 (idle GPU, warm-up, sync; 'not on the dataset'); concurrency probe 2.77x. Response agrees.", [R+"04_results/latency/latency_r3.json"], "None."),
"R4-4d": (F, "tex tab:cryo_efficiency l.710-722, tab:nu_efficiency l.883: EPCS-Net 120.42 M total / 26.68 M trainable; matches latency_r3.csv. Response agrees.", [R+"04_results/latency/latency_r3.csv"], "None."),
}
assert set(X) == {c[0] for c in C_OLD}, set(X) ^ {c[0] for c in C_OLD}
C = []
for (i, rv, tx, ty, pr, st, done, ev, rem, tk) in C_OLD:
    ns, nd, extra, nr = X[i]
    C.append((i, rv, tx, ty, pr, ns, nd, list(dict.fromkeys([TEX, RESP] + extra)), nr, tk))

# metric, requested by, definition, script (rel. to 07_revision3/), output (rel. to 07_revision3/), status
M = [
("Dice","R1,R3,R4 (all tables)","per-image Dice","03_code/shared/protocol_r3.py","04_results/analysis/all_runs.csv","computed, final (both datasets)"),
("IoU","tables","per-image IoU","03_code/shared/protocol_r3.py","04_results/analysis/all_runs.csv","computed, final (both datasets)"),
("Precision","tables","TP/(TP+FP)","03_code/shared/protocol_r3.py","04_results/analysis/all_runs.csv","computed, final (both datasets)"),
("Recall (pixel)","R3-m1","TP/(TP+FN); NOT missed-nuclei rate","03_code/shared/protocol_r3.py","04_results/analysis/all_runs.csv","computed, final (both datasets)"),
("Specificity","tables","TN/(TN+FP)","03_code/shared/protocol_r3.py","04_results/analysis/all_runs.csv","computed, final (both datasets)"),
("Accuracy","tables","pixel accuracy","03_code/shared/protocol_r3.py","04_results/analysis/all_runs.csv","computed, final (both datasets)"),
("Parameters total + trainable","R4-4d","count_params","03_code/analysis/latency_r3.py","04_results/latency/latency_r3.csv","final (idle-GPU run)"),
("Latency (GPU model, warm-up 20, sync, no dataloader)","R3-m2,R4-4c","timed forward","03_code/analysis/latency_r3.py","04_results/latency/latency_r3.json","final (idle-GPU run)"),
("Concurrency / latency-discrepancy explanation","R3-m2,R4-4c","training-step benchmark","03_code/runners/probe_concurrency.py","04_results/logs/concurrency_probe.json","done (explanatory only)"),
("VRAM peak","R3-m2 (implied)","max_memory_allocated","03_code/analysis/latency_r3.py","04_results/latency/latency_r3.json","final (idle-GPU run)"),
("Grad-CAM overlap with GT","R2-3","CAM vs mask overlap per image","03_code/analysis/gradcam_r3.py","04_results/analysis/gradcam_report_epcs_net__nuinsseg__seed2024.json","final (seed 2024, test split)"),
("Silhouette","R3-M5","geometric separation only","03_code/analysis/dpnp_r3.py","04_results/analysis/dpnp_report_r3_nuinsseg.json","final on test predictions (r3 files; seed 2024, single run)"),
("ARI (bootstrap stability)","R4-3c","adjusted Rand vs subsamples","03_code/analysis/dpnp_r3.py","04_results/analysis/dpnp_report_r3_nuinsseg.json","final on test predictions (r3 files; seed 2024, single run)"),
("Cramer's V (organ/image/species)","R4-3c","chi-square + V","03_code/analysis/dpnp_r3.py","04_results/analysis/dpnp_cluster_confounds_r3_nuinsseg.csv","final on test predictions (r3 files; seed 2024, single run)"),
("PCA loading mass","R4-3b","loading share on interpretable vs embedding dims","03_code/analysis/dpnp_r3.py","04_results/analysis/dpnp_pca_loading_mass_r3_nuinsseg.csv","final on test predictions (r3 files; seed 2024, single run)"),
("Component-vs-instance merge rate","R4-3a,R3-M5c","components merging >=2 GT nuclei etc.","03_code/analysis/instances_r3.py","04_results/analysis/instance_vs_component_summary_r3.json","final on test predictions (r3 files; seed 2024, single run)"),
("Feature bias components vs instances","R4-3a","ratio of means + p","03_code/analysis/instances_r3.py","04_results/analysis/feature_bias_components_vs_instances_r3.csv","final on test predictions (r3 files; seed 2024, single run)"),
("Mean + SD","R1-1,R3-M3","across matched runs","03_code/analysis/stats_r3.py","04_results/analysis/aggregated_by_model.csv","final (both datasets)"),
("95% CI (across runs)","R3-M3","t-based CI","03_code/analysis/stats_r3.py","04_results/analysis/aggregated_by_model.csv","final (both datasets)"),
("Paired t-test","R1-1,R2-1,R3-M3","matched seed/fold pairs","03_code/analysis/stats_r3.py","04_results/analysis/paired_significance_tests.csv","final (both datasets)"),
("Wilcoxon signed-rank","R2-1,R3-M3","matched pairs","03_code/analysis/stats_r3.py","04_results/analysis/paired_significance_tests.csv","final (both datasets)"),
("Holm correction","R2-1,R3-M3","within (dataset, family)","03_code/analysis/stats_r3.py","04_results/analysis/paired_significance_tests.csv","final (both datasets)"),
("Fold-averaged paired test (n=3)","pseudo-replication (audit S5)","conservative check","03_code/analysis/stats_r3.py","04_results/analysis/fold_averaged_tests.csv","final (both datasets)"),
("Bootstrap CI (images)","R3-M3","per-image, run-averaged","03_code/analysis/stats_r3.py","04_results/analysis/bootstrap_per_image.csv","final (both datasets)"),
("Per-organ test Dice","R1-3 (implied)","per organ","03_code/analysis/stats_r3.py","04_results/analysis/per_organ_test_dice.csv","final"),
("MedSAM zero-shot / oracle Dice","R1-4","zero-shot whole box; GT-component box","03_code/medsam_r3","04_results/medsam_r3/summary_all.json","done"),
]


def check(paths):
    miss = [p for p in paths if not os.path.exists(os.path.join(ROOT, p))]
    return (not miss), miss


def hdr(ws, cols, widths):
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="305496")
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for i, w in enumerate(widths):
        ws.column_dimensions[chr(65 + i)].width = w
    ws.freeze_panes = "A2"


def wrap(ws):
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")


def main():
    wb = Workbook()
    ws = wb.active
    ws.title = "Coverage"
    hdr(ws, ["ID", "Reviewer", "Comment (short)", "Requirement type", "Priority", "Status", "Done so far",
             "Evidence file path(s)", "Evidence exists", "Missing paths", "What remains", "Tracker task IDs"],
        [9, 11, 45, 13, 8, 30, 45, 55, 10, 40, 50, 22])
    ids, cnt, noowner = set(), Counter(), []
    for (i, rv, tx, ty, pr, st, done, ev, rem, tk) in C:
        assert i not in ids and st in STATUSES
        ids.add(i)
        ok, miss = check(ev)
        if st == F and (not ok or "pending" in rem.lower()):
            raise SystemExit(f"{i}: cannot be evidence-final (missing evidence or pending remains)")
        if st == PA and not ok:
            raise SystemExit(f"{i}: PARTIAL needs existing evidence files")
        if "NO OWNER" in tk:
            noowner.append(i)
        cnt[st] += 1
        ws.append([i, rv, tx, ty, pr, st, done, "\n".join(ev), "TRUE" if ok else "FALSE",
                   "\n".join(miss), rem, tk])
    wrap(ws)
    ws2 = wb.create_sheet("Metrics requested")
    hdr(ws2, ["Metric", "Requested by", "Definition / note", "Computed in (script)", "Output stored",
              "Script exists", "Output exists", "Status"], [34, 24, 38, 36, 60, 10, 10, 34])
    for (m, by, d, sc, out, st) in M:
        ws2.append([m, by, d, R + sc, R + out, str(os.path.exists(os.path.join(ROOT, R + sc))).upper(),
                    str(os.path.exists(os.path.join(ROOT, R + out))).upper(), st])
    wrap(ws2)
    ws3 = wb.create_sheet("Summary")
    hdr(ws3, ["Status", "Count"], [45, 60])
    for s in STATUSES:
        ws3.append([s, cnt[s]])
    ws3.append(["TOTAL", sum(cnt.values())])
    ws3.append([])
    ws3.append(["Comments with no dedicated owner task", ", ".join(noowner) or "none"])
    ws3.append(["Note", "Final audit (FULFILMENT_AUDIT_FINAL.md). RESOLVED-evidence-final = FULFILLED."])
    wb.save(OUT)
    print("saved", OUT)
    print(dict(cnt), "no-owner:", noowner)


if __name__ == "__main__":
    main()
