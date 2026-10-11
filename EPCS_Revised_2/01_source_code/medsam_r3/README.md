# MedSAM on the round-3 test splits (Reviewer 1, comment 4)

Round-2 code in `OLD_ROUNDS_1_2/02_code_MedSAM/` is frozen and untouched. `medsam_r3.py` is new.

## Protocol
Evaluated on the TEST partition only of `04_results/splits/`: CryoNuSeg folds 1-3 and the NuInsSeg 70/15/15 split. Weights: `wanglab/medsam-vit-base` via transformers `SamModel`/`SamProcessor`, frozen, no fine-tuning, no threshold tuning. Metrics: `metric_row` from `shared/protocol_r3.py` (same definitions as every other model), computed at the ORIGINAL image resolution as in round 2 (the trained models score at 256x256, so values are not bit-comparable in resolution).

| mode | prompt | role |
|---|---|---|
| `zero_shot_wholebox` | one box covering the whole image, 3 candidate masks, keep best predicted IoU | round-2 protocol, sanity check |
| `oracle_boxes` | one box per 8-connected component of the GT binary mask, single-mask output, masks unioned | ORACLE upper bound (uses GT localisation) |

Per-component counts are saved (`n_gt_components` per image; mean/median/min/max/total in the summary). Components touching each other in the GT form one component, so the box count under-states the instance count in dense clusters.

## Run
```
# logic test, no weights, CPU only
CUDA_VISIBLE_DEVICES="" python medsam_r3.py --dry-run
# real run (weights must already be in the HF cache; fails with a clear message otherwise)
python medsam_r3.py                       # CPU default
python medsam_r3.py --device cuda         # only when the GPU is free
python medsam_r3.py --allow-download      # the ONLY way it will touch the network
```
Options: `--modes`, `--datasets cryonuseg_fold1 ... nuinsseg`, `--limit N`.
Outputs: `04_results/medsam_r3/<mode>/<dataset>/per_image_*.csv, summary_*.json, predictions/`, plus `summary_all.json`. Dry-run writes to `medsam_r3/_dry_run/` (stub numbers; never cite them).

## Manuscript wording
"MedSAM is a box-prompted model and has no mechanism to localise nuclei by itself. With the round-2 single whole-image box it is a stress test, not its intended use (reported as a sanity check). To bound what prompt-based adaptation could achieve, we also gave MedSAM one box per ground-truth connected component. This is an oracle upper bound: it uses ground-truth localisation that is unavailable at test time, and it is therefore not a competitor to the fully automatic models in Table X. Our models receive no prompts."
Do not rank the oracle row against the other models, and state that it is a single deterministic pass (no seeds, no training).

## Tested / not tested
Tested (CPU, stub): split reading, test filtering, box extraction, union, metric computation, file writing, synthetic check (3 rectangles: 3 components, Dice ~1.0), real-mask stub check (recall 1.0). Real-mask stub Dice is ~0.8 by construction (bounding boxes overfill irregular nuclei), so it is not expected near 1.
NOT tested: the actual MedSAM path (`MedSAMModel`), including box rescaling into the processor frame, `get_image_embeddings` + `image_embeddings=` forward, post-processing of batched box masks, the cache-missing error, and runtime/memory. Verify with `--limit 2` on a dummy first run before the full run. Dependency `transformers` was not installed in the default python at write time.
