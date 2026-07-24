# s=0 fix — remaining scope closed: train split + the other 2 ablation variants — 2026-07-25

## What was done

Extended the corrected (native-symlink, single-resize-hop) s=0 fix to the two
scopes flagged as not-yet-covered:

1. **Train split** (`outputs/refine-2/socalfire/train/scale0_fixed`, symlinked
   from `datasets/remote/socalfire/train/pre`, n=769) — covers
   `tab:app_refined_scale`'s and `tab:app_continuous_generation_scale`'s Train row.
2. **The other two `tab:app_ablation_multiscale` variants** — "w/o Refinement"
   (`outputs/infer/socalfire`, the raw unrefined generator) and "w/o Scale-aware
   Refiner" (`outputs/ablation-2/socalfire/no_scale_finalbest`) — previously only
   "Ours" and "w/o Pseudo Supervision" had been fixed, leaving the 4-row table
   internally inconsistent (2 rows on pixel-identity x_pre, 2 rows on old
   model-generated s=0). All 4 rows are now on the same fix.

Ran on `cuda:4`, `cuda:0` untouched. Full CSVs:
`outputs/eval/scale0fix_remaining/{train_refined,train_direct,test_direct,test_no_scale}/`.

**CAS(s=0) is exactly 0.5000 in every one of these new runs**, matching the other
5 categories fixed earlier today — confirms the fix is now uniform and correct
everywhere it's been applied, not just for the two variants done first.

## tab:app_ablation_multiscale — now fully consistent, all 4 rows

CAS/PCR (test split, n=246), 0/0.25/0.5/0.75/1:

| Variant | s=0 | s=0.25 | s=0.5 | s=0.75 | s=1 |
|---|---|---|---|---|---|
| w/o Refinement | 0.5000 | 0.6504 | 0.6545 | 0.6707 | 0.6850 |
| w/o Scale-aware Refiner | 0.5000 | 0.7154 | 0.8943 | 0.9756 | 0.9837 |
| w/o Pseudo Supervision | 0.5000 | 0.9898 | 0.9898 | 0.9898 | 0.9898 |
| Ours | 0.5000 | 0.7378 | 0.9085 | 0.9837 | 0.9878 |

(0.25/0.5/0.75/1 columns unchanged from the existing clean-classifier data
`table14_clean/socalfire_ablation_multiscale_cas_test.csv` — only s=0 changed, now
uniformly 0.5000 for all four, matching the corrected PCR definition's floor.)

## Train split (wildfire, n=769)

`tab:app_refined_scale` Train row (full pipeline):

| s | 0 | 0.25 | 0.3 | 0.5 | 0.7 | 0.75 | 1 |
|---|---|---|---|---|---|---|---|
| LPIPS | 0.2821 | 0.2586 | 0.2541 | 0.2350 | 0.2183 | 0.2149 | 0.2000 |
| SSIM | 0.4949 | 0.5575 | 0.5626 | 0.5805 | 0.5938 | 0.5965 | 0.6137 |
| PSNR | 21.511 | 22.480 | 22.588 | 22.970 | 23.277 | 23.352 | 24.098 |
| CAS/PCR | 0.5000 | 0.7120 | 0.7406 | 0.9064 | 0.9837 | 0.9883 | 0.9948 |

`tab:app_continuous_generation_scale` Train row (raw/unrefined generator):

| s | 0 | 0.25 | 0.3 | 0.5 | 0.7 | 0.75 | 1 |
|---|---|---|---|---|---|---|---|
| LPIPS | 0.2821 | 0.3289 | 0.3320 | 0.3240 | 0.3156 | 0.3145 | 0.3119 |
| SSIM | 0.4949 | 0.4624 | 0.4635 | 0.4651 | 0.4634 | 0.4629 | 0.4608 |
| PSNR | 21.511 | 21.692 | 21.712 | 21.710 | 21.675 | 21.668 | 21.635 |
| CAS/PCR | 0.5000 | 0.6489 | 0.6424 | 0.6508 | 0.6704 | 0.6756 | 0.6964 |

Note: train's LPIPS(s=0)=0.2821 is smaller than test's LPIPS(s=0)=0.3075 — expected,
this is the same "LPIPS(x_pre, x_post) — real semantic pre/post difference" measure
discussed for test split, just computed on the train split's own (different) real
image pairs; not a discrepancy.

## Scope now fully closed

All 4 `tab:app_ablation_multiscale` rows, both `tab:app_refined_scale`/
`tab:app_continuous_generation_scale` splits (train+test already existed for test,
train done now), and the 5-category cross-category test-split tables are on the
same, consistent, corrected s=0 fix. Only remaining open item from this whole
thread: the 2026-07-17 cross-category `table10_11_percategory_CLEAN.csv`
reproducibility question, still deliberately not touched per earlier instruction.
