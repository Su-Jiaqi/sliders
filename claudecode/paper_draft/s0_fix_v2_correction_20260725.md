# s=0 fix v2 — traced "1.2%" to its exact source, found and fixed a bug in my own v1 fix — 2026-07-25

## The "1.2%" number is real and now exactly reproduced

Traced to `output-models/classifier/socalfire_cls_clean_split/clean_split_report.json`,
the held-out test confusion matrix from `train_classifier_clean_split.py`'s final,
selection-untouched evaluation: `[[243, 3], [2, 244]]` (rows=true[pre,post],
cols=pred[pre,post]) on n=246 real pre / n=246 real post images. Row 0 gives
**3/246 = 0.0122** real pre-disaster photos misclassified as post-disaster — exactly
the paper's cited "1.2%" (and 244/246=0.9919 for the post row, also matching the
paper's cited number). This is a real-photo classifier sanity check, not a CAS-style
generated-vs-real comparison — good, unambiguous ground truth.

## Root cause of the 62.6% vs 1.2% mismatch: a bug in my own v1 fix, not a general classifier problem

v1 of the s=0 fix (`s0_pixel_identity_fix_results_20260725.md`) pre-resized x_pre to
256x256 and saved a new file, which the eval pipeline's own per-metric resize then
processed *again* — a redundant double-resize hop that doesn't exist anywhere else
in this project: every other scale folder is written once by the generation model at
its native output resolution and read once by the eval script; the paper's "1.2%"
baseline likewise reads the real pre-disaster photo directly from its native
1024px source and resizes it exactly once. Confirmed this hop is the entire cause:
recomputed CAS using the native 1024px x_pre file directly (bypassing my
intermediate save) and got **CAS=0.5000 exactly, with the generated-predicted-as-post
component = 3/246 = 0.0122 — matching the paper's "1.2%" number to 3 significant
figures.**

**Fixed**: `build_scale0_pixel_identity.py` now symlinks the native-resolution
x_pre file directly instead of pre-resizing and saving a copy. Reran the full 5-
category, 7-scale eval (`cuda:4`; `cuda:0` untouched).

## This does *not* generalize to a project-wide classifier problem

Important reassurance, since this was the user's biggest worry: **the double-resize
bug was specific to my v1 implementation of this one fix.** Every other table this
session touched (Table 2, cross-category, seed-stability) scores images that were
generated once at their native resolution and read once by the eval script — the
same single-resize-hop pattern the "1.2%" baseline uses. None of them have the extra
intermediate-save step that caused this. No reason from this finding alone to
re-doubt those numbers.

## Corrected results (v2, all 5 categories, native symlink, single resize hop)

CAS at s=0 is now **exactly 0.5000 for all 5 categories** (theoretically correct
value for a truly identical real-pre/generated pair under CAS's own definition —
confirms the fix now behaves exactly as intended everywhere, not just socalfire).

| Category | LPIPS(s=0) | SSIM(s=0) | PSNR(s=0) | CAS(s=0) |
|---|---|---|---|---|
| Wildfire (n=246) | 0.3075 | 0.4646 | 21.256 | 0.5000 |
| Hurricane (n=97) | 0.3232 | 0.4673 | 20.162 | 0.5000 |
| Flooding (n=80) | 0.3641 | 0.4382 | 20.350 | 0.5000 |
| Santa Rosa (n=47) | 0.3094 | 0.4535 | 19.996 | 0.5000 |
| Volcano (n=4) | 0.4493 | 0.3508 | 20.303 | 0.5000 |

LPIPS is unchanged from v1 (that metric's own resize already converged to the same
result regardless of the intermediate hop). **SSIM and PSNR did shift slightly from
v1** (e.g. Wildfire SSIM 0.5030→0.4646, PSNR 21.626→21.256) — the SSIM/PSNR path
uses LANCZOS resizing, which behaves differently on an already-downsampled 256px
intermediate vs. the native 1024px source; v2's single-hop numbers are the correct
ones to use, superseding v1's for these two metrics. Scale0.25 onward are completely
unaffected (unchanged from before any of this) — this only ever touches the s=0
column.

## CAS-labeled figure, v2

Regenerated again with the corrected scale0: both curves now show CAS=0.5000,
LPIPS=0.3075 at s=0 (`outputs/eval/cas_figure_clean/cas_lpips_vs_scale_test_CLEAN.{png,pdf}`).
Scale0.25-1.0 unchanged from the previous regeneration.

| scale | Ours CAS | Ours LPIPS | w/o Pseudo CAS | w/o Pseudo LPIPS |
|---|---|---|---|---|
| 0 | 0.5000 | 0.3075 | 0.5000 | 0.3075 |
| 0.25 | 0.7378 | 0.2813 | 0.9898 | 0.2122 |
| 0.3 | 0.7561 | 0.2761 | 0.9898 | 0.2121 |
| 0.5 | 0.9085 | 0.2539 | 0.9898 | 0.2119 |
| 0.7 | 0.9797 | 0.2338 | 0.9898 | 0.2120 |
| 0.75 | 0.9837 | 0.2296 | 0.9898 | 0.2120 |
| 1.0 | 0.9878 | 0.2166 | 0.9898 | 0.2123 |

## Still open (unchanged from before, not addressed this round)

- Cross-category `table10_11_percategory_CLEAN.csv` (2026-07-17) still doesn't
  reproduce today (Hurricane s=0.25: stored 0.5206 vs. fresh 0.5928) — this is the
  separate environment-drift-affects-classifier-inference finding, unrelated to the
  resize-hop bug fixed here (that file was never touched by my double-resize code).
  Per your instruction, left this for a later decision.
