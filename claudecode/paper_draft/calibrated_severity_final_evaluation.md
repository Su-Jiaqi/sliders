# Final evaluation plan: calibrated-severity variant vs. production model

## 0. What actually changed, restated precisely

Production model: for every training pair, `s=1` is defined as "this pair's own real
post-disaster image," regardless of how severe that pair's real damage actually was.
So `s=1` does not mean "maximum damage" in any universal sense — it means "whatever
happened to this particular scene."

Calibrated model: for the 366/769 train pairs (103/246 test pairs) with a reliable
`S_real` label, `s=1` is redefined to mean "maximum universal severity," and that
pair's own real post image is anchored at `s_i = S_real/3 < 1` instead. Training only
directly supervises `s ∈ [0, s_i]` for that pair; `s > s_i` is never given a direct
target for that specific scene and can only be learned by generalizing across other
training pairs whose real endpoints sit higher on the scale. Unlabeled pairs (403/769
train, 143/246 test) still use `s_i=1.0`, unchanged from production.

This means the interesting, falsifiable claim to test is narrow and specific:

> For scenes whose real damage was low (S_real≈0), does requesting `s=1` from the
> calibrated model produce something *more damaged-looking* than requesting `s=1`
> from the production model — where production, unable to do anything but reproduce
> that scene's own (near-undamaged) real post image, necessarily fails to look
> "maximally damaged"?

Everything else (image quality at points where a real target exists, whether the
generation module still trains stably, whether the refiner still works) is a
sanity check, not the headline result.

## 1. Why CAS accuracy as previously defined does NOT directly transfer

The original CAS metric checks: does real-pre classify as "pre" and does gen-post
(at s=1, using each pair's OWN post image as ground truth) classify as "post"? That
metric is meaningless here for two reasons:
- For labeled pairs with low S_real, there is no real image at s=1 to check against
  (by construction, we're asking the model to extrapolate past what happened).
- The classifier itself is a binary pre/post classifier, not a severity regressor —
  it was never designed to detect "how damaged," only "damaged at all."

So CAS accuracy computed the old way would be uninformative or misleading here — do
NOT reuse Tables 7-11's CAS numbers as if they were comparable to the calibrated
model without adjustment.

## 2. Concrete evaluation script plan

### 2.1 Core test: per-scene P(damaged) vs S_real at requested s=1 (headline result)

Using the clean, leak-free classifier (`output-models/classifier/socalfire_cls_clean_split/best.pt`,
already the one used for Phase 0/1 fixes elsewhere in the paper):

For all 103 labeled test pairs, load the **refined** (f0/fu/f1 pipeline output, not
raw generator output) `scale1` image from:
- production: `outputs/refine-2/socalfire/test/scale1/{id}.png`
- calibrated: `outputs/refine-2/socalfire_calibrated/test/scale1/{id}.png`

Compute `P(damaged)` (softmax prob of the "post" class) for each image, for both
models. Join against `S_real` from `claudecode/data/socalfire/severity_labels.csv`.

Report, for both models:
- Spearman correlation between `S_real` and `P(damaged)@s=1`.
  - Production is EXPECTED to show a positive correlation here (bad — it means
    "s=1" output severity still just tracks whatever really happened to that scene,
    i.e., s=1 isn't universal max-severity, it's pair-specific).
  - Calibrated is HOPED to show a much weaker (ideally near-zero or slightly negative)
    correlation — meaning s=1 pushes most scenes toward high P(damaged) regardless
    of their real outcome.
- Mean `P(damaged)@s=1` split into the same 4 severity buckets used elsewhere in the
  paper (0 none / 1 minor / 2 major / 3 destroyed; test-set counts: 49/20/17/17).
  This is the most intuitive, defensible way to present the result: a bar chart or
  table of mean P(damaged) by bucket, for both models side by side.

**Honesty checkpoint**: if calibrated does NOT show a flatter correlation than
production, or if its P(damaged) for low-S_real scenes is not meaningfully higher
than production's, report that plainly. Do not cherry-pick a different metric to
manufacture a positive result (per the user's explicit earlier instruction not to
metric-shop).

### 2.2 Secondary test: same analysis at intermediate calibrated scales

Repeat 2.1's per-scene P(damaged)-vs-S_real correlation at s=0.5 and s=0.75 (using
the calibrated model's own `s` values directly, which now should correspond to an
absolute severity level ~1.5 and ~2.25 out of 3). If the calibration is working, a
scene with real S_real≈1.5 driven to s=0.5 should show similar P(damaged) to a scene
with real S_real≈2.25 driven to s=0.75 — i.e., matching REQUESTED absolute severity
should now predict P(damaged) better than the scene's own real S_real does, which is
the reverse of a well-calibrated system's failure mode. This reuses the exact
methodology already used in `scene_level_calibration.md` (Phase 1), just re-run for
the calibrated model in addition to production, so the same script can largely be
reused/adapted rather than rewritten from scratch.

### 2.3 Sanity check: image quality where ground truth exists

Restrict to labeled test pairs with S_real ≥ 2.5 (already near s_i≈1, i.e., their
own real post image approximately IS the s=1 target for both models — 17 test pairs
per the "3 destroyed" bucket, possibly also include "2 major" as a softer cutoff for
more samples). For these pairs only, compute LPIPS / SSIM / PSNR between the refined
`scale1` output and the real post image, for both production and calibrated. This
checks that retraining f1/fu on the smaller calibrated subset (81 train pairs vs.
production's full 769) has not visibly hurt reconstruction quality where a genuine
target exists. Do NOT compute this metric for low-severity pairs — there is no valid
target there, and doing so anyway (comparing an intentionally-more-damaged
extrapolation against an undamaged real image) would manufacture an artificially bad
number that misrepresents what the model is being asked to do.

### 2.4 What NOT to do

- Do not run the old Table 7-11 style aggregate CAS accuracy on the calibrated
  model's full scale sweep and present it side-by-side with production's numbers as
  if directly comparable — the semantics of `s` differ, so an "improvement" or
  "regression" in that number would not mean what a reader would assume it means.
- Do not compute LPIPS/SSIM/PSNR at s=1 against real post images for LOW-severity
  labeled pairs (discussed above).
- Do not swap metrics after seeing an unfavorable result. If 2.1's headline
  correlation comparison comes out null or unfavorable, report that as the finding,
  the same way the smoothness ablation (`lambda_smooth_ablation.md`) and the naive
  interpolation confound were reported honestly earlier in this revision.

## 3. Execution order

1. Write `claudecode/code/calibrated_severity/eval_scene_calibration_compare.py`
   implementing 2.1 and 2.2 (reusing `build_classifier`/`classifier_transform`/
   `classify_batch` helpers already written in
   `claudecode/code/classifier_fix/recompute_cross_tables.py`).
2. Run it, producing a CSV of per-scene `(id, S_real, bucket, P_damaged_production,
   P_damaged_calibrated)` at s=1, 0.75, 0.5, plus the aggregate correlation/bucket-mean
   summary.
3. Separately compute 2.3's quality sanity check for the high-severity subset.
4. Write up the honest result in
   `claudecode/paper_draft/calibrated_severity_final_evaluation_results.md`
   (a distinct file from this plan, so the plan and the findings don't get
   conflated if the finding is later revised).
5. Update `claudecode/paper_draft/OVERVIEW.md` to point at both files and mark
   whether the calibrated variant is presented in the paper as (a) an accepted
   improvement, (b) an honestly-reported mixed/null finding used to preempt a
   reviewer objection, or (c) future work — decision deferred until step 4's actual
   numbers are in.
