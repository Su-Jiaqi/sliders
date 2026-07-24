# appendices/04's `cas_lpips_vs_scale_test.png` — source script/data found, plus a staleness caveat — 2026-07-25

## What was flagged

The figure's title and y-axis still say "CAS Accuracy," but this label is being
renamed project-wide (see the CAS-rename decision). Couldn't redraw it because (a)
no plotting script could be found, and (b) no "w/o Pseudo Supervision" mid-scale
LPIPS data could be found either.

## Both blockers are resolved — the script and data exist

- **Script**: `eval/cas-lpips-scale.py`. Takes `--ours_json` / `--no_pseudo_json`,
  plots two panels ((a) CAS-vs-scale, (b) LPIPS-vs-scale) for "Ours" vs. "w/o Pseudo
  Supervision" across all 7 scales, saves to the exact `outputs/eval/cas_lpips_vs_scale_test.{png,pdf}`
  paths the figure lives at. Renaming the y-axis label is a one-line edit
  (`ax.set_ylabel("CAS Accuracy", ...)` → the new name) and a rerun — no new
  evaluation needed for the rename itself.
- **Data**: the script's own default `--no_pseudo_json` already points at
  `outputs/eval/ablation/socalfire_unified_no_pseudo_20260422_103055/socalfire-infered_metrics.json`,
  which has full 7-scale rows (LPIPS, CAS, SSIM, PSNR) for the "w/o Pseudo
  Supervision" variant — nothing is missing.

## New finding while checking this: the figure predates the classifier/pipeline fixes found elsewhere this project

The script's default `--ours_json` is `outputs/eval/socalfire-refine/socalfire-refined_metrics.json`
— **the old refine-1 pipeline** (`infer_root: outputs/refine/socalfire`), not current
production `refine-2`. This is the same stale-pipeline file already identified and
fixed in `appendix_table_consistency_20260724.md` / `appendix_cross_category_tables_20260724.md`.

Checked classifier provenance for both curves in this figure: neither JSON records
its `classifier_ckpt` (both echo `None`, consistent with how these early-project eval
runs logged nothing). But timestamps rule out any classifier except the very first
one: the "Ours"/refine-1 file was generated 2026-04-21, the "w/o Pseudo Supervision"
file 2026-04-22 — and as of those dates, `output-models/classifier/socalfire_cls_20260421_201243/best.pt`
(2026-04-21 20:15) was the *only* classifier checkpoint that existed anywhere in the
project (`_shared_real_classifier`, `socalfire_cls_real_fresh`, and
`socalfire_cls_clean_split` were all created in May or July). So both curves in this
figure are internally consistent with each other (same classifier, not a
cross-contamination like Table 2's), but both rest on the original leaked-selection
checkpoint, and the "Ours" curve is the stale pipeline.

## Recommendation

Two separable decisions:
1. **CAS label rename**: mechanical, no new compute — edit the script's axis label
   and rerun on the existing JSONs.
2. **Whether to also regenerate this figure on the clean classifier + current
   refine-2 pipeline** (for full consistency with the Table 2 / cross-category fixes
   already applied elsewhere): a real option, but a separate decision from the
   rename, since it would require rerunning "w/o Pseudo Supervision" through the
   clean classifier (straightforward re-scoring, images already exist) and possibly
   regenerating "Ours" from refine-2 at all 7 scales (already exists at
   `outputs/refine-2/socalfire/test/scale*`, so also just re-scoring, no new
   generation). Flagging this rather than doing it unprompted, since it changes the
   figure's actual curve shape, not just its label.
