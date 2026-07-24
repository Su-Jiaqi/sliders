# Table 2 / tab:app_ablation_multiscale CAS mismatch vs. tab:app_continuous_generation_scale — root-caused

## The discrepancy reported

User found that `tab:app_continuous_generation_scale` (unrefined generator, s=1,
socalfire test) reports CAS=0.6850, while Table 2 / `tab:app_ablation_multiscale`'s
"w/o Refinement" row at s=1 reports CAS=0.7093 — both describing the exact same
thing (raw generator output, no refiner, s=1, socalfire test split). LPIPS/SSIM/
PSNR/FID/CLIP-I/DINO-I agree exactly between the two; only CAS differs.

## Root cause: a third, previously-unaudited classifier checkpoint — same bug class as the Table 1 CAS finding, not a new category

Both numbers trace to real files, generated from the **same underlying images**
(`outputs/infer/socalfire/test/scale1`), scored by **two different classifier
checkpoints**:

- **0.6850** (`tab:app_continuous_generation_scale`) traces exactly to
  `claudecode/result/classifier_fix/table14_clean/socalfire_ablation_multiscale_cas_test.csv`
  (0.6849593495934959) and independently to
  `claudecode/result/classifier_fix/cross_tables/table7_unrefined_socalfire.csv`
  (Phase 0, 2026-07-17), both scored with the confirmed leak-free
  `output-models/classifier/socalfire_cls_clean_split/best.pt`.

- **0.7093** (Table 2 / `tab:app_ablation_multiscale`, "w/o Refinement") traces
  exactly to `outputs/eval/ablation/socalfire_scale1_compare_full/direct/socalfire-infered_metrics.csv`
  (0.709349593495935, dated 2026-05-03), which feeds
  `outputs/eval/ablation/ablation-compare/socalfire_ablation_multiscale_cas_test.csv`
  — the actual source file behind both Table 2 and the appendix table (they cite
  identical numbers because they are literally the same csv). This run
  (`logs/ablation-2/run_socalfire_scale1_ablation_eval.sh`) used
  `output-models/cas_cross_domain/socalfire_endpoint_formal/_shared_real_classifier/best.pt`
  — **the exact same third-checkpoint already flagged in `table1_significance.md`**
  as the source of Table 1's stale RiskSlider CAS (0.9939, should be 0.9878). That
  checkpoint is methodologically clean (val-selected on a train-only 80/20 split,
  confirmed via its training code in `eval/cas_cross_domain/run_cross_domain_cas.py`)
  but is a *separate training run* from Phase 0's `socalfire_cls_clean_split`, so it
  scores images differently.

**This is the same bug already identified for Table 1, now confirmed to also
contaminate Table 2 / `tab:app_ablation_multiscale`.** `run_socalfire_scale1_ablation_eval.sh`
scores all four ablation variants (direct/refined/no_pseudo/no_scale) with this one
`_shared_real_classifier` checkpoint, so the whole table — not just the
"w/o Refinement" row — was never scored by the same classifier as Phase 0's
recompute, even though `SUMMARY.md` claimed "Table 2's CAS column is identical to
Table 14's s=1 column" (that claim was only ever verified against the *documentation*,
never against the actual csv/tex sitting in `outputs/eval/ablation/`).

## Effect on all four rows (s=1)

| variant | current (Table 2 / app_ablation_multiscale) | correct (table14_clean, clean classifier) | Δ |
|---|---|---|---|
| w/o Refinement | 0.7093 | 0.6850 | -0.0244 |
| w/o Scale-aware Refiner | 0.9837 | 0.9837 | 0.0000 (coincides) |
| w/o Pseudo Supervision | 0.9919 | 0.9898 | -0.0020 |
| Ours | 0.9939 | 0.9878 | -0.0061 |

Largest shift is on "w/o Refinement" — consistent with the already-documented
pattern (`SUMMARY.md`) that classifier-choice sensitivity is largest for the most
visually ambiguous/borderline images, which the unrefined generator produces most of.

## Recommended fix

No new compute needed. `table14_clean/socalfire_ablation_multiscale_cas_test.tex`
already has the correct, single-classifier numbers for all four variants at all five
scales (0/0.25/0.5/0.75/1). Replace both Table 2's CAS column and
`tab:app_ablation_multiscale`'s CAS column with that file's numbers directly — same
fix pattern as the Table 1 CAS finding and the `appendix_table_consistency_20260724.md`
finding (both of which also resolve to "standardize on `socalfire_cls_clean_split`,
everywhere CAS is cited"). LPIPS/SSIM/PSNR/FID/CLIP-I/DINO-I in Table 2 are
classifier-independent and need no change.

Not yet checked: whether Table 2's non-CAS-column citation elsewhere, or any other
appendix table, still cites `_shared_real_classifier`-scored CAS values — worth a
full grep sweep for `_shared_real_classifier` usages the next time Table 1/2/appendix
edits are made, to catch any remaining instance in one pass rather than one at a time.
