# appendices/03_experimental_results.tex "Cross-Category Quantitative Results" — Wildfire row root-caused, other 4 rows verified clean

## What was reported

Two tables in the same appendix section, both with a "Wildfire (primary)" row:

- `tab:app_cross_category_s1` (line ~416): single s=1 snapshot per category
  (LPIPS/SSIM/PSNR/CAS). Wildfire row: LPIPS 0.2166, SSIM 0.5954, PSNR 23.81,
  CAS 0.9939.
- `tab:app_cross_category_cas` (line ~437): multiscale CAS trend per category
  (s=0/0.25/0.5/0.75/1). Wildfire row: 0.618, 0.720, 0.880, 0.976, 0.994.

The non-CAS metrics in `tab:app_cross_category_s1`'s Wildfire row match
`tab:app_refined_scale`'s Test s=1 row exactly, but CAS doesn't (0.9939 vs the
corrected 0.9878) — same symptom pattern as every other instance of this bug found
this session. Separately, `tab:app_cross_category_cas`'s whole Wildfire row differs
from `tab:app_refined_scale`'s Test row by more than rounding at every intermediate
scale, suggesting a genuinely different source, not just a different classifier.

## Root cause 1 — tab:app_cross_category_s1's Wildfire row: same `_shared_real_classifier` bug as Table 1/2

Traced exactly: `outputs/refine-2/socalfire/eval-full/socalfire-infered_metrics.csv`,
test/scale1 row — LPIPS 0.21660283650082301, SSIM 0.5953810033072361,
PSNR 23.806210092130513, CAS 0.9939024390243902. **This is the identical row already
flagged in `table1_significance.md`** as Table 1's stale RiskSlider CAS, scored with
`output-models/cas_cross_domain/socalfire_endpoint_formal/_shared_real_classifier/best.pt`.
Fix: same as Table 1/2 — swap CAS to `socalfire_cls_clean_split`'s 0.9878048780487805
(`table14_clean` / `table10_11_percategory_CLEAN.csv`, SoCal Wildfire row, s=1). No
new compute needed.

**Follow-up check (2026-07-24, later same day)**: user flagged that
`tab:app_cross_category_s1`'s Hurricane/Flooding/Santa Rosa/Volcano CAS values
(0.9948, 1.0000, 0.8617, 1.0000) don't match the clean s=1 values recommended below
for those categories (0.954, 1.000, 0.723, 0.500) — Volcano in particular looks like a
flat contradiction (1.0 "always classified post-disaster" vs 0.5 "coin flip"), and
asked whether this exposes a second, un-investigated problem in the endpoint table
before applying any fix. **Checked and confirmed it is not a new problem**: all four
values trace exactly to `table10_11_percategory_LEAKED_verify.csv`'s s=1 column
(Hurricane 0.9948453608247423, Flooding 1.0, Santa Rosa 0.8617021276595744,
Volcano 1.0 — exact matches, re-confirmed against the underlying per-category
`eval-full`/`eval-full-rerun` csvs for Hurricane and Santa Rosa specifically). This is
the same per-category **leaked** classifier (`classifier_real_fresh`) already
identified as the source of `tab:app_cross_category_cas`'s other four rows — i.e.
`tab:app_cross_category_s1`'s endpoint values for these four categories are just the
s=1 slice of the exact same (leaked) computation as the trend table, not an
independent, differently-sourced number. The apparent "contradiction" the user saw is
expected and by construction: it's the leaked-vs-clean gap, the same gap already being
fixed for every other row in both tables — not evidence of a third source. Volcano's
1.0→0.5 swing in particular is fully consistent with N=4 (2/4 correct either way is a
40-percentage-point step) and the paper's own "too small for reliable estimation"
caveat. **Both tables can be fixed together from the single `table10_11_percategory_CLEAN.csv`
source with no remaining inconsistency between them.**

## Root cause 2 — tab:app_cross_category_cas's Wildfire row: a completely different, older generation run + the original leaked classifier (not `_shared_real_classifier` this time)

Traced exactly (all 5 points, to 10+ decimal places) to
`outputs/eval/socalfire-refine/socalfire-refined_metrics.csv`, test split:
s=0→0.6178861788617886, s=0.25→0.7195121951219512, s=0.5→0.8800813008130082,
s=0.75→0.975609756097561, s=1→0.9939024390243902 (all round to the reported
0.618/0.720/0.880/0.976/0.994 at 3 decimals — note this appendix table is printed to
3 decimals, not 4, which is why it wasn't caught by a naive 4-decimal string
comparison against `tab:app_refined_scale`).

This csv's `infer_root` is `outputs/refine/socalfire` — **the same old, pre-`refine-2`
generation run already identified in `appendix_table_consistency_20260724.md`** as the
source of `tab:app_refined_scale`'s Test row. It was evaluated on 2026-04-22 15:14
(`logs/eval/socalfire-refine-infered_20260422_151423.log`). No `classifier_ckpt` is
recorded in the run's own metadata, but at that timestamp **the only classifier
checkpoint that existed anywhere on this machine** was
`output-models/classifier/socalfire_cls_20260421_201243/best.pt` — the exact checkpoint
`table1_significance.md` already confirmed is the leaked one (test-set-selected via
`classifier/train.py:326`) used to score the four Table 1 baselines. So this row is
doubly wrong: wrong generation run (superseded `refine-1`, not production `refine-2`)
*and* the original leaked classifier (not even `_shared_real_classifier` — an older,
worse problem).

**Correction to a claim in `appendix_table_consistency_20260724.md`**: that file states
the old run's images "no longer exist on this machine" and "cannot be regenerated or
independently re-verified." This is now known to be inaccurate — the images are still
present at `outputs/refine-1/socalfire/test/scale*` (all 7 scales, n=246 at every
reported scale). I re-ran LPIPS/SSIM/PSNR on this directory and got a bit-for-bit
match to `socalfire-refined_metrics.csv`'s test row at every scale (n=246 exactly,
metric values identical to the digits recorded) — `outputs/refine/socalfire` was
simply renamed to `outputs/refine-1/socalfire` at some point (natural, given
`refine-2` was introduced later), not deleted. This doesn't change the fix
recommendation (still standardize on current `refine-2` + clean classifier, per the
existing convention), but the "unrecoverable" framing in that earlier file should be
corrected — I also verified the clean-classifier CAS trend *of that old run itself*
for completeness (not the recommended replacement, just a sanity data point):
0.622 / 0.726 / 0.870 / 0.976 / 0.992 at s=0/0.25/0.5/0.75/1
(`claudecode/result/classifier_fix/table_app_cross_category_wildfire_refine1_clean/`).

## The other four rows (Hurricane / Flooding / Santa Rosa / Volcano) — checked against user-supplied PDF text, confirmed clean/consistent, NOT the same run as Wildfire

User supplied the actual `tab:app_cross_category_cas` text for the other four
categories:

| Category | s=0 | s=0.25 | s=0.5 | s=0.75 | s=1 |
|---|---|---|---|---|---|
| Hurricane (N=97) | 0.500 | 0.500 | 0.686 | 0.938 | 0.995 |
| Flooding (N=80) | 0.500 | 0.525 | 0.956 | 1.000 | 1.000 |
| Santa Rosa (N=47) | 0.489 | 0.500 | 0.564 | 0.723 | 0.862 |
| Volcano (N=4) | 0.750 | 0.875 | 1.000 | 1.000 | 1.000 |

These match `claudecode/result/classifier_fix/cross_tables/table10_11_percategory_LEAKED_verify.csv`
(the per-category **leaked** classifiers, `output-models/refine-2/{category}/classifier_real_fresh/best.pt`,
already characterized in Phase 0/`SUMMARY.md`) almost exactly: **Flooding and Volcano
match at all 5 points**; **Hurricane and Santa Rosa match at 4/5 points, off only at
s=0.75** (Hurricane: table says 0.938, LEAKED_verify says 0.964; Santa Rosa: table
says 0.723, LEAKED_verify says 0.766) — this is the *exact* discrepancy pattern
`SUMMARY.md` already documented and attributed to inference-seed variance
(paper's own Table 3 noise floor), not a new issue.

**Conclusion: these four rows are genuinely Table 10/11-family data (current
`refine-2` pipeline, per-category classifiers), computed by the same script/run as
each other, and NOT the same run as the Wildfire row.** This confirms the user's
suspicion was right in spirit (worth checking whether all 5 rows shared one script)
but the actual finding is sharper: it's structurally impossible for Wildfire to share
a run with these four, because the old `refine-1`-based Wildfire eval (April 22, 2026)
predates the existence of `refine-2` and any per-category refiner/classifier for
Hurricane/Flooding/Santa Rosa/Volcano (all created 2026-05-02/05-03, per
`output-models/refine-2/{category}/` timestamps). The table was assembled from two
different eras of experiments, not one run — Wildfire is the sole outlier, not a
symptom of a table-wide problem.

## Recommended fix

Replace the entire `tab:app_cross_category_cas` table (all 5 rows) with
`table10_11_percategory_CLEAN.csv` — already computed, already the designated fix for
Table 10/11, and already internally verified consistent with `table14_clean` for the
Wildfire row specifically:

| Category | s=0 | s=0.25 | s=0.5 | s=0.75 | s=1 |
|---|---|---|---|---|---|
| SoCal Wildfire (primary) | 0.636 | 0.738 | 0.909 | 0.984 | 0.988 |
| Hurricane | 0.479 | 0.521 | 0.763 | 0.938 | 0.954 |
| Flooding | 0.506 | 0.544 | 0.969 | 1.000 | 1.000 |
| Santa Rosa Wildfire | 0.500 | 0.500 | 0.553 | 0.660 | 0.723 |
| Volcano (N=4) | 0.500 | 0.625 | 0.500 | 0.500 | 0.500 |

Note Volcano's clean-classifier trend is flat/non-monotonic — already covered by the
paper's own "too small for reliable estimation, N=4" caveat, no new framing needed.

For `tab:app_cross_category_s1`, only the CAS column needs replacing (per category,
s=1 value from the table above); LPIPS/SSIM/PSNR are classifier-independent and
unaffected.

No new compute needed for any of this — every number above already exists in
`claudecode/result/classifier_fix/cross_tables/table10_11_percategory_CLEAN.csv`.
