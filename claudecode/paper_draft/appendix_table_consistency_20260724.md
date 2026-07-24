# Appendix table cross-consistency check: tab:app_refined_scale vs. tab:app_ablation_multiscale ("Ours" row)

## The discrepancy reported

User found that two appendix tables, which should both describe the same trained
RiskSlider model on the same socalfire test split, disagree at every intermediate
scale:

| s | tab:app_refined_scale | tab:app_ablation_multiscale ("Ours") |
|---|---|---|
| 0.25 | 0.7195 | 0.7846 |
| 0.50 | 0.8801 | 0.9248 |
| 0.75 | 0.9756 | 0.9837 |
| 1.00 | 0.9939 | 0.9939 (matches) |

Only the endpoint agrees; the gap at intermediate scales (0.06-0.07) is far larger
than the paper's own seed-noise floor (0.001-0.002, Table 3).

## Root cause: found, and it is NOT the same classifier issue as Table 1 — it's a different checkpoint entirely

Traced both tables to their exact source CSVs on this machine.

**tab:app_refined_scale** exactly matches
`outputs/eval/socalfire-refine/socalfire-refined_metrics.csv` (dated 2026-04-22) to
10 decimal places: 0.25→0.7195121951219512, 0.5→0.8800813008130082,
0.75→0.975609756097561, 1→0.9939024390243902. This eval's `infer_root` (recorded in
its sibling `.json`) is **`outputs/refine/socalfire`** — note: `refine`, not
`refine-2`. This is an *earlier* refiner pipeline iteration, generated and evaluated
in April, before the current production "refine-2" pipeline
(`unified_refiner_scale1stronger`, used for Table 1 and everything else in this
project going forward) existed.

**tab:app_ablation_multiscale's "Ours" row** matches (3 of 4 points exactly, 1 point
off by ~1 image out of 492) `outputs/refine-2/socalfire/eval-full/socalfire-infered_metrics.csv`
test-split row: 0.25→0.7865853658536586 (table says 0.7846, off by ~0.002), 0.5→0.9247967479674797
(exact), 0.75→0.983739837398374 (exact), 1→0.9939024390243902 (exact). This is the
**current production "refine-2" pipeline** -- the same one that generates
`outputs/refine-2/socalfire/test/scale1`, which feeds Table 1's RiskSlider row.

**So the two appendix tables are not disagreeing because of a classifier mismatch --
they are simply two different model/generation runs from two different points in the
project's development, both labeled "Ours"/"RiskSlider" as if they were the same
result.** `outputs/refine/socalfire` (the older run) predates `outputs/refine-2`
entirely -- different refiner training run, quite possibly a different generator
checkpoint underneath it too, given the April timestamp is close to several other
early-stage experiment iterations. The two runs happen to converge to a very similar
number at s=1 (both refiners are strongly endpoint-supervised, so the trained
endpoint is where they agree most), but diverge substantially at intermediate scales,
where the two different refiner training runs behave differently.

**Correction (2026-07-24, later same day)**: this section originally claimed the old
run's images "no longer exist on this machine" (`outputs/refine/socalfire` was
checked and found missing). That was wrong — the images survive under a renamed path,
`outputs/refine-1/socalfire/test/scale*` (all 7 scales, n=246). Verified by re-running
LPIPS/SSIM/PSNR on that directory: bit-for-bit identical to
`socalfire-refined_metrics.csv`'s test row at every scale. See
`appendix_cross_category_tables_20260724.md` for the eval that surfaced this (found
while root-causing a third, related instance of this bug in the cross-category
appendix table). This doesn't change the recommendation below — still standardize on
the current, reproducible `refine-2` pipeline, not the superseded `refine-1` one —
but the "unrecoverable" framing was inaccurate and worth flagging in case a future
pass wants to double-check the old run directly instead of trusting only the
surviving CSV.

## Recommended fix

**Both appendix tables' "Ours"/RiskSlider row should be replaced with the same
numbers, from the same (current, reproducible) pipeline, scored by the same (clean,
leak-free) classifier** -- i.e. `table8_9_refined_socalfire.csv` / `table14_clean`
(`claudecode/result/classifier_fix/cross_tables/table8_9_refined_socalfire.csv` and
`claudecode/result/classifier_fix/table14_clean/socalfire_ablation_multiscale_cas_test.csv`,
already computed in Phase 0, and independently confirmed to be internally consistent
with each other -- they agree exactly at every shared scale point):

| s | Correct, consistent value (both tables) |
|---|---|
| 0.25 | 0.7378 |
| 0.50 | 0.9085 |
| 0.75 | 0.9837 |
| 1.00 | 0.9878 |

This is the same underlying issue category as the Table 1 CAS finding
(`table1_significance.md`) -- different tables in the paper, generated at different
points during iterative development, ended up citing results from different
checkpoints/classifiers without anyone noticing, because each table was individually
correct at the time it was made. The fix in both cases is the same: standardize on
the current production pipeline (`refine-2` generation + `unified_refiner_scale1stronger`
refiner) scored by the confirmed leak-free classifier (`socalfire_cls_clean_split`),
and replace every appendix/main-text citation of "Ours"/"RiskSlider" CAS numbers with
that single, consistent source.

## Caveat
This does not affect LPIPS/SSIM/PSNR/FID/CLIP-I/DINO-I in either table -- those are
classifier-independent, and (for the refine-2/eval-full source at least) already
confirmed correct in the Table 1 provenance audit. Only CAS accuracy is affected,
because it's the only metric that depends on which classifier scores it, and here
also depends on which generation/refiner run produced the underlying images.
