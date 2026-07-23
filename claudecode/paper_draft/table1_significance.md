## 🔴 CRITICAL, HIGHEST PRIORITY (found 2026-07-24): Table 1's CAS column mixes THREE different classifier checkpoints — must be replaced before submission

This was found during a server-side audit triggered by an external reviewer's
suggestion to double-check "is CAS consistent across Table 1's rows." It is NOT a new
experiment — it is a provenance bug in numbers already sitting in the compiled PDF, and
it is more urgent than any of the exploratory experiments discussed elsewhere in this
directory (C1-C7 style ideas). **Read this before doing anything else with Table 1.**

**What's wrong**: Table 1's CAS Accuracy column currently in the compiled PDF
(`RiskSlider.pdf`, Table 1, page 6) is:

| Method | Published CAS |
|---|---|
| ControlNet | 0.9980 |
| Pix2Pix | 0.9919 |
| CycleGAN | 0.9898 |
| Palette | 0.9891 |
| RiskSlider (Ours) | 0.9939 |

These five numbers were **not all produced by the same classifier checkpoint**, which
makes CAS Accuracy in the current Table 1 an apples-to-oranges comparison, not a single
consistent metric:

- ControlNet/Pix2Pix/CycleGAN/Palette's original CAS numbers trace to eval runs from
  2026-04-21, using classifier checkpoint
  `output-models/classifier/socalfire_cls_20260421_201243/best.pt` — **this is the
  exact checkpoint Phase 0 (`claudecode/result/classifier_fix/SUMMARY.md`) flagged as
  leaked** (best-checkpoint selected directly on the 246-pair test split,
  `classifier/train.py:326`).
- RiskSlider's published CAS (0.9939) traces to
  `outputs/refine-2/socalfire/eval-full/socalfire-infered_metrics.csv` (dated
  2026-05-03), which was scored with a **third, different classifier**:
  `output-models/cas_cross_domain/socalfire_endpoint_formal/_shared_real_classifier/best.pt`
  (created 2026-05-02, via `eval/cas_cross_domain/run_cross_domain_cas.py`). This
  checkpoint is itself methodologically clean — I traced its training code
  (`train_classifier` in that script) and confirmed it selects the best checkpoint via
  held-out validation accuracy on an 80/20 split of the *train* pool only, never
  touching the test set — so it is **not** the leaked checkpoint. But it is a
  *different* classifier from both the leaked one (used for the 4 baselines) and
  Phase 0's `socalfire_cls_clean_split` fix, so RiskSlider's row was never actually
  scored by the same yardstick as any other row in the table, at any point in this
  project's history.
- Phase 0 (2026-07-17) already recomputed a fully-consistent CAS column — all five
  methods scored by the same classifier, `socalfire_cls_clean_split` — see
  `SUMMARY.md`'s "Leaked CAS / Clean CAS" table. **This consistent recomputation was
  never applied to the actual Table 1 / PDF.** I directly re-verified RiskSlider's
  clean-classifier number myself just now (re-running
  `eval/socalfire_infered_eval_metrics.py` on the exact production folder
  `outputs/refine-2/socalfire/test/scale1`, n=246, with
  `socalfire_cls_clean_split/best.pt`): **0.9878048780487805**, matching SUMMARY.md's
  Phase 0 number to 10 decimal places. This is the correct, defensible RiskSlider CAS
  number — not 0.9939.

**The correct, single-classifier CAS column for Table 1** (combining Phase 0's
`socalfire_cls_clean_split` recompute with the already-resolved Palette n=246 fix
below):

| Method | Correct CAS (socalfire_cls_clean_split, consistent across all 5) |
|---|---|
| ControlNet | 0.9939 |
| Palette | 0.9919 |
| Pix2Pix | 0.9898 |
| RiskSlider (Ours) | 0.9878 |
| CycleGAN | 0.9837 |

**This changes the paper's narrative, not just the numbers.** Under the currently
published (inconsistent) column, RiskSlider's CAS (0.9939) reads as tied-for-best.
Under the corrected, single-classifier column, **RiskSlider is 4th of 5** — beating
only CycleGAN, and behind ControlNet/Palette/Pix2Pix. This is exactly the finding
SUMMARY.md already flagged in its own words ("RiskSlider can no longer be framed as
CAS-competitive") but it was apparently never carried through to the actual
submission. LPIPS/SSIM/PSNR/FID/CLIP-I/DINO-I are unaffected (they don't depend on a
classifier) — only the CAS column needs replacing. **Action: before submission,
replace Table 1's entire CAS column with the table above, and adjust any prose that
currently frames RiskSlider as winning or tying on CAS** — the honest framing is that
RiskSlider wins decisively on structural fidelity (LPIPS/SSIM/PSNR, all p<1e-28) and
semantic similarity against most baselines (CLIP-I/DINO-I, p<1e-7 except CycleGAN),
but is not the best method by this particular classifier-based proxy metric.

**Real-image reference point (new, pure eval, no generation involved)**: to
contextualize how tight these CAS differences actually are, I directly evaluated
`socalfire_cls_clean_split` on the genuine real test images
(`datasets/remote/socalfire/test/{pre,post}`, n=492, ImageFolder-loaded via the same
`classifier/train.py` utilities used for training) — this is the classifier's own
ceiling, no generative model involved at all. Confusion matrix (rows=true, cols=pred,
[pre,post]): `[[243,3],[2,244]]`, overall accuracy 0.98984. In particular, **real
post-disaster photos are classified as "post" 244/246 = 0.9919 of the time** — this is
the practical ceiling for the metric CAS is approximating at s=1. RiskSlider's
generated s=1 images are classified as "post" 243/246 = 0.9878 of the time — a gap of
exactly **one image out of 246** relative to genuine real photos. So while RiskSlider
is 4th of 5 among the generative methods on this metric, it is essentially
indistinguishable from real post-disaster imagery by the classifier's own standard;
the entire competitive spread between all 5 methods (0.9837-0.9939) sits within about
2-3 images of the real-image ceiling itself (0.9919), and CAS at s=1 is close to
saturated for every method tested, real photos included. Useful honest context to pair
with the "4th of 5" finding above — the ranking is real, but the absolute practical
gap it reflects is tiny.

---

# Table 1 per-image paired significance — draft insert

Insert location: footnote under Table 1, or one sentence in "Comparison with Baselines"
right after the Table 1 discussion paragraph.

## Results (socalfire test split, n=246 for ALL five methods — see "Palette n=246 fix" below)

| Baseline | metric | RiskSlider mean | Baseline mean | Wilcoxon p |
|---|---|---|---|---|
| ControlNet | LPIPS | 0.2166 | 0.4519 | 4.2e-42 |
| ControlNet | SSIM | 0.5954 | 0.3813 | 4.2e-42 |
| ControlNet | PSNR | 23.81 | 17.38 | 4.7e-42 |
| Pix2Pix | LPIPS | 0.2166 | 0.3022 | 4.2e-42 |
| Pix2Pix | SSIM | 0.5954 | 0.4729 | 4.2e-42 |
| Pix2Pix | PSNR | 23.81 | 21.80 | 1.4e-41 |
| CycleGAN | LPIPS | 0.2166 | 0.2739 | 5.8e-42 |
| CycleGAN | SSIM | 0.5954 | 0.5182 | 1.2e-41 |
| CycleGAN | PSNR | 23.81 | 21.80 | 5.2e-42 |
| Palette | LPIPS | 0.2166 | 0.6016 | 4.2e-42 |
| Palette | SSIM | 0.5954 | 0.3213 | 4.2e-42 |
| Palette | PSNR | 23.81 | 16.27 | 4.6e-42 |

All comparisons two-sided Wilcoxon signed-rank test, paired by test scene.
Full script: `claudecode/code/classifier_fix/paired_significance_table1.py`.

## Draft prose (one sentence, for the main Results paragraph after Table 1)

To confirm these structural-fidelity gains are not an artifact of aggregate
averaging, we additionally ran a two-sided Wilcoxon signed-rank test, pairing
LPIPS/SSIM/PSNR by test scene between RiskSlider and each baseline; RiskSlider's
advantage on all three metrics is significant against every baseline at p < 1e-28
(Appendix Table [S]), including CycleGAN, the baseline closest to RiskSlider on the
distributional/semantic metrics (FID/CLIP-I/DINO-I).

## Notes
- This is unambiguously good news: every LPIPS/SSIM/PSNR margin in Table 1 is
  robust at the per-image level, not just in aggregate. This directly answers
  the "is your FID/structural advantage real or cherry-picked averaging"
  question a reviewer would likely raise given CycleGAN's competitive FID
  (178.72 vs 175.19).

## CLIP-I / DINO-I per-image paired significance (done)

Ran the same per-image paired Wilcoxon protocol on CLIP-I/DINO-I (per-image cosine
similarity between each method's output and the real post-disaster target, paired by
test scene). Script: `claudecode/code/classifier_fix/paired_significance_clip_dino.py`.

| Baseline | metric | RiskSlider mean | Baseline mean | Wilcoxon p |
|---|---|---|---|---|
| ControlNet | CLIP-I | 0.9281 | 0.8437 | 9.7e-41 |
| ControlNet | DINO-I | 0.7986 | 0.5993 | 4.8e-42 |
| Pix2Pix | CLIP-I | 0.9281 | 0.9025 | 1.0e-11 |
| Pix2Pix | DINO-I | 0.7986 | 0.7560 | 5.2e-08 |
| CycleGAN | CLIP-I | 0.9281 | 0.9312 | 1.2e-02 |
| CycleGAN | DINO-I | 0.7986 | 0.8127 | 2.5e-06 |
| Palette | CLIP-I | 0.9281 | 0.7581 | 4.2e-42 |
| Palette | DINO-I | 0.7986 | 0.3983 | 4.2e-42 |

All comparisons two-sided Wilcoxon signed-rank test, n=246 paired test scenes.
Aggregate means reproduce the paper's own Table 1 CLIP-I/DINO-I columns almost
exactly (ControlNet/Pix2Pix/CycleGAN/RiskSlider match to 4 decimal places; Palette
differs because the paper's original row used the incomplete n=138 subset -- see
the n=246 fix below).

**Honest finding, not fully favorable**: against ControlNet, Pix2Pix, and Palette,
RiskSlider's CLIP-I/DINO-I advantage is significant at p<1e-7, same pattern as
LPIPS/SSIM/PSNR. But against **CycleGAN specifically**, the paper's own Table 1
numbers already show RiskSlider *trailing* CycleGAN on both CLIP-I (0.9281 vs
0.9312) and DINO-I (0.7986 vs 0.8127) -- this per-image test confirms that gap is
a real, statistically significant reversal (p=0.012 and p=2.5e-06), not noise. This
is consistent with CycleGAN also being RiskSlider's closest competitor on FID
(178.72 vs 175.19): CycleGAN produces outputs that are marginally more
semantically/distributionally similar to the specific real target by these two
metrics, even though RiskSlider wins decisively on every structural-fidelity metric
(LPIPS/SSIM/PSNR, p<1e-28) and on CAS. **Recommended framing**: don't claim an
unqualified sweep on Table 1 -- CycleGAN is a genuine, significant exception on
CLIP-I/DINO-I specifically, and the paper should say so rather than only citing the
metrics where RiskSlider wins.
- **Palette n=246 fix (RESOLVED, no retraining or re-inference needed)**: the
  paper's *original* Table 1 Palette row was computed on only n=138
  (`logs/eval/palette_eval_20260421_211943.log` reproduces the paper's exact
  published numbers at n=138 -- LPIPS 0.5962, SSIM 0.3265, PSNR 16.4847, FID
  472.78, CAS 0.9891 -- confirming this is a pre-existing issue in the paper
  itself, not an artifact introduced by this repo or by our recompute). The
  root cause: Palette's inference for the test split was run/saved across
  *two separate result folders*
  (`outputs/baselines/palette/test_palette_socalfire_epoch40_testsplit_260421_185108/`,
  169 images, and `outputs/baselines/palette/infer_palette_socalfire_TEST_epoch40_260421_185108/`,
  77 images), which together are disjoint and their union is exactly the full
  246-image test set. Every downstream script (the paper's original eval, and
  our earlier recompute) only ever pointed at one of the two folders, silently
  evaluating an incomplete subset. Fix: merged both folders into
  `outputs/baselines/palette/test_socalfire_epoch40_merged_246/test/0`
  (verified 246 unique GT/Out pairs, zero overlap between the two source
  folders) and reran all downstream metrics -- **no retraining or new
  inference was required**, since the complete generations already existed on
  disk. Corrected full-246 Palette numbers vs. the original incomplete-n=138
  paper numbers:

  | metric | original (n=138) | corrected (n=246) | Δ |
  |---|---|---|---|
  | LPIPS | 0.5962 | 0.6016 | +0.0054 |
  | SSIM | 0.3265 | 0.3213 | -0.0052 |
  | PSNR | 16.4847 | 16.2690 | -0.2157 |
  | FID | 472.78 | 434.56 | -38.22 |
  | CLIP-I | 0.7619 | 0.7581 | -0.0038 |
  | DINO-I | 0.4061 | 0.3983 | -0.0078 |
  | CAS (clean clf) | 0.9882 | 0.9919 | +0.0037 |

  The CAS row also needed the n=246 fix: Phase 0's leaked-vs-clean recompute
  (`SUMMARY.md`) reported Palette's clean-classifier CAS as 0.9882, but that number
  was computed on the same incomplete n=138 folder as the other metrics (CAS's
  triplet matching draws on Palette's own result folder too, so it wasn't immune to
  the split-folder bug). Recomputing CAS on the merged n=246 folder with the clean
  classifier gives 0.9919 (leaked-classifier n=246: 0.9939, close to SUMMARY.md's
  n=138 leaked value of 0.9941 since CAS is less sample-size-sensitive at this
  accuracy level than LPIPS/SSIM/PSNR/FID). **Action**: replace Palette's CAS in
  both Table 1 and any leaked-vs-clean CAS table with the n=246 clean value (0.9919),
  not the n=138 value in SUMMARY.md.

  **CLIP-I/DINO-I nan issue (RESOLVED)**: the first attempt at this corrected-n=246
  recompute (2026-07-17) silently returned `nan` for CLIP-I/DINO-I -- not a bug in
  the metric code, but a transient inability of this machine's `huggingface_hub`
  client to fetch `openai/clip-vit-base-patch32` / `facebook/dino-vitb16` through the
  configured HF mirror at that time (the mirror's HEAD-request redirect handling was
  incompatible with the installed `huggingface_hub` version, and separately the
  `hf_xet` chunked-download backend hung indefinitely on the DINO checkpoint with no
  retry). Worked around by disabling `hf_xet`-path resolution and downloading both
  checkpoints directly via plain HTTPS with resume support, then placing them in the
  standard HF cache layout; both metrics now compute cleanly (`HF_HUB_OFFLINE=1`
  afterward confirms no further network dependency). No change to any other metric
  or to the qualitative conclusion -- Palette remains the worst method on every
  metric.

  LPIPS/SSIM/PSNR/CAS shift only marginally; FID shifts more substantially
  (-38.2, ~8% relative), plausibly because FID is more sensitive to sample
  size and the original n=138 was a biased/noisy estimate. Palette remains by
  far the worst method on every metric either way -- this fix does not change
  any qualitative conclusion, it just makes Table 1 genuinely apples-to-apples
  across all five methods at n=246. **Action for the paper**: replace Table 1's
  Palette row with the corrected numbers above (FID especially, since 472.78
  is the exact number currently in the submitted/drafted table).

## Effect sizes + Holm correction for the per-image Wilcoxon tests (done, 2026-07-24)

The per-image Wilcoxon tests above (LPIPS/SSIM/PSNR/CLIP-I/DINO-I x 4 baselines = 20
tests) reported only p-values, with no effect size and no multiple-comparison
correction. Added both: median paired difference, rank-biserial correlation r
(Kerby 2014: r = (W+ - W-)/(W+ + W-), the standard effect size for the Wilcoxon
signed-rank test), and Holm-Bonferroni step-down correction across the full 20-test
family. Script: `claudecode/code/classifier_fix/table1_effect_sizes.py`. Full data:
`claudecode/result/classifier_fix/table1_effect_sizes.csv`.

**Result: every one of the 20 tests remains significant after Holm correction** (the
weakest, CycleGAN vs. CLIP-I, p_holm=1.18e-2). Effect sizes are large (|r| > 0.98) for
LPIPS/SSIM/PSNR against every baseline including CycleGAN -- the structural-fidelity
advantage is not just statistically significant but a near-maximal, near-unanimous
per-image effect (r close to ±1 means almost every single paired comparison favors
RiskSlider in the same direction). CLIP-I/DINO-I effect sizes are more modest and vary
by baseline: large against ControlNet/Palette (r≈0.98-1.0), moderate against Pix2Pix
(r=0.40-0.50), and small-to-moderate *against* RiskSlider for CycleGAN specifically
(r=-0.185 for CLIP-I, r=-0.346 for DINO-I) -- consistent with, and now quantifying,
the already-documented CycleGAN exception on these two metrics.

## FID/KID bootstrap confidence intervals (done, 2026-07-24)

Addresses whether Table 1's FID margin between RiskSlider (175.19) and CycleGAN
(178.72, the closest competitor) is distinguishable from sampling noise at n=246.
Computed KID (Binkowski et al. 2018 unbiased polynomial-kernel MMD^2 estimator, using
the same Inception-v3 features already used for Table 1's FID) and bootstrap 95% CIs
for both FID and KID. Full 2048-dim FID/KID sqrtm computation is too slow to
bootstrap directly (thousands of 2048x2048 matrix-sqrt calls); CIs were computed on a
150-dim PCA projection fit jointly per comparison (n=246 samples means the true
covariance rank is <= 245 regardless of original dimensionality, so 150 components
retains most of the real signal) -- point estimates reported alongside are the exact,
full-dimensional values matching Table 1. Script:
`claudecode/code/classifier_fix/fid_kid_bootstrap.py`. Data:
`claudecode/result/classifier_fix/fid_kid_bootstrap.csv` (marginal CIs) and
`fid_kid_paired_diff_bootstrap.csv` (paired RiskSlider-vs-baseline difference CIs, the
more direct/higher-power test).

**Caught and fixed a real bug before trusting these numbers**: the first version of
the paired-difference bootstrap used two different, independently-fit PCA bases for
the shared "real" reference features across the two comparisons being differenced,
producing a mathematically inconsistent result (a CycleGAN comparison whose CI didn't
even bracket the sign of the point-estimate difference). Fixed by fitting one PCA
basis jointly across all three arrays (real, method A, method B) being compared in
each paired test. The numbers below are from the corrected version.

**Result: RiskSlider's FID/KID advantage over CycleGAN specifically is NOT
statistically significant.** Paired-difference 95% CIs (FID_ours - FID_baseline):

| Baseline | FID diff 95% CI | excludes 0? | KID diff 95% CI | excludes 0? |
|---|---|---|---|---|
| ControlNet | [-91.94, -64.17] | yes | [-0.899, -0.604] | yes |
| Pix2Pix | [-30.90, -6.12] | yes | [-0.267, -0.034] | yes |
| **CycleGAN** | **[-13.49, +12.88]** | **no** | **[-0.044, +0.141]** | **no** |
| Palette | [-273.77, -233.99] | yes | [-4.284, -3.744] | yes |

RiskSlider is significantly better than ControlNet, Pix2Pix, and Palette on both
FID and KID (CIs entirely below zero). **But against CycleGAN, the CI straddles
zero for both metrics** -- the 175.19 vs. 178.72 FID gap (and the corresponding KID
gap) that Table 1 reports as a point estimate is not distinguishable from sampling
noise at n=246. **Action for the paper**: do not claim RiskSlider "wins" on FID/KID
against CycleGAN specifically; the honest claim is a tie on this metric, consistent
with the already-established pattern that CycleGAN is RiskSlider's one genuine,
metric-specific close competitor (also seen in the CLIP-I/DINO-I Wilcoxon results
above). RiskSlider's advantage over CycleGAN is still real and decisive on
LPIPS/SSIM/PSNR (Holm-corrected p<1e-40, r>0.99, see effect-size section above) and on
population-level FID/KID against the other three baselines -- this is specifically an
"our FID edge over CycleGAN in particular is not proven" finding, not a general
weakening of Table 1's claims.
