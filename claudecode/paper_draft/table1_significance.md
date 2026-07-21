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
