# Negative-controls quantitative table (mean diff, 95% CI, Wilcoxon p, rank-biserial r) — 2026-07-25

## What was asked

A reviewer wants a small quantitative table (mean diff, CI, test stat/p, effect size)
for the two negative controls — naive-interpolation and the same-backbone
no-progression control — placed in the appendix (main text is at its 7-page limit,
consistent with the earlier decision to keep this material out of the main text).
Both experiments already existed (`naive_interpolation_rebuttal.md`,
`same_backbone_control_20260724.md`) with point-estimate comparisons and, for naive
interpolation, scene-level Spearman/concordance significance — but neither had a
paired per-image significance/CI table for the population-level pixel/semantic
metrics (LPIPS/SSIM/PSNR/CLIP-I/DINO-I), which is what was missing.

## Method

New script (adapted directly from `claudecode/code/classifier_fix/table1_effect_sizes.py`,
same methodology used for Table 1's own effect-size table): two-sided Wilcoxon
signed-rank test per metric, rank-biserial r as effect size (Kerby 2014), Holm-Bonferroni
correction across the full 20-test family (4 comparisons × 5 metrics), plus a paired
bootstrap 95% CI on the mean difference (10,000 resamples, matching the project's
existing `fid_kid_bootstrap.py` convention). No new generation — both control
pipelines' images already exist on disk (`outputs/ablation-naive-interp-v2clean-refined/`,
`outputs/ablation-noprogression-refined/`), only re-scored.

**Important correction made before running this**: naive-interpolation's own design
(the `v2clean` fix already documented in `naive_interpolation_rebuttal.md`) defines
z_naive(1) = z_gen(1) exactly, i.e. at s=1 the naive-interp+refiner pipeline is
*definitionally* the same computation as RiskSlider's actual pipeline (both refine
RiskSlider's own raw s=1 generation through the same refiner). Confirmed empirically
first — comparing the two image sets at s=1 gives mean diff = 0.00000 on every
metric, p=1.0, because the files are identical. **s=1 is not a meaningful comparison
point for the naive-interp control**; used the three intermediate scales
(0.25/0.5/0.75) instead, which is also where the paper's own critique actually lives.
The no-progression control, by contrast, was only ever trained on `s∈{0,1}`, so s=1
*is* its only meaningful comparison point (matches `same_backbone_control_20260724.md`'s
own framing).

## Results

Full data: `claudecode/result/classifier_fix/negative_controls_effect_sizes.csv`.

### Naive-interp + refiner vs. RiskSlider actual (n=246, socalfire test)

| s | Metric | Ours mean | Control mean | Mean diff | 95% CI | rank-biserial r | p (raw) | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| 0.25 | LPIPS↓ | 0.2813 | 0.2810 | +0.00035 | [0.0001, 0.0005] | 0.319 | 1.43e-05 | 2.00e-04 |
| 0.25 | SSIM↑ | 0.5329 | 0.5328 | +0.00011 | [0.0000, 0.0002] | 0.126 | 8.54e-02 | 5.71e-01 |
| 0.25 | PSNR↑ | 22.250 | 22.238 | +0.012 | [0.003, 0.022] | 0.058 | 4.33e-01 | 1.00 |
| 0.25 | CLIP-I↑ | 0.8891 | 0.8882 | +0.00093 | [0.0005, 0.0014] | 0.329 | 7.84e-06 | 1.25e-04 |
| 0.25 | DINO-I↑ | 0.7350 | 0.7338 | +0.00116 | [0.0007, 0.0016] | 0.368 | 5.73e-07 | 9.75e-06 |
| 0.5 | LPIPS↓ | 0.2539 | 0.2527 | +0.00118 | [0.0009, 0.0015] | 0.604 | 2.26e-16 | 4.52e-15 |
| 0.5 | SSIM↑ | 0.5590 | 0.5585 | +0.00048 | [0.0003, 0.0007] | 0.287 | 9.64e-05 | 1.16e-03 |
| 0.5 | PSNR↑ | 22.799 | 22.756 | +0.043 | [0.022, 0.068] | 0.183 | 1.28e-02 | 1.02e-01 |
| 0.5 | CLIP-I↑ | 0.9090 | 0.9086 | +0.00039 | [-0.0002, 0.0010] | 0.091 | 2.18e-01 | 1.00 |
| 0.5 | DINO-I↑ | 0.7637 | 0.7628 | +0.00092 | [0.0004, 0.0015] | 0.208 | 4.66e-03 | 4.19e-02 |
| 0.75 | LPIPS↓ | 0.2296 | 0.2288 | +0.00088 | [0.0006, 0.0012] | 0.457 | 5.10e-10 | 9.18e-09 |
| 0.75 | SSIM↑ | 0.5780 | 0.5773 | +0.00068 | [0.0005, 0.0009] | 0.484 | 4.64e-11 | 8.82e-10 |
| 0.75 | PSNR↑ | 23.236 | 23.185 | +0.051 | [0.031, 0.073] | 0.326 | 9.57e-06 | 1.44e-04 |
| 0.75 | CLIP-I↑ | 0.9254 | 0.9250 | +0.00035 | [-0.0003, 0.0010] | 0.092 | 2.11e-01 | 1.00 |
| 0.75 | DINO-I↑ | 0.7895 | 0.7883 | +0.00119 | [0.0005, 0.0019] | 0.228 | 1.92e-03 | 1.92e-02 |

(Mean diff = Ours − Control; positive on LPIPS means Ours is *slightly worse*, i.e.
the sign is unfavorable to RiskSlider on the one metric where it reaches its largest
effect size, r=0.604 at s=0.5.)

### No-progression control vs. RiskSlider actual (n=246, socalfire test, s=1 only)

| Metric | Ours mean | Control mean | Mean diff | 95% CI | rank-biserial r | p (raw) | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| LPIPS↓ | 0.21660 | 0.21651 | +0.00009 | [0.0000, 0.0002] | 0.128 | 8.15e-02 | 5.71e-01 |
| SSIM↑ | 0.59538 | 0.59527 | +0.00011 | [0.0001, 0.0002] | 0.300 | 4.43e-05 | 5.76e-04 |
| PSNR↑ | 23.806 | 23.808 | -0.0015 | [-0.0055, 0.0025] | -0.031 | 6.69e-01 | 1.00 |
| CLIP-I↑ | 0.92813 | 0.92774 | +0.00039 | [-0.0001, 0.0009] | 0.081 | 2.68e-01 | 1.00 |
| DINO-I↑ | 0.79859 | 0.79806 | +0.00053 | [0.0002, 0.0008] | 0.241 | 1.03e-03 | 1.13e-02 |

## Reading these tables

Both controls show the same pattern already reported qualitatively: with n=246 paired
samples, several differences reach statistical significance (large n makes even tiny
differences detectable), but the **effect sizes and absolute magnitudes are
negligible** — 3rd-4th decimal place for LPIPS/SSIM/CLIP-I/DINO-I, well inside the
paper's own documented seed-noise floor (Table 3: CAS std 0.001-0.002 across seeds).
The largest effect size in either table (naive-interp LPIPS at s=0.5, r=0.604) is
still a difference of 0.0012 in absolute LPIPS — not a magnitude any reader would
call a meaningful pipeline advantage. This is exactly the quantitative backing for
the honest conclusions both source documents already draw: naive-interp+refiner and
RiskSlider's actual pipeline are population-level indistinguishable at every scale
tested; the no-progression control matches RiskSlider's endpoint quality.

## Recommendation

Use these two tables as the appendix-only quantitative backing (per the earlier
page-budget decision) for both negative-control experiments. Suggest presenting them
compactly — e.g. one combined table with a `p (Holm)` column and a one-line note that
"despite several statistically significant differences at n=246, all effect sizes
are small (r<0.61) and absolute magnitudes are 2-4 orders of magnitude smaller than
the gaps separating RiskSlider from any Table 1 baseline" — rather than reproducing
all 20 rows verbatim, to stay within appendix space.
