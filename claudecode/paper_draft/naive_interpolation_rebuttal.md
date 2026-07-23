# Rebuttal: "you don't need s-conditioned generation, just interpolate between endpoints"

## The critique this answers

A sharp reviewer could argue: RiskSlider's severity-conditioned latent generator is
unnecessary complexity. Just generate the post-disaster endpoint once (unconditional
on s), then linearly interpolate between the pre- and post-disaster latents for any
intermediate s -- exactly the quantity already used internally as the training-time
pseudo-target y_u(s)=D(z_s), Eq. 5. If this trivial baseline performs comparably to
the full s-conditioned pipeline, the generator's severity conditioning is redundant.

## Experimental design

Three stages of comparison, all on the primary SoCal wildfire test split (n=246,
n=103 with reliable S_real for the scene-level metrics):

1. **Naive interpolation, generation stage only**: VAE-encode real x_pre and
   RiskSlider's own generated s=1 endpoint (unrefined) x_gen(1); for each s, decode
   z_naive(s) = z_pre + s*(z_gen(1) - z_pre). No conditional UNet denoising at all --
   this is the "fair," actually-deployable version of the reviewer's proposed
   shortcut (does not peek at the real post-disaster image, which is unavailable at
   inference time).
2. **Naive interpolation, refined**: route the same z_naive(s) outputs through
   RiskSlider's own, unmodified production refiner -- f0 for s=0
   (`output-models/refine-1/socalfire/socalfire_scale0_refiner_20260421_192435`),
   f1 for s=1 (`output-models/refine-2/socalfire/scale1_teacher_stronger`), fu for
   intermediate s (`output-models/refine-2/socalfire/unified_refiner_scale1stronger`)
   -- identical branch routing to how RiskSlider's own generation output is refined.
3. Compare both against RiskSlider's actual, fully s-conditioned generation +
   refinement pipeline (existing data from Section "Scene-level Severity
   Calibration" and Tables 8/9).

Scripts: `claudecode/code/classifier_fix/naive_interp_baseline.py` (stage 1),
manual invocation of `refine/scale0_only_refiner.py`, `refine/unified_scale_refiner.py`,
`refine/scale1_only_refiner_stronger.py` in `refine` mode (stage 2).

## Results

### Stage 1 (generation only, before any refinement)

Scene-level correlation with real S_real, s=1: naive interpolation rho=-0.027
(p=0.79) vs. RiskSlider's own raw s-conditioned generation module rho=-0.014
(p=0.89, existing data). **Statistically indistinguishable -- both non-significant.**
At the raw-generation stage, s-conditioning shows no measurable advantage over
naive interpolation. (Full data: `claudecode/result/classifier_fix/naive_interp_eval/experiment_a_naive_interp.csv`
and `.../multiscale_cas.csv`; note the CAS-accuracy numbers there are confounded
by an extra VAE encode-decode round-trip that inflates naive interpolation's
apparent CAS uniformly across all scales including s=0 -- scene-level Spearman
correlation is the metric to trust, not CAS, for this specific comparison.)

### Stage 2 (through the identical, unmodified production refiner)

| metric | scale | naive-interp + refiner | RiskSlider actual (generation + refiner) |
|---|---|---|---|
| Spearman rho vs S_real | 0.5 | 0.021 (p=0.83) | 0.015 (p=0.88) |
| | 0.7 | 0.096 (p=0.33) | 0.091 (p=0.36) |
| | 0.75 | 0.120 (p=0.23) | 0.116 (p=0.24) |
| | **1.0** | **0.272 (p=0.0055)** | **0.388 (p=5.1e-5)** |
| concordance rate | 0.5 | 44.3% (p~1) | 43.8% (p~1) |
| | 0.7 | 48.6% (p=0.92) | 48.6% (p=0.92) |
| | 0.75 | 49.8% (p=0.59) | 49.7% (p=0.62) |
| **| 1.0** | **59.9% (p=7.9e-23)** | **65.2% (p=3.5e-52)** |

Full data: `claudecode/result/classifier_fix/naive_interp_eval/experiment_a_refined.csv`,
`.../experiment_b_refined/pairwise_concordance.csv`.

**CAS accuracy trend, for reference** (population-level, not scene-level): naive-interp
+ refiner reaches 0.610/0.734/0.764/0.919/0.978/0.986/0.994 across the 7 scales vs.
RiskSlider's actual 0.636/0.738/0.756/0.909/0.980/0.984/0.988 (Table 8/9 data) --
**nearly identical**. CAS accuracy alone cannot distinguish these two very different
pipelines; only the scene-level analysis reveals a difference.

## Consolidated three-way comparison table (new, 2026-07-24, closes the C2-style gap)

Filled in the missing LPIPS/DINO-I cells (previously only rho/concordance/CAS existed)
by directly re-evaluating the already-generated naive-interpolation images
(`outputs/ablation-naive-interp/gen_endpoints`, `outputs/ablation-naive-interp-refined/gen_endpoints`,
both already on disk from the original experiment, no new generation needed) with
`eval/socalfire_infered_eval_metrics.py`, using the same confirmed leak-free classifier
(`socalfire_cls_clean_split`) as the Table 1 CAS fix above, so all three CAS columns
here are on a consistent footing. RiskSlider-actual's numbers use the same clean
classifier (LPIPS/DINO-I are classifier-independent, unaffected either way).

| s | Method | CAS↑ | LPIPS↓ | DINO-I↑ | scene-level ρ (p) | concordance (p) |
|---|---|---|---|---|---|---|
| 0.25 | RiskSlider (actual) | 0.7378 | 0.2813 | 0.7350 | -0.055 (0.58) | 0.401 (~1) |
| 0.25 | Naive interp (unrefined) | 0.6728 | 0.3135 | 0.6835 | -- | -- |
| 0.25 | Naive interp + refiner | 0.7337 | 0.2810 | 0.7338 | -0.059 (0.55) | -- |
| 0.5 | RiskSlider (actual) | 0.9085 | 0.2539 | 0.7637 | +0.015 (0.88) | 0.438 (~1) |
| 0.5 | Naive interp (unrefined) | 0.7033 | 0.3177 | 0.6770 | -0.079 (0.43) | -- |
| 0.5 | Naive interp + refiner | 0.9207 | 0.2527 | 0.7628 | +0.021 (0.83) | 0.435 (~1) |
| 0.75 | RiskSlider (actual) | 0.9838 | 0.2296 | 0.7895 | +0.116 (0.24) | 0.497 (0.62) |
| 0.75 | Naive interp (unrefined) | 0.7378 | 0.3208 | 0.6703 | -- | -- |
| 0.75 | Naive interp + refiner | 0.9858 | 0.2287 | 0.7883 | +0.120 (0.23) | 0.498 (0.59) |

("--" = not computed for that variant/scale in the original experiment design; scene-
level ρ/concordance for the unrefined naive baseline were only run at s=1 in the
original design, see Stage 1 above.)

**Adjacent-scale smoothness** (Δ CAS between consecutive tested scales, all three
variants monotonic/non-decreasing across 0.25→0.5→0.75, no backward steps):

| Method | ΔCAS(0.25→0.5) | ΔCAS(0.5→0.75) |
|---|---|---|
| RiskSlider (actual) | +0.1707 | +0.0753 |
| Naive interp (unrefined) | +0.0305 | +0.0345 |
| Naive interp + refiner | +0.1870 | +0.0651 |

**Reading this table**: LPIPS and DINO-I are essentially tied between RiskSlider's
actual pipeline and naive-interp+refiner at every scale checked (differences in the
2nd-3rd decimal) -- confirming, with two more metrics, the already-reported finding
that population-level/pixel-level metrics cannot distinguish the two pipelines. The
refiner is clearly what drives the steep CAS jump between 0.25 and 0.5 for both
refined variants (RiskSlider +0.17, naive+refiner +0.19) versus the much gentler,
still-positive climb for the unrefined naive baseline (+0.03) -- consistent with the
paper's own framing that the refinement/pseudo-target-supervision stage, not
s-conditioning per se, is the primary driver of the smooth population-level
trajectory (see `lambda_smooth_ablation.md`'s full-pipeline check for the same
conclusion from a different angle). The one place s-conditioning earns its keep is
scene-level correlation at s=1 (already reported above: RiskSlider ρ=0.388 vs.
naive+refiner ρ=0.272, both significant but RiskSlider's stronger) -- everywhere else
in this table, s-conditioning is not distinguishable from naive interpolation.

## Honest conclusion

**The rebuttal is real but modest, not overwhelming.** At intermediate scales
(0.5/0.7/0.75), refined naive interpolation and RiskSlider's actual pipeline are
statistically indistinguishable -- neither shows significant scene-level calibration
there (a shared, already-acknowledged weakness, see "Scene-level Severity
Calibration"). At s=1, RiskSlider's actual s-conditioned pipeline is meaningfully
better: correlation is ~43% relatively higher (0.388 vs. 0.272) and reaches a much
stronger significance level (p=5.1e-5 vs. p=0.0055); concordance is 5.3 points higher
(65.2% vs. 59.9%). Both reach significance at s=1, so this is not a case of "one
works and one doesn't" -- it is a case of "both carry some real signal at the trained
endpoint, and RiskSlider's carries more."

**Recommended framing for the paper**: do not claim s-conditioned generation
dramatically outperforms a naive interpolation baseline across the board -- it does
not, especially at intermediate severities, and the population-level CAS metric is
not sensitive enough to show any difference at all. Instead, claim precisely what the
data supports: conditioning the generator on s provides additional scene-specific
severity information beyond what post-hoc endpoint interpolation and refinement can
recover, measurably so at the trained endpoint (Appendix Table [N]), while
acknowledging this advantage does not yet extend with statistical significance to
intermediate severities -- consistent with, and reinforcing, the paper's own
observation elsewhere that intermediate-scale calibration remains an open challenge
rather than a solved problem. This is a more defensible, reviewer-resistant claim
than an unqualified "s-conditioning is clearly necessary," and it directly
preempts the "just interpolate" critique with data rather than assertion.

## Caveats
- Single test category (SoCal wildfire, the primary benchmark). Not verified on
  other disaster categories.
- The "naive interpolation" here uses RiskSlider's OWN generated endpoint as the
  interpolation target, which itself required training a full conditional
  generator to produce (it is not the s=1 generation of a completely different,
  cheaper baseline). A stricter version of the reviewer's critique -- "don't even
  train a severity-conditioned generator, just fine-tune an ordinary pre-to-post
  translation model once and interpolate" -- is not directly tested here; the
  existing Concept Sliders comparison (Table 6) is the closest available evidence
  against unconditioned/differently-conditioned generation baselines, though Table
  6 cannot currently be reproduced on this machine (no Concept Slider generations
  available).
- Single run, no seed variance estimate for this specific ablation.
