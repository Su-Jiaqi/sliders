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
