# Calibrated-severity variant: final results (honest, not decisive)

See `calibrated_severity_final_evaluation.md` for the plan and rationale. This
document reports what was actually found, from
`claudecode/code/calibrated_severity/eval_scene_calibration_compare.py`,
`eval_quality_sanity_check.py`, and (the more trustworthy test, added after the
binary-classifier version below turned out to be ceiling-limited)
`claudecode/code/experiment_a/multiscale_correlation.py` — the SAME linear-probe
tool used for the paper's original scene-level calibration section (Phase 1),
re-run here against the calibrated model's outputs for a direct, apples-to-apples
comparison. All numbers on the 103 labeled socalfire test pairs (and the 22-pair
high-severity subset for the quality check).

## Update: re-run with the linear severity probe (supersedes the binary-classifier
test below for the correlation question — read this section first)

The binary pre/post classifier used in the first pass (section below) saturates
near P=1.0 regardless of real severity once an image looks post-disaster at all, so
it cannot really test the calibration claim. The paper's own Phase 1 scene-level
calibration section already solved this with a different tool: a frozen Ridge
regression probe fit on real training images' features (predicting continuous
`S_real`, not a binary post/pre label), applied to generated test images. Re-running
that exact tool (`--ckpt output-models/classifier/socalfire_cls_clean_split/best.pt`,
same clean classifier backbone) against both models gives a sanity-checked,
directly-comparable result: production reproduces almost exactly the number already
reported in `scene_level_calibration.md` (ρ=0.392 here vs. 0.388 previously), so the
tool and setup are trustworthy.

Spearman ρ between real `S_real` and probe-predicted severity `Ŝ`, refined
(f0/fu/f1) output, at each requested scale:

| requested s | production ρ (p) | calibrated ρ (p) |
|---|---|---|
| 0    | 0.202 (0.041) | 0.214 (0.030) |
| 0.25 | 0.193 (0.051) | 0.170 (0.085) |
| 0.3  | 0.187 (0.058) | 0.153 (0.122) |
| 0.5  | 0.247 (0.012) | 0.150 (0.131) |
| 0.7  | 0.254 (0.010) | 0.079 (0.426) |
| 0.75 | 0.252 (0.010) | 0.051 (0.610) |
| 1.0  | 0.392 (4.3e-05) | 0.300 (0.002) |

Mean predicted severity `Ŝ` (0-3 scale) at each requested scale:

| requested s | production mean Ŝ | calibrated mean Ŝ |
|---|---|---|
| 0    | 0.182 | 0.191 |
| 0.25 | 0.234 | 0.289 |
| 0.3  | 0.248 | 0.317 |
| 0.5  | 0.336 | 0.471 |
| 0.7  | 0.460 | 0.720 |
| 0.75 | 0.491 | 0.784 |
| 1.0  | 0.576 | 0.511 |

**This is a genuinely mixed result, and it cuts both ways:**

- **In the hoped direction**: at every intermediate scale (0.5/0.7/0.75), calibrated
  shows a clearly weaker correlation with real severity than production, becoming
  non-significant (p=0.13-0.61) where production stays significant (p≤0.012). This
  is exactly the pattern you'd expect if requested severity became somewhat less
  tied to "what actually happened to this scene" at intermediate points — a real,
  qualitatively different signature from production, not noise.
- **Not fully decoupled**: at s=1 specifically — the scale that matters most for the
  paper's "s must be given before generation" rebuttal — calibrated still shows a
  significant correlation with real severity (ρ=0.300, p=0.002), only somewhat
  weaker than production's (ρ=0.392). So s=1 has NOT become fully scene-independent;
  it is still measurably influenced by what actually happened to that scene.
- **New concern, not previously visible**: production's mean predicted severity
  rises smoothly and monotonically across the entire sweep (0.18→0.58). Calibrated's
  does NOT — it rises faster than production through s=0.75 (reaching 0.78, already
  higher than production ever reaches) and then **drops** to 0.51 at s=1. A model
  whose own top requested severity produces a lower mean predicted severity than an
  earlier point on its own scale is a real oddity worth flagging plainly, not
  smoothing over. Plausible causes (not confirmed): f1 was retrained on only 81
  high-severity pairs vs. production's 769, or extrapolating s>s_i for the majority
  of scenes (whose own real endpoint sits below 1) behaves unevenly right at the
  extreme end of the range. Quick qualitative check: visually compared s=0.75 vs.
  s=1 calibrated outputs for several near-zero-real-severity test scenes (e.g. IDs
  10, 138) — the two images look nearly identical to the eye, no obvious visual
  collapse or dramatic quality failure at s=1. This suggests the numeric dip is
  more likely a subtle sensitivity of the probe to fine-grained pixel statistics
  under out-of-distribution extrapolation than a visible model failure, but this
  is a light spot-check (a handful of images), not a systematic investigation —
  flag as an open question, not a resolved one.
- Unrefined (pre-refiner) generation-module output shows essentially flat, ~zero,
  non-significant correlation across ALL 7 scales for calibrated (ρ between -0.041
  and -0.014, mean Ŝ pinned around 0.385-0.394 regardless of requested s) — flatter
  than production's already-weak unrefined numbers. The refiner is doing all of the
  calibration work in both models, more so in the calibrated variant.

## Cross-category replication: flooding

Ran the identical pipeline (calibrated generation module retrained with the hybrid
data scheme, f1 retrained on the high-severity subset, new pseudo-targets, fu
retrained) on `midwest-flooding` (279 train / 80 test pairs; only 17 train pairs
have `S_real >= 2.0`, vs. socalfire's 81 — flooding's severity distribution is
skewed heavily toward "no damage," so this category stress-tests the method under a
much smaller high-severity training set for f1). Same probe tool, same clean
classifier convention (`output-models/refine-2/midwest-flooding/classifier_clean_split/best.pt`).

Sanity check: production reproduces the number already on record for flooding
(ρ=0.494 at s=1, matching the "flooding calibrates better" note in
`psi_generalization_and_experiment_c.md`/OVERVIEW.md).

Spearman ρ between real `S_real` and probe-predicted severity, refined output:

| requested s | production ρ (p) | calibrated ρ (p) |
|---|---|---|
| 0    | 0.015 (0.901) | -0.018 (0.879) |
| 0.25 | 0.113 (0.340) | 0.068 (0.570) |
| 0.3  | 0.135 (0.256) | 0.095 (0.425) |
| 0.5  | 0.281 (0.016) | 0.138 (0.245) |
| 0.7  | 0.320 (0.006) | 0.132 (0.265) |
| 0.75 | 0.327 (0.005) | 0.101 (0.394) |
| 1.0  | 0.494 (8.9e-06) | 0.267 (0.022) |

Mean predicted severity `Ŝ`:

| requested s | production mean Ŝ | calibrated mean Ŝ |
|---|---|---|
| 0    | 0.753 | 0.750 |
| 0.25 | 0.609 | 0.514 |
| 0.3  | 0.542 | 0.458 |
| 0.5  | 0.274 | 0.377 |
| 0.7  | 0.177 | 0.474 |
| 0.75 | 0.173 | 0.497 |
| 1.0  | 0.209 | 0.200 |

**This replicates the socalfire pattern on an independent disaster category, almost
exactly, on all three fronts:**

1. **Intermediate-scale decoupling, confirmed**: at s=0.5/0.7/0.75, production stays
   significant (p=0.016/0.006/0.005) while calibrated drops to non-significant
   (p=0.245/0.265/0.394) — the same qualitative shift seen on socalfire, on a
   completely different disaster type and a much smaller high-severity training set
   for f1 (17 pairs vs. 81). This is no longer a single-category coincidence.
2. **s=1 still not fully decoupled, confirmed**: calibrated's correlation at s=1
   (ρ=0.267, p=0.022) is weaker than production's (ρ=0.494) but remains
   statistically significant — exactly the same partial-success pattern as
   socalfire (0.300 vs 0.392 there).
3. **Non-monotonic top-of-range trajectory, confirmed**: calibrated's mean predicted
   severity rises through s=0.7/0.75 (0.474, 0.497) — actually exceeding production's
   values at those points — then drops sharply at s=1 (0.200). This is the same
   qualitative anomaly found on socalfire (peaks at 0.75, drops at 1.0), now seen
   twice, on both categories tested. This raises the anomaly from "maybe a one-off
   quirk" to "a real, reproducible weakness of the current training scheme" —
   directly motivating the v2 fix below (see
   `calibrated_severity_v2_improvement_plan.md`): the model is never given any
   direct training signal for `s > s_i`, so nothing constrains its behavior at the
   very top of the requested range, and this now shows up consistently across two
   categories.

Note production's own mean-severity trajectory is *not* monotonic on flooding either
(0.75→0.61→0.54→0.27→0.18→0.17→0.21, i.e. it also dips through the middle before a
small uptick at s=1) — unlike socalfire where production was smoothly monotonic.
Flooding scenes evidently behave differently under this probe than wildfire scenes
in general (plausibly a domain-shift effect on the probe itself, fit on socalfire's
clean classifier features), so some caution is warranted before treating "production
monotonic, calibrated not" as a universal contrast — the cleaner three-way read is
simply: **the intermediate-scale decoupling and the residual/anomalous s=1 behavior
both replicate across categories**, which is the operative finding motivating v2.

## v2 improvement (severity-probe extrapolation loss)

**Diagnosis motivating v2**: `sample_s_calibrated()` in the v1 training loop never
samples `s > s_i` for any given pair — for the majority of pairs (whose real
severity anchors `s_i < 1`), the model receives literally zero direct training
signal for the region between its own anchor and `s=1`. Everything it does there
is pure extrapolation from what it learned on other pairs. This was the leading
candidate explanation for both v1 anomalies above: the residual s=1 correlation
with real severity (not yet fully decoupled) and, more concerning, the
non-monotonic drop in mean predicted severity between s=0.75 and s=1.

**Fix implemented**: added a severity-probe-guided extrapolation loss
(`train/train_xbd_paired_slider_calibrated_v2.py`). During training, when
`s_i < 1`, sample an extra `s_extrap` in `(s_i, 1]`, decode the model's predicted
`x0` at that scale, score it with a frozen, differentiable copy of the same Ridge
severity probe used for evaluation, and penalize MSE against a target of
`s_extrap * 3`. This gives the model, for the first time, an explicit training
signal for the entire extrapolation region — the direct mechanism by which we
hoped to fix both the non-monotonicity and firm up what `s=1` means. Trained for
15,000 steps on socalfire (same hyperparameters as v1 otherwise), full downstream
pipeline (train/test inference, f1 retrain, pseudo-targets, fu retrain, refine)
re-run identically to v1's.

**Result: the fix did not achieve its goal, on either front measured.**

Spearman ρ between real `S_real` and probe-predicted severity, refined output,
socalfire test set (n=103, same labeled subset used throughout):

| requested s | production ρ (p) | v1-calibrated ρ (p) | v2 ρ (p) |
|---|---|---|---|
| 0    | 0.202 (0.041) | 0.214 (0.030) | 0.208 (0.035) |
| 0.25 | 0.193 (0.051) | 0.170 (0.085) | 0.184 (0.062) |
| 0.3  | 0.187 (0.058) | 0.153 (0.122) | 0.170 (0.086) |
| 0.5  | 0.247 (0.012) | 0.150 (0.131) | 0.170 (0.086) |
| 0.7  | 0.254 (0.010) | 0.079 (0.426) | 0.128 (0.196) |
| 0.75 | 0.252 (0.010) | 0.051 (0.610) | 0.124 (0.213) |
| 1.0  | 0.392 (4.3e-05) | 0.300 (0.002) | 0.364 (1.6e-04) |

Mean predicted severity Ŝ (0-3 scale) at each requested scale:

| requested s | production mean Ŝ | v1-calibrated mean Ŝ | v2 mean Ŝ |
|---|---|---|---|
| 0    | 0.182 | 0.191 | 0.192 |
| 0.25 | 0.234 | 0.289 | 0.309 |
| 0.3  | 0.248 | 0.317 | 0.340 |
| 0.5  | 0.336 | 0.471 | 0.518 |
| 0.7  | 0.460 | 0.720 | 0.770 |
| 0.75 | 0.491 | 0.784 | 0.821 |
| 1.0  | 0.576 | 0.511 | 0.508 |

(Full precision numbers are in `claudecode/result/classifier_fix/probe_v2_socalfire.csv`,
`probe_calibrated_socalfire.csv`, `probe_production_socalfire.csv`.)

**Non-monotonicity: not fixed, if anything slightly worse.** v1's mean Ŝ rose to
0.784 at s=0.75 then dropped to 0.511 at s=1 (a drop of 0.273). v2's mean Ŝ rises
even higher, to 0.821 at s=0.75, then drops to 0.508 at s=1 — a drop of 0.313,
larger in absolute terms than v1's. The severity-probe extrapolation loss gave the
model a training signal in the extrapolation region, but it did not translate into
a smooth, monotonic approach to s=1 in the refined pipeline output. The anomaly
this fix specifically targeted persists essentially unchanged.

**s=1 decoupling: moved in the wrong direction, though within noise.** v1's s=1
correlation with real severity was ρ=0.300 (95% CI [0.106, 0.458]) — weaker than
production's 0.392, i.e. partially decoupled. v2's s=1 correlation is ρ=0.364 (95%
CI [0.171, 0.517]) — higher than v1's, moving *closer* to production's fully-coupled
number rather than further from it. The two confidence intervals overlap
substantially, so this specific shift cannot be called statistically significant
on n=103 — but there is no evidence of improvement here either, and the point
estimate moved the opposite way from what the fix was intended to produce.

**Intermediate scales: the v1 finding survives, but is somewhat eroded.** At
s=0.5/0.7/0.75, v2's correlations (0.170/0.128/0.124) remain non-significant
(p=0.086-0.213), same qualitative pattern as v1's non-significant
0.150/0.079/0.051 — so the core "intermediate decoupling" result from v1 still
holds directionally in v2. But every v2 point estimate at these three scales is
higher than v1's, drifting slightly back toward production's significant
0.247/0.254/0.252. Not a reversal, but not an improvement either.

**Unrefined generation-module output**: stays flat and slightly negative across
all scales (ρ between -0.094 and -0.018), essentially identical in shape to v1's
flat unrefined numbers — confirms the extrapolation loss (applied during LoRA
generation-module training, before the refiners) did not change the qualitative
picture that the refiner does essentially all of the scale-calibration work in
this pipeline.

**What was NOT tested in this round**: no image-quality sanity check (LPIPS/SSIM
against the earlier quality-check protocol) was re-run for v2 specifically, so
whether the extrapolation loss introduced any generation-quality regression is an
open question, not verified either way. The other secondary improvements listed in
`calibrated_severity_v2_improvement_plan.md` (severity-weighted f1 sampling,
data augmentation, using the probe as a primary training/selection metric) were
not implemented in this round — only the severity-probe extrapolation loss itself
was built and tested.

**Bottom line**: the severity-probe-guided extrapolation loss, despite being a
reasonable, correctly-implemented fix for a real diagnosed gap (zero training
signal for `s > s_i`), did not fix the non-monotonic top-of-range trajectory and
did not improve (arguably slightly worsened, within noise) the s=1 decoupling
result. The most likely explanation is that decoding `x0` and scoring it with the
probe during training gives a comparatively weak, indirect gradient signal
relative to the main diffusion objective, and/or that a single scalar MSE target
per extrapolated sample is too coarse a supervisory signal to meaningfully shape
behavior at the very top of the range within 15,000 steps. This does not
invalidate the v1 finding (which stands on its own, replicated across two disaster
categories) — it means the specific fix attempted for v1's residual anomalies did
not work as hypothesized, and a different mechanism (e.g. more aggressive f1
retraining data, or a stronger/more direct signal on decoded pixels rather than a
downstream probe score) would be needed to actually resolve them.

## v3 improvement (pixel-space monotonicity loss)

**Diagnosis motivating v3**: v2's failure pointed to its gradient signal being
too weak and indirect — it had to flow through a frozen classifier backbone and
a linear probe on top of a decoded image, several steps removed from the pixels
the LoRA can actually influence directly. v3 replaces that with a direct,
classifier-free pixel-space constraint: decode the model's prediction at the
anchor scale `s_i` (real pixel target exists here) and at an extrapolated scale
`s_extrap > s_i` (no real target, same synthetic-overshoot construction as v1/v2),
then penalize any case where the extrapolated image is *less* visually different
from the pre-disaster image than the anchor is — a hinge loss directly targeting
the exact observed failure mode (higher requested severity producing an image
that looks less damaged).

**A second bug found and fixed during v3's smoke test**: the synthetic
overshoot target `z_pre + (s_extrap/s_i)·(z_post - z_pre)`, inherited unchanged
from v1/v2, has a slope of `1/s_i`. For the ~58% of pairs with near-zero real
severity (`s_i` near its `EPS_SI=0.001` floor), this slope explodes (up to
~1000x), producing numerically extreme "extrapolation targets" that are not
meaningful images at all — pure noise for the model to chase. This was
discovered because v3's monotonicity loss was trivially ~0 for low-`s_i` samples
in the initial smoke test (the extrapolated image was already absurdly far from
`pre` due to the blowup, not because the model had learned anything). Fixed by
flooring the slope's denominator at `EXTRAP_SLOPE_FLOOR=0.15` (only affects
low-`s_i` pairs; unaffected for the majority with moderate-to-high `s_i`). This
bug was present in v2 as well and likely undermined a majority of v2's training
signal too, though v2 was not re-run to isolate this effect in isolation.

Trained 15,000 steps on socalfire, identical downstream pipeline (f1 retrain,
pseudo-targets, fu retrain, refine) as v1/v2.

**Result: real, measurable (if partial) progress — the first actual improvement
in this series, though the anomaly is reduced, not eliminated.**

Spearman ρ between real `S_real` and probe-predicted severity, refined output,
socalfire test set (n=103):

| requested s | production ρ | v1-calibrated ρ | v2 ρ | v3 ρ |
|---|---|---|---|---|
| 0    | 0.202 | 0.214 | 0.208 | 0.208 |
| 0.25 | 0.193 | 0.170 | 0.184 | 0.175 |
| 0.3  | 0.187 | 0.153 | 0.170 | 0.185 |
| 0.5  | 0.247 | 0.150 | 0.170 | 0.153 |
| 0.7  | 0.254 | 0.079 | 0.128 | 0.100 |
| 0.75 | 0.252 | 0.051 | 0.124 | 0.083 |
| 1.0  | 0.392 | 0.300 | 0.364 | 0.310 |

(p-values, CIs, and n in `claudecode/result/classifier_fix/probe_v3_socalfire.csv`;
production/v1/v2 numbers cross-referenced from `probe_production_socalfire.csv`,
`probe_calibrated_socalfire.csv`, `probe_v2_socalfire.csv`.)

Mean predicted severity Ŝ (0-3 scale) at each requested scale:

| requested s | production mean Ŝ | v1-calibrated mean Ŝ | v2 mean Ŝ | v3 mean Ŝ |
|---|---|---|---|---|
| 0    | 0.182 | 0.191 | 0.192 | 0.190 |
| 0.25 | 0.234 | 0.289 | 0.309 | 0.304 |
| 0.3  | 0.248 | 0.317 | 0.340 | 0.331 |
| 0.5  | 0.336 | 0.471 | 0.518 | 0.489 |
| 0.7  | 0.460 | 0.720 | 0.770 | 0.719 |
| 0.75 | 0.491 | 0.784 | 0.821 | 0.768 |
| 1.0  | 0.576 | 0.511 | 0.508 | 0.525 |

**Non-monotonicity: reduced, not fixed.** The s=0.75→s=1 drop is still present
in v3 (0.768 → 0.525), but it is the smallest drop of the three variants tried:
v1 dropped 0.273 (0.784→0.511), v2 dropped 0.313 (0.821→0.508), v3 dropped
**0.243** (0.768→0.525). This is genuine, measurable progress in the direction
the fix was designed for — but the qualitative anomaly (the trajectory still
peaks before s=1 rather than rising monotonically to it) has not been
eliminated. Framing this as "solved" would overclaim; framing it as "no
different from v1/v2" would understate a real, if partial, effect.

**s=1 decoupling: v2's regression undone, back in line with v1.** v2 had moved
the s=1 correlation up to ρ=0.364 (closer to production's fully-coupled 0.392,
i.e. less decoupled — an unwanted direction). v3's s=1 correlation is ρ=0.310
(95% CI [0.119, 0.463]), nearly identical to v1's 0.300 (95% CI [0.106, 0.458])
and clearly below v2's 0.364. In other words: whatever v2 did that pushed s=1
back toward "not decoupled," v3 does not reproduce — it restores v1's original,
more-decoupled s=1 behavior almost exactly.

**Intermediate scales: also closer to v1 than v2, i.e. the erosion v2 caused is
mostly reversed.** At s=0.5/0.7/0.75, v3's correlations (0.153/0.100/0.083) sit
between v1's (0.150/0.079/0.051) and v2's (0.170/0.128/0.124) — closer to v1 in
every case, and all three remain non-significant (p=0.12-0.41), same as v1's
qualitative pattern. The "intermediate-scale decoupling" finding from v1 survives
intact in v3, without the partial erosion v2 introduced.

**Unrefined generation-module output**: still flat and non-significant across
all scales (ρ between -0.058 and -0.019), consistent with v1/v2 — the refiner
continues to do essentially all of the scale-calibration work.

**What this does and doesn't mean**: v3 is a genuine improvement over v2 on
every axis measured (smaller top-of-range drop, more-decoupled s=1, less erosion
of the intermediate-scale finding) — largely by both (a) giving the model a more
direct training signal for the extrapolation region, and (b) fixing the
extrapolation-slope numerical bug that likely corrupted a majority of v2's
training signal. It does **not** fully resolve the non-monotonicity anomaly
(there is still a dip, just a smaller one), and no additional image-quality
sanity check (LPIPS/SSIM) was re-run for v3 specifically, so a quality
regression relative to v1/production cannot be ruled out from this evaluation
alone. Given the direction and magnitude of improvement, a natural next step
(not attempted in this round) would be increasing `lambda_monotonic_extrap` or
extending training steps, to see whether the same mechanism, pushed harder,
closes the remaining gap — but that is future work, not something claimed here.

## Original headline test (binary classifier — superseded by the section above for
the correlation question, kept for the quality-relevant part)

Spearman correlation between real `S_real` and classifier `P(damaged)` at a fixed
requested `s`, production vs. calibrated:

| requested s | production ρ | calibrated ρ |
|---|---|---|
| 0.5  | 0.186 | 0.140 |
| 0.75 | 0.290 | 0.155 |
| 1.0  | 0.276 | 0.209 |

Calibrated shows a somewhat weaker correlation than production at every scale
tested (most notably at s=0.75, roughly half of production's), which is the
direction the redesign was intended to move things in. **But the effect is small,
n is modest (103 pairs total, as few as 17 per severity bucket), and neither
correlation was tested for statistical significance** — treat this as a mild,
suggestive signal, not a demonstrated fix.

## Why s=1 specifically is a weak place to look for this effect

Mean `P(damaged)` by real-severity bucket at s=1:

| bucket | n | mean P(damaged) production | mean P(damaged) calibrated |
|---|---|---|---|
| 0 none | 49 | 0.934 | 0.947 |
| 1 minor | 20 | 0.966 | 0.940 |
| 2 major | 17 | 0.994 | 0.990 |
| 3 destroyed | 17 | 0.963 | 0.957 |

Both models are already near-ceiling (0.93-0.99) at s=1 across every severity
bucket, for both models. This is a **classifier limitation, not a model-generation
result**: the classifier used here is a binary pre/post *domain* classifier (does
this image look like a post-disaster photo at all — burnt vegetation, smoke haze,
color shift), not a severity regressor. It saturates near 1.0 once an image has any
post-disaster visual character, regardless of how severely damaged the buildings
actually are. This ceiling effect means s=1 is close to the worst place to look for
evidence that calibration changed anything — there just isn't headroom left in the
metric. The s=0.75 result (where neither model is fully saturated yet) is the more
informative data point, and it's also the one showing the largest gap between
production and calibrated correlations.

**What this does NOT establish**: that the calibrated model reliably produces
"more damaged" images for low-severity real scenes than production does. The
per-bucket mean P(damaged) numbers above do not show calibrated pulling the "0 none"
bucket meaningfully higher than production at s=1 (0.947 vs 0.934 — a difference
that's small relative to typical scene-to-scene noise in this classifier). A
severity-sensitive classifier (e.g. a 4-way severity classifier, or a
building-damage-detection model) would be needed to test the core claim properly;
the binary domain classifier used throughout this paper's other CAS tables is not
built for it. This is a genuine limitation of the evaluation, not just of the model.

## Quality sanity check: no regression where ground truth exists

Restricted to the 22 test pairs with `S_real >= 2.0` (where s=1's target is
approximately each pair's own real post image, for both models), refined-output
quality at s=1 vs. the real post image:

| model | n | LPIPS ↓ | SSIM ↑ | PSNR ↑ |
|---|---|---|---|---|
| production | 22 | 0.2104 | 0.5842 | 23.92 |
| calibrated | 22 | 0.2169 | 0.5791 | 23.78 |

Essentially unchanged (LPIPS +0.0065, SSIM -0.0051, PSNR -0.14dB), despite f1/fu
being retrained on a much smaller subset (81 high-severity train pairs vs.
production's full 769). This is the one clearly positive, low-ambiguity result:
the calibrated redesign does not cost reconstruction quality where a genuine target
exists.

## Overall assessment

- **Training pipeline**: fully executed end-to-end (generation module LoRA, f1 on
  the 81-pair high-severity subset, new pseudo-targets, fu retrained), no
  instabilities, all losses converged normally.
- **Quality**: no regression (positive, solid finding).
- **The core calibration claim** (does s=1 mean universal max-severity rather than
  "this pair's own fate"): genuinely mixed, using the paper's own linear-probe
  methodology (the trustworthy test, not the ceiling-limited binary classifier).
  Intermediate scales (0.5/0.7/0.75) show real, qualitatively different behavior —
  correlation with real severity drops to non-significant where production stays
  significant. But at s=1 specifically, the scale the rebuttal is actually about,
  correlation with real severity is weaker than production but still significant
  (ρ=0.300, p=0.002) — not decoupled. And the calibrated model's mean predicted
  severity is non-monotonic across its own scale (peaks at s=0.75, drops at s=1),
  which is a new, unexplained oddity that a reviewer could reasonably seize on if
  presented without caveat.

## Recommendation for how to use this in the paper

Given the honesty standard applied to every other finding in this revision
(smoothness ablation, naive-interpolation confound), this should NOT be presented
as a validated improvement, and should NOT lean on the binary-classifier numbers
now that the linear-probe re-run is available (it is strictly more informative and
already the paper's own established tool). Two defensible options:
1. **Present as a supplementary, exploratory variant** (matching what the user
   originally proposed) — describe the redesign, report the quality-preservation
   result plainly, report the intermediate-scale decoupling as a genuine partial
   effect, and explicitly flag both (a) that s=1 itself is not fully decoupled from
   real severity, and (b) the non-monotonic mean-severity trajectory as an open
   question. This is an honest, defensible "we tried this, here's what actually
   happened, here's what's still unresolved" narrative — it preempts the reviewer's
   objection by showing genuine engagement without overclaiming a fix.
2. **Future work**: mention the redesign and hybrid data scheme briefly in a
   limitations/future-work paragraph, without running it, if page budget is tight.
   Given the pipeline is already fully run, the quality check is good news, and the
   intermediate-scale result is a real (if partial) finding, option 1 is more
   convincing than (2) — but only if the non-monotonicity and s=1-still-correlated
   caveats are stated alongside it, not omitted.

Do not present option 3 (claim this fixes the pair-relative-interpolation
objection outright) — the evidence here does not support that framing, and the
s=1 correlation + non-monotonic trajectory would be exactly the kind of detail a
careful reviewer checks for and penalizes if it's glossed over.
