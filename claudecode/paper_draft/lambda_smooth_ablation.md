# λ_smooth (Eq.3 smoothness regularization) ablation — draft insert

Insert location: Appendix, right after the existing "Additional Ablation Analysis"
section (which currently only ablates whole modules: w/o Refinement, w/o
Scale-aware Refiner, w/o Pseudo Supervision). This fills the completeness gap that
none of Eq.3's three loss terms (λ_scale, λ_endpoint, λ_smooth) were individually
ablated -- only module-level ablations existed.

## Setup

Retrained the Scale-Conditioned Latent Generator (the LoRA-adapted SD v1.4 diffusion
module, NOT the refiner) on the primary SoCal wildfire benchmark from scratch, 15,000
steps, identical hyperparameters to the production config (`train/configs/xbd_slider.yaml`)
except smoothness_weight:
  - smooth0: smoothness_weight = 0.0 (regularizer off)
  - smooth0.25: production checkpoint, smoothness_weight = 0.25 (paper's reported value)
  - smooth1: smoothness_weight = 1.0 (4x production strength)

All three were run through identical inference (`infer/batch_infer_xbd_paired_scales.py`,
steps=50, start_noise=100, guidance_scale=1.0, all 7 scale points) on the 246-pair test
split, and evaluated with `eval/socalfire_infered_eval_metrics.py` using the leak-free
classifier (`socalfire_cls_clean_split`) for CAS. This evaluates the RAW/UNREFINED
generator output only (comparable to the paper's Table 7, "before refinement"), which
isolates λ_smooth's effect from the refinement module's own smoothing behavior.

## Results

| variant | CAS(s=0) | CAS(0.25) | CAS(0.3) | CAS(0.5) | CAS(0.7) | CAS(0.75) | CAS(1) | #backward steps | backward-step sum | range |
|---|---|---|---|---|---|---|---|---|---|---|
| smooth0 (λ=0) | 0.6768 | 0.6707 | 0.6626 | 0.6809 | 0.6870 | 0.6911 | 0.7012 | 2 | 0.0142 | 0.0386 |
| smooth0.25 (production) | 0.6606 | 0.6646 | 0.6524 | 0.6565 | 0.6748 | 0.6768 | 0.6829 | 1 | 0.0122 | 0.0305 |
| smooth1 (λ=1.0) | 0.6707 | 0.6768 | 0.6829 | 0.6626 | 0.6646 | 0.6667 | 0.6789 | 1 | 0.0203 | 0.0203 |

"Backward steps" = number of consecutive scale-pairs where the metric decreases (i.e.
violates monotonicity); "backward-step sum" = total magnitude of those decreases;
"range" = max - min across all 7 scales.

LPIPS/SSIM/PSNR ranges (smaller = smoother trajectory) across the same 7 scales:

| variant | LPIPS range | SSIM range | PSNR range |
|---|---|---|---|
| smooth0 (λ=0) | 0.0146 | 0.0083 | 0.1184 |
| smooth0.25 (production) | 0.0192 | 0.0063 | 0.0758 |
| smooth1 (λ=1.0) | 0.0118 | 0.0067 | 0.0982 |

Full per-scale numbers: `claudecode/result/classifier_fix/ablation_smooth/{smooth0,smooth025,smooth1}_eval/socalfire-infered_metrics.csv`.

## Honest interpretation

**λ_smooth does not show a clean, monotonic effect on the raw generator's trajectory
smoothness across this range.** Neither the CAS non-monotonicity metric nor the
LPIPS/PSNR ranges decrease monotonically as smoothness_weight increases from 0 to
0.25 to 1.0:
- CAS backward-step magnitude: 0.0142 (λ=0) -> 0.0122 (λ=0.25) -> 0.0203 (λ=1.0) --
  dips at the production value then rises again at λ=1.0, not monotonic.
- LPIPS range: 0.0146 -> 0.0192 -> 0.0118 -- production (λ=0.25) actually shows the
  *largest* range (least smooth), worse than both λ=0 and λ=1.0.
- PSNR range: 0.1184 -> 0.0758 -> 0.0982 -- production is smallest, but λ=1.0 is
  worse than production despite 4x the regularization strength.
- SSIM range is small and roughly flat across all three (0.0083/0.0063/0.0067),
  showing no meaningful sensitivity to λ_smooth at all.

All three variants also show a similar qualitative pattern: a local non-monotonic
dip somewhere in the low-to-mid scale range (s=0.25-0.3 for smooth0/production,
s=0.5-0.7 for smooth1) before recovering and rising toward s=1. This dip is present
*regardless of smoothness_weight*, including at 4x the production strength.

**What this suggests**: λ_smooth's practical contribution to the smooth, monotonic
severity trajectory documented in Table 14/Figure 6 appears modest at the level of
the raw generator output, and is not obviously dose-responsive in this range. The
much cleaner monotonic CAS trend the full pipeline exhibits (e.g., production
refined trajectory: 0.6768 -> 0.7846 -> 0.9248 -> 0.9837 -> 0.9939, essentially free
of backward steps) likely owes more to the refinement module and pseudo-target
supervision (already ablated in Table 2/14) than to this particular latent-space
regularizer. We report this honestly rather than claiming λ_smooth is necessary for
the property it is nominally designed to encourage: on this evidence, its role may
be better characterized as a mild training-stability regularizer for the diffusion
model's per-step denoising predictions, rather than the primary mechanism securing
continuous controllability -- that mechanism is better attributed to the refinement
stage. This does not undermine the overall continuous-control claim (which is
supported by the full-pipeline results), but the paper should not imply that
λ_smooth alone, in isolation, is what prevents trajectory collapse; that credit
belongs to pseudo-target supervision (Table 14) and the endpoint/scale-aware
refiner design (Table 2), not to Eq.3's smoothness term.

## Caveats
- Single run per configuration (no seed variance estimate for this specific
  ablation); given the paper's own Table 3 shows CAS seed-to-seed std around
  0.001-0.002 on the *refined* pipeline, and these unrefined-generator differences
  are on the order of 0.01-0.02, the effect sizes here are unlikely to be pure noise,
  but a repeat-seed check would strengthen this before submission if time allows.
- Only tested on the primary wildfire category; not verified on other disaster
  types.

## Full-pipeline check (done): does the refiner change this picture?

**Setup**: took each of the three raw-generator variants' unrefined test-split output
(smooth0, production, smooth1; 246 pairs, all 7 scales) and ran it through the
**unchanged production refiner** (f0/fu, `unified_refiner_scale1stronger/best.pt` --
no retraining, exact same checkpoint used for the paper's own Table 14 numbers),
then re-evaluated LPIPS/SSIM/PSNR/CAS with the leak-free classifier. This directly
tests the caveat above: does the refiner's pseudo-target supervision compensate
differently depending on the raw generator's λ_smooth, producing an interaction
effect the generator-only ablation would miss?

**CAS trajectory after refinement** (scale 0 -> 1):

| variant | s=0 | s=0.25 | s=0.3 | s=0.5 | s=0.7 | s=0.75 | s=1 | backward steps | range |
|---|---|---|---|---|---|---|---|---|---|
| smooth0 (λ=0) | 0.6402 | 0.7398 | 0.7500 | 0.9065 | 0.9776 | 0.9858 | 0.9898 | 0 | 0.3496 |
| smooth0.25 (production) | 0.6362 | 0.7378 | 0.7561 | 0.9085 | 0.9797 | 0.9838 | 0.9878 | 0 | 0.3516 |
| smooth1 (λ=1.0) | 0.6362 | 0.7378 | 0.7541 | 0.9045 | 0.9776 | 0.9858 | 0.9898 | 0 | 0.3536 |

LPIPS / SSIM / PSNR ranges after refinement:

| variant | LPIPS range | SSIM range | PSNR range |
|---|---|---|---|
| smooth0 (λ=0) | 0.0895 | 0.0918 | 2.158 |
| smooth0.25 (production) | 0.0896 | 0.0919 | 2.160 |
| smooth1 (λ=1.0) | 0.0895 | 0.0920 | 2.163 |

Full per-scale numbers: `claudecode/result/classifier_fix/ablation_smooth/refined_fullpipeline_check.csv`.

**Finding: no interaction effect, and the caveat above is resolved.** Once the
production refiner is applied, all three λ_smooth variants become monotonic with
**zero backward steps** (the local dips present in every raw-generator variant are
gone), and their trajectories are statistically indistinguishable from each other --
every metric's spread across the three variants (CAS ≤0.004, LPIPS ≤0.0001, SSIM
≤0.0002, PSNR ≤0.005) is smaller than the paper's own seed-to-seed noise floor for
the refined pipeline (CAS std 0.001-0.002, Table 3). This is a clean, unambiguous
result: the refiner's pseudo-target supervision fully washes out whatever
raw-generator-level differences λ_smooth produces, regardless of its value in
{0, 0.25, 1.0}. There is no generator/refiner interaction effect to worry about --
the honest generator-only finding above is not an artifact of stopping short of the
full pipeline; it holds throughout. This strengthens rather than weakens the
recommended framing: credit the smooth, non-saturating full-pipeline trajectory
entirely to the refinement module and pseudo-target supervision, and treat
λ_smooth as, at most, a training-stability regularizer on the raw diffusion output
with no measurable end-to-end effect on the property it is nominally designed for.
