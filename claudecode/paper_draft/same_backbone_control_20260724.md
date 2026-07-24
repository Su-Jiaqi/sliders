# Same-backbone, no-progression-conditioning control — done, 2026-07-24, honest negative result

## The question this answers

The single most-recommended remaining experiment from the reviewer checklist: does
RiskSlider's advantage come from the progression-conditioning *method* (training the
LoRA on a continuous severity scale s, with the smoothness/scale-alignment losses in
Eq. 3), or simply from fine-tuning the same pretrained SD v1.4 backbone on this
pre-to-post disaster translation task at all? If a much simpler endpoint-only
fine-tune matches RiskSlider's endpoint quality, the progression-conditioning
methodology isn't what's earning the Table 1 numbers.

## Setup

Trained a new LoRA from scratch with **identical** hyperparameters to the production
socalfire config (`train/configs/xbd_slider_socalfire_noprogression.yaml`): same SD
v1.4 backbone, rank=16, alpha=16.0, `training_method=full`, 15,000 iterations,
lr=5e-5, same 769 train pairs, same loss weights (x0/smoothness/img-endpoint terms
all identical). The **only** change: `s_warmup_steps` set to cover the entire run
(15000, vs. production's 3000), which forces the scale-sampling function to draw
only `s in {0, 1}` for every single training step (confirmed in the training log --
`s~[0.00,0.00]` or `s~[1.00,1.00]` on every printed step, never an intermediate
value). The model architecturally still has the scale-conditioning input channel
(no code changes needed) -- it simply never receives a training signal at any
intermediate severity, which is the direct operationalization of "no progression
conditioning": an endpoint-only translation model trained on the same backbone with
the same recipe.

Generated the test-set s=1 endpoint with this checkpoint
(`infer/batch_infer_xbd_paired_scales.py`, identical inference hyperparameters:
steps=50, start_noise=100, guidance_scale=1.0), refined through the **unchanged
production refiner** (`unified_refiner_scale1stronger`, the same one used for every
other RiskSlider number in this project), and evaluated with the same Table 1
metrics using the confirmed leak-free classifier for CAS.

## Result: no measurable difference at the trained endpoint

| Metric | RiskSlider (progression-conditioned, actual) | No-progression control |
|---|---|---|
| LPIPS↓ | 0.2166 | 0.2165 |
| SSIM↑ | 0.5954 | 0.5953 |
| PSNR↑ | 23.806 | 23.808 |
| FID↓ | 175.19 | 175.27 |
| CLIP-I↑ | 0.9281 | 0.9277 |
| DINO-I↑ | 0.7986 | 0.7981 |
| CAS↑ (clean classifier) | 0.9878 | **0.9898** |
| Structural IoU (n=108) | 0.2892 | 0.2900 |

Every metric is essentially identical -- differences are 2-3 orders of magnitude
smaller than the gaps that separate RiskSlider from any baseline in Table 1, and
smaller than or comparable to the paper's own seed-noise floor (Table 3: CAS std
0.001-0.002). The no-progression control is not worse on any metric, and is
marginally *better* on CAS (0.9898 vs. 0.9878) and structural IoU (0.2900 vs.
0.2892) -- though these tiny differences are themselves within noise.

## Honest interpretation

**At the trained endpoint (s=1), progression-conditioned training provides no
measurable quality advantage over a much simpler endpoint-only fine-tune of the same
backbone.** RiskSlider's Table 1 endpoint numbers do not depend on the
progression-conditioning methodology (the continuous s-scale training, Eq. 1's
pseudo-target construction, or Eq. 3's smoothness/scale-alignment losses) -- they
depend on fine-tuning the pretrained SD v1.4 backbone on this specific pre-to-post
disaster translation task at all, which the simplest possible recipe already
achieves. This is consistent with, and reinforces, two other findings from this same
audit: the clean naive-interpolation rerun (`naive_interpolation_rebuttal.md`), which
found RiskSlider's actual pipeline is statistically indistinguishable from
naive-interpolation+refiner at s=1; and the SDEdit finding
(`structural_consistency_and_sdedit_20260724.md`), which showed *some* form of
training is clearly necessary (SDEdit without any fine-tuning is dramatically worse).
Put together: fine-tuning matters, but *how* you fine-tune (progression-conditioned
vs. endpoint-only) does not appear to matter for endpoint quality on this evidence.

**Important scope limitation, not covered by this experiment**: this control was
never trained on intermediate severities, so it cannot produce a meaningful
continuous severity slider at all -- evaluating it at s=0.5, for instance, would be
extrapolating the scale-conditioning channel to a value it never saw during training,
which is not a fair or meaningful test. This experiment says nothing about whether
progression-conditioning is necessary for the *continuous control* capability itself
(arguably the paper's actual core contribution, distinct from endpoint image
quality) -- only that it isn't necessary for matching Table 1's specific endpoint
numbers. **Recommended framing**: the paper should not claim progression-conditioned
training is what makes the *endpoint* generation good -- that credit likely belongs
to backbone fine-tuning in general. The paper's continuous-severity-control claim
(the actual "slider" contribution) is a separate claim, not tested or undermined by
this experiment, and should be argued on its own terms (e.g., the scene-level
calibration and monotonic-trajectory results elsewhere in this project) rather than
leaning on endpoint quality as evidence that progression-conditioning is doing
something special.

## Caveats
- Single run, no seed variance estimate for this specific control (though the
  differences observed are far smaller than what seed noise alone would need to be
  to explain a real effect, based on Table 3's documented noise floor).
- Only tested at the trained endpoint (s=1) on the primary wildfire benchmark; not
  verified on other disaster categories.
- Training script/config: `train/configs/xbd_slider_socalfire_noprogression.yaml`,
  `train/train_xbd_paired_slider.py` (unmodified). Checkpoint:
  `output-models/train/socalfire_slider_noprogression_paired_alpha16.0_rank16_full/`.
  Generation/refine/eval outputs under `outputs/ablation-noprogression*/socalfire/`.
