# s=0 -> s=0.01 boundary continuity check — 2026-07-25

## What was checked

External reviewer flagged a real risk: hardcoding s=0 to a pixel-identical x_pre
passthrough could introduce a new discontinuity right at the s=0 boundary that the
paper's existing "Continuity Across the Branch Boundaries" appendix analysis
wouldn't have accounted for. This had not been checked before this round.

Generated a genuine s=0.01 sample through the actual model (not a shortcut): base
diffusion generation (`infer/batch_infer_xbd_paired_scales.py --scales 0.01`,
socalfire test split, n=246, same hyperparameters as production: rank=16,
alpha=16.0, steps=50, start_noise=400, guidance_scale=1.0) using the best surviving
reconstruction of the production LoRA checkpoint (the original path referenced by
`infer/run_socalfire_scales.sh` no longer exists on disk — used
`output-models/train/socalfire_slider_paired_alpha16.0_rank16_full/...safetensors`,
same config, contemporaneous timestamp, same caveat already noted in
`headline_vs_seed_stability_gap_20260725.md`), then refined through the same
production refiner (`unified_refiner_scale1stronger/best.pt`). Evaluated with the
standard script, clean classifier. Ran on `cuda:7`; `cuda:0` untouched.

## Result: s=0.01 metrics

| scale | LPIPS | SSIM | PSNR | CAS/PCR |
|---|---|---|---|---|
| 0.01 | 0.3058 | 0.5044 | 21.664 | 0.6179 |

## Per-metric jump size at each boundary (test split, socalfire)

| Boundary | ΔLPIPS | ΔSSIM | ΔPSNR | ΔCAS |
|---|---|---|---|---|
| **0 → 0.01 (new, the boundary in question)** | **-0.0017** | **+0.0398** | **+0.408** | **+0.1179** |
| 0.01 → 0.25 | -0.0245 | +0.0285 | +0.586 | +0.1199 |
| 0.25 → 0.3 | -0.0052 | +0.0057 | +0.118 | +0.0183 |
| 0.3 → 0.5 | -0.0222 | +0.0204 | +0.431 | +0.1524 |
| 0.5 → 0.7 | -0.0201 | +0.0158 | +0.352 | +0.0712 |
| 0.7 → 0.75 | -0.0042 | +0.0033 | +0.085 | +0.0040 |
| 0.75 → 1 | -0.0130 | +0.0174 | +0.570 | +0.0041 |

## Honest verdict: mixed, real but not alarming — one metric genuinely gets worse

- **LPIPS**: the new 0→0.01 jump (-0.0017) is the *smallest* magnitude jump anywhere
  in the curve. No issue.
- **PSNR**: the new jump (+0.408) is mid-pack, smaller than 3 of the other 6
  boundary jumps (0.01→0.25, 0.3→0.5, 0.75→1 are all larger). No issue.
- **CAS/PCR**: the new jump (+0.1179) is comparable to the adjacent 0.01→0.25 jump
  (+0.1199) and smaller than 0.3→0.5 (+0.1524). Within the range already present in
  the trajectory. No issue.
- **SSIM**: the new jump (+0.0398) is now the *single largest* boundary jump in the
  entire curve (previous largest was 0.01→0.25 at +0.0285). This is a real,
  legitimate finding — the reviewer's concern is validated specifically for SSIM,
  though the new jump (0.0398) is not dramatically larger than the previous largest
  (0.0285), roughly 40% bigger, not an order of magnitude.

## Why this happens (mechanistic explanation, not just the numbers)

Before the fix, the model's own s=0 output was itself a diffusion+refiner sample
close in signal characteristics to a genuinely-sampled s=0.01 (both went through the
same noise-then-denoise-then-refine pipeline, just with a different conditioning
value) — so the two were naturally similar. After the fix, s=0 is now a real,
unprocessed photograph, while s=0.01 is still a model sample carrying the
pipeline's own generation/refinement signature (slight blur, texture smoothing,
etc. typical of diffusion+refiner outputs) — introducing a small but real
"domain gap" between a real photo and a model sample that didn't exist when both
ends of that boundary were model samples. SSIM is the most locally-sensitive of
these metrics to exactly this kind of structural/texture mismatch, which is
consistent with it being the one metric where the new jump stands out.

## Recommendation

The existing appendix continuity claim likely needs a caveat added for this one
metric at this one boundary, rather than being fully rewritten — SSIM's 0→0.01 step
is now the trajectory's largest single jump (though still a modest ~0.04 absolute
change, not a cliff). LPIPS/PSNR/CAS all remain within the range already present
elsewhere in the curve and need no caveat. Recommend disclosing this specifically
(not silently), e.g.: "the s=0 boundary, now defined as an exact real-image
passthrough, produces a modestly larger discontinuity in SSIM specifically (Δ=0.040
vs. a next-largest step of 0.029) than adjacent transitions, reflecting the
domain gap between an unprocessed photograph and a model sample; LPIPS, PSNR, and
PCR are unaffected."

## Caveat on this check itself

Used the best surviving reconstruction of the production LoRA checkpoint (original
path is gone from disk — same limitation as the headline-vs-seed-stability
finding). If that substitution isn't byte-identical to the true production weights,
the exact jump magnitudes above could shift slightly, though the qualitative
pattern (LPIPS/PSNR/CAS fine, SSIM is the one metric that changes) is unlikely to
flip given how it's derived (real-photo-vs-model-sample domain gap, not a
checkpoint-specific artifact).
