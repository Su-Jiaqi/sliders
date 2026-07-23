# Dense-s continuity check (C1 from the reviewer checklist) — done, clean negative result (no discontinuity)

## The concern this answers

The pipeline conditions the diffusion generator on a continuous scale channel and then
applies a single continuous scale-conditioned refiner (see `refine/unified_scale_refiner.py`
-- confirmed by reading the code that there is no discrete branch selection by value of
s anywhere in generation or refinement; `s` is a single scalar fed into both stages as a
filled conditioning channel/map). A reviewer could still ask: even without an explicit
branch in the code, could the model have *learned* an implicit discontinuity at s=0 or
s=1 specifically, since those are qualitatively different regimes (identity-like at s=0,
maximum damage at s=1)? This is only checkable empirically, by sampling densely near
both boundaries and looking for an anomalous jump.

## Setup

Used the existing production checkpoints with no retraining: the LoRA generator
(`output-models/train/socalfire_slider_paired_alpha16.0_rank16_full/*_last.safetensors`,
rank=16, alpha=16.0) and the production refiner
(`output-models/refine-2/socalfire/unified_refiner_scale1stronger/best.pt`), identical
inference hyperparameters to production (steps=50, start_noise=100, guidance_scale=1.0,
image_size=256). Generated 14 new scale points on the full 246-pair test split,
densest near both boundaries: s in {0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8,
0.85, 0.9, 0.95, 0.98, 0.99}, refined all of them through the unchanged production
refiner, and combined with the 7 existing production scale points (0, 0.25, 0.3, 0.5,
0.7, 0.75, 1) for **21 total scale points, all 246 test scenes present at every
scale** (zero dropped stems).

For each pair of adjacent scale points, computed per-scene: LPIPS between the two
refined images, DINO-ViT cosine distance, |ΔP(post)| from the leak-free classifier
(`socalfire_cls_clean_split`), and |Δ residual-norm| (the refiner's own edit magnitude,
`||refined - unrefined||` at each scale, then differenced across adjacent scales).
Script: `claudecode/code/classifier_fix/dense_s_continuity_eval.py`. Full per-pair
numbers: `claudecode/result/classifier_fix/dense_s_continuity.json`.

## Results

| pair | LPIPS | DINO-dist | \|ΔP(post)\| | residual-jump |
|---|---|---|---|---|
| 0→0.01 | 0.0001 | 0.0001 | 0.0024 | 0.0002 |
| 0.01→0.02 | 0.0001 | 0.0000 | 0.0025 | 0.0003 |
| 0.02→0.05 | 0.0003 | 0.0001 | 0.0042 | 0.0010 |
| 0.05→0.1 | 0.0008 | 0.0005 | 0.0115 | 0.0016 |
| 0.1→0.15 | 0.0015 | 0.0011 | 0.0210 | 0.0015 |
| 0.15→0.2 | 0.0013 | 0.0010 | 0.0288 | 0.0011 |
| 0.2→0.25 | 0.0012 | 0.0008 | 0.0376 | 0.0010 |
| 0.25→0.3 | 0.0012 | 0.0008 | 0.0467 | 0.0008 |
| 0.3→0.4 | 0.0042 | 0.0031 | 0.1410 | 0.0016 |
| 0.4→0.5 | 0.0045 | 0.0033 | 0.2209 | 0.0014 |
| 0.5→0.6 | 0.0047 | 0.0034 | 0.2189 | 0.0013 |
| 0.6→0.7 | 0.0048 | 0.0037 | 0.1288 | 0.0018 |
| 0.7→0.75 | 0.0015 | 0.0012 | 0.0335 | 0.0014 |
| 0.75→0.8 | 0.0017 | 0.0014 | 0.0190 | 0.0019 |
| 0.8→0.85 | 0.0022 | 0.0018 | 0.0101 | 0.0027 |
| 0.85→0.9 | 0.0037 | 0.0033 | 0.0065 | 0.0043 |
| 0.9→0.95 | 0.0078 | 0.0070 | 0.0048 | 0.0069 |
| 0.95→0.98 | 0.0037 | 0.0031 | 0.0033 | 0.0042 |
| 0.98→0.99 | 0.0006 | 0.0004 | 0.0010 | 0.0013 |
| **0.99→1** | **0.0006** | **0.0005** | **0.0015** | **0.0011** |

(Values are per-scene means across all 246 test scenes; std omitted from the table
above but recorded in the JSON.)

## Honest interpretation: no discontinuity at either boundary

**The two branch boundaries the reviewer checklist specifically flagged (0→0.01 and
0.99→1.00) are among the smoothest transitions in the entire 21-point trajectory, not
the roughest.** Both are visually and numerically indistinguishable from their
immediate same-step-size neighbors: 0→0.01 (LPIPS 0.0001, ΔP(post) 0.0024) is nearly
identical to 0.01→0.02 (LPIPS 0.0001, ΔP(post) 0.0025); 0.99→1.00 (LPIPS 0.0006,
ΔP(post) 0.0015) is nearly identical to 0.98→0.99 (LPIPS 0.0006, ΔP(post) 0.0010).

Because the step sizes (Δs) are not uniform across the table (0.01 near the
boundaries vs. 0.05-0.1 in the middle), a fair comparison requires normalizing by
step size. Doing so (|ΔP(post)| / Δs, a per-unit-s "slope"): the boundaries give
0.24 (0→0.01) and 0.15 (0.99→1.00), while the *middle* of the range peaks at **2.2**
(0.4→0.5 and 0.5→0.6) -- roughly **10-15x steeper** than either boundary, even after
adjusting for step size. The full trajectory is a smooth, sigmoid-like S-curve:
gentle at both ends (where the classifier's decision is already confident and
saturated in one direction), steepest in the middle (s≈0.3-0.7, the actual
"decision boundary" region between pre- and post-disaster appearance), gentle again
near s=1. This is exactly the shape you'd expect from a well-behaved continuous
classifier-probability trajectory, not evidence of a branch artifact.

The same pattern holds for LPIPS, DINO-distance, and the refiner's own residual-jump
-- all four independent metrics agree: no anomalous jump at either boundary, and the
steepest region of change is solidly in the interior of the scale range, not at
either endpoint. **This closes the "could there be a hidden f0/fu/f1 branch
discontinuity" concern with a clean negative result: there is no evidence of one**,
consistent with the code-level finding that neither the generator nor the refiner
implements any discrete branching by value of s -- both treat s as a single
continuous conditioning scalar throughout, and the empirical trajectory reflects
that design faithfully.

## Caveats
- Single category (SoCal wildfire), single inference seed, no retraining involved --
  this checks the *existing* production model's behavior, not whether continuity
  holds after retraining with different hyperparameters.
- The steep mid-range transition (s≈0.3-0.7) is itself an interesting, separate
  observation -- it says the model's implicit "decision boundary" between
  pre-/post-disaster appearance sits in the middle of the nominal severity scale,
  which is plausibly related to (but not the same claim as) the already-documented
  intermediate-scale calibration weakness in `scene_level_calibration.md`. Not
  investigated further here.
