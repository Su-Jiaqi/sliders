# Pixel-space alpha-blending trivial control — result confirms the reviewer's core critique — 2026-07-25

## What was tested

The reviewer's sharpest methodological point: PCR (balanced pre/post classifier
accuracy) measures "how post-disaster does this look," not "how severe/controlled
is this," so a smooth PCR trajectory doesn't by itself demonstrate learned
continuous control. Table 22 (naive latent interpolation + the real refiner) already
showed RiskSlider is statistically indistinguishable from a semi-trivial baseline.
This pushes one step further: the most trivial possible "continuous" baseline —
**no model anywhere, just pixel-space linear blending of the real pre/post pair**,
$x_{\text{blend}}(s) = (1-s)\cdot x_{\text{pre}} + s\cdot x_{\text{post}}$ — computed
directly, no generator, no refiner, no learned component at all
(`claudecode/code/classifier_fix/build_alpha_blend_control.py`), evaluated with the
identical pipeline/classifier used everywhere else this session.

## Result: PCR trajectory shape is nearly indistinguishable from RiskSlider's

| s | RiskSlider PCR | Alpha-blend PCR | RiskSlider LPIPS | Alpha-blend LPIPS |
|---|---|---|---|---|
| 0 | 0.5000 | 0.6260 | 0.3075 | 0.3075 |
| 0.25 | 0.7378 | 0.7093 | 0.2813 | 0.2505 |
| 0.3 | 0.7561 | 0.7500 | 0.2761 | 0.2358 |
| 0.5 | 0.9085 | 0.8902 | 0.2539 | 0.1619 |
| 0.7 | 0.9797 | 0.9797 | 0.2338 | 0.0727 |
| 0.75 | 0.9837 | 0.9837 | 0.2296 | 0.0527 |
| 1 | 0.9878 | 0.9939 | 0.2166 | 0.0000 |

**PCR: the two trajectories are within 1-3 points of each other at every scale
from 0.25 onward, and identical to 4 decimal places at s=0.7/0.75.** This directly
confirms the reviewer's claim: PCR alone cannot distinguish RiskSlider's learned
generation from literally averaging two real photographs pixel-by-pixel. Table 22
already showed this against a semi-trivial baseline (real generator + real
refiner, no s-conditioning); this shows it holds against a fully trivial one too.

## The one place the two are NOT interchangeable, and it matters for interpretation

**LPIPS tells the opposite story, and for a structural reason, not a quality one.**
Alpha-blending's LPIPS drops to exactly 0.0 at s=1 and is dramatically lower than
RiskSlider's at every intermediate scale (e.g. 0.073 vs. 0.234 at s=0.7). This is
not because blending "looks better" — it's because alpha-blending has **direct
pixel access to the real post-disaster target it's being scored against** (at s=1
it literally *is* the target image; at intermediate s, half its pixels are copied
straight from the target). RiskSlider, by contrast, has to actually hallucinate
content it never observed — a real generation task, not a compositing task. This
asymmetry means alpha-blending's near-perfect LPIPS is not evidence it's a
"better" method, and RiskSlider's LPIPS gap from it is not evidence RiskSlider is
"worse" — the two are not doing comparable tasks by this metric. **This is exactly
why PCR looking so similar between them is the finding that matters**: PCR is
supposed to be the metric specifically designed to not require ground-truth pixel
access (per the paper's own framing, `tab:app_psi_prob`/PCR's whole point is to
work at intermediate $s$ where no ground truth exists) — and it still can't tell
the two apart.

## Recommendation

This is the strongest, cheapest piece of evidence available for engaging with the
reviewer's core critique honestly. Recommend adding this as a second row in
whatever table/figure ends up presenting Table 22's naive-interpolation control —
framed exactly as above: PCR does not distinguish RiskSlider from trivial pixel
blending, while acknowledging LPIPS is not a fair comparator here due to blending's
direct target access. This directly motivates the reviewer's suggestion to find or
design a metric that can separate "continuous control" from "trivial mixing,"
rather than leaning on PCR as the primary controllability evidence.

## Caveat

One data point (DINO-I at s=0.3) failed to compute due to a transient network
error (HF mirror proxy connection refused mid-run) — everything else in the table
is complete; can be backfilled in under a minute if needed, doesn't affect any
conclusion above (PCR/LPIPS, the two metrics this finding rests on, are both fully
populated at every scale).
