# Draft: "zero marginal cost controllability" reframing — 2026-07-25

## Purpose

Per the review-response plan, reviewer item 5: move the negative controls
(Table 22/23, now also the alpha-blend control) into the main text and frame them
proactively rather than let them read as something the paper got caught by. This
is a writing/narrative draft only — I don't have access to the LaTeX source, so
this is prose to paste in, not an applied edit.

## Honest framing constraint

Don't oversell this. The reviewer's verdict was explicit: "in a single inference
pass, without a separate post-hoc interpolation step" is engineering convenience,
not a methods contribution, and reframing alone will not flip a 4/10 to accept.
What this reframe CAN honestly do: turn a finding that currently reads as "reviewer
caught us" into "we tested this ourselves, here's exactly what holds and what
doesn't" — which is a real, if modest, credibility improvement, and is table stakes
for whatever venue this eventually lands in.

## Draft paragraph (for wherever Table 22/23 land in the main text)

> **Where does progression-conditioning's value actually lie?** We stress-test this
> directly with two controls. First, an endpoint-only variant of the same backbone
> trained with identical hyperparameters but no intermediate-severity supervision
> (Table 23) matches RiskSlider's endpoint quality on every metric to within the
> 3rd-4th decimal place — progression-conditioned training provides no measurable
> endpoint-quality advantage over simply fine-tuning on the same task. Second, we
> compare against the simplest possible continuous baseline: linearly interpolating
> RiskSlider's own generated endpoint through the same refiner (Table 22), and
> separately, linearly blending the real pre/post pixels directly with no model at
> all (Appendix~\ref{app:negative_controls}). Both land within noise of RiskSlider's
> full pipeline on population-level metrics, and PCR specifically cannot
> distinguish RiskSlider's trajectory from either control — a genuine limitation of
> PCR as controllability evidence, not just of these baselines. **What
> progression-conditioning does provide, and what neither control can, is a single
> forward pass that produces any requested severity directly**, without a
> post-hoc interpolation or compositing step, at zero measurable cost to output
> quality relative to the alternatives we tested. We report this as the honest
> scope of the contribution: RiskSlider does not learn a severity representation
> distinguishable from these controls on the evidence gathered here (see
> Appendix~\ref{app:scene_calibration} for the scene-level calibration results,
> which are also weak outside the trained endpoint) — its value is architectural
> and operational (single-pass, model-native control), not a demonstrated gain in
> generation quality or severity fidelity over substantially simpler alternatives.

## What to cut/move alongside this (mechanical, per reviewer item 5)

- Move Table 22 (naive interpolation) and Table 23 (same-backbone control) from
  appendix to main text, adjacent to this paragraph.
- Add the alpha-blend control result (`alpha_blend_trivial_control_20260725.md`)
  as a third row/panel in whichever table currently holds Table 22 — same
  structure, one more baseline.
- Drop Volcano (N=4) from every main-text and appendix table; keep a one-line
  footnote noting it was excluded for sample-size reasons if the category needs to
  be mentioned at all elsewhere.

## What this reframe does NOT fix

Per the reviewer's own priority ranking, this does not address: (1) s not carrying
calibrated severity information (needs real retraining with xBD damage-level
supervision — out of scope for 3 days), (2) missing Changen2/DisasterGAN baselines,
(3) the user study design. Recommend pairing this paragraph with an explicit
"Limitations" bullet naming these three as future work, rather than implying the
zero-cost framing resolves them.
