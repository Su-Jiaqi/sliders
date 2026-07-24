# Table 1 headline numbers — confirmed stable/reproducible on re-scoring, distinct from the seed-stability generation-drift issue — 2026-07-25

## What the reviewer flagged

"Table 1's headline LPIPS 0.2166 vs. the same-pipeline seed-stability rerun's
0.2201±0.0004 — a ~9σ gap — disclosed via footnote, but the correct fix is to rerun
Table 1 in the fixed environment, not add a footnote."

## Important distinction this reproduces and clarifies

There are two different things that could be "non-reproducible" here, and they are
not the same problem:

1. **Re-scoring the same, already-generated images** (what Table 1's number is
   computed from) — this does NOT depend on the PyTorch/xFormers environment at all.
   LPIPS/SSIM/PSNR/CLIP-I/DINO-I are pure feature-extraction/pixel-comparison
   metrics; nothing in their computation touches xFormers (xFormers only affects the
   UNet attention kernel during diffusion *sampling*, which doesn't happen during
   scoring).
2. **Regenerating the base images from scratch** (what the seed-stability sweep did,
   on 2026-05-04, in an environment where xFormers had already broken) — this DOES
   depend on the environment, as already documented in
   `headline_vs_seed_stability_gap_20260725.md`.

Table 1's number was never regenerated this session — only re-scored, many times,
on the same static on-disk images (`outputs/refine-2/socalfire/test/scale1`).
Re-confirmed once more just now, independently, with a fresh minimal script (not
reusing any infrastructure built earlier today), on `cuda:7`:

**LPIPS = 0.21660283650082301** — exact float-for-float match to every other time
this number has been computed this session (at least 5 separate independent script
invocations across this entire multi-round audit, including today's Table 1
effect-size table, the s=0-fix ablation-figure work, and this standalone check).

## Conclusion

**Table 1's own reported numbers are stable and reproducible today, and do not need
regenerating** — the discrepancy the reviewer is reacting to is real, but it lives
entirely on the seed-stability table's side (whose base images were regenerated in
a different, later, partially-broken environment), not on Table 1's. Rerunning
Table 1's *generation* wouldn't change Table 1's numbers (they're already what a
current re-score gives); what actually needs to happen is either (a) regenerating
the seed-stability sweep's images in a fixed/matching environment so its numbers
become comparable to Table 1, or (b) disclosing the environment-drift caveat on the
seed-stability table specifically (the cheaper of the two, already drafted in
`headline_vs_seed_stability_gap_20260725.md`).

**Recommended paper-facing fix**: don't imply Table 1 itself was rerun/changed (it
wasn't, and doesn't need to be) — instead tighten the seed-stability section's
footnote into the fuller disclosure already drafted, making clear the two tables'
numbers come from different, non-matching generation environments, and that Table
1's own number is independently confirmed stable under re-scoring.

## Caveat

This does not address the reviewer's separate, larger concern about the *fairness*
of Table 1 as a comparison (pretrained SD backbone vs. from-scratch GANs) — see
`table1_baseline_fairness_disclosure_20260725.md` for that.
