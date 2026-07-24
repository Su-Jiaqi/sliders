# s=0 pixel-identity fix — executed, results, and two important caveats — 2026-07-25

## What was done

Per the decided fix (`s0_pre_disaster_gap_20260725.md`), built pixel-identity scale0
folders for all 5 categories (`claudecode/code/classifier_fix/build_scale0_pixel_identity.py`
— real x_pre resized once, bilinear, to the pipeline's 256x256 working resolution, no
VAE/model in the loop) and re-ran the standard eval script
(`eval/socalfire_infered_eval_metrics.py`, clean classifier `socalfire_cls_clean_split`)
on all 7 scales per category, swapping only scale0. Ran on `cuda:5`/`cuda:7`; `cuda:0`
untouched. Full per-category CSVs: `outputs/eval/scale0fix/<category>/socalfire-infered_metrics.csv`.

## Result 1: LPIPS/SSIM/PSNR barely move, and the 0→0.25 boundary stays smooth

| Category | LPIPS old→new (s=0) | SSIM old→new | PSNR old→new | Δ(0→0.25) LPIPS old / new |
|---|---|---|---|---|
| Wildfire (socalfire, n=246) | 0.3062→0.3075 | 0.5035→0.5030 | 21.646→21.626 | -0.0249 / -0.0262 |
| Hurricane (n=97) | 0.3236→0.3232 | 0.4890→0.4910 | 20.326→20.329 | -0.0118 / -0.0113 |
| Flooding (n=80) | 0.3641→0.3641 | 0.4646→0.4654 | 20.605→20.577 | -0.0355 / -0.0355 |
| Santa Rosa (n=47) | 0.3091→0.3094 | 0.4723→0.4735 | 20.277→20.236 | -0.0140 / -0.0143 |
| Volcano (n=4) | 0.4483→0.4493 | 0.3849→0.3868 | 20.561→20.549 | -0.0269 / -0.0279 |

Every value changes only in the 3rd-4th decimal, and the size of the jump to s=0.25
is essentially unchanged before vs. after the fix. **This means the paper's own
"gap" language (63.6% vs. 1.2%) is not describing a pixel-similarity gap** — the
model's old s=0 output was already close to x_pre in raw LPIPS/SSIM/PSNR terms.
Whatever "1.2%" measures in the paper's text, it is not LPIPS/SSIM/PSNR as computed
here — I don't have the exact sentence/metric definition, so I can't confirm this
fix produces "1.2%" on that specific metric. **Please check this against the exact
wording before using these numbers.**

## Result 2: CAS at s=0 — the "63.6%" number is very likely just this, but the fix does NOT bring it near zero, and CAS at s=0 turns out not to be a clean signal at all

Socalfire's OLD (pre-fix) clean-classifier CAS at s=0 is **0.6362** — this almost
certainly *is* the paper's "63.6%" number (exact match to 3 significant figures).

After the pixel-identity fix, socalfire's CAS at s=0 is **0.6260** — barely changed,
**not anywhere near 1.2%**. Full comparison, all freshly recomputed together in one
run (see caveat below on why "freshly together" matters):

| Category | OLD CAS(s=0) | NEW CAS(s=0), pixel-identity | n |
|---|---|---|---|
| Wildfire | 0.6362 | 0.6260 | 246 |
| Hurricane | 0.5928 | 0.6031 | 97 |
| Flooding | 0.6250 | 0.6000 | 80 |
| Santa Rosa | 0.6596 | 0.6596 | 47 |
| Volcano | 0.7500 | 0.6250 | 4 |

None of these approach 50% (the value you'd expect if the classifier treated a true
copy of x_pre symmetrically with the real x_pre reference) let alone 1.2%. Dug into
why: **the classifier is measurably sensitive to a resize-pipeline artifact specific
to this fix.** Spot-checked 30 socalfire test images, comparing the classifier's
prediction on (a) the real x_pre file loaded directly from its 1024px source
(single resize straight to the classifier's 224px input) vs. (b) the same file after
my one intermediate 256px save-and-reload (which is otherwise how every scale folder
in this whole project is stored) — **8/30 (27%) flip prediction entirely**, several
with dramatic confidence swings (e.g. one image: 99.5% "pre" via the direct path,
91.1% "post" via the double-resize path, despite the underlying image content being
identical). Verified this isn't a bug in the copy itself (my saved 256px file is a
bit-exact match to a fresh direct resize) — it's the classifier reacting to a small
(~2/255 mean pixel difference) but apparently decision-relevant artifact from the
extra resize step. **This means CAS at s=0 is not a trustworthy signal either way,
independent of which fix is applied** — the classifier's behavior at s=0 is
dominated by this resize sensitivity, not by whether the image is semantically
pre- or post-disaster. Recommend not leaning on the CAS(s=0) number at all in the
paper's honest-disclosure framing; LPIPS/SSIM/PSNR are unaffected by this issue
(they use the project's standard single 256px eval resize, same as every other scale).

## Important caveat: classifier scoring is not perfectly reproducible across this project's own environment drift

While cross-checking, found that the previously-stored
`claudecode/result/classifier_fix/cross_tables/table10_11_percategory_CLEAN.csv`
(generated 2026-07-17, same day as the classifier checkpoint, images from
2026-05-03 — nothing on disk has changed since) **does not reproduce when
recomputed today**: stored Hurricane CAS at s=0.25 = 0.5206; rerunning the identical
script/checkpoint/images right now gives 0.5928 (confirmed deterministic across 3
repeated calls in this session, so not run-to-run randomness). This is the same
underlying phenomenon documented in `headline_vs_seed_stability_gap_20260725.md`
(the machine's PyTorch/CUDA/xFormers stack changed at some point this project) —
except that finding was about diffusion generation; **this shows the same
environment drift also perturbs classifier inference determinism**, likely via
different cuDNN/CUDA convolution kernels flipping borderline predictions, the same
mechanism as the resize-sensitivity finding above. Practical consequence: **the
Cross-Category fix recommended earlier this session
(`appendix_cross_category_tables_20260724.md`, using `table10_11_percategory_CLEAN.csv`
as the replacement data) may itself need a fresh recompute in the current
environment before being trusted numerically** — the qualitative conclusion (that
table was leaked-classifier-contaminated) still stands, but the specific clean
replacement values in that CSV are not guaranteed bit-reproducible today. Flagging
this rather than silently recomputing and replacing it, since that's a separate
decision from what was asked this round.

## What to do next

- LPIPS/SSIM/PSNR at s=0: safe to use the new pixel-identity numbers above, the fix
  is uncontroversial and doesn't disturb the trajectory shape.
- CAS at s=0: recommend dropping it from the honest-disclosure claim entirely (it's
  not measuring what it appears to measure), or if it must stay, disclose the
  resize-sensitivity caveat alongside it.
- Need the exact sentence/metric behind "1.2%" to know whether this fix actually
  addresses what the paper claims, or whether the paper's real concern is a
  different metric I haven't touched.
