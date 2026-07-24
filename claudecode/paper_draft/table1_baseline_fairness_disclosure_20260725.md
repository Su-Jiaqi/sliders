# Table 1 baseline-fairness disclosure — draft text, 2026-07-25

## What the reviewer flagged

Pix2Pix/CycleGAN/Palette are trained from scratch, no pretraining, no
hyperparameter search; RiskSlider fine-tunes a pretrained SD v1.4 backbone. This
asymmetry is already noted in supplementary material, but Table 1 is still the
paper's headline comparison, and numbers like Palette FID=434 / SDEdit FID=651 read
as "we're better" when they're really "these from-scratch baselines weren't trained
to convergence" — which undermines the whole table's credibility once a reader
notices it. Compounding this: CycleGAN significantly beats RiskSlider on CLIP-I
(p=0.0118) and DINO-I (p<1e-5), and on building-footprint IoU (0.364 vs. RiskSlider's
0.289, actually higher than the real-image ceiling of 0.282 — itself evidence the
IoU proxy isn't fully trustworthy). FID's CycleGAN-vs-Ours difference has a bootstrap
CI that crosses zero ([-13.5, +12.9], not significant). On the one dimension
(CLIP-I/DINO-I, structural IoU) where the comparison is closest to fair (both
methods are scored the same way, and CycleGAN itself is at least a real trained
GAN, not a from-scratch SDEdit-style failure), RiskSlider has no significant
advantage — CycleGAN wins two of three, ties on FID.

## Verified numbers (all from this session's own re-derivations, cited above)

| Baseline | LPIPS | SSIM | PSNR | CLIP-I | DINO-I | Building IoU | FID |
|---|---|---|---|---|---|---|---|
| RiskSlider (Ours) | 0.2166 | 0.5954 | 23.81 | 0.9281 | 0.7986 | 0.289 | — |
| CycleGAN | 0.2739 (Ours wins, p<1e-40) | 0.5182 (Ours wins) | 21.80 (Ours wins) | **0.9312 (CycleGAN wins, p=0.012)** | **0.8127 (CycleGAN wins, p<1e-5)** | **0.364 (CycleGAN wins; exceeds real-image ceiling 0.282)** | 178.7 vs 144.2, CI [-13.5,+12.9] crosses 0 — not significant |
| Palette | — | — | — | — | — | — | 434.47 |
| SDEdit (training-free) | — | — | — | — | — | — | 651.55 |
| Real images (reference ceiling) | — | — | — | — | — | 0.282 | — |

## Recommended disclosure text (draft, for near Table 1)

> **A note on baseline comparability.** Pix2Pix, CycleGAN, and Palette are trained
> from scratch on our 769-pair training set with no pretraining and no per-method
> hyperparameter search, while RiskSlider fine-tunes a pretrained Stable Diffusion
> v1.4 backbone; this asymmetry favors RiskSlider on any metric sensitive to general
> image quality and should temper the strength of claims drawn from Table 1 alone.
> Consistent with this, on the two metrics closest to a content-agnostic
> pretrained-feature comparison (CLIP-I, DINO-I), CycleGAN — the strongest-trained
> of the three from-scratch baselines — significantly outperforms RiskSlider
> ($p=0.012$ and $p<10^{-5}$ respectively, Holm-corrected), and the two methods are
> statistically indistinguishable on FID (bootstrap 95% CI for the paired
> difference crosses zero). RiskSlider's advantage in Table 1 is concentrated in
> pixel-level fidelity (LPIPS/SSIM/PSNR), where its pretrained backbone's prior is
> the most direct explanation, not evidence of a better disaster-generation method
> per se; see Appendix~\ref{app:limitations} for the full baseline-training
> protocol.

## Recommendation

This text (or a tightened version of it) should sit directly under/near Table 1,
not only in supplementary material, per the reviewer's specific complaint that the
asymmetry is disclosed but not where it affects how the headline table is read.
