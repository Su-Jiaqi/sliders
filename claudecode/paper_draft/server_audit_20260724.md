# Server-side audit, 2026-07-24 — dataset/result/protocol/statistics

Triggered by an external reviewer-style checklist asking to double-check Table 1's
dataset counts, per-baseline result provenance, training protocol disclosures, and
statistical methodology before doing any new experiments. All four items below are
read-only verification, not new experiments; the one substantive finding (Table 1's
CAS column) is written up in full in `table1_significance.md`'s top section and
summarized in `OVERVIEW.md`'s Phase 0 section — this file is the supporting detail for
the other three, clean audits.

## 1. Dataset audit — clean, no issues found

- socalfire train=769, test=246 pairs confirmed via direct file count; no source
  anywhere in the repo claims 770 (only a couple of unrelated `0.770` metric values
  matched the grep, not a dataset-size claim).
- **Table 1 is confirmed category-specific**: `train/configs/*.yaml` shows exactly one
  active `dataset.root` per training config, never mixed across disaster types — one
  LoRA per category. Table 10/11 (cross-category) uses separate per-category
  classifiers, documented in `SUMMARY.md`.
- **Train/test split is tile-disjoint by construction, not just by accident.**
  Verified via md5 hashing: all 769 `datasets/remote/socalfire/train/pre` files match
  the original xBD dataset's own 823 official train-tile hashes exactly, and all 246
  `test/pre` files match xBD's own 307 official test-tile hashes exactly, with zero
  cross-matches in either direction, and zero overlap between xBD's own train/test
  hash sets. `datasets/remote/socalfire` is a filtered subset of xBD's own
  already-geographically-disjoint partition — there is no train/test leakage at the
  tile level.
- Other categories' train/test counts (for Table 10/11 context): santarosa 72/47,
  volcano 15/4, hurricane-florence 292/97, midwest-flooding 279/80. Volcano's N=4 test
  set matches the paper's own "too small for reliable estimation" caveat.

## 2. Result audit (Table 1 provenance) — clean, no issues found (except the known Palette n=246 fix and the CAS finding above)

All four non-Palette Table 1 rows (ControlNet, Pix2Pix, CycleGAN, RiskSlider) are
solid:
- Exactly n=246 matched stems for all four, zero duplicates, zero mismatches against
  the 246 real post-disaster GT filenames.
- Filename/stem matching (suffix-stripping for Pix2Pix/CycleGAN, plain-name for
  ControlNet/RiskSlider) is unambiguous — spot-checked, no off-by-one bugs.
- LPIPS/SSIM/PSNR/FID/CLIP-I/DINO-I for all four trace exactly to dedicated eval
  CSV/JSON outputs (`outputs/eval/{controlnet,pix2pix,cyclegan}/*_metrics.csv`,
  `outputs/refine-2/socalfire/eval-full/socalfire-infered_metrics.csv`), matching
  Table 1 to 4-5 decimal places — not hand-typed.
- No conflicting citations of these four methods' numbers found elsewhere in
  `paper_draft/*.md`.
- **Only the CAS column has a real provenance problem** — see
  `table1_significance.md`'s critical-fix section and `OVERVIEW.md` Phase 0.

## 3. Training protocol audit — clean, all four baselines properly disclosed-if-asked

- **ControlNet**: fine-tuned from pretrained `runwayml/stable-diffusion-v1-5`
  (UNet/VAE/text-encoder), with a freshly-initialized ControlNet branch via
  `ControlNetModel.from_unet()` — the standard official recipe, not from scratch.
  15,000 steps, lr=1e-5, batch 1 x grad-accum 8, res=256.
- **Pix2Pix**: trained fully from scratch (`continue_train: False`, no checkpoint
  loaded), lr=2e-4, unet_256 generator, 100+100 epochs (linear decay), batch 1,
  res 256 (crop from 286).
- **CycleGAN**: trained fully from scratch, resnet_9blocks generator + PatchGAN
  discriminator, lr=2e-4, 100+100 epochs, batch 1, res 256.
- **Palette**: trained fully from scratch (`resume_state: null`, kaiming init),
  custom 6-channel-input guided-diffusion UNet, lr=5e-5, res 256, batch 1,
  n_timestep=2000; training halted around epoch 40 by a stability-monitor heuristic,
  not a fixed epoch budget.
- **RiskSlider** fine-tunes pretrained `CompVis/stable-diffusion-v1-4` via LoRA
  (rank=16, alpha=16.0, `train_method=full`), production checkpoint
  `output-models/train/socalfire_slider_paired_alpha16.0_rank16_full/socalfire_slider_paired_alpha16.0_rank16_full_last.safetensors`.
- **Fairness note worth adding to the paper**: RiskSlider and ControlNet both start
  from a pretrained SD backbone; Pix2Pix/CycleGAN/Palette are trained from scratch on
  the ~769-pair training set. This is worth stating explicitly in the baselines
  paragraph (it's an honest, defensible asymmetry — GAN/non-diffusion baselines don't
  have an equivalent large-scale pretrained initialization to start from — but a
  reviewer will ask, so better to state it first).
- **Open, not-yet-applied recommendation (separate from the CAS column fix)**:
  `SUMMARY.md` recommended swapping in a refiner retrained with the clean classifier
  as semantic teacher ψ (`socalfire_scale1_refiner_clean_psi_20260717/best.pt`) as the
  production f1 checkpoint. Confirmed this was never actually applied — production's
  `outputs/refine-2/socalfire/test/scale1` is still generated by the original
  `unified_refiner_scale1stronger` checkpoint (teacher: `socalfire_cls_real_fresh`).
  This is a separate, lower-urgency item from the CAS-column fix (it's about which
  refiner *produces* the images, not which classifier *scores* them) — flagging for
  the user's judgment on whether to apply it; not done as part of this audit since it
  would require re-generating and re-verifying every downstream number.

## 4. Statistical audit — clean, no issues found

Two methods used throughout, both consistent and code-verified:
- Paired two-sided Wilcoxon signed-rank test (per-image LPIPS/SSIM/PSNR/CLIP-I/DINO-I),
  implemented in `claudecode/code/classifier_fix/paired_significance_table1.py` and
  `paired_significance_clip_dino.py`, matching `scipy.stats.wilcoxon` calls exactly.
- One-sided binomial test against chance (0.5) for concordance rate (scene-level
  calibration), implemented in `claudecode/code/experiment_b/group_ceiling_and_concordance.py`,
  matching the methodology described in `scene_level_calibration.md`'s table caption.
- No Mann-Whitney anywhere in the codebase (the concern that concordance p-values
  "should be" Mann-Whitney was based on a mismatched premise — a one-sided binomial
  test against chance is the correct test for a binary concordance rate, not a
  substitute for a two-sample test). The 3.5e-52 concordance p-value (scale=1,
  n_pairs=2442) is a real, correctly-computed, consistently-reported number, not a
  stale artifact — traced to `claudecode/result/socalfire/experiment_b/pairwise_concordance_imagenet_probe.csv`
  and reproduced identically in `scene_level_calibration.md`.
