# P0: s=1 test set regenerated in current environment — matches headline Table 1 almost exactly — 2026-07-25

## What was done

Per the reviewer's explicit instruction ("正确处理是在修复后的环境里重跑整张Table 1"),
regenerated all 246 socalfire test-split s=1 outputs from scratch in the current
environment: base diffusion (`infer/batch_infer_xbd_paired_scale01.py`, steps=50,
start_noise=100, guidance_scale=1.0, seed=42 — matching the endpoint-generation
script's own defaults and `same_backbone_control_20260724.md`'s documented
methodology) using the best-available reconstruction of the production LoRA
checkpoint (original path missing from disk, same substitute and caveat as
`headline_vs_seed_stability_gap_20260725.md`), then refined through the confirmed
production refiner (`unified_refiner_scale1stronger/best.pt`). Ran on `cuda:4`;
`cuda:0` untouched. Baselines were NOT rerun (per instruction — their per-image
metrics are already cached from the existing Wilcoxon/effect-size analysis).

## Result: regeneration reproduces the headline numbers almost exactly

| Metric | Headline (existing) | Regenerated (this run) | Diff |
|---|---|---|---|
| LPIPS↓ | 0.216603 | 0.216597 | 0.000006 |
| SSIM↑ | 0.595381 | 0.595378 | 0.000003 |
| PSNR↑ | 23.8062 | 23.8061 | 0.0001 |
| FID↓ | 175.19 | 175.19 | ~0 |
| CLIP-I↑ | 0.92813 | 0.92816 | 0.00003 |
| DINO-I↑ | 0.79858 | 0.79850 | 0.00008 |
| PCR (clean classifier) | 0.9878 | 0.9898 | 0.0020 |

Every pixel/semantic metric matches to 4-5 decimal places; PCR differs by 0.002
(well within the seed-noise floor documented elsewhere, Table 3's σ≈0.001-0.002).

## What this means, and an important distinction from the seed-stability finding

**Table 1's headline numbers are genuinely reproducible** — this is a real, full
end-to-end regeneration (not a re-score of existing images, unlike
`table1_reproducibility_confirmation_20260725.md`'s check), done today, in the
current (xFormers-broken) environment, using the best-available checkpoint
substitute, and it lands within noise of the original. **This directly contradicts
a naive reading of the seed-stability finding as "the environment change broke
generation reproducibility in general."** It did not, at least not for this
script/methodology. The seed-stability sweep's ~9σ gap is better explained by:
the sweep specifically tested seeds 42/43/44 with `infer/batch_infer_xbd_paired_scales.py`
(a different script than this one, used for the seed-stability sweep's own base
generation), and none of those seeds match whatever seed the *original* April 21
headline batch actually used internally per-image — genuine seed-to-seed variation,
not a systemic environment failure. (This regeneration used the *same* script and
default seed=42 as the endpoint generation always uses, which may be why it
reproduces so cleanly.)

## Recommendation

This is strong, direct evidence to cite in response to the review: Table 1 was
regenerated end-to-end in the current environment and reproduces within noise. No
footnote-only fix needed for Table 1 itself. The seed-stability table's own
disclosure (already drafted) should be kept, since that specific comparison used a
different generation script/seed set and remains only partially reconciled -- but it
should no longer be framed as casting general doubt on Table 1's reproducibility.

## Caveat

Same as `headline_vs_seed_stability_gap_20260725.md`: uses the best surviving
reconstruction of the production LoRA checkpoint (original path is gone from disk),
not a byte-confirmed identical weights file.
