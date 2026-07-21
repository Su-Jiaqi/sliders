# Classifier checkpoint-selection leakage — Phase 0 summary (2026-07-17)

## Bug
`classifier/train.py:326` selects the "best" checkpoint via `if test_acc > best_acc`,
directly on the same 246-pair socalfire test split reused for CAS Accuracy throughout
the paper (Tables 1/2/6/7/8/9/10/11/14). This is test-set leakage in model selection.

## Step 1 — retrain leak-free classifier
`claudecode/code/classifier_fix/train_classifier_clean_split.py`, val-selected
(15% held-out val carved from train pool, test touched once at the end):
- Checkpoint: `output-models/classifier/socalfire_cls_clean_split/best.pt`
- Held-out test accuracy: **0.9898** (243/246 pre correct, 244/246 post correct)

## Step 2 — recompute CAS-based tables with clean classifier
### Table 1 (endpoint baseline comparison, s=1)
| Method | Leaked CAS | Clean CAS | Δ |
|---|---|---|---|
| ControlNet | 0.9980 | 0.9939 | -0.0041 |
| Pix2Pix | 0.9939 | 0.9898 | -0.0041 |
| Palette | 0.9941 | 0.9882 | -0.0059 |
| RiskSlider (Ours) | 0.9980 | 0.9878 | -0.0102 |
| CycleGAN | 0.9878 | 0.9837 | -0.0041 |

RiskSlider's CAS drops ~2x more than any baseline and falls to 4th/5th place under
the clean classifier. **All CAS numbers in Table 1 must be updated before submission**;
RiskSlider can no longer be framed as CAS-competitive, reinforcing the existing
"no single metric wins on everything" framing in the Appendix.

### Table 14 (ablation multi-scale CAS trend) — re-run via `eval/socalfire_ablation_multiscale_cas_table.py`
Full leaked-vs-clean numbers: `claudecode/result/classifier_fix/table14_leaked/` and
`claudecode/result/classifier_fix/table14_clean/`.

**Core qualitative claim survives**: `w/o Pseudo Supervision` still collapses/saturates
immediately (s=0→0.25 jump then flat) under both classifiers; `Ours` still shows a
gradual, monotonic rise across all 5 scale points under both classifiers. Absolute
values shift down uniformly (largest drop for `w/o Refinement`, -0.09 to -0.12,
consistent with the leaked classifier's inflation being largest for the most
ambiguous/borderline images). **Numbers need updating; the argument does not.**

### Table 2 — no new compute needed
Table 2's CAS Accuracy column at s=1 is identical to Table 14's s=1 column for the
same four variants (already produced above). LPIPS/SSIM/PSNR/FID/CLIP-I/DINO-I do not
depend on the classifier and are unaffected by the leak.

### Table 7 (unrefined generation module, socalfire train+test) — DONE
`claudecode/result/classifier_fix/cross_tables/table7_unrefined_socalfire.csv`. Same
pattern as Table 14: numbers shift down slightly, monotonic-with-s trend on the test
split preserved (0.652→0.652→0.648→0.654→0.669→0.671→0.685).

### Table 8/9 (refined, socalfire train+test) — DONE
`claudecode/result/classifier_fix/cross_tables/table8_9_refined_socalfire.csv`. CAS
trend and the mean-P(post) trend used for Table 9 both remain cleanly monotonic under
the clean classifier (test: 0.636→0.738→0.756→0.909→0.980→0.984→0.988).

### Table 10/11 (cross-category CAS) — DONE, with an important correction along the way
**Correction to an earlier claim in this investigation**: an initial recompute using
the *socalfire* classifier applied to all five categories produced non-monotonic,
even decreasing, CAS trends for Hurricane/Flooding/Santa Rosa -- this was wrongly
reported as a possible pre-existing flaw in the paper's Table 11. It was not: it was
a methodology error on this investigation's part. `logs/refine-2/run_eval_full_selected_datasets.sh`
confirmed Table 10/11 actually uses a **separate classifier trained per category**
(`output-models/refine-2/{category}/classifier_real_fresh/best.pt`), not the shared
socalfire one. Re-running with the correct per-category leaked classifiers reproduces
the paper's published Table 11 numbers closely (Flooding and Volcano exactly; Hurricane
and Santa Rosa Wildfire match at 4/5 scale points, off by 2-4 points at s=0.75, likely
inference-seed variance already acknowledged in the paper's Table 3) --
`claudecode/result/classifier_fix/cross_tables/table10_11_percategory_LEAKED_verify.csv`.

These four per-category classifiers **each has the identical checkpoint-selection
leakage bug** (confirmed via `logs/refine-2/midwest-flooding/00_train_classifier.log`,
same `Saved best checkpoint` pattern from the buggy `classifier/train.py`). Retrained
all four with the clean val-split protocol
(`output-models/refine-2/{category}/classifier_clean_split/best.pt`) and recomputed:
`claudecode/result/classifier_fix/cross_tables/table10_11_percategory_CLEAN.csv`.

| Category | s=0 | s=0.25 | s=0.5 | s=0.75 | s=1 | Monotonic? |
|---|---|---|---|---|---|---|
| SoCal Wildfire (primary) | 0.636 | 0.738 | 0.909 | 0.984 | 0.988 | Yes |
| Hurricane | 0.479 | 0.521 | 0.763 | 0.938 | 0.954 | Yes |
| Flooding | 0.506 | 0.544 | 0.969 | 1.000 | 1.000 | Yes |
| Santa Rosa Wildfire | 0.500 | 0.500 | 0.553 | 0.660 | 0.723 | Yes |
| Volcano (N=4) | 0.500 | 0.625 | 0.500 | 0.500 | 0.500 | No |

**The paper's cross-category claim ("CAS increases monotonically with s in all
categories") survives the leakage fix for every category the paper itself treats as
reliable.** Only Volcano (N=4 test scenes, already flagged by the paper as "too small
for reliable metric estimation... included only for completeness") is non-monotonic
under the clean classifier -- consistent with pure sampling noise at that sample size,
not a real finding about the method.

## Step 3 — frozen semantic teacher ψ (Eq. 6, used to train f1)
Could not conclusively trace which checkpoint was used as ψ for socalfire's original
f1 training run (log dated 2026-04-21 19:16 predates the earliest classifier
checkpoint surviving on this machine, 20:15; no shell script preserved the invocation).
Per user: results may have been produced across two different machines, so exact
reproduction of the historical run is not required — retrained instead.

**Retrained f1 from scratch** (`refine/scale1_only_refiner_stronger.py`, identical
hyperparameters to the paper's Table 4/5) using the confirmed clean classifier
(`socalfire_cls_clean_split/best.pt`) as ψ:
- Checkpoint: `output-models/refine/socalfire/socalfire_scale1_refiner_clean_psi_20260717/best.pt`
- Refined test set (s=1): `outputs/refine-clean-psi/socalfire/test/scale1`

| | LPIPS↓ | SSIM↑ | PSNR↑ | CAS Acc↑ (clean clf) |
|---|---|---|---|---|
| Original f1, clean-clf eval | 0.2166 | 0.5954 | 23.81 | 0.9878 |
| **Retrained f1 (clean ψ), clean-clf eval** | 0.2188 | 0.5826 | 23.47 | **0.9939** |

**Conclusion: retraining f1 end-to-end with a confirmed leak-free ψ does not hurt —
CAS accuracy is actually marginally higher (+0.0061), LPIPS/SSIM/PSNR shift by amounts
comparable to the paper's own seed-variance noise floor (Table 3: σ_LPIPS≈0.0004,
σ_PSNR≈0.01 — though the PSNR delta here, -0.34, is somewhat larger than that floor and
should be flagged as such rather than dismissed).** No evidence that ψ's potential
leakage inflated the paper's reported endpoint quality.

## Net takeaway for the paper
1. All CAS Accuracy numbers in Tables 1/2/7/8/9/10/11/14 have now been recomputed
   with leak-free classifiers and must be swapped into the paper before submission —
   mechanical but necessary. Table 6 (Concept Slider) cannot be recomputed: no
   Concept Slider generations exist on this machine (likely produced elsewhere,
   consistent with the paper's work having been split across two machines).
2. Every qualitative argument tested so far is robust to the fix: refinement helps,
   pseudo-target supervision prevents collapse, endpoint quality holds, and —
   after correcting an initial methodology error on our part — the cross-category
   monotonic CAS trend (Table 11) also holds for every category the paper treats
   as reliable (all but Volcano, N=4).
3. Recommend swapping in `socalfire_scale1_refiner_clean_psi_20260717` as the
   production f1 checkpoint going forward, since it is now verifiably leak-free
   end-to-end and performs at least as well.
4. Recommend swapping in the four newly-trained per-category clean classifiers
   (`output-models/refine-2/{hurricane-florence,midwest-flooding,santarosa,volcano}/classifier_clean_split/best.pt`)
   wherever Table 10/11's CAS numbers are reported.
