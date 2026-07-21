# RiskSlider revision — consolidated overview

This indexes everything produced across Phase 0-2 of the revision plan. Each item
below says: what it is, where it goes in the paper, whether it's a "must fix before
submission" or a "new finding to write up honestly", and which file/data backs it.

---

## Phase 0 — Classifier checkpoint-selection leakage (MUST FIX)

**File**: `claudecode/result/classifier_fix/SUMMARY.md`

- Bug: `classifier/train.py:326` selects the best checkpoint via `if test_acc > best_acc`,
  directly on the same 246-pair test split reused for CAS Accuracy throughout the paper.
- Fixed: retrained with a proper held-out validation split
  (`output-models/classifier/socalfire_cls_clean_split/best.pt`, test acc 0.9898).
- **DONE — all mechanical CAS recomputes complete**: Tables 1/2/7/8/9/10/11/14 have
  all been recomputed with leak-free classifiers. Every qualitative argument survives
  (endpoint quality, refinement helps, pseudo-target supervision prevents collapse,
  cross-category monotonicity). Table 6 (Concept Slider) cannot be recomputed — no
  generations exist on this machine.
- **Important sub-finding**: Table 10/11 (cross-category) uses a *separate classifier
  trained per disaster category* (`output-models/refine-2/{category}/classifier_real_fresh/best.pt`),
  not the shared socalfire one — an initial attempt using the wrong (socalfire) classifier
  produced misleadingly non-monotonic cross-category trends and was wrongly reported as a
  possible new flaw; this was a methodology error on our part, corrected once the actual
  per-category classifier setup was found. All four per-category classifiers have the
  identical leakage bug and have been retrained clean; the cross-category monotonic-CAS
  claim holds for every category the paper treats as reliable (all but Volcano, N=4).
- Also retrained f1 (endpoint refiner) with the confirmed clean classifier as semantic
  teacher ψ — quality holds, CAS actually ticks up slightly (0.9878→0.9939). Recommend
  swapping in `output-models/refine/socalfire/socalfire_scale1_refiner_clean_psi_20260717/best.pt`
  as the production f1 checkpoint, and the four new per-category clean classifiers
  wherever Table 10/11's numbers are reported.

---

## Phase 1 — Core narrative additions

### 1. Scene-level Severity Calibration (NEW SECTION, addresses the biggest risk)
**File**: `claudecode/paper_draft/scene_level_calibration.md`
**Insert**: new main-text section after "Ablation Study", before "User Study"; full
data table in Appendix after "Additional Ablation Analysis".

- Distinguishes population-level claim (already proven: Fig.6/Table 14) from the
  stronger scene-level claim (does the model respect each scene's *own* real
  severity, not just push everything toward a generic template?).
- Honest finding: scene-level calibration is weak-to-absent at intermediate scales
  on the primary wildfire benchmark, only becoming significant at s=1 (ρ=0.388,
  p=5.1e-5); unrefined generator shows *reversed* ordering at low s (concordance
  below chance). Flooding (secondary category) calibrates better.
- All numbers now use the leak-free classifier throughout (was updated after Phase 0).
- Includes a rewritten paragraph tying this to Eq.1's linear-interpolation prior:
  the pseudo-target is constructed *within* each scene's own two endpoints, so it
  carries no cross-scene calibration signal — this is the root-cause explanation,
  and doubles as the defense against "why do you trust linear latent interpolation."

### 2. Ablation cross-check: pseudo-target supervision vs. scene-level calibration
**Same file, section "Ablation cross-check"**
- Surprising, honest finding: `w/o Pseudo Supervision` is *better* at scene-level
  calibration at intermediate scales than the full model (not worse, as originally
  hypothesized) — it's a genuine trade-off (smooth absolute trajectory vs. relative
  scene discrimination), not a simple "ablated component is worse on every axis" story.
- `w/o Scale-aware Refiner` is uniformly worse, as expected — strengthens that
  existing ablation argument with harder, scene-level evidence.

### 3. ψ cross-domain generalization
**File**: `claudecode/paper_draft/psi_generalization_and_experiment_c.md`, part 1
**Insert**: Limitations and Future Work, after "Disaster-specific adaptation".
- ImageNet-only features generalize to flooding (unseen category) better than
  wildfire-tuned ψ features (ρ=0.511 vs. 0.389 at s=1, using the clean classifier —
  this gap is *larger* than it first appeared with the leaked checkpoint, so fixing
  Phase 0 strengthened rather than weakened this finding).
- Tempers the implicit generalization claim of reusing one wildfire-tuned teacher
  across disaster types.

### 4. Experiment C — severity-distribution alignment by damage bucket
**Same file, part 2**
**Insert**: "Failure Cases and Limitations" or the Discussion after it.
- Extended from flooding-only to the primary wildfire category (was flagged as a
  gap — now fixed).
- Wildfire shows a richer pattern than flooding: destroyed-bucket scenes are
  undershot (Wasserstein distance 2.11 at s=1, worst of all buckets, same direction
  as flooding), but *additionally* no-damage-bucket scenes get pulled toward higher
  severity as s→1 (distance rises 0.33→0.78) — real "no damage" status isn't
  protected at large s. This is independent distributional evidence for the same
  weak-calibration-at-large-s finding from item 1.

---

## Phase 2 — Statistical rigor

### 5. Table 1 per-image paired significance
**File**: `claudecode/paper_draft/table1_significance.md`
**Insert**: footnote under Table 1, or one sentence after its discussion paragraph.
- Good news: Wilcoxon signed-rank test shows RiskSlider's LPIPS/SSIM/PSNR advantage
  over every baseline is significant at p<1e-28. Means reproduce the paper's Table 1
  exactly.
- **CLIP-I/DINO-I paired Wilcoxon test — DONE**: extended the same per-image test to
  CLIP-I/DINO-I. RiskSlider beats ControlNet/Pix2Pix/Palette significantly (p<1e-7),
  same pattern as the structural metrics. **But against CycleGAN specifically, the
  paper's own Table 1 numbers already show RiskSlider trailing on both CLIP-I (0.9281
  vs 0.9312) and DINO-I (0.7986 vs 0.8127) — this test confirms that gap is a real,
  significant reversal (p=0.012, p=2.5e-06), not noise.** Don't claim an unqualified
  sweep on Table 1; CycleGAN is a genuine exception on these two metrics specifically,
  consistent with it also being RiskSlider's closest FID competitor.
- **Palette n=246 fix — RESOLVED**: confirmed the paper's *original* published Table 1
  Palette row was computed on only n=138 (not a repo/analysis artifact — traced to
  the exact original eval log, numbers match the paper exactly). Root cause: Palette's
  test-split generations were split across two disjoint result folders (169 + 77 =
  246, zero overlap) and every downstream script only ever read one of them. No
  retraining or new inference needed — merged both folders
  (`outputs/baselines/palette/test_socalfire_epoch40_merged_246/`) and reran all
  metrics, including CLIP-I/DINO-I (see below) and CAS. LPIPS/SSIM/PSNR/CLIP-I/DINO-I
  shift only slightly; FID shifts more (472.78→434.56, -8% relative, plausibly a
  small-sample bias in the original n=138 estimate); CAS also needed the same n=246
  fix (clean-classifier CAS 0.9882→0.9919 — SUMMARY.md's Phase 0 clean-CAS number for
  Palette was itself still on the incomplete n=138 folder). Palette remains the worst
  method on every metric either way — **replace Table 1's Palette row with the
  corrected n=246 numbers throughout, including CAS**.
- **CLIP-I/DINO-I `nan` for Palette's n=246 recompute — RESOLVED**: root cause was not
  a code bug but a broken HF-checkpoint download path on this machine (mirror
  redirect handling + an `hf_xet` backend hang with no retry on the DINO checkpoint).
  Fixed by downloading both checkpoints directly over plain HTTPS with resume and
  placing them in the standard HF cache layout; all four baselines plus Palette now
  have complete CLIP-I/DINO-I numbers (see `table1_significance.md`).

### 6. λ_smooth (Eq.3) ablation
**File**: `claudecode/paper_draft/lambda_smooth_ablation.md`
**Insert**: Appendix, after "Additional Ablation Analysis".
- Fills the gap that Eq.3's three loss terms (scale/endpoint/smooth) were never
  individually ablated, only whole modules were.
- Honest, non-flattering finding (raw generator only): λ_smooth shows no clean
  monotonic effect on the raw generator's trajectory smoothness across
  λ∈{0, 0.25, 1.0} — CAS non-monotonicity and LPIPS/PSNR ranges don't move
  consistently with regularization strength; production (λ=0.25) is even the *worst*
  of the three on LPIPS range.
- **Full-pipeline check — DONE**: ran all three raw-generator variants through the
  unchanged production refiner (no retraining) and re-evaluated the Table-14-style
  multiscale trend. Result: once refined, all three variants become fully monotonic
  (zero backward steps, vs. local dips in every raw-generator variant) and are
  statistically indistinguishable from each other (spread across variants smaller
  than the paper's own seed-noise floor). **No generator/refiner interaction effect**
  — this closes the caveat and strengthens the recommended framing: credit the
  smooth, non-saturating trajectory in Table 14 entirely to pseudo-target supervision
  and the refiner design, not to λ_smooth, which shows no measurable end-to-end
  effect on the property it's nominally designed for.

---

## Resolved: the "just interpolate between endpoints" rebuttal

**File**: `claudecode/paper_draft/naive_interpolation_rebuttal.md` — DONE, real but
modest effect confirmed.

Built a naive latent-interpolation baseline and pushed it all the way through the
decisive full-pipeline test: VAE-encode real x_pre and RiskSlider's own generated
s=1 endpoint, linearly interpolate in latent space (no conditional UNet denoising),
decode -- then route through the *identical, unmodified* production refiner (f0/fu/f1)
the same way RiskSlider's own generation output is refined at inference time.

- **Generation stage alone**: s-conditioning shows no advantage over naive
  interpolation (both non-significant at s=1: rho=-0.014 vs. -0.027).
- **After the same refiner is applied to both**: RiskSlider's actual pipeline pulls
  ahead at s=1 -- rho=0.388 (p=5.1e-5) vs. naive-interp-refined's rho=0.272
  (p=0.0055); concordance 65.2% (p=3.5e-52) vs. 59.9% (p=7.9e-23). Both reach
  significance, but RiskSlider's is meaningfully stronger (~43% higher correlation,
  5.3 points higher concordance).
- At intermediate scales (0.5/0.7/0.75) the two are statistically indistinguishable
  -- neither shows significant scene-level calibration there, consistent with the
  already-documented intermediate-scale weakness.
- Population-level CAS accuracy trend is nearly identical between the two pipelines
  -- the paper's headline metric cannot detect this difference at all; only the
  scene-level analysis reveals it.

**Recommended paper framing**: claim precisely what the data supports -- s-conditioning
provides additional, measurable scene-specific severity information beyond what
post-hoc interpolation + refinement can recover, at the trained endpoint -- and
explicitly acknowledge this advantage is not yet significant at intermediate
severities, rather than claiming an unqualified, across-the-board win. See the file
for full numbers and caveats (single category tested, single run, and note the
CAS-accuracy numbers for the *unrefined* naive baseline are confounded by an extra
VAE round-trip and should not be used for that specific comparison -- scene-level
Spearman correlation is the trustworthy metric throughout).

## Phase 3 — Calibrated-severity variant (rebuttal to "s isn't needed before generation")

**Files**: `claudecode/paper_draft/calibrated_severity_redesign_plan.md` (design),
`calibrated_severity_final_evaluation.md` (eval plan),
`calibrated_severity_final_evaluation_results.md` (results — READ THIS ONE).

Full pipeline actually retrained end-to-end on socalfire (not just designed): LoRA
generation module (15000 steps, hybrid data scheme — labeled pairs anchor s=1 at
their own S_real/3, unlabeled pairs keep s=1.0 as before), f1 refiner retrained on
the 81 train pairs with S_real>=2.0, new pseudo-targets, fu retrained on the full
calibrated scale sweep. Train+test full inference and refine pipeline completed.

- **Quality**: no regression vs. production (LPIPS 0.217 vs 0.210, SSIM 0.579 vs
  0.584, PSNR 23.8 vs 23.9dB, on the 22 high-severity test pairs where a genuine
  s=1 target exists for both models). Solid, unambiguous result.
- **Core calibration claim** (does s=1 now mean universal max-severity instead of
  "this scene's own fate"): re-tested with the paper's own linear-probe tool
  (`experiment_a/multiscale_correlation.py`, reproduces the known production
  number almost exactly — ρ=0.392 vs. previously reported 0.388, good sanity
  check). Genuinely mixed result: at intermediate scales (0.5/0.7/0.75), calibrated
  shows a real, qualitatively different signature — correlation with real severity
  drops to non-significant (p=0.13-0.61) where production stays significant
  (p≤0.012). But at s=1 itself — the scale the rebuttal is actually about —
  calibrated is still significantly correlated with real severity (ρ=0.300,
  p=0.002), only somewhat weaker than production's 0.392, so NOT fully decoupled.
  New unexplained oddity: calibrated's mean predicted severity is non-monotonic
  across its own scale (rises to a peak at s=0.75, then drops at s=1) unlike
  production's smooth monotonic increase — flagged as an open question, not
  investigated further.
- **Recommendation**: present as a supplementary, exploratory variant (per the
  user's own suggestion), not as a validated fix. Report the quality-preservation
  result plainly, report the intermediate-scale decoupling as a genuine partial
  effect, and explicitly flag both the residual s=1 correlation and the
  non-monotonic trajectory as open limitations. Do not claim this resolves the
  "s doesn't need to be given before generation" reviewer objection outright —
  the evidence doesn't support that framing (see results file for full reasoning).
- **Cross-category replication (flooding)**: the full pipeline was re-run on
  midwest-flooding (17 high-severity train pairs for f1, vs. socalfire's 81 — a
  genuine stress test of small-N). All three findings from socalfire replicated
  almost exactly: intermediate-scale decoupling becomes significant→non-significant
  the same way, s=1 residual correlation persists but weaker (0.267 vs 0.494), and
  the same non-monotonic top-of-range dip appears. This raises the "extrapolation
  gap" diagnosis from single-category anecdote to a reproducible pattern across two
  disaster types, and directly motivates v2 (see
  `calibrated_severity_v2_improvement_plan.md`).
- **v2 fix attempt (severity-probe-guided extrapolation loss), negative result**:
  implemented and trained the diagnosed fix — an explicit loss during v1's
  never-sampled `s > s_i` region, penalizing the frozen severity probe's score on
  the decoded prediction against `s*3` — full pipeline re-run on socalfire. Did
  **not** fix either targeted anomaly: the s=0.75→s=1 mean-severity drop is still
  present and slightly larger in absolute terms (0.821→0.508, vs. v1's
  0.784→0.511), and the s=1 correlation with real severity moved slightly *toward*
  production's fully-coupled number (ρ=0.364 vs. v1's 0.300, CIs overlap so not
  significant) rather than away from it. Intermediate-scale decoupling still holds
  directionally but every point estimate drifted slightly back toward production.
  Bottom line: a well-motivated, correctly-implemented fix for a real diagnosed gap
  that did not work in practice — full numbers and discussion in
  `calibrated_severity_final_evaluation_results.md`. The v1 finding itself is
  unaffected by this negative result.
- **v3 fix attempt (pixel-space monotonicity loss), partial positive result**:
  replaced v2's frozen-classifier probe loss with a direct hinge loss comparing
  two decoded images (anchor at s_i vs. extrapolated s>s_i), penalizing any case
  where the extrapolated image is visually less damaged than the anchor — plus a
  fix to a numerical bug (found via smoke test) where the extrapolation target's
  slope explodes for the ~58% of near-zero-severity pairs, likely also
  undermining v2. Full pipeline re-run on socalfire. Result: the s=0.75→s=1
  mean-severity drop shrank from v1's 0.273 and v2's 0.313 to **0.243** — smaller,
  not eliminated. The s=1 correlation (ρ=0.310) landed almost exactly back at
  v1's 0.300, undoing v2's unwanted drift toward production's 0.392, and the
  intermediate-scale decoupling finding survived with less erosion than v2
  caused. Net: v3 is a genuine, measurable improvement over v2 on every axis
  checked, but does not fully resolve the non-monotonicity anomaly — a smaller
  version of the same open question remains. No quality (LPIPS/SSIM) re-check was
  done for v3. Full numbers in `calibrated_severity_final_evaluation_results.md`.

---

## What's NOT done yet (explicit open items)

1. Everything else from the original priority list not covered above (writing
   polish, related-work defensive framing, anonymity check across the full 21-page
   PDF including the appendix) is still pending and lower-priority.
2. **Page-budget triage not done**: the main text appears to be ~6-7 pages before
   any of Phase 1's new main-text insertions (Scene-level Calibration section, ψ
   generalization paragraph, Experiment C discussion, Table 1 significance footnote).
   Nobody has decided what actually fits in a fixed AAAI page limit vs. gets pushed
   to the appendix.
3. **Reproducibility checklist** (AAAI requires one) has not been started.
4. None of the seven markdown files' findings have actually been merged into the
   real LaTeX source yet — see next steps.

## Suggested next steps, in order
1. Decide what belongs in the main text vs. appendix given the page budget (item 2
   above) — this is a judgment call only you can make.
2. Draft the actual LaTeX edits into your real paper source using the prose in
   these seven files as source material — I don't have access to that source, only
   the compiled PDF, so this step needs to happen in your own editor.
3. Do a full-document (not just page-1) anonymity pass and fill in the
   reproducibility checklist before submission.

## Resolved since last pass (2026-07-22)
- **λ_smooth full-pipeline check** (item 6): done, no interaction effect found —
  see item 6 above.
- **CLIP-I/DINO-I Wilcoxon test** (item 5): done — see item 5 above, including the
  honest CycleGAN exception.
- **Palette CLIP-I/DINO-I `nan`** (item 5): root-caused and fixed (broken HF
  checkpoint download path on this machine, not a code bug) — see item 5 above.
