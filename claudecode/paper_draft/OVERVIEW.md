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
- **🔴 NEW, 2026-07-24, NOT actually applied to the PDF — see `table1_significance.md`
  top section for full detail**: the "all mechanical CAS recomputes complete" claim
  above is true of the *documentation* (SUMMARY.md's Leaked/Clean tables exist and are
  correct) but was never carried through into Table 1's actual published numbers.
  Server-side audit found Table 1's current CAS column mixes **three different
  classifier checkpoints** across its five rows: the four baselines trace to the
  leaked checkpoint (`socalfire_cls_20260421_201243`), while RiskSlider's row
  (published 0.9939) traces to yet another, separately-trained checkpoint
  (`_shared_real_classifier`, from an unrelated cross-domain-CAS experiment) that
  Phase 0 never touched. I re-verified directly: scoring the actual production
  RiskSlider images with the confirmed leak-free classifier gives **0.9878**, not
  0.9939. Replacing the whole column with one consistent classifier
  (`socalfire_cls_clean_split`) gives: ControlNet 0.9939, Palette 0.9919, Pix2Pix
  0.9898, **RiskSlider 0.9878**, CycleGAN 0.9837 — RiskSlider drops from
  looking tied-for-best to **4th of 5**. This is a bigger, more urgent fix than
  anything else in this file — do this before any other Table 1 edit.
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
- **🔴 NEW, 2026-07-24, Table 2 / `tab:app_ablation_multiscale` also NOT applied to
  the PDF** — see `table2_ablation_classifier_mismatch_20260724.md`. Same
  `_shared_real_classifier` third-checkpoint problem as Table 1, independently
  contaminating the whole ablation table (all 4 variants, largest effect on
  "w/o Refinement": 0.7093→0.6850 at s=1). Same fix: replace with `table14_clean`.

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
- **New, 2026-07-24**: filled in the previously-missing LPIPS/DINO-I cells (a C2-style
  gap from the reviewer checklist) by re-evaluating the already-generated naive-interp
  images with the confirmed leak-free classifier -- consolidated three-way table
  (RiskSlider / naive-interp / naive-interp+refiner) at s=.25/.5/.75 now in
  `naive_interpolation_rebuttal.md`. LPIPS/DINO-I are essentially tied between
  RiskSlider-actual and naive-interp+refiner at every scale (2nd-3rd decimal), and
  both refined variants show a much steeper CAS jump (0.25->0.5: +0.17/+0.19) than the
  unrefined naive baseline (+0.03) -- reinforcing that the refiner, not s-conditioning,
  drives the population-level trajectory shape; s-conditioning's one measurable edge
  remains scene-level correlation at s=1 specifically.
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

## Server-side audit, 2026-07-24 (new)

**File**: `claudecode/paper_draft/server_audit_20260724.md` (dataset/result/protocol/
statistics, all clean) + `table1_significance.md` top section (the one real finding).

- 🔴 **Highest-priority finding, do this before any other Table 1 edit**: Table 1's
  CAS column currently mixes three different classifier checkpoints across its five
  rows (the four baselines use the leaked checkpoint; RiskSlider's published 0.9939
  uses yet another, unrelated checkpoint that Phase 0 never audited). The correct,
  single-classifier column (re-verified directly): ControlNet 0.9939, Palette 0.9919,
  Pix2Pix 0.9898, **RiskSlider 0.9878**, CycleGAN 0.9837 — RiskSlider drops from
  looking tied-for-best to 4th of 5. Full detail in `table1_significance.md`.
- Also added: a real-image CAS reference point (real post-disaster photos are
  classified "post" 244/246=0.9919 of the time by the same clean classifier — the
  practical ceiling for this metric) to contextualize how tight the 4th-of-5 gap
  actually is.
- Dataset, Table-1-provenance (minus CAS), training-protocol, and statistical-test
  audits all came back clean — no other issues found. Confirmed via md5 hashing that
  socalfire's train/test split is tile-disjoint by construction (inherits xBD's own
  official partition), and confirmed each baseline's pretrained-vs-scratch status for
  a fairness disclosure the paper should probably state explicitly.
- Also confirmed (separate, lower-urgency, not yet acted on): SUMMARY.md's
  recommendation to swap in a clean-ψ-trained refiner as the production f1 checkpoint
  was never actually applied — production still uses the original refiner. This is
  about *which refiner generates* the images, distinct from the CAS fix above (which
  is about *which classifier scores* them).
- **Dense-s continuity check (C1) — DONE, clean negative result**. File:
  `claudecode/paper_draft/dense_s_continuity.md`. Generated 14 new scale points
  (densest right at both boundaries: 0.01, 0.02, 0.05, ..., 0.95, 0.98, 0.99) with the
  unchanged production checkpoints, combined with the 7 existing points for 21 total,
  and checked adjacent-scale LPIPS/DINO-distance/ΔP(post)/residual-jump across all
  246 test scenes. **No discontinuity at either s=0/0.01 or s=0.99/1.00** — both
  boundaries are among the *smoothest* transitions in the whole trajectory, not the
  roughest; even after normalizing by step size, the boundary "slopes" are 10-15x
  gentler than the steepest region, which sits in the interior (s≈0.3-0.7, a normal
  sigmoid-shaped classifier-probability transition). Consistent with the code-level
  fact that neither the generator nor the refiner branches discretely on s anywhere —
  both treat it as one continuous conditioning scalar.
- **Naive-interpolation three-way table (C2 gap) — DONE**. File:
  `naive_interpolation_rebuttal.md`, new "Consolidated three-way comparison table"
  section. Filled in previously-missing LPIPS/DINO-I cells at s=.25/.5/.75 for
  RiskSlider / naive-interp / naive-interp+refiner using already-generated images, no
  new generation needed. LPIPS/DINO-I are essentially tied between RiskSlider-actual
  and naive-interp+refiner at every scale checked; both refined variants show a much
  steeper CAS jump between s=0.25→0.5 than the unrefined naive baseline, reinforcing
  that the refiner (not s-conditioning) drives the population-level trajectory shape.

## Appendix cross-table discrepancy, 2026-07-24 (new, root-caused)

**File**: `claudecode/paper_draft/appendix_table_consistency_20260724.md`.

User found tab:app_refined_scale and tab:app_ablation_multiscale's "Ours" row
disagree by 0.06-0.07 at every intermediate scale (only s=1 matches). Root cause
found: **these two tables trace to two entirely different model runs**, not a
classifier issue this time. tab:app_refined_scale exactly matches an April eval of
`outputs/refine/socalfire` (an older, no-longer-existing-on-disk refiner iteration);
tab:app_ablation_multiscale's "Ours" row matches the current production `refine-2`
pipeline (`eval-full`, same source as Table 1's stale CAS). The old run's images are
gone from this machine — cannot be reconciled further, only standardized going
forward. **Fix: replace both tables' "Ours" row with the same, single source** —
`table8_9_refined_socalfire.csv` / `table14_clean` (Phase 0's clean-classifier
recompute on the current `refine-2` pipeline, confirmed internally consistent with
each other): 0.25→0.7378, 0.5→0.9085, 0.75→0.9837, 1→0.9878. Same fix pattern as the
Table 1 CAS finding — standardize on the current, reproducible pipeline + the
confirmed leak-free classifier, everywhere "Ours"/RiskSlider CAS is cited.

## Table 2 / appendix ablation CAS mismatch, 2026-07-24 (new, root-caused)

**File**: `claudecode/paper_draft/table2_ablation_classifier_mismatch_20260724.md`.

User found `tab:app_continuous_generation_scale` (CAS=0.6850, unrefined generator,
s=1) disagrees with Table 2 / `tab:app_ablation_multiscale`'s "w/o Refinement" row
(CAS=0.7093) even though both describe the identical images and every other metric
(LPIPS/SSIM/PSNR/FID/CLIP-I/DINO-I) matches exactly. Root cause: **the same third
classifier checkpoint already flagged in the Table 1 CAS finding**
(`_shared_real_classifier`, from the unrelated `cas_cross_domain` experiment) also
scored the entire Table 2 / `tab:app_ablation_multiscale` table
(`outputs/eval/ablation/ablation-compare/`, dated 2026-05-03), not just Table 1's
RiskSlider row. All four ablation variants are affected to varying degrees (largest
on "w/o Refinement", -0.0244 at s=1). **Fix: replace Table 2's and
`tab:app_ablation_multiscale`'s CAS columns with `table14_clean`'s numbers** — no
new compute needed, same fix pattern as the Table 1 CAS finding and the
`appendix_table_consistency_20260724.md` finding. One open item: a full sweep for
any other table still citing `_shared_real_classifier`-scored CAS has not been done.

## Cross-category appendix tables CAS mismatch, 2026-07-24 (new, root-caused — third instance of the same bug family)

**File**: `claudecode/paper_draft/appendix_cross_category_tables_20260724.md`.

User read the full `appendices/03_experimental_results.tex` for the first time this
pass and found a third instance: `tab:app_cross_category_s1` and
`tab:app_cross_category_cas`'s "Wildfire (primary)" rows are both wrong, and wrong
for two *different* reasons layered together.
`tab:app_cross_category_s1`'s Wildfire CAS (0.9939) is the same
`_shared_real_classifier` bug as Table 1/2 — trivial fix, swap to 0.9878, no new
compute. `tab:app_cross_category_cas`'s whole Wildfire row (0.618/0.720/0.880/0.976/0.994)
is worse: it traces to the *old, pre-`refine-2` generation run* (already flagged in
`appendix_table_consistency_20260724.md`) scored by the *original* leaked classifier
(`socalfire_cls_20260421_201243`, the one used for Table 1's baselines) — the only
classifier that existed on this machine on 2026-04-22 when that eval ran. Checked the
other four rows (Hurricane/Flooding/Santa Rosa/Volcano, supplied by the user from the
actual tex text) against the two existing Table 10/11 recomputes: they match the
per-category **leaked** classifier numbers (`table10_11_percategory_LEAKED_verify.csv`)
almost exactly (Flooding/Volcano exact, Hurricane/Santa Rosa off only at s=0.75 by the
same already-documented seed-variance amount) — confirming those four rows are
genuinely current-pipeline Table 10/11 data, and structurally *cannot* share a run
with the Wildfire row (the other four categories' `refine-2` pipelines didn't exist
yet in April when Wildfire's row was computed). **Fix: replace the entire
`tab:app_cross_category_cas` table with `table10_11_percategory_CLEAN.csv`** (already
computed, no new compute needed) and swap `tab:app_cross_category_s1`'s CAS column
the same way. Side finding: corrected an inaccurate claim in
`appendix_table_consistency_20260724.md` that the old run's images were unrecoverable
— they survive, renamed to `outputs/refine-1/socalfire`, verified bit-for-bit via a
fresh LPIPS/SSIM/PSNR rerun (used `cuda:1` to avoid the training job running on
`cuda:0` at the time).

**Follow-up, same day**: user cross-checked the recommended fix values against
`tab:app_cross_category_s1` (the endpoint table) and found its own
Hurricane/Flooding/Santa Rosa/Volcano CAS values don't match the clean numbers either
(Volcano especially: 1.0 vs 0.5, a direction-flipping gap) — asked whether this was a
third, uninvestigated source before touching either table. Checked: it isn't — those
four endpoint-table values trace exactly to the same per-category leaked
`table10_11_percategory_LEAKED_verify.csv` already identified as the trend table's
source (re-confirmed against the underlying `eval-full`/`eval-full-rerun` csvs for
Hurricane and Santa Rosa). It's the same leaked-vs-clean gap being fixed everywhere
else, not a new inconsistency — both tables fix cleanly from the single
`table10_11_percategory_CLEAN.csv` source with nothing left over.

## Seed-stability table CAS mismatch, 2026-07-24 (new, root-caused — 4th instance, a genuinely new checkpoint)

**File**: `claudecode/paper_draft/appendix_seed_stability_20260724.md`.

User spotted `tab:app_seed_stability` (3 seeds, full pipeline, s=1): CAS consistently
~0.008-0.010 *above* the cross-referenced clean baseline (0.9878) while
LPIPS/SSIM/PSNR sit consistently *below* it — the signature of a different classifier
scoring the same images, not seed noise, but wanted it confirmed before touching the
table since the evidence was less direct than the first three instances. Confirmed:
re-scored the still-on-disk seed images (`outputs/stability/socalfire_seed{42,43,44}/refined/test/scale1`)
against four candidate checkpoints on `cuda:1` and got exact digit-for-digit matches
to the published numbers with `output-models/classifier/socalfire_cls_real_fresh/best.pt`
— a **4th distinct leaked checkpoint**, different from all three already found
(`socalfire_cls_20260421_201243`, `_shared_real_classifier`,
`socalfire_cls_clean_split`). Confirmed leaked via the repo's own
`train_classifier_clean_split.py` docstring and `recompute_table1_cas.py`'s explicit
`"leaked (original, test_acc-selected)"` label — this checkpoint is also, separately,
the semantic teacher ψ used to train the production refiner (already a known,
lower-urgency open item). Also ruled out the Cross-Category-style compounding bug
(stale generation run): direct LPIPS check confirms the seed images are
pipeline-consistent with current production `refine-2` (mean LPIPS 0.0100 vs. 0.0459
against the old, superseded `refine-1` pipeline) — this table's LPIPS/SSIM/PSNR need
no change. **Fix: replace only the CAS column** — clean-classifier rescore already
done, no new generation needed: 0.9878 / 0.9858 / 0.9878 (mean 0.9871±0.0012), saved
to `claudecode/result/classifier_fix/seed_stability_clean/seed_stability_clean_cas.csv`.

## Second batch of reviewer-checklist items, 2026-07-24

Five items requested; three done (pure eval / cheap generation), two remain (need
new segmentation model / new training, see next section).

- **Effect sizes + Holm correction (done)**. File: `table1_significance.md`, "Effect
  sizes + Holm correction" section. All 20 per-image Wilcoxon tests (4 baselines x
  LPIPS/SSIM/PSNR/CLIP-I/DINO-I) remain significant after Holm-Bonferroni correction
  (worst case p_holm=1.18e-2). Effect sizes (rank-biserial r) are near-maximal
  (|r|>0.98) for structural metrics against every baseline; CLIP-I/DINO-I effect
  sizes are smaller and, for CycleGAN specifically, favor CycleGAN (r=-0.19/-0.35).
- **FID/KID bootstrap CIs (done)**. File: `table1_significance.md`, "FID/KID
  bootstrap confidence intervals" section. Caught and fixed a real bug in my own
  first attempt (inconsistent PCA bases across a paired comparison, producing a
  sign-inconsistent result for CycleGAN) before trusting or reporting it. Corrected
  result: **RiskSlider's FID/KID edge over CycleGAN specifically is not
  statistically significant** (paired-diff 95% CI straddles zero for both metrics) --
  significant and decisive against the other three baselines. Recommend not claiming
  an FID/KID win over CycleGAN specifically.
- **Clean naive-interpolation rerun, no VAE round-trip confound (done) — reverses a
  previous conclusion**. File: `naive_interpolation_rebuttal.md`, new "SUPERSEDING
  UPDATE" section (the old "Honest conclusion" below it is now marked superseded,
  not deleted, for the record). The original naive-interp baseline unnecessarily
  VAE-round-tripped the s=0/s=1 endpoints even though those exactly equal existing
  images with no interpolation needed. Reran with exact endpoints (no VAE pass) and
  only genuinely-interpolated intermediate scales going through the VAE. **Result:
  the previously-reported s=1 scene-level-correlation advantage for RiskSlider's
  actual pipeline (0.388 vs. naive-interp+refiner's 0.272) disappears entirely** --
  the clean naive-interp+refiner number is 0.388 (p=5.12e-5), statistically identical
  to RiskSlider's own 0.388 (p=5.1e-5). This directly answers the paper's "we did not
  have the opportunity to rerun before submission" sentence (appendices/06 line 63):
  we did rerun it, and the s-conditioning advantage this sentence was hedging around
  does not survive. **Recommend removing the claim that s-conditioning provides
  measurable scene-level calibration advantage over naive interpolation** -- on this
  evidence it does not, anywhere on the tested scale range.

## Structural consistency + SDEdit baseline — DONE, 2026-07-24

**File**: `claudecode/paper_draft/structural_consistency_and_sdedit_20260724.md`.

- **Structural consistency (building footprint IoU)**: found xBD's own building
  polygon labels, trained a lightweight U-Net building segmenter from scratch (no
  pretrained segmentation package available on this machine), matched it to the
  anonymized test set via md5 hashing, and computed IoU against ground truth on a
  fixed, method-independent subset (n=108/246 scenes with >=1 real building).
  **Caught and fixed a real rasterization bug** (coordinates scaled before drawing
  on a still-full-resolution canvas, squeezing all buildings into one corner) --
  visually verified against source imagery before trusting the masks. **Result:
  RiskSlider (0.289) is essentially at parity with real post-disaster photos
  (0.282)** -- clearly better than ControlNet/Pix2Pix/Palette/SDEdit. **CycleGAN is
  again the one exception** (0.364, higher than even the real-image ceiling) --
  another data point for the already-established "CycleGAN is RiskSlider's one
  genuine structural-fidelity-adjacent competitor" framing.
- **SDEdit baseline (training-free)**: partial-noise + denoise with the plain
  pretrained SD v1.4 backbone (no LoRA, no learned conditioning), one endpoint-style
  comparison. **Result: dramatically worse than every trained baseline** on
  LPIPS/SSIM/PSNR/FID/CLIP-I/DINO-I (e.g. FID 651.55 vs. Palette's 434.47, the
  current Table 1 worst) and equally poor on structural IoU. Directly answers "maybe
  you don't need to fine-tune at all" -- on this evidence, fine-tuning does real,
  necessary work.

## Same-backbone-no-progression-conditioning control — DONE, 2026-07-24 (honest negative result)

**File**: `claudecode/paper_draft/same_backbone_control_20260724.md`.

Trained a new LoRA from scratch, identical to the production socalfire recipe in
every hyperparameter (backbone, rank=16, alpha=16.0, 15,000 steps, lr, loss weights),
except forcing the scale-sampling to draw only `s in {0,1}` for the entire run
(confirmed in the training log -- never an intermediate value) -- i.e. the same
backbone, fine-tuned with the same recipe, but never given a progression/severity
training signal. Generated the test-set endpoint, refined through the unchanged
production refiner, evaluated with the same Table 1 metrics.

**Result: no measurable difference from RiskSlider's actual endpoint performance on
any metric** (LPIPS/SSIM/PSNR/FID/CLIP-I/DINO-I/structural-IoU all within noise; CAS
is marginally *higher* for the control, 0.9898 vs. 0.9878). **Progression-conditioned
training provides no measurable endpoint-quality advantage over a much simpler
endpoint-only fine-tune of the same backbone** -- Table 1's endpoint numbers reflect
that fine-tuning happened at all, not the specific progression-conditioning
methodology. Consistent with, and reinforcing, the clean naive-interpolation
reversal finding above. **Important scope limit**: this control was never trained on
intermediate severities and cannot produce a meaningful continuous slider -- this
result is about endpoint quality only, and says nothing about whether
progression-conditioning is needed for the continuous-control capability itself
(a separate claim, argued elsewhere via scene-level calibration / monotonicity
results, not undermined by this experiment). **Recommended framing**: don't credit
progression-conditioning for endpoint image quality; that credit belongs to
backbone fine-tuning in general (also consistent with the SDEdit finding that some
form of fine-tuning is clearly necessary, just not specifically this form).

## Negative-controls quantitative table (mean diff/CI/p/effect size) — DONE, 2026-07-25

**File**: `claudecode/paper_draft/negative_controls_quantitative_table_20260725.md`.
Data: `claudecode/result/classifier_fix/negative_controls_effect_sizes.csv`.

Reviewer wanted a quantitative table (mean diff, CI, p/effect size) for the two
negative controls, appendix-only per the page-budget decision. Built it by adapting
`table1_effect_sizes.py`'s paired-Wilcoxon + rank-biserial-r + Holm-correction
methodology, plus a new paired bootstrap 95% CI, re-scoring images that already
existed on disk (no new generation). **Caught one subtlety before running it**:
naive-interpolation's own v2clean design makes s=1 definitionally identical to
RiskSlider's actual pipeline (confirmed empirically: diff=0.00000, p=1.0 there) — used
the three intermediate scales (0.25/0.5/0.75) instead for that comparison, and s=1
(its only trained point) for the no-progression control. **Result for both**: several
differences reach statistical significance at n=246 (large-n effect), but all effect
sizes are small (rank-biserial r ≤ 0.61) and absolute magnitudes are 2-4 orders of
magnitude smaller than any Table 1 baseline gap — quantitatively confirms both
documents' existing qualitative "indistinguishable at the population level" claims.

## appendices/04 CAS-labeled figure (`cas_lpips_vs_scale_test.png`) source — DONE, 2026-07-25

**File**: `claudecode/paper_draft/cas_figure_appendix04_source_20260725.md`.

Both blockers reported (no plotting script, no mid-scale "w/o Pseudo Supervision"
data) turned out to be resolved already: script is `eval/cas-lpips-scale.py`, default
data path `outputs/eval/ablation/socalfire_unified_no_pseudo_20260422_103055/socalfire-infered_metrics.json`
has full 7-scale rows. CAS→rename is a one-line label edit + rerun, no new compute.
**New caveat found along the way**: the script's default "Ours" curve is the old
refine-1 pipeline (not current refine-2), and both curves almost certainly rest on
the original leaked-selection classifier (`socalfire_cls_20260421_201243` — the only
classifier that existed as of the figure's April 2026 generation dates) — same bug
class as Table 2/cross-category, flagged but not fixed (separate decision from the
rename, would change the curve shape not just the label).

## s=0 vs. x_pre gap — root cause found, code-only fix available, NOT applied — 2026-07-25

**File**: `claudecode/paper_draft/s0_pre_disaster_gap_20260725.md`.

`infer/batch_infer_xbd_paired_scale01.py:299-323` runs s=0 through a full
noise-then-denoise diffusion pass instead of special-casing it as an identity
passthrough of x_pre — even though the paper's own severity definition makes s=0's
ground truth trivially known. A code-only special case (skip the model entirely for
s=0, save x_pre directly) is feasible without any retraining and without touching
`cuda:0`. **Not implemented** — regenerating scale0 would ripple into every table
citing s=0 metrics, and the exact intended behavior (pixel-identical copy vs. one VAE
round-trip, and what the paper's cited "1.2%" floor actually measures) needs
confirmation first.

## Headline LPIPS (Table 1) vs. seed-stability mean — real gap, root cause found, NOT reconciled — 2026-07-25

**File**: `claudecode/paper_draft/headline_vs_seed_stability_gap_20260725.md`.

Different bug class from every classifier-leak finding above — this one is a
base-generation-stage environment/checkpoint provenance issue, not a classifier
mix-up. Headline 0.2166 traces to `outputs/refine-2/socalfire/test/scale1`
(genuine production output, base images generated 2026-04-21). The seed-stability
sweep's base images were freshly regenerated 2026-05-04, in an environment where
xFormers' compiled CUDA extensions had broken (confirmed: log warns of a
PyTorch/CUDA version mismatch; current environment shows the identical warning
today) — direct LPIPS distance between the two nominally-"seed=42" base-image sets
is 0.0771, far too large for a true identical-seed rerun. Separately, the exact LoRA
checkpoint file the headline script references no longer exists on disk (likely
renamed in a later reorg, unconfirmed byte-for-byte). **No clean substitute data
exists to swap in** (unlike the CAS-leak fixes) — reconciling this for real requires
regenerating one of the two runs in a shared environment, which is a visible change
to load-bearing numbers and needs sign-off before I touch anything. Recommended
minimum fix: disclose the environment-drift caveat in the seed-stability section.
All checking done on `cuda:5`; `cuda:0` untouched.
