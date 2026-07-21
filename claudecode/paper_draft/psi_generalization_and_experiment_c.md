# ψ cross-domain generalization + Experiment C — draft inserts

## 1. ψ cross-domain generalization — insert into "Limitations and Future Work",
   right after the "Disaster-specific adaptation" paragraph

Draft prose:

Semantic teacher generalization. RiskSlider's semantic teacher psi (Eq. 6) is
pretrained on SoCal wildfire imagery, the same category as the primary evaluation
benchmark. To check whether this category-specific fine-tuning transfers to other
disaster types, we compare psi's features (using the leak-free checkpoint confirmed
in our classifier-selection audit, Appendix Table [C]) against generic ImageNet
features as the backbone for the scene-level severity probe of Appendix Table [X]
(Section "Scene-level Severity Calibration"), evaluated on the flooding category --
i.e., a domain psi never saw during its own training. On refined outputs at s=1,
the ImageNet-only probe achieves a clearly stronger correlation with real severity
(rho=0.511, p=3.8e-6) than the wildfire-tuned psi features (rho=0.389, p=6.6e-4);
the gap holds, and grows, across intermediate scales too (e.g., s=0.5: ImageNet
rho=0.381 vs. wildfire-tuned rho=0.308; s=0.75: 0.454 vs. 0.380; full numbers in
Appendix Table [X]). This suggests that fine-tuning psi on wildfire imagery does
not improve, and mildly trades off against, its ability to read out severity on an
unseen disaster category, relative to generic pretrained features. Because psi is
only used as a training-time supervisory signal for the post-disaster endpoint
refiner f1 within each category-specific model (Section "Semantic-Aware Residual
Refinement"), this does not affect the paper's within-category results, but it does
temper the generalization claim implicit in reusing a single semantic teacher
design across disaster types, and suggests that a future unified multi-disaster
model (discussed above) may benefit from either a disaster-agnostic teacher or
teachers specific to each disaster type rather than one wildfire-tuned teacher
reused everywhere.

## 2. Experiment C (severity-distribution alignment by real damage bucket) —
   insert into "Failure Cases and Limitations" / "Dataset-dependent Failure Modes",
   or as a new short paragraph in "Discussion" (the one after Failure Cases)

Context: Experiment C buckets flooding test scenes by their real xBD damage label
(no-damage / minor / major / destroyed) and computes the Wasserstein distance between
each bucket's real severity distribution and RiskSlider's generated severity
distribution Shat_gen(s), plus a bucket-conditional mean cosine similarity between
generated features and each bucket's real feature centroid.

Key numbers, refined outputs, Wasserstein distance between Shat_gen(s) and each
bucket's real S_real distribution (full data in
`claudecode/result/experiment_c/severity_distribution_wasserstein_results.csv` for
flooding and `claudecode/result/socalfire/experiment_c/severity_distribution_wasserstein_results.csv`
for the primary SoCal wildfire category):

Flooding (pilot), s=1: no-damage 0.273, minor 0.491, major 1.683, destroyed 2.631 --
monotonically increasing with real damage extent, i.e. the model undershoots
severity most for the most heavily damaged scenes.

SoCal Wildfire (primary) shows a richer, non-monotonic pattern across s:

| bucket | s=0 | s=0.5 | s=1 |
|---|---|---|---|
| no-damage | 0.327 | 0.534 | 0.776 |
| minor | 0.578 | 0.348 | 0.240 |
| major | 1.611 | 1.370 | 1.118 |
| destroyed | 2.603 | 2.362 | 2.110 |

For minor/major/destroyed, distance to the real bucket distribution *decreases*
monotonically as s -> 1 (as intended: generated severity approaches the real
distribution of more heavily damaged scenes), but destroyed remains the worst-
matched bucket throughout (2.11 at s=1, still far larger than any other bucket) --
the same undershoot-at-the-extreme pattern seen in flooding. For no-damage,
distance *increases* as s -> 1 (0.327 -> 0.776): scenes that are genuinely
undamaged in reality are still pushed toward higher generated severity as s
increases, i.e. their real "no damage" status is not preserved at large s. This is
the same failure mode already identified from the population-level side in
Section "Scene-level Severity Calibration" -- weak scene-level calibration at
large s means genuinely low-severity scenes are not protected from the overall
upward pull of s -- now visible from the distributional side as well.

Draft prose:

Severity-distribution alignment by real damage extent. Beyond scene-level rank
correlation (Section "Scene-level Severity Calibration"), we check whether
RiskSlider's generated severity distribution matches the real distribution
conditioned on xBD's four official damage buckets (no-damage/minor/major/destroyed).
On the primary SoCal wildfire benchmark, the Wasserstein distance between generated
and real severity distributions decreases with s for the minor/major/destroyed
buckets, as intended, but destroyed remains the worst-matched bucket at every scale
(2.11 at s=1) -- RiskSlider systematically undershoots severity for the most
heavily damaged real scenes, consistent with the same pattern on the secondary
flooding category (Appendix Table [W]). Conversely, the no-damage bucket's distance
*increases* with s (0.33 at s=0 to 0.78 at s=1): scenes with no real damage are
still pulled toward higher generated severity as s grows, indicating their true
"no damage" status is not protected at large s. Both observations are consistent
with, and provide independent distributional evidence for, the weak scene-level
calibration at large s already identified in Section "Scene-level Severity
Calibration": the model's severity output at a given s is influenced more by s
itself than by each scene's real damage extent. We view closing this gap -- e.g.,
via stronger supervision at the tail of the severity distribution, explicit
oversampling of high-severity training pairs, or a scene-conditioned floor/ceiling
constraint that ties Shat_gen(s) more tightly to each scene's own real endpoints --
as a concrete direction for future work, complementing the auxiliary damage-label
supervision already discussed above.

## Notes
- Both inserts are self-contained and can be dropped in independently of the
  "Scene-level Severity Calibration" section, though the psi paragraph references
  its Appendix table and should be placed after it, or the cross-reference removed.
- Experiment C was only run for the flooding pilot category; if time allows, rerun
  for socalfire (primary) before citing this as more than a secondary-category
  observation -- currently this section would read as evidence from a non-headline
  category only, which a reviewer may discount somewhat.
