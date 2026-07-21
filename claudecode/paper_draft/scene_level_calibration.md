# Scene-level Severity Calibration — draft insert

Insert location: after "Ablation Study", before "User Study" (main text);
appendix table goes after "Additional Ablation Analysis".

## Main text (draft prose)

Scene-level Severity Calibration
The multi-scale CAS trends in Figure 6 and Table 14 establish a
population-level property: averaged over the test set, post-disaster
semantics strengthen monotonically as s -> 1. This does not by itself
establish a stronger, scene-level property that continuous severity
control implicitly promises: for a fixed s, does a scene with greater
real damage receive a higher predicted severity than a scene with less
real damage? A model could satisfy the population-level trend while
applying a generic "post-disaster template" uniformly across scenes,
irrespective of each scene's true severity -- the population average
would still rise with s even if s carries no scene-specific information.

To test this directly, we use xBD's building-level damage annotations
to compute a real, continuous severity score S_real in [0,3] for each
scene (no-damage/minor/major/destroyed proportions weighted by affected
building count), and fit a linear ridge probe on frozen backbone
features of REAL training images only (5-fold CV for regularization
strength; the probe never sees generated images or the test split during
fitting). We then apply this frozen probe to RiskSlider's generated
test images at each scale to obtain Shat_gen(s), and report (i) the
Spearman correlation between Shat_gen(s) and each scene's own S_real,
and (ii) a calibration-free complement: splitting test scenes into
S_real=0 ("no visible damage") and S_real>0 ("visible damage") groups
and measuring the rank concordance rate of Shat_gen(s) between all
low/high pairs, with a one-sided binomial test against chance (0.5).

Table [X] reports results on the primary SoCal wildfire benchmark
(n=103 test scenes with reliable S_real labels) using an ImageNet-only
probe, i.e., features that never saw any xBD training signal, so the
result is not confounded by circularity with the generation pipeline.
Before refinement, Shat_gen(s) shows no significant scene-level
correlation with S_real at any scale (rho in [-0.06, 0.02], p > 0.6
throughout), and concordance between high- and low-damage scenes is at
or below chance (42-44%). After refinement, scene-level correlation
remains non-significant at intermediate scales (s=0.5: rho=0.015,
p=0.88; s=0.75: rho=0.116, p=0.24) and only becomes significant at the
trained endpoint (s=1: rho=0.388, p=5.1e-5; concordance=65.2%,
p=3.5e-52). A secondary flooding category, where inundation produces a
visually salient color shift, shows a more complete picture: refined
correlation is already significant from s=0.25 onward and reaches
rho=0.511 (p=3.8e-6) at s=1 (Appendix Table [Y]).

We interpret this pattern as follows. Real, scene-specific severity
information enters training only at the two endpoints, through
lambda_endpoint's direct supervision toward x_pre/x_post; the
intermediate pseudo-target y_u(s)=D(z_s) (Eq. 5) is constructed by
interpolating each scene's own two endpoints and therefore carries no
signal about how one scene's severity compares to another's. RiskSlider
is consequently well-calibrated to real damage extent at the endpoints
and, for disaster types with visually subtle intermediate damage (e.g.,
wildfire), behaves closer to a shared intensity trajectory than a
scene-calibrated one at intermediate s. This is consistent with the
disaster-dependent behavior already noted in the Failure Cases section
(flooding's salient inundation vs. wildfire's subtler texture change)
and the category-level user-study gap (Table 15: flooding 4.45 vs.
wildfire 3.43 progression scores). We view scene-level calibration at
intermediate severities -- which would require a cross-scene training
signal that current endpoint-only paired supervision does not provide
-- as an open problem rather than one resolved by this work, and discuss
candidate directions (e.g., auxiliary damage-extent labels, relative
ranking losses across scenes) in Limitations. Interestingly, an ablation
cross-check (Appendix Table [Z]) shows this is not simply a matter of
adding back more supervision: the pseudo-target supervision that
prevents premature saturation of the *absolute* severity trajectory
(Table 14) trades off against, rather than improves, *relative*
scene-level discrimination at intermediate scales -- see Appendix for
the full analysis.

## Main-text condensed table (s = 0, 0.5, 0.75, 1; ImageNet probe; SoCal wildfire)

| s | rho, unrefined | rho, refined | concordance, unrefined | concordance, refined |
|---|---|---|---|---|
| 0 | -0.040 (p=0.69) | -0.099 (p=0.32) | 0.440 (p~1) | 0.384 (p~1) |
| 0.5 | -0.022 (p=0.83) | +0.015 (p=0.88) | 0.435 (p~1) | 0.438 (p~1) |
| 0.75 | -0.021 (p=0.83) | +0.116 (p=0.24) | 0.427 (p~1) | 0.497 (p=0.62) |
| 1 | -0.014 (p=0.89) | +0.388 (p=5.1e-5) | 0.426 (p~1) | 0.652 (p=3.5e-52) |

## Appendix — full table (both categories, both probes, all 7 scale points)

```latex
% SoCal Wildfire (primary), n=103 test scenes with reliable S_real
% probe = frozen linear ridge on backbone features, fit on REAL train images only
\begin{table*}[t]
\centering
\small
\begin{tabular}{l cccc cccc}
\toprule
& \multicolumn{4}{c}{Spearman $\rho$ vs. $S_{real}$ (p-value)} & \multicolumn{4}{c}{Concordance rate (p-value)} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9}
Scale $s$ & ImageNet, unref. & ImageNet, ref. & Wildfire-tuned, unref. & Wildfire-tuned, ref.
          & ImageNet, unref. & ImageNet, ref. & Wildfire-tuned, unref. & Wildfire-tuned, ref. \\
\midrule
0.00 & -0.040 (6.9e-1) & -0.099 (3.2e-1) & +0.067 (5.0e-1) & +0.202 (4.1e-2) & 0.440 (1.0) & 0.384 (1.0) & 0.514 (8.1e-2) & 0.570 (2.4e-12) \\
0.25 & -0.064 (5.2e-1) & -0.055 (5.8e-1) & +0.044 (6.6e-1) & +0.193 (5.1e-2) & 0.427 (1.0) & 0.401 (1.0) & 0.516 (5.5e-2) & 0.557 (7.9e-9) \\
0.30 & -0.046 (6.5e-1) & -0.040 (6.9e-1) & +0.027 (7.9e-1) & +0.187 (5.8e-2) & 0.436 (1.0) & 0.410 (1.0) & 0.511 (1.4e-1) & 0.556 (2.0e-8) \\
0.50 & -0.022 (8.3e-1) & +0.015 (8.8e-1) & +0.075 (4.5e-1) & +0.247 (1.2e-2) & 0.435 (1.0) & 0.438 (1.0) & 0.540 (3.9e-5) & 0.590 (2.6e-19) \\
0.70 & -0.021 (8.3e-1) & +0.091 (3.6e-1) & +0.097 (3.3e-1) & +0.254 (9.6e-3) & 0.428 (1.0) & 0.486 (9.2e-1) & 0.546 (2.6e-6) & 0.588 (1.1e-18) \\
0.75 & -0.021 (8.3e-1) & +0.116 (2.4e-1) & +0.084 (4.0e-1) & +0.252 (1.0e-2) & 0.427 (1.0) & 0.497 (6.2e-1) & 0.544 (6.7e-6) & 0.583 (1.5e-16) \\
1.00 & -0.014 (8.9e-1) & +0.388 (5.1e-5) & +0.055 (5.8e-1) & +0.392 (4.3e-5) & 0.426 (1.0) & 0.652 (3.5e-52) & 0.526 (5.7e-3) & 0.633 (3.7e-40) \\
\bottomrule
\end{tabular}
\caption{Scene-level severity calibration on the primary SoCal wildfire benchmark. ``Unref.''/``ref.'' denote before/after semantic-aware residual refinement; ``ImageNet''/``Wildfire-tuned'' denote the frozen backbone used for the linear probe (Wildfire-tuned uses the leak-free checkpoint from our classifier-selection audit). Concordance p-values are one-sided binomial tests against chance (0.5); p$\sim$1 for unrefined at low $s$ indicates the ordering is significantly \emph{reversed} relative to real severity, not merely non-significant.}
\end{table*}

% Flooding (secondary/pilot category), n=73 test scenes with reliable S_real
\begin{table*}[t]
\centering
\small
\begin{tabular}{l cccc cccc}
\toprule
& \multicolumn{4}{c}{Spearman $\rho$ vs. $S_{real}$ (p-value)} & \multicolumn{4}{c}{Concordance rate (p-value)} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9}
Scale $s$ & ImageNet, unref. & ImageNet, ref. & Wildfire-tuned, unref. & Wildfire-tuned, ref.
          & ImageNet, unref. & ImageNet, ref. & Wildfire-tuned, unref. & Wildfire-tuned, ref. \\
\midrule
0.00 & +0.166 (1.6e-1) & +0.166 (1.6e-1) & +0.232 (4.9e-2) & +0.165 (1.6e-1) & 0.520 (7.3e-2) & 0.521 (6.6e-2) & 0.563 (2.3e-6) & 0.518 (9.9e-2) \\
0.25 & +0.166 (1.6e-1) & +0.277 (1.8e-2) & +0.242 (3.9e-2) & +0.166 (1.6e-1) & 0.520 (7.3e-2) & 0.559 (8.3e-6) & 0.562 (3.9e-6) & 0.518 (9.9e-2) \\
0.30 & +0.185 (1.2e-1) & +0.294 (1.2e-2) & +0.255 (2.9e-2) & +0.180 (1.3e-1) & 0.528 (2.3e-2) & 0.565 (1.4e-6) & 0.568 (4.5e-7) & 0.528 (2.3e-2) \\
0.50 & +0.210 (7.5e-2) & +0.381 (8.8e-4) & +0.267 (2.2e-2) & +0.308 (8.0e-3) & 0.528 (2.3e-2) & 0.613 (6.0e-17) & 0.562 (3.9e-6) & 0.592 (8.4e-12) \\
0.70 & +0.225 (5.6e-2) & +0.447 (7.3e-5) & +0.279 (1.7e-2) & +0.368 (1.4e-3) & 0.535 (6.3e-3) & 0.651 (8.5e-29) & 0.565 (1.0e-6) & 0.629 (1.7e-21) \\
0.75 & +0.205 (8.2e-2) & +0.454 (5.5e-5) & +0.262 (2.5e-2) & +0.380 (9.1e-4) & 0.525 (3.7e-2) & 0.656 (9.8e-31) & 0.556 (2.8e-5) & 0.637 (7.0e-24) \\
1.00 & +0.210 (7.5e-2) & +0.511 (3.8e-6) & +0.259 (2.7e-2) & +0.389 (6.6e-4) & 0.527 (2.6e-2) & 0.688 (7.3e-44) & 0.551 (1.1e-4) & 0.641 (2.2e-25) \\
\bottomrule
\end{tabular}
\caption{Scene-level severity calibration on the secondary flooding category (pilot). Wildfire-tuned uses the leak-free checkpoint from our classifier-selection audit.}
\end{table*}
```

## Ablation cross-check: does pseudo-target supervision help scene-level calibration too?

We additionally ran the same scene-level probe on the `w/o Pseudo Supervision` and
`w/o Scale-aware Refiner` ablation variants (SoCal wildfire, ImageNet probe), to test
whether the two ablated components help scene-level calibration in the same direction
they help the population-level CAS trend. The results are **not symmetric** and should
be written up as a genuine trade-off, not a second confirmation of the existing story:

| Variant | s | rho vs S_real (p) | concordance (p) | mean $\hat S_{gen}$ |
|---|---|---|---|---|
| Ours (full model) | 0.25 | -0.055 (0.58) | 0.401 (~1) | 0.405 |
| Ours (full model) | 0.50 | +0.015 (0.88) | 0.438 (~1) | 0.543 |
| Ours (full model) | 0.75 | +0.116 (0.25) | 0.497 (0.62) | 0.695 |
| Ours (full model) | 1.00 | +0.388 (5.1e-5) | 0.652 (3.5e-52) | 0.796 |
| w/o Pseudo Supervision | 0.25 | +0.400 (2.8e-5) | 0.665 (4.7e-61) | 0.808 |
| w/o Pseudo Supervision | 0.50 | +0.412 (1.6e-5) | 0.673 (7.2e-67) | 0.805 |
| w/o Pseudo Supervision | 0.75 | +0.418 (1.1e-5) | 0.676 (1.0e-69) | 0.798 |
| w/o Pseudo Supervision | 1.00 | +0.417 (1.2e-5) | 0.680 (2.5e-72) | 0.799 |
| w/o Scale-aware Refiner | 0.25 | -0.064 (0.52) | 0.402 (~1) | 0.365 |
| w/o Scale-aware Refiner | 0.50 | +0.045 (0.65) | 0.459 (~1) | 0.562 |
| w/o Scale-aware Refiner | 0.75 | +0.097 (0.33) | 0.495 (0.71) | 0.705 |
| w/o Scale-aware Refiner | 1.00 | +0.106 (0.29) | 0.495 (0.68) | 0.720 |

**`w/o Scale-aware Refiner` is uniformly worse than the full model** at scene-level
calibration, at every scale including the trained endpoint (s=1: rho=0.106 vs. 0.388,
concordance 49.5% vs. 65.2%, not even significant). This cleanly strengthens the
existing "scale-aware conditioning is important" ablation argument with harder,
scene-level evidence (not just "less consistent intermediate behavior" from a figure).

**`w/o Pseudo Supervision` is the opposite of what we expected**: its mean predicted
severity collapses to a near-constant ~0.80 from s=0.25 onward (this is exactly the
"premature saturation" the paper already documents via CAS in Table 14/Figure 6) --
but *within* that collapsed absolute level, scene-to-scene relative ordering is
*better* calibrated to real severity than the full model's, and significant at every
intermediate scale (unlike the full model, which is only significant at s=1).

The honest interpretation: pseudo-target supervision buys a smooth, gradually-rising
*absolute* severity trajectory (which is what population-level CAS trends measure),
at the cost of *relative*, scene-level discrimination at intermediate scales. Removing
it lets the refiner apply a stronger, less-dampened correction that happens to
separate scenes by real severity more cleanly, even though the resulting absolute
level looks prematurely saturated. This should be reported as a genuine trade-off in
the paper -- e.g., "pseudo-target supervision is necessary for a smooth, non-collapsing
severity trajectory in the aggregate sense that downstream augmentation/simulation use
cases likely care about, but Table [Z] shows this is not free: it comes with weaker
scene-level relative calibration at intermediate severities than a variant without it."
Do **not** write this section as if pseudo-target supervision improves both properties
simultaneously -- the data does not support that.

## Notes / things to double check before submitting
- n for socalfire experiment_a/b (103 test scenes) differs from the paper's headline n=246
  test pairs, because S_real requires a minimum building count in the xBD label json
  (`--min_buildings`, see `compute_flood_severity_labels.py` / the socalfire equivalent) --
  scenes with too few labeled buildings are dropped when computing S_real, but are still
  used elsewhere in the paper (e.g. CAS accuracy, which doesn't need S_real). State this
  explicitly in the table caption to avoid an "why does n change" reviewer question.
- DONE: the "wildfire-tuned" probe columns above have been rerun with the leak-free
  `socalfire_cls_clean_split` checkpoint (was previously using the leaked
  `socalfire_cls_real_fresh`). Numbers shifted meaningfully for flooding (e.g. s=1:
  rho 0.479->0.389), negligibly for socalfire (s=1: rho 0.393->0.392). The direction
  of the shift *strengthens* the psi cross-domain generalization finding (see
  `psi_generalization_and_experiment_c.md`), so this is not a case where fixing the
  leak weakens an existing claim.
