# Calibrated-severity v2: closing the extrapolation gap

## 0. Diagnosis, precisely

Confirmed by reading `sample_s_calibrated()` in `train/train_xbd_paired_slider_calibrated.py`:

```python
s = torch.rand(batch_size, device=device) * s_i
if endpoint_prob > 0:
    ...
    s_endpoint = torch.where(coin2 >= 0.5, s_i, torch.zeros_like(s_i))
    s = torch.where(endpoint_mask, s_endpoint, s)
return s
```

For any given training pair, `s` is **always** sampled in `[0, s_i]`, never above it. So for a
low-severity pair (`s_i` small, e.g. 0.1), the model never once sees a direct training
target at `s=0.9` *for that specific pair*. Everything the model knows about "what does
s=0.9 look like" comes purely from *other* pairs whose own `s_i` happens to be near 0.9 —
generalization has to happen entirely through the network's smoothness in the continuous
conditioning input `s`, with zero explicit supervision pushing it to happen correctly.

This is the root cause of both v1 findings:
- **Residual correlation with real severity at s=1**: nothing ever taught the model that
  "requested max severity" should look similar regardless of scene identity — it only ever
  saw "max severity" *for pairs that were themselves already severe*.
- **Non-monotonic mean-severity trajectory**: the extrapolation region `(s_i, 1]` is
  completely unconstrained for the majority of pairs (`s_i` distribution is skewed low —
  mean 0.281 for socalfire, 0.110 for flooding), so nothing prevents erratic behavior right
  at the top of the range where extrapolation distance is largest.

v1's hybrid data scheme (real anchors for labeled pairs, s=1 fallback for unlabeled) was a
necessary first step, but it does not, by itself, give the model a reason to behave
consistently in the region it was never shown. v2 adds that reason directly.

## 1. Proposed fix: severity-probe-guided extrapolation loss

**Idea**: for `s > s_i` (the region with no real pixel target), don't leave it
unsupervised. Instead, decode the model's `x0` estimate at that `s` and penalize the
*classifier/probe-predicted severity* of the decoded image against the severity level the
requested `s` implies (`target ≈ s * 3`, since `S_real` is 0-3). This gives the model an
explicit, differentiable reason to make "requested s" and "perceived severity" track each
other, regardless of which scene it is — directly targeting the exact failure mode
measured in v1's evaluation.

This reuses infrastructure that already exists in the training loop:
- The image-endpoint loss already decodes `x0_pred` to pixels with a differentiable VAE
  decode (`decode_latents_with_grad`) to compute a reconstruction loss at s=0/s=1 — the
  same decode path can produce pixels for the new loss at extrapolated `s`.
- The frozen clean classifier (`socalfire_cls_clean_split/best.pt`) already provides
  penultimate features; the existing Ridge severity probe (fit in
  `experiment_a/multiscale_correlation.py`) is linear (`y = w^T x + b`), so it converts
  directly into a frozen `nn.Linear` layer — fully differentiable, no new model to train.

### 1.1 Concretely

1. Export the fitted Ridge probe's coefficients (`ridge.coef_`, `ridge.intercept_`) to a
   `.npz` file once per category (reuse the exact fit already done in
   `multiscale_correlation.py` — just add a `--save_probe` flag to dump the weights instead
   of only printing correlations).
2. New module: `SeverityProbeHead` — frozen MobileNetV3-Small feature extractor (same
   clean classifier backbone) + frozen linear layer loaded from the exported weights.
   Given a pixel batch, returns a differentiable scalar predicted severity per image.
3. In `train_xbd_paired_slider_calibrated.py`, add a new sampling branch: with some
   probability (e.g. 30% of steps, only for pairs with `s_i < 1`), additionally sample
   `s_extrap ~ Uniform(s_i, 1]` for that pair, run the existing `forward_eps_and_x0` at
   `s_extrap` to get `x0_pred_extrap`, decode it, run through `SeverityProbeHead`, and add:
   `loss_severity_extrap = MSE(probe(decode(x0_pred_extrap)), s_extrap * 3)`
   weighted by a new `lambda_severity_extrap` (start at 0.1-0.2, tune by watching whether
   it destabilizes the main eps loss).
4. Everything else (data scheme, f1/fu retraining, evaluation protocol) stays identical to
   v1, so the v1-vs-v2 comparison isolates the effect of this one change.

### 1.2 Why this is better than the alternatives considered

- **Cross-pair pixel borrowing** (use another higher-severity pair's real post image as a
  pseudo-target for the extrapolation region): more complex (needs a severity-matching
  scheme across pairs), and risks the model memorizing a small set of "donor" textures
  rather than learning a general severity concept.
- **Feature-prototype matching** (pull x0 features toward the mean feature of real images
  in the target severity bucket): plausible alternative, similar spirit, but the scalar
  severity-regression loss is simpler to implement, reuses the exact evaluation tool
  already trusted in this paper, and gives a directly interpretable training curve
  (predicted severity vs. target severity) to sanity-check during training.
- **Just extending sample_s_calibrated's range unconditionally** (sample s in [0,1]
  regardless of s_i, using the SAME real post image as target even when s > s_i): this
  silently reintroduces v0's original bug — it would explicitly teach the model
  "s=0.9 for this low-severity pair should look like ITS OWN unremarkable real post image,"
  actively undoing the calibration. Must not do this.

## 2. Secondary, cheaper improvements (worth doing alongside, low cost)

1. **Lower or remove the hard S_real>=2.0 cutoff for f1's training subset**, replace with
   severity-weighted sampling over all labeled pairs (weight ∝ S_real, or a soft threshold
   with a wider band). Directly addresses the small-N problem, especially severe for
   flooding (17 pairs vs. socalfire's 81) — a hard cutoff at 2.0 throws away pairs at 1.5-2.0
   that still carry useful high-severity signal.
2. **Light data augmentation (horizontal flip, small rotation) for f1's training subset**,
   free effective-sample-count multiplier for the smallest, most overfitting-prone stage
   of the pipeline.
3. **Report probe-based evaluation, not the binary-classifier CAS test**, as the primary
   metric going forward — already adopted for socalfire's results file, should be the
   only evaluation path for the flooding replication and any v2 comparison, since it's the
   one that actually has statistical power to see the effect (v1's binary-classifier test
   was ceiling-limited and uninformative).

## 3. What NOT to do

- Do not retune hyperparameters (lr, iterations, LoRA rank) purely to chase a better-looking
  number on this specific test set — the v1→v2 comparison should isolate the ONE
  architectural change (the extrapolation loss), not become a general hyperparameter search.
- Do not drop the s=1 residual-correlation or non-monotonicity findings from the writeup
  even if v2 fixes them — report v1 as the honest first attempt and v2 as the fix,
  showing the diagnosis → fix → re-measurement arc explicitly. This is a stronger, more
  convincing narrative for a reviewer than only showing the final (fixed) numbers.
- Do not run v2 on flooding until socalfire's v2 result is in — socalfire has 4.8x more
  high-severity labeled pairs (81 vs 17) and is the better testbed for first confirming the
  fix works at all before stress-testing it on a smaller-data category.

## 4. Execution order

1. Let the currently-running flooding v1 replication (Tasks #8-14) finish — needed
   regardless, to know whether v1's pattern (intermediate decoupling, residual s=1
   correlation, non-monotonic top) replicates across categories before layering v2 on top.
2. Implement `SeverityProbeHead` + probe-weight export + the new sampling/loss branch in a
   NEW script `train/train_xbd_paired_slider_calibrated_v2.py` (copy of the v1 script,
   modified) — keep v1 unmodified so it remains reproducible.
3. Train v2 on socalfire only, first (15000 steps, same hyperparameters as v1 plus the new
   `lambda_severity_extrap`).
4. Run the identical downstream pipeline (train-split inference, f1 retrain — now with the
   severity-weighted sampling from section 2.1, pseudo-targets, fu retrain, test-split
   inference + refine).
5. Re-run the probe-based evaluation (`multiscale_correlation.py`) comparing v2 against
   both v1 and production on socalfire. Specifically check: (a) is the s=1 correlation with
   real severity reduced further than v1's 0.300? (b) is the mean-severity trajectory now
   monotonic? Report honestly either way.
6. Only if v2 clearly improves on v1's socalfire result, consider extending v2 to flooding
   as well — otherwise, report v1 as the presented supplementary variant and v2 as a
   documented but unsuccessful further attempt (equally honest, equally valid to report).
