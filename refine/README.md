# Refine Overview

This folder contains the image-space refinement stage used by the current paper implementation.

## What Is Actually Used

The refinement pipeline in the current codebase is composed of four scripts:

- `scale0_only_refiner.py`
- `scale1_only_refiner.py`
- `build_pseudo_targets.py`
- `unified_scale_refiner.py`

In paper terms, this maps to:

- endpoint-specific refiners
  - `scale0_only_refiner.py`
  - `scale1_only_refiner.py`
- unified scale-conditioned refiner
  - `unified_scale_refiner.py`
- pseudo-target generation for intermediate scales
  - `build_pseudo_targets.py`

## How Refinement Is Organized

The implemented refinement logic is:

- `s = 0`
  - refine the generated `scale0` image with `scale0_only_refiner.py`
- `s = 1`
  - refine the generated `scale1` image with `scale1_only_refiner.py`
- `0 < s < 1`
  - first build pseudo targets for intermediate scales with `build_pseudo_targets.py`
  - then refine intermediate scales with `unified_scale_refiner.py`

This matches the paper description:

- endpoint-specific refiners are used at `s=0` and `s=1`
- a unified scale-conditioned refiner is used for intermediate scales

## Practical Interpretation

The current refinement stage is not a single script.

It is a two-endpoint plus one-unified design:

1. Train a `scale0` refiner for the pre-disaster endpoint.
2. Train a `scale1` refiner for the post-disaster endpoint.
3. Use the endpoint refiners to build pseudo supervision for intermediate scales.
4. Train a unified refiner that takes `(generated_image, pre_image, scale)` and predicts a residual for intermediate scales.

## Main Files

- `scale0_only_refiner.py`
  - endpoint refiner for `scale0`
- `scale1_only_refiner.py`
  - endpoint refiner for `scale1`
- `build_pseudo_targets.py`
  - generates pseudo targets for intermediate scales
- `unified_scale_refiner.py`
  - unified scale-aware residual refiner for non-endpoint scales

## Not The Main Path

These files exist in the folder, but they are not the main socalfire refinement path summarized above:

- `endpoint_refiner.py`
- `scale1_only_refiner_stronger.py`

They may be useful for experiments or variants, but the current paper-aligned refinement path is the four-script pipeline listed above.

## Related Note

For a more detailed step-by-step data and training description, see:

- [DATA_PREP_AND_PIPELINE.md](./DATA_PREP_AND_PIPELINE.md)
