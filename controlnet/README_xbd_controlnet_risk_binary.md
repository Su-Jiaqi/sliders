# xBD binary-risk ControlNet starter

This package is a practical **first experiment** for your task:

- input: pre-disaster image
- target: either pre or post
- severity:
  - `0` means target = pre
  - `1` means target = post

This is **not yet the final continuous risk-slider model**, but it is the correct first step if you only have:

- xBD paired pre/post images
- one unified prompt
- no per-image prompts

## Why this version makes sense

The official ControlNet training recipe expects a dataset with:

- target image (`jpg`)
- control image (`hint`)
- prompt (`txt`)

For your task, we map it to:

- `target image` = `post` (or `pre` for identity samples)
- `control image` = `pre`
- `prompt` = one shared prompt for all samples

This follows the official ControlNet framing of target / control / prompt.

## Recommended prompt

Use one generic prompt, for example:

- `aerial post-disaster image of the same location after wildfire damage`
- `satellite image of the same location after disaster damage`
- `overhead remote sensing image of the same location after wildfire`

Do **not** make it too specific at first.

## Recommended first experiment

Train:
- `severity=1` samples: `(pre -> post)`
- `severity=0` samples: `(pre -> pre)`

Then sample with:
- `0, 0.25, 0.5, 0.75, 1.0`

and see whether geometry stays stable while damage increases.

## Example commands

Train:
```bash
python xbd_controlnet_risk_binary.py train \
  --pre_dir /path/to/xbd/pre \
  --post_dir /path/to/xbd/post \
  --output_dir ./outputs/xbd_controlnet_risk \
  --pretrained_model runwayml/stable-diffusion-v1-5 \
  --prompt "aerial post-disaster image of the same location after wildfire damage" \
  --resolution 512 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_train_steps 20000 \
  --learning_rate 1e-5 \
  --save_every 1000 \
  --mixed_precision bf16
```

Sample:
```bash
python xbd_controlnet_risk_binary.py sample \
  --controlnet_dir ./outputs/xbd_controlnet_risk/final_controlnet \
  --pretrained_model runwayml/stable-diffusion-v1-5 \
  --input_pre /path/to/xbd/pre/example.png \
  --prompt "aerial post-disaster image of the same location after wildfire damage" \
  --output_dir ./outputs/xbd_controlnet_risk_samples \
  --severities 0,0.25,0.5,0.75,1.0 \
  --num_inference_steps 30 \
  --guidance_scale 7.5 \
  --seed 42
```

## Important caveat

Because this binary starter only uses severity `0/1`, the intermediate values `0.25/0.5/0.75` are an extrapolation through ControlNet strength, not a guaranteed physically correct continuous disaster model.

That is okay for phase 1.
If this base experiment works, the next phase is:
- build a better scalar severity label
- add explicit severity embedding in the network
- maybe add region-aware loss or damage masks
