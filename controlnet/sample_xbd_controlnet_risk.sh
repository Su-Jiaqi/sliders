#!/usr/bin/env bash
set -e

cd /home/sjq/concept_sliders

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

# CONTROLNET_DIR="/home/sjq/concept_sliders/output-models/xbd_controlnet_risk_binary/final_controlnet"
CONTROLNET_DIR="/data2/sjq/output-models/xbd_controlnet_risk_binary/final_controlnet"
INPUT_PRE="/home/sjq/concept_sliders/datasets/remote/socalfire/pre/2.png"
# OUTPUT_DIR="/home/sjq/concept_sliders/outputs/xbd_controlnet_risk_samples"
OUTPUT_DIR="/data2/sjq/output-models/xbd_controlnet_risk_binary"
PROMPT="aerial post-disaster image of the same location after wildfire damage"

mkdir -p "$OUTPUT_DIR"

python controlnet/xbd_controlnet_risk_binary.py sample \
  --controlnet_dir "$CONTROLNET_DIR" \
  --pretrained_model runwayml/stable-diffusion-v1-5 \
  --input_pre "$INPUT_PRE" \
  --prompt "$PROMPT" \
  --output_dir "$OUTPUT_DIR" \
  --severities 0,0.25,0.5,0.75,1.0 \
  --num_inference_steps 30 \
  --guidance_scale 7.5 \
  --mixed_precision bf16 \
  --seed 42
