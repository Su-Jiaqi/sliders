#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1

PRE_DIR="datasets/remote/socalfire/pre"
POST_DIR="datasets/remote/socalfire/post"

# OUTPUT_DIR="output-models/xbd_controlnet_risk_binary"

OUTPUT_DIR="output-models/xbd_controlnet_risk_binary"
PROMPT="aerial post-disaster image of the same location after wildfire damage"

mkdir -p logs
LOG="logs/train_xbd_controlnet_risk_$(date +%Y%m%d_%H%M%S).log"

python controlnet/xbd_controlnet_risk_binary.py train \
  --pre_dir "$PRE_DIR" \
  --post_dir "$POST_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --pretrained_model runwayml/stable-diffusion-v1-5 \
  --prompt "$PROMPT" \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_workers 4 \
  --max_train_steps 20000 \
  --learning_rate 1e-5 \
  --save_every 1000 \
  --log_every 20 \
  --mixed_precision bf16 \
  --seed 42 \
  2>&1 | tee "$LOG"
