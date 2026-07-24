#!/usr/bin/env bash
set -euo pipefail

source /home/xjtucxy/miniconda3/etc/profile.d/conda.sh
conda activate sliders

ROOT=/home/xjtucxy/sjq/sliders
INFER_SCRIPT="$ROOT/infer/batch_infer_xbd_paired_scales.py"
# original headline LoRA path (output-models/lora/...) no longer exists on disk;
# best surviving candidate: same alpha/rank config, contemporaneous (2026-04-20).
MODEL_PATH="$ROOT/output-models/train/socalfire_slider_paired_alpha16.0_rank16_full/socalfire_slider_paired_alpha16.0_rank16_full_last.safetensors"
DATA_ROOT="$ROOT/datasets/remote/socalfire"
OUT_ROOT="$ROOT/outputs/eval/s001_continuity/infer"
REFINED_OUT="$ROOT/outputs/eval/s001_continuity/refined"
LOG_ROOT="$ROOT/logs/s001_continuity"
DEVICE="${1:-cuda:7}"

mkdir -p "$OUT_ROOT" "$REFINED_OUT" "$LOG_ROOT"

python -u "$INFER_SCRIPT" \
  --lora_path "$MODEL_PATH" \
  --pre_dir "$DATA_ROOT/test/pre" \
  --post_dir "$DATA_ROOT/test/post" \
  --output_root "$OUT_ROOT" \
  --scales "0.01" \
  --rank 16 \
  --alpha 16.0 \
  --train_method full \
  --device "$DEVICE" \
  --precision fp16 \
  --image_size 256 \
  --steps 50 \
  --start_noise 400 \
  --guidance_scale 1.0 \
  2>&1 | tee "$LOG_ROOT/01_infer_s001.log"

# refine through the same production refiner used for every other scale
python -u "$ROOT/refine/unified_scale_refiner.py" refine \
  --checkpoint "$ROOT/output-models/refine-2/socalfire/unified_refiner_scale1stronger/best.pt" \
  --pre_dir "$DATA_ROOT/test/pre" \
  --input_dir "$OUT_ROOT/test/scale0.01" \
  --output_dir "$REFINED_OUT/test/scale0.01" \
  --scale_value 0.01 \
  2>&1 | tee "$LOG_ROOT/02_refine_s001.log"

echo "s=0.01 continuity-check generation done."
