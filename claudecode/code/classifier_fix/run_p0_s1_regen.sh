#!/usr/bin/env bash
set -euo pipefail

source /home/xjtucxy/miniconda3/etc/profile.d/conda.sh
conda activate sliders

ROOT=/home/xjtucxy/sjq/sliders
INFER_SCRIPT="$ROOT/infer/batch_infer_xbd_paired_scale01.py"
# original production LoRA path (output-models/lora/...) no longer exists on disk;
# best surviving candidate, same config, contemporaneous timestamp -- same caveat
# already disclosed in headline_vs_seed_stability_gap_20260725.md.
MODEL_PATH="$ROOT/output-models/train/socalfire_slider_paired_alpha16.0_rank16_full/socalfire_slider_paired_alpha16.0_rank16_full_last.safetensors"
DATA_ROOT="$ROOT/datasets/remote/socalfire"
OUT_ROOT="$ROOT/outputs/eval/p0_s1_regen/infer"
REFINED_OUT="$ROOT/outputs/eval/p0_s1_regen/refined"
LOG_ROOT="$ROOT/logs/p0_s1_regen"
DEVICE="${1:-cuda:4}"

mkdir -p "$OUT_ROOT" "$REFINED_OUT" "$LOG_ROOT"

python -u "$INFER_SCRIPT" \
  --lora_path "$MODEL_PATH" \
  --pre_dir "$DATA_ROOT/test/pre" \
  --output_root "$OUT_ROOT" \
  --rank 16 \
  --alpha 16.0 \
  --train_method full \
  --device "$DEVICE" \
  --precision bf16 \
  --image_size 256 \
  --steps 50 \
  --start_noise 100 \
  --guidance_scale 1.0 \
  --seed 42 \
  2>&1 | tee "$LOG_ROOT/01_infer_scale01.log"

# refine scale1 through the production refiner (scale0 output from this script
# is discarded here -- production s=0 already uses the pixel-identity fix)
python -u "$ROOT/refine/unified_scale_refiner.py" refine \
  --checkpoint "$ROOT/output-models/refine-2/socalfire/unified_refiner_scale1stronger/best.pt" \
  --pre_dir "$DATA_ROOT/test/pre" \
  --input_dir "$OUT_ROOT/scale1" \
  --output_dir "$REFINED_OUT/test/scale1" \
  --scale_value 1.0 \
  2>&1 | tee "$LOG_ROOT/02_refine_scale1.log"

echo "P0 s=1 regeneration done."
