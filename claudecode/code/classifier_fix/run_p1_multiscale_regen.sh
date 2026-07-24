#!/usr/bin/env bash
set -euo pipefail

source /home/xjtucxy/miniconda3/etc/profile.d/conda.sh
conda activate sliders

ROOT=/home/xjtucxy/sjq/sliders
INFER_SCRIPT="$ROOT/infer/batch_infer_xbd_paired_scales.py"
MODEL_PATH="$ROOT/output-models/train/socalfire_slider_paired_alpha16.0_rank16_full/socalfire_slider_paired_alpha16.0_rank16_full_last.safetensors"
DATA_ROOT="$ROOT/datasets/remote/socalfire"
OUT_ROOT="$ROOT/outputs/eval/p1_multiscale_regen/infer"
REFINED_OUT="$ROOT/outputs/eval/p1_multiscale_regen/refined"
LOG_ROOT="$ROOT/logs/p1_multiscale_regen"
DEVICE="${1:-cuda:7}"

mkdir -p "$OUT_ROOT" "$REFINED_OUT" "$LOG_ROOT"

python -u "$INFER_SCRIPT" \
  --lora_path "$MODEL_PATH" \
  --pre_dir "$DATA_ROOT/test/pre" \
  --post_dir "$DATA_ROOT/test/post" \
  --output_root "$OUT_ROOT" \
  --scales "0.25,0.3,0.5,0.7,0.75" \
  --rank 16 \
  --alpha 16.0 \
  --train_method full \
  --device "$DEVICE" \
  --precision fp16 \
  --image_size 256 \
  --steps 50 \
  --start_noise 400 \
  --guidance_scale 1.0 \
  --seed 42 \
  2>&1 | tee "$LOG_ROOT/01_infer_multiscale.log"

for s in 0.25 0.3 0.5 0.7 0.75; do
  python -u "$ROOT/refine/unified_scale_refiner.py" refine \
    --checkpoint "$ROOT/output-models/refine-2/socalfire/unified_refiner_scale1stronger/best.pt" \
    --pre_dir "$DATA_ROOT/test/pre" \
    --input_dir "$OUT_ROOT/scale$s" \
    --output_dir "$REFINED_OUT/test/scale$s" \
    --scale_value "$s" \
    2>&1 | tee "$LOG_ROOT/02_refine_scale${s}.log"
done

echo "P1 multiscale regeneration done."
