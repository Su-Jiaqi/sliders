#!/usr/bin/env bash
set -euo pipefail

source /home/xjtucxy/miniconda3/etc/profile.d/conda.sh
conda activate sliders

ROOT=/home/xjtucxy/sjq/sliders
CLASSIFIER_CKPT="$ROOT/output-models/classifier/socalfire_cls_clean_split/best.pt"
DATA_ROOT="$ROOT/datasets/remote/socalfire"
TMP_ROOT="$ROOT/outputs/eval/scale0fix_remaining/_tmp_inputs"
OUT_ROOT="$ROOT/outputs/eval/scale0fix_remaining_fullmetrics"
LOG_ROOT="$ROOT/logs/scale0fix_remaining_fullmetrics"
DEVICE="${1:-cuda:6}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

run_one() {
  local name="$1" split="$2"
  python "$ROOT/eval/socalfire_infered_eval_metrics.py" \
    --infer_root "$TMP_ROOT/$name" \
    --data_root "$DATA_ROOT" \
    --splits "$split" \
    --batch_size 16 \
    --device "$DEVICE" \
    --run_clip \
    --run_dino \
    --run_fid \
    --classifier_ckpt "$CLASSIFIER_CKPT" \
    --output_dir "$OUT_ROOT/$name" \
    2>&1 | tee "$LOG_ROOT/${name}.log"
}

run_one "train_refined" train
run_one "train_direct" train
run_one "test_direct" test
run_one "test_no_scale" test

echo "All full-metrics reruns finished."
