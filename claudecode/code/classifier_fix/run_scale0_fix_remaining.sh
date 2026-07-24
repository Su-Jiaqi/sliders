#!/usr/bin/env bash
set -euo pipefail

source /home/xjtucxy/miniconda3/etc/profile.d/conda.sh
conda activate sliders

ROOT=/home/xjtucxy/sjq/sliders
CLASSIFIER_CKPT="$ROOT/output-models/classifier/socalfire_cls_clean_split/best.pt"
DATA_ROOT="$ROOT/datasets/remote/socalfire"
TMP_ROOT="$ROOT/outputs/eval/scale0fix_remaining/_tmp_inputs"
OUT_ROOT="$ROOT/outputs/eval/scale0fix_remaining"
LOG_ROOT="$ROOT/logs/scale0fix_remaining"
DEVICE="${1:-cuda:4}"

mkdir -p "$TMP_ROOT" "$OUT_ROOT" "$LOG_ROOT"

# name : source-root : split : scale0_fixed dir
build_tmp_root() {
  local name="$1" src="$2" split="$3" scale0fixed="$4"
  local dst="$TMP_ROOT/$name/$split"
  mkdir -p "$dst"
  rm -f "$dst"/scale*
  ln -s "$scale0fixed" "$dst/scale0"
  for s in scale0.25 scale0.3 scale0.5 scale0.7 scale0.75 scale1; do
    ln -s "$src/$split/$s" "$dst/$s"
  done
}

TEST_S0FIX="$ROOT/outputs/refine-2/socalfire/test/scale0_fixed"
TRAIN_S0FIX="$ROOT/outputs/refine-2/socalfire/train/scale0_fixed"

build_tmp_root "train_refined"      "$ROOT/outputs/refine-2/socalfire"    train "$TRAIN_S0FIX"
build_tmp_root "train_direct"       "$ROOT/outputs/infer/socalfire"       train "$TRAIN_S0FIX"
build_tmp_root "test_direct"        "$ROOT/outputs/infer/socalfire"       test  "$TEST_S0FIX"
build_tmp_root "test_no_scale"      "$ROOT/outputs/ablation-2/socalfire/no_scale_finalbest" test "$TEST_S0FIX"

run_one() {
  local name="$1" split="$2"
  python "$ROOT/eval/socalfire_infered_eval_metrics.py" \
    --infer_root "$TMP_ROOT/$name" \
    --data_root "$DATA_ROOT" \
    --splits "$split" \
    --batch_size 16 \
    --device "$DEVICE" \
    --classifier_ckpt "$CLASSIFIER_CKPT" \
    --output_dir "$OUT_ROOT/$name" \
    2>&1 | tee "$LOG_ROOT/${name}.log"
}

run_one "train_refined" train
run_one "train_direct" train
run_one "test_direct" test
run_one "test_no_scale" test

echo "All remaining scale0-fix eval jobs finished."
