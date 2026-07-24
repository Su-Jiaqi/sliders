#!/usr/bin/env bash
set -euo pipefail

source /home/xjtucxy/miniconda3/etc/profile.d/conda.sh
conda activate sliders

ROOT=/home/xjtucxy/sjq/sliders
CLASSIFIER_CKPT="$ROOT/output-models/classifier/socalfire_cls_clean_split/best.pt"
DATA_ROOT="$ROOT/datasets/remote/socalfire"
OUT_ROOT="$ROOT/outputs/eval/cas_figure_clean"
LOG_ROOT="$ROOT/logs/cas_figure_clean"
TMP_ROOT="$ROOT/outputs/eval/cas_figure_clean/_tmp_inputs"
DEVICE="${1:-cuda:7}"
SCALE0_FIXED="$ROOT/outputs/refine-2/socalfire/test/scale0_fixed"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

# s=0 is pipeline-independent (it's always x_pre by definition), so both curves
# use the same pixel-identity scale0 fix, not each variant's own (buggy) scale0.
build_tmp_root() {
  local name="$1"
  local src="$2"
  local dst="$TMP_ROOT/$name/test"
  mkdir -p "$dst"
  rm -f "$dst"/scale*
  ln -s "$SCALE0_FIXED" "$dst/scale0"
  for s in scale0.25 scale0.3 scale0.5 scale0.7 scale0.75 scale1; do
    ln -s "$src/$s" "$dst/$s"
  done
}

build_tmp_root "ours_refine2" "$ROOT/outputs/refine-2/socalfire/test"
build_tmp_root "no_pseudo_refine2" "$ROOT/outputs/ablation-2/socalfire/no_pseudo_finalbest/test"

run_one() {
  local name="$1"
  python "$ROOT/eval/socalfire_infered_eval_metrics.py" \
    --infer_root "$TMP_ROOT/$name" \
    --data_root "$DATA_ROOT" \
    --splits test \
    --batch_size 16 \
    --device "$DEVICE" \
    --classifier_ckpt "$CLASSIFIER_CKPT" \
    --output_dir "$OUT_ROOT/$name" \
    2>&1 | tee "$LOG_ROOT/${name}.log"
}

run_one "ours_refine2"
run_one "no_pseudo_refine2"

echo "Both CAS-figure eval jobs finished."
