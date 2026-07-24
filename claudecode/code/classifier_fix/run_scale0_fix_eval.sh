#!/usr/bin/env bash
set -euo pipefail

source /home/xjtucxy/miniconda3/etc/profile.d/conda.sh
conda activate sliders

ROOT=/home/xjtucxy/sjq/sliders
CLASSIFIER_CKPT="$ROOT/output-models/classifier/socalfire_cls_clean_split/best.pt"
TMP_ROOT="$ROOT/outputs/eval/scale0fix/_tmp_inputs"
OUT_ROOT="$ROOT/outputs/eval/scale0fix"
LOG_ROOT="$ROOT/logs/scale0fix"
DEVICE="${1:-cuda:5}"

mkdir -p "$TMP_ROOT" "$OUT_ROOT" "$LOG_ROOT"

CATEGORIES="socalfire hurricane-florence midwest-flooding santarosa volcano"

for cat in $CATEGORIES; do
  SRC="$ROOT/outputs/refine-2/$cat/test"
  DST="$TMP_ROOT/$cat/test"
  mkdir -p "$DST"
  rm -f "$DST"/scale*
  ln -s "$ROOT/outputs/refine-2/$cat/test/scale0_fixed" "$DST/scale0"
  for s in scale0.25 scale0.3 scale0.5 scale0.7 scale0.75 scale1; do
    ln -s "$SRC/$s" "$DST/$s"
  done
done

for cat in $CATEGORIES; do
  echo "=== $cat ==="
  python "$ROOT/eval/socalfire_infered_eval_metrics.py" \
    --infer_root "$TMP_ROOT/$cat" \
    --data_root "$ROOT/datasets/remote/$cat" \
    --splits test \
    --batch_size 16 \
    --device "$DEVICE" \
    --run_clip \
    --run_dino \
    --classifier_ckpt "$CLASSIFIER_CKPT" \
    --output_dir "$OUT_ROOT/$cat" \
    2>&1 | tee "$LOG_ROOT/${cat}.log"
done

echo "All scale0-fix eval jobs finished."
