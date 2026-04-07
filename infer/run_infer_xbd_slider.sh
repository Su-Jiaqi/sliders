#!/usr/bin/env bash
# Run xBD slider inference from repo root. Defaults match infer_xbd_slider.py.
# Usage:
#   ./infer/run_infer_xbd_slider.sh
#   DEVICE=cuda:1 ./infer/run_infer_xbd_slider.sh --input_image path/to/pre.png
#   ./infer/run_infer_xbd_slider.sh --no_strip --slider_scale 2.0
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DEVICE="${DEVICE:-cuda:0}"

exec python infer/infer_xbd_slider.py \
  --device "$DEVICE" \
  "$@"
