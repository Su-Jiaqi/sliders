#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INFER_SCRIPT="$REPO_ROOT/infer/batch_infer_xbd_paired_scales.py"
MODEL_PATH="$REPO_ROOT/output-models/lora/xbd_paired_slider_endpoint_paired_alpha16.0_rank16_full/xbd_paired_slider_endpoint_paired_alpha16.0_rank16_full_last.safetensors"
DATA_ROOT="$REPO_ROOT/datasets/remote/socalfire"
PRE_DIR="$DATA_ROOT/pre"
POST_DIR="$DATA_ROOT/post"
OUT_ROOT="$REPO_ROOT/outputs/infer/socalfire"
LOG_FILE="$OUT_ROOT/infer_socalfire.log"
PID_FILE="$OUT_ROOT/infer_socalfire.pid"
CONDA_ENV="sliders"

SCALES="0.25,0.5,0.75"
RANK=16
ALPHA=16.0
TRAIN_METHOD="full"
DEVICE="cuda:0"
PRECISION="fp16"
IMAGE_SIZE=256
STEPS=50
START_NOISE=400
GUIDANCE_SCALE=1.0

mkdir -p "$OUT_ROOT"

cmd=(
  "python" "$INFER_SCRIPT"
  "--lora_path" "$MODEL_PATH"
  "--pre_dir" "$PRE_DIR"
  "--post_dir" "$POST_DIR"
  "--output_root" "$OUT_ROOT"
  "--scales" "$SCALES"
  "--rank" "$RANK"
  "--alpha" "$ALPHA"
  "--train_method" "$TRAIN_METHOD"
  "--device" "$DEVICE"
  "--precision" "$PRECISION"
  "--image_size" "$IMAGE_SIZE"
  "--steps" "$STEPS"
  "--start_noise" "$START_NOISE"
  "--guidance_scale" "$GUIDANCE_SCALE"
  "--skip_existing"
)

start() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running with PID $(cat "$PID_FILE")."
    echo "Log: $LOG_FILE"
    exit 0
  fi

  local run_cmd
  run_cmd="source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate \"$CONDA_ENV\" && PYTHONUNBUFFERED=1 ${cmd[*]}"
  nohup bash -lc "$run_cmd" >"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  echo "Started background inference. PID: $(cat "$PID_FILE")"
  echo "Log file: $LOG_FILE"
}

status() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Running. PID: $(cat "$PID_FILE")"
    echo "Log file: $LOG_FILE"
  else
    echo "Not running."
    echo "Log file: $LOG_FILE"
  fi
}

stop() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    kill "$(cat "$PID_FILE")"
    echo "Stopped PID $(cat "$PID_FILE")."
    rm -f "$PID_FILE"
  else
    echo "Not running."
    rm -f "$PID_FILE"
  fi
}

case "${1:-start}" in
  start) start ;;
  status) status ;;
  stop) stop ;;
  *)
    echo "Usage: $0 {start|status|stop}"
    exit 1
    ;;
esac
