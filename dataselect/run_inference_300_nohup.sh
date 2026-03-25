#!/bin/bash
# Run inference 300 with nohup, log under logs/inference_300_scale01_<timestamp>.log
cd "$(dirname "$0")"
mkdir -p logs
LOG="logs/inference_300_scale01_$(date +%Y%m%d_%H%M%S).log"
nohup python run_inference_300_scale01.py --device cuda:2 > "$LOG" 2>&1 &
echo "PID: $!"
echo "Log: $LOG"
echo "tail -f $LOG  # to follow"
