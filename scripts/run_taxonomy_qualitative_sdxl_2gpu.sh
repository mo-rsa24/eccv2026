#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <shared run_taxonomy_qualitative_sdxl.py args...>" >&2
  echo "Example:" >&2
  echo "  $0 --output-dir experiments/eccv2026/sdxl_screening/my_run --seed 42 --num-inference-steps 50 --guidance-scale 7.5" >&2
  exit 1
fi

ROOT_ARGS=("$@")

echo "[worker 0] GPU 0"
python scripts/run_taxonomy_qualitative_sdxl.py \
  "${ROOT_ARGS[@]}" \
  --gpu-id 0 \
  --num-workers 2 \
  --worker-index 0 &
PID0=$!

echo "[worker 1] GPU 1"
python scripts/run_taxonomy_qualitative_sdxl.py \
  "${ROOT_ARGS[@]}" \
  --gpu-id 1 \
  --num-workers 2 \
  --worker-index 1 &
PID1=$!

FAIL=0
wait "$PID0" || FAIL=1
wait "$PID1" || FAIL=1

if [[ "$FAIL" -ne 0 ]]; then
  echo "One or more SDXL workers failed." >&2
  exit 1
fi

echo "Both SDXL workers completed successfully."
