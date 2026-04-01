#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_pointwise_scaffold_bridge.sh [DATA_DIR] [OUTPUT_DIR] [extra plot args...]
#
# Examples:
#   scripts/run_pointwise_scaffold_bridge.sh
#   scripts/run_pointwise_scaffold_bridge.sh \
#       experiments/inversion/gap_analysis/small_20260302_143000
#   scripts/run_pointwise_scaffold_bridge.sh \
#       experiments/inversion/gap_analysis/small_20260302_143000 \
#       experiments/inversion/gap_analysis/small_20260302_143000/figures \
#       --and-anchor mean
#   CONDA_ENV_NAME=superdiff scripts/run_pointwise_scaffold_bridge.sh

DATA_DIR="${1:-experiments/inversion/gap_analysis}"
OUTPUT_DIR="${2:-${DATA_DIR}/figures}"
shift $(( $# >= 1 ? 1 : 0 ))
shift $(( $# >= 1 ? 1 : 0 ))
EXTRA_ARGS=("$@")

ENV_NAME="${CONDA_ENV_NAME:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLOTS=(29 30 34 31 32 33 24)

echo "Data dir:   ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
if [[ -n "${ENV_NAME}" ]]; then
  echo "Conda env:  ${ENV_NAME}"
else
  echo "Conda env:  <none> (using active ${PYTHON_BIN})"
fi
echo "Plot stack: ${PLOTS[*]}"

for plot_id in "${PLOTS[@]}"; do
  echo
  echo "[${plot_id}] running..."
  if [[ -n "${ENV_NAME}" ]]; then
    conda run -n "${ENV_NAME}" "${PYTHON_BIN}" scripts/plot_gap_analysis.py \
      --data-dir "${DATA_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --plot "${plot_id}" \
      "${EXTRA_ARGS[@]}"
  else
    "${PYTHON_BIN}" scripts/plot_gap_analysis.py \
      --data-dir "${DATA_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --plot "${plot_id}" \
      "${EXTRA_ARGS[@]}"
  fi
done

echo
echo "Done. Wrote scaffold figures to ${OUTPUT_DIR}"
