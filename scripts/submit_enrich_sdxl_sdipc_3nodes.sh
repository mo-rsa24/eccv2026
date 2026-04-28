#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SDXL_FINAL_DIR="${SDXL_FINAL_DIR:-${ROOT_DIR}/experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_frozen}"
SDXL_SDIPC_DIR="${SDXL_SDIPC_DIR:-${ROOT_DIR}/experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_frozen_sdipc}"
export SDXL_FINAL_DIR SDXL_SDIPC_DIR

NODE0="${NODE0:-mscluster107}"
SUBMIT_LOG_ROOT="${SUBMIT_LOG_ROOT:-${SDXL_SDIPC_DIR}/slurm_logs}"

mkdir -p "${SDXL_SDIPC_DIR}/logs"
mkdir -p "${SUBMIT_LOG_ROOT}"

echo "Submitting 1 node job (uses both GPUs on ${NODE0}):"
echo "  ${NODE0}: gpu0 group1 + group2 + group3, gpu1 group4 + group5 + group6"
echo ""
echo "Source: ${SDXL_FINAL_DIR}"
echo "Output: ${SDXL_SDIPC_DIR}"
echo "Slurm logs: ${SUBMIT_LOG_ROOT}"
echo ""

JOB1="$(sbatch --parsable --nodelist "${NODE0}" --output "${SUBMIT_LOG_ROOT}/sdxl-sdipc-enrich-%j.out" --error "${SUBMIT_LOG_ROOT}/sdxl-sdipc-enrich-%j.err" scripts/slurm_enrich_sdxl_sdipc_batch.sh 1)"

echo "Submitted:"
echo "  ${NODE0} batch 1 job: ${JOB1}"
echo ""
echo "Monitor:"
echo "  squeue -j ${JOB1}"
echo "Logs:"
echo "  tail -f ${SUBMIT_LOG_ROOT}/sdxl-sdipc-enrich-${JOB1}.out"
echo "Worker logs:"
echo "  ls -lah \"${SDXL_SDIPC_DIR}/logs\" | tail"
