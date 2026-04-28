#!/usr/bin/env bash
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --job-name=sdxl-sdipc-enrich

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <BATCH_ID: 1>" >&2
  exit 1
fi

BATCH_ID="$1"

case "$BATCH_ID" in
  1)
    GROUPS0=(
      "group1_cooccurrence"
      "group2_factorization"
      "group3_role_separable_object_scene"
    )
    GROUPS1=(
      "group4_dual_object_composition"
      "group5_concept_prior_entanglement"
      "group6_coherent_collision"
    )
    ;;
  *)
    echo "Invalid BATCH_ID: $BATCH_ID (expected 1)" >&2
    exit 1
    ;;
esac

SDXL_FINAL_DIR="${SDXL_FINAL_DIR:-${ROOT_DIR}/experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_frozen}"
SDXL_SDIPC_DIR="${SDXL_SDIPC_DIR:-${ROOT_DIR}/experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_frozen_sdipc}"
export SDXL_FINAL_DIR SDXL_SDIPC_DIR

CONDA_ENV="${CONDA_ENV:-jaxstack}"
KEEP_CLIP_ON_GPU="${KEEP_CLIP_ON_GPU:-1}"
COPY_MODE="${COPY_MODE:-symlink}"

LOG_ROOT="${SDXL_SDIPC_DIR}/logs"
mkdir -p "$LOG_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH on compute node. Load your module or adjust PATH." >&2
  exit 1
fi

# Ensure conda activate works in non-interactive shells.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

COMMON_ARGS=(
  --source-dir "$SDXL_FINAL_DIR"
  --output-dir "$SDXL_SDIPC_DIR"
  --groups
)

if [[ "$COPY_MODE" != "symlink" && "$COPY_MODE" != "copy" ]]; then
  echo "Invalid COPY_MODE: $COPY_MODE (expected 'symlink' or 'copy')" >&2
  exit 1
fi

COMMON_ARGS+=(--copy-mode "$COPY_MODE")

if [[ "$KEEP_CLIP_ON_GPU" == "1" ]]; then
  COMMON_ARGS+=(--keep-clip-on-gpu)
fi

run_worker() {
  local gpu="$1"
  shift
  local groups=("$@")
  local joined_groups
  joined_groups="$(IFS=,; printf '%s' "${groups[*]}")"
  local log_file="${LOG_ROOT}/enrich_${SLURM_JOB_ID}_gpu${gpu}_${joined_groups//,/__}.log"

  echo "[$(date '+%F %T')] job=${SLURM_JOB_ID} gpu=${gpu} groups=${joined_groups} host=$(hostname)" | tee -a "$log_file"
  for group in "${groups[@]}"; do
    echo "[$(date '+%F %T')] start group=${group}" | tee -a "$log_file"
    CUDA_VISIBLE_DEVICES="$gpu" USE_TF=0 PYTHONUNBUFFERED=1 \
      python scripts/enrich_sdxl_final_with_sdipc.py \
        "${COMMON_ARGS[@]}" "$group" \
        >>"$log_file" 2>&1
    echo "[$(date '+%F %T')] done group=${group}" | tee -a "$log_file"
  done
}

run_worker 0 "${GROUPS0[@]}" &
PID0=$!
run_worker 1 "${GROUPS1[@]}" &
PID1=$!

EC=0
wait "$PID0" || EC=1
wait "$PID1" || EC=1

if [[ "$EC" -ne 0 ]]; then
  echo "One or more enrichment workers failed (job ${SLURM_JOB_ID})." >&2
  exit "$EC"
fi

echo "Completed batch ${BATCH_ID}: gpu0=$(IFS=,; echo "${GROUPS0[*]}") gpu1=$(IFS=,; echo "${GROUPS1[*]}")"
