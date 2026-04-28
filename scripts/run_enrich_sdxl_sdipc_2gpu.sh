#!/usr/bin/env bash
# Run SDXL SD-IPC enrichment locally on a single node using GPUs 0 and 1.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOURCE_DIR="${1:-${SDXL_FINAL_DIR:-${ROOT_DIR}/experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_frozen}}"
OUTPUT_DIR="${2:-${SDXL_SDIPC_DIR:-${ROOT_DIR}/experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_frozen_sdipc}}"

CONDA_ENV="${CONDA_ENV:-co3}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
COPY_MODE="${COPY_MODE:-symlink}"
KEEP_CLIP_ON_GPU="${KEEP_CLIP_ON_GPU:-1}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_DIR}/logs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_SPLIT="${RUN_SPLIT:-single_node}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_ROOT"

if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "Conda activation script not found: ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 1
fi

SEEDS_ARGS=()
if [[ -n "${SDXL_FINAL_SEEDS:-}" ]]; then
  # shellcheck disable=SC2206
  SEEDS_LIST=(${SDXL_FINAL_SEEDS})
  SEEDS_ARGS=(--seeds "${SEEDS_LIST[@]}")
fi
if [[ -n "${SDXL_GRID_SEED:-}" ]]; then
  SEEDS_ARGS+=(--grid-seed "${SDXL_GRID_SEED}")
fi

COMMON_ARGS=(
  --source-dir "$SOURCE_DIR"
  --output-dir "$OUTPUT_DIR"
  --copy-mode "$COPY_MODE"
)
COMMON_ARGS+=("${SEEDS_ARGS[@]}")

if [[ "$KEEP_CLIP_ON_GPU" == "1" ]]; then
  COMMON_ARGS+=(--keep-clip-on-gpu)
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=($EXTRA_ARGS)
  COMMON_ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

resolve_groups() {
  local which="$1"
  local override_csv=""
  local -a resolved=()

  if [[ "$which" == "gpu0" ]]; then
    override_csv="${GPU0_GROUPS_CSV:-}"
  else
    override_csv="${GPU1_GROUPS_CSV:-}"
  fi

  if [[ -n "$override_csv" ]]; then
    IFS=',' read -r -a resolved <<< "$override_csv"
    printf '%s\n' "${resolved[@]}"
    return
  fi

  case "$RUN_SPLIT" in
    single_node)
      if [[ "$which" == "gpu0" ]]; then
        resolved=(
          "group1_cooccurrence"
          "group2_factorization"
          "group3_role_separable_object_scene"
        )
      else
        resolved=(
          "group4_dual_object_composition"
          "group5_concept_prior_entanglement"
          "group6_coherent_collision"
        )
      fi
      ;;
    node106)
      if [[ "$which" == "gpu0" ]]; then
        resolved=(
          "group1_cooccurrence"
          "group2_factorization"
        )
      else
        resolved=(
          "group3_role_separable_object_scene"
        )
      fi
      ;;
    node107)
      if [[ "$which" == "gpu0" ]]; then
        resolved=(
          "group4_dual_object_composition"
          "group5_concept_prior_entanglement"
        )
      else
        resolved=(
          "group6_coherent_collision"
        )
      fi
      ;;
    *)
      echo "Unknown RUN_SPLIT: $RUN_SPLIT" >&2
      echo "Expected one of: single_node, node106, node107" >&2
      exit 1
      ;;
  esac

  printf '%s\n' "${resolved[@]}"
}

mapfile -t GPU0_GROUPS < <(resolve_groups gpu0)
mapfile -t GPU1_GROUPS < <(resolve_groups gpu1)

if [[ "${#GPU0_GROUPS[@]}" -eq 0 && "${#GPU1_GROUPS[@]}" -eq 0 ]]; then
  echo "No groups assigned to either GPU." >&2
  exit 1
fi

run_worker() {
  local gpu="$1"
  shift
  local groups=("$@")
  if [[ "${#groups[@]}" -eq 0 ]]; then
    echo "[$(date '+%F %T')] gpu=${gpu} has no assigned groups; skipping."
    return 0
  fi
  local joined_groups
  joined_groups="$(IFS=,; printf '%s' "${groups[*]}")"
  local log_file="${LOG_ROOT}/enrich_local_${RUN_SPLIT}_gpu${gpu}_${joined_groups//,/__}.log"

  echo "[$(date '+%F %T')] gpu=${gpu} groups=${joined_groups} host=$(hostname)" | tee -a "$log_file"
  for group in "${groups[@]}"; do
    echo "[$(date '+%F %T')] start group=${group}" | tee -a "$log_file"
    CUDA_VISIBLE_DEVICES="$gpu" USE_TF=0 PYTHONUNBUFFERED=1 \
      bash -lc "source \"${CONDA_BASE}/etc/profile.d/conda.sh\" && conda activate \"${CONDA_ENV}\" && python scripts/enrich_sdxl_final_with_sdipc.py ${COMMON_ARGS[*]@Q} --groups ${group@Q}" \
      2>&1 | tee -a "$log_file"
    echo "[$(date '+%F %T')] done group=${group}" | tee -a "$log_file"
  done
}

echo "Source: $SOURCE_DIR"
echo "Output: $OUTPUT_DIR"
echo "Logs: $LOG_ROOT"
echo "Conda env: $CONDA_ENV"
echo "Run split: $RUN_SPLIT"
echo "GPU 0 groups: ${GPU0_GROUPS[*]}"
echo "GPU 1 groups: ${GPU1_GROUPS[*]}"

run_worker 0 "${GPU0_GROUPS[@]}" &
PID0=$!
run_worker 1 "${GPU1_GROUPS[@]}" &
PID1=$!

EC=0
wait "$PID0" || EC=1
wait "$PID1" || EC=1

if [[ "$EC" -ne 0 ]]; then
  echo "One or more local enrichment workers failed." >&2
  exit "$EC"
fi

echo "Completed local SDXL SD-IPC enrichment on GPUs 0 and 1."
