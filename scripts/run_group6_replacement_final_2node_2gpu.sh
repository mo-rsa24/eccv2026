#!/usr/bin/env bash
# Run the frozen replacement Group 6 across two nodes with two visible GPUs each.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NODE_ROLE="${NODE_ROLE:-}"
MODE="${MODE:-run}"

EDITED_FINAL_DIR="${SDXL_EDIT_FINAL_DIR:-${ROOT_DIR}/experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_group6_edit}"
EDITED_SDIPC_DIR="${SDXL_EDIT_SDIPC_DIR:-${ROOT_DIR}/experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_group6_edit_sdipc}"
REPLACEMENT_ROSTER="${SDXL_GROUP6_EDIT_ROSTER:-${ROOT_DIR}/experiments/eccv2026/sdxl_final/group6_replacement_roster.json}"
CONDA_ENV="${CONDA_ENV:-co3}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
KEEP_CLIP_ON_GPU="${KEEP_CLIP_ON_GPU:-1}"
COPY_MODE="${COPY_MODE:-symlink}"
LOG_ROOT="${LOG_ROOT:-${EDITED_SDIPC_DIR}/logs/group6_replacement_final}"
GROUP_KEY="${GROUP_KEY:-group6_coherent_collision}"
GRID_SEED="${SDXL_GRID_SEED:-42}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
PROJECTION="${PROJECTION:-mds}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"

if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "Conda activation script not found: ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 1
fi

if [[ ! -f "$REPLACEMENT_ROSTER" ]]; then
  echo "Missing replacement roster: $REPLACEMENT_ROSTER" >&2
  exit 1
fi

if [[ -z "${SDXL_FINAL_SEEDS:-}" ]]; then
  echo "Missing SDXL_FINAL_SEEDS in environment." >&2
  exit 1
fi

# shellcheck disable=SC2206
ALL_SEEDS=(${SDXL_FINAL_SEEDS})
if [[ "${#ALL_SEEDS[@]}" -eq 0 ]]; then
  echo "No seeds found in SDXL_FINAL_SEEDS." >&2
  exit 1
fi

mkdir -p "$LOG_ROOT"

assign_seeds_to_shards() {
  local -n shard0_ref=$1
  local -n shard1_ref=$2
  local -n shard2_ref=$3
  local -n shard3_ref=$4
  shard0_ref=()
  shard1_ref=()
  shard2_ref=()
  shard3_ref=()
  local idx seed
  for idx in "${!ALL_SEEDS[@]}"; do
    seed="${ALL_SEEDS[$idx]}"
    case $(( idx % 4 )) in
      0) shard0_ref+=("$seed") ;;
      1) shard1_ref+=("$seed") ;;
      2) shard2_ref+=("$seed") ;;
      3) shard3_ref+=("$seed") ;;
    esac
  done
}

join_by_space() {
  local first=1
  local item
  for item in "$@"; do
    if [[ $first -eq 1 ]]; then
      printf '%s' "$item"
      first=0
    else
      printf ' %s' "$item"
    fi
  done
}

run_worker() {
  local gpu="$1"
  shift
  local seeds=("$@")
  local seeds_joined
  seeds_joined="$(join_by_space "${seeds[@]}")"
  local log_file="${LOG_ROOT}/group6_${NODE_ROLE}_gpu${gpu}.log"

  if [[ "${#seeds[@]}" -eq 0 ]]; then
    echo "[$(date '+%F %T')] node=${NODE_ROLE} gpu=${gpu} no assigned seeds; skipping" | tee -a "$log_file"
    return 0
  fi

  echo "[$(date '+%F %T')] node=${NODE_ROLE} gpu=${gpu} render+sdipc seeds=${seeds_joined}" | tee -a "$log_file"

  local keep_clip_flag=""
  if [[ "$KEEP_CLIP_ON_GPU" == "1" ]]; then
    keep_clip_flag="--keep-clip-on-gpu"
  fi

  (
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"

    CUDA_VISIBLE_DEVICES="$gpu" USE_TF=0 PYTHONUNBUFFERED=1 \
      python scripts/run_group6_replacement_final_sdxl.py \
        --edited-final-dir "$EDITED_FINAL_DIR" \
        --edited-sdipc-dir "$EDITED_SDIPC_DIR" \
        --replacement-roster "$REPLACEMENT_ROSTER" \
        --seeds "${seeds[@]}" \
        --grid-seed "$GRID_SEED" \
        --num-inference-steps "$NUM_INFERENCE_STEPS" \
        --guidance-scale "$GUIDANCE_SCALE" \
        --projection "$PROJECTION" \
        --height "$HEIGHT" \
        --width "$WIDTH" \
        --gpu-id 0 \
        --skip-manifest-update

    if [[ "$KEEP_CLIP_ON_GPU" == "1" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" USE_TF=0 PYTHONUNBUFFERED=1 \
        python scripts/enrich_sdxl_final_with_sdipc.py \
          --source-dir "$EDITED_FINAL_DIR" \
          --output-dir "$EDITED_SDIPC_DIR" \
          --groups "$GROUP_KEY" \
          --seeds "${seeds[@]}" \
          --grid-seed "$GRID_SEED" \
          --gpu-id 0 \
          --copy-mode "$COPY_MODE" \
          --keep-clip-on-gpu
    else
      CUDA_VISIBLE_DEVICES="$gpu" USE_TF=0 PYTHONUNBUFFERED=1 \
        python scripts/enrich_sdxl_final_with_sdipc.py \
          --source-dir "$EDITED_FINAL_DIR" \
          --output-dir "$EDITED_SDIPC_DIR" \
          --groups "$GROUP_KEY" \
          --seeds "${seeds[@]}" \
          --grid-seed "$GRID_SEED" \
          --gpu-id 0 \
          --copy-mode "$COPY_MODE"
    fi
  ) 2>&1 | tee -a "$log_file"

  echo "[$(date '+%F %T')] node=${NODE_ROLE} gpu=${gpu} done seeds=${seeds_joined}" | tee -a "$log_file"
}

sync_manifests() {
  echo "Synchronizing edited manifests from frozen replacement roster..."
  (
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    # shellcheck disable=SC2206
    sync_seeds=(${SDXL_FINAL_SEEDS})
    USE_TF=0 PYTHONUNBUFFERED=1 \
      python scripts/run_group6_replacement_final_sdxl.py \
        --edited-final-dir "$EDITED_FINAL_DIR" \
        --edited-sdipc-dir "$EDITED_SDIPC_DIR" \
        --replacement-roster "$REPLACEMENT_ROSTER" \
        --seeds "${sync_seeds[@]}" \
        --grid-seed "$GRID_SEED" \
        --num-inference-steps "$NUM_INFERENCE_STEPS" \
        --guidance-scale "$GUIDANCE_SCALE" \
        --sync-manifests-only
  )
}

if [[ "$MODE" == "sync" ]]; then
  sync_manifests
  exit 0
fi

if [[ "$NODE_ROLE" != "node106" && "$NODE_ROLE" != "node107" ]]; then
  echo "Set NODE_ROLE=node106 or NODE_ROLE=node107." >&2
  exit 1
fi

assign_seeds_to_shards SHARD0 SHARD1 SHARD2 SHARD3

if [[ "$NODE_ROLE" == "node106" ]]; then
  GPU0_SEEDS=("${SHARD0[@]}")
  GPU1_SEEDS=("${SHARD1[@]}")
else
  GPU0_SEEDS=("${SHARD2[@]}")
  GPU1_SEEDS=("${SHARD3[@]}")
fi

echo "Node role: $NODE_ROLE"
echo "Edited final: $EDITED_FINAL_DIR"
echo "Edited SD-IPC: $EDITED_SDIPC_DIR"
echo "Roster: $REPLACEMENT_ROSTER"
echo "Conda env: $CONDA_ENV"
echo "Logs: $LOG_ROOT"
echo "GPU0 seeds: $(join_by_space "${GPU0_SEEDS[@]}")"
echo "GPU1 seeds: $(join_by_space "${GPU1_SEEDS[@]}")"

run_worker 0 "${GPU0_SEEDS[@]}" &
PID0=$!
run_worker 1 "${GPU1_SEEDS[@]}" &
PID1=$!

EC=0
wait "$PID0" || EC=1
wait "$PID1" || EC=1

if [[ "$EC" -ne 0 ]]; then
  echo "One or more Group 6 workers failed on ${NODE_ROLE}." >&2
  exit "$EC"
fi

echo "Completed Group 6 replacement work on ${NODE_ROLE}."
echo "After both nodes finish, run once:"
echo "  MODE=sync bash scripts/run_group6_replacement_final_2node_2gpu.sh"
