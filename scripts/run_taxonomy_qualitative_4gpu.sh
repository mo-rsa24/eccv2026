#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_taxonomy_qualitative_4gpu.sh
#
# Multi-node launcher for scripts/run_taxonomy_qualitative.py.
# It mirrors run_gap_4gpu.sh, but shards by taxonomy group instead of pair range.
#
# Default layout for the Group-3 sub-regime qualitative run:
#   mscluster106 GPU 0 : group3a_missing_support
#   mscluster106 GPU 1 : group3b_entanglement
#   mscluster107 GPU 0 : group3c_interference
#
# Usage:
#   bash scripts/run_taxonomy_qualitative_4gpu.sh [OUTPUT_DIR]
#
# Override behavior via environment:
#   GROUPS="group3a_missing_support group3b_entanglement group3c_interference" \
#   SEED=4 STEPS=50 SCALE=7.5 CONDA_ENV=co3 \
#   EXTRA_ARGS="--with-co3 --co3-filename co3_sd14.png" \
#   bash scripts/run_taxonomy_qualitative_4gpu.sh /path/to/out
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${1:-${ROOT_DIR}/experiments/eccv2026/taxonomy_qualitative_parallel}"
CONDA_ENV="${CONDA_ENV:-co3}"
SEED="${SEED:-4}"
STEPS="${STEPS:-50}"
SCALE="${SCALE:-7.5}"
MODEL="${MODEL:-CompVis/stable-diffusion-v1-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Default to the exact subgroup run referenced by the user.
# Avoid Bash's readonly special variable GROUPS.
TAXONOMY_GROUPS_STR="${TAXONOMY_GROUPS:-group3a_missing_support group3b_entanglement group3c_interference}"
read -r -a TAXONOMY_GROUPS <<< "$TAXONOMY_GROUPS_STR"

# Worker slots in launch order.
WORKER_NODES=(mscluster106 mscluster106 mscluster107 mscluster107)
WORKER_GPUS=(0 1 0 1)

if [[ ${#TAXONOMY_GROUPS[@]} -eq 0 ]]; then
    echo "No groups specified."
    exit 1
fi

if [[ ${#TAXONOMY_GROUPS[@]} -gt ${#WORKER_NODES[@]} ]]; then
    echo "Requested ${#TAXONOMY_GROUPS[@]} groups but only ${#WORKER_NODES[@]} worker slots are configured."
    echo "Set TAXONOMY_GROUPS to at most ${#WORKER_NODES[@]} groups, or extend WORKER_NODES/WORKER_GPUS."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Output: $OUTPUT_DIR"
echo "Seed: $SEED  Steps: $STEPS  Scale: $SCALE"
echo "Groups: ${TAXONOMY_GROUPS[*]}"
echo "Launching ${#TAXONOMY_GROUPS[@]} worker(s) ..."

worker_cmd() {
    local node=$1 gpu=$2 group_name=$3
    local log_file="${OUTPUT_DIR}/worker_${node}_gpu${gpu}_${group_name}.log"
    # shellcheck disable=SC2016
    echo "ssh ${node} 'cd ${ROOT_DIR} && CUDA_VISIBLE_DEVICES=${gpu} USE_TF=0 \
        conda run --no-capture-output -n ${CONDA_ENV} \
        python scripts/run_taxonomy_qualitative.py \
        --groups ${group_name} \
        --seed ${SEED} \
        --steps ${STEPS} \
        --scale ${SCALE} \
        --model ${MODEL} \
        --output-dir ${OUTPUT_DIR} \
        ${EXTRA_ARGS} \
        > ${log_file} 2>&1'"
}

PIDS=()
LOGS=()
for idx in "${!TAXONOMY_GROUPS[@]}"; do
    node="${WORKER_NODES[$idx]}"
    gpu="${WORKER_GPUS[$idx]}"
    group_name="${TAXONOMY_GROUPS[$idx]}"
    log_file="${OUTPUT_DIR}/worker_${node}_gpu${gpu}_${group_name}.log"
    eval "$(worker_cmd "$node" "$gpu" "$group_name")" &
    pid=$!
    PIDS+=("$pid")
    LOGS+=("$log_file")
    echo "  ${node} GPU${gpu}  ${group_name}  (PID ${pid})"
done

echo ""
echo "Tail logs with:"
for log_file in "${LOGS[@]}"; do
    echo "  tail -f ${log_file}"
done
echo ""

EC=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || { echo "Worker PID $pid failed (exit $?)"; EC=1; }
done

if [[ $EC -eq 0 ]]; then
    echo "All workers finished successfully."
else
    echo "One or more workers failed. Check logs above."
    exit 1
fi
