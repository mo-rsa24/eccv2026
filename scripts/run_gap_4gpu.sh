#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_gap_4gpu.sh — split --regime large across 4 GPUs on 2 SLURM nodes
#
# Layout (24 pairs ÷ 4 workers = 6 pairs each):
#   mscluster107 GPU 0 : pairs  0–5
#   mscluster107 GPU 1 : pairs  6–11
#   mscluster109 GPU 0 : pairs 12–17
#   mscluster109 GPU 1 : pairs 18–23
#
# Usage:
#   bash scripts/run_gap_4gpu.sh [OUTPUT_DIR]
#
# Override measure_composability_gap.py flags via environment:
#   EXTRA_ARGS="--steps 20" bash scripts/run_gap_4gpu.sh
#
# Optional SLURM overrides:
#   SLURM_PARTITION=biggpu SLURM_TIME=72:00:00 \
#   bash scripts/run_gap_4gpu.sh
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${1:-${ROOT_DIR}/results/gap_large_4gpu}"
CONDA_ENV="${CONDA_ENV:-jaxstack}"
CONDA_EXE="${CONDA_EXE:-${HOME}/miniforge3/bin/conda}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SEED_BATCH_SIZE="${SEED_BATCH_SIZE:-8}"
SLURM_PARTITION="${SLURM_PARTITION:-biggpu}"
SLURM_TIME="${SLURM_TIME:-72:00:00}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

if [[ ! -x "$CONDA_EXE" ]]; then
    echo "Conda executable not found or not executable: $CONDA_EXE" >&2
    exit 1
fi

COMMON_ARGS=(
    --paper-only
    --regime large
    --pstar-source sdipc
    --model-family sd14
    --output-dir "$OUTPUT_DIR"
    --seed-batch-size "$SEED_BATCH_SIZE"
)

if [[ -n "$EXTRA_ARGS" ]]; then
    # Intentionally split EXTRA_ARGS on shell word boundaries.
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=($EXTRA_ARGS)
    COMMON_ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

SBATCH_COMMON=(
    --parsable
    --ntasks 1
    --cpus-per-task "$SLURM_CPUS_PER_TASK"
    --partition "$SLURM_PARTITION"
    --time "$SLURM_TIME"
)

submit_node_job() {
    local node=$1
    local pair0_start=$2
    local pair0_end=$3
    local pair1_start=$4
    local pair1_end=$5

    local job_name="gap-${node}"
    local slurm_out="${LOG_DIR}/${job_name}-%j.out"
    local slurm_err="${LOG_DIR}/${job_name}-%j.err"

    sbatch \
        "${SBATCH_COMMON[@]}" \
        --job-name "$job_name" \
        --nodelist "$node" \
        --output "$slurm_out" \
        --error "$slurm_err" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "$ROOT_DIR"
mkdir -p "$OUTPUT_DIR"

run_worker() {
    local gpu=\$1
    local pair_start=\$2
    local pair_end=\$3
    local log_file="${OUTPUT_DIR}/worker_${node}_gpu\${gpu}.log"

    echo "[$(date '+%F %T')] ${node} gpu\${gpu} pairs \${pair_start}-\$((pair_end - 1))" | tee -a "\$log_file"
    CUDA_VISIBLE_DEVICES="\$gpu" USE_TF=0 \
        "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
        python scripts/measure_composability_gap.py \
        ${COMMON_ARGS[*]} --pair-start "\$pair_start" --pair-end "\$pair_end" \
        >> "\$log_file" 2>&1
}

run_worker 0 $pair0_start $pair0_end &
PID0=\$!
run_worker 1 $pair1_start $pair1_end &
PID1=\$!

EC=0
wait "\$PID0" || EC=1
wait "\$PID1" || EC=1

if [[ \$EC -ne 0 ]]; then
    echo "One or more workers failed on ${node}" >&2
    exit \$EC
fi
EOF
}

echo "Output: $OUTPUT_DIR"
echo "Submitting 2 SLURM jobs across 4 GPUs ..."

JOB107="$(submit_node_job mscluster107 0 6 6 12)"
JOB109="$(submit_node_job mscluster109 12 18 18 24)"

echo "Submitted:"
echo "  mscluster107 GPUs 0,1 pairs  0-11  (job ${JOB107})"
echo "  mscluster109 GPUs 0,1 pairs 12-23  (job ${JOB109})"
echo ""
echo "Monitor with:"
echo "  squeue -j ${JOB107},${JOB109}"
echo "  tail -f ${LOG_DIR}/gap-mscluster107-${JOB107}.out"
echo "  tail -f ${LOG_DIR}/gap-mscluster109-${JOB109}.out"
echo "  tail -f ${OUTPUT_DIR}/worker_mscluster107_gpu0.log"
echo "  tail -f ${OUTPUT_DIR}/worker_mscluster107_gpu1.log"
echo "  tail -f ${OUTPUT_DIR}/worker_mscluster109_gpu0.log"
echo "  tail -f ${OUTPUT_DIR}/worker_mscluster109_gpu1.log"
