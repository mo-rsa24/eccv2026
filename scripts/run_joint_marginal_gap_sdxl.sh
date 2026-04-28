#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_joint_marginal_gap_sdxl.sh — re-run all 4 joint-marginal diagnostic pairs
#
# Runs the 4 pairs from the original experiments sequentially on a single GPU:
#   1. "A dog on the left of the photo" / "A cat on the right of the photo"  (Group 3: spatial)
#   2. "a butterfly" / "a flower meadow"                                      (Group 1/2: factorized)
#   3. "a dog" / "oil painting style"                                         (Group 2: factorized style)
#   4. "a picnic table" / "a snowstorm"                                       (Group 3/4: context collision)
#
# Each run is independent; on failure the script continues and reports at the end.
#
# Usage:
#   bash scripts/run_joint_marginal_gap_sdxl.sh
#
# Environment overrides:
#   CONDA_ENV=jaxstack
#   CONDA_EXE=~/miniforge3/bin/conda
#   CUDA_DEVICE=0
#   SLURM_PARTITION=biggpu
#   SLURM_TIME=12:00:00
#   SLURM_NODE=mscluster109   # pin to a specific node
#   EXTRA_ARGS="--num-inference-steps 20"
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${ROOT_DIR}/experiments/eccv2026/joint_marginal_gap_sdxl"
CONDA_ENV="${CONDA_ENV:-jaxstack}"
CONDA_EXE="${CONDA_EXE:-${HOME}/miniforge3/bin/conda}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SLURM_PARTITION="${SLURM_PARTITION:-biggpu}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

if [[ ! -x "$CONDA_EXE" ]]; then
    echo "Conda executable not found or not executable: $CONDA_EXE" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Pair definitions: (prompt_a, prompt_b, joint_prompt)
# ---------------------------------------------------------------------------
declare -a PROMPT_A=(
    "A dog on the left of the photo"
    "a butterfly"
    "a dog"
    "a picnic table"
)
declare -a PROMPT_B=(
    "A cat on the right of the photo"
    "a flower meadow"
    "oil painting style"
    "a snowstorm"
)
declare -a JOINT_PROMPT=(
    "A dog on the left of the photo and a cat on the right of the photo"
    "a butterfly in a flower meadow"
    "an oil painting of a dog"
    "a picnic table in a snowstorm"
)

# ---------------------------------------------------------------------------
# Build the heredoc body that will run all 4 pairs
# ---------------------------------------------------------------------------
JOB_NAME="jmgap-sdxl-rerun"
SLURM_OUT="${LOG_DIR}/${JOB_NAME}-%j.out"
SLURM_ERR="${LOG_DIR}/${JOB_NAME}-%j.err"

SLURM_NODE="${SLURM_NODE:-}"
NODE_ARGS=()
if [[ -n "$SLURM_NODE" ]]; then
    NODE_ARGS=(--nodelist "$SLURM_NODE")
fi

JOB_ID=$(sbatch \
    --parsable \
    --job-name "$JOB_NAME" \
    --partition "$SLURM_PARTITION" \
    --time "$SLURM_TIME" \
    --ntasks 1 \
    --cpus-per-task "$SLURM_CPUS_PER_TASK" \
    --output "$SLURM_OUT" \
    --error "$SLURM_ERR" \
    "${NODE_ARGS[@]}" <<EOF
#!/usr/bin/env bash
set -uo pipefail

cd "$ROOT_DIR"

FAILED=0

run_pair() {
    local idx=\$1
    local pa=\$2
    local pb=\$3
    local pj=\$4
    local log_file="${LOG_DIR}/${JOB_NAME}-pair\${idx}.log"

    echo "[\\$(date '+%F %T')] Starting pair \${idx}: '\${pa}' / '\${pb}'" | tee -a "\$log_file"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" USE_TF=0 \
        "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
        python scripts/diagnose_joint_marginal_gap_sdxl.py \
            --prompt-a "\${pa}" \
            --prompt-b "\${pb}" \
            --joint-prompt "\${pj}" \
            --seed 42 \
            --num-inference-steps 50 \
            --guidance-scale 7.5 \
            --height 1024 \
            --width 1024 \
            --output-dir "$OUTPUT_DIR" \
            --device cuda \
            $EXTRA_ARGS \
        >> "\$log_file" 2>&1 \
        && echo "[\\$(date '+%F %T')] Pair \${idx} done." | tee -a "\$log_file" \
        || { echo "[\\$(date '+%F %T')] Pair \${idx} FAILED." | tee -a "\$log_file"; return 1; }
}

run_pair 1 "${PROMPT_A[0]}" "${PROMPT_B[0]}" "${JOINT_PROMPT[0]}" || FAILED=1
run_pair 2 "${PROMPT_A[1]}" "${PROMPT_B[1]}" "${JOINT_PROMPT[1]}" || FAILED=1
run_pair 3 "${PROMPT_A[2]}" "${PROMPT_B[2]}" "${JOINT_PROMPT[2]}" || FAILED=1
run_pair 4 "${PROMPT_A[3]}" "${PROMPT_B[3]}" "${JOINT_PROMPT[3]}" || FAILED=1

if [[ \$FAILED -ne 0 ]]; then
    echo "One or more pairs failed. Check per-pair logs in $LOG_DIR." >&2
    exit 1
fi

echo "All 4 pairs completed successfully. Results in $OUTPUT_DIR"
EOF
)

echo "Submitted job ${JOB_ID}"
echo ""
echo "Monitor with:"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out"
echo "  tail -f ${LOG_DIR}/${JOB_NAME}-pair1.log"
echo "  tail -f ${LOG_DIR}/${JOB_NAME}-pair2.log"
echo "  tail -f ${LOG_DIR}/${JOB_NAME}-pair3.log"
echo "  tail -f ${LOG_DIR}/${JOB_NAME}-pair4.log"
echo ""
echo "Results will appear under:"
echo "  $OUTPUT_DIR"
