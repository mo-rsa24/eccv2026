#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R3_DIR="$ROOT_DIR/compositions/reduce_reuse_recycle"

OUT_DIR="${1:-$ROOT_DIR/experiments/eccv2026/reduce_reuse_recycle}"

CKPT_PATH="${CKPT_PATH:-$R3_DIR/ebm-49x1874.pt}"
CONCEPT_A="${CONCEPT_A:-sphere}"
CONCEPT_B="${CONCEPT_B:-cylinder}"
SAMPLER="${SAMPLER:-MALA}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_RUNS="${NUM_RUNS:-4}"
SEED="${SEED:-0}"
OUT_FILE="${OUT_FILE:-$OUT_DIR/${CONCEPT_A}_and_${CONCEPT_B}_${SAMPLER}.png}"

mkdir -p "$OUT_DIR"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint not found: $CKPT_PATH" >&2
  echo "Set CKPT_PATH to your trained EBM checkpoint path before running." >&2
  exit 1
fi

(
  cd "$R3_DIR"
  python inf_sample.py \
    --ckpt_path "$CKPT_PATH" \
    --sampler "$SAMPLER" \
    --concept-a "$CONCEPT_A" \
    --concept-b "$CONCEPT_B" \
    --guidance-scale "$GUIDANCE_SCALE" \
    --batch-size "$BATCH_SIZE" \
    --num-runs "$NUM_RUNS" \
    --seed "$SEED" \
    --out "$OUT_FILE"
)

echo "Done. Output saved to: $OUT_FILE"
