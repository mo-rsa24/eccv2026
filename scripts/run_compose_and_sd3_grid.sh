#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/compositions/super-diffusion/scripts/compose_and_sd3.py"

OUT_DIR="${1:-$ROOT_DIR/experiments/eccv2026/grid_figure/compose_and_sd3}"
MODE="${MODE:-chimera}"
STEPS="${STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-21}"
SCALE="${SCALE:-4.5}"

mkdir -p "$OUT_DIR"

slugify() {
  echo "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

PAIRS=(
  "a donut|rainbow sprinkles"
  "a red sports car|green tires"
  "a white mug|blue polka dots"
  "a black umbrella|neon pink trim"
)

total="${#PAIRS[@]}"
for i in "${!PAIRS[@]}"; do
  pair="${PAIRS[$i]}"
  obj="${pair%%|*}"
  bg="${pair#*|}"
  obj_slug="$(slugify "$obj")"
  bg_slug="$(slugify "$bg")"
  out_file="$OUT_DIR/$(printf '%02d' "$((i + 1))")_${obj_slug}__${bg_slug}.png"

  echo "[$((i + 1))/$total] obj='$obj' bg='$bg' -> $out_file"
  python "$SCRIPT_PATH" \
    --obj "$obj" \
    --bg "$bg" \
    --mode "$MODE" \
    --steps "$STEPS" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --scale "$SCALE" \
    --out "$out_file"
done

echo "Done. Images saved to: $OUT_DIR"
