#!/usr/bin/env bash
set -euo pipefail

# SD3.5 guided hybrid:
#   v_final = (1-alpha) * v_monolithic + alpha * v_superdiff_and
# with count-focused sub-prompts to "excite" numerosity.
#
# Run from repo root:
#   bash scripts/run_sd35_guided_count_excite.sh
#
# Override with env vars, e.g.:
#   PROMPTS="2 cats|two cats|3 dogs|three dogs" \
#   MONOLITHIC="2 cats and 3 dogs in a field" \
#   ALPHAS="0.2 0.4 0.7" \
#   OUTPUT_DIR="experiments/trajectory_dynamics/count_excite" \
#   bash scripts/run_sd35_guided_count_excite.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/scripts/trajectory_dynamics_experiment.py"

MODEL_ID="${MODEL_ID:-stabilityai/stable-diffusion-3.5-medium}"
STEPS="${STEPS:-50}"
GUIDANCE="${GUIDANCE:-4.5}"
SEED="${SEED:-42}"
NUM_SEEDS="${NUM_SEEDS:-1}"
LIFT="${LIFT:-0.0}"
SUPERDIFF_VARIANT="${SUPERDIFF_VARIANT:-fm_ode}"
PROJECTION="${PROJECTION:-pca}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/experiments/trajectory_dynamics/count_excite}"
NO_CLIP_PROBE="${NO_CLIP_PROBE:-1}"

# Pipe-separated sub-prompts used in AND composition.
# Include numeric and lexical count forms by default.
PROMPTS="${PROMPTS:-2 cats|two cats|3 dogs|three dogs}"

# Monolithic SD3.5 prompt used for the anchor branch in guided mode.
MONOLITHIC="${MONOLITHIC:-2 cats and 3 dogs in a field}"

# Space-separated alphas for guided blending.
ALPHAS="${ALPHAS:-0.2 0.4 0.7}"

mkdir -p "$OUTPUT_DIR"

readarray -t PROMPT_ARR < <(
  printf '%s\n' "$PROMPTS" \
  | tr '|' '\n' \
  | sed -E 's/^[[:space:]]+|[[:space:]]+$//g' \
  | sed '/^$/d'
)

if [[ "${#PROMPT_ARR[@]}" -lt 2 ]]; then
  echo "Need at least 2 prompts in PROMPTS. Got: ${#PROMPT_ARR[@]}"
  exit 1
fi

read -r -a ALPHA_ARR <<< "$ALPHAS"
if [[ "${#ALPHA_ARR[@]}" -lt 1 ]]; then
  echo "ALPHAS is empty."
  exit 1
fi

CMD=(
  python "$SCRIPT_PATH"
  --prompts "${PROMPT_ARR[@]}"
  --monolithic "$MONOLITHIC"
  --guided
  --alpha "${ALPHA_ARR[@]}"
  --model-id "$MODEL_ID"
  --steps "$STEPS"
  --guidance "$GUIDANCE"
  --seed "$SEED"
  --num-seeds "$NUM_SEEDS"
  --lift "$LIFT"
  --superdiff-variant "$SUPERDIFF_VARIANT"
  --projection "$PROJECTION"
  --output-dir "$OUTPUT_DIR"
)

if [[ "$NO_CLIP_PROBE" == "1" ]]; then
  CMD+=(--no-clip-probe)
fi

echo "Running SD3.5 guided count-excitation experiment:"
echo "  prompts    : ${PROMPT_ARR[*]}"
echo "  monolithic : $MONOLITHIC"
echo "  alphas     : ${ALPHA_ARR[*]}"
echo "  output_dir : $OUTPUT_DIR"
echo

"${CMD[@]}"

echo
echo "Done. Results saved under: $OUTPUT_DIR"
