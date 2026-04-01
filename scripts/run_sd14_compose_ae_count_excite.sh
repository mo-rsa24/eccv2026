#!/usr/bin/env bash
set -euo pipefail

# SD 1.4 Composable AND + Attend-and-Excite (A&E) runner
# focused on count-token excitation (e.g., "2 cats and 3 dogs").
#
# Run:
#   bash scripts/run_sd14_compose_ae_count_excite.sh
#
# First inspect token indices:
#   SHOW_TOKEN_MAP=1 bash scripts/run_sd14_compose_ae_count_excite.sh
#
# Override example:
#   JOINT_PROMPT="3 dogs and 4 cats in a park" \
#   SUBJECT_PROMPTS="3 dogs | 4 cats" \
#   INDICES_TO_ALTER="1,2,4,5" \
#   bash scripts/run_sd14_compose_ae_count_excite.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$ROOT_DIR/compositions/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch"
SCRIPT_PATH="$REPO_DIR/scripts/image_sample_compose_and_ae_stable_diffusion.py"

JOINT_PROMPT="${JOINT_PROMPT:-2 cats and 3 dogs in a field}"
SUBJECT_PROMPTS="${SUBJECT_PROMPTS:-2 cats | 3 dogs}"

# 1-based token indices in JOINT_PROMPT for A&E.
# Default indices commonly map to: "2", "cats", "3", "dogs".
INDICES_TO_ALTER="${INDICES_TO_ALTER:-1,2,4,5}"

COMP_WEIGHTS="${COMP_WEIGHTS:-7.5 | 7.5}"
STEPS="${STEPS:-50}"
SCALE="${SCALE:-7.5}"
SEED="${SEED:-42}"
NUM_IMAGES="${NUM_IMAGES:-1}"
MODEL_PATH="${MODEL_PATH:-CompVis/stable-diffusion-v1-4}"
SCHEDULER="${SCHEDULER:-ddim}"
MAX_ITER_TO_ALTER="${MAX_ITER_TO_ALTER:-25}"
LOSS_MODE="${LOSS_MODE:-sum}"

# Keep pure AND+A&E by default (no monolithic blend in-loop).
ALPHA="${ALPHA:-1.0}"
ALPHA_END="${ALPHA_END:-1.0}"
ALPHA_SCHEDULE="${ALPHA_SCHEDULE:-constant}"

# Set SHOW_TOKEN_MAP=1 to print tokenizer index map and exit.
SHOW_TOKEN_MAP="${SHOW_TOKEN_MAP:-0}"

CMD=(
  python "$SCRIPT_PATH"
  --joint_prompt "$JOINT_PROMPT"
  --subject_prompts "$SUBJECT_PROMPTS"
  --indices_to_alter "$INDICES_TO_ALTER"
  --comp_weights "$COMP_WEIGHTS"
  --steps "$STEPS"
  --scale "$SCALE"
  --seed "$SEED"
  --num_images "$NUM_IMAGES"
  --model_path "$MODEL_PATH"
  --scheduler "$SCHEDULER"
  --max_iter_to_alter "$MAX_ITER_TO_ALTER"
  --loss_mode "$LOSS_MODE"
  --alpha "$ALPHA"
  --alpha_end "$ALPHA_END"
  --alpha_schedule "$ALPHA_SCHEDULE"
)

if [[ "$SHOW_TOKEN_MAP" == "1" ]]; then
  CMD+=(--show_token_map)
fi

echo "Running SD1.4 AND + A&E count excitation:"
echo "  joint prompt   : $JOINT_PROMPT"
echo "  subject prompts: $SUBJECT_PROMPTS"
echo "  indices        : $INDICES_TO_ALTER"
echo "  show token map : $SHOW_TOKEN_MAP"
echo

(
  cd "$REPO_DIR"
  "${CMD[@]}"
)

if [[ "$SHOW_TOKEN_MAP" != "1" ]]; then
  echo
  echo "Done. Output image saved under:"
  echo "  $REPO_DIR"
fi
