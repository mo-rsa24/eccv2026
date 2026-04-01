#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$ROOT_DIR/compositions/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch"
SCRIPT_PATH="$REPO_DIR/scripts/image_sample_compose_stable_diffusion.py"

PROMPT_A="${PROMPT_A:-a cat}"
PROMPT_B="${PROMPT_B:-a dog}"
PROMPTS="${PROMPTS:-$PROMPT_A | $PROMPT_B}"
WEIGHTS="${WEIGHTS:-7.5 | 7.5}"

STEPS="${STEPS:-50}"
SCALE="${SCALE:-7.5}"
SEED="${SEED:-42}"
NUM_IMAGES="${NUM_IMAGES:-1}"
MODEL_PATH="${MODEL_PATH:-CompVis/stable-diffusion-v1-4}"
SCHEDULER="${SCHEDULER:-ddim}"

R3_SAMPLER="${R3_SAMPLER:-ula}"
R3_ULA_STEPS="${R3_ULA_STEPS:-1}"
R3_ULA_STEP_SCALE="${R3_ULA_STEP_SCALE:-2.0}"
R3_ULA_T_MIN="${R3_ULA_T_MIN:-500}"
R3_ULA_NOISE_SCALE="${R3_ULA_NOISE_SCALE:-1.0}"

(
  cd "$REPO_DIR"
  python "$SCRIPT_PATH" \
    --prompts "$PROMPTS" \
    --weights "$WEIGHTS" \
    --steps "$STEPS" \
    --scale "$SCALE" \
    --seed "$SEED" \
    --num_images "$NUM_IMAGES" \
    --model_path "$MODEL_PATH" \
    --scheduler "$SCHEDULER" \
    --r3_sampler "$R3_SAMPLER" \
    --r3_ula_steps "$R3_ULA_STEPS" \
    --r3_ula_step_scale "$R3_ULA_STEP_SCALE" \
    --r3_ula_t_min "$R3_ULA_T_MIN" \
    --r3_ula_noise_scale "$R3_ULA_NOISE_SCALE"
)

echo "Done. Check generated image(s) in: $REPO_DIR"
