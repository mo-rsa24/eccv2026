#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$ROOT_DIR/compositions/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch"
SCRIPT_PATH="$REPO_DIR/scripts/image_sample_compose_stable_diffusion.py"

OUT_DIR="${1:-$ROOT_DIR/experiments/eccv2026/grid_figure/composable_diffusion_sd14}"
STEPS="${STEPS:-50}"
SCALE="${SCALE:-7.5}"
SEED="${SEED:-42}"
NUM_IMAGES="${NUM_IMAGES:-1}"
SCHEDULER="${SCHEDULER:-ddim}"
MODEL_PATH="${MODEL_PATH:-CompVis/stable-diffusion-v1-4}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUT_DIR"

if [[ -n "${WEIGHTS:-}" ]]; then
  PAIR_WEIGHTS="$WEIGHTS"
else
  PAIR_WEIGHTS="$SCALE | $SCALE"
fi

slugify() {
  echo "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

DEFAULT_PAIRS=(
  "a hot air balloon|a mountain landscape"
  "a lighthouse|a stormy sea"
  "a lion|a savanna at sunset"
  "a sailboat|cloudy blue sky"
  "a butterfly|a flower meadow"
)

if [[ -n "${PAIRS_JSON:-}" && -n "${PAIRS_FILE:-}" ]]; then
  echo "Set only one of PAIRS_JSON or PAIRS_FILE." >&2
  exit 1
fi

declare -a PAIRS

if [[ -n "${PAIRS_JSON:-}" ]]; then
  mapfile -t PAIRS < <(
    python - "${PAIRS_JSON}" <<'PY'
import json
import sys

items = json.loads(sys.argv[1])
if not isinstance(items, list):
    raise SystemExit("PAIRS_JSON must decode to a JSON list.")

for item in items:
    if isinstance(item, str):
        pair = item.strip()
    elif isinstance(item, list) and len(item) == 2:
        pair = f"{str(item[0]).strip()}|{str(item[1]).strip()}"
    else:
        raise SystemExit(
            "Each PAIRS_JSON entry must be either 'prompt_a|prompt_b' or [prompt_a, prompt_b]."
        )
    if "|" not in pair:
        raise SystemExit(f"Pair is missing '|': {pair!r}")
    print(pair)
PY
  )
elif [[ -n "${PAIRS_FILE:-}" ]]; then
  if [[ ! -f "${PAIRS_FILE}" ]]; then
    echo "PAIRS_FILE does not exist: ${PAIRS_FILE}" >&2
    exit 1
  fi
  mapfile -t PAIRS < <(sed '/^[[:space:]]*$/d' "${PAIRS_FILE}")
else
  PAIRS=("${DEFAULT_PAIRS[@]}")
fi

if [[ "${#PAIRS[@]}" -eq 0 ]]; then
  echo "No prompt pairs were provided." >&2
  exit 1
fi

total="${#PAIRS[@]}"
for i in "${!PAIRS[@]}"; do
  pair="${PAIRS[$i]}"
  if [[ "${pair}" != *"|"* ]]; then
    echo "Invalid pair (missing '|'): ${pair}" >&2
    exit 1
  fi
  obj="${pair%%|*}"
  bg="${pair#*|}"
  prompts="$obj | $bg"

  src_name="$(echo "$prompts" | awk -F'|' '{gsub(/^ +| +$/,"",$1); gsub(/^ +| +$/,"",$2); print $1"_AND_"$2}' | sed 's/ /_/g')_seed${SEED}.png"
  dst_name="$(printf '%02d' "$((i + 1))")_$(slugify "$obj")__$(slugify "$bg").png"
  dst_path="$OUT_DIR/$dst_name"

  echo "[$((i + 1))/$total] prompts='$prompts' -> $dst_path"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  DRY_RUN=1 -> skipping generation"
    continue
  fi
  (
    cd "$REPO_DIR"
    python "$SCRIPT_PATH" \
      --prompts "$prompts" \
      --weights "$PAIR_WEIGHTS" \
      --scale "$SCALE" \
      --steps "$STEPS" \
      --seed "$SEED" \
      --num_images "$NUM_IMAGES" \
      --scheduler "$SCHEDULER" \
      --model_path "$MODEL_PATH"
  )

  if [[ -f "$REPO_DIR/$src_name" ]]; then
    mv "$REPO_DIR/$src_name" "$dst_path"
  else
    echo "Expected output not found: $REPO_DIR/$src_name" >&2
    exit 1
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete. Planned outputs under: $OUT_DIR"
else
  echo "Done. Images saved to: $OUT_DIR"
fi
