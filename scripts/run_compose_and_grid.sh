#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/compositions/super-diffusion/scripts/compose_and.py"

OUT_DIR="${1:-$ROOT_DIR/experiments/eccv2026/grid_figure/compose_and_sd14}"
MODE="${MODE:-deterministic}"
STEPS="${STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-21}"
SCALE="${SCALE:-7.5}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUT_DIR"

slugify() {
  echo "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

DEFAULT_PAIRS=(
  "a donut|rainbow sprinkles"
  "a red sports car|green tires"
  "a white mug|blue polka dots"
  "a black umbrella|neon pink trim"
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
  obj_slug="$(slugify "$obj")"
  bg_slug="$(slugify "$bg")"
  out_file="$OUT_DIR/$(printf '%02d' "$((i + 1))")_${obj_slug}__${bg_slug}.png"

  echo "[$((i + 1))/$total] obj='$obj' bg='$bg' -> $out_file"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  DRY_RUN=1 -> skipping generation"
    continue
  fi
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

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete. Planned outputs under: $OUT_DIR"
else
  echo "Done. Images saved to: $OUT_DIR"
fi
