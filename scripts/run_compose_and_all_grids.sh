#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_DIR="${1:-$ROOT_DIR/experiments/eccv2026/grid_figure}"

mkdir -p "$BASE_DIR"

"$ROOT_DIR/scripts/run_compose_and_grid.sh" "$BASE_DIR/compose_and_sd14"
"$ROOT_DIR/scripts/run_compose_and_sd3_grid.sh" "$BASE_DIR/compose_and_sd3"

echo "Done. Outputs:"
echo "  - $BASE_DIR/compose_and_sd14"
echo "  - $BASE_DIR/compose_and_sd3"
