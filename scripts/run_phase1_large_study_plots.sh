#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-compose_ebm}"
INPUT_DIR="${1:-experiments/eccv2026/taxonomy_qualitative}"
TABLES_DIR="${2:-${INPUT_DIR}/phase1_tables}"
FIGURES_DIR="${3:-${INPUT_DIR}/phase1_figures}"

CYCLE_JSON_DEFAULT="experiments/eccv2026/cycle_consistency/cycle_consistency_taxonomy.json"
REACHABILITY_JSON="${REACHABILITY_JSON:-}"

AGGREGATE_CMD=(
  conda run -n "$CONDA_ENV_NAME" python scripts/aggregate_phase1_taxonomy_study.py
  --input-dir "$INPUT_DIR"
  --output-dir "$TABLES_DIR"
)

if [[ -f "$CYCLE_JSON_DEFAULT" ]]; then
  AGGREGATE_CMD+=(--cycle-json "$CYCLE_JSON_DEFAULT")
fi

if [[ -n "$REACHABILITY_JSON" ]]; then
  AGGREGATE_CMD+=(--reachability-json "$REACHABILITY_JSON")
fi

PLOT_CMD=(
  conda run -n "$CONDA_ENV_NAME" python scripts/plot_phase1_taxonomy_figures.py
  --tables-dir "$TABLES_DIR"
  --output-dir "$FIGURES_DIR"
)

printf '\n[Phase 1] Aggregating taxonomy study tables\n'
"${AGGREGATE_CMD[@]}"

printf '\n[Phase 1] Rendering group figures\n'
"${PLOT_CMD[@]}"

printf '\n[Phase 1] Done\n'
printf '  Tables : %s\n' "$TABLES_DIR"
printf '  Figures: %s\n' "$FIGURES_DIR"
