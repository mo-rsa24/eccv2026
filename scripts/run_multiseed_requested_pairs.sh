#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEEDS=(1 2 3 4)

PAIRS=(
  "group1_cooccurrence a_dolphin__x__an_ocean_wave"
  "group1_cooccurrence a_sailboat__x__a_harbor"
  "group3_role_separable_object_scene a_picnic_table__x__a_snowstorm"
  "group5_concept_prior_entanglement fluffy__x__a_stone"
)

for entry in "${PAIRS[@]}"; do
  read -r group pair_slug <<<"$entry"
  pair_root="experiments/eccv2026/multiseed/${pair_slug}"

  for seed in "${SEEDS[@]}"; do
    python scripts/run_taxonomy_qualitative_sdxl.py \
      --pairs "${group}/${pair_slug}" \
      --seed "$seed" \
      --output-dir "${pair_root}/seed_${seed}"
  done

  python scripts/plot_multiseed_combined.py \
    --pair-dirs \
      "${pair_root}/seed_1/${group}/${pair_slug}" \
      "${pair_root}/seed_2/${group}/${pair_slug}" \
      "${pair_root}/seed_3/${group}/${pair_slug}" \
      "${pair_root}/seed_4/${group}/${pair_slug}" \
    --output "${pair_root}/multiseed_combined.png" \
    --overlay-output "${pair_root}/multiseed_combined_terminal_overlays.png"
done
