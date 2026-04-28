#!/usr/bin/env bash
# run_cat_dog_ablation.sh
# ========================
# Test Methods 11, 12, 13, and 14 on cat+dog concept pairs.
#
# Usage:
#   bash scripts/run_cat_dog_ablation.sh [MODEL_ID]
#
# MODEL_ID defaults to SDXL.  For a quick smoke test on SD1.4:
#   bash scripts/run_cat_dog_ablation.sh CompVis/stable-diffusion-v1-4
#
# Outputs are written to:
#   results/cat_dog_ablation/a_cat_a_dog/seed_{N}/
#     11_tweedie_poe_corrector.png
#     12_overlap_penalty_corrector.png
#     13_spatial_routing.png
#     14_poe_anchored_contrastive.png
#     15_corrected_spatial_masking.png
#     *_infos.json   (per-step diagnostics)
#     summary.json

set -e

MODEL_ID="${1:-stabilityai/stable-diffusion-xl-base-1.0}"
SEEDS="42 123 456"
STEPS=50
GUIDANCE=7.5
OUT_DIR="results/cat_dog_ablation"

echo "======================================================"
echo "  Cat+Dog Ablation: Methods 11, 12, 13, 14, 15, 16"
echo "  Model   : ${MODEL_ID}"
echo "  Seeds   : ${SEEDS}"
echo "  Steps   : ${STEPS}"
echo "  Guidance: ${GUIDANCE}"
echo "  Out dir : ${OUT_DIR}"
echo "======================================================"

for SEED in $SEEDS; do
    echo ""
    echo "--- Seed ${SEED} ---"
    python scripts/run_repair_methods.py \
        --c1 "a cat" \
        --c2 "a dog" \
        --model_id "${MODEL_ID}" \
        --method "11,12,13,14,15,16" \
        --seed "${SEED}" \
        --steps "${STEPS}" \
        --guidance "${GUIDANCE}" \
        --out_dir "${OUT_DIR}"
done

echo ""
echo "======================================================"
echo "  Done. Results: ${OUT_DIR}/a_cat_a_dog/"
echo "======================================================"

# ---- Diagnostic summary ----
# After running, verify the following in the *_infos.json files:
#
# Method 11 (11_tweedie_poe_corrector_infos.json):
#   - cos_d1_d2 key present (diagnostic)
#   - delta_norm > 0 at t_frac in [0.2, 0.5] (divergence correction active)
#   - apply_outer_cfg=True now (fixed from previous broken default)
#
# Method 12 (12_overlap_penalty_corrector_infos.json):
#   - correction_grad_norm_mean should be O(0.1) not O(0.001)
#   - lambda_overlap_t > 0 for t_frac in [0.05, 0.60]
#   - cos_delta1_delta2 and dominance_ratio keys present
#
# Method 13 (13_spatial_routing_infos.json):
#   - phase transitions: 1 (steps 0-12), 2 (steps 12-35), 3 (steps 35-50)
#   - spatial_iou decreasing from ~0.5 toward <0.15 by mid-trajectory
#   - x0_disagree growing in early steps
#
# Method 14 (14_poe_anchored_contrastive_infos.json):
#   - anchor_w should ramp from 1.0 → ~1.8 → 1.2 across trajectory
#   - contrastive_delta > 0 during [0.15, 0.75] t_frac
#   - x0_disagree growing (concepts separating) vs decreasing (collapsing)
#   - cos_d1_d2 > 0 means constructive interference (chimera risk)
#
# Method 15 (15_corrected_spatial_masking_infos.json):
#   - blend_weight ramps 0→1 over t_frac [0.20, 0.50], then holds at 1.0
#   - mask_sharpness should increase from ~0.5 (soft) toward ~0.9 (binary)
#   - spatial_iou should decrease (concepts claiming different territories)
#   - leakage_removed > 0 confirms leakage filter is active at boundaries
#
# Method 16 (16_ir_poe_infos.json):
#   - phase=1.0 for steps 0..19 (t_frac<0.40), phase=2.0 for steps 20..49
#   - mask_sharpness: starts ~0.5 (pure noise), should grow toward ~0.9 by step 20
#   - spatial_iou: decreasing across Phase 1 (concepts claiming territory)
#   - conflict: cos(S1,S2); if >0 → constructive chimera risk; if <conflict_threshold → soft mask engaged
#   - suppression_delta > 0 confirms cross-negative passes fired (if use_cross_suppression=True)
