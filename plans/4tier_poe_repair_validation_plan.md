# Plan: 4-Tier PoE Repair Validation — Six-Group SDXL Pipeline

## Context

Vanilla PoE omits the interaction term rₜ(xₜ; c₁, c₂), causing characteristic failures in Group 3 (interference) and Group 6 (coherent collision) regimes. This plan validates a 4-tier theoretical framework using the canonical six-group, 10-pairs-per-group SDXL taxonomy.

**Current state (2026-04-14):**
- All 60 canonical pairs rendered in `sdxl_six_group_supplemental_seed42_steps50_cfg7p5_20260413/` (all 6 groups × 10 pairs, each with solo_a, solo_b, monolithic, poe, decoded_images, trajectory_manifold)
- Audit manifest complete: all 60 entries `decision=keep`, `rationale_tags=["paper_clean"]`, auto-reviewed
- Frozen roster exists: `experiments/eccv2026/sdxl_screening/sdxl_paper_roster.json` (manifest v3, all 6 groups selection_complete=True)
- **Stage 3 (fresh final run with 24 seeds) has NOT been run yet** — that is the next step per `FIGURE_PIPELINE.md`

The 10 repair methods are fully implemented in `compositions/repair_methods/`. The runner is `scripts/run_repair_methods.py`.

---

## Pipeline Position (per `scripts/FIGURE_PIPELINE.md`)

| Stage | Description | Status |
|---|---|---|
| Stage 0 | Refresh screening pool (60 pairs) | ✅ Complete |
| Stage 1 | Init & export audit manifest | ✅ Complete |
| Stage 2 | Freeze SDXL paper roster | ✅ Complete |
| **Stage 3** | **Fresh final SDXL run (24 seeds per pair)** | ❌ Not run |
| **Stage 4** | **Paper figure rendering** | ❌ Not run |
| **Repair** | **Run repair methods on G4/G6 pairs** | ❌ Not planned yet |

---

## Critical Code Issue to Fix Before Any GPU Work

**SDXL `cond_kwargs` is always `None`** in `run_repair_methods.py` lines 171–182 (the `is_sdxl` branch returns `emb, None`). The SDXL UNet requires `added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}` or the result is garbage/NaN.

Fix: read `notebooks/utils.py` to confirm the SDXL `get_text_embedding` signature — it already accepts `tokenizer_2` / `text_encoder_2` and returns a concatenated embedding, but the pooled output from `text_encoder_2` must be extracted separately and `time_ids = [[H, W, 0, 0, H, W]]` constructed before returning.

---

## Pair Selection

**Treatment — Group 6 (Coherent Collision), 5 pairs from frozen roster:**

All sourced from `experiments/eccv2026/sdxl_screening/sdxl_six_group_supplemental_seed42_steps50_cfg7p5_20260413/group6_coherent_collision/`

| Pair | Dir slug | Rationale |
|---|---|---|
| fox / wolf | `a_fox__x__a_wolf` | Canonical representative (taxonomy_manifest.py) |
| raven / crow | `a_raven__x__a_crow` | Near-identical concepts — strongest collision signal |
| leopard / cheetah | `a_leopard__x__a_cheetah` | Cross-feline, visually similar coloration |
| convertible / roadster | `a_convertible__x__a_roadster` | Cross-domain (vehicles) — generality outside biology |
| wolf / husky | `a_wolf__x__a_husky` | Wildtype vs. domestic — prior entanglement edge case |

**Treatment — Group 4 (Dual-Object Composition), 3 pairs:**

All sourced from `experiments/eccv2026/sdxl_screening/sdxl_six_group_supplemental_seed42_steps50_cfg7p5_20260413/group4_dual_object_composition/`

| Pair | Dir slug | Rationale |
|---|---|---|
| typewriter / cactus | `a_typewriter__x__a_cactus` | Legacy anchor pair, highest historical gap score |
| bathtub / streetlamp | `a_bathtub__x__a_streetlamp` | Legacy anchor pair |
| drum set / snowman | `a_drum_set__x__a_snowman` | Thematically incongruent objects |

**Control — Groups 1 and 3 (should NOT improve with repair):**

| Group | Pair | Dir slug |
|---|---|---|
| G1 | butterfly / flower meadow | `a_butterfly__x__a_flower_meadow` |
| G1 | dolphin / ocean wave | `a_dolphin__x__an_ocean_wave` |
| G3 | bookcase / glacier | `a_bookcase__x__a_glacier` |
| G3 | picnic table / snowstorm | `a_picnic_table__x__a_snowstorm` |

---

## Methods to Run

**Active (8):** 01, 02, 03, 04, 06, 07, 09, 10
**Excluded:**
- 08 (`branch_and_select`) — known `linalg.matrix_norm` error
- 05 (`residual_adapter`) — requires offline-trained weights; note in paper as supervised upper bound

---

## Step-by-Step Implementation Plan

### Phase 0 — Code Fixes (no GPU, ~1h)

**0.1 — Fix SDXL `cond_kwargs` in `run_repair_methods.py`**

Location: lines 169–182 (`is_sdxl` branch of the `encode()` closure).

Steps:
1. Read `notebooks/utils.py` — find how `get_text_embedding` handles SDXL (tokenizer_2, text_encoder_2). Confirm whether pooled_output is returned or must be obtained via a separate forward pass.
2. Update the `is_sdxl` branch to build and return `{"text_embeds": pooled_embeds, "time_ids": add_time_ids}` where `add_time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], dtype=dtype, device=device)`.

**0.2 — Add `group6_coherent_collision` to `eval_joint_probes.py`**

Find `PAIR_TYPE_BY_GROUP` dict and add:
```python
"group6_coherent_collision": "collision",
```
Currently only `group4_coherent_collision` maps to "collision". The canonical rename silently falls back to the wrong probe battery.

**0.3 — Add repair method condition aliases to `eval_blip_vqa.py`**

In `CONDITION_ALIASES`, add:
```python
"01_adaptive_weighting": ["01_adaptive_weighting"],
"02_gradient_surgery":   ["02_gradient_surgery"],
# ... 03, 04, 06, 07, 09, 10
```

### Phase 1 — Stage 3: Fresh Final SDXL Run (GPU, ~4–8h)

Per `FIGURE_PIPELINE.md` Stage 3, run the frozen roster with 24 seeds:

```bash
conda activate jaxstack && export USE_TF=NO

python scripts/run_taxonomy_qualitative_sdxl.py \
  --output-dir "experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_frozen" \
  --roster-manifest "experiments/eccv2026/sdxl_screening/sdxl_paper_roster.json" \
  --seeds 42 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
  --grid-seed 42 \
  --num-inference-steps 50 \
  --guidance-scale 7.5
```

Output layout (seed-aware):
```
experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_frozen/
  seed_42/<group>/<pair>/{solo_a,solo_b,monolithic,poe,decoded_images,trajectory_manifold}.png
  seed_1/...
  ...
  seed_23/...
```

This is the source of truth for all paper figures (Stage 4 onwards).

### Phase 2 — Stage 4: Paper Figures from Final Root (no GPU, ~2h)

Per `FIGURE_PIPELINE.md` Stage 4:

```bash
FINAL_DIR="experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_frozen"
FIGURES="paper/neurips/Comparing Semantic and Logical Composition Using Latent Diffusion Models/figures"

python scripts/render_taxonomy_paper_figure.py \
  --data-dir "$FINAL_DIR" \
  --seed 42 \
  --mode figure1 \
  --out "$FIGURES/trajectory_3x2_sdxl.png"
```

### Phase 3 — Run Repair Methods on G4/G6 Pairs (GPU, ~8–20h)

Baseline images (solo_a, solo_b, monolithic, poe, grid_assets.json) are copied from the Stage 3 final root (seed 42) rather than from the legacy screening runs.

**3.1 — Treatment pairs: G6 (5 pairs) + G4 (3 pairs), seeds [42, 1, 2, 3, 4]**

```bash
MODEL=stabilityai/stable-diffusion-xl-base-1.0
FINAL_SEED42="experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_frozen/seed_42"
OUTDIR="experiments/eccv2026/repair_methods/sdxl_repair_seed42_steps50_cfg7p5_20260414"

# G6 pairs
for PAIR_ARGS in \
    '"a fox" "a wolf"' \
    '"a raven" "a crow"' \
    '"a leopard" "a cheetah"' \
    '"a convertible" "a roadster"' \
    '"a wolf" "a husky"'; do
  for SEED in 42 1 2 3 4; do
    python scripts/run_repair_methods.py \
      --c1 $(echo $PAIR_ARGS | cut -d' ' -f1-3) \
      --c2 $(echo $PAIR_ARGS | cut -d' ' -f4-6) \
      --model_id $MODEL --method 01,02,03,04,06,07,09,10 \
      --seed $SEED --steps 50 --guidance 7.5 --out_dir $OUTDIR
  done
done

# G4 pairs (same loop with typewriter/cactus, bathtub/streetlamp, drum set/snowman)
# G1/G3 controls (seed 42 only)
```

**3.2 — Copy baselines from Stage 3 final root into output tree**

For each pair, copy from `$FINAL_SEED42/group6_coherent_collision/<pair>/` into `$OUTDIR/group6_coherent_collision/<pair>/`. This places solo_a, solo_b, monolithic, poe, grid_assets.json alongside repair method images so the evaluator sees the full condition set.

**3.3 — Ablation runs on fox/wolf, seed 42**

Ablation A — Signal isolation (method_01, 4 configs):
- `m01_no_signal`: k_disagree=0, balance=False → degenerates to vanilla PoE
- `m01_Dt_only`: k_disagree=1.0, balance=False
- `m01_Rt_only`: k_disagree=0, balance=True
- `m01_full`: default config

Ablation B — Scheduling (method_01, 3 configs):
- `schedule="constant"`, `"linear_increase"`, `"cosine_increase"`

Run via a small Python driver script that constructs modified `AdaptiveWeightingConfig` objects and calls `run_adaptive_weighting` directly, saving to `sdxl_repair_ablations/a_fox__x__a_wolf/`.

### Phase 4 — Evaluation (GPU, ~2h)

**4.1 — BLIP-VQA**
```bash
python scripts/eval_blip_vqa.py \
    --data-dir $OUTDIR \
    --conditions c1 c2 mono poe \
                 01_adaptive_weighting 02_gradient_surgery 03_trust_region \
                 04_mask_gated_poe 06_curvature_preconditioning \
                 07_mcmc_corrector 09_probe_energy 10_adaptive_diagnostics
```

**4.2 — Joint probes**
```bash
python scripts/eval_joint_probes.py --data-dir $OUTDIR
```

### Phase 5 — Figures and Tables (no GPU, ~4h)

**New scripts to create:**

| Script | Purpose | Tier |
|---|---|---|
| `scripts/plot_repair_methods_qualitative_grid.py` | Grid: solo_a \| solo_b \| monolithic \| poe \| best_method (fox/wolf) | T4 |
| `scripts/plot_repair_ablation_bars.py` | Grouped bars: signal contribution + scheduling ablations | T2/T3 |
| `scripts/plot_repair_group_selectivity.py` | G6/G4 improves, G1/G3 flat | T4 |
| `scripts/build_repair_paper_tables.py` | LaTeX Table 1 (main) + Table 2 (ablation) | T4 |

**Reuse existing:**
- `scripts/plots/joint_probe_bar.py` — base for ablation bar chart
- `scripts/plots/groupwise.py` — base for group selectivity chart
- `scripts/render_taxonomy_paper_figure.py` — canonical overview figure (Stage 4)

---

## Output Structure

```
experiments/eccv2026/
  sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_frozen/   ← Stage 3 (paper source of truth)
    seed_42/group1_cooccurrence/...
    seed_42/group2_factorization/...
    seed_42/group3_role_separable_object_scene/...
    seed_42/group4_dual_object_composition/...
    seed_42/group5_concept_prior_entanglement/...
    seed_42/group6_coherent_collision/...
    seed_1/ ... seed_23/

  repair_methods/sdxl_repair_seed42_steps50_cfg7p5_20260414/   ← Phase 3
    group6_coherent_collision/
      a_fox__x__a_wolf/
        solo_a.png, solo_b.png, monolithic.png, poe.png  (from Stage 3)
        grid_assets.json
        01_adaptive_weighting.png ... 10_adaptive_diagnostics.png
        summary.json
      a_raven__x__a_crow/ ...
      a_leopard__x__a_cheetah/ ...
      a_convertible__x__a_roadster/ ...
      a_wolf__x__a_husky/ ...
    group4_dual_object_composition/
      a_typewriter__x__a_cactus/ ...
      a_bathtub__x__a_streetlamp/ ...
      a_drum_set__x__a_snowman/ ...
    group3_role_separable_object_scene/
      a_bookcase__x__a_glacier/ ...
    group1_cooccurrence/
      a_butterfly__x__a_flower_meadow/ ...
    blip_vqa_scores.json
    joint_probe_scores.json

  repair_methods/sdxl_repair_ablations/
    a_fox__x__a_wolf/
      m01_no_signal.png, m01_Dt_only.png, m01_Rt_only.png, m01_full.png
      m01_const.png, m01_linear.png, m01_cosine.png
```

---

## Tier-to-Experiment Mapping

| Tier | Claim | Validated By |
|---|---|---|
| **T1** | Vanilla PoE omits rₜ | Table 1: `poe` < `monolithic` on G6/G4, not on G1 |
| **T1** | Surrogates from observable signals work | Any method_0X ≥ poe on G6/G4 treatment pairs |
| **T2** | x₀-consistency is strongest signal | Ablation A: m01_Dt_only > m01_Rt_only (joint_correctness) |
| **T2** | PCGrad is least-invasive | Method_02 shows no G1/G3 regression while improving G6 |
| **T3** | Scheduling matters | Ablation B: m01_linear > m01_const on G6 |
| **T4** | Corrections improve G4/G6, not G1/G3 | Table 1 group columns + group selectivity figure |

---

## Critical Files

| File | Change Type | Purpose |
|---|---|---|
| `scripts/run_repair_methods.py` (lines 169–182) | **Modify** | Fix SDXL `cond_kwargs` — single most critical change |
| `notebooks/utils.py` | Read-only | Confirm SDXL `get_text_embedding` signature for the fix |
| `scripts/eval_joint_probes.py` | **Modify** | Add `"group6_coherent_collision": "collision"` to `PAIR_TYPE_BY_GROUP` |
| `scripts/eval_blip_vqa.py` | **Modify** | Add repair method names to `CONDITION_ALIASES` |
| `scripts/run_taxonomy_qualitative_sdxl.py` | Run-only | Stage 3 final SDXL run |
| `scripts/render_taxonomy_paper_figure.py` | Run-only | Stage 4 overview figure |
| `experiments/eccv2026/sdxl_screening/sdxl_paper_roster.json` | Read-only | Frozen roster for Stage 3 |
| `experiments/eccv2026/sdxl_screening/sdxl_six_group_supplemental_seed42_steps50_cfg7p5_20260413/` | Read-only | Screening baseline source |
| `compositions/repair_methods/method_01_adaptive_weighting.py` | Read-only | `AdaptiveWeightingConfig` fields for ablation |

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| SDXL `cond_kwargs` fix incorrect `time_ids` shape | Dry-run 5 steps on fox/wolf before full batch |
| Method_07 (MCMC) is 5–10× slower | Run seed 42 only first; extend if results are promising |
| Method_04 (mask-gated) attention hooks on SDXL | Test on fox/wolf alone first; delta-norm fallback is already implemented |
| BLIP-VQA can't distinguish raven vs. crow | Use `hybrid_score` not `cue_presence`; raven/crow primarily tests collapse axis |
| Stage 3 run takes >8h | Parallelise across 4 GPUs by group: G1/G2 on GPU0, G3/G5 on GPU1, G4 on GPU2, G6 on GPU3 |

---

## Verification

After Phase 0 fix, dry-run:
```bash
python scripts/run_repair_methods.py \
    --c1 "a fox" --c2 "a wolf" \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --method 01 --seed 42 --steps 5 \
    --out_dir /tmp/repair_test
```
Expect: `01_adaptive_weighting.png` saved, no NaN, image is not uniform gray.

After Stage 3, verify coverage:
```bash
python scripts/audit_and_clean_sdxl_pairs.py \
  --screening-roots \
    "experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_frozen/seed_42" \
  --audit-manifest "experiments/eccv2026/sdxl_screening/sdxl_paper_audit_manifest.json" \
  --action report
```
Expect: all 60 pairs present, 0 missing assets.
