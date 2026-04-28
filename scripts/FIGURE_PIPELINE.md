# Figure Pipeline — Six-Group SDXL Screening, Freeze, Final Run

This document is the canonical paper-facing SDXL workflow.

## Errata / Bug Fixes (2026-04-18)

Three bugs were identified and fixed. Any SDIPC dirs produced before this date must be
repaired before rendering figure2 or figure_seed_sheet with `--seed-condition pstar_sdipc`.

### Bug 1 — `merge_grid_assets` wrote mixed-projection `grid_assets.json` *(fixed)*

`scripts/sdxl_sdipc_utils.py`: `merge_grid_assets()` previously overwrote
`trajectory_projection` with a mix of the old 4-condition MDS and the new 5-condition MDS
coordinates, producing incompatible coordinate systems. The `trajectory_projection` write
has been removed from `merge_grid_assets`. It is now written **only** by
`write_canonical_grid_assets`, which always performs a fully joint MDS over all conditions
present. **Rule:** always call `write_canonical_grid_assets` after `merge_grid_assets`
(both `enrich_sdxl_final_with_sdipc.py` and `repair_sdxl_sdipc_root.py` already do this).

### Bug 2 — Misaligned `x_T` for pstar_sdipc in figure2 *(repair required for existing data)*

If SDIPC dirs were produced before the Bug 1 fix and `write_canonical_grid_assets` exited
early or was interrupted for any pair, the cached `trajectory_projection` in that
`grid_assets.json` may mix projections from two MDS runs. The `pstar_sdipc` trajectory will
appear to start at a different `x_T` than the other conditions.

**Repair existing SDIPC dirs before rendering any figure2:**

```bash
conda activate jaxstack
export USE_TF=NO

python scripts/_repair_sdipc_projections.py
```

The script rewrites every real (non-symlinked) `grid_assets.json` in both SDIPC roots with
a correct joint MDS projection and prints `x_T OK` / `x_T MISMATCH` for the six
representative pairs.

### Bug 3 — No step-count validation in `measure_composability_gap.py` *(fixed)*

Added an assertion that all condition trackers have the same number of denoising steps
before per-step distance computation. A mismatch now raises immediately instead of silently
misaligning trajectory distances.

The paper narrative now uses the frozen six-group taxonomy throughout:

1. `G1 Co-occurrence`: manifold-supported, low conflict
2. `G2 Factorization`: attribute/style transfer with preserved factor roles
3. `G3 Object-Scene`: role-separable but structurally strained
4. `G4 Dual-Object`: composition without natural support
5. `G5 Prior Entanglement`: concept priors dominate or hijack
6. `G6 Coherent Collision`: semantically adjacent concepts collapse or fuse

Interpretation layer:

- `easy`: G1, G2
- `intermediate`: G3
- `hard`: G4, G5, G6

The paper-facing rule is strict: after freeze, all qualitative SDXL figures must read from the frozen multi-seed final root only.

## Environment

```bash
conda activate jaxstack

export USE_TF=NO

export SDXL_BASE_RUN="experiments/eccv2026/sdxl_screening/sdxl_taxonomy_seed42_steps50_cfg7p5_20260413"
export SDXL_G34_EXPANSION_RUN="experiments/eccv2026/sdxl_screening/sdxl_g34_simple_expansion_seed42_steps50_cfg7p5_20260413"
export SDXL_G34_SCENE_RUN="experiments/eccv2026/sdxl_screening/sdxl_g34_scene_subgroup_seed42_steps50_cfg7p5_20260413"
export SDXL_G34_REPLACEMENTS_RUN="experiments/eccv2026/sdxl_screening/sdxl_g34_replacements_seed42_steps50_cfg7p5_20260413"
export SDXL_SUPPLEMENTAL_RUN="experiments/eccv2026/sdxl_screening/sdxl_six_group_supplemental_seed42_steps50_cfg7p5_20260413"
export SDXL_AUDIT_MANIFEST="experiments/eccv2026/sdxl_screening/sdxl_paper_audit_manifest.json"
export SDXL_FINAL_DIR="experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_frozen"
export SDXL_SDIPC_DIR="experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_frozen_sdipc"
export SDXL_GROUP6_CANDIDATE_MANIFEST="scripts/group6_collision_candidates.json"
export SDXL_GROUP6_EDIT_AUDIT_MANIFEST="experiments/eccv2026/sdxl_final/group6_replacement_audit_manifest.json"
export SDXL_GROUP6_EDIT_ROSTER="experiments/eccv2026/sdxl_final/group6_replacement_roster.json"
export SDXL_EDIT_FINAL_DIR="experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_group6_edit"
export SDXL_EDIT_SDIPC_DIR="experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_group6_edit_sdipc"
export FIGURES_DIR="paper/neurips/Comparing Semantic and Logical Composition Using Latent Diffusion Models/figures"
export WAVE1_INCLUDE_SDIPC=0

# Quantitative gap-analysis root. This is separate from SDXL_FINAL_DIR and must
# be the direct output root produced by the six-group measure_composability_gap.py
# stage documented below.
export SDXL_GAP_RUN="results/sdxl_six_group_gap_sd35_sdipc_seed42_1to23"
export SDXL_REPORT_DIR="$SDXL_GAP_RUN/report_bundle"
```

Canonical final seeds:

```bash
export SDXL_FINAL_SEEDS="42 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"
```

## Stage 0 — Refresh The Screening Pool

The canonical taxonomy lives in `scripts/taxonomy_manifest.py`:

- 6 groups
- 10 pairs per group
- 60 total screening pairs

Reuse existing screening roots whenever possible. Only render missing canonical pairs into the supplemental root.

Check missing canonical assets before any supplemental render:

```bash
python scripts/audit_and_clean_sdxl_pairs.py \
  --screening-roots \
    "$SDXL_BASE_RUN" \
    "$SDXL_G34_EXPANSION_RUN" \
    "$SDXL_G34_SCENE_RUN" \
    "$SDXL_G34_REPLACEMENTS_RUN" \
    "$SDXL_SUPPLEMENTAL_RUN" \
  --audit-manifest "$SDXL_AUDIT_MANIFEST" \
  --action report
```

Expected per-pair outputs:

- `summary.json`
- `grid_assets.json`
- `trajectory_manifold.png`
- `decoded_images.png`
- `solo_a.png`
- `solo_b.png`
- `monolithic.png`
- `poe.png`

## Stage 1 — Manual Audit Manifest

Initialize the canonical 60-pair review manifest:

```bash
python scripts/audit_and_clean_sdxl_pairs.py \
  --screening-roots \
    "$SDXL_BASE_RUN" \
    "$SDXL_G34_EXPANSION_RUN" \
    "$SDXL_G34_SCENE_RUN" \
    "$SDXL_G34_REPLACEMENTS_RUN" \
    "$SDXL_SUPPLEMENTAL_RUN" \
  --audit-manifest "$SDXL_AUDIT_MANIFEST" \
  --action init-manifest
```

Export the manual review pack:

```bash
python scripts/audit_and_clean_sdxl_pairs.py \
  --screening-roots \
    "$SDXL_BASE_RUN" \
    "$SDXL_G34_EXPANSION_RUN" \
    "$SDXL_G34_SCENE_RUN" \
    "$SDXL_G34_REPLACEMENTS_RUN" \
    "$SDXL_SUPPLEMENTAL_RUN" \
  --audit-manifest "$SDXL_AUDIT_MANIFEST" \
  --action export-review
```

Manual review policy:

1. Inspect `trajectory_manifold.png`, `decoded_images.png`, `monolithic.png`, and `poe.png`.
2. Use `solo_a.png` and `solo_b.png` whenever the composition is ambiguous.
3. Set each canonical pair to `keep`, `drop`, or `unreviewed`.
4. Fill `rationale_tags`, `notes`, `reviewer`, and `reviewed_at`.
5. Do not freeze while any entry remains `unreviewed`.
6. If a group drops below 10 keeps, rerender replacements before freeze.

## Stage 2 — Freeze The Canonical Paper Roster

```bash
python scripts/freeze_sdxl_taxonomy_roster.py \
  --audit-manifest "$SDXL_AUDIT_MANIFEST"
```

Outputs:

- `experiments/eccv2026/sdxl_screening/sdxl_paper_roster.json`
- `experiments/eccv2026/sdxl_screening/sdxl_paper_roster_pairs.csv`
- `experiments/eccv2026/sdxl_screening/sdxl_paper_roster_groups.csv`

Freeze rules:

- exactly 10 `keep` decisions per group
- zero `unreviewed` entries
- no metric ranking or probe thresholds
- freeze fails loudly if any group is short

## Stage 3 — Fresh Final SDXL Run From The Frozen Roster

```bash
python scripts/run_taxonomy_qualitative_sdxl.py \
  --output-dir "$SDXL_FINAL_DIR" \
  --roster-manifest "experiments/eccv2026/sdxl_screening/sdxl_paper_roster.json" \
  --seeds $SDXL_FINAL_SEEDS \
  --grid-seed 42 \
  --num-inference-steps 50 \
  --guidance-scale 7.5
```

Expected final-root layout:

```text
$SDXL_FINAL_DIR/
  sdxl_qualitative_run_manifest.json
  seed_42/group1_cooccurrence/a_butterfly__x__a_flower_meadow/
  seed_1/group1_cooccurrence/a_butterfly__x__a_flower_meadow/
  ...
  seed_23/group6_coherent_collision/a_fox__x__a_wolf/
```

Rule:

- all paper-facing qualitative SDXL figures read from `SDXL_FINAL_DIR`
- screening roots remain audit-only inputs after freeze

## Stage 3B — Taxonomy Edit Branch: Replace Group 6 After Initial Freeze

Use this branch only when the frozen final roots already exist and you need to repair Group 6 without rerunning Groups 1–5.

The edited workflow preserves the original near-neighbor Group 6 under a non-paper archive bucket:

- archived key: `group7_archived_near_neighbor_collision`
- replacement key: `group6_coherent_collision`

### Step 3B.1 — Prepare Edited Roots And Archive Old Group 6

This step derives new edited roots from the existing frozen roots, archives the old Group 6 as Group 7, keeps Groups 1–5 untouched, and optionally stages seed-42 screening renders for the replacement Group 6 candidate pool.

```bash
python scripts/prepare_group6_replacement_sdxl.py \
  --source-final-dir "$SDXL_FINAL_DIR" \
  --source-sdipc-dir "$SDXL_SDIPC_DIR" \
  --output-final-dir "$SDXL_EDIT_FINAL_DIR" \
  --output-sdipc-dir "$SDXL_EDIT_SDIPC_DIR" \
  --candidate-manifest "$SDXL_GROUP6_CANDIDATE_MANIFEST" \
  --copy-mode symlink \
  --stage-screening \
  --gpu-id 0
```

Candidate source:

- `scripts/group6_collision_candidates.json`
- manifest-driven replacement pool, larger than 10
- intended for seed-42 screening only before freeze

### Step 3B.2 — Audit Replacement Group 6 Candidates At Seed 42

Initialize the replacement-group audit manifest:

```bash
python scripts/audit_group6_replacement_sdxl.py \
  --screening-root "$SDXL_EDIT_FINAL_DIR" \
  --candidate-manifest "$SDXL_GROUP6_CANDIDATE_MANIFEST" \
  --audit-manifest "$SDXL_GROUP6_EDIT_AUDIT_MANIFEST" \
  --action init-manifest
```

Export the review pack:

```bash
python scripts/audit_group6_replacement_sdxl.py \
  --screening-root "$SDXL_EDIT_FINAL_DIR" \
  --candidate-manifest "$SDXL_GROUP6_CANDIDATE_MANIFEST" \
  --audit-manifest "$SDXL_GROUP6_EDIT_AUDIT_MANIFEST" \
  --action export-review
```

Import your completed review CSV or JSON back into the audit manifest:

```bash
python scripts/audit_group6_replacement_sdxl.py \
  --screening-root "$SDXL_EDIT_FINAL_DIR" \
  --candidate-manifest "$SDXL_GROUP6_CANDIDATE_MANIFEST" \
  --audit-manifest "$SDXL_GROUP6_EDIT_AUDIT_MANIFEST" \
  --review-csv "${SDXL_GROUP6_EDIT_AUDIT_MANIFEST%.json}.review.csv" \
  --action import-review
```

Check the current keep/drop/unreviewed counts:

```bash
python scripts/audit_group6_replacement_sdxl.py \
  --screening-root "$SDXL_EDIT_FINAL_DIR" \
  --candidate-manifest "$SDXL_GROUP6_CANDIDATE_MANIFEST" \
  --audit-manifest "$SDXL_GROUP6_EDIT_AUDIT_MANIFEST" \
  --action report
```

Candidate review policy:

1. inspect `trajectory_manifold.png`, `decoded_images.png`, `monolithic.png`, and `poe.png`
2. use `solo_a.png` and `solo_b.png` to check that both concepts remain visually grounded
3. mark exactly 10 candidate pairs as `keep`
4. do not freeze while any candidate remains `unreviewed`

### Step 3B.3 — Freeze The Replacement Group 6 Final 10

```bash
python scripts/freeze_group6_replacement_sdxl.py \
  --audit-manifest "$SDXL_GROUP6_EDIT_AUDIT_MANIFEST" \
  --candidate-manifest "$SDXL_GROUP6_CANDIDATE_MANIFEST" \
  --output "$SDXL_GROUP6_EDIT_ROSTER"
```

Freeze rules:

- exactly 10 `keep` decisions
- zero `unreviewed` entries
- scoped only to replacement Group 6

### Step 3B.4 — Run The Final 24 Seeds For Replacement Group 6

This reruns only the new Group 6 pairs into the edited final root and then enriches only the replacement Group 6 with SD-IPC.

```bash
python scripts/run_group6_replacement_final_sdxl.py \
  --edited-final-dir "$SDXL_EDIT_FINAL_DIR" \
  --edited-sdipc-dir "$SDXL_EDIT_SDIPC_DIR" \
  --replacement-roster "$SDXL_GROUP6_EDIT_ROSTER" \
  --seeds $SDXL_FINAL_SEEDS \
  --grid-seed 42 \
  --num-inference-steps 50 \
  --guidance-scale 7.5 \
  --gpu-id 0 \
  --enrich-sdipc
```

Outputs:

- edited final root with Groups 1–5 unchanged
- archived old Group 6 preserved under `group7_archived_near_neighbor_collision`
- replacement Group 6 fully rendered at 24 seeds
- replacement Group 6 SD-IPC assets appended into `SDXL_EDIT_SDIPC_DIR`

Rule:

- once the replacement Group 6 is accepted, all paper-facing qualitative SDXL figures should read from `SDXL_EDIT_FINAL_DIR` or `SDXL_EDIT_SDIPC_DIR`
- Group 7 is reference-only and must not be used by default six-panel paper renderers

### Step 3B.5 — Distributed Two-Node Execution For The Final 24 Seeds

When both `mscluster106` and `mscluster107` are available, use seed sharding across the 4 visible GPUs. This is race-free because each worker owns a disjoint seed subset for the same frozen 10-pair Group 6 roster.

Shard rule:

- `mscluster106`, GPU 0 → seeds at indices `0,4,8,...`
- `mscluster106`, GPU 1 → seeds at indices `1,5,9,...`
- `mscluster107`, GPU 0 → seeds at indices `2,6,10,...`
- `mscluster107`, GPU 1 → seeds at indices `3,7,11,...`

Run on `mscluster106`:

```bash
NODE_ROLE=node106 bash scripts/run_group6_replacement_final_2node_2gpu.sh
```

Run on `mscluster107`:

```bash
NODE_ROLE=node107 bash scripts/run_group6_replacement_final_2node_2gpu.sh
```

Each local worker does:

1. SDXL final render for its assigned seeds only
2. SD-IPC enrichment for the same seeds only
3. no manifest rewrite during shard execution

After both nodes finish, run exactly once on either node to synchronize the edited manifests:

```bash
MODE=sync bash scripts/run_group6_replacement_final_2node_2gpu.sh
```

This updates:

- `SDXL_EDIT_FINAL_DIR/sdxl_qualitative_run_manifest.json`
- `SDXL_EDIT_SDIPC_DIR/sdxl_qualitative_run_manifest.json`

from the frozen replacement roster, without rerendering anything.

## Stage 4 — Paper Figure Rendering From The Frozen Final Root

The paper figure plan is intentionally split into two waves.

### Wave 1 — Available Now

These figures establish the scientific narrative in this order: taxonomy first, representative evidence second, seed robustness third.

**Source root after Stage 3B (Group 6 edit):** use `SDXL_EDIT_FINAL_DIR` / `SDXL_EDIT_SDIPC_DIR`
instead of `SDXL_FINAL_DIR` / `SDXL_SDIPC_DIR` for all paper-facing renders. The edited dirs
preserve Groups 1–5 unchanged and replace Group 6 with the corrected Coherent Collision pairs.

Wave 1 flag:

- `WAVE1_INCLUDE_SDIPC=0`: baseline Wave 1
- `WAVE1_INCLUDE_SDIPC=1`: SD-IPC-inclusive Wave 1

The flag is a paper-workflow switch. It does not change renderer semantics; it selects between already-supported modes and explicit condition lists.

When `WAVE1_INCLUDE_SDIPC=1`, first enrich the frozen root into `SDXL_SDIPC_DIR`:

```bash
python scripts/enrich_sdxl_final_with_sdipc.py \
  --source-dir "$SDXL_FINAL_DIR" \
  --output-dir "$SDXL_SDIPC_DIR" \
  --seeds $SDXL_FINAL_SEEDS \
  --grid-seed 42 \
  --copy-mode symlink
```

This enrichment pass is narrower than rerunning `run_taxonomy_qualitative_sdxl.py`:

- it reads the existing frozen multi-seed final root
- it computes only `pstar_sdipc` from the existing PoE outputs
- it writes the new decoded image and trajectory/grid metadata into the derived root
- all SD-IPC-inclusive Wave 1 commands below must read from `SDXL_SDIPC_DIR`, not `SDXL_FINAL_DIR`

After enrichment, repair and validate the derived root before rendering any SD-IPC figure:

```bash
python scripts/repair_sdxl_sdipc_root.py \
  --source-dir "$SDXL_FINAL_DIR" \
  --output-dir "$SDXL_SDIPC_DIR" \
  --seeds 42 1 7 13 \
  --pairs \
    a_butterfly__x__a_flower_meadow \
    a_dog__x__oil_painting_style \
    a_picnic_table__x__a_snowstorm \
    a_typewriter__x__a_cactus \
    fluffy__x__a_stone \
    a_fox__x__a_wolf \
  --rerun-missing-base \
  --copy-mode symlink

python scripts/validate_sdxl_sdipc_enrichment.py \
  --source-dir "$SDXL_FINAL_DIR" \
  --output-dir "$SDXL_SDIPC_DIR" \
  --seeds 42 1 7 13 \
  --groups \
    group1_cooccurrence \
    group2_factorization \
    group3_role_separable_object_scene \
    group4_dual_object_composition \
    group5_concept_prior_entanglement \
    group6_coherent_collision
```

Paper-ready SD-IPC contract:

- each representative pair-seed directory must retain the base decoded assets from `SDXL_FINAL_DIR`
- each representative pair-seed directory must contain `trajectory_flat_prompt_a.npy`, `trajectory_flat_prompt_b.npy`, `trajectory_flat_monolithic.npy`, `trajectory_flat_poe.npy`, and `trajectory_flat_pstar_sdipc.npy`
- `grid_assets.json` must be rebuilt from the actual flat trajectories so `figure2` and other shared-noise SD-IPC manifold plots use a true joint MDS, never a mixture of cached 2D projections and one fresh trajectory
- **after any enrichment or repair, always run `python scripts/_repair_sdipc_projections.py` before rendering figure2** to guarantee all `grid_assets.json` hold a fully joint MDS projection; verify `x_T OK` is printed for all six representative pairs

#### Figure 1 — Six-Group Taxonomy Overview

- `scientific_claim`: the six regimes are visually distinct under a shared-noise rendering protocol
- `source_root`: baseline `SDXL_FINAL_DIR`; SD-IPC-on `SDXL_SDIPC_DIR`
- `dependency_status`: `available_now`
- `placement`: `main_body`
- `renderer`: baseline `scripts/render_taxonomy_paper_figure.py --mode figure1 --seed 42`; SD-IPC-on `--mode figure2 --seed 42`
- `output`: baseline `figures/trajectory_3x2_sdxl.png`; SD-IPC-on `figures/trajectory_3x2_sdxl_sdipc.png`

Baseline command (post-Stage-3B):

```bash
python scripts/render_taxonomy_paper_figure.py \
  --data-dir "$SDXL_EDIT_FINAL_DIR" \
  --seed 42 \
  --mode figure1 \
  --out "$FIGURES_DIR/trajectory_3x2_sdxl.png"
```

SD-IPC-on command (post-Stage-3B — use edited roots):

```bash
# Run _repair_sdipc_projections.py first if SDXL_EDIT_SDIPC_DIR was produced before 2026-04-18
python scripts/_repair_sdipc_projections.py

python scripts/render_taxonomy_paper_figure.py \
  --data-dir "$SDXL_EDIT_SDIPC_DIR" \
  --seed 42 \
  --mode figure2 \
  --out "$FIGURES_DIR/trajectory_3x2_sdxl_sdipc.png"
```

`figure2` requires full flat-trajectory coverage for `prompt_a`, `prompt_b`, `monolithic`, `poe`, and `pstar_sdipc`. The renderer fails loudly if the SD-IPC root contains only cached 2D projections. All five trajectories must share the same `x_T` origin — verify this by checking that `_repair_sdipc_projections.py` prints `x_T OK` for all six pairs.

#### Figure 2 — Representative Decoded Endpoints

- `scientific_claim`: the trajectory differences correspond to concrete decoded outputs, not only latent geometry
- `source_root`: baseline `SDXL_FINAL_DIR`; SD-IPC-on `SDXL_SDIPC_DIR`
- `dependency_status`: `available_now`
- `placement`: `main_body`
- `renderer`: baseline `scripts/render_taxonomy_paper_figure.py --mode figure1_endpoints --seed 42`; SD-IPC-on `--mode custom --conditions prompt_a prompt_b monolithic poe pstar_sdipc --seed 42`
- `output`: baseline `figures/representative_endpoints_3x2.png`; SD-IPC-on `figures/representative_endpoints_3x2_sdipc.png`

Baseline command (post-Stage-3B):

```bash
python scripts/render_taxonomy_paper_figure.py \
  --data-dir "$SDXL_EDIT_FINAL_DIR" \
  --seed 42 \
  --mode figure1_endpoints \
  --out "$FIGURES_DIR/representative_endpoints_3x2.png"
```

SD-IPC-on command (post-Stage-3B):

```bash
python scripts/render_taxonomy_paper_figure.py \
  --data-dir "$SDXL_EDIT_SDIPC_DIR" \
  --seed 42 \
  --mode custom \
  --conditions prompt_a prompt_b monolithic poe pstar_sdipc \
  --out "$FIGURES_DIR/representative_endpoints_3x2_sdipc.png"
```

#### Figure 3 — Representative Seed Sheet

- `scientific_claim`: some regimes are visually stable across seeds while others show structured variability
- `source_root`: baseline `SDXL_FINAL_DIR`; SD-IPC-on `SDXL_SDIPC_DIR`
- `dependency_status`: `available_now`
- `placement`: `main_body` or `appendix` depending on page budget
- `renderer`: baseline `scripts/render_taxonomy_paper_figure.py --mode figure_seed_sheet --seeds 42 1 7 13 --seed-condition poe`; SD-IPC-on `--seed-condition pstar_sdipc`
- `output`: baseline `figures/representative_seed_sheet_poe.png`; SD-IPC-on `figures/representative_seed_sheet_pstar_sdipc.png`

Baseline command (post-Stage-3B):

```bash
python scripts/render_taxonomy_paper_figure.py \
  --data-dir "$SDXL_EDIT_FINAL_DIR" \
  --mode figure_seed_sheet \
  --seeds 42 1 7 13 \
  --seed-condition poe \
  --out "$FIGURES_DIR/representative_seed_sheet_poe.png"
```

SD-IPC-on command (post-Stage-3B):

```bash
python scripts/render_taxonomy_paper_figure.py \
  --data-dir "$SDXL_EDIT_SDIPC_DIR" \
  --mode figure_seed_sheet \
  --seeds 42 1 7 13 \
  --seed-condition pstar_sdipc \
  --out "$FIGURES_DIR/representative_seed_sheet_pstar_sdipc.png"
```

#### Figure 4a — BLIP-VQA Hybrid Summary

- `scientific_claim`: simple cue-presence bars can look superficially reasonable while the decoded outputs reveal why marginal prompts are insufficient
- `source_root`: baseline `SDXL_FINAL_DIR` for decoded images; SD-IPC-on `SDXL_SDIPC_DIR` for decoded images
- `dependency_status`: `requires_additional_eval`
- `placement`: `main_body` or `appendix`, depending on pedagogy and page budget
- `renderer`: baseline `scripts/render_group_probe_endpoint_hybrid.py --metric blip_vqa`; SD-IPC-on adds `--endpoint-conditions prompt_a prompt_b monolithic poe pstar_sdipc`
- `output`: baseline `figures/blip_vqa_endpoint_hybrid_3x2.png`; SD-IPC-on `figures/blip_vqa_endpoint_hybrid_3x2_sdipc.png`

Baseline command pattern:

```bash
python scripts/eval_blip_vqa.py \
  --data-dir "$SDXL_FINAL_DIR/seed_42" \
  --conditions c1 c2 mono poe

python scripts/render_group_probe_endpoint_hybrid.py \
  --data-dir "$SDXL_FINAL_DIR" \
  --metrics-dir "$SDXL_FINAL_DIR/seed_42" \
  --metric blip_vqa \
  --seed 42 \
  --pairs \
    a_butterfly__x__a_flower_meadow \
    a_dog__x__oil_painting_style \
    a_picnic_table__x__a_snowstorm \
    a_typewriter__x__a_cactus \
    fluffy__x__a_stone \
    a_fox__x__a_wolf \
  --out "$FIGURES_DIR/blip_vqa_endpoint_hybrid_3x2.png"
```

SD-IPC-on command pattern:

```bash
python scripts/eval_blip_vqa.py \
  --data-dir "$SDXL_SDIPC_DIR/seed_42" \
  --conditions c1 c2 mono poe pstar_sdipc

python scripts/render_group_probe_endpoint_hybrid.py \
  --data-dir "$SDXL_SDIPC_DIR" \
  --metrics-dir "$SDXL_SDIPC_DIR/seed_42" \
  --metric blip_vqa \
  --seed 42 \
  --endpoint-conditions prompt_a prompt_b monolithic poe pstar_sdipc \
  --pairs \
    a_butterfly__x__a_flower_meadow \
    a_dog__x__oil_painting_style \
    a_picnic_table__x__a_snowstorm \
    a_typewriter__x__a_cactus \
    fluffy__x__a_stone \
    a_fox__x__a_wolf \
  --out "$FIGURES_DIR/blip_vqa_endpoint_hybrid_3x2_sdipc.png"
```

#### Figure 4b — Joint-Probe Hybrid Summary

- `scientific_claim`: pair-type-aware joint probes align better with the actual failure patterns visible in the decoded outputs
- `source_root`: baseline `SDXL_FINAL_DIR` for decoded images; SD-IPC-on `SDXL_SDIPC_DIR` for decoded images
- `dependency_status`: `requires_additional_eval`
- `placement`: `main_body`
- `renderer`: baseline `scripts/render_group_probe_endpoint_hybrid.py --metric joint_probe`; SD-IPC-on adds `--endpoint-conditions prompt_a prompt_b monolithic poe pstar_sdipc`
- `output`: baseline `figures/joint_probe_endpoint_hybrid_3x2.png`; SD-IPC-on `figures/joint_probe_endpoint_hybrid_3x2_sdipc.png`

Baseline command pattern:

```bash
python scripts/eval_joint_probes.py \
  --data-dir "$SDXL_FINAL_DIR/seed_42"

python scripts/render_group_probe_endpoint_hybrid.py \
  --data-dir "$SDXL_FINAL_DIR" \
  --metrics-dir "$SDXL_FINAL_DIR/seed_42" \
  --metric joint_probe \
  --seed 42 \
  --pairs \
    a_butterfly__x__a_flower_meadow \
    a_dog__x__oil_painting_style \
    a_picnic_table__x__a_snowstorm \
    a_typewriter__x__a_cactus \
    fluffy__x__a_stone \
    a_fox__x__a_wolf \
  --out "$FIGURES_DIR/joint_probe_endpoint_hybrid_3x2.png"
```

SD-IPC-on command pattern:

```bash
python scripts/eval_joint_probes.py \
  --data-dir "$SDXL_SDIPC_DIR/seed_42"

python scripts/render_group_probe_endpoint_hybrid.py \
  --data-dir "$SDXL_SDIPC_DIR" \
  --metrics-dir "$SDXL_SDIPC_DIR/seed_42" \
  --metric joint_probe \
  --seed 42 \
  --endpoint-conditions prompt_a prompt_b monolithic poe pstar_sdipc \
  --pairs \
    a_butterfly__x__a_flower_meadow \
    a_dog__x__oil_painting_style \
    a_picnic_table__x__a_snowstorm \
    a_typewriter__x__a_cactus \
    fluffy__x__a_stone \
    a_fox__x__a_wolf \
  --out "$FIGURES_DIR/joint_probe_endpoint_hybrid_3x2_sdipc.png"
```

Both hybrid variants must use the same explicit six-pair list in canonical `G1..G6` order.

Hybrid renderer note:

- `render_group_probe_endpoint_hybrid.py` now keeps the current four-endpoint strip by default
- pass `--endpoint-conditions ... pstar_sdipc` only when the corresponding decoded SD-IPC asset exists
- the renderer fails clearly if a requested endpoint condition is missing from the per-pair assets

Current limitation:

- `eval_blip_vqa.py` and `eval_joint_probes.py` do not yet aggregate directly over the frozen multi-seed `seed_*` layout
- the commands above therefore produce seed-42 diagnostics rather than full 24-seed final-root aggregates
- once the evaluators are upgraded for the multi-seed layout, switch `--data-dir` and `--metrics-dir` back to `"$SDXL_FINAL_DIR"`

Common failure:

- `No pair directories found under experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_frozen`
  means the evaluator was pointed at the frozen multi-seed root instead of a concrete seed directory; for the current BLIP-VQA and joint-probe workflow use `"$SDXL_FINAL_DIR/seed_42"` as the evaluator `--data-dir`

#### Figure 4c — Single-Endpoint Probe Explainer

- `scientific_claim`: the joint-correctness score is built from structured sub-questions rather than a single naive prompt
- `source_root`: `SDXL_FINAL_DIR`
- `dependency_status`: `available_now`
- `placement`: `main_body` or `appendix`
- `renderer`: `scripts/render_probe_explainer_panel.py`
- `output`: baseline `figures/monolithic_probe_explainer.png`; SD-IPC-on `figures/pstar_sdipc_probe_explainer.png`

Baseline command pattern:

```bash
python scripts/render_probe_explainer_panel.py \
  --data-dir "$SDXL_FINAL_DIR" \
  --pair a_typewriter__x__a_cactus \
  --group group4_dual_object_composition \
  --seed 42 \
  --condition monolithic \
  --out "$FIGURES_DIR/monolithic_probe_explainer.png"
```

SD-IPC-on command pattern:

```bash
python scripts/render_probe_explainer_panel.py \
  --data-dir "$SDXL_FINAL_DIR" \
  --pair a_typewriter__x__a_cactus \
  --group group4_dual_object_composition \
  --seed 42 \
  --condition pstar_sdipc \
  --out "$FIGURES_DIR/pstar_sdipc_probe_explainer.png"
```

For the Group 6 edited branch, run the same repair + validation sequence against `SDXL_EDIT_FINAL_DIR` and `SDXL_EDIT_SDIPC_DIR` before any SD-IPC-inclusive render:

```bash
python scripts/repair_sdxl_sdipc_root.py \
  --source-dir "$SDXL_EDIT_FINAL_DIR" \
  --output-dir "$SDXL_EDIT_SDIPC_DIR" \
  --seeds 42 1 7 13 \
  --pairs \
    a_butterfly__x__a_flower_meadow \
    a_dog__x__oil_painting_style \
    a_picnic_table__x__a_snowstorm \
    a_typewriter__x__a_cactus \
    fluffy__x__a_stone \
    a_fox__x__a_wolf \
  --rerun-missing-base \
  --copy-mode symlink

python scripts/validate_sdxl_sdipc_enrichment.py \
  --source-dir "$SDXL_EDIT_FINAL_DIR" \
  --output-dir "$SDXL_EDIT_SDIPC_DIR" \
  --seeds 42 1 7 13 \
  --groups \
    group1_cooccurrence \
    group2_factorization \
    group3_role_separable_object_scene \
    group4_dual_object_composition \
    group5_concept_prior_entanglement \
    group6_coherent_collision
```

## Stage 5 — Produce The Six-Group Quantitative Gap Run

This stage creates the quantitative root consumed later by `build_gap_report_bundle.py`.

Required outputs under `"$SDXL_GAP_RUN"`:

- `metrics/per_seed_distances.json`
- `metrics/trajectory_distances.json`
- `metrics/within_and_distances.json`
- `metrics/all_pairs_gap.json`
- `pairs/*/grid_assets.json`

Important separation:

- `SDXL_FINAL_DIR` is the frozen qualitative multi-seed root used for paper-facing qualitative figures
- `SDXL_GAP_RUN` is a separate six-group quantitative export root produced by `scripts/measure_composability_gap.py`
- `build_gap_report_bundle.py` consumes `SDXL_GAP_RUN`; it does not read directly from `SDXL_FINAL_DIR`

Canonical paper-scale command:

```bash
python scripts/measure_composability_gap.py \
  --output-dir "$SDXL_GAP_RUN" \
  --model-family sd35 \
  --pstar-source sdipc \
  --monolithic-baseline naive \
  --seeds $SDXL_FINAL_SEEDS \
  --grid-seed 42 \
  --pairs \
    "a butterfly+a flower meadow" \
    "a camel+a desert landscape" \
    "a deer+a forest clearing" \
    "a dolphin+an ocean wave" \
    "a duck+a pond" \
    "a flamingo+a lagoon" \
    "a horse+a grassy field" \
    "a lighthouse+an ocean with stormy waves" \
    "a polar bear+an iceberg" \
    "a sailboat+a harbor" \
    "a dog+oil painting style" \
    "a lighthouse+watercolour style" \
    "a bicycle+sketch style" \
    "a teapot+claymation style" \
    "a barn+pencil drawing style" \
    "a cactus+mosaic style" \
    "a camera+watercolor style" \
    "a castle+stained glass style" \
    "a cat+charcoal drawing style" \
    "a train+pixel art style" \
    "a bookcase+a glacier" \
    "a candle+a waterfall" \
    "a fire hydrant+a snowfield" \
    "a lamppost+a desert dune" \
    "a lighthouse+a desert dune" \
    "a mailbox+a snowfield" \
    "a park bench+a sand dune" \
    "a phone booth+a tropical beach" \
    "a picnic table+a snowstorm" \
    "a rowboat+a cactus garden" \
    "a bathtub+a streetlamp" \
    "a birdcage+a watering can" \
    "a briefcase+a ceramic bowl" \
    "a chessboard+a lantern" \
    "a drum set+a snowman" \
    "a feather pillow+a cast iron pan" \
    "a lab microscope+a hay bale" \
    "a microwave+a potted plant" \
    "a suitcase+a desk fan" \
    "a typewriter+a cactus" \
    "fluffy+a stone" \
    "striped+a sphere" \
    "small+an elephant" \
    "a transparent glass+a dog" \
    "a fur coat+a goldfish" \
    "a winter coat+a tropical parrot" \
    "a wool scarf+a jellyfish" \
    "a ballerina+a spacesuit" \
    "a wedding dress+a lobster" \
    "a tuxedo+a flamingo" \
    "a convertible+a roadster" \
    "a coupe+a sedan" \
    "a fox+a wolf" \
    "a goose+a swan" \
    "a leopard+a cheetah" \
    "a lion+a leopard" \
    "a pickup truck+an SUV" \
    "a raven+a crow" \
    "a sedan+an SUV" \
    "a wolf+a husky"
```

Paper-path rules:

- do not use `--regime large` for this canonical run; that preset is the legacy 24-pair ECCV regime, not the six-group 60-pair paper taxonomy
- do not use `--paper-only`; in the current script it switches to `sd14`, which is not the intended `sd35` paper path here

Common failure:

- `Not found: per_seed_distances.json`
  means `build_gap_report_bundle.py` was run before this stage; first create `"$SDXL_GAP_RUN/metrics/per_seed_distances.json"` with `measure_composability_gap.py`, then rerun the report-bundle command

Ordered execution from the completed `"$SDXL_EDIT_SDIPC_DIR"` state:

1. Run the six-group quantitative export once to populate `"$SDXL_GAP_RUN"`:

```bash
python scripts/measure_composability_gap.py \
  --output-dir "$SDXL_GAP_RUN" \
  --model-family sd35 \
  --pstar-source sdipc \
  --monolithic-baseline naive \
  --seeds $SDXL_FINAL_SEEDS \
  --grid-seed 42 \
  --pairs \
    "a butterfly+a flower meadow" \
    "a camel+a desert landscape" \
    "a deer+a forest clearing" \
    "a dolphin+an ocean wave" \
    "a duck+a pond" \
    "a flamingo+a lagoon" \
    "a horse+a grassy field" \
    "a lighthouse+an ocean with stormy waves" \
    "a polar bear+an iceberg" \
    "a sailboat+a harbor" \
    "a dog+oil painting style" \
    "a lighthouse+watercolour style" \
    "a bicycle+sketch style" \
    "a teapot+claymation style" \
    "a barn+pencil drawing style" \
    "a cactus+mosaic style" \
    "a camera+watercolor style" \
    "a castle+stained glass style" \
    "a cat+charcoal drawing style" \
    "a train+pixel art style" \
    "a bookcase+a glacier" \
    "a candle+a waterfall" \
    "a fire hydrant+a snowfield" \
    "a lamppost+a desert dune" \
    "a lighthouse+a desert dune" \
    "a mailbox+a snowfield" \
    "a park bench+a sand dune" \
    "a phone booth+a tropical beach" \
    "a picnic table+a snowstorm" \
    "a rowboat+a cactus garden" \
    "a bathtub+a streetlamp" \
    "a birdcage+a watering can" \
    "a briefcase+a ceramic bowl" \
    "a chessboard+a lantern" \
    "a drum set+a snowman" \
    "a feather pillow+a cast iron pan" \
    "a lab microscope+a hay bale" \
    "a microwave+a potted plant" \
    "a suitcase+a desk fan" \
    "a typewriter+a cactus" \
    "fluffy+a stone" \
    "striped+a sphere" \
    "small+an elephant" \
    "a transparent glass+a dog" \
    "a fur coat+a goldfish" \
    "a winter coat+a tropical parrot" \
    "a wool scarf+a jellyfish" \
    "a ballerina+a spacesuit" \
    "a wedding dress+a lobster" \
    "a tuxedo+a flamingo" \
    "a convertible+a roadster" \
    "a coupe+a sedan" \
    "a fox+a wolf" \
    "a goose+a swan" \
    "a leopard+a cheetah" \
    "a lion+a leopard" \
    "a pickup truck+an SUV" \
    "a raven+a crow" \
    "a sedan+an SUV" \
    "a wolf+a husky"
```

2. Confirm that Stage 5 actually wrote the quantitative artifacts before plotting:

```bash
ls "$SDXL_GAP_RUN/metrics"
find "$SDXL_GAP_RUN/pairs" -maxdepth 2 -name 'grid_assets.json' | wc -l
```

Expected after this check:

- `"$SDXL_GAP_RUN/metrics/per_seed_distances.json"`
- `"$SDXL_GAP_RUN/metrics/trajectory_distances.json"`
- `"$SDXL_GAP_RUN/metrics/within_and_distances.json"`
- `"$SDXL_GAP_RUN/metrics/all_pairs_gap.json"`
- one `grid_assets.json` per pair under `"$SDXL_GAP_RUN/pairs/"`

3. Build the canonical paper-facing quantitative bundle:

```bash
python scripts/build_gap_report_bundle.py \
  --data-dir "$SDXL_GAP_RUN" \
  --output-dir "$SDXL_REPORT_DIR" \
  --monolithic-baseline naive \
  --taxonomy-view groupwise \
  --dpi 180
```

4. If you want the core paper figures as standalone exports in `"$FIGURES_DIR"`, render them in narrative order:

```bash
python scripts/plot_gap_analysis.py \
  --data-dir "$SDXL_GAP_RUN" \
  --output-dir "$FIGURES_DIR" \
  --plot 11 \
  --taxonomy-view groupwise

python scripts/plot_gap_analysis.py \
  --data-dir "$SDXL_GAP_RUN" \
  --output-dir "$FIGURES_DIR" \
  --plot 06 \
  --taxonomy-view groupwise \
  --pstar-sources none

python scripts/plot_gap_analysis.py \
  --data-dir "$SDXL_GAP_RUN" \
  --output-dir "$FIGURES_DIR" \
  --plot 17 \
  --taxonomy-view groupwise

python scripts/plot_gap_analysis.py \
  --data-dir "$SDXL_GAP_RUN" \
  --output-dir "$FIGURES_DIR" \
  --plot 15 \
  --taxonomy-view groupwise

python scripts/plot_gap_analysis.py \
  --data-dir "$SDXL_GAP_RUN" \
  --output-dir "$FIGURES_DIR" \
  --plot 20 \
  --taxonomy-view groupwise

python scripts/plot_gap_analysis.py \
  --data-dir "$SDXL_GAP_RUN" \
  --output-dir "$FIGURES_DIR" \
  --plot 21 \
  --taxonomy-view groupwise
```

5. If you also want the pooled / non-groupwise diagnostic plots from the same run, render them after the groupwise paper set:

```bash
python scripts/plot_gap_analysis.py \
  --data-dir "$SDXL_GAP_RUN" \
  --output-dir "$FIGURES_DIR"
```

Notes on scope:

- Step 3 is the canonical paper export path
- Step 4 is useful when you want direct copies of the core paper plots in `"$FIGURES_DIR"`
- Step 5 is optional and renders the broader pooled plot suite, including diagnostics not used in the main paper sequence
- for this SD3.5 paper path, do not add `--paper-only`
- for this SD-IPC-only paper run, do not add `--merge` unless you later decide to accumulate extra p* sources such as `inverter` or `pez`

## Stage 6 — Build The Groupwise Quantitative Report Bundle

### Wave 2 — Requires Additional Evaluation

These figures are part of the intended paper arc, but they are blocked until six-group evaluation artifacts are regenerated from the frozen final root.

#### Figure 5 — Groupwise Quantitative Summary

- `scientific_claim`: the six-group visual taxonomy aligns with measured groupwise behavior
- `source_root`: frozen final root plus six-group evaluation outputs
- `dependency_status`: `requires_additional_eval`
- `placement`: `main_body`
- `preferred_first_metric`: pair-type-aware joint correctness
- `blocked_on`:
  - `joint_probe_scores.json`
  - finalized six-group metrics regenerated from the frozen final root

What already exists:

- the canonical quantitative export path is `scripts/build_gap_report_bundle.py`
- the report bundle already exports the core six-group groupwise figures:
  - `plot_11_within_and_noise_floor_groupwise.png`
  - `plot_06_stacked_bar_groupwise.png`
  - `plot_17_pstar_strip_groupwise.png`
  - `plot_15_pstar_kde_groupwise.png`
  - `plot_20_ecdf_groupwise.png`
  - `plot_21_jeffreys_heatmap_groupwise.png`
- the current bundle order already preserves the intended reveal:
  - baseline evidence first: `plot_11`, then `plot_06`
  - p* reveal next: `plot_17`, then `plot_15`
  - distributional generalization next: `plot_20`, then `plot_21`

Paper-facing reading of that sequence:

- `plot_06`: first clear baseline taxonomy separation in temporal divergence
- `plot_11`: calibrates the gap against the within-anchor noise floor
- `plot_17` and `plot_15`: reveal p* later, rather than front-loading it
- `plot_20` and `plot_21`: support general claims about the distribution of distances across groups

Canonical one-shot command:

```bash
python scripts/build_gap_report_bundle.py \
  --data-dir "$SDXL_GAP_RUN" \
  --output-dir "$SDXL_REPORT_DIR" \
  --monolithic-baseline naive \
  --taxonomy-view groupwise \
  --dpi 180
```

This writes the paper-facing exports to:

- `"$SDXL_REPORT_DIR/paper_exports/"`
- `"$SDXL_REPORT_DIR/figures/01_taxonomy/"`
- `"$SDXL_REPORT_DIR/figures/02_validity/"`
- `"$SDXL_REPORT_DIR/figures/03_dynamics/"`
- `"$SDXL_REPORT_DIR/figures/04_reachability/"`
- `"$SDXL_REPORT_DIR/figures/05_distribution/"`

Individual commands for the core groupwise figures:

```bash
python scripts/plot_gap_analysis.py \
  --data-dir "$SDXL_GAP_RUN" \
  --output-dir "$FIGURES_DIR" \
  --plot 11 \
  --taxonomy-view groupwise

python scripts/plot_gap_analysis.py \
  --data-dir "$SDXL_GAP_RUN" \
  --output-dir "$FIGURES_DIR" \
  --plot 06 \
  --taxonomy-view groupwise \
  --pstar-sources none

python scripts/plot_gap_analysis.py \
  --data-dir "$SDXL_GAP_RUN" \
  --output-dir "$FIGURES_DIR" \
  --plot 17 \
  --taxonomy-view groupwise

python scripts/plot_gap_analysis.py \
  --data-dir "$SDXL_GAP_RUN" \
  --output-dir "$FIGURES_DIR" \
  --plot 15 \
  --taxonomy-view groupwise

python scripts/plot_gap_analysis.py \
  --data-dir "$SDXL_GAP_RUN" \
  --output-dir "$FIGURES_DIR" \
  --plot 20 \
  --taxonomy-view groupwise

python scripts/plot_gap_analysis.py \
  --data-dir "$SDXL_GAP_RUN" \
  --output-dir "$FIGURES_DIR" \
  --plot 21 \
  --taxonomy-view groupwise
```

Expected filenames from those commands:

- `plot_11_within_and_noise_floor_groupwise.png`
- `plot_06_stacked_bar_groupwise.png`
- `plot_17_pstar_strip_groupwise.png`
- `plot_15_pstar_kde_groupwise.png`
- `plot_20_ecdf_groupwise.png`
- `plot_21_jeffreys_heatmap_groupwise.png`

Individual commands for the supportive probe bars:

```bash
python scripts/plot_gap_analysis.py \
  --data-dir "$SDXL_GAP_RUN" \
  --output-dir "$FIGURES_DIR" \
  --plot joint_probes

python scripts/plot_gap_analysis.py \
  --data-dir "$SDXL_GAP_RUN" \
  --output-dir "$FIGURES_DIR" \
  --plot blip_vqa
```

Run these only if the corresponding score JSONs already exist under `"$SDXL_GAP_RUN"` or another evaluation root:

- `joint_probe_scores.json`
- `blip_vqa_scores.json`

Secondary quantitative follow-ups:

- terminal composability gap
- within-AND noise floor
- temporal dynamics

#### Figure 6 — Reachability / Recovery Analysis

- `scientific_claim`: some targets are language-recoverable while harder groups remain unreachable
- `source_root`: frozen final root plus p* reruns and summary metrics
- `dependency_status`: `requires_additional_eval`
- `placement`: `late_main_body`
- `blocked_on`:
  - `pstar_sdipc.png` or equivalent rerun assets
  - reachability metrics and groupwise ECDF exports

Supportive but still blocked:

- `blip_vqa_scores.json`
- `joint_probe_scores.json`
- `blip_vqa_endpoint_hybrid_3x2.png`
- `joint_probe_endpoint_hybrid_3x2.png`
- `blip_vqa_grouped_bar.png`
- `blip_vqa_grouped_bar_pstar.png`

Canonical note:

- Wave 2 is not blocked because the groupwise plotting path is missing
- it is blocked because the final six-group evaluation artifacts still need to be regenerated and finalized from the frozen root
- use the report bundle as the canonical paper-facing quantitative export path once those artifacts exist

Command pattern for a reachability-specific SDXL subset run:

```bash
python scripts/run_reachability_validation_sdxl.py \
  --out experiments/eccv2026/sdxl_reachability_validation \
  --pairs \
    a_picnic_table__x__a_snowstorm \
    a_typewriter__x__a_cactus \
    fluffy__x__a_stone \
    a_fox__x__a_wolf \
  --seeds 42 1 7 13 \
  --steps 50 \
  --scale 7.5
```

After that subset run exists, render the semantics-qualified reachability figure with:

```bash
python scripts/plot_gap_analysis.py \
  --data-dir experiments/eccv2026/sdxl_reachability_validation \
  --output-dir "$FIGURES_DIR" \
  --plot reachability_semantic \
  --taxonomy-view groupwise
```

## Validation Checklist

Before treating the paper figure set as canonical, verify:

- the six representative pairs resolve from `scripts/taxonomy_manifest.py`
- all Wave 1 figures render exclusively from `SDXL_FINAL_DIR`
- overview and endpoint panels use the same group order and labels
- multi-seed panels use only seeds listed in `sdxl_qualitative_run_manifest.json`
- `FIGURE_PIPELINE.md`, `figures_plan.md`, and paper text all describe a six-group taxonomy
- Wave 2 text distinguishes between existing groupwise exports and still-missing evaluation artifacts

## Paper-Facing Rule

- `scripts/taxonomy_manifest.py` is the source of truth for the six-group taxonomy
- `scripts/freeze_sdxl_taxonomy_roster.py` is the only canonical freeze entrypoint
- `scripts/run_taxonomy_qualitative_sdxl.py` is the canonical SDXL runner
- `scripts/render_taxonomy_paper_figure.py` is the canonical qualitative paper figure renderer
- do not mix screening outputs into paper figures after freeze
