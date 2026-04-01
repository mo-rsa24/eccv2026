# Scripts Catalog

Quick reference for every script in this directory.
Run all commands from the `eccv2026/` root with the `eccv2026` conda environment active.

---

## Core Pipeline — run in order

### [P0] `trajectory_dynamics_experiment.py`  ✅ Active
**Phase 0 — Geometric evidence**
Runs four conditions (c₁ alone, c₂ alone, monolithic AND, SuperDiff AND) from the
same initial noise x_T, records full latent trajectories, projects to 2D (PCA/MDS),
and computes per-step divergence and kappa dynamics.
**Claims addressed:** C1, C2, C3, C7

```bash
python scripts/trajectory_dynamics_experiment.py \
    --prompt-a "a cat" --prompt-b "a dog" \
    --model-id stabilityai/stable-diffusion-3.5-medium \
    --steps 50 --guidance 4.5 --seed 42 \
    --output-dir experiments/trajectory_dynamics
# Guided hybrid sweep (Claim C7):
    --guidance-sweep 0.0 0.1 0.3 0.5 0.7 1.0
```

---

### [P1] `generate_inversion_training_data.py`  ✅ Active
**Phase 1 — Inverter training data**
Generates (image, SD3.5-conditioning) pairs for training f_θ.
Self-supervised closed loop — no external captions needed.
**Claims addressed:** C6 (prerequisite)

```bash
python scripts/generate_inversion_training_data.py \
    --model-id stabilityai/stable-diffusion-3.5-medium \
    --output-dir experiments/inversion/training_data \
    --images-per-prompt 8 --steps 50 --guidance 4.5
```

---

### [P2] `train_inverter.py`  ✅ Active
**Phase 2 — Train f_θ (CLIP → SD3.5 conditioning MLP)**
Architecture: frozen CLIP ViT-L/14 → pooled head (MLP 768→2048) +
sequence head (cross-attention decoder, 154 queries → 4096-dim).
**Claims addressed:** C6

```bash
python scripts/train_inverter.py \
    --data-dir experiments/inversion/training_data \
    --ckpt-dir ckpt/inverter \
    --epochs 50 --batch-size 16 --lr 1e-4
```

---

### [P3] `measure_composability_gap.py`  ✅ Active  (heavily extended)
**Phase 3 — Core gap measurement**
Runs AND, p* (multiple sources), mono, c₁, c₂ from the same x_T per seed.
Saves per_seed_distances.json, trajectory_distances.json, all_pairs_gap.json,
within_and_distances.json.
**Claims addressed:** C1, C4, C5, C6

```bash
# Base run (inverter only):
python scripts/measure_composability_gap.py \
    --ckpt ckpt/inverter/best.pt \
    --model-id stabilityai/stable-diffusion-3.5-medium \
    --output-dir experiments/inversion/gap_analysis \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

# Add p* sources incrementally with --merge:
python scripts/measure_composability_gap.py --pstar-source pez    --merge [flags]
python scripts/measure_composability_gap.py --pstar-source vlm    --merge [flags]
python scripts/measure_composability_gap.py --pstar-source z2t    --merge [flags]

# Mean-anchor sensitivity check (reviewer robustness):
python scripts/measure_composability_gap.py --anchor mean [flags]

# If you hit CUDA OOM on VLM runs, offload BLIP-2 captioning to CPU:
python scripts/measure_composability_gap.py --pstar-source vlm --vlm-device cpu [flags]
```

**Key flags:**

| Flag | Description |
|------|-------------|
| `--pstar-source {inverter,pez,vlm,z2t,all}` | Which p* inversion method to run |
| `--merge` | Accumulate new columns into existing JSONs (don't overwrite) |
| `--anchor {seed,mean}` | Per-seed paired (default) or average-AND anchor (sensitivity check) |
| `--pez-tokens N` | Number of PEZ soft token slots (default 16) |
| `--vlm-model-id` | BLIP-2 model (default `Salesforce/blip2-opt-2.7b`) |

---

### [P4] `characterize_gap.py`  ✅ Active
**Phase 4 — Gap taxonomy**
Correlates gap magnitude with CLIP cosine similarity between concept embeddings.
Produces scatter plots and regression summary.
**Claims addressed:** C5

```bash
python scripts/characterize_gap.py \
    --gap-dir experiments/inversion/gap_analysis \
    --output-dir experiments/inversion/gap_analysis
```

---

## Analysis

### [ABL] `ablation_semantic_vs_logical.py`  ✅ Active
Direct ablation: CLIP AND vs SuperDiff AND terminal cosine similarity.
Complements Phase 0 with a broader prompt set.
**Claims addressed:** C1, C3

### [ABL] `manifold_cooccurrence_vs_hybrid.py`  ✅ Active
CLIP hypersphere S^767 visualization: individual concepts, monolithic AND,
SuperDiff AND, real COCO co-occurrence images, geodesic midpoint.
**Claims addressed:** C3

---

## Visualisation

### [VIS-primary] `plot_gap_analysis.py`  ✅ Active  — 23 plots
Primary visualisation suite. See README.md §8 for the full plot table.

```bash
# All 23 plots:
python scripts/plot_gap_analysis.py \
    --data-dir experiments/inversion/gap_analysis \
    --output-dir experiments/inversion/gap_analysis/figures

# Key individual plots:
python scripts/plot_gap_analysis.py --plot 11   # noise floor validity
python scripts/plot_gap_analysis.py --plot 17   # primary p* result
python scripts/plot_gap_analysis.py --plot 22   # expressiveness ladder (THE paper figure)

# Plot 04 anchor sensitivity (same data dir; requires *_meananchor columns):
python scripts/plot_gap_analysis.py --plot 04 --and-anchor seed
python scripts/plot_gap_analysis.py --plot 04 --and-anchor mean

# Plot 04 clarity preset (zoom + sharper KDE + larger figure):
python scripts/plot_gap_analysis.py --plot 04 --and-anchor mean --plot04-separate
# Optional manual controls:
#   --plot04-xmax-quantile 0.97   --plot04-bw-scale 0.75   --plot04-scale 1.25
```

### [VIS] `plot_trajectory_analysis.py`  ✅ Active  — 14 plots
Trajectory-level evidence suite. Reads Phase 0 outputs.

```bash
python scripts/plot_trajectory_analysis.py \
    --data-dir experiments/trajectory_dynamics/<YYYYMMDD_HHMMSS>
```

### [VIS-phase1] `aggregate_phase1_taxonomy_study.py` + `plot_phase1_taxonomy_figures.py`  ✅ Active
Phase 1 large-study aggregation and plotting pipeline built around
`trajectory_dynamics_experiment.py` taxonomy outputs. Anchors every main metric to
PoE, uses pair means as the inferential unit, and emits:

- `phase1_group_terminal_gap.png`
- `phase1_group_trajectory_gap.png`
- `phase1_group_phase_contributions.png`
- `phase1_group_reachability.png`

```bash
python scripts/aggregate_phase1_taxonomy_study.py \
    --input-dir experiments/eccv2026/taxonomy_qualitative \
    --output-dir experiments/eccv2026/taxonomy_qualitative/phase1_tables

python scripts/plot_phase1_taxonomy_figures.py \
    --tables-dir experiments/eccv2026/taxonomy_qualitative/phase1_tables \
    --output-dir experiments/eccv2026/taxonomy_qualitative/phase1_figures
```

For the one-command wrapper:

```bash
CONDA_ENV_NAME=compose_ebm scripts/run_phase1_large_study_plots.sh
```

### [VIS-phase1] `export_phase1_sdipc_reachability.py`  ✅ Active
Exports SD-IPC reachability records for the Phase 1 taxonomy study. For each
pair-seed run already present under `taxonomy_qualitative/`, it:

- loads the saved PoE image,
- computes the SD-IPC conditioning from that image,
- reruns SD 1.4 with trajectory tracking from the same seed,
- reruns the PoE anchor from the same seed,
- writes `d_t_sdipc_poe`, `d_T_sdipc_poe`, CLIP similarity to the source PoE image,
  and cycle-close rate.

```bash
python scripts/export_phase1_sdipc_reachability.py \
    --input-dir experiments/eccv2026/taxonomy_qualitative \
    --output-json experiments/eccv2026/reachability/phase1_sdipc_reachability.json

REACHABILITY_JSON=experiments/eccv2026/reachability/phase1_sdipc_reachability.json \
CONDA_ENV_NAME=compose_ebm \
scripts/run_phase1_large_study_plots.sh
```

### [VIS-phase1] `plot_phase1_reachability_qualitative.py`  ✅ Active
Builds a side-by-side qualitative figure from the exported SD-IPC reachability JSON.
Columns are:

- semantic composition (`monolithic.png`)
- PoE (`poe.png`)
- SD-IPC regeneration (`sdipc.png`)

```bash
python scripts/plot_phase1_reachability_qualitative.py \
    --reachability-json experiments/eccv2026/reachability/phase1_sdipc_reachability.json \
    --output experiments/eccv2026/reachability/phase1_sdipc_qualitative_grid.png

# Or include every available row instead of one representative per group:
python scripts/plot_phase1_reachability_qualitative.py \
    --reachability-json experiments/eccv2026/reachability/phase1_sdipc_reachability.json \
    --selection all \
    --output experiments/eccv2026/reachability/phase1_sdipc_qualitative_grid_all.png
```

### [VIS] `make_comparison_figure.py`  ✅ Active
Publication comparison grids: real COCO images vs generated outputs.
Used for the paper's qualitative section.

---

## Utilities

### [UTIL] `generate_sd35.py`  ✅ Active
Simple standalone SD3.5 generation. Useful for quick image checks.

### [UTIL] `run_sd35_guided_count_excite.sh`  ✅ Active
Convenience launcher for SD3.5 guided hybrid:
`(1-α)·monolithic + α·SuperDiff-AND`, with count-focused sub-prompts
(defaults: `2/two cats`, `3/three dogs`) to encourage numerosity fidelity.

```bash
bash scripts/run_sd35_guided_count_excite.sh
```

### [UTIL] `run_sd14_compose_ae_count_excite.sh`  ✅ Active
Convenience launcher for SD1.4 Composable AND + Attend-and-Excite (A&E),
with count-token targeting (defaults for `2 cats and 3 dogs`).
Supports `SHOW_TOKEN_MAP=1` to inspect token indices before generation.

```bash
bash scripts/run_sd14_compose_ae_count_excite.sh
SHOW_TOKEN_MAP=1 bash scripts/run_sd14_compose_ae_count_excite.sh
```

### [UTIL] `generate_compositional_model_grid.py`  ✅ Active
Generates a rows×columns grid comparing multiple model IDs on one prompt.
Useful for supplementary material or model selection, not in the core pipeline.

---

## Superseded Scripts  ⚠️ Not part of active pipeline

These scripts pre-date `plot_gap_analysis.py` and have been fully superseded.
Moved to `scripts/superseded/` — do not include in paper pipeline runs.

| Script | Superseded by | What to use instead |
|--------|--------------|---------------------|
| `superseded/plot_composability_histogram.py` | `plot_gap_analysis.py` 00, 03 | `--plot 00` or `--plot 03` |
| `superseded/plot_gap_clean.py` | `plot_gap_analysis.py` 00, 17 | `--plot 17` |
| `superseded/plot_trajectory_histogram.py` | `plot_gap_analysis.py` 06–10 | `--plot 06` through `--plot 10` |

---

## Run Order Summary

```
P0: trajectory_dynamics_experiment.py       → experiments/trajectory_dynamics/
P1: generate_inversion_training_data.py     → experiments/inversion/training_data/
P2: train_inverter.py                       → ckpt/inverter/best.pt
P3: measure_composability_gap.py            → experiments/inversion/gap_analysis/
    (repeat with --pstar-source + --merge for each p* source)
P4: characterize_gap.py                     → experiments/inversion/gap_analysis/

VIS: plot_gap_analysis.py --plot all        → experiments/inversion/gap_analysis/figures/
VIS: plot_trajectory_analysis.py            → experiments/trajectory_dynamics/<run>/figures/
ABL: ablation_semantic_vs_logical.py        → experiments/eccv2026/ablations/
ABL: manifold_cooccurrence_vs_hybrid.py     → experiments/eccv2026/manifold/
```
