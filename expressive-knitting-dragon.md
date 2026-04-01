# H1 Test Matrix: What Phase 1 Requires

## H1 (The Composability Gap Hypothesis)

> Logical composition (score addition / PoE) diverges from semantic composition (joint training),
> and this divergence is **pair-type-predictable**: worst when neither the Bayes co-occurrence regime
> nor feature-space disentanglement holds. When both safety nets are absent, the two score fields
> conflict and produce outputs structurally unreachable by any natural-language prompt.
> The pair-type dependence is theoretically grounded in Bradley et al. (2025) and measurable as
> divergence in latent trajectories across the denoising process.

H1 splits into three falsifiable sub-claims:

| Sub-claim | What would confirm it | What would falsify it |
|-----------|----------------------|-----------------------|
| **C1** Gap exists and is observable | Decoded PoE outputs visibly differ from monolithic; groups 3–4 show hybridisation / dominance | PoE and monolithic indistinguishable across all groups |
| **C2** Gap is pair-type-predictable | Terminal distance $d_T$ increases monotonically G1→G4; KDE shifts right G1→G4 | All groups overlap; no group ordering |
| **C3** Gap is trajectory-level, not terminal coincidence | Trajectories occupy different manifold regions from early steps; late amplification >50% of total gap | All conditions co-terminate; gap only at final step |
| **C4** Gap is structural (off-manifold for SD) | Cycle consistency breaks for PoE G3–G4; SD-IPC maps to dominant concept | Cycle closes equally for PoE and monolithic |

---

## 1. Experiments to Conduct

### E1 — Decoded Images Grid (DONE — taxonomy_qualitative/)
**Script:** `trajectory_dynamics_experiment.py --taxonomy-grid`
**What:** 4 conditions × all 16 pairs from TAXONOMY_GROUPS, SD 1.4, shared noise x_T, seed 42
**Output:** `decoded_images_grid.png` → `media/decoded_images_grid.png`
**Status:** ✅ results exist in `experiments/eccv2026/taxonomy_qualitative/`

### E2 — Latent Trajectory MDS (DONE — per-pair)
**Script:** `trajectory_dynamics_experiment.py --taxonomy-grid`
**What:** Full latent trajectory per condition per pair, projected via MDS, fan-out plot
**Output:** `trajectory_manifold.png` per pair + `trajectory_manifold_grid.png` (all groups)
**Status:** ✅ per-pair manifolds exist; all-group grid needs to be assembled

### E3 — Terminal Distance Measurement (TODO)
**Script:** `measure_composability_gap.py`
**What:** For each of 16 pairs × 24 seeds:
  - Run PoE, monolithic, solo A, solo B from same x_T
  - Record terminal latent MSE $d_T$ (PoE vs monolithic)
  - Record per-step $d_t$ for temporal decomposition
**Output:** `per_seed_distances.json`, `trajectory_distances.json`, `within_and_distances.json`
**Scale:** 16 pairs × 24 seeds = 384 pair-seed records × 4 conditions = 1,536 runs
**Note:** Use SD 1.4 (legacy backend) to match E1/E2

### E4 — Temporal Decomposition (TODO — depends on E3)
**Script:** `plot_gap_analysis.py` plots 06–10
**What:** Bin 50-step denoising into 10-step intervals; show where gap accumulates
**Output:** `temporal_decomposition.png` → `media/temporal_decomposition.png`
**Confirms:** C3 — late amplification accounts for >50% of total gap

### E5 — Taxonomy Bar Chart / KDE (TODO — depends on E3)
**Script:** `plot_gap_analysis.py` plots 00–05, 17
**What:** KDE of $d_T$ stratified by group; bar chart of mean $d_T$ per group
**Output:** `fig:temporal_decomposition` left panel; `fig:taxonomy_bars`
**Confirms:** C2 — monotonic increase G1→G4

### E6 — Cycle Consistency / SD-IPC (TODO)
**Script:** `cycle_consistency_analysis.py` (or `cycle_consistency_analysis_sd3.py`)
**What:** For each PoE output (G1–G4 representative pairs):
  1. PoE image → SD-IPC projection → text embedding → regenerate from same x_T
  2. Monolithic image → same pipeline (control)
  3. Report: CLIP cosine similarity between original and regenerated
**Output:** `cycle_consistency_multiseed.png`
**Scale:** 4 representative pairs × 5 seeds = 20 cycle tests
**Confirms:** C4 — cycle-break rate significantly higher for PoE G3–G4 vs monolithic

### E7 — Orthogonality vs d_T Scatter (TODO — depends on E3)
**Script:** `composability_orthogonality_heuristic.py` + join with E3 results
**What:** Plot CLIP score-delta orthogonality ($\delta_1 \cdot \delta_2$) vs observed $d_T$ across pairs
**Output:** `fig:orth_scatter` (supplementary)
**Confirms:** Bradley et al. Lemma 8.1 predicts difficulty; positive correlation expected

---

## 2. Code to Write

### W1 — Taxonomy result aggregator (NEW — needed for E5 bar chart)
**File:** `scripts/aggregate_taxonomy_results.py`
**Purpose:** Scrape `summary.json` from all `taxonomy_qualitative/group*/*/` dirs; compute per-group mean $d_T$ (monolithic vs PoE endpoint distance); write `taxonomy_d_T_summary.csv`
**Reuse:** `trajectory_dynamics_experiment.py` summary.json schema → field `endpoint_distances_l2`
**Inputs:** `experiments/eccv2026/taxonomy_qualitative/**/summary.json`
**Outputs:** `experiments/eccv2026/taxonomy_qualitative/taxonomy_d_T_summary.csv`

### W2 — Taxonomy group comparison figure (NEW — proposal Fig. taxonomy_bars)
**File:** `scripts/plot_taxonomy_bars.py`
**Purpose:** Load `taxonomy_d_T_summary.csv`; produce 2-panel figure:
  - Left: bar chart of mean $d_T$ (PoE vs monolithic) ± std, 4 groups, monotonically increasing
  - Right: KDE of per-seed $d_T$ stratified by group
**Output:** `media/taxonomy_d_T_bars.png`

### W3 — Multi-group MDS grid figure (NEW — proposal Fig. trajectory_manifold_grid)
**File:** Could be added to `trajectory_dynamics_experiment.py` or standalone `scripts/plot_trajectory_grid.py`
**Purpose:** Load per-pair `trajectory_data.json` from all 16 pairs; run joint MDS per group; render 1×4 panel
**Output:** `media/trajectory_manifold_grid.png`
**Reuse:** `build_trajectory_figure()` already partially implemented in `run_taxonomy_qualitative.py`

### W4 — Cycle consistency runner for taxonomy pairs (EXTEND existing)
**File:** `scripts/cycle_consistency_analysis.py` — add `--taxonomy-pairs` batch mode
**Purpose:** Run SD-IPC cycle test for each taxonomy pair, report CLIP sim + cycle-break flag
**Output:** `media/cycle_consistency_multiseed.png`

---

## 3. Qualitative Evidence & Visual Illustrations Required

| Figure | File path (media/) | What it shows | Confirms |
|--------|--------------------|---------------|---------|
| **Fig composition_grid** | `decoded_images_grid.png` | 16 pairs × 4 conditions; rows by group | C1 |
| **Fig trajectory_manifold** | `trajectory_manifold_grid.png` | 4 MDS panels (one per group); fan-out; same x_T start | C3 |
| **Fig temporal_decomposition** | `temporal_decomposition.png` | Left: $d_T$ KDE by group; Right: 10-step bin stacked bar | C2 + C3 |
| **Fig cycle_consistency** | `cycle_consistency_multiseed.png` | PoE vs monolithic cycle test; green=closed, red=broken | C4 |
| **Fig taxonomy_bars** | `taxonomy_d_T_bars.png` | Mean $d_T$ monotonically increasing G1→G4 | C2 |
| **Fig orth_scatter** (supp) | `orth_scatter.png` | Orthogonality dot product vs $d_T$ per pair | Theory link |

**Critical visual prediction per group:**
- **G1 (co-occurrence):** PoE ≈ monolithic visually; both show both concepts present
- **G2 (disentangled):** PoE succeeds; output between two solo attractors (e.g., dog rendered as oil painting)
- **G3 (OOD):** Mixed — some concept dominance, some incoherence; MDS trajectories separate earlier
- **G4 (collision):** Hybridisation (cat-owl chimera) or concept dominance; trajectory most separated

---

## 4. Quantities and Evaluations Required

### Primary quantitative evidence for H1

| Metric | How computed | Expected value | Script |
|--------|-------------|----------------|--------|
| $d_T$ per group (mean ± std) | MSE(PoE terminal, monolithic terminal) in latent space | Monotonic: G1 < G2 < G3 < G4 | `measure_composability_gap.py` |
| $d_T$ KDE group separation | KDE overlap of G1/G2 vs G3/G4 | G3–G4 distributions right-shifted vs G1–G2 | `plot_gap_analysis.py` |
| Temporal decomposition | Fraction of total gap accumulated in each 10-step bin | Final 2–3 bins account for >50% | `plot_gap_analysis.py` |
| Divergence onset step | First step where $d_t >$ 1% max distance | All groups: first 10–15 steps | `trajectory_dynamics_experiment.py` |
| Cycle-break rate | CLIP sim(original PoE, regenerated) < 0.80 threshold | G3–G4 > G1–G2 break rate | `cycle_consistency_analysis.py` |
| Monolithic cycle-close rate | Same test for monolithic | >0.80 across all groups (on-manifold control) | `cycle_consistency_analysis.py` |

### Secondary / supporting quantities

| Metric | What it shows |
|--------|--------------|
| Orthogonality dot product $\delta_1 \cdot \delta_2$ per pair | Theoretical predictor of difficulty; correlates with observed $d_T$ |
| Within-AND noise floor (cross-seed $d_T$) | Validates gap is signal not stochasticity — gap >> noise floor |
| Total path length per condition | PoE traverses longer path (competing gradients); monolithic shorter |
| Final tangent cosine similarity | Direction of final velocity for PoE vs monolithic — divergent for G3–G4 |
| CLIP classifier accuracy at terminal step | PoE G4 maps to dominant concept, not both — quantifies concept dominance |

### Success criteria for H1 to hold

1. $d_T$ increases monotonically G1→G4 (Kruskal-Wallis + post-hoc p < 0.05)
2. KDE of G3–G4 is statistically right of G1–G2 (Wilcoxon rank-sum p < 0.05)
3. Late bins (steps 35–50) account for ≥50% of cumulative gap for all groups
4. Cycle-break rate: PoE G3–G4 > PoE G1–G2 > Monolithic (χ² test p < 0.05)
5. Orthogonality $\delta_1 \cdot \delta_2$ is Spearman-correlated with $d_T$ (r > 0.5)

---

## Execution Order

```
E1 ✅ Qualitative images (16 pairs, SD1.4, --taxonomy-grid)
E2 ✅ Per-pair trajectory manifolds
W3    Build trajectory_manifold_grid.png from E2 outputs
E3    Run measure_composability_gap.py × 16 pairs × 24 seeds
E4    Temporal decomposition (depends E3)
E5    Taxonomy bars + KDE (depends E3) — W1 + W2 first
E6    Cycle consistency runs (independent of E3)
E7    Orthogonality scatter (depends E3 + existing orth_dot values)
```

---

## Critical Files

| File | Role |
|------|------|
| `scripts/trajectory_dynamics_experiment.py` | E1, E2 (--taxonomy-grid mode) |
| `scripts/measure_composability_gap.py` | E3 (terminal + trajectory distances) |
| `scripts/plot_gap_analysis.py` | E4, E5 (temporal decomp, KDE, bars) |
| `scripts/cycle_consistency_analysis.py` | E6 (SD-IPC cycle test) |
| `scripts/composability_orthogonality_heuristic.py` | E7 (orth_dot per pair) |
| `scripts/aggregate_taxonomy_results.py` | W1 (NEW — scrape summary.json) |
| `scripts/plot_taxonomy_bars.py` | W2 (NEW — main quantitative figure) |
| `proposal/.../media/decoded_images_grid.png` | Fig composition_grid |
| `proposal/.../media/trajectory_manifold_grid.png` | Fig trajectory_manifold |
| `proposal/.../media/temporal_decomposition.png` | Fig temporal_decomposition |
| `proposal/.../media/cycle_consistency_multiseed.png` | Fig cycle_consistency |
| `proposal/.../media/taxonomy_d_T_bars.png` | Fig taxonomy_bars |

---

## Taxonomy Pairs (from binary-honking-lampson.md §1.1)

```python
GROUPS = {
    "group1_cooccurrence": [          # Setting C / Bayes regime
        ("a camel",           "a desert landscape"),
        ("a butterfly",       "a flower meadow"),
        ("a dolphin",         "an ocean wave"),
        ("a lion",            "a savanna at sunset"),
        ("a lighthouse",      "a stormy sea"),
        ("a sailboat",        "a lighthouse on shore"),
    ],
    "group2_disentangled": [          # Theorem 6.1 / style-content
        ("a dog",             "oil painting style"),
        ("a lighthouse",      "watercolor style"),
        ("a bicycle",         "sketch style"),
        ("a teapot",          "claymation style"),
        ("a barn",            "pencil drawing style"),
        ("a cactus",          "mosaic style"),
    ],
    "group3_ood": [                   # Lemma 8.1 high orth_dot, no joint attractor
        ("a desk lamp",       "a glacier"),            # orth_dot 0.251
        ("a bathtub",         "a streetlamp"),         # orth_dot 0.253
        ("a lab microscope",  "a hay bale"),           # orth_dot 0.267
        ("a bathtub",         "a hot air balloon"),    # orth_dot 0.293
        ("a black grand piano", "a white vase"),       # orth_dot 0.311
        ("a typewriter",      "a cactus"),             # orth_dot 0.315
    ],
    "group4_collision": [             # single semantic slot competition
        ("a cat",             "a dog"),
        ("a cat",             "an owl"),
        ("a cat",             "a bear"),
        ("a teddy bear",      "a panda"),
        ("an otter",          "a duck"),
        ("a tiger",           "a lion"),
    ],
}
```

---

## File 1: `scripts/run_control_landscape.py`

### Change 1A — Replace GROUPS dict (lines 59–124)
Replace the entire `GROUPS = { ... }` block with the 4-group taxonomy above.

### Change 1B — Remove SD1.4 model-id override (line 146)
Remove this line from the `cmd` list:
```python
        "--model-id", "CompVis/stable-diffusion-v1-4",
```
SD3.5 (`stabilityai/stable-diffusion-3.5-medium`) is the default in trajectory_dynamics_experiment.py;
no flag is needed.

### Change 1C — Pass `--taxonomy-group` in the cmd list
In `run_group()`, add to the `cmd` list (after the existing `--superdiff-variant` line):
```python
        "--taxonomy-group", group_name,
```

### Change 1D — Update default output directory (line 178)
Change:
```python
base_out = Path(args.out) if args.out else PROJECT_ROOT / "experiments" / "control_landscape"
```
To:
```python
base_out = Path(args.out) if args.out else PROJECT_ROOT / "experiments" / "eccv2026" / "taxonomy"
```

---

## File 2: `scripts/trajectory_dynamics_experiment.py`

Three surgical changes only. No logic changes.

### Change 2A — Add `taxonomy_group` field to `TrajectoryExperimentConfig` dataclass (around line 110)
After the `uniform_color` field (or wherever the block ends before the closing), add:
```python
    taxonomy_group: str = ""  # taxonomy group label for Phase 1 classification
```

### Change 2B — Add `--taxonomy-group` CLI argument (around line 3822, near `--seed`)
After:
```python
    parser.add_argument("--seed", type=int, default=42)
```
Add:
```python
    parser.add_argument("--taxonomy-group", type=str, default="",
                        help="Taxonomy group label stored in summary.json (e.g. group1_cooccurrence).")
```

### Change 2C — Add to `base_cfg_kwargs` dict (around lines 3917–3936)
In the `base_cfg_kwargs = dict(...)` block (grid mode path), add:
```python
            taxonomy_group=args.taxonomy_group,
```

### Change 2D — Add to summary.json config block (around line 3551)
In the `summary["config"] = { ... }` block, after `"no_poe": cfg.no_poe,`, add:
```python
        "taxonomy_group": cfg.taxonomy_group,
```

Note: `replace(base_cfg, prompt_a=..., prompt_b=...)` in grid mode preserves all other fields
including `taxonomy_group`, so all per-pair summaries will carry the group label automatically.

---

## Run Command (after changes)

```bash
conda activate attend_excite
python scripts/run_control_landscape.py --seed 42 --steps 50
```

Runs 4 groups × 6 pairs = 24 pairs. Per pair: 5 conditions (solo a, solo b, monolithic,
PoE primary, SuperDiff AND fm_ode secondary), SD3.5 Medium, seed 42.

Output:
```
experiments/eccv2026/taxonomy/
    group1_cooccurrence/
        camel__x__desert_landscape/
            decoded_images.png        ← qualitative Phase 1 result
            trajectory_manifold.png
            summary.json              ← includes "taxonomy_group": "group1_cooccurrence"
        ...
    group2_disentangled/
    group3_ood/
    group4_collision/
```

---

## Critical Files

| File | Lines affected |
|------|---------------|
| `scripts/run_control_landscape.py` | 59–124 (GROUPS), 146 (model-id removal), 147 (taxonomy-group add), 178 (output dir) |
| `scripts/trajectory_dynamics_experiment.py` | ~110 (dataclass field), ~3822 (argparse), ~3930 (base_cfg_kwargs), ~3551 (summary) |

## Verification Checklist

- [ ] `run_control_landscape.py` GROUPS has exactly 4 keys with 6 pairs each
- [ ] No `--model-id CompVis/...` in the cmd list
- [ ] `--taxonomy-group group_name` IS in the cmd list
- [ ] `--no-poe` is NOT in the cmd list (PoE runs)
- [ ] `--superdiff-variant fm_ode` remains (SD3.5 secondary condition)
- [ ] Default output dir is `experiments/eccv2026/taxonomy/`
- [ ] `TrajectoryExperimentConfig.taxonomy_group` field exists
- [ ] `--taxonomy-group` argparse argument exists
- [ ] `taxonomy_group` in `base_cfg_kwargs`
- [ ] `"taxonomy_group"` in summary.json config block
