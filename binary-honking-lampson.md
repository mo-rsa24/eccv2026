# Plan: Phase 1 & 2 Reconceptualisation After Taxonomy Failure

## Context

Experiments with `Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch` revealed that the original Phase 1 concept-pair taxonomy (Independent/Orthogonal → Relational → Adversarial) was based on human semantic intuition, not on the theory in Bradley et al. (2502.04549). As a result, many "control" pairs that should have worked under the old taxonomy failed, and some "test" pairs worked unexpectedly. The theory explains this cleanly: Bayes composition (p_b = p_u) only works when the training distribution contains co-occurring joint modes for the composed concepts, OR when the concept pair is implicitly feature-space disentangled (style/content). Without either condition, score fields combine conflicting gradients with no training attractor, and the trajectory wanders.

This plan covers what must be reconsidered, rewritten, coded, run, and visualised so that Phase 1 (H1) and Phase 2 (H2) are scientifically coherent and theoretically aligned. Phase 3 (medical imaging / H3) is deferred.

---

## Part 1 — Conceptual Reconsidering

### 1.1 The New Four-Group Taxonomy (replaces old three-group taxonomy)

Each group is defined by conditions from Bradley et al., not by semantic intuition.

---

**Group 1 — Manifold-Supported Co-occurrence (Setting C / Bayes regime)**

**Definition:** pairs where the training distribution contains abundant joint-mode examples, so the unconditional model p_u is approximately factorised for this pair. Score-delta orthogonality may not hold, but composition works because the joint mode exists in training data.

**Prediction:** PoE and monolithic converge to similar terminal latents. d_T is small.

**Examples:** camel+desert, butterfly+flower meadow, dolphin+ocean wave, lion+savanna (from `positive_object_scene` — these mostly passed).

**Why it matters for H1:** establishes the baseline where composition is expected to succeed. Confirms theory (Setting C). Small gap here validates that the d_T metric can detect variation across groups.

---

**Group 2 — Feature-Space Disentangled (Theorem 6.1 / Style-Content regime)**

**Definition:** pairs where concepts occupy approximately disjoint coordinate subspaces in the model's feature space by construction — not because of training co-occurrence but because the concepts belong to decoupled feature axes (style ⊥ content in CLIP/UNet space).

**Prediction:** AND composition works, moderate-to-small d_T. Trajectory panels show AND between the two solo attractors, not collapsing to either.

**Examples:** dog+oil painting style, lighthouse+watercolor style, bicycle+sketch style (from `orthogonal_object_style` — these mostly passed).

**Why it matters for H1:** demonstrates that composition can succeed for genuinely OOD combinations when implicit feature disentanglement holds. Acts as a controlled positive control showing Theorem 6.1 in practice.

---

**Group 3 — Low Co-occurrence, OOD for Bayes Composition**

**Definition:** pairs where (a) the training distribution lacks abundant joint modes (Bayes regime fails; no attractor), AND (b) the CLIP score-delta orthogonality is marginal (Lemma 8.1 does not clearly hold). Both conditions for composition success are absent simultaneously.

> **IMPORTANT CAVEAT:** the theory predicts these pairs will be HARDER, not that they will categorically fail. Some may succeed if implicit disentanglement exists in a feature space not captured by CLIP text embeddings. Empirical verification is required; do not overstate the theoretical prediction.

**Examples** (from `composability_orthogonality_heuristic.py`, `background=""`):
- a desk lamp + a glacier               (orth_dot = 0.251)
- a bathtub + a streetlamp              (orth_dot = 0.253)
- a lab microscope + a hay bale         (orth_dot = 0.267)
- a bathtub + a hot air balloon         (orth_dot = 0.293)
- a black grand piano + a white vase    (orth_dot = 0.311)
- a typewriter + a cactus               (orth_dot = 0.315)
- And 5 more (see results.json).

**Why it matters for H1:** this is where the structural gap lives. If d_T is systematically larger here than in Groups 1 and 2, the gap is pair-type-predictable in a theoretically grounded way, which is the core H1 claim.

---

**Group 4 — Adversarial Collision (Single Semantic Slot Competition)**

**Definition:** pairs where both concepts compete for the same spatial slot and semantic category. Factorised Conditionals fail because both score fields impose overlapping constraints. The PoE product-of-experts mode is a chimeric blend, not a scene containing both distinct objects.

**Prediction:** AND produces chimeras or dominant-concept collapse. d_T from monolithic is largest. Cycle consistency breaks (SD-IPC maps AND output to dominant concept, not the hybrid).

**Examples:** cat+dog, cat+owl, cat+bear, two animals of same scale.

**Why it matters for H1:** the starkest failure mode and strongest evidence. Both hybridisation and concept dominance are pair-type-predictable, exactly what H1 claims.

---

### 1.2 What the Old "Works/Fails" Pattern Actually Showed

- `positive_object_scene` passed → Group 1: manifold-supported co-occurrence, Bayes regime works
- `orthogonal_object_style` passed → Group 2: Theorem 6.1, feature-space disentangled
- `orthogonal_layered_layout` mostly failed, sailboat+lighthouse passed → Group 1: sailboat+lighthouse is a natural coastal co-occurrence; all others are OOD (airplane+tractor, etc.) → Group 3
- `color_distinctive` failed → Group 3/4: objects competing for main-subject slot, color doesn't help
- `spatially_disjoint` failed → Group 3: genuine OOD pairs, no joint training mode, no disentanglement

### 1.3 SuperDiff AND in the Primary Narrative

> **DECISION:** Demote SuperDiff AND from primary comparator to secondary illustrator.
>
> **Reason:** Bradley et al. only theorises about linear score addition (PoE). SuperDiff's κ solver is a different operation with no closed-form theoretical guarantee from Bradley et al. SuperDiff also produces hybrids that are hard to classify (success or failure?), making the primary H1 narrative harder to defend.

Primary comparison: semantic composition (monolithic) vs. logical composition (PoE). SuperDiff AND is retained as a secondary condition (still included in trajectory experiments) but the paper text and captions should ground all theoretical claims in PoE, with SuperDiff as corroborating visual evidence of a related phenomenon.

### 1.4 Metric Defensibility

| Metric | Role | Justification | Action |
|--------|------|---------------|--------|
| Decoded images | PRIMARY | Direct visual evidence; non-expert readable | Keep; must span all 4 groups |
| MDS latent trajectories | PRIMARY | Mechanistic structural evidence; distance-centric; shared x_T controls stochasticity | Keep; report MDS stress |
| Terminal d_T KDE | PRIMARY | Quantitative backbone of H1; pooled across 576 records | Keep; add taxonomy-stratified variant |
| Cycle consistency (SD-IPC) | SECONDARY | Lower-bound structural claim; the strongest argument | Keep; extend to multi-pair |
| Temporal decomposition (10-step bins) | SECONDARY | Gap timing / mechanistic interpretation | Keep as plot_26-right |
| Within-AND noise floor | VALIDITY CHECK | Confirms gap is structural, not stochastic AND noise | Keep as supplementary |
| CLIP terminal similarity | SECONDARY | Semantic sanity check on terminal decodes | Keep as plot_12 |
| Jeffrey's divergence / KL analysis | SUPPLEMENTARY | Redundant with KDE for primary claim | Move to supplementary, not main paper |

- **ADD:** Taxonomy-stratified d_T bar chart (Group 1–4 on x-axis, conditions as bar clusters). This is the key new quantitative figure that links the theory-grounded taxonomy to the measured gap.
- **ADD:** Orthogonality-dot vs. d_T scatter (Supplementary S2): Lemma 8.1 heuristic score on x-axis, observed mean d_T on y-axis, colored by group. Shows theory predicts difficulty quantitatively.

---

## Part 2 — Proposal Rewrites

### 2.1 research_goal.tex

**Goal 1 (Phase 1):** Add one sentence clarifying that the concept-pair taxonomy is grounded in the conditions from Bradley et al. (2502.04549) — specifically Factorised Conditionals, Bayes regime co-occurrence, and score-delta orthogonality — rather than in semantic intuition.

**Goal 2 (new Phase 2):** Replace the current vague description with: the goal is to empirically determine whether the composability gap is caused by representation entanglement rather than by the inference-time composition rule. The testbed is CLEVR (controlled, ground-truth factorisation) rather than CXR. The success criterion is that the 2×2 ablation (entangled vs factorised representation × simple vs improved composition rule) shows that only the representation axis matters, not the rule axis.

**Goal 3 (formerly Phase 2, now Phase 3):** Rename to Goal 3. Keep as is. Refers to CXR + H3.

### 2.2 research_hypothesis.tex

**H1 —** Add: "The pair-type dependence of these failure modes is theoretically predictable using the Factorised Conditionals framework of Bradley et al. (2502.04549). Specifically, compositions where neither the Bayes co-occurrence regime nor feature-space disentanglement applies are predicted to exhibit the largest gap, as the score fields combine conflicting gradients without a nearby training attractor." Remove the relational/contextual/adversarial framing.

**H2 —** Add: "Specifically, a VAE trained with explicit mutual-information minimisation and orthogonality constraints between concept-specific latent heads will produce significantly higher composition quality under an identical composition rule than a VAE trained without such constraints. Conversely, applying a geometrically improved composition rule (SLERP on the hypersphere) to an entangled VAE will not significantly improve composition quality, confirming that representation is the active bottleneck." Add the CLEVR SLERP falsification language.

**H3 —** Keep as is.

### 2.3 phase_1.tex (Taxonomy Section, Subsec. concept_pair_taxonomy)

Replace the three-group taxonomy (Independent/Relational/Adversarial) with the four-group theory-grounded taxonomy from Section 1.1 of this plan. Cite Bradley et al. for each group's definition. Change the figure caption for `decoded_images_grid.png` to reference the four groups.

Change the PoE/SuperDiff framing: make PoE the primary logical composition operator throughout phase_1.tex. Change "SuperDiff AND" from primary to "SuperDiff AND, a more aggressive variant of logical composition". All theoretical derivations cite Bradley et al. on linear score addition.

Add a new short subsection (1-2 paragraphs): "Theoretical Basis for Group Assignment". Cites Bradley et al. Explains Factorised Conditionals in one sentence. Explains why Groups 1-2 are positive controls and Groups 3-4 are stress tests. Reference the orthogonality heuristic (Lemma 8.1) as the operationalisation of group assignment.

Update `taxonomy.tex` figure reference: the 4×5 grid figure (new `plot_taxonomy_figure.py`).

### 2.4 New phase_2.tex (H2: CLEVR + SepVAE)

This is a new file. Structure:

**Section header:** "Phase 2: Testing the Representation Hypothesis with Controlled Compositional Learning"

**Subsection 2.1:** Motivation — Phase 1 established that the gap is pair-type-predictable and trajectory-level. The next question is whether the cause is in the representation or the inference rule. Without settling this, H2 remains unestablished.

**Subsection 2.2:** CLEVR as the Primary Testbed — Why CLEVR (ground-truth factorisation, cheap, reproducible, no ethics barrier). Why not CXR here (reserved for H3 / Phase 3).

**Subsection 2.3:** The SepVAE Architecture — z_c, z_d1, z_d2. Training stages d0→d3. Loss terms (reconstruction, nulling, MI minimisation, orthogonality). Reference h2_phase2_experimental_design.md.

**Subsection 2.4:** The 2×2 Ablation — The load-bearing falsification test. Four cells. Case B is the critical test. Prediction: d0 row fails regardless of κ rule (including SLERP). d3 row passes.

**Subsection 2.5:** The MI Causal Curve — λ_mi sweep, composition_score vs MI(z_d1, z_d2). Monotonic negative correlation is the causal evidence.

**Subsection 2.6 (Secondary):** Bridging to Phase 1 — Group 3 + VAE bridge experiment. Demonstrates the principle in the natural-image domain.

### 2.5 Old phase_2.tex → renamed phase_3.tex

Move the medical imaging content (VinBigData, Cardiomegaly, Pleural Thickening, spatial orthogonality) to a new `phase_3.tex`. Update all internal references. Phase 3 is now the H3 clinical application. This section is mostly complete and needs only structural renaming.

---

## Part 3 — Code Changes

### 3.1 Modify (targeted changes)

**`scripts/run_control_landscape.py`**
- Replace `GROUPS` dict with four-group theory-grounded taxonomy: `group1_cooccurrence`, `group2_disentangled`, `group3_stress`, `group4_collision`
- Re-sort existing pairs into the four new groups based on heuristic results
- This file's `GROUPS` is imported by `run_composable_diffusion_control_groups.py` so both update together

**`scripts/run_composable_diffusion_control_groups.py`**
- Update `EXTRA_GROUPS` to align with new taxonomy group names
- Re-sort `positive_object_scene` → group1, `orthogonal_object_style` → group2, `orthogonal_layered_layout` pairs → group3 (except sailboat+lighthouse → group1)

**`scripts/trajectory_dynamics_experiment.py`**
- Add `taxonomy_group` field to summary.json output so results can be aggregated by group later
- No changes to the measurement logic

**`scripts/cycle_consistency_analysis.py`**
- Extend from hardcoded single pair (cat+dog) to accept a list of pairs + taxonomy group labels
- Loop over 2-3 representative pairs per group (8-12 pairs total), 3 seeds each
- Output per-pair per-seed cycle consistency scores in a JSON that `aggregate_taxonomy_results.py` can read

**`scripts/characterize_gap.py`**
- Extend `PAIR_ANNOTATIONS` to include all pairs from the four new taxonomy groups
- Add taxonomy group as an annotation field so the scatter plot (orth_dot vs d_T) can be group-colored

### 3.2 Create (new scripts)

| Script | Purpose | Priority |
|--------|---------|----------|
| `scripts/run_taxonomy_control_groups.py` | Replacement for old control group runner; accepts `--groups group1 group2 group3 group4`; runs `trajectory_dynamics_experiment.py` for all pairs with taxonomy labels | Phase 1, HIGH |
| `scripts/aggregate_taxonomy_results.py` | Reads all summary.json from taxonomy runs; computes per-group mean d_T ± std, cycle-break rate; writes `taxonomy_summary.json` | Phase 1, HIGH |
| `scripts/plot_taxonomy_figure.py` | Figure A: 4×5 decoded image grid with per-group d_T bars | Phase 1, HIGH |
| `scripts/score_field_schematic.py` | Figure B: two-Gaussian score field schematic (pure matplotlib, no model inference) | Phase 1, MEDIUM |
| `models/clevr_sepvae.py` | SepVAE for CLEVR; three encoder heads (z_c, z_d1, z_d2); CNN; MI loss (MINE or CLUB); orthogonality loss | Phase 2, HIGH |
| `scripts/train_clevr_sepvae.py` | Multi-stage trainer d0→d3; logs composition_score at each stage checkpoint; accepts `--stage` and `--mi_weight` | Phase 2, HIGH |
| `scripts/eval_h2_ablation.py` | 2×2 ablation evaluator; loads d0 + d3 checkpoints; runs κ_subst and κ_slerp (port SLERP from `run/analyze_sepvae_manifold.py`); outputs CSV | Phase 2, HIGH |
| `scripts/plot_h2_ablation.py` | 2×2 heatmap figure (Figure H2-2) | Phase 2, HIGH |
| `scripts/plot_mi_vs_composition.py` | MI vs composition_score causal curve (Figure H2-3) | Phase 2, HIGH |
| `scripts/group3_vae_bridge.py` | Secondary bridge experiment; Group 3 pairs; small orthogonality-constrained VAE on natural images; rerun PoE | Phase 2, SECONDARY |

### 3.3 Keep unchanged

- `scripts/measure_composability_gap.py` — comprehensive; no changes
- `scripts/plot_gap_analysis.py` — keep all 35 plots; add one new plot (group-stratified KDE)
- `scripts/plot_trajectory_analysis.py` — keep all 14 plots
- `scripts/composability_orthogonality_heuristic.py` — keep as utility
- `notebooks/composition_experiments.py` — keep; all trajectory infrastructure
- `notebooks/utils.py` — keep

### 3.4 Remove / Archive

- `scripts/superseded/` — already archived; confirm excluded from all pipeline calls
- `scripts/composability_gap_clip_analysis.py` — functionality superseded by `measure_composability_gap.py --pstar-source vlm`. Archive after confirming no unique logic.

### 3.5 Import from Original Repos (Not Reimplement)

- PoE composition logic: import `ComposableStableDiffusionPipeline` from `compositions/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch/` directly; do not reimplement the score addition loop in new scripts
- SuperDiff on SD3.5: the local adaptation in `trajectory_dynamics_experiment.py` is necessary because the original repo targets SD1.x; keep the local adaptation but document it derives from `compositions/super-diffusion/scripts/compose_and_sd3.py`

---

## Part 4 — Run Sequence

### Phase 1 Sequence

```bash
# Step 1: Update taxonomy group definitions in code (no experiments)
# Modify run_control_landscape.py and run_composable_diffusion_control_groups.py

# Step 2: Re-label existing results under new taxonomy
python scripts/aggregate_taxonomy_results.py \
  --relabel-map taxonomy_relabel.json \   # maps old pair slugs to new group names
  --out experiments/eccv2026/taxonomy/

# Step 3: Run trajectory experiments for any Group 3 / Group 4 pairs
# not yet measured under the 5-condition protocol
conda activate attend_excite
python scripts/run_taxonomy_control_groups.py \
  --groups group3_stress group4_collision \
  --seeds 42 43 44

# Step 4: Run extended cycle consistency over all four groups
conda activate attend_excite
python scripts/cycle_consistency_analysis.py \
  --taxonomy-json experiments/eccv2026/taxonomy/taxonomy_summary.json \
  --seeds 42 43 44 \
  --out experiments/eccv2026/cycle_consistency/

# Step 5: Aggregate and produce taxonomy summary
python scripts/aggregate_taxonomy_results.py \
  --out experiments/eccv2026/taxonomy/taxonomy_summary.json

# Step 6: Produce Phase 1 figures
python scripts/plot_taxonomy_figure.py \
  --taxonomy-summary experiments/eccv2026/taxonomy/taxonomy_summary.json
python scripts/score_field_schematic.py \
  --out paper/eccv/figures/score_field_schematic.pdf
```

### Phase 2 Sequence

```bash
# Step 1: Download CLEVR (or generate synthetic CLEVR with 2 objects per scene)
# single-object images for training: ~20k images per concept (color/shape attribute)

# Step 2: Train d0 (entangled baseline)
conda activate cxr   # or a new clevr env
python scripts/train_clevr_sepvae.py \
  --stage d0 --epochs 50 \
  --out runs/clevr_d0/

# Step 3: Train d3 (fully disentangled)
python scripts/train_clevr_sepvae.py \
  --stage d3 --mi_weight 0.10 --orth_weight 0.05 --epochs 50 \
  --out runs/clevr_d3/

# Step 4: Run 2×2 ablation (Cases A, B, C, D)
python scripts/eval_h2_ablation.py \
  --d0_ckpt runs/clevr_d0/best.pt \
  --d3_ckpt runs/clevr_d3/best.pt \
  --rules kappa_subst kappa_slerp kappa_diff \
  --out results/h2_ablation.csv

# Step 5: MI sweep for causal curve (λ_mi ∈ {0, 0.01, 0.05, 0.10, 0.20})
for lmi in 0.0 0.01 0.05 0.10 0.20; do
  python scripts/train_clevr_sepvae.py \
    --stage d2 --mi_weight $lmi --epochs 50 \
    --run_name clevr_d2_lmi${lmi}
done

# Step 6: Produce Phase 2 figures
python scripts/plot_h2_ablation.py --results results/h2_ablation.csv
python scripts/plot_mi_vs_composition.py --results_dir results/

# Step 7 (Secondary): Bridge experiment on Group 3 pairs
python scripts/group3_vae_bridge.py \
  --pairs "a desk lamp,a glacier" "a typewriter,a cactus" "a bathtub,a hot air balloon" \
  --out experiments/eccv2026/group3_vae_bridge/
```

---

## Part 5 — Figures to Produce

### Phase 1 Figures

| Figure | Description | Script | Supports |
|--------|-------------|--------|----------|
| Fig 1: Taxonomy Grid | 4 rows (groups) × 5 cols (conditions); decoded images, shared x_T per row | `plot_taxonomy_figure.py` | H1: gap is group-dependent |
| Fig 2: Trajectory MDS | 4 panels (one per group's representative pair); fan-out from shared x_T | `plot_trajectory_analysis.py` | H1: divergence is structural |
| Fig 3: d_T KDE + Temporal | Left: pooled KDE; Right: 10-step temporal decomposition | `plot_gap_analysis.py` (plot_26) | H1: gap is quantitative, late-amplified |
| Fig 4: Taxonomy d_T Bars | 4-group × 3-condition bar chart with error bars | `plot_taxonomy_figure.py` | H1: gap increases Group 1→4 |
| Fig 5: Cycle Consistency | AND vs monolithic cycle-break grid, 2-3 Group 3+4 pairs | `cycle_consistency_analysis.py` | H1: AND target is structurally unreachable |
| Supp S1: Noise Floor | Within-AND d_T vs mono-AND d_T — validity check | `plot_gap_analysis.py` (plot_11) | Methodological validity |
| Supp S2: Orth vs d_T | Scatter: Lemma 8.1 orth_dot × observed d_T, group-colored | new in `plot_taxonomy_figure.py` | Theory-empirical link |
| Schematic: Score Field | Two-Gaussian illustration; training manifold; conflicting gradients | `score_field_schematic.py` | Mechanistic intuition |

### Phase 2 Figures

| Figure | Description | Script | Supports |
|--------|-------------|--------|----------|
| Fig H2-1: CLEVR Examples | 3 rows (d0, d3, GT) × 4 cols (src A, src B, κ_subst, κ_slerp) | `eval_h2_ablation.py` | H2: qualitative intuition |
| Fig H2-2: 2×2 Heatmap | 2×2 (entangled/factorised × subst/slerp); composition_score; red/green | `plot_h2_ablation.py` | H2: load-bearing result |
| Fig H2-3: MI Causal Curve | x: MI(z_d1,z_d2); y: composition_score; monotonic negative | `plot_mi_vs_composition.py` | H2: causal mechanism |
| Fig H2-4: Stage Evolution | x: training stage (d0→d3); y: composition_score; bar per stage | `train_clevr_sepvae.py` logging | H2: which stage closes the gap |
| Supp H2-5: Bridge | Group 3 pairs; baseline PoE vs VAE-conditioned PoE vs d_T bars | `group3_vae_bridge.py` | H1→H2 bridge |

---

## Part 6 — Metric-to-Claim Mapping

| Metric | Hypothesis Claim | Confirms When | Falsifies When |
|--------|-----------------|---------------|----------------|
| MDS trajectory fan-out | H1: divergence is structural and early | Trajectories separate within first 5 steps; monolithic and AND terminate in distinct manifold regions | All trajectories co-terminate; MDS stress is too high to interpret |
| Terminal d_T KDE | H1: semantic ≠ logical, quantitatively | Monolithic KDE is right-shifted vs within-AND noise floor; consistent across seeds | Monolithic KDE overlaps within-AND floor |
| Cycle consistency | H1: AND target is semantically unreachable | AND cycle-break rate >> monolithic cycle-break rate; pattern holds across groups | AND and monolithic have statistically equivalent cycle-close rates |
| Temporal decomposition | H1: gap amplifies late in denoising | Top 2-3 late bins account for >50% of total cumulative gap | Gap accumulates uniformly |
| Taxonomy-stratified d_T bars | H1: gap is group-type-predictable | d_T increases monotonically from Group 1 → Group 4 | Groups do not separate; all bars at same height |
| 2×2 ablation composition_score | H2: representation is the bottleneck | d0 row (both κ_subst and κ_slerp) is low; d3 row is high | κ_slerp fixes d0 (Case B passes): H2 is FALSIFIED |
| MI vs composition_score | H2: MI reduction is the active mechanism | Monotonic negative correlation across λ_mi sweep | No correlation; composition_score is independent of MI |

---

## Part 7 — What Is Kept, Modified, Removed, Deferred

| Item | Decision | Reason |
|------|----------|--------|
| Three-group taxonomy | REMOVED | Not grounded in Bradley et al.; replaced by four-group theory-grounded taxonomy |
| SuperDiff AND as primary condition | DEMOTED to secondary | Theory (Bradley et al.) only covers linear score addition; SuperDiff is a different operation |
| SuperDiff AND as secondary condition | KEPT | Still in trajectory experiments; useful corroborating visual evidence |
| MDS trajectories | KEPT as PRIMARY | Distance-centric; mechanistically grounded; shared x_T controls stochasticity |
| Terminal d_T KDE | KEPT as PRIMARY | Quantitative backbone |
| Cycle consistency (SD-IPC) | KEPT as SECONDARY | Lower-bound structural claim; the strongest structural argument in H1 |
| Jeffrey's divergence / KL analysis | DEMOTED to supplementary | Redundant with KDE for primary claim |
| Domains A-F (MNIST, CelebA, etc.) | REMOVED from Phase 2 | CLEVR is more controlled; domains A-F dilute focus without improving the H2 argument |
| CXR as Phase 2 testbed | DEFERRED to Phase 3 | Ethics/access barriers; reserved for H3 |
| CLEVR as Phase 2 testbed | ADDED | Ground-truth factorisation; cheap; reproducible; directly replicates Setting A from Bradley et al. |
| Group 3 + VAE bridge experiment | KEPT as SECONDARY | Valuable H1→H2 bridge but structurally weaker than CLEVR 2×2 |
| 2×2 ablation (Case B SLERP) | CRITICAL ADDITION | The load-bearing falsification test for H2; without it the thesis is vulnerable |
| Phase 2 (medical imaging) | RENAMED to Phase 3 | Renumber all .tex files and references |
| Phase 3 (Riemannian LDM) | DEFERRED / DROPPED | Speculative, underdeveloped; focus resources on Phases 1-2 |

---

## Critical Files

| File | Change |
|------|--------|
| `proposal/proposal_stage_3/chapters/research_objectives/research_goal.tex` | Update Goals 1-3 |
| `proposal/proposal_stage_3/chapters/research_objectives/research_hypothesis.tex` | Update H1, H2 wording |
| `proposal/proposal_stage_3/chapters/research_method/phase_1.tex` | Replace taxonomy section; demote SuperDiff |
| `proposal/proposal_stage_3/chapters/research_method/phase_2.tex` | NEW: CLEVR + SepVAE + 2×2 ablation |
| `proposal/proposal_stage_3/chapters/research_method/phase_3.tex` | Rename from old phase_2; keep content |
| `scripts/run_control_landscape.py` | Replace GROUPS dict with four-group taxonomy |
| `scripts/run_composable_diffusion_control_groups.py` | Update EXTRA_GROUPS to match |
| `scripts/trajectory_dynamics_experiment.py` | Add taxonomy_group to summary.json output |
| `scripts/cycle_consistency_analysis.py` | Extend to multi-pair loop |
| `scripts/characterize_gap.py` | Extend PAIR_ANNOTATIONS to all four groups |
| `plans/h2_phase2_experimental_design.md` | Add CLEVR domain, Case B SLERP spec, domain simplification rationale |

---

## Verification

**Phase 1 is verified when:**
- [ ] Taxonomy-stratified d_T bar chart shows monotonic increase from Group 1 → Group 4
- [ ] MDS trajectory panels show fan-out pattern for all four groups
- [ ] Cycle consistency rate for AND is significantly lower than for monolithic across Groups 3-4
- [ ] Within-AND noise floor (Supp S1) confirms the gap is structural, not stochastic

**Phase 2 is verified when:**
- [ ] Case B (d0 + κ_slerp) fails — this is the falsification test; if it passes, H2 is false
- [ ] Cases C and D (d3 + κ_subst/κ_slerp) both pass
- [ ] MI vs composition_score curve is monotonically negative across λ_mi sweep
- [ ] Per-stage composition logging shows d3 improvement > d2 improvement (orthogonality is the key stage)
