# Hybridization vs. Co-Presence in Compositional Diffusion Models

**ECCV 2026 submission** · Python 3.11 · PyTorch 2.10 · Stable Diffusion 3.5 Medium

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Hypothesis](#2-hypothesis)
3. [Claims and Objectives](#3-claims-and-objectives)
4. [Theoretical Background](#4-theoretical-background)
5. [Repository Structure](#5-repository-structure)
6. [Installation](#6-installation)
7. [Experimental Pipeline](#7-experimental-pipeline)
   - [Phase 0 — Geometric Evidence](#phase-0--geometric-evidence)
   - [Phase 1 — Inverter Training Data](#phase-1--inverter-training-data)
   - [Phase 2 — Train the Inverter](#phase-2--train-the-inverter)
   - [Phase 3 — Measure the Composability Gap](#phase-3--measure-the-composability-gap)
   - [Phase 4 — Characterise the Gap Taxonomy](#phase-4--characterise-the-gap-taxonomy)
   - [Ablation Study](#ablation-study)
   - [Manifold Analysis](#manifold-analysis)
8. [Visualisation Suite](#8-visualisation-suite)
9. [Expected Outputs and How to Read Them](#9-expected-outputs-and-how-to-read-them)
10. [Falsification Conditions](#10-falsification-conditions)
11. [Paper Reference](#11-paper-reference)

---

## 1. Introduction

Text-to-image diffusion models are routinely used for *compositional generation* — creating images that contain multiple concepts, satisfy multiple constraints, or blend multiple attributes. Two fundamentally different approaches exist, and they are almost never distinguished:

**Semantic composition** encodes the composition into a natural-language prompt. Writing `"a cat and a dog"` asks the model to find the learned representation of that phrase in its training distribution. This is fast, intuitive, and produces visually coherent images — but it is ultimately a lookup in a corpus-statistics table. The model generates what images captioned with `"a cat and a dog"` looked like during training.

**Logical composition** (SuperDiff AND, product-of-experts) does not use a joint prompt. It runs separate score networks for each concept — one for `"a cat"`, one for `"a dog"` — and combines their score functions at every denoising step. The result is defined by the joint probability density under both individual models, not by any single conditioning vector. This is compositionally correct by construction, but it bypasses the learned language-to-image associations entirely.

**The problem.** These two paradigms are frequently treated as interchangeable alternatives, or as members of the same family of methods. Papers benchmark them side by side as if the difference were only quantitative. This paper argues they are structurally distinct operations, that conflating them leads to wrong conclusions about what each can and cannot express, and that the right framework is not to choose between them but to understand exactly where and how they diverge — and how to bridge that gap.

---

## 2. Hypothesis

> *From a shared noise origin* $x_T$*, semantic composition (monolithic prompt) and logical composition (SuperDiff AND) carve out distinct trajectories through the latent flow field of SD 3.5, converging to different terminal basins. This divergence is structural — a consequence of different inductive biases — not stochastic noise.*

**Semantic composition** encodes a single point in SD 3.5's conditioning space: a learned embedding that the model maps to its corpus mode for that phrase. The denoising trajectory is pulled toward a co-occurrence attractor shaped by training statistics. The model is doing *pragmatic* reasoning: what does this phrase typically look like?

**Logical composition** operates on the score field directly. At each denoising step, it computes the gradient of the log-probability under each individual concept model and combines them (via SuperDiff's Proposition 6 normalisation). The trajectory is pulled toward the region of genuinely highest joint density under both marginals. The model is doing *logical* reasoning: what would satisfy both constraints simultaneously?

**The geometric suspicion.** The two trajectories diverge early (within the first 2–4 denoising steps) because the inductive bias is injected during structure formation, not during fine-detail refinement. Late-stage trajectories run roughly in parallel, separated in latent space. The terminal gap — measured as per-element MSE between final latents — is the operational signature of two different inductive biases operating in the same denoising manifold.

---

## 3. Claims and Objectives

### Claims

| # | Claim | What would falsify it |
|---|---|---|
| C1 | Semantic AND ≠ logical AND | Terminal latent MSE is trivially small across all pairs and seeds |
| C2 | The two paradigms diverge early (steps 2–4), not late | Divergence onset is concentrated in the last third of denoising |
| C3 | SuperDiff AND occupies a geometrically distinct regime | PCA/MDS trajectory projections from shared $x_T$ overlap completely |
| C4 | Solo conditioning is a weak proxy for composition | $d_T^{c_1} \approx 0$ or $d_T^{c_2} \approx 0$, i.e. one concept dominates |
| C5 | The gap magnitude varies by concept pair | All pairs produce identical gap distributions |
| C6 | A learned inverter $f_\theta$ can partially (not fully) bridge the gap | $d_T^{p^*} \approx d_T^{\text{mono}}$ (inverter gains nothing) or $d_T^{p^*} \approx 0$ (gap fully closed) |
| C7 | Guided hybrid $v_\alpha = (1-\alpha)v_{\text{mono}} + \alpha v_{\text{superdiff}}$ reduces the gap | No value of $\alpha \in (0,1)$ outperforms either pure condition |

### Objectives

1. Establish trajectory-level geometric evidence that CLIP AND and SuperDiff AND are dynamically distinct.
2. Quantify the composability gap at three levels: image (CLIP cosine, LPIPS), latent (MSE at terminal step), and trajectory (step-wise MSE and cosine similarity).
3. Learn the best-possible semantic proxy $p^*$ for each SuperDiff AND output and measure the residual gap — the *semantic representability horizon*.
4. Characterise what structural property of the concept pair (CLIP cosine similarity between $e(c_1)$ and $e(c_2)$) predicts the gap magnitude.
5. Evaluate guided hybrid composition as a practical bridge.

### Contributions

- **Shared-noise trajectory analysis protocol** comparing monolithic CFG, SuperDiff AND (fm\_ode variant), PoE, and guided hybrid from a fixed $x_T$, with per-step divergence diagnostics and kappa tracking across 67 reproducible runs.
- **Approximate inversion framework**: a closed-loop CLIP+MLP inverter $f_\theta$ ($\approx$8M parameters) trained entirely within SD 3.5 to find the best single conditioning vector $p^*$ approximating SuperDiff AND outputs, without external image-caption data.
- **Three-level composability gap metric**: image, latent, and trajectory levels.
- **Composability gap taxonomy** correlating residual gap magnitude with the CLIP geometry of the concept pair.

---

## 4. Theoretical Background

### 4.1 Score-based diffusion and flow matching

SD 3.5 uses a flow-matching objective. The denoising process solves an ODE:

$$\frac{dx_t}{dt} = v_\theta(x_t, t, c)$$

where $v_\theta$ is the velocity field predicted by the transformer, conditioned on a text embedding $c$. Classifier-free guidance scales the conditional component:

$$v_{\text{cfg}} = v_{\text{uncond}} + w \cdot (v_{\text{cond}} - v_{\text{uncond}})$$

### 4.2 Semantic composition via a monolithic prompt

Writing `"a cat and a dog"` maps to a single point $c_{\text{mono}}$ in SD 3.5's conditioning space (pooled and sequence embeddings from CLIP-L, CLIP-G, and T5-XXL). The velocity field $v_\theta(x_t, t, c_{\text{mono}})$ steers the trajectory toward the corpus mode for that phrase. The composition happens implicitly through attention: the model has seen many images captioned with this phrase and learned their statistics.

**Key limitation.** If the training corpus never contained images of cats and dogs together, the model has no mode to find. More subtly, even if it did, it learned the *joint* distribution conditioned on co-occurrence — not the product of marginals.

### 4.3 Logical composition via SuperDiff AND

SuperDiff AND (the `fm_ode` variant) combines velocity fields from two separate conditioning vectors at every step:

$$v_{\text{AND}} = \kappa_1(t) \cdot v_\theta(x_t, t, c_1) + \kappa_2(t) \cdot v_\theta(x_t, t, c_2)$$

where $\kappa_1(t), \kappa_2(t)$ are adaptive weights computed from the Proposition 6 normalisation condition, which ensures the composed score function remains a valid log-probability gradient. This targets the product-of-experts distribution:

$$p_{\text{AND}}(x) \propto p(x \mid c_1) \cdot p(x \mid c_2)$$

**Key difference.** There is no joint embedding. The operator never asks "what does a cat-and-dog image look like?" — it asks "what direction would simultaneously increase the probability under both marginals?"

### 4.4 The composability gap

The composability gap is the discrepancy between these two operations, measured at the level of final latents:

$$\Delta_T = \text{MSE}(z_T^{\text{mono}}, z_T^{\text{AND}})$$

and the residual gap after inversion:

$$\Delta_T^{p^*} = \text{MSE}(z_T^{p^*}, z_T^{\text{AND}})$$

where $p^* = f_\theta(\text{decode}(z_T^{\text{AND}}))$ is the best single SD 3.5 conditioning vector that approximates the AND output. The portion $\Delta_T^{p^*}$ that survives inversion is structurally inexpressible in SD 3.5's semantic space — it is the *semantic representability horizon* of logical composition.

### 4.5 Guided hybrid

A linear velocity interpolation defines a family of trajectories:

$$v_\alpha = (1-\alpha) \cdot v_{\text{mono}} + \alpha \cdot v_{\text{AND}}, \quad \alpha \in [0, 1]$$

This is not a probabilistic mixture — it is a geometric interpolation in velocity space. At $\alpha=0$ we recover monolithic CFG; at $\alpha=1$ we recover pure SuperDiff AND. At intermediate $\alpha$, the trajectory is pulled toward a region that is semantically plausible (anchored by the monolithic velocity) and logically balanced (corrected by the AND velocity).

---

## 5. Repository Structure

```
eccv2026/
│
├── scripts/                       # Experiment and analysis scripts
│   ├── trajectory_dynamics_experiment.py    # Phase 0: geometric evidence
│   ├── generate_inversion_training_data.py  # Phase 1: training data
│   ├── train_inverter.py                    # Phase 2: train f_θ
│   ├── measure_composability_gap.py         # Phase 3: gap measurement
│   ├── characterize_gap.py                  # Phase 4: gap taxonomy
│   ├── ablation_semantic_vs_logical.py      # Ablation study
│   ├── manifold_cooccurrence_vs_hybrid.py   # Manifold geometry
│   ├── generate_sd35.py                     # Utility: SD3.5 generation
│   ├── generate_compositional_model_grid.py # Utility: model comparison
│   ├── make_comparison_figure.py            # Utility: publication figures
│   ├── plot_gap_analysis.py                 # Vis: gap suite (plots 00–13)
│   ├── plot_trajectory_analysis.py          # Vis: trajectory suite (plots 00–13)
│   ├── plot_composability_histogram.py      # Vis: violin + strip plots
│   ├── plot_gap_clean.py                    # Vis: clean strip chart
│   └── plot_trajectory_histogram.py         # Vis: time-resolved histograms
│
├── notebooks/                     # Core library modules (imported by scripts)
│   ├── utils.py                   # Model loading, text encoding
│   ├── composition_experiments.py # Trajectory tracking, sampling functions
│   └── dynamics.py                # Velocity, latent, SuperDiff core
│
├── models/
│   └── sd35_inverter.py           # SD35ConditioningInverter architecture
│
├── paper/
│   ├── eccv2016submission.tex     # Main paper
│   ├── abstract.tex               # Standalone abstract
│   ├── research_blueprint.md      # Full research narrative
│   └── egbib.bib                  # Bibliography
│
├── datasets/
│   └── coco_common_pairs.json     # Curated concept pair definitions
│
├── experiments/                   # All experiment outputs (moved here)
│   ├── trajectory_dynamics/       # Phase 0 runs (timestamped)
│   ├── inversion/
│   │   ├── training_data/         # Phase 1 output
│   │   └── gap_analysis/          # Phase 3 output
│   ├── composition_analysis_*/    # Earlier composition analysis runs
│   ├── eccv2026/                  # Structured ablation/guidance runs
│   └── semantic_prompt_search_*/  # Phase 0 semantic search outputs
│
├── requirements_eccv.txt          # Pinned dependencies (PyTorch stack only)
├── INSTALL.md                     # Step-by-step installation guide
└── README.md                      # This file
```

All scripts use `sys.path.insert(0, PROJECT_ROOT)` where `PROJECT_ROOT` is resolved as `Path(__file__).parent.parent`. Run all commands from this `eccv2026/` directory root.

---

## 6. Installation

See [INSTALL.md](INSTALL.md) for full instructions including GPU-specific PyTorch builds and RunPod setup.

**Quick start (local machine with CUDA 12.x):**

```bash
# 1. Create a dedicated conda environment
conda create -n eccv2026 python=3.11 -y
conda activate eccv2026

# 2. Install PyTorch with CUDA support
#    Adjust cu121 to match your CUDA version (cu118, cu124, etc.)
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements_eccv.txt --no-deps --upgrade

# 4. Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
python -c "from diffusers import SD3Transformer2DModel; print('diffusers OK')"
```

**Model access.** The SD 3.5 Medium model requires a HuggingFace account and acceptance of the model licence at `stabilityai/stable-diffusion-3.5-medium`. Authenticate once with:

```bash
huggingface-cli login
```

---

## 7. Experimental Pipeline

The pipeline has five ordered phases. Each phase produces outputs that subsequent phases consume. Run them in the order shown. All examples assume you are in the `eccv2026/` root directory with the `eccv2026` conda environment active.

---

### Phase 0 — Geometric Evidence

**Script:** `scripts/trajectory_dynamics_experiment.py`

**Research claim addressed:** *C1, C2, C3 — Semantic AND ≠ logical AND; they diverge within the first 2–4 denoising steps and terminate in geometrically separated regions of latent space.*

**What it does.** Runs four conditions from the *same initial Gaussian noise* $x_T$:
1. Prompt A alone
2. Prompt B alone
3. Monolithic CLIP AND (`"prompt_a and prompt_b"`)
4. SuperDiff AND (`fm_ode` variant)

Records the full latent trajectory at every denoising step via `LatentTrajectoryCollector`, projects all conditions jointly into 2D via PCA and MDS, and computes pairwise L2 distances and kappa dynamics.

**Expected output.** A timestamped directory in `experiments/trajectory_dynamics/<YYYYMMDD_HHMMSS>/` containing:
- `trajectories_pca.png` / `trajectories_mds.png` — 2D projections with time-coloured paths; monolithic and SuperDiff paths should be visually separated
- `pairwise_l2_distances.png` — divergence curves showing onset at steps 2–4
- `kappa_dynamics.png` — per-step composition weights $\kappa_1(t), \kappa_2(t)$ rebalancing in the first third of denoising
- `summary.json` — machine-readable divergence onset, endpoint L2, CLIP probe labels

**Basic run (single pair, SD3.5):**

```bash
python scripts/trajectory_dynamics_experiment.py \
    --prompt-a "a cat" \
    --prompt-b "a dog" \
    --model-id stabilityai/stable-diffusion-3.5-medium \
    --steps 50 \
    --guidance 4.5 \
    --seed 42 \
    --output-dir experiments/trajectory_dynamics
```

**Guided hybrid sweep** (claim C7 — interpolating velocities reduces the gap at $\alpha \approx 0.3$):

```bash
python scripts/trajectory_dynamics_experiment.py \
    --prompt-a "a book on the left" \
    --prompt-b "a bird on the right" \
    --model-id stabilityai/stable-diffusion-3.5-medium \
    --guidance-sweep 0.0 0.1 0.3 0.5 0.7 1.0 \
    --steps 50 --guidance 4.5 --seed 42 \
    --output-dir experiments/trajectory_dynamics
```

> **Interpretation.** If C1–C3 hold, the PCA plot will show two clearly separated trajectory clusters: monolithic (pulled toward corpus attractor) and SuperDiff (navigating the joint-density region). Divergence should appear in the leftmost portion of the denoising curve (high-noise regime), not at the end.

---

### Phase 1 — Inverter Training Data

**Script:** `scripts/generate_inversion_training_data.py`

**Research claim addressed:** *C6 — A learned inverter $f_\theta$ can partially bridge the gap. This phase generates the closed-loop training data that makes it possible to learn $f_\theta$ at all.*

**What it does.** For each prompt in a curated set (single concepts, simple conjunctions, attribute-rich variants), generates $K$ images with SD 3.5 and saves the paired `(image, conditioning)` ground truth. The conditioning tensors are SD 3.5's own pooled and sequence embeddings — this is a self-supervised closed loop.

**Expected output.** `experiments/inversion/training_data/` with one subdirectory per prompt containing PNG images and `conditioning.pt` files. The default run produces ~680 images from 85 prompts (8 seeds each) at 512×512.

```bash
python scripts/generate_inversion_training_data.py \
    --model-id stabilityai/stable-diffusion-3.5-medium \
    --output-dir experiments/inversion/training_data \
    --images-per-prompt 8 \
    --steps 50 \
    --guidance 4.5 \
    --image-size 512
```

> **Design rationale.** The held-out test pairs (cat/dog, person/umbrella, person/car, car/truck) are excluded from training entirely. This ensures that the gap measured in Phase 3 is not an artefact of memorisation.

---

### Phase 2 — Train the Inverter

**Script:** `scripts/train_inverter.py`

**Research claim addressed:** *C6 — The inverter $f_\theta$ is the semantic representability probe. Its quality (measured by $d_T^{p^*}$) directly quantifies how much of the logical composition is recoverable via a single prompt embedding.*

**What it does.** Loads the Phase 1 dataset and trains `SD35ConditioningInverter`:
- **Backbone**: frozen CLIP ViT-L/14 (768-dim patch tokens)
- **Pooled head**: MLP $768 \to 1024 \to 2048$ (predicts `pooled_prompt_embeds`)
- **Sequence head**: cross-attention decoder (154 learned queries attending over patch tokens, projected to 4096-dim) predicts the CLIP portion of `prompt_embeds`; T5 portion is zeroed

**Loss:** $\mathcal{L} = \text{MSE}(\hat{e}_{\text{pool}}, e_{\text{pool}}) + \lambda \cdot \text{MSE}(\hat{e}_{\text{seq}}, e_{\text{seq}}[:154])$

**Expected output.** Checkpoint files in `ckpt/inverter/` (default path in superdiff-ldm), with `best.pt` selected by validation loss. Training log at `training_log.json`.

```bash
python scripts/train_inverter.py \
    --data-dir experiments/inversion/training_data \
    --ckpt-dir ckpt/inverter \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-4 \
    --lambda-seq 0.1 \
    --val-fraction 0.1 \
    --clip-model-id openai/clip-vit-large-patch14
```

> **Sanity check before proceeding.** Run a closed-loop test: encode a known prompt → generate an image → invert the image with $f_\theta$ → regenerate from $p^*$. The two generated images should be visually close. If they are not, the inverter has not converged and Phase 3 results will reflect inverter failure, not composability gap.

---

### Phase 3 — Measure the Composability Gap

**Script:** `scripts/measure_composability_gap.py`

**Research claims addressed:** *C1, C4, C5, C6 — Quantifies all gap metrics across four held-out concept pairs and multiple seeds.*

**What it does.** For each held-out pair $(c_1, c_2)$ and each seed:

1. Generates from SuperDiff AND → records trajectory $z_t^{\text{AND}}$
2. Inverts the terminal image with $f_\theta$ → gets $p^*$
3. Generates from SD 3.5($p^*$) from the *same initial noise* → records $z_t^{p^*}$
4. Generates from SD 3.5(`"c1 and c2"`) from the *same initial noise* → records $z_t^{\text{mono}}$
5. Generates from SD 3.5($c_1$ alone) and SD 3.5($c_2$ alone) → records $z_t^{c_1}, z_t^{c_2}$

Computes terminal MSE anchored to AND: $d_T^{\text{mono}}, d_T^{p^*}, d_T^{c_1}, d_T^{c_2}$.

**Expected output.** In `experiments/inversion/gap_analysis/`:
- `per_seed_distances.json` — one row per (seed, pair) with all four gap values
- `all_pairs_gap.json` — summary statistics per pair
- `trajectory_distances.json` — step-wise MSE at every denoising step
- Per-pair subdirectories with image grids comparing AND vs $p^*$ vs mono outputs

```bash
python scripts/measure_composability_gap.py \
    --ckpt ckpt/inverter/best.pt \
    --model-id stabilityai/stable-diffusion-3.5-medium \
    --output-dir experiments/inversion/gap_analysis \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
    --steps 50 \
    --guidance 4.5 \
    --image-size 512 \
    --dtype float16
```

> **Key number to watch.** The ordering $d_T^{p^*} < d_T^{\text{mono}}$ and $d_T^{p^*} > 0$ is the central quantitative result of the paper. If $d_T^{p^*} \approx d_T^{\text{mono}}$, inversion has gained nothing and Claim C6 fails. If $d_T^{p^*} \approx 0$, the gap is fully closed by a single prompt, which would undermine the structural inexpressibility argument.

---

### Phase 4 — Characterise the Gap Taxonomy

**Script:** `scripts/characterize_gap.py`

**Research claim addressed:** *C5 — The gap is not symmetric across concept pairs. Its magnitude is predicted by the CLIP cosine similarity between the individual concept embeddings.*

**What it does.** Loads all gap metrics from Phase 3, computes CLIP cosine similarity between the concept embeddings $e(c_1)$ and $e(c_2)$, and produces:
- Scatter plots of CLIP cosine similarity vs each gap metric
- A taxonomy CSV annotating pairs as complementary / hierarchical / co-occurring / same-category
- Trajectory gap curves (step-wise MSE per pair)
- A regression summary JSON

**Expected output.** Figures and CSVs in `experiments/inversion/gap_analysis/`.

```bash
python scripts/characterize_gap.py \
    --gap-dir experiments/inversion/gap_analysis \
    --output-dir experiments/inversion/gap_analysis \
    --model-id stabilityai/stable-diffusion-3.5-medium
```

> **Expected taxonomy.** Based on the hypothesis, same-category pairs (car/truck — high CLIP similarity) should show the smallest gap; complementary pairs (cat/dog) should show a moderate gap; hierarchical pairs (person/umbrella) a larger gap; and co-occurring pairs (person/car) the largest, because their joint training distribution differs most from the product of marginals.

---

### Ablation Study

**Script:** `scripts/ablation_semantic_vs_logical.py`

**Research claim addressed:** *C1, C3 — Direct ablation: do CLIP AND and SuperDiff AND converge to the same latent-space region?*

**What it does.** For a curated set of prompt pairs, runs Prompt A, Prompt B, CLIP AND, and SuperDiff AND from the same $x_T$. Measures cosine similarity between the CLIP AND and SuperDiff AND terminal latents, and produces a summary figure with per-pair trajectory panels and a bar chart of cosine similarities.

```bash
python scripts/ablation_semantic_vs_logical.py \
    --model-id stabilityai/stable-diffusion-3.5-medium \
    --output-dir experiments/eccv2026/ablations \
    --steps 50 --guidance 4.5 --seed 42
```

---

### Manifold Analysis

**Script:** `scripts/manifold_cooccurrence_vs_hybrid.py`

**Research claim addressed:** *C3 — SuperDiff AND occupies a distinct geometric regime; this script visualises that regime on the CLIP hypersphere $S^{767}$ relative to the text anchors and COCO co-occurrence images.*

**What it does.** Embeds all of the following into CLIP ViT-L/14 space:
- Individual text anchors: `"a car"`, `"a truck"`
- Compositional text: `"a car and a truck"`
- Real COCO images containing both objects (genuine co-occurrence)
- SuperDiff AND decoded outputs (logical composition / hybrid)
- Standard SD 3.5 monolithic AND outputs (semantic composition)
- Geodesic midpoint: $\text{normalize}(e_a + e_b)$

Produces MDS/PCA projections and a geodesic decomposition showing tangential vs residual components relative to the great-circle arc between the two concept anchors.

```bash
python scripts/manifold_cooccurrence_vs_hybrid.py \
    --pair "car+truck" \
    --experiment-dir experiments/trajectory_dynamics/<latest_run> \
    --output-dir experiments/eccv2026/manifold
```

---

## 8. Visualisation Suite

All visualisation scripts read from saved JSON/npy outputs. Run them *after* the corresponding experiment phase has completed.

### Gap analysis suite — 14 plots

**Script:** `scripts/plot_gap_analysis.py`

Reads `per_seed_distances.json` and `trajectory_distances.json` from Phase 3.

| Plot | Type | What it shows |
|------|------|---------------|
| 00 | Strip by pair | Raw dots (one per seed) + mean tick; foundation plot |
| 01 | Grouped bar | Mean ± std per pair per condition |
| 02–05 | KDE / histograms | Distribution shape by pair and pooled |
| 06–10 | Stacked bars + bands | When during denoising the gap accumulates |
| 11 | Strip $p^*$ | **Key result**: $p^*$ vs mono vs solo $c_1$ vs solo $c_2$ |
| 12 | CLIP comparison | AND↔$p^*$ vs AND↔mono cosine similarity per pair |
| 13 | JS divergence | $JS^2(P_{p^*}, P_{\text{mono}})$: lower = $p^*$ is a better AND proxy |

```bash
# All 14 plots
python scripts/plot_gap_analysis.py \
    --data-dir experiments/inversion/gap_analysis \
    --output-dir experiments/inversion/gap_analysis/figures

# Single plot
python scripts/plot_gap_analysis.py --plot 11 \
    --data-dir experiments/inversion/gap_analysis \
    --output-dir experiments/inversion/gap_analysis/figures
```

### Trajectory analysis suite — 14 plots

**Script:** `scripts/plot_trajectory_analysis.py`

Reads trajectory data from `experiments/trajectory_dynamics/<run>/`.

| Plots | Layer | What it shows |
|-------|-------|---------------|
| 00 | Identity anchor | Decoded images as the result reference |
| 01–03 | Raw scalars | Latent norms, velocity magnitudes, pairwise L2 |
| 04–06 | Divergence structure | Onset bar chart, path lengths, terminal heatmap |
| 07–09 | Geometry | PCA scree, manifold overlay, per-condition panels |
| 10–11 | Mechanism | Kappa dynamics $\kappa(t)$, log-likelihood per concept |
| 12–13 | Semantic | CLIP cosine similarity to class prompts |

```bash
python scripts/plot_trajectory_analysis.py \
    --data-dir experiments/trajectory_dynamics/<YYYYMMDD_HHMMSS>
```

### Composability histogram (Plotly)

**Script:** `scripts/plot_composability_histogram.py`

Produces violin + strip overlays for terminal distances. Outputs interactive HTML; PNG export requires Chrome (see [INSTALL.md](INSTALL.md)).

```bash
python scripts/plot_composability_histogram.py \
    --data-dir experiments/inversion/gap_analysis \
    --output-dir experiments/inversion/gap_analysis
```

### Clean strip chart

```bash
python scripts/plot_gap_clean.py \
    --data-dir experiments/inversion/gap_analysis \
    --output-dir experiments/inversion/gap_analysis/figures
```

### Time-resolved distance histogram

```bash
python scripts/plot_trajectory_histogram.py \
    --data-dir experiments/inversion/gap_analysis \
    --output-dir experiments/inversion/gap_analysis/figures \
    --n-buckets 8
```

---

## 9. Expected Outputs and How to Read Them

### The central result (Plot 11)

Plot 11 from `plot_gap_analysis.py` is the paper's key figure. It shows a strip chart of terminal MSE distances anchored to SuperDiff AND, with four conditions side by side:

```
d_T^{p*}    < d_T^{mono}   ≈ d_T^{c1}   ≈ d_T^{c2}
  │                │                │              │
  └── inverter     └── best naive   └── single     └── single
      reduces gap      baseline         concept A      concept B
```

If the ordering holds:
- **Inversion works**: $p^*$ is meaningfully closer to AND than a monolithic prompt
- **Gap is not trivial**: $d_T^{p^*} > 0$ means some portion of AND is structurally inexpressible
- **Single concepts are poor proxies**: $d_T^{c_1} \approx d_T^{\text{mono}}$ means AND is not simply dominated by one concept

### Trajectory divergence (Plots 01–03)

Divergence onset at steps 2–4 is the structural signature of Claim C2. Late-onset divergence (steps 40+) would mean composition is a fine-detail effect — a fundamentally different story with different implications for where to intervene.

### Kappa dynamics (Plot 10)

$\kappa(t)$ should show large variation in the first third of denoising (steps 0–17 of 50) and stabilise thereafter. If $\kappa(t)$ is constant, Proposition 6 is not having an effect and the run may have degenerated to a simple PoE.

### Guided hybrid trade-off

In the Phase 0 `--guidance-sweep` run, the per-$\alpha$ L2 to the CLIP AND path should form a concave curve with a minimum at $\alpha \approx 0.3$. This is Claim C7. A flat curve would indicate the hybrid is equivalent to pure monolithic.

---

## 10. Falsification Conditions

The paper's claims are falsifiable. The following observations would require revising or retracting claims:

| Observation | Implication |
|---|---|
| $d_T^{\text{mono}} \approx 0$ for all pairs | C1 fails: semantic AND = logical AND |
| Trajectory PCA shows complete overlap | C3 fails: no geometric distinction |
| Divergence onset concentrated in steps 40–50 | C2 fails: divergence is late-stage fine-detail |
| $d_T^{p^*} \approx d_T^{\text{mono}}$ | C6 fails: inversion gains nothing |
| $d_T^{p^*} \approx 0$ | C6 fails in the other direction: gap is fully closeable |
| All pairs have identical gap distributions | C5 fails: no pair-difficulty effect |
| No $\alpha \in (0,1)$ reduces L2 to CLIP path | C7 fails: hybrid is not useful |

---

## 11. Paper Reference

```
Hybridization vs. Co-Presence in Compositional Diffusion Models.
ECCV 2026 submission.
```

**Keywords:** compositional generation, diffusion models, latent trajectories, approximate inversion, composability gap, logical operators, score composition, flow matching

**Model:** Stable Diffusion 3.5 Medium (`stabilityai/stable-diffusion-3.5-medium`)

**All experimental runs** are stored in `experiments/` with timestamped directories and `summary.json` files providing one-to-one traceability from artifact to paper claim.

---

*For questions about experimental design, see `paper/research_blueprint.md`. For installation issues, see `INSTALL.md`.*
