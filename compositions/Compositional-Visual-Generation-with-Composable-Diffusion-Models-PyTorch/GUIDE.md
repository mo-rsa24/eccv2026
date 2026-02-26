# Compositional Visual Generation — Run Guide

**Paper:** Compositional Visual Generation with Composable Diffusion Models (Liu et al., ECCV 2022)
**Repo:** `compositions/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch`
**Environment:** `compose_diff` (Python 3.8)

---

## What the script does

`scripts/image_sample_compose_stable_diffusion.py` generates a **single image** that
simultaneously satisfies multiple text prompts by composing their diffusion scores at
every denoising step. Prompts are separated by `|`; each carries an independent
guidance weight.

Example:
```bash
python scripts/image_sample_compose_stable_diffusion.py \
  --prompts "mystical trees | A magical pond | dark" \
  --weights "7.5 | 7.5 | 7.5" \
  --scale 7.5 --steps 50 --seed 2
```

---

## Model used

**Stable Diffusion v1-4** (`CompVis/stable-diffusion-v1-4`) — a Latent Diffusion Model (LDM).

| Component | Architecture | Role |
|---|---|---|
| VAE encoder/decoder | KL-regularised CNN | 512×512 image ↔ 64×64×4 latent (factor-8 compression) |
| Text encoder | CLIP ViT-L/14 (`clip-vit-large-patch14`) | Encodes each prompt → 77×768 embedding |
| Denoising backbone | U-Net with cross-attention | Predicts noise residual ε in latent space |
| Scheduler | DDIM (default), LMS, DDPM, PNDM | Steps latents from xₜ → x₀ |

---

## Equations implemented

### 1. Standard Classifier-Free Guidance (CFG) — single prompt baseline

At each timestep *t* the denoised score estimate is:

```
ε̃(zₜ, c) = ε(zₜ, ∅) + w · (ε(zₜ, c) − ε(zₜ, ∅))
```

where:
- `ε(zₜ, c)` — U-Net noise prediction conditioned on prompt *c*
- `ε(zₜ, ∅)` — unconditional (empty prompt) prediction
- `w` — guidance scale (`--scale`, default 7.5)

### 2. Composable Diffusion — AND composition (this paper's contribution)

For *n* prompts c₁, …, cₙ each with weight wᵢ:

```
ε̃(zₜ, c₁,…,cₙ) = ε(zₜ, ∅) + Σᵢ wᵢ · (ε(zₜ, cᵢ) − ε(zₜ, ∅))
```

**Implemented at** `pipeline_composable_stable_diffusion.py:539`:
```python
noise_pred = noise_pred_uncond + (weights * (noise_pred_text - noise_pred_uncond)).sum(dim=0, keepdims=True)
```

#### Theoretical grounding (energy-based composition)

Each prompt *cᵢ* defines an energy:
```
Eᵢ(x) ∝ −log p(cᵢ | x)
```

Under the **AND** (conjunction) operator, the composed distribution is:
```
p(x | c₁ AND … AND cₙ) ∝ p(x) · Πᵢ p(cᵢ | x)
```

Taking the score (gradient of log-probability):
```
∇ log p(x | c₁,…,cₙ) = ∇ log p(x) + Σᵢ ∇ log p(cᵢ | x)
```

Approximating each conditional score with CFG:
```
∇ log p(cᵢ | x) ≈ wᵢ · (ε(zₜ, cᵢ) − ε(zₜ, ∅))
```

yields the composition formula above.

### 3. NOT composition (negation)

Pass a **negative** weight (e.g., `--weights "7.5 | -7.5"`) for concept cᵢ to steer
generation *away* from it:
```
p(x | c₁ AND NOT c₂) ∝ p(x) · p(c₁|x) / p(c₂|x)
```

Score form:
```
∇ log p(x | c₁ AND NOT c₂) = ∇ log p(x) + w₁·(ε(zₜ,c₁)−ε(zₜ,∅)) − w₂·(ε(zₜ,c₂)−ε(zₜ,∅))
```

---

## What remains before running

### Step 1 — Install the local package (not yet done)
```bash
conda activate compose_diff
cd compositions/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch
pip install -e .
```

### Step 2 — Accept the model licence on HuggingFace
`CompVis/stable-diffusion-v1-4` is a gated model.
1. Go to https://huggingface.co/CompVis/stable-diffusion-v1-4
2. Accept the licence (one-time, requires HF account)
3. Log in locally:
```bash
huggingface-cli login
```

### Step 3 — Run
```bash
conda activate compose_diff
cd compositions/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch
python scripts/image_sample_compose_stable_diffusion.py \
  --prompts "mystical trees | A magical pond | dark" \
  --weights "7.5 | 7.5 | 7.5" \
  --scale 7.5 --steps 50 --seed 2
```

Output is saved as a PNG named after the prompts and weights in the current directory.
