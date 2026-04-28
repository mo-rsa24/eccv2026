# SDXL Trajectory Dynamics Guide

This guide explains how to generate `trajectory_2x2_sdxl.png` and other SDXL-based trajectory visualizations.

## What Changed from SD 1.4

### Model Architecture
- **Text Encoding**: SDXL uses **dual text encoders** (CLIP-L + CLIP-G)
  - Returns concatenated embeddings: `(B, 77, 2048)` instead of SD 1.4's `(B, 77, 768)`
  - Both encoders are required for proper conditioning

- **Latent Space**: Same as SD 1.4
  - 4-channel latents with shape `(B, 4, 128, 128)` for 1024×1024 images
  - Same scaling factors and VAE architecture

- **UNet**: SDXL's UNet accepts the dual-encoder embeddings directly
  - No additional processing needed beyond concatenation

### Key Implementation Details

1. **Text Encoder Loading** (`notebooks/utils.py`):
   ```python
   models = get_sd_models(
       model_id="stabilityai/stable-diffusion-xl-base-1.0",
       dtype=torch.float16,
       device=device,
   )
   # Returns: tokenizer, text_encoder, tokenizer_2, text_encoder_2, unet, vae, is_sdxl=True
   ```

2. **Text Embedding** (`notebooks/utils.py::get_text_embedding`):
   ```python
   prompt_embeds = get_text_embedding(
       prompt,
       tokenizer, text_encoder,
       device=device,
       tokenizer_2=tokenizer_2,
       text_encoder_2=text_encoder_2,
       return_pooled=False,
   )
   # Returns: (B, 77, 2048) for SDXL
   ```

3. **PoE Composition**: Identical to SD 1.4
   ```python
   noise_pred = uncond + guidance_scale * (cond_a - uncond) + guidance_scale * (cond_b - uncond)
   ```

## Scripts

### 1. `trajectory_dynamics_experiment_sdxl.py`
Core functions for trajectory tracking with SDXL.

**Main Functions:**
- `poe_sdxl_with_trajectory_tracking()`: PoE composition with trajectory tracking
- `poe_sdxl_monolithic_with_trajectory_tracking()`: Single "A and B" prompt with tracking

**Usage (test mode):**
```bash
conda activate jaxstack
python scripts/trajectory_dynamics_experiment_sdxl.py \
    --prompt-a "a dog" \
    --prompt-b "a cat" \
    --guidance-scale 7.5 \
    --num-inference-steps 50 \
    --seed 42
```

### 2. `run_taxonomy_qualitative_sdxl.py`
Full taxonomy qualitative experiments for SDXL (adapts `run_taxonomy_qualitative.py`).

**Key Features:**
- Runs all 4 taxonomy groups (or specified subset)
- Generates trajectory 2D projections (PCA or MDS)
- Decodes final images to PNG
- Saves summary JSON with metrics

**Output Structure:**
```
experiments/eccv2026/taxonomy_qualitative_sdxl/
├── group1_cooccurrence/
│   ├── a_butterfly__x__a_flower_meadow/
│   │   ├── summary.json              ← config + d_T metric
│   │   ├── trajectory_manifold.png   ← 2D projection
│   │   ├── decoded_images.png        ← PoE vs Monolithic
│   │   ├── poe.png
│   │   └── monolithic.png
│   └── ...
├── group2_disentangled/
│   └── ...
└── ...
```

**Usage Examples:**

Single group (1 pair):
```bash
conda activate jaxstack
python scripts/run_taxonomy_qualitative_sdxl.py \
    --groups group1_cooccurrence \
    --max-per-group 1
```

All 4 groups:
```bash
python scripts/run_taxonomy_qualitative_sdxl.py
```

Specific pair:
```bash
python scripts/run_taxonomy_qualitative_sdxl.py \
    --pairs group1_cooccurrence/a_butterfly__x__a_flower_meadow
```

Custom SDXL variant (e.g., with LoRA or different checkpoint):
```bash
python scripts/run_taxonomy_qualitative_sdxl.py \
    --model-id "stabilityai/stable-diffusion-xl-base-1.0" \
    --output-dir experiments/eccv2026/taxonomy_qualitative_sdxl_custom
```

### 3. Plotting from Results
Once experiments complete, generate the 2x2 figure:

```bash
python scripts/plot_trajectory_2x2.py \
    --pairs \
        group1_cooccurrence/a_butterfly__x__a_flower_meadow \
        group2_disentangled/a_dog__x__oil_painting_style \
        group3_ood/a_desk_lamp__x__a_glacier \
        group4_collision/a_cat__x__a_dog \
    --out experiments/eccv2026/grid_figure/trajectory_2x2_sdxl.png
```

Or use SDXL-specific input directory:
```bash
# Modify plot_trajectory_2x2.py to point to taxonomy_qualitative_sdxl:
python scripts/plot_trajectory_2x2.py \
    --input-dir experiments/eccv2026/taxonomy_qualitative_sdxl \
    --out experiments/eccv2026/grid_figure/trajectory_2x2_sdxl.png
```

## Implementation Summary

### What's Different from SD 1.4 Code

| Aspect | SD 1.4 | SDXL |
|--------|--------|------|
| Text encoders | 1 (CLIP-L) | 2 (CLIP-L + CLIP-G) |
| Embedding dim | 768 | 2048 (after concat) |
| Text encoding function | N/A | `get_text_embedding()` with `tokenizer_2` + `text_encoder_2` |
| UNet input | `(B, 77, 768)` | `(B, 77, 2048)` |
| Scheduling | Euler / DDIM | Same (Euler / DDIM) |
| Latent size | `(B, 4, 64, 64)` for 512×512 | `(B, 4, 128, 128)` for 1024×1024 |
| PoE formula | Same | **Same** (composition is model-agnostic) |

### Key SDXL Adaptations in New Code

1. **Text Encoding** (`trajectory_dynamics_experiment_sdxl.py::_encode_sdxl`):
   - Passes both tokenizer/encoder pairs to `get_text_embedding()`
   - Returns properly concatenated embeddings

2. **PoE Functions**:
   - Accept `tokenizer_2`, `text_encoder_2` as parameters
   - Call `_encode_sdxl()` for all prompt encodings
   - Rest of PoE logic is identical to SD 1.4

3. **Trajectory Collection**:
   - No changes — uses same `LatentTrajectoryCollector` class
   - Works with any latent shape

4. **Model Loading**:
   - Uses `get_sd_models()` which auto-detects SDXL and loads dual encoders

## Performance Notes

- SDXL is slower than SD 1.4 (~2-3x depending on batch size)
- Memory usage: ~15GB for single-batch inference on A100
- Latent resolution doubled (64→128) but number of channels same
- Consider using `--num-inference-steps 30` for faster iterations

## Reproducing Results

To generate the exact same `trajectory_2x2_sdxl.png` figure:

```bash
# 1. Run full taxonomy with fixed seed
conda activate jaxstack
python scripts/run_taxonomy_qualitative_sdxl.py \
    --seed 42 \
    --num-inference-steps 50 \
    --guidance-scale 7.5

# 2. Plot 2x2 grid from results
python scripts/plot_trajectory_2x2.py \
    --input-dir experiments/eccv2026/taxonomy_qualitative_sdxl \
    --out experiments/eccv2026/grid_figure/trajectory_2x2_sdxl.png
```

## Troubleshooting

**Issue: "No module named 'diffusers'"**
- Solution: `conda activate jaxstack` before running

**Issue: CUDA out of memory**
- Reduce `--num-inference-steps` (e.g., 30 instead of 50)
- Use `dtype=torch.float32` if float16 fails
- Check for competing GPU processes with `nvidia-smi`

**Issue: Text embeddings have wrong shape**
- Ensure you're using an SDXL model ID (contains "xl" in name)
- Check `models["is_sdxl"]` is `True`
- Verify both text encoders are loaded

**Issue: Generated images look strange**
- SDXL default guidance_scale is often lower (4.5–5.5) vs SD 1.4 (7.5)
- Try `--guidance-scale 5.0`
- Ensure seed is fixed for reproducibility

## Next Steps

1. **Validation**: Run on single pair first to check memory/timing
   ```bash
   python scripts/run_taxonomy_qualitative_sdxl.py --groups group1_cooccurrence --max-per-group 1
   ```

2. **Full Suite**: Run all 4 groups
   ```bash
   python scripts/run_taxonomy_qualitative_sdxl.py
   ```

3. **Visualization**: Generate 2x2 figure and compare to SD 1.4 baseline

4. **Analysis**: Use generated JSON summaries to compute aggregate metrics across groups
