# SDXL Trajectory Experiments — Quick Start

## TL;DR

Generate `trajectory_2x2_sdxl.png` in 3 commands:

```bash
conda activate jaxstack

# 1. Run all 4 taxonomy groups (takes ~6-10 hours on single GPU)
python scripts/run_taxonomy_qualitative_sdxl.py

# 2. Generate 2x2 figure from results
python scripts/plot_trajectory_2x2.py \
    --input-dir experiments/eccv2026/taxonomy_qualitative_sdxl \
    --out experiments/eccv2026/grid_figure/trajectory_2x2_sdxl.png

# Result: trajectory_2x2_sdxl.png ready to use!
```

## What You Get

Same output structure as SD 1.4 version, but with SDXL:
```
experiments/eccv2026/taxonomy_qualitative_sdxl/
├── group1_cooccurrence/
│   ├── a_butterfly__x__a_flower_meadow/
│   │   ├── summary.json
│   │   ├── trajectory_manifold.png  ← PoE vs Monolithic trajectories
│   │   ├── decoded_images.png       ← Final generated images
│   │   ├── poe.png                  ← PoE result
│   │   └── monolithic.png           ← Monolithic "A and B" result
│   └── ...
└── grid_figure/trajectory_2x2_sdxl.png  ← Final 2x2 figure
```

## Key Parameters

```bash
# Skip to specific group (faster testing)
python scripts/run_taxonomy_qualitative_sdxl.py --groups group1_cooccurrence --max-per-group 1

# Use custom model
python scripts/run_taxonomy_qualitative_sdxl.py --model-id "your/custom-sdxl-model"

# Adjust quality/speed
python scripts/run_taxonomy_qualitative_sdxl.py --num-inference-steps 30  # Faster
python scripts/run_taxonomy_qualitative_sdxl.py --guidance-scale 5.0     # May look better

# Specific pair only
python scripts/run_taxonomy_qualitative_sdxl.py \
    --pairs group1_cooccurrence/a_butterfly__x__a_flower_meadow
```

## What Changed from SD 1.4

| Aspect | SD 1.4 | SDXL |
|--------|--------|------|
| Text encoder count | 1 | **2** (CLIP-L + CLIP-G) |
| Embedding dimension | 768 | **2048** |
| Latent resolution | 64×64 | **128×128** |
| PoE formula | `uncond + g*(a-uncond) + g*(b-uncond)` | **Same** ✓ |
| Speed | Baseline | ~2-3x slower |
| Memory | Baseline | Similar (~15GB single-batch) |

## Files Added

```
scripts/
├── trajectory_dynamics_experiment_sdxl.py    ← Core functions
├── run_taxonomy_qualitative_sdxl.py          ← Full experiment
├── SDXL_TRAJECTORY_GUIDE.md                  ← Detailed guide
└── ...

SDXL_IMPLEMENTATION_NOTES.md                   ← Technical details
SDXL_QUICK_START.md                            ← This file
```

## Troubleshooting

**Memory error?**
→ Reduce steps: `--num-inference-steps 30`

**Text encoder shape error?**
→ Check model_id contains "xl": `stabilityai/stable-diffusion-xl-base-1.0`

**ImportError for diffusers?**
→ Activate correct env: `conda activate jaxstack`

**Results look different?**
→ Try lower guidance: `--guidance-scale 5.0`

## Next: Detailed Docs

- **Full guide**: See `scripts/SDXL_TRAJECTORY_GUIDE.md`
- **Implementation details**: See `SDXL_IMPLEMENTATION_NOTES.md`
- **SD 1.4 original**: `scripts/trajectory_dynamics_experiment.py`

## One Pair Test (5 min)

Validate setup with a single pair:
```bash
conda activate jaxstack
python scripts/trajectory_dynamics_experiment_sdxl.py \
    --prompt-a "a butterfly" \
    --prompt-b "a flower" \
    --num-inference-steps 20 \
    --seed 42
```

If this runs without errors, full taxonomy will work!

---

**Status**: ✅ Ready to run | **Time estimate**: 6-10 hrs (all groups) | **GPU memory**: ~15GB
