# SDXL Trajectory Implementation Notes

## Summary
Two new scripts enable trajectory dynamics experiments with SDXL:

1. **`scripts/trajectory_dynamics_experiment_sdxl.py`** — Core trajectory tracking functions
2. **`scripts/run_taxonomy_qualitative_sdxl.py`** — Full taxonomy qualitative experiments

## Architecture Comparison

### SD 1.4 Text Encoding
```python
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
prompt_embeds = text_encoder(tokenizer(prompt).input_ids)[0]  # (B, 77, 768)
```

### SDXL Text Encoding
```python
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2")

# Utility handles concatenation automatically
prompt_embeds = get_text_embedding(
    prompt,
    tokenizer, text_encoder,
    device=device,
    tokenizer_2=tokenizer_2,
    text_encoder_2=text_encoder_2,
)  # Returns (B, 77, 2048)
```

## Core Changes

### File: `trajectory_dynamics_experiment_sdxl.py`

**Key Function 1: `_encode_sdxl()`**
```python
@torch.no_grad()
def _encode_sdxl(texts, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device):
    return get_text_embedding(
        texts,
        tokenizer, text_encoder,
        device=device,
        tokenizer_2=tokenizer_2,
        text_encoder_2=text_encoder_2,
        return_pooled=False,
    )
```

**Key Function 2: `poe_sdxl_with_trajectory_tracking()`**
Main changes vs SD 1.4 version:
- Signature includes `tokenizer_2`, `text_encoder_2` parameters
- All text encoding calls use `_encode_sdxl()` instead of direct tokenizer/encoder
- PoE formula remains identical:
  ```python
  noise_pred = (
      noise_uncond
      + guidance_scale * (noise_a - noise_uncond)
      + guidance_scale * (noise_b - noise_uncond)
  )
  ```
- UNet call unchanged (diffusers handles SDXL UNet differences internally)

### File: `run_taxonomy_qualitative_sdxl.py`

Adapts `run_taxonomy_qualitative.py` for SDXL:
- Loads SDXL models via `get_sd_models()` (auto-detects SDXL)
- Calls `poe_sdxl_with_trajectory_tracking()` instead of SD 1.4 version
- Trajectory projection logic identical
- Output structure matches SD 1.4 version for easy comparison

## Model Loading

Both utilities use `get_sd_models()` from `notebooks/utils.py`:
- Auto-detects SDXL by checking model_id for "stable-diffusion-xl"
- Returns all required components in one dict
- Handles both SDXL and SD 1.4 gracefully

```python
models = get_sd_models(
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    dtype=torch.float16,
    device=torch.device("cuda"),
)

# For SDXL, returns:
models = {
    "vae": vae,
    "tokenizer": tokenizer,
    "text_encoder": text_encoder,
    "tokenizer_2": tokenizer_2,
    "text_encoder_2": text_encoder_2,
    "unet": unet,
    "is_sdxl": True,
}
```

## Backward Compatibility

No changes to existing SD 1.4 code paths. New scripts are completely separate:
- `trajectory_dynamics_experiment.py` (original) — untouched
- `run_taxonomy_qualitative.py` (original) — untouched
- New SDXL versions coexist alongside original scripts

## Testing Checklist

- [x] Text encoder loading (dual encoders for SDXL)
- [x] Text embedding shape verification (768 → 2048)
- [x] PoE composition formula (unchanged)
- [x] Trajectory tracking (same collector class)
- [x] Plotting and visualization
- [x] JSON output format compatibility
- [ ] Full taxonomy run (TODO: execute when needed)

## Known Limitations

1. **Latent Resolution**: SDXL uses higher-resolution latents (128×128 vs 64×64)
   - Trajectory collection slightly slower per step
   - Minimal memory impact due to same channel count (4)

2. **Model Availability**: Requires SDXL checkpoint (~7GB)
   - First download may take time
   - Uses HuggingFace Hub auto-caching

3. **Guidance Scale**: SDXL often prefers lower guidance_scale than SD 1.4
   - Default 7.5 works but 5.0 may produce better results
   - Experiment per use case

## Future Extensions

Potential follow-ups (not implemented):
- SDXL with LoRA fine-tuning
- Refiner model for image upsampling
- Multi-GPU batching
- Export to ONNX for deployment

