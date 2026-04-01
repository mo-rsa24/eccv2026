"""
Phase 1 Taxonomy Qualitative Decoded Images — SDXL
===================================================
Generates the four conditions (solo_a, solo_b, monolithic, poe) for the
multi-model taxonomy grid using Stable Diffusion XL Base 1.0.

PoE (logical composition) is implemented as direct score addition in the
denoising loop — the same product-of-experts formula used by ComposableStableDiffusion
for SD 1.4, adapted to SDXL's dual text-encoder and added-conditioning API:

    ε_PoE = ε_uncond + scale*(ε_A - ε_uncond) + scale*(ε_B - ε_uncond)

Shared-noise protocol: all four conditions within a pair start from the same
x_T (same generator seed), so differences are attributable to the operator alone.

Taxonomy pairs (3 per group, as selected for the multi-model figure):
    Group 1 — co-occurring concepts
    Group 2 — feature-space disentangled (style/content)
    Group 3 — OOD / feature overlap
    Group 4 — adversarial collision

Usage
-----
    conda activate diffgen
    python scripts/run_taxonomy_qualitative_sdxl.py
    python scripts/run_taxonomy_qualitative_sdxl.py --groups group4_collision
    python scripts/run_taxonomy_qualitative_sdxl.py --seed 43 --steps 30

Output structure
----------------
experiments/eccv2026/taxonomy_qualitative_multimodel/sdxl/
    group1_cooccurrence/
        a_camel__x__a_desert_landscape/
            solo_a.png  solo_b.png  monolithic.png  poe.png
        ...
    group2_disentangled/
    group3_ood/
    group4_collision/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Taxonomy pairs — 3 per group (most visually distinct selection)
# ---------------------------------------------------------------------------
TAXONOMY_GROUPS = {
    "group1_cooccurrence": [
        ("a camel",             "a desert landscape"),
        ("a butterfly",         "a flower meadow"),
        ("a lion",              "a savanna at sunset"),
    ],
    "group2_disentangled": [
        ("a dog",               "oil painting style"),
        ("a lighthouse",        "watercolour style"),
        ("a bicycle",           "sketch style"),
    ],
    "group3_ood": [
        ("a bathtub",           "a streetlamp"),
        ("a black grand piano", "a white vase"),
        ("a typewriter",        "a cactus"),
    ],
    "group4_collision": [
        ("a cat",               "a dog"),
        ("a cat",               "an owl"),
        ("a cat",               "a bear"),
    ],
}

CONDITIONS = ["solo_a", "solo_b", "monolithic", "poe"]

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace(",", "")
        .replace("'", "")
        .replace("/", "")
    )


def _load_models(model_id: str, device: torch.device, dtype: torch.dtype):
    """Load SDXL components individually (avoids pipeline API version issues)."""
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler

    print(f"Loading {model_id} ...")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype,
    ).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=dtype,
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=dtype,
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype,
    ).to(device)
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    text_encoder.eval()
    text_encoder_2.eval()
    unet.eval()
    vae.eval()
    print("Models loaded.\n")
    return tokenizer, tokenizer_2, text_encoder, text_encoder_2, unet, vae, scheduler


@torch.no_grad()
def _encode(
    prompt: str,
    tokenizer, tokenizer_2,
    text_encoder, text_encoder_2,
    device: torch.device,
    dtype: torch.dtype,
):
    """Encode a single prompt with SDXL's dual text encoders.

    Returns:
        combined_emb: (1, 77, 2048)  — penultimate hidden states of both encoders concatenated
        pooled_emb:   (1, 1280)      — pooled output from CLIP-G (text_encoder_2)
    """
    # CLIP-L (text_encoder): penultimate hidden state shape (1, 77, 768)
    tok1 = tokenizer(
        [prompt], padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    out1 = text_encoder(
        tok1.input_ids.to(device), output_hidden_states=True,
    )
    emb1 = out1.hidden_states[-2]  # (1, 77, 768)

    # CLIP-G (text_encoder_2): penultimate hidden state (1, 77, 1280) + pooled output (1, 1280)
    tok2 = tokenizer_2(
        [prompt], padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True, return_tensors="pt",
    )
    out2 = text_encoder_2(
        tok2.input_ids.to(device), output_hidden_states=True,
    )
    emb2 = out2.hidden_states[-2]  # (1, 77, 1280)
    pooled = out2[0]               # (1, 1280)

    # Concatenate on feature dim → (1, 77, 2048)
    combined = torch.cat([emb1, emb2], dim=-1).to(dtype)
    pooled = pooled.to(dtype)
    return combined, pooled


@torch.no_grad()
def _decode(latents, vae, dtype):
    """VAE decode latents → PIL Image."""
    latents = latents.to(dtype=vae.dtype)
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    image = (image * 255).round().astype(np.uint8)
    return Image.fromarray(image)


def _make_time_ids(height: int, width: int, device: torch.device, dtype: torch.dtype):
    """Build SDXL added-time-ids conditioning tensor.

    Format: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
    """
    return torch.tensor(
        [[height, width, 0, 0, height, width]],
        device=device, dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Single-pair generation
# ---------------------------------------------------------------------------

def run_pair(
    concept_a: str,
    concept_b: str,
    out_dir: Path,
    tokenizer, tokenizer_2,
    text_encoder, text_encoder_2,
    unet, vae, scheduler,
    scale: float,
    steps: int,
    seed: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """Generate all 4 conditions for one concept pair, shared-noise protocol."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-encode all prompts
    emb_uncond, pool_uncond = _encode(
        "", tokenizer, tokenizer_2, text_encoder, text_encoder_2, device, dtype
    )
    emb_a, pool_a = _encode(
        concept_a, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device, dtype
    )
    emb_b, pool_b = _encode(
        concept_b, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device, dtype
    )
    emb_mono, pool_mono = _encode(
        f"{concept_a} and {concept_b}",
        tokenizer, tokenizer_2, text_encoder, text_encoder_2, device, dtype
    )

    time_ids = _make_time_ids(height, width, device, dtype)

    scheduler.set_timesteps(steps)

    for cond in CONDITIONS:
        # Shared noise: reseed generator identically for each condition
        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(
            (1, unet.config.in_channels, height // 8, width // 8),
            generator=generator, device=device, dtype=dtype,
        )
        latents = latents * scheduler.init_noise_sigma

        # Pick the conditioning embeddings for this condition
        if cond == "solo_a":
            emb_cond, pool_cond = emb_a, pool_a
        elif cond == "solo_b":
            emb_cond, pool_cond = emb_b, pool_b
        elif cond == "monolithic":
            emb_cond, pool_cond = emb_mono, pool_mono
        else:  # poe
            emb_cond = None  # handled in loop
            pool_cond = None

        for t in scheduler.timesteps:
            t_tensor = t.unsqueeze(0).to(device)
            # Apply the scheduler's input scaling (÷√(σ²+1) for EulerDiscreteScheduler).
            # The original ComposableStableDiffusionPipeline calls this on every forward pass.
            scaled_latents = scheduler.scale_model_input(latents, t)

            if cond == "poe":
                # PoE: three forward passes, score addition.
                # Matches _predict_composed_noise in the original pipeline exactly:
                #   ε_PoE = ε_uncond + scale*(ε_A - ε_uncond) + scale*(ε_B - ε_uncond)
                with torch.no_grad():
                    eps_uncond = unet(
                        scaled_latents, t_tensor,
                        encoder_hidden_states=emb_uncond,
                        added_cond_kwargs={"text_embeds": pool_uncond, "time_ids": time_ids},
                        return_dict=False,
                    )[0]
                    eps_a = unet(
                        scaled_latents, t_tensor,
                        encoder_hidden_states=emb_a,
                        added_cond_kwargs={"text_embeds": pool_a, "time_ids": time_ids},
                        return_dict=False,
                    )[0]
                    eps_b = unet(
                        scaled_latents, t_tensor,
                        encoder_hidden_states=emb_b,
                        added_cond_kwargs={"text_embeds": pool_b, "time_ids": time_ids},
                        return_dict=False,
                    )[0]
                noise_pred = (
                    eps_uncond
                    + scale * (eps_a - eps_uncond)
                    + scale * (eps_b - eps_uncond)
                )
            else:
                # Standard CFG: two forward passes batched together for efficiency.
                scaled_cat = torch.cat([scaled_latents, scaled_latents])
                emb_cat = torch.cat([emb_uncond, emb_cond])
                pool_cat = torch.cat([pool_uncond, pool_cond])
                t_cat = t_tensor.repeat(2)
                with torch.no_grad():
                    eps_out = unet(
                        scaled_cat, t_cat,
                        encoder_hidden_states=emb_cat,
                        added_cond_kwargs={
                            "text_embeds": pool_cat,
                            "time_ids": time_ids.repeat(2, 1),
                        },
                        return_dict=False,
                    )[0]
                eps_uncond_i, eps_cond_i = eps_out.chunk(2)
                noise_pred = eps_uncond_i + scale * (eps_cond_i - eps_uncond_i)

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        img = _decode(latents, vae, dtype)
        img.save(out_dir / f"{cond}.png")

    print(f"    [{concept_a}]  ×  [{concept_b}]  →  {out_dir.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 taxonomy qualitative images — SDXL."
    )
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--steps",  type=int,   default=50)
    parser.add_argument("--scale",  type=float, default=7.5)
    parser.add_argument("--model",  type=str,   default=MODEL_ID)
    parser.add_argument("--height", type=int,   default=1024)
    parser.add_argument("--width",  type=int,   default=1024)
    parser.add_argument("--out",    type=str,   default="")
    parser.add_argument(
        "--groups", nargs="+",
        choices=sorted(TAXONOMY_GROUPS.keys()),
        default=list(TAXONOMY_GROUPS.keys()),
    )
    args = parser.parse_args()

    base_out = (
        Path(args.out) if args.out
        else PROJECT_ROOT / "experiments" / "eccv2026" / "taxonomy_qualitative_multimodel" / "sdxl"
    )

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    dtype = torch.float16  # SDXL fp16

    total_pairs = sum(len(TAXONOMY_GROUPS[g]) for g in args.groups)
    print(f"\nPhase 1 taxonomy qualitative — SDXL")
    print(f"  Model  : {args.model}")
    print(f"  Seed   : {args.seed}   Steps: {args.steps}   Scale: {args.scale}")
    print(f"  Groups : {', '.join(args.groups)}")
    print(f"  Pairs  : {total_pairs}  ×  4 conditions = {total_pairs * 4} images")
    print(f"  Output : {base_out}\n")

    tokenizer, tokenizer_2, text_encoder, text_encoder_2, unet, vae, scheduler = \
        _load_models(args.model, device, dtype)

    for group_name in args.groups:
        pairs = TAXONOMY_GROUPS[group_name]
        group_out = base_out / group_name
        print(f"{'='*60}")
        print(f"  {group_name}  ({len(pairs)} pairs)")
        print(f"{'='*60}")

        for concept_a, concept_b in pairs:
            pair_slug = f"{_slugify(concept_a)}__x__{_slugify(concept_b)}"
            pair_out = group_out / pair_slug
            run_pair(
                concept_a, concept_b, pair_out,
                tokenizer, tokenizer_2, text_encoder, text_encoder_2, unet, vae, scheduler,
                scale=args.scale, steps=args.steps, seed=args.seed,
                height=args.height, width=args.width,
                device=device, dtype=dtype,
            )
        print()

    print("All groups complete.")
    print(f"Output: {base_out}")


if __name__ == "__main__":
    main()
