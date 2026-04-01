"""
Phase 1 Taxonomy Qualitative Decoded Images — SD 3.5 Medium
============================================================
Generates the four conditions (solo_a, solo_b, monolithic, poe) for the
multi-model taxonomy grid using Stable Diffusion 3.5 Medium.

PoE (logical composition) for flow matching uses score addition over the
velocity field — the same formula adapted to SD 3.5's Transformer and
FlowMatchEulerDiscreteScheduler:

    v_PoE = v_uncond + scale*(v_A - v_uncond) + scale*(v_B - v_uncond)

Model loading and text encoding logic is adapted from
    compositions/super-diffusion/scripts/compose_and_sd3.py

Shared-noise protocol: all four conditions within a pair start from the
same x_T (same generator seed), so differences are attributable to the
operator alone.

Taxonomy pairs (3 per group, most visually distinct):
    Group 1 — co-occurring concepts
    Group 2 — feature-space disentangled (style/content)
    Group 3 — OOD / feature overlap
    Group 4 — adversarial collision

Usage
-----
    conda activate diffgen
    python scripts/run_taxonomy_qualitative_sd3.py
    python scripts/run_taxonomy_qualitative_sd3.py --groups group4_collision
    python scripts/run_taxonomy_qualitative_sd3.py --seed 43 --steps 30

Output structure
----------------
experiments/eccv2026/taxonomy_qualitative_multimodel/sd3/
    group1_cooccurrence/
        a_camel__x__a_desert_landscape/
            solo_a.png  solo_b.png  monolithic.png  poe.png
        ...
    group2_disentangled/
    group3_ood/
    group4_collision/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"


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


def _load_models(model_id: str, device: torch.device, dtype: torch.dtype, t5_seq: int):
    """Load SD 3.5 components (Transformer + triple text encoders + VAE + scheduler)."""
    from transformers import (
        CLIPTextModelWithProjection, CLIPTokenizer,
        T5EncoderModel, T5TokenizerFast,
    )
    from diffusers import AutoencoderKL, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler

    print(f"Loading {model_id} ...")
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype, use_safetensors=True,
    ).to(device)
    transformer = SD3Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=dtype, use_safetensors=True,
    ).to(device)

    tokenizer    = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype, use_safetensors=True,
    ).to(device)

    tokenizer_2    = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=dtype, use_safetensors=True,
    ).to(device)

    tokenizer_3    = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_3")
    text_encoder_3 = T5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder_3", torch_dtype=dtype, use_safetensors=True,
    ).to(device)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler",
    )

    vae.eval(); transformer.eval()
    text_encoder.eval(); text_encoder_2.eval(); text_encoder_3.eval()
    print("Models loaded.\n")

    return (
        tokenizer, tokenizer_2, tokenizer_3,
        text_encoder, text_encoder_2, text_encoder_3,
        transformer, vae, scheduler,
    )


@torch.no_grad()
def _encode(
    prompt: str,
    tokenizer, tokenizer_2, tokenizer_3,
    text_encoder, text_encoder_2, text_encoder_3,
    device: torch.device, dtype: torch.dtype,
    t5_seq: int,
):
    """Encode a prompt with SD 3.5's triple text encoders.

    Returns:
        prompt_embeds: (1, 333, 4096)  — CLIP-L 77 + CLIP-G 77 padded, then cat with T5 256
        pooled_embeds: (1, 2048)       — CLIP-L pooled ⊕ CLIP-G pooled
    """
    if isinstance(prompt, str):
        prompt = [prompt]

    # CLIP-L
    inp1 = tokenizer(
        prompt, padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    out1  = text_encoder(inp1.input_ids.to(device), output_hidden_states=True)
    emb1  = out1.hidden_states[-2]   # (1, 77, 768)
    pool1 = out1[0]                  # (1, 768)

    # CLIP-G
    inp2 = tokenizer_2(
        prompt, padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True, return_tensors="pt",
    )
    out2  = text_encoder_2(inp2.input_ids.to(device), output_hidden_states=True)
    emb2  = out2.hidden_states[-2]   # (1, 77, 1280)
    pool2 = out2[0]                  # (1, 1280)

    # T5-XXL
    inp3   = tokenizer_3(
        prompt, padding="max_length",
        max_length=t5_seq,
        truncation=True, return_tensors="pt",
    )
    t5_emb = text_encoder_3(inp3.input_ids.to(device))[0]  # (1, t5_seq, 4096)

    # Pack CLIP tokens: concat on feature dim, pad to T5 hidden size, then cat on seq dim
    clip_emb = torch.cat([emb1, emb2], dim=-1)                              # (1, 77, 2048)
    clip_emb = F.pad(clip_emb, (0, t5_emb.shape[-1] - clip_emb.shape[-1])) # (1, 77, 4096)
    prompt_embeds = torch.cat([clip_emb, t5_emb], dim=-2)                   # (1, 77+t5_seq, 4096)

    pooled = torch.cat([pool1, pool2], dim=-1)  # (1, 2048)
    return prompt_embeds.to(dtype), pooled.to(dtype)


@torch.no_grad()
def _decode(latents, vae, dtype):
    """VAE decode latents → PIL Image (SD 3.5 shift_factor corrected)."""
    latents = latents.to(dtype=vae.dtype)
    shift_factor = getattr(vae.config, "shift_factor", 0.0)
    image = vae.decode(
        latents / vae.config.scaling_factor + shift_factor, return_dict=False,
    )[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    image = (image * 255).round().astype(np.uint8)
    return Image.fromarray(image)


@torch.no_grad()
def _get_vel(latents, t, ctx, pooled, transformer, device):
    """Single transformer velocity prediction (no JVP divergence)."""
    return transformer(
        hidden_states=latents,
        timestep=t.expand(latents.shape[0]).to(device),
        encoder_hidden_states=ctx,
        pooled_projections=pooled,
        return_dict=False,
    )[0]


# ---------------------------------------------------------------------------
# Single-pair generation
# ---------------------------------------------------------------------------

def run_pair(
    concept_a: str,
    concept_b: str,
    out_dir: Path,
    tokenizer, tokenizer_2, tokenizer_3,
    text_encoder, text_encoder_2, text_encoder_3,
    transformer, vae, scheduler,
    scale: float,
    steps: int,
    seed: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    t5_seq: int,
):
    """Generate all 4 conditions for one pair using the shared-noise protocol."""
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_kwargs = dict(
        tokenizer=tokenizer, tokenizer_2=tokenizer_2, tokenizer_3=tokenizer_3,
        text_encoder=text_encoder, text_encoder_2=text_encoder_2, text_encoder_3=text_encoder_3,
        device=device, dtype=dtype, t5_seq=t5_seq,
    )

    emb_uncond, pool_uncond = _encode("", **enc_kwargs)
    emb_a,      pool_a      = _encode(concept_a, **enc_kwargs)
    emb_b,      pool_b      = _encode(concept_b, **enc_kwargs)
    emb_mono,   pool_mono   = _encode(f"{concept_a} and {concept_b}", **enc_kwargs)

    scheduler.set_timesteps(steps)

    for cond in CONDITIONS:
        # Shared noise: reseed identically for each condition
        if device.type == "cuda":
            generator = torch.cuda.manual_seed(seed)
        else:
            generator = torch.manual_seed(seed)

        # SD 3.5 uses 16-channel latents
        latents = torch.randn(
            (1, 16, height // 8, width // 8),
            generator=generator, device=device, dtype=dtype,
        )

        if cond == "solo_a":
            ctx, pooled = emb_a, pool_a
        elif cond == "solo_b":
            ctx, pooled = emb_b, pool_b
        elif cond == "monolithic":
            ctx, pooled = emb_mono, pool_mono
        else:
            ctx, pooled = None, None  # PoE handled in loop

        for i, t in enumerate(scheduler.timesteps):
            dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]

            if cond == "poe":
                v_uncond = _get_vel(latents, t, emb_uncond, pool_uncond, transformer, device)
                v_a      = _get_vel(latents, t, emb_a,      pool_a,      transformer, device)
                v_b      = _get_vel(latents, t, emb_b,      pool_b,      transformer, device)
                # PoE velocity: sum of CFG-guided contributions from each concept
                vel = v_uncond + scale * (v_a - v_uncond) + scale * (v_b - v_uncond)
            else:
                # Standard CFG
                v_uncond = _get_vel(latents, t, emb_uncond, pool_uncond, transformer, device)
                v_cond   = _get_vel(latents, t, ctx,        pooled,      transformer, device)
                vel = v_uncond + scale * (v_cond - v_uncond)

            latents = latents + dsigma * vel

        img = _decode(latents, vae, dtype)
        img.save(out_dir / f"{cond}.png")

    print(f"    [{concept_a}]  ×  [{concept_b}]  →  {out_dir.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 taxonomy qualitative images — SD 3.5 Medium."
    )
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--steps",  type=int,   default=50)
    parser.add_argument("--scale",  type=float, default=4.5)
    parser.add_argument("--model",  type=str,   default=MODEL_ID)
    parser.add_argument("--height", type=int,   default=1024)
    parser.add_argument("--width",  type=int,   default=1024)
    parser.add_argument("--t5_seq", type=int,   default=256,
                        help="T5 max sequence length (default: 256)")
    parser.add_argument("--out",    type=str,   default="")
    parser.add_argument(
        "--groups", nargs="+",
        choices=sorted(TAXONOMY_GROUPS.keys()),
        default=list(TAXONOMY_GROUPS.keys()),
    )
    args = parser.parse_args()

    base_out = (
        Path(args.out) if args.out
        else PROJECT_ROOT / "experiments" / "eccv2026" / "taxonomy_qualitative_multimodel" / "sd3"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16  # SD 3.5 bfloat16

    total_pairs = sum(len(TAXONOMY_GROUPS[g]) for g in args.groups)
    print(f"\nPhase 1 taxonomy qualitative — SD 3.5 Medium")
    print(f"  Model  : {args.model}")
    print(f"  Seed   : {args.seed}   Steps: {args.steps}   Scale: {args.scale}")
    print(f"  Groups : {', '.join(args.groups)}")
    print(f"  Pairs  : {total_pairs}  ×  4 conditions = {total_pairs * 4} images")
    print(f"  Output : {base_out}\n")

    (
        tokenizer, tokenizer_2, tokenizer_3,
        text_encoder, text_encoder_2, text_encoder_3,
        transformer, vae, scheduler,
    ) = _load_models(args.model, device, dtype, args.t5_seq)

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
                tokenizer, tokenizer_2, tokenizer_3,
                text_encoder, text_encoder_2, text_encoder_3,
                transformer, vae, scheduler,
                scale=args.scale, steps=args.steps, seed=args.seed,
                height=args.height, width=args.width,
                device=device, dtype=dtype, t5_seq=args.t5_seq,
            )
        print()

    print("All groups complete.")
    print(f"Output: {base_out}")


if __name__ == "__main__":
    main()
