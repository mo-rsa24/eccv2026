#!/usr/bin/env python3
"""
Generate a 4-panel grid for spatial concept pairs showing:
  Col 0 : Concept A alone
  Col 1 : Concept B alone
  Col 2 : Semantic composition  (single monolithic prompt "A and B")
  Col 3 : PoE composition       (Product-of-Experts via composable score summing,
                                 replicating the method from
                                 "Compositional Visual Generation with Composable
                                 Diffusion Models" – Liu et al., ECCV 2022)

SD v1.x  PoE formula (noise prediction):
    eps_composed = eps_uncond
                 + w_A * (eps_A - eps_uncond)
                 + w_B * (eps_B - eps_uncond)

SD 3.5   PoE formula (velocity prediction, flow matching):
    v_composed = v_uncond
               + w_A * (v_A - v_uncond)
               + w_B * (v_B - v_uncond)

The premise: CLIP is trained with a "A photo of" lead phrase, so purely spatial
descriptions like "A cat on the left of the photo" may confuse composition when
baked into a monolithic (semantic) prompt, but separate PoE conditioning can
respect each concept independently.

Usage:
    # SD v1.4 (default)
    python scripts/generate_spatial_composition_grid.py

    # SD 3.5 Medium
    python scripts/generate_spatial_composition_grid.py --sd35

    # SD 3.5 with custom model
    python scripts/generate_spatial_composition_grid.py \\
        --sd35 --model stabilityai/stable-diffusion-3.5-medium

    # Quick test: first 2 pairs only
    python scripts/generate_spatial_composition_grid.py --pairs 2
"""

import argparse
import gc
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Concept pair definitions
# Format: (concept_A, concept_B)
#   - concept_A / concept_B  → solo images and PoE conditioning signals
#   - semantic prompt         → f"{concept_A} and {concept_B}"
#
# All pairs use the "of the photo" phrasing to match the CLIP training
# lead-phrase ("A photo of …") hypothesis.  Directions covered:
#   left/right, top/bottom, top-left/bottom-right, top-right/bottom-left.
# ---------------------------------------------------------------------------

# Contamination risk is annotated per pair.  Higher risk = the *conjunction*
# of these two concepts in this spatial layout is more likely to have appeared
# in LAION captions during SD training.  Varying the risk lets us test whether
# "semantic composition works" is explained by memorisation (high-risk pairs)
# or genuine compositional understanding (low-risk pairs).
#
# Risk levels: HIGH · MEDIUM-HIGH · MEDIUM · LOW · VERY LOW
#
SPATIAL_CONCEPT_PAIRS: List[Tuple[str, str]] = [
    # ── HIGH contamination risk ─────────────────────────────────────────────
    # Sky-above-ground is the dominant landscape composition in web imagery.
    ("A blue sky on the top of the photo",
     "A green meadow on the bottom of the photo"),
    # Bird-above-water is classic nature photography; co-occurrence is frequent.
    ("A bird on the top of the photo",
     "A fish on the bottom of the photo"),

    # ── MEDIUM-HIGH contamination risk ──────────────────────────────────────
    # Rainbow + body of water are co-described in weather/landscape captions.
    ("A cat",
     "A dog"),
    # Cat + dog together are ubiquitous in pet photography captions.
    ("A cat on the left of the photo",
     "A dog on the right of the photo"),

    # ── MEDIUM contamination risk ────────────────────────────────────────────
    # Sun + shadow: natural pairing but rarely frame-relative in captions.
    ("A cat on the left",
     "A dog on the right"),
    # Apple + banana: common still-life but left/right placement is unusual.
    ("A butterfly",
     "A flower meadow"),

    # ── LOW contamination risk ───────────────────────────────────────────────
    # Bicycle + motorbike: similar category, but spatial co-description rare.
    ("A bicycle on the left of the photo",
     "A motorbike on the right of the photo"),
    # Moon + wolf: imagery is common but diagonal framing is rarely in alt-text.
    ("A moon in the top-right of the photo",
     "A wolf in the bottom-left of the photo"),

    # ── VERY LOW contamination risk ──────────────────────────────────────────
    # Guitar + piano: unusual to describe two instruments with frame coords.
    ("A guitar on the left of the photo",
     "A piano on the right of the photo"),
    # Candle + book: the objects co-occur but diagonal placement is not natural.
    ("A candle in the top-left of the photo",
     "A book in the bottom-right of the photo"),
    # Campfire + telescope: semantically unrelated, diagonal, rarely co-captioned.
    ("A campfire in the bottom-left of the photo",
     "A telescope in the top-right of the photo"),
    # Kite + anchor: conceptually opposite (air vs sea), unusual combination.
    ("A kite in the top-left of the photo",
     "An anchor in the bottom-right of the photo"),
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate spatial composition grids "
                    "(concept A | concept B | semantic | PoE)"
    )
    p.add_argument(
        "--sd35",
        action="store_true",
        help="Use SD 3.5 (flow matching) instead of SD v1.x.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "HuggingFace model ID. "
            "Defaults to 'CompVis/stable-diffusion-v1-4' for SD v1.x "
            "or 'stabilityai/stable-diffusion-3.5-medium' for --sd35."
        ),
    )
    p.add_argument("--seed",  type=int, default=42,  help="Random seed.")
    p.add_argument("--steps", type=int, default=None,
                   help="Denoising steps. Defaults: 50 (SD v1.x), 28 (SD 3.5).")
    p.add_argument(
        "--guidance_scale", type=float, default=None,
        help="CFG scale. Defaults: 7.5 (SD v1.x), 4.5 (SD 3.5).",
    )
    p.add_argument("--height", type=int, default=None,
                   help="Image height. Defaults: 512 (SD v1.x), 512 (SD 3.5).")
    p.add_argument("--width",  type=int, default=None,
                   help="Image width.  Defaults: 512 (SD v1.x), 512 (SD 3.5).")
    p.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "spatial_composition_grids",
        help="Directory where grids are saved.",
    )
    p.add_argument(
        "--pairs",
        type=int,
        default=None,
        help="Limit to the first N concept pairs (default: all 12).",
    )
    p.add_argument(
        "--select",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Run only specific pairs by 1-based index, e.g. --select 4 "
            "for cat/dog. Overrides --pairs."
        ),
    )
    return p.parse_args()


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in model/step/guidance/size defaults based on --sd35 flag."""
    if args.sd35:
        args.model    = args.model    or "stabilityai/stable-diffusion-3.5-medium"
        args.steps    = args.steps    or 28
        args.guidance_scale = args.guidance_scale or 4.5
        args.height   = args.height   or 512
        args.width    = args.width    or 512
    else:
        args.model    = args.model    or "CompVis/stable-diffusion-v1-4"
        args.steps    = args.steps    or 50
        args.guidance_scale = args.guidance_scale or 7.5
        args.height   = args.height   or 512
        args.width    = args.width    or 512
    return args


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------
def make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        return torch.Generator(device=device).manual_seed(seed)
    except Exception:
        return torch.Generator().manual_seed(seed)


# ===========================================================================
# SD v1.x  (DDIM, UNet, noise prediction)
# ===========================================================================

def load_pipeline_sd1(model_id: str, device: torch.device, dtype: torch.dtype):
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    return pipe.to(device)


@torch.no_grad()
def generate_standard_sd1(pipe, prompt, seed, steps, guidance, height, width, device):
    gen = make_generator(device, seed)
    return pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width,
        generator=gen,
    ).images[0]


@torch.no_grad()
def generate_poe_sd1(pipe, concept_a, concept_b, seed, steps, guidance, height, width, device):
    """
    PoE for SD v1.x:
        eps_composed = eps_uncond + w*(eps_A - eps_uncond) + w*(eps_B - eps_uncond)
    """
    tokenizer    = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet         = pipe.unet
    scheduler    = pipe.scheduler
    vae          = pipe.vae

    def encode(text):
        tok = tokenizer(
            text, padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        return text_encoder(tok.input_ids.to(device))[0]

    emb_uncond = encode("")
    emb_a      = encode(concept_a)
    emb_b      = encode(concept_b)

    gen = make_generator(device, seed)
    latents = torch.randn(
        (1, unet.config.in_channels, height // 8, width // 8),
        generator=gen, device=device, dtype=emb_a.dtype,
    ) * scheduler.init_noise_sigma

    scheduler.set_timesteps(steps, device=device)

    for t in scheduler.timesteps:
        lat = scheduler.scale_model_input(latents, t)
        eps_uncond = unet(lat, t, encoder_hidden_states=emb_uncond).sample
        eps_a      = unet(lat, t, encoder_hidden_states=emb_a).sample
        eps_b      = unet(lat, t, encoder_hidden_states=emb_b).sample
        eps = eps_uncond + guidance * (eps_a - eps_uncond) + guidance * (eps_b - eps_uncond)
        latents = scheduler.step(eps, t, latents).prev_sample

    latents = latents / vae.config.scaling_factor
    img_t = vae.decode(latents).sample
    img_t = (img_t / 2 + 0.5).clamp(0, 1)
    img_np = img_t.cpu().float().permute(0, 2, 3, 1).numpy()[0]
    return Image.fromarray((img_np * 255).astype("uint8"))


# ===========================================================================
# SD 3.5  (FlowMatch Euler, Transformer, velocity prediction)
# ===========================================================================

def load_pipeline_sd35(model_id: str, device: torch.device, dtype: torch.dtype):
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
    return pipe.to(device)


def encode_sd35(pipe, text: str, device: torch.device):
    """
    Encode a single prompt with all three SD 3.5 text encoders.
    Returns (prompt_embeds [1, seq, dim], pooled_prompt_embeds [1, pooled_dim]).
    """
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=text,
        prompt_2=text,
        prompt_3=text,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,  # we handle CFG manually
    )
    return prompt_embeds, pooled_prompt_embeds


@torch.no_grad()
def generate_standard_sd35(pipe, prompt, seed, steps, guidance, height, width, device):
    gen = make_generator(device, seed)
    return pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width,
        generator=gen,
    ).images[0]


@torch.no_grad()
def generate_poe_sd35(pipe, concept_a, concept_b, seed, steps, guidance, height, width, device):
    """
    PoE for SD 3.5 (flow matching):
        v_composed = v_uncond + w*(v_A - v_uncond) + w*(v_B - v_uncond)

    Three transformer forward passes per denoising step.
    """
    transformer = pipe.transformer
    scheduler   = pipe.scheduler
    vae         = pipe.vae

    # Encode all three concepts (triple encoders → seq + pooled)
    emb_uncond, pool_uncond = encode_sd35(pipe, "",        device)
    emb_a,      pool_a      = encode_sd35(pipe, concept_a, device)
    emb_b,      pool_b      = encode_sd35(pipe, concept_b, device)

    # Initial latents: 16-channel, no init_noise_sigma scaling for flow matching
    gen = make_generator(device, seed)
    latent_channels = transformer.config.in_channels
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latents = torch.randn(
        (1, latent_channels, height // vae_scale_factor, width // vae_scale_factor),
        generator=gen, device=device, dtype=emb_a.dtype,
    )

    # SD 3.5 Medium uses dynamic shifting: mu must be computed from image
    # sequence length and passed to set_timesteps, otherwise the noise schedule
    # is incorrect (timesteps are not properly shifted toward high-noise end).
    if scheduler.config.get("use_dynamic_shifting", False):
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift
        _, _, h, w = latents.shape
        image_seq_len = (h // transformer.config.patch_size) * (w // transformer.config.patch_size)
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.16),
        )
        scheduler.set_timesteps(steps, device=device, mu=mu)
    else:
        scheduler.set_timesteps(steps, device=device)

    for t in scheduler.timesteps:
        # SD 3.5 transformer expects a scalar timestep tensor
        t_batch = t.expand(latents.shape[0])

        v_uncond = transformer(
            hidden_states=latents,
            timestep=t_batch,
            encoder_hidden_states=emb_uncond,
            pooled_projections=pool_uncond,
        ).sample
        v_a = transformer(
            hidden_states=latents,
            timestep=t_batch,
            encoder_hidden_states=emb_a,
            pooled_projections=pool_a,
        ).sample
        v_b = transformer(
            hidden_states=latents,
            timestep=t_batch,
            encoder_hidden_states=emb_b,
            pooled_projections=pool_b,
        ).sample

        # PoE velocity combination
        v_composed = v_uncond + guidance * (v_a - v_uncond) + guidance * (v_b - v_uncond)

        latents = scheduler.step(v_composed, t, latents).prev_sample

    # Decode: SD 3.5 VAE uses both scaling_factor and shift_factor
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    img_t  = vae.decode(latents).sample
    img_t  = (img_t / 2 + 0.5).clamp(0, 1)
    img_np = img_t.cpu().float().permute(0, 2, 3, 1).numpy()[0]
    return Image.fromarray((img_np * 255).astype("uint8"))


# ===========================================================================
# Grid helpers
# ===========================================================================
COLUMN_LABELS = [
    "Concept A\n(solo)",
    "Concept B\n(solo)",
    "Semantic\n(A and B)",
    "PoE\n(Composable)",
]


def save_pair_grid(images, concept_a, concept_b, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, img, label in zip(axes, images, COLUMN_LABELS):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label, fontsize=11, pad=6)
    fig.suptitle(f'A: "{concept_a}"\nB: "{concept_b}"', fontsize=10, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def save_combined_grid(all_images, all_pairs, out_path, model_tag):
    n_rows, n_cols = len(all_pairs), 4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.8, n_rows * 3.8),
        squeeze=False,
    )
    for r, (images, (ca, cb)) in enumerate(zip(all_images, all_pairs)):
        for c, img in enumerate(images):
            ax = axes[r][c]
            ax.imshow(img)
            ax.axis("off")
            if r == 0:
                ax.set_title(COLUMN_LABELS[c], fontsize=12, pad=6)
            if c == 0:
                short_a = (ca[:38] + "…") if len(ca) > 38 else ca
                short_b = (cb[:38] + "…") if len(cb) > 38 else cb
                ax.set_ylabel(
                    f"A: {short_a}\nB: {short_b}",
                    fontsize=7.5, rotation=0,
                    ha="right", va="center", labelpad=170,
                )
    fig.suptitle(
        f"Spatial Concept Composition [{model_tag}] — "
        "Concept A  |  Concept B  |  Semantic  |  PoE",
        fontsize=13, y=1.004,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved combined grid → {out_path}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()
    args = resolve_defaults(args)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    dtype  = torch.float16 if device_str == "cuda" else torch.float32

    pairs = SPATIAL_CONCEPT_PAIRS
    if args.pairs is not None:
        pairs = pairs[: args.pairs]

    model_tag = "SD3.5" if args.sd35 else "SD1.x"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = args.output_dir / f"run_{model_tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Spatial Composition Grid Generator  [{model_tag}]")
    print(f"  Model        : {args.model}")
    print(f"  Device       : {device}  (dtype={dtype})")
    print(f"  Seed / Steps : {args.seed} / {args.steps}")
    print(f"  Guidance     : {args.guidance_scale}")
    print(f"  Resolution   : {args.width}×{args.height}")
    print(f"  Pairs        : {len(pairs)}")
    print(f"  Output       : {run_dir}")
    print("=" * 80)

    print("\nLoading pipeline …")
    if args.sd35:
        pipe = load_pipeline_sd35(args.model, device, dtype)
        fn_standard = generate_standard_sd35
        fn_poe      = generate_poe_sd35
    else:
        pipe = load_pipeline_sd1(args.model, device, dtype)
        fn_standard = generate_standard_sd1
        fn_poe      = generate_poe_sd1

    all_row_images: List[List[Image.Image]] = []

    for idx, (concept_a, concept_b) in enumerate(pairs):
        print(f"\n[{idx + 1}/{len(pairs)}]")
        print(f"  A = '{concept_a}'")
        print(f"  B = '{concept_b}'")
        semantic = f"{concept_a} and {concept_b}"

        print("  → concept A …")
        img_a   = fn_standard(pipe, concept_a, args.seed, args.steps,
                               args.guidance_scale, args.height, args.width, device)
        print("  → concept B …")
        img_b   = fn_standard(pipe, concept_b, args.seed, args.steps,
                               args.guidance_scale, args.height, args.width, device)
        print("  → semantic …")
        img_sem = fn_standard(pipe, semantic,   args.seed, args.steps,
                               args.guidance_scale, args.height, args.width, device)
        print("  → PoE …")
        img_poe = fn_poe(pipe, concept_a, concept_b, args.seed, args.steps,
                         args.guidance_scale, args.height, args.width, device)

        row = [img_a, img_b, img_sem, img_poe]
        all_row_images.append(row)

        safe = (
            concept_a[:28].replace(" ", "_").replace("/", "-") + "__" +
            concept_b[:28].replace(" ", "_").replace("/", "-")
        )
        save_pair_grid(row, concept_a, concept_b,
                       run_dir / f"pair_{idx + 1:02d}_{safe}.png")

    save_combined_grid(all_row_images, pairs,
                       run_dir / "all_pairs_combined.png",
                       model_tag=model_tag)

    del pipe
    gc.collect()
    if device_str == "cuda":
        torch.cuda.empty_cache()

    print(f"\nAll done. Outputs in: {run_dir}")


if __name__ == "__main__":
    main()
