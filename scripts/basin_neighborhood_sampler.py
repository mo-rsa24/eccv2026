"""
Basin Neighborhood Sampler for SuperDiff AND chimera seeds.

Given a known chimera seed (e.g., seed=25 for "a bird" ∧ "a cat"), this
script reconstructs the original latent z_T, perturbs it at several epsilon
levels, runs SuperDiff AND on each perturbed latent, then VLM-captions the
decoded images and classifies each caption as chimeric or not.

The resulting "chimera fraction vs epsilon" curve reveals whether the seed
landed in a broad latent basin (structural, reliable) or a narrow saddle
(one-off fluke).

Epsilon is the standard-deviation scale of the additive Gaussian perturbation:
    z_T_perturbed = z_T  +  eps * delta,     delta ~ N(0, I)

For SD3.5 16-ch latents (16 × 128 × 128), a perturbation of eps=0.1 moves
≈ sqrt(16*128*128) * 0.1 ≈ 18 units in L2 norm while z_T itself has norm
≈ 181; so eps=0.1 is a gentle 10% nudge in each dimension.

Usage
-----
# Quick run: seed 25, "a bird" ∧ "a cat", 4 samples per epsilon level:
conda run -n superdiff python scripts/basin_neighborhood_sampler.py \\
    --obj "a bird" --bg "a cat" \\
    --chimera-seed 25 \\
    --epsilons "0.0 0.05 0.1 0.2 0.4 0.8" \\
    --samples-per-eps 4

# Larger sweep with deterministic mode:
conda run -n superdiff python scripts/basin_neighborhood_sampler.py \\
    --obj "a bird" --bg "a cat" \\
    --chimera-seed 25 \\
    --mode average --scale 3.0 --lift 3.0 \\
    --epsilons "0.0 0.1 0.2 0.4 0.8 1.6" \\
    --samples-per-eps 8 \\
    --out-dir outputs/basin_bird_cat_seed25

Output
------
  outputs/basin_<slug>/
    basin_report.json       — per-epsilon chimera fractions + all captions
    basin_curve.png         — chimera fraction vs epsilon plot
    eps_<e>/grid.png        — decoded image grid for each epsilon level
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Chimera classification (mirrors mine_chimera_captions.py)
# ---------------------------------------------------------------------------

_BODY_PARTS = (
    r"face|head|body|eye|eyes|fur|feathers?|legs?|tail|beak|paws?|wings?|"
    r"ears?|nose|snout|muzzle|mouth|claw|claws|hooves?|horns?|scales?|skin|"
    r"neck|belly|coat|plumage|bill|talons?"
)

_MORPHOLOGICAL = [
    re.compile(rf"\b\w+\s*'s\s+(?:{_BODY_PARTS})\b", re.I),
    re.compile(r"\bhybrid\b", re.I),
    re.compile(r"\bchimera\b", re.I),
    re.compile(r"\bcross\s+between\b", re.I),
    re.compile(
        r"\bpart[\s-](?:bird|cat|dog|horse|fish|human|person|woman|man|animal)\b", re.I
    ),
    re.compile(
        r"\b(?:bird|cat|dog|horse|fish|human|person)[\s-]like\s+"
        r"(?:bird|cat|dog|horse|fish|human|person|woman|man)\b",
        re.I,
    ),
    re.compile(rf"\bwith\b.{{1,30}}\b(?:{_BODY_PARTS})\b", re.I),
]

_ANATOMICAL = [
    re.compile(
        r"\b(?:cat|dog|horse|person|woman|man)\b.{1,50}"
        r"\b(?:beak|feathers?|wings?|talons?|plumage|bill)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:bird|parrot|sparrow|eagle|owl)\b.{1,50}"
        r"\b(?:fur|paws?|hooves?|mane|whiskers?)\b",
        re.I,
    ),
]


def classify_caption(caption: str) -> str:
    """Return 'strong', 'medium', or 'none'."""
    for pat in _MORPHOLOGICAL:
        if pat.search(caption):
            return "strong"
    for pat in _ANATOMICAL:
        if pat.search(caption):
            return "medium"
    return "none"


def is_chimeric(caption: str) -> bool:
    return classify_caption(caption) in ("strong", "medium")


# ---------------------------------------------------------------------------
# SD3.5 model helpers (self-contained, mirrors compose_and_sd3.py)
# ---------------------------------------------------------------------------

def load_sd3_models(model_id: str, dtype: torch.dtype, device: torch.device):
    from transformers import (
        CLIPTextModelWithProjection, CLIPTokenizer,
        T5EncoderModel, T5TokenizerFast,
    )
    from diffusers import (
        AutoencoderKL, SD3Transformer2DModel,
        FlowMatchEulerDiscreteScheduler,
    )

    print(f"  Loading SD3.5 models from {model_id} ...")
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype, use_safetensors=True,
    ).to(device).eval()

    transformer = SD3Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=dtype, use_safetensors=True,
    ).to(device).eval()

    tokenizer    = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype, use_safetensors=True,
    ).to(device).eval()

    tokenizer_2    = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=dtype, use_safetensors=True,
    ).to(device).eval()

    tokenizer_3    = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_3")
    text_encoder_3 = T5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder_3", torch_dtype=dtype, use_safetensors=True,
    ).to(device).eval()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler",
    )

    encoders = dict(
        tokenizer=tokenizer,       text_encoder=text_encoder,
        tokenizer_2=tokenizer_2,   text_encoder_2=text_encoder_2,
        tokenizer_3=tokenizer_3,   text_encoder_3=text_encoder_3,
    )
    return vae, transformer, scheduler, encoders


@torch.no_grad()
def get_text_embedding(prompt, encoders, device, dtype, t5_seq=256):
    """SD3.5 triple-encoder text embedding."""
    if isinstance(prompt, str):
        prompt = [prompt]

    tok1, te1  = encoders["tokenizer"],   encoders["text_encoder"]
    tok2, te2  = encoders["tokenizer_2"], encoders["text_encoder_2"]
    tok3, te3  = encoders["tokenizer_3"], encoders["text_encoder_3"]

    inp1  = tok1(prompt, padding="max_length",
                 max_length=tok1.model_max_length,
                 truncation=True, return_tensors="pt")
    out1  = te1(inp1.input_ids.to(device), output_hidden_states=True)
    emb1, pool1 = out1.hidden_states[-2], out1[0]

    inp2  = tok2(prompt, padding="max_length",
                 max_length=tok2.model_max_length,
                 truncation=True, return_tensors="pt")
    out2  = te2(inp2.input_ids.to(device), output_hidden_states=True)
    emb2, pool2 = out2.hidden_states[-2], out2[0]

    inp3  = tok3(prompt, padding="max_length",
                 max_length=t5_seq,
                 truncation=True, return_tensors="pt")
    t5_emb = te3(inp3.input_ids.to(device))[0]

    clip_emb = torch.cat([emb1, emb2], dim=-1)
    clip_emb = F.pad(clip_emb, (0, t5_emb.shape[-1] - clip_emb.shape[-1]))
    prompt_embeds = torch.cat([clip_emb, t5_emb], dim=-2)
    pooled = torch.cat([pool1, pool2], dim=-1)
    return prompt_embeds, pooled


def get_vel(t, latents, ctx, pooled, transformer, device, dtype, eps=None, get_div=False):
    """Single transformer forward (or JVP) for velocity prediction."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def _fwd(_x):
        return transformer(
            hidden_states=_x,
            timestep=t.expand(_x.shape[0]).to(device),
            encoder_hidden_states=ctx,
            pooled_projections=pooled,
            return_dict=False,
        )[0]

    if get_div:
        with sdpa_kernel(SDPBackend.MATH):
            vel, jvp_out = torch.func.jvp(_fwd, (latents,), (eps,))
        div = -(eps * jvp_out).sum((1, 2, 3))
    else:
        with torch.no_grad():
            vel = _fwd(latents)
        div = torch.zeros(latents.shape[0], device=device, dtype=dtype)

    return vel, div


def superdiff_and_denoise(
    z_T: torch.Tensor,       # (1, 16, H//8, W//8)
    obj_ctx, obj_pooled,
    bg_ctx,  bg_pooled,
    unc_ctx, unc_pooled,
    transformer, scheduler,
    device, dtype,
    mode: str   = "average",
    scale: float = 4.5,
    lift: float  = 0.0,
    steps: int   = 50,
) -> torch.Tensor:
    """Run SuperDiff AND denoising on a single latent. Returns decoded latents."""
    latents = z_T.clone()
    w       = scale
    T       = steps
    scheduler.set_timesteps(T)

    kappa = 0.5 * torch.ones((T + 1, 1), device=device, dtype=dtype)

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma  = scheduler.sigmas[i]

        if mode == "deterministic":
            eps = torch.randint_like(latents, 2, dtype=latents.dtype) * 2 - 1
            vel_obj,    dlog_obj = get_vel(t, latents, obj_ctx, obj_pooled,
                                           transformer, device, dtype, eps, True)
            vel_bg,     dlog_bg  = get_vel(t, latents, bg_ctx,  bg_pooled,
                                           transformer, device, dtype, eps, True)
            vel_uncond, _        = get_vel(t, latents, unc_ctx, unc_pooled,
                                           transformer, device, dtype)

            denom = w * ((vel_obj - vel_bg) ** 2).sum((1, 2, 3))
            kappa[i + 1] = (
                sigma * (dlog_obj - dlog_bg)
                + ((vel_obj - vel_bg) * (vel_obj + vel_bg)).sum((1, 2, 3))
                + lift / dsigma * sigma / T
                - ((vel_obj - vel_bg) * (vel_uncond + w * (vel_bg - vel_uncond))).sum((1, 2, 3))
            ) / denom

        elif mode == "stochastic":
            vel_obj,    _ = get_vel(t, latents, obj_ctx, obj_pooled, transformer, device, dtype)
            vel_bg,     _ = get_vel(t, latents, bg_ctx,  bg_pooled,  transformer, device, dtype)
            vel_uncond, _ = get_vel(t, latents, unc_ctx, unc_pooled, transformer, device, dtype)

            sigma_safe = sigma.clamp(min=1e-4)
            noise    = torch.sqrt(2 * torch.abs(dsigma) * sigma_safe) * torch.randn_like(latents)
            dx_ind   = dsigma * (vel_uncond + w * (vel_bg - vel_uncond)) + noise
            denom    = dsigma * w * ((vel_obj - vel_bg) ** 2).sum((1, 2, 3))
            kappa[i + 1] = (
                (torch.abs(dsigma) * (vel_bg - vel_obj) * (vel_bg + vel_obj)).sum((1, 2, 3))
                - (dx_ind * (vel_obj - vel_bg)).sum((1, 2, 3))
                + sigma_safe * lift / T
            ) / denom

        else:  # "average" (default — best for chimera discovery)
            vel_obj,    _ = get_vel(t, latents, obj_ctx, obj_pooled, transformer, device, dtype)
            vel_bg,     _ = get_vel(t, latents, bg_ctx,  bg_pooled,  transformer, device, dtype)
            vel_uncond, _ = get_vel(t, latents, unc_ctx, unc_pooled, transformer, device, dtype)

        if mode in ("deterministic", "stochastic", "average"):
            kap = kappa[i + 1][:, None, None, None] if mode != "average" \
                  else torch.tensor(0.5, device=device, dtype=dtype)
            vf = vel_uncond + w * ((vel_bg - vel_uncond) + kap * (vel_obj - vel_bg))
            latents = latents + dsigma * vf

    return latents


@torch.no_grad()
def decode(latents, vae, dtype):
    latents = latents.to(dtype=vae.dtype)
    shift   = getattr(vae.config, "shift_factor", 0.0)
    imgs    = vae.decode(latents / vae.config.scaling_factor + shift, return_dict=False)[0]
    imgs    = (imgs / 2 + 0.5).clamp(0, 1).float()
    return imgs  # (B, 3, H, W) float32 in [0,1]


# ---------------------------------------------------------------------------
# VLM captioning helpers
# ---------------------------------------------------------------------------

def load_blip2(model_id: str, device: torch.device):
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    print(f"  Loading BLIP-2 ({model_id}) ...")
    proc  = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16,
    ).to(device).eval()
    return proc, model


@torch.no_grad()
def caption_images(pil_images: list, proc, blip2, device, max_new_tokens=60) -> List[str]:
    captions = []
    for pil in pil_images:
        inputs  = proc(images=pil, return_tensors="pt").to(device, torch.float16)
        out_ids = blip2.generate(**inputs, max_new_tokens=max_new_tokens)
        captions.append(proc.decode(out_ids[0], skip_special_tokens=True).strip() or "a photo")
    return captions


# ---------------------------------------------------------------------------
# Image grid helper
# ---------------------------------------------------------------------------

def make_grid_pil(tensors: List[torch.Tensor], ncols: int) -> Image.Image:
    """Stack a list of (3, H, W) float [0,1] tensors into a PIL grid."""
    imgs_np = [(t.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8") for t in tensors]
    H, W    = imgs_np[0].shape[:2]
    nrows   = (len(imgs_np) + ncols - 1) // ncols
    canvas  = np.zeros((nrows * H, ncols * W, 3), dtype="uint8")
    for i, img in enumerate(imgs_np):
        r, c = divmod(i, ncols)
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = img
    return Image.fromarray(canvas)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_basin_curve(eps_values, frac_chimeric, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(eps_values, frac_chimeric, marker="o", linewidth=2, color="#e63946")
        ax.fill_between(eps_values, frac_chimeric, alpha=0.15, color="#e63946")
        ax.set_xlabel("Perturbation ε (additive Gaussian std)", fontsize=12)
        ax.set_ylabel("Fraction of images classified chimeric", fontsize=12)
        ax.set_title("Basin Neighbourhood: chimera fraction vs ε", fontsize=13)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Basin curve saved → {out_path}")
    except ImportError:
        print("  (matplotlib not available; skipping basin_curve.png)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Basin neighbourhood sampler for SuperDiff AND chimera seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--obj",            required=True, help="Object prompt (concept A)")
    p.add_argument("--bg",             required=True, help="Background prompt (concept B)")
    p.add_argument("--chimera-seed",   type=int, required=True,
                   help="The known chimera seed whose z_T to reconstruct")
    p.add_argument("--epsilons",       default="0.0 0.1 0.2 0.4 0.8",
                   help="Space-separated epsilon values to test (default: '0.0 0.1 0.2 0.4 0.8')")
    p.add_argument("--samples-per-eps", type=int, default=4,
                   help="Number of perturbed samples to draw per epsilon level")
    p.add_argument("--perturbation-seed", type=int, default=0,
                   help="RNG seed for the perturbation vectors δ (reproducibility)")
    p.add_argument("--mode",   default="average",
                   choices=["deterministic", "stochastic", "average"],
                   help="SuperDiff AND composition mode (default: average — best for chimeras)")
    p.add_argument("--scale",  type=float, default=3.0,
                   help="CFG guidance scale (default: 3.0 for chimera-friendly soft guidance)")
    p.add_argument("--lift",   type=float, default=0.0,
                   help="Log-likelihood lift (deterministic mode only)")
    p.add_argument("--steps",  type=int,   default=50,
                   help="Denoising steps")
    p.add_argument("--height", type=int,   default=512,
                   help="Image height (use 512 for speed; SD3.5 is happy with any multiple of 8)")
    p.add_argument("--width",  type=int,   default=512)
    p.add_argument("--model-id", default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--vlm-model-id",   default="Salesforce/blip2-opt-2.7b")
    p.add_argument("--vlm-max-tokens", type=int, default=60)
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: outputs/basin_<obj>_<bg>_s<seed>)")
    p.add_argument("--t5-seq", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    epsilons = [float(e) for e in args.epsilons.split()]
    N        = args.samples_per_eps

    slug = (
        f"{args.obj[:20].replace(' ', '_')}"
        f"_{args.bg[:20].replace(' ', '_')}"
        f"_s{args.chimera_seed}"
    )
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"outputs/basin_{slug}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBasin Neighbourhood Sampler")
    print(f"  obj : {args.obj!r}")
    print(f"  bg  : {args.bg!r}")
    print(f"  chimera seed : {args.chimera_seed}")
    print(f"  epsilons     : {epsilons}")
    print(f"  samples/eps  : {N}")
    print(f"  mode         : {args.mode}  scale={args.scale}  lift={args.lift}")
    print(f"  output dir   : {out_dir}\n")

    # ------------------------------------------------------------------
    # Step 1: Load SD3.5 and reconstruct z_T(chimera_seed)
    # ------------------------------------------------------------------
    print("[1/3] Loading SD3.5 models ...")
    vae, transformer, scheduler, encoders = load_sd3_models(
        args.model_id, dtype, device
    )

    H, W  = args.height, args.width
    t5seq = args.t5_seq

    # Reconstruct the chimera latent (same RNG as compose_and_sd3.py)
    gen_base = (
        torch.cuda.manual_seed(args.chimera_seed)
        if device.type == "cuda"
        else torch.manual_seed(args.chimera_seed)
    )
    z_chimera = torch.randn(
        (1, 16, H // 8, W // 8),
        generator=gen_base, device=device, dtype=dtype,
    )
    print(f"  z_T(seed={args.chimera_seed}) reconstructed.  "
          f"norm = {z_chimera.norm().item():.2f}")

    # Encode text once
    print("  Encoding text prompts ...")
    obj_ctx, obj_pooled = get_text_embedding(
        [args.obj], encoders, device, dtype, t5seq
    )
    bg_ctx,  bg_pooled  = get_text_embedding(
        [args.bg],  encoders, device, dtype, t5seq
    )
    unc_ctx, unc_pooled = get_text_embedding(
        [""],        encoders, device, dtype, t5seq
    )

    # ------------------------------------------------------------------
    # Step 2: Generate images for each (epsilon, sample) pair
    # ------------------------------------------------------------------
    print("[2/3] Generating perturbed images ...")

    # Shared perturbation RNG for reproducibility across runs
    gen_perturb = torch.Generator(device=device)
    gen_perturb.manual_seed(args.perturbation_seed)

    all_pil_images   = {}   # eps -> List[PIL.Image]
    all_latent_norms = {}   # eps -> List[float]  (cosine similarity to z_chimera)

    for eps in epsilons:
        eps_key   = f"{eps:.3f}"
        eps_dir   = out_dir / f"eps_{eps_key}"
        eps_dir.mkdir(exist_ok=True)

        pil_imgs  = []
        cos_sims  = []

        for sample_idx in range(N):
            if eps == 0.0:
                z_perturbed = z_chimera.clone()
            else:
                delta       = torch.randn_like(z_chimera, generator=gen_perturb)
                z_perturbed = z_chimera + eps * delta

            # Cosine similarity between perturbed and original (diagnostic)
            cos_sim = F.cosine_similarity(
                z_perturbed.flatten().unsqueeze(0),
                z_chimera.flatten().unsqueeze(0),
            ).item()
            cos_sims.append(cos_sim)

            # Run SuperDiff AND denoising
            scheduler_copy = scheduler.__class__.from_config(scheduler.config)
            final_latents  = superdiff_and_denoise(
                z_perturbed,
                obj_ctx, obj_pooled,
                bg_ctx,  bg_pooled,
                unc_ctx, unc_pooled,
                transformer, scheduler_copy,
                device, dtype,
                mode=args.mode,
                scale=args.scale,
                lift=args.lift,
                steps=args.steps,
            )

            # Decode to PIL
            img_tensor = decode(final_latents, vae, dtype)[0]  # (3, H, W)
            pil_img    = Image.fromarray(
                (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            )
            pil_imgs.append(pil_img)

            print(f"  eps={eps:.3f}  sample {sample_idx + 1}/{N}"
                  f"  cosim={cos_sim:.4f}")

        all_pil_images[eps_key]   = pil_imgs
        all_latent_norms[eps_key] = cos_sims

        # Save image grid for this epsilon level
        tensors = [
            torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            for img in pil_imgs
        ]
        grid    = make_grid_pil(tensors, ncols=min(N, 4))
        grid.save(eps_dir / "grid.png")
        print(f"  Grid saved → {eps_dir / 'grid.png'}")

    # Free transformer memory before loading BLIP-2
    del transformer
    for enc in encoders.values():
        del enc
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 3: VLM-caption all images and classify
    # ------------------------------------------------------------------
    print("\n[3/3] VLM captioning with BLIP-2 ...")
    blip2_proc, blip2_model = load_blip2(args.vlm_model_id, device)

    report        = []
    frac_chimeric = []

    for eps in epsilons:
        eps_key  = f"{eps:.3f}"
        pil_imgs = all_pil_images[eps_key]
        captions = caption_images(pil_imgs, blip2_proc, blip2_model, device,
                                   max_new_tokens=args.vlm_max_tokens)

        classifications = [classify_caption(c) for c in captions]
        n_chimeric      = sum(1 for cl in classifications if cl != "none")
        frac            = n_chimeric / len(captions) if captions else 0.0
        frac_chimeric.append(frac)

        eps_entry = {
            "epsilon":         eps,
            "n_samples":       len(captions),
            "n_chimeric":      n_chimeric,
            "frac_chimeric":   frac,
            "mean_cosim":      float(np.mean(all_latent_norms[eps_key])),
            "samples":         [
                {
                    "sample_idx":     i,
                    "cosim_to_base":  all_latent_norms[eps_key][i],
                    "caption":        captions[i],
                    "classification": classifications[i],
                }
                for i in range(len(captions))
            ],
        }
        report.append(eps_entry)

        print(f"  eps={eps:.3f}  chimeric={n_chimeric}/{len(captions)}"
              f"  ({frac:.0%})")
        for i, (cap, cl) in enumerate(zip(captions, classifications)):
            marker = "✓" if cl != "none" else "·"
            print(f"    [{marker}] sample {i}  [{cl}]  {cap!r}")

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    full_report = {
        "obj":            args.obj,
        "bg":             args.bg,
        "chimera_seed":   args.chimera_seed,
        "mode":           args.mode,
        "scale":          args.scale,
        "lift":           args.lift,
        "steps":          args.steps,
        "model_id":       args.model_id,
        "perturbation_seed": args.perturbation_seed,
        "results":        report,
    }

    report_path = out_dir / "basin_report.json"
    report_path.write_text(json.dumps(full_report, indent=2) + "\n")
    print(f"\nReport saved → {report_path}")

    plot_basin_curve(epsilons, frac_chimeric, out_dir / "basin_curve.png")

    # Console summary
    print(f"\n{'─'*55}")
    print(f"  {'epsilon':>8}  {'cosim':>7}  {'chimeric':>10}  captions")
    print(f"{'─'*55}")
    for entry in report:
        eps_key = f"{entry['epsilon']:.3f}"
        cos_str = f"{entry['mean_cosim']:.4f}"
        chi_str = f"{entry['n_chimeric']}/{entry['n_samples']} ({entry['frac_chimeric']:.0%})"
        print(f"  {eps_key:>8}  {cos_str:>7}  {chi_str:>10}")
    print(f"{'─'*55}")

    # Estimate basin radius: largest eps where frac > 0.5
    basin_radius = 0.0
    for entry in report:
        if entry["frac_chimeric"] >= 0.5:
            basin_radius = entry["epsilon"]
    print(f"\n  Chimera basin radius (50% threshold): eps ≤ {basin_radius:.3f}")
    print(f"  {'BROAD BASIN — structural chimera mode!' if basin_radius > 0.1 else 'NARROW BASIN — saddle-point / lucky seed'}")
    print()


if __name__ == "__main__":
    main()
