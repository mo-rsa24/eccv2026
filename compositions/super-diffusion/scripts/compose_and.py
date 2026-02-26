"""
SuperDiff AND composition with Stable Diffusion v1-4.
Extracted from notebooks/superposition_AND.ipynb.

Usage:
    python scripts/compose_and.py \
        --obj "a sailboat" \
        --bg  "cloudy blue sky" \
        --mode deterministic \
        --steps 100 \
        --batch_size 4 \
        --seed 1 \
        --out output.png

Modes:
    deterministic   Adaptive kappa via Jacobian-vector product (get_div=True).
                    More accurate but slower (~2x UNet calls per step).
    stochastic      Adaptive kappa from SDE Langevin noise estimate.
                    Faster, matches stochastic sampler (Euler-Maruyama).
    average         Fixed kappa=0.5 (equal mix, no adaptation).
    single          Standard CFG on --obj prompt only (baseline).
"""

import argparse
import torch
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
import torchvision.utils as tvu
import numpy as np

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--obj",        type=str,   default="a sailboat",
                    help="Object / foreground prompt (concept A)")
parser.add_argument("--bg",         type=str,   default="cloudy blue sky",
                    help="Background / context prompt (concept B)")
parser.add_argument("--mode",       type=str,   default="deterministic",
                    choices=["deterministic", "stochastic", "average", "single"],
                    help="Composition mode (see module docstring)")
parser.add_argument("--steps",      type=int,   default=100,
                    help="Number of denoising steps")
parser.add_argument("--scale",      type=float, default=7.5,
                    help="Classifier-free guidance scale")
parser.add_argument("--batch_size", type=int,   default=4,
                    help="Number of images to generate in parallel")
parser.add_argument("--seed",       type=int,   default=1)
parser.add_argument("--height",     type=int,   default=512)
parser.add_argument("--width",      type=int,   default=512)
parser.add_argument("--lift",       type=float, default=0.0,
                    help="Log-likelihood lift term (deterministic mode only)")
parser.add_argument("--model_id",   type=str,   default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--out",        type=str,   default="superdiff_and.png",
                    help="Output image path (.png)")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float32

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

print(f"Loading {args.model_id} ...")
vae          = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae",          use_safetensors=True).to(device)
tokenizer    = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder", use_safetensors=True).to(device)
unet         = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet",  use_safetensors=True).to(device)
scheduler    = EulerDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler")

vae.eval(); text_encoder.eval(); unet.eval()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_text_embedding(prompt):
    text_input = tokenizer(
        prompt, padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    return text_encoder(text_input.input_ids.to(device))[0]


@torch.no_grad()
def get_vel(t, sigma, latents, embeddings, eps=None, get_div=False):
    """U-Net velocity prediction (flow matching convention).

    v(x, c) = unet(x / sqrt(σ²+1), t, c)

    If get_div=True, also returns the scalar divergence estimate
    d log p / d latents via a Jacobian-vector product (deterministic mode).
    """
    v = lambda _x, _e: unet(_x / ((sigma ** 2 + 1) ** 0.5), t,
                             encoder_hidden_states=_e).sample
    embeds      = torch.cat(embeddings)
    latent_input = latents

    if get_div:
        with sdpa_kernel(SDPBackend.MATH):
            vel, jvp = torch.func.jvp(v, (latent_input, embeds),
                                       (eps, torch.zeros_like(embeds)))
            div = -(eps * jvp).sum((1, 2, 3))
    else:
        vel = v(latent_input, embeds)
        div = torch.zeros([len(embeds)], device=device)

    return vel, div


@torch.no_grad()
def decode_latents(latents, nrow, ncol):
    images = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    images = (images / 2 + 0.5).clamp(0, 1)
    images = (images.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
    rows = []
    for r in range(nrow):
        cols = [images[r * ncol + c] for c in range(ncol)]
        rows.append(np.hstack(cols))
    return Image.fromarray(np.vstack(rows))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

B      = args.batch_size
H, W   = args.height, args.width
w      = args.scale
T      = args.steps
lift   = args.lift

generator = torch.cuda.manual_seed(args.seed) if device.type == "cuda" \
            else torch.manual_seed(args.seed)

obj_emb   = get_text_embedding([args.obj]   * B)
bg_emb    = get_text_embedding([args.bg]    * B)
uncond_emb = get_text_embedding([""]        * B)

latents = torch.randn((B, unet.config.in_channels, H // 8, W // 8),
                      generator=generator, device=device)
scheduler.set_timesteps(T)
latents = latents * scheduler.init_noise_sigma

print(f"Mode: {args.mode}  |  obj: '{args.obj}'  |  bg: '{args.bg}'")
print(f"Steps: {T}  |  guidance: {w}  |  batch: {B}  |  seed: {args.seed}")

# ---------------------------------------------------------------------------
# Denoising loop
# ---------------------------------------------------------------------------

kappa = 0.5 * torch.ones((T + 1, B), device=device)

for i, t in enumerate(scheduler.timesteps):
    dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
    sigma  = scheduler.sigmas[i]

    if args.mode == "deterministic":
        # ── Adaptive κ via JVP divergence estimate ──────────────────────────
        eps = torch.randint_like(latents, 2, dtype=latents.dtype) * 2 - 1
        vel_obj,   dlog_obj = get_vel(t, sigma, latents, [obj_emb],   eps, True)
        vel_bg,    dlog_bg  = get_vel(t, sigma, latents, [bg_emb],    eps, True)
        vel_uncond, _       = get_vel(t, sigma, latents, [uncond_emb])

        denom = w * ((vel_obj - vel_bg) ** 2).sum((1, 2, 3))
        kappa[i + 1] = (
            sigma * (dlog_obj - dlog_bg)
            + ((vel_obj - vel_bg) * (vel_obj + vel_bg)).sum((1, 2, 3))
            + lift / dsigma * sigma / T
            - ((vel_obj - vel_bg) * (vel_uncond + w * (vel_bg - vel_uncond))).sum((1, 2, 3))
        ) / denom

        vf = vel_uncond + w * ((vel_bg - vel_uncond)
             + kappa[i + 1][:, None, None, None] * (vel_obj - vel_bg))
        latents += dsigma * vf

    elif args.mode == "stochastic":
        # ── Adaptive κ from SDE noise estimate ──────────────────────────────
        vel_obj,    _ = get_vel(t, sigma, latents, [obj_emb])
        vel_bg,     _ = get_vel(t, sigma, latents, [bg_emb])
        vel_uncond, _ = get_vel(t, sigma, latents, [uncond_emb])

        noise    = torch.sqrt(2 * torch.abs(dsigma) * sigma) * torch.randn_like(latents)
        dx_ind   = 2 * dsigma * (vel_uncond + w * (vel_bg - vel_uncond)) + noise
        denom    = 2 * dsigma * w * ((vel_obj - vel_bg) ** 2).sum((1, 2, 3))
        kappa[i + 1] = (
            (torch.abs(dsigma) * (vel_bg - vel_obj) * (vel_bg + vel_obj)).sum((1, 2, 3))
            - (dx_ind * (vel_obj - vel_bg)).sum((1, 2, 3))
            + sigma * lift / T
        ) / denom

        vf = vel_uncond + w * ((vel_bg - vel_uncond)
             + kappa[i + 1][:, None, None, None] * (vel_obj - vel_bg))
        latents += 2 * dsigma * vf + noise

    elif args.mode == "average":
        # ── Fixed κ = 0.5 ────────────────────────────────────────────────────
        vel_obj,    _ = get_vel(t, sigma, latents, [obj_emb])
        vel_bg,     _ = get_vel(t, sigma, latents, [bg_emb])
        vel_uncond, _ = get_vel(t, sigma, latents, [uncond_emb])

        vf = vel_uncond + w * ((vel_bg - vel_uncond) + 0.5 * (vel_obj - vel_bg))
        latents += 2 * dsigma * vf + \
                   torch.sqrt(2 * torch.abs(dsigma) * sigma) * torch.randn_like(latents)

    elif args.mode == "single":
        # ── Standard CFG on obj prompt only (baseline) ───────────────────────
        vel_obj,    _ = get_vel(t, sigma, latents, [obj_emb])
        vel_uncond, _ = get_vel(t, sigma, latents, [uncond_emb])

        vf = vel_uncond + w * (vel_obj - vel_uncond)
        latents += 2 * dsigma * vf + \
                   torch.sqrt(2 * torch.abs(dsigma) * sigma) * torch.randn_like(latents)

    if (i + 1) % 10 == 0:
        print(f"  step {i + 1}/{T}")

# ---------------------------------------------------------------------------
# Decode and save
# ---------------------------------------------------------------------------

ncol = min(B, 4)
nrow = (B + ncol - 1) // ncol
image = decode_latents(latents, nrow, ncol)
image.save(args.out)
print(f"Saved → {args.out}")
