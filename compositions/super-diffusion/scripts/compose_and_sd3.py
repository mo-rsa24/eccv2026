"""
SuperDiff AND composition with Stable Diffusion 3.5 Medium.
Straightforward model swap of compose_and.py — SD3 Transformer replaces UNet;
triple text encoders replace single CLIP; FlowMatchEulerDiscreteScheduler
replaces EulerDiscreteScheduler.  All composition math (κ formula, JVP
divergence, denoising update equations) is unchanged, with new chimera-optimised
modes added on top.

Usage:
    python scripts/compose_and_sd3.py \\
        --obj "a sailboat" \\
        --bg  "cloudy blue sky" \\
        --mode chimera \\
        --steps 50 \\
        --batch_size 4 \\
        --seed 1 \\
        --out output.png

Modes:
    chimera         RECOMMENDED for morphological fusion.  Phase-locked composition:
                    early steps force κ=0.5 (both concepts shape structure equally);
                    late steps use adaptive κ hard-clamped to [kappa_min, kappa_max]
                    to prevent either concept from dominating fine detail.
                    Use --slerp for angular-correct velocity blending.
    deterministic   Adaptive κ via Jacobian-vector product (get_div=True).
                    κ is now clamped to [kappa_min, kappa_max] and the denominator
                    is guarded against near-zero collapse.
    stochastic      Adaptive κ from SDE Langevin noise estimate.
    average         Fixed κ (default 0.5, override with --kappa).
                    Use --slerp for spherical velocity interpolation.
    single          Standard CFG on --obj prompt only (baseline).

Chimera-quality flags (all modes):
    --kappa         Fixed κ for average mode (0 = bg only, 1 = obj only). [0.5]
    --kappa_min     Lower κ clamp for chimera/deterministic modes. [0.2]
    --kappa_max     Upper κ clamp for chimera/deterministic modes. [0.8]
    --denom_eps     Minimum denominator in κ formula (prevents NaN). [1e-4]
    --phase_split   Fraction of steps in locked phase (chimera mode). [0.6]
    --kappa_gain    Sigmoid gain for soft-κ saturation in late phase. [2.0]
    --slerp         Use spherical velocity interpolation instead of linear.

SD3.5 differences vs. SD1.4:
    - SD3Transformer2DModel    (no /√(σ²+1) input scaling — removed)
    - 3 text encoders          (CLIP-L + CLIP-G + T5-XXL, packed to 333×4096)
    - 16-channel latents       (vs. 4 channels)
    - FlowMatchEulerDiscreteScheduler
    - VAE decode: shift_factor correction
    - Default guidance scale 4.5  (vs. 7.5)
    - bfloat16                 (vs. float32)
"""

import argparse
import math
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import (
    CLIPTextModelWithProjection, CLIPTokenizer,
    T5EncoderModel, T5TokenizerFast,
)
from diffusers import (
    AutoencoderKL, SD3Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
import numpy as np

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--obj",        type=str,   default="a sailboat",
                    help="Object / foreground prompt (concept A)")
parser.add_argument("--bg",         type=str,   default="cloudy blue sky",
                    help="Background / context prompt (concept B)")
parser.add_argument("--mode",       type=str,   default="chimera",
                    choices=["chimera", "deterministic", "stochastic", "average", "single"],
                    help="Composition mode (see module docstring)")
parser.add_argument("--steps",      type=int,   default=50,
                    help="Number of denoising steps")
parser.add_argument("--scale",      type=float, default=4.5,
                    help="Classifier-free guidance scale")
parser.add_argument("--batch_size", type=int,   default=4,
                    help="Number of images to generate in parallel")
parser.add_argument("--seed",       type=int,   default=1)
parser.add_argument("--height",     type=int,   default=1024)
parser.add_argument("--width",      type=int,   default=1024)
parser.add_argument("--lift",       type=float, default=0.0,
                    help="Log-likelihood lift term (deterministic mode only)")
parser.add_argument("--model_id",   type=str,
                    default="stabilityai/stable-diffusion-3.5-medium")
parser.add_argument("--t5_seq",     type=int,   default=256,
                    help="T5 max sequence length")
parser.add_argument("--out",        type=str,   default="superdiff_and_sd3.png",
                    help="Output image path (.png)")
# ── Chimera-quality flags ──────────────────────────────────────────────────
parser.add_argument("--kappa",      type=float, default=0.5,
                    help="Fixed κ for average mode (0=bg only, 1=obj only). [0.5]")
parser.add_argument("--kappa_min",  type=float, default=0.2,
                    help="Lower κ clamp for chimera/deterministic modes. [0.2]")
parser.add_argument("--kappa_max",  type=float, default=0.8,
                    help="Upper κ clamp for chimera/deterministic modes. [0.8]")
parser.add_argument("--denom_eps",  type=float, default=1e-4,
                    help="Minimum κ denominator — guards against near-zero ‖v_A-v_B‖². [1e-4]")
parser.add_argument("--phase_split",type=float, default=0.6,
                    help="Fraction of steps in locked phase for chimera mode. [0.6]")
parser.add_argument("--kappa_gain", type=float, default=2.0,
                    help="Sigmoid gain for soft-κ in chimera late phase. "
                         "Higher = sharper saturation at kappa_min/max. [2.0]")
parser.add_argument("--slerp",      action="store_true",
                    help="Spherical velocity interpolation instead of linear (average/chimera).")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.bfloat16

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

print(f"Loading {args.model_id} ...")
vae = AutoencoderKL.from_pretrained(
    args.model_id, subfolder="vae", torch_dtype=dtype, use_safetensors=True,
).to(device)
transformer = SD3Transformer2DModel.from_pretrained(
    args.model_id, subfolder="transformer", torch_dtype=dtype, use_safetensors=True,
).to(device)

tokenizer   = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
text_encoder = CLIPTextModelWithProjection.from_pretrained(
    args.model_id, subfolder="text_encoder", torch_dtype=dtype, use_safetensors=True,
).to(device)

tokenizer_2   = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer_2")
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
    args.model_id, subfolder="text_encoder_2", torch_dtype=dtype, use_safetensors=True,
).to(device)

tokenizer_3   = T5TokenizerFast.from_pretrained(args.model_id, subfolder="tokenizer_3")
text_encoder_3 = T5EncoderModel.from_pretrained(
    args.model_id, subfolder="text_encoder_3", torch_dtype=dtype, use_safetensors=True,
).to(device)

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    args.model_id, subfolder="scheduler",
)

vae.eval(); transformer.eval()
text_encoder.eval(); text_encoder_2.eval(); text_encoder_3.eval()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_text_embedding(prompt):
    """Encode prompt with SD3 triple text encoders.

    Returns (prompt_embeds, pooled_prompt_embeds):
        prompt_embeds:        (B, 333, 4096)  — CLIP 77 + T5 256 tokens
        pooled_prompt_embeds: (B, 2048)       — CLIP-L pooled ⊕ CLIP-G pooled
    """
    if isinstance(prompt, str):
        prompt = [prompt]

    # CLIP-L
    inp1 = tokenizer(
        prompt, padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    out1 = text_encoder(inp1.input_ids.to(device), output_hidden_states=True)
    emb1   = out1.hidden_states[-2]   # (B, 77, 768)
    pool1  = out1[0]                   # (B, 768)

    # CLIP-G
    inp2 = tokenizer_2(
        prompt, padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True, return_tensors="pt",
    )
    out2 = text_encoder_2(inp2.input_ids.to(device), output_hidden_states=True)
    emb2   = out2.hidden_states[-2]   # (B, 77, 1280)
    pool2  = out2[0]                   # (B, 1280)

    # T5-XXL
    inp3 = tokenizer_3(
        prompt, padding="max_length",
        max_length=args.t5_seq,
        truncation=True, return_tensors="pt",
    )
    t5_emb = text_encoder_3(inp3.input_ids.to(device))[0]  # (B, 256, 4096)

    # Pack: concat CLIP-L + CLIP-G on feature dim, pad to T5 hidden size, then concat on seq dim
    clip_emb = torch.cat([emb1, emb2], dim=-1)             # (B, 77, 2048)
    clip_emb = F.pad(clip_emb, (0, t5_emb.shape[-1] - clip_emb.shape[-1]))  # (B, 77, 4096)
    prompt_embeds = torch.cat([clip_emb, t5_emb], dim=-2)  # (B, 333, 4096)

    pooled = torch.cat([pool1, pool2], dim=-1)              # (B, 2048)
    return prompt_embeds, pooled


def get_vel(t, latents, ctx, pooled, eps=None, get_div=False):
    """Transformer velocity prediction (flow matching, no input scaling).

    If get_div=True, also returns scalar divergence estimate via JVP
    (deterministic mode).  ctx/pooled are the conditioning tensors.
    """
    def _forward(_x):
        return transformer(
            hidden_states=_x,
            timestep=t.expand(_x.shape[0]).to(device),
            encoder_hidden_states=ctx,
            pooled_projections=pooled,
            return_dict=False,
        )[0]

    if get_div:
        with sdpa_kernel(SDPBackend.MATH):
            vel, jvp_out = torch.func.jvp(_forward, (latents,), (eps,))
        div = -(eps * jvp_out).sum((1, 2, 3))
    else:
        with torch.no_grad():
            vel = _forward(latents)
        div = torch.zeros(latents.shape[0], device=device, dtype=dtype)

    return vel, div


@torch.no_grad()
def decode_latents(latents, nrow, ncol):
    latents = latents.to(dtype=vae.dtype)
    shift_factor = getattr(vae.config, "shift_factor", 0.0)
    images = vae.decode(
        latents / vae.config.scaling_factor + shift_factor, return_dict=False,
    )[0]
    images = (images / 2 + 0.5).clamp(0, 1)
    images = (images.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
    h, w, c = images[0].shape
    blank = np.zeros((h, w, c), dtype=np.uint8)
    frames = list(images) + [blank] * (nrow * ncol - len(images))
    rows = []
    for r in range(nrow):
        cols = [frames[r * ncol + c_] for c_ in range(ncol)]
        rows.append(np.hstack(cols))
    return Image.fromarray(np.vstack(rows))


def _slerp_vel(
    v0: torch.Tensor, v1: torch.Tensor, t: float = 0.5, eps: float = 1e-6,
) -> torch.Tensor:
    """Spherical linear interpolation between two velocity fields (per sample).

    Unlike linear blending, slerp preserves the angular geometry of both
    velocity directions when they point in very different directions — critical
    when obj and bg score fields oppose each other in latent space.

    Returns a velocity with the interpolated direction and the average magnitude.
    """
    B   = v0.shape[0]
    v0f = v0.reshape(B, -1).float()
    v1f = v1.reshape(B, -1).float()

    mag0 = v0f.norm(dim=-1, keepdim=True).clamp(min=eps)
    mag1 = v1f.norm(dim=-1, keepdim=True).clamp(min=eps)

    v0n = v0f / mag0
    v1n = v1f / mag1

    cos_th  = (v0n * v1n).sum(-1).clamp(-1.0, 1.0)   # (B,)
    theta   = cos_th.acos()                            # (B,)
    sin_th  = theta.sin()
    degen   = sin_th.abs() < eps                       # (B,) nearly parallel

    w0 = torch.where(degen, torch.full_like(theta, 1 - t),
                     ((1 - t) * theta).sin() / sin_th)
    w1 = torch.where(degen, torch.full_like(theta, t),
                     (t       * theta).sin() / sin_th)

    avg_mag = (mag0 + mag1) / 2                         # (B, 1)
    result  = w0.unsqueeze(-1) * v0n + w1.unsqueeze(-1) * v1n
    result  = F.normalize(result, dim=-1) * avg_mag
    return result.reshape(v0.shape).to(v0.dtype)


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

obj_ctx,    obj_pooled    = get_text_embedding([args.obj] * B)
bg_ctx,     bg_pooled     = get_text_embedding([args.bg]  * B)
uncond_ctx, uncond_pooled = get_text_embedding([""]       * B)

# SD3.5 uses 16-channel latents; spatial resolution is H//8 × W//8
latents = torch.randn(
    (B, 16, H // 8, W // 8),
    generator=generator, device=device, dtype=dtype,
)
scheduler.set_timesteps(T)
# FlowMatchEulerDiscreteScheduler has no init_noise_sigma; FM latents are pure N(0,1)

print(f"Mode: {args.mode}  |  obj: '{args.obj}'  |  bg: '{args.bg}'")
print(f"Steps: {T}  |  guidance: {w}  |  batch: {B}  |  seed: {args.seed}")
if args.mode in ("chimera", "deterministic"):
    print(f"κ ∈ [{args.kappa_min}, {args.kappa_max}]  |  "
          f"phase_split: {args.phase_split if args.mode == 'chimera' else 'n/a'}  |  "
          f"denom_eps: {args.denom_eps}  |  slerp: {args.slerp}")

# ---------------------------------------------------------------------------
# Denoising loop  (composition math identical to compose_and.py)
# ---------------------------------------------------------------------------

kappa = args.kappa * torch.ones((T + 1, B), device=device, dtype=dtype)

for i, t in enumerate(scheduler.timesteps):
    dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
    sigma  = scheduler.sigmas[i]

    if args.mode == "chimera":
        # ── Phase-locked chimera mode ────────────────────────────────────────
        # Early phase (first phase_split fraction of steps): κ = args.kappa
        #   Both concepts contribute equally to coarse structure; neither can
        #   dominate while the large-σ flow is laying out global topology.
        # Late phase: adaptive κ hard-clamped to [kappa_min, kappa_max].
        #   Fine-detail specialisation is allowed, but clamping prevents full
        #   concept takeover — ensuring both concepts remain visible at decode.
        vel_obj,    _ = get_vel(t, latents, obj_ctx,    obj_pooled)
        vel_bg,     _ = get_vel(t, latents, bg_ctx,     bg_pooled)
        vel_uncond, _ = get_vel(t, latents, uncond_ctx, uncond_pooled)

        phase_step = int(T * args.phase_split)
        if i < phase_step:
            # Locked early phase: kappa already initialised to args.kappa (0.5)
            pass
        else:
            # Late phase: adaptive κ with tanh soft-saturation.
            # Hard clamp risks κ sticking at the boundary for many consecutive
            # steps (≡ one concept dominating late detail).  Tanh asymptotically
            # approaches kappa_min/max so both concepts always contribute.
            denom = (w * ((vel_obj - vel_bg) ** 2).sum((1, 2, 3))).clamp(min=args.denom_eps)
            kappa_adaptive = (
                ((vel_obj - vel_bg) * (vel_obj + vel_bg)).sum((1, 2, 3))
                - ((vel_obj - vel_bg) * (vel_uncond + w * (vel_bg - vel_uncond))).sum((1, 2, 3))
            ) / denom
            center     = (args.kappa_max + args.kappa_min) / 2          # 0.5 by default
            half_range = (args.kappa_max - args.kappa_min) / 2          # 0.3 by default
            k_norm     = (kappa_adaptive - center) * args.kappa_gain / half_range
            kappa[i + 1] = center + half_range * torch.tanh(k_norm)

        kap = kappa[i + 1][:, None, None, None]
        if args.slerp:
            v_cfg_bg  = vel_uncond + w * (vel_bg  - vel_uncond)
            v_cfg_obj = vel_uncond + w * (vel_obj - vel_uncond)
            # Blend at the current κ; then add the uncond residual back
            vf = _slerp_vel(v_cfg_bg, v_cfg_obj, t=float(kappa[i + 1].mean()))
        else:
            vf = vel_uncond + w * ((vel_bg - vel_uncond) + kap * (vel_obj - vel_bg))
        latents = latents + dsigma * vf

    elif args.mode == "deterministic":
        # ── Adaptive κ via JVP divergence estimate ──────────────────────────
        eps = torch.randint_like(latents, 2, dtype=latents.dtype) * 2 - 1
        vel_obj,    dlog_obj = get_vel(t, latents, obj_ctx,    obj_pooled,    eps, True)
        vel_bg,     dlog_bg  = get_vel(t, latents, bg_ctx,     bg_pooled,     eps, True)
        vel_uncond, _        = get_vel(t, latents, uncond_ctx, uncond_pooled)

        # Guard denominator: when ‖v_A - v_B‖² ≈ 0 (velocities nearly parallel),
        # the raw κ formula diverges.  Clamping prevents NaN propagation.
        denom = (w * ((vel_obj - vel_bg) ** 2).sum((1, 2, 3))).clamp(min=args.denom_eps)
        kappa[i + 1] = (
            sigma * (dlog_obj - dlog_bg)
            + ((vel_obj - vel_bg) * (vel_obj + vel_bg)).sum((1, 2, 3))
            + lift / dsigma * sigma / T
            - ((vel_obj - vel_bg) * (vel_uncond + w * (vel_bg - vel_uncond))).sum((1, 2, 3))
        ) / denom
        kappa[i + 1] = kappa[i + 1].clamp(args.kappa_min, args.kappa_max)

        vf = vel_uncond + w * ((vel_bg - vel_uncond)
             + kappa[i + 1][:, None, None, None] * (vel_obj - vel_bg))
        latents = latents + dsigma * vf

    elif args.mode == "stochastic":
        # ── Adaptive κ from SDE noise estimate — FM ODE step ────────────────
        # Noise is used only to estimate κ (dx_ind); the latent update is a
        # clean ODE step.  The Langevin SDE noise is not appropriate for SD3.5.
        vel_obj,    _ = get_vel(t, latents, obj_ctx,    obj_pooled)
        vel_bg,     _ = get_vel(t, latents, bg_ctx,     bg_pooled)
        vel_uncond, _ = get_vel(t, latents, uncond_ctx, uncond_pooled)

        sigma_safe = sigma.clamp(min=1e-4)
        noise    = torch.sqrt(2 * torch.abs(dsigma) * sigma_safe) * torch.randn_like(latents)
        dx_ind   = dsigma * (vel_uncond + w * (vel_bg - vel_uncond)) + noise
        denom    = (dsigma * w * ((vel_obj - vel_bg) ** 2).sum((1, 2, 3))).clamp(min=args.denom_eps)
        kappa[i + 1] = (
            (torch.abs(dsigma) * (vel_bg - vel_obj) * (vel_bg + vel_obj)).sum((1, 2, 3))
            - (dx_ind * (vel_obj - vel_bg)).sum((1, 2, 3))
            + sigma_safe * lift / T
        ) / denom

        vf = vel_uncond + w * ((vel_bg - vel_uncond)
             + kappa[i + 1][:, None, None, None] * (vel_obj - vel_bg))
        latents = latents + dsigma * vf   # clean ODE step

    elif args.mode == "average":
        # ── Fixed κ (default 0.5), optionally with spherical blending ────────
        vel_obj,    _ = get_vel(t, latents, obj_ctx,    obj_pooled)
        vel_bg,     _ = get_vel(t, latents, bg_ctx,     bg_pooled)
        vel_uncond, _ = get_vel(t, latents, uncond_ctx, uncond_pooled)

        if args.slerp:
            v_cfg_bg  = vel_uncond + w * (vel_bg  - vel_uncond)
            v_cfg_obj = vel_uncond + w * (vel_obj - vel_uncond)
            vf = _slerp_vel(v_cfg_bg, v_cfg_obj, t=args.kappa)
        else:
            vf = vel_uncond + w * ((vel_bg - vel_uncond) + args.kappa * (vel_obj - vel_bg))
        latents = latents + dsigma * vf   # clean ODE step, no Langevin noise

    elif args.mode == "single":
        # ── Standard CFG on obj prompt only — pure FM ODE (matches SD3 pipeline) ─
        vel_obj,    _ = get_vel(t, latents, obj_ctx,    obj_pooled)
        vel_uncond, _ = get_vel(t, latents, uncond_ctx, uncond_pooled)

        vf = vel_uncond + w * (vel_obj - vel_uncond)
        latents = latents + dsigma * vf

    if (i + 1) % 10 == 0:
        print(f"  step {i + 1}/{T}")

# ---------------------------------------------------------------------------
# Decode and save
# ---------------------------------------------------------------------------

ncol  = min(B, 4)
nrow  = (B + ncol - 1) // ncol
image = decode_latents(latents, nrow, ncol)
image.save(args.out)
print(f"Saved → {args.out}")
