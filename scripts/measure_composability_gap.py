"""
Phase 3: Measure the composability gap between SuperDiff-AND and SD3.5(p*).

For each held-out test pair (c₁, c₂):
  1. Generate images from SuperDiff-AND(c₁, c₂) with trajectory tracking
  2. Invert each image via f_θ to get p* = (pooled_embeds_pred, seq_embeds_pred)
  3. Generate images from SD3.5(p*) with same initial noise + trajectory tracking
  4. Generate images from SD3.5("c₁ and c₂") as the naive monolithic baseline
  5. Compute gap metrics at three levels:
       - Image:      CLIP cosine similarity, LPIPS
       - Latent:     MSE in VAE latent space at the final step
       - Trajectory: step-wise MSE and cosine similarity

Results are saved as gap_metrics.json + image grids per pair.

Usage
-----
# Training-free quick run (no checkpoint needed; auto-timestamped output dir):
conda run -n superdiff python scripts/measure_composability_gap.py \
    --pstar-source vlm --regime small \
    --steps 50 --guidance 4.5

# Full inverter run (requires trained checkpoint):
conda run -n superdiff python scripts/measure_composability_gap.py \
    --pstar-source inverter --ckpt ckpt/inverter/best.pt \
    --regime medium --steps 50 --guidance 4.5

# Accumulate a second source into an existing run dir:
conda run -n superdiff python scripts/measure_composability_gap.py \
    --pstar-source pez --regime medium \
    --output-dir experiments/inversion/gap_analysis/medium_<timestamp> --merge

# Explicit output dir (disables auto-timestamp):
conda run -n superdiff python scripts/measure_composability_gap.py \
    --pstar-source vlm --regime small \
    --output-dir experiments/inversion/gap_analysis/my_run \
    --steps 50 --guidance 4.5

# Choose which seed is used for plot_gap_analysis.py --plot grid assets
# while running the trusted training-free p* path (currently VLM-only):
conda run -n superdiff python scripts/measure_composability_gap.py \
    --pstar-source all --regime small --seeds 0 1 2 42 \
    --grid-seed 42
"""

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Trajectory MDS will be skipped.")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from notebooks.utils import get_sd3_models, get_sd3_text_embedding
from notebooks.composition_experiments import (
    LatentTrajectoryCollector,
    sample_sd3_with_trajectory_tracking,
    get_vel_sd3,
)
from scripts.trajectory_dynamics_experiment import (
    superdiff_fm_ode_sd3,
    poe_sd3_with_trajectory_tracking,
    project_trajectories,
)
from models.sd35_inverter import load_inverter, make_clip_preprocessor

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not installed. LPIPS metric will be skipped.")
    print("  Install with: pip install lpips")


# ---------------------------------------------------------------------------
# Test pairs (held out from training)
# ---------------------------------------------------------------------------

# Core 4 pairs — fixed set used for trajectory/decoded-image grid figures (seed 42)
# _PAIRS_CORE = [
#     ("a cat",    "a dog"),
#     ("a man with black hair and a black shirt", "a red umbrella"),
#     ("a bird", "a book"),
#     ("a red bmw",    "a white canopy truck"),
# ]

# _PAIRS_CORE = [
#     ("an blue fat striped cat",    "a green skinny cat"),
#     ("a man with black hair and a black shirt", "a hat with a nike logo on it"),
#     ("a beautiful bird on a tree", "a bird with a short blue beak"),
#     ("a red bmw parked on the road",    "a car with green tyres"),
# ]

_PAIRS_CORE = [
    ("a butterfly", "a flower meadow"),
    ("a dog", "oil painting style"),
    ("a desk lamp", "a glacier"),
    ("a cat", "a dog"),
]


# Large-regime extensions.
# Two balanced groups:
#  - complementary: usually co-operative blends (distinct entities or additive attributes)
#  - competing: intentionally conflicting concepts to probe failure modes
_PAIRS_EXTENDED_COMPLEMENTARY = [
    ("a red sports car", "a car with green tyres"),
    ("a glass teapot", "a teapot with golden handles"),
    ("a blue backpack", "a backpack with orange zippers"),
    ("a white ceramic mug", "a mug with blue stripes"),
    ("a wooden chair", "a chair with metal legs"),
    ("a silver bicycle", "a bicycle with a wicker basket"),
    ("a violin", "a black music stand"),
    ("a chef in a white coat", "a stainless steel stove"),
    ("a lighthouse", "an ocean with stormy waves"),
    ("a fox", "a snow-covered pine forest"),
]

_PAIRS_EXTENDED_COMPETING = [
    ("a red car", "a blue car"),
    ("a green apple", "a red apple"),
    ("a wooden table", "a glass table"),
    ("a sunny beach", "a snowy mountain"),
    ("a desert at noon", "a heavy rainstorm"),
    ("a city street at night", "a city street in bright daylight"),
    ("a person facing left", "a person facing right"),
    ("a brand-new building", "a ruined building"),
    ("a calm lake", "a rough ocean"),
    ("a flying airplane", "a parked airplane"),
]

_PAIRS_EXTENDED = _PAIRS_EXTENDED_COMPLEMENTARY + _PAIRS_EXTENDED_COMPETING

assert len(_PAIRS_CORE) == 4, f"Expected 4 core pairs, got {len(_PAIRS_CORE)}"
assert len(_PAIRS_EXTENDED) == 20, f"Expected 20 extended pairs, got {len(_PAIRS_EXTENDED)}"

_PAIRS_CORE_SET = set(_PAIRS_CORE)
_PAIRS_EXTENDED_COMPLEMENTARY_SET = set(_PAIRS_EXTENDED_COMPLEMENTARY)
_PAIRS_EXTENDED_COMPETING_SET = set(_PAIRS_EXTENDED_COMPETING)

# Pair-specific naturalized monolithic prompts for fair SD3.5 baseline comparison.
_PAIR_MONO_NATURAL_PROMPT = {
    ("a cat", "an owl"): "a cat and an owl",
    ("a bird", "a book"): "a bird and a book",
    ("a man with black hair and black shirt", "a red umbrella"):
        "a man with black hair and black shirt holding a red umbrella",
    ("a red bmw", "a white canopy truck"): "a red bmw and a white canopy truck",
    ("a red sports car", "a car with green tyres"): "a red sports car with green tyres",
    ("a glass teapot", "a teapot with golden handles"): "a glass teapot with golden handles",
    ("a blue backpack", "a backpack with orange zippers"): "a blue backpack with orange zippers",
    ("a white ceramic mug", "a mug with blue stripes"): "a white ceramic mug with blue stripes",
    ("a wooden chair", "a chair with metal legs"): "a wooden chair with metal legs",
    ("a silver bicycle", "a bicycle with a wicker basket"): "a silver bicycle with a wicker basket",
    ("a violin", "a black music stand"): "a violin on a black music stand",
    ("a chef in a white coat", "a stainless steel stove"):
        "a chef in a white coat standing at a stainless steel stove",
    ("a lighthouse", "an ocean with stormy waves"): "a lighthouse by an ocean with stormy waves",
    ("a fox", "a snow-covered pine forest"): "a fox in a snow-covered pine forest",
    ("a red car", "a blue car"): "a car that is both red and blue",
    ("a green apple", "a red apple"): "an apple that is both green and red",
    ("a wooden table", "a glass table"): "a table made of both wood and glass",
    ("a sunny beach", "a snowy mountain"): "a scene that is both a sunny beach and a snowy mountain",
    ("a desert at noon", "a heavy rainstorm"): "a desert at noon during a heavy rainstorm",
    ("a city street at night", "a city street in bright daylight"):
        "a city street that is both at night and in bright daylight",
    ("a person facing left", "a person facing right"): "a person facing both left and right",
    ("a brand-new building", "a ruined building"): "a building that is both brand-new and ruined",
    ("a calm lake", "a rough ocean"): "water that is both a calm lake and a rough ocean",
    ("a flying airplane", "a parked airplane"): "an airplane that is both flying and parked",
}

assert len(_PAIR_MONO_NATURAL_PROMPT) == 24, (
    f"Expected natural prompts for 24 pairs, got {len(_PAIR_MONO_NATURAL_PROMPT)}"
)


def _pair_group(c1: str, c2: str) -> str:
    pair = (c1, c2)
    if pair in _PAIRS_EXTENDED_COMPLEMENTARY_SET:
        return "complementary"
    if pair in _PAIRS_EXTENDED_COMPETING_SET:
        return "competing"
    if pair in _PAIRS_CORE_SET:
        return "core"
    return "other"


def _pair_monolithic_prompts(c1: str, c2: str) -> tuple:
    naive = f"{c1} and {c2}"
    natural = _PAIR_MONO_NATURAL_PROMPT.get((c1, c2), naive)
    return naive, natural


REGIME_PAIRS = {
    "small":  _PAIRS_CORE,
    "medium": _PAIRS_CORE,
    "large":  _PAIRS_CORE + _PAIRS_EXTENDED,   # 24 pairs total
}

REGIME_SEEDS = {
    # "small":  list(range(8)),    #  4 pairs ×  8 seeds =  32 records
    # "small":  list(range(4)),    #  4 pairs ×  4 seeds =  16 records
    "small":  [42] + list(range(1,4)),    #  4 pairs ×  4 seeds =  16 records
    "medium": list(range(16)),   #  4 pairs × 16 seeds =  64 records  (default)
    "large":  [42] + list(range(1,24)),   # 24 pairs × 24 seeds = 576 records
}

# Keep for backward compatibility (used when --regime is not set)
TEST_PAIRS = _PAIRS_CORE


# ---------------------------------------------------------------------------
# SD3.5 sampling with pre-computed conditioning
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_sd3_with_precomputed_cond(
    latents: torch.Tensor,
    cond_embeds: torch.Tensor,
    cond_pooled: torch.Tensor,
    uncond_embeds: torch.Tensor,
    uncond_pooled: torch.Tensor,
    scheduler,
    transformer,
    guidance_scale: float = 4.5,
    num_inference_steps: int = 50,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
) -> tuple:
    """
    CFG sampling using pre-computed conditioning embeddings.
    Returns (final_latents, LatentTrajectoryCollector).
    """
    B = latents.shape[0]
    tracker = LatentTrajectoryCollector(
        num_inference_steps, B,
        latents.shape[1], latents.shape[2], latents.shape[3],
    )

    # Expand pre-computed conditioning to batch size
    c_embeds = cond_embeds.expand(B, -1, -1).to(device=device, dtype=dtype)
    c_pooled = cond_pooled.expand(B, -1).to(device=device, dtype=dtype)
    u_embeds = uncond_embeds.expand(B, -1, -1).to(device=device, dtype=dtype)
    u_pooled = uncond_pooled.expand(B, -1).to(device=device, dtype=dtype)

    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(scheduler.timesteps):
        sigma = scheduler.sigmas[i]

        vel_cond  = get_vel_sd3(transformer, t, latents, c_embeds, c_pooled,
                                device=device, dtype=dtype)
        vel_uncond = get_vel_sd3(transformer, t, latents, u_embeds, u_pooled,
                                  device=device, dtype=dtype)

        vf = vel_uncond + guidance_scale * (vel_cond - vel_uncond)
        dt = scheduler.sigmas[i + 1] - sigma
        tracker.store_step(i, latents, vf, float(sigma), t.item())
        latents = latents + dt * vf

    tracker.store_final(latents)
    return latents, tracker


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    latents = latents.to(dtype=vae.dtype)
    shift_factor = getattr(vae.config, "shift_factor", 0.0)
    images = vae.decode(latents / vae.config.scaling_factor + shift_factor, return_dict=False)[0]
    return ((images / 2 + 0.5).clamp(0, 1)).float()


_CLIP_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                         [0.26862954, 0.26130258, 0.27577711]),
])


def images_to_clip_input(images_01: torch.Tensor) -> torch.Tensor:
    """Convert (B, 3, H, W) [0,1] float to CLIP-preprocessed (B, 3, 224, 224)."""
    result = []
    for img in images_01.cpu():
        pil = transforms.ToPILImage()(img)
        result.append(_CLIP_EVAL_TRANSFORM(pil))
    return torch.stack(result)


def _clip_feature_tensor(output, feature_kind: str) -> torch.Tensor:
    """
    Unwrap CLIP feature outputs across transformers versions.

    Older versions return a raw projected feature tensor. Newer versions can
    return a BaseModelOutputWithPooling whose `pooler_output` has already been
    replaced with the projected feature vector.
    """
    if torch.is_tensor(output):
        return output

    if feature_kind == "image":
        attr_names = ("image_embeds", "pooler_output")
    elif feature_kind == "text":
        attr_names = ("text_embeds", "pooler_output")
    else:
        attr_names = ("pooler_output",)

    for attr_name in attr_names:
        value = getattr(output, attr_name, None)
        if torch.is_tensor(value):
            return value

    if isinstance(output, (tuple, list)):
        # Prefer the first rank-2 tensor, which matches projected feature shape.
        for value in output:
            if torch.is_tensor(value) and value.ndim == 2:
                return value
        for value in output:
            if torch.is_tensor(value):
                return value

    raise TypeError(
        f"Unsupported CLIP {feature_kind} feature output type: {type(output).__name__}"
    )


def _clip_image_features(clip_model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Return projected CLIP image features as a float tensor."""
    output = clip_model.get_image_features(pixel_values=pixel_values)
    return _clip_feature_tensor(output, feature_kind="image").float()


def _clip_text_features(clip_model, **tokens) -> torch.Tensor:
    """Return projected CLIP text features as a float tensor."""
    output = clip_model.get_text_features(**tokens)
    return _clip_feature_tensor(output, feature_kind="text").float()


# ---------------------------------------------------------------------------
# Gap metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_image_gap(
    images_a: torch.Tensor,
    images_b: torch.Tensor,
    clip_model,
    lpips_fn=None,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """
    images_a, images_b: (N, 3, H, W) float32 [0, 1]
    Returns dict with clip_cos, lpips (if available).
    """
    a_clip = images_to_clip_input(images_a).to(device)
    b_clip = images_to_clip_input(images_b).to(device)

    feat_a = _clip_image_features(clip_model, a_clip)
    feat_b = _clip_image_features(clip_model, b_clip)

    feat_a = F.normalize(feat_a, dim=-1)
    feat_b = F.normalize(feat_b, dim=-1)

    clip_cos = (feat_a * feat_b).sum(dim=-1).mean().item()

    result = {"clip_cos": clip_cos}

    if lpips_fn is not None:
        # lpips expects [-1, 1]; kept on CPU to avoid OOM alongside the transformer
        a_lp = images_a.cpu().float() * 2 - 1
        b_lp = images_b.cpu().float() * 2 - 1
        lp_vals = lpips_fn(a_lp, b_lp)
        result["lpips"] = lp_vals.mean().item()

    return result


@torch.no_grad()
def compute_latent_gap(
    vae,
    images_a: torch.Tensor,
    images_b: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    latent_batch_size: int = 4,
) -> dict:
    """MSE in VAE latent space between corresponding images.

    Encodes in mini-batches to avoid OOM in large-regime runs.
    """
    n = int(images_a.shape[0])
    if n != int(images_b.shape[0]):
        raise ValueError(
            f"compute_latent_gap expects same batch size, got {images_a.shape[0]} and {images_b.shape[0]}"
        )

    # Optional env override for quick troubleshooting without code edits.
    env_bs = os.environ.get("LATENT_GAP_BATCH_SIZE")
    if env_bs:
        try:
            latent_batch_size = int(env_bs)
        except ValueError:
            pass
    latent_batch_size = max(1, int(latent_batch_size))

    total_sqerr = 0.0
    total_count = 0

    for i in range(0, n, latent_batch_size):
        a = images_a[i:i + latent_batch_size].to(device=device, dtype=dtype) * 2 - 1
        b = images_b[i:i + latent_batch_size].to(device=device, dtype=dtype) * 2 - 1

        lat_a = vae.encode(a).latent_dist.mean.float()
        lat_b = vae.encode(b).latent_dist.mean.float()

        diff = (lat_a - lat_b).pow(2)
        total_sqerr += diff.sum().item()
        total_count += diff.numel()

        del a, b, lat_a, lat_b, diff

    lat_mse = float(total_sqerr / max(total_count, 1))
    return {"lat_mse": lat_mse}


def compute_trajectory_gap(
    tracker_a: LatentTrajectoryCollector,
    tracker_b: LatentTrajectoryCollector,
) -> dict:
    """
    Step-wise MSE and cosine similarity between two trajectory collectors.
    Both must have the same number of steps and batch size.
    Returns: {traj_mse_mean, traj_cos_mean, traj_mse_per_step, traj_cos_per_step}
    """
    T = tracker_a.trajectories.shape[0] - 1  # exclude final stored step
    mse_per_step = []
    cos_per_step = []

    for t in range(T):
        a_t = tracker_a.trajectories[t].flatten(1)  # (B, D)
        b_t = tracker_b.trajectories[t].flatten(1)

        mse = F.mse_loss(a_t, b_t).item()
        cos = F.cosine_similarity(a_t, b_t, dim=-1).mean().item()

        mse_per_step.append(mse)
        cos_per_step.append(cos)

    return {
        "traj_mse_mean": float(sum(mse_per_step) / len(mse_per_step)),
        "traj_cos_mean": float(sum(cos_per_step) / len(cos_per_step)),
        "traj_mse_per_step": [float(v) for v in mse_per_step],
        "traj_cos_per_step": [float(v) for v in cos_per_step],
    }


# ---------------------------------------------------------------------------
# Best-of-K candidate selection
# ---------------------------------------------------------------------------

@torch.no_grad()
def select_best_p_star(
    candidates: list,
    x_T: torch.Tensor,
    target_img: torch.Tensor,
    transformer,
    scheduler,
    uncond_embeds: torch.Tensor,
    uncond_pooled: torch.Tensor,
    guidance_scale: float,
    vae,
    clip_eval,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    """
    Pick the best p* from K candidates using a single-NFE x₀ prediction.

    Formula:  x₀_pred = x_T - σ_max · v_cfg(x_T, t_max, p*_k)

    This is the direct clean-image prediction at the first (noisiest)
    timestep — one transformer call per candidate, no ODE solve.
    The winner is the candidate whose predicted x₀ has the highest
    CLIP cosine similarity to the reference AND image.

    Scheduler must already have set_timesteps called (e.g. by the AND
    run that preceded this call) — we read sigmas[0] without resetting.

    Parameters
    ----------
    candidates  : list of k (pooled: Tensor(1,2048), seq: Tensor(1,410,4096))
    x_T         : (1, 16, H, W) — shared initial noise for this seed
    target_img  : (1, 3, H, W) float32 [0,1] — AND image for this seed
    ...

    Returns
    -------
    (best_pooled, best_seq, best_score)
    best_score is None when len(candidates)==1.
    """
    if len(candidates) == 1:
        return candidates[0][0], candidates[0][1], None

    # Timestep / sigma at t_max — read from already-configured scheduler
    t_max     = scheduler.timesteps[0]
    sigma_max = float(scheduler.sigmas[0])   # ≈ 1.0 for flow matching

    # Unconditional velocity at x_T — same for all candidates, compute once
    vel_uncond = get_vel_sd3(
        transformer, t_max, x_T,
        uncond_embeds.expand(1, -1, -1),
        uncond_pooled.expand(1, -1),
        device=device, dtype=dtype,
    )

    # CLIP features of the AND target image — computed once
    target_clip_in = images_to_clip_input(target_img.cpu()).to(device)
    feat_target = F.normalize(
        _clip_image_features(clip_eval, target_clip_in), dim=-1
    )  # (1, D_clip)

    best_score  = -float("inf")
    best_pooled, best_seq = candidates[0]

    for pooled_k, seq_k in candidates:
        c_embeds = seq_k.to(device=device, dtype=dtype)
        c_pooled = pooled_k.to(device=device, dtype=dtype)

        vel_cond = get_vel_sd3(
            transformer, t_max, x_T, c_embeds, c_pooled,
            device=device, dtype=dtype,
        )
        vf = vel_uncond + guidance_scale * (vel_cond - vel_uncond)

        # Direct x₀ prediction (flow-matching: x₀ = x_T - σ · v)
        x0_pred = x_T - sigma_max * vf

        img_pred = decode_latents(vae, x0_pred)
        approx_clip = images_to_clip_input(img_pred.cpu()).to(device)
        feat_approx = F.normalize(
            _clip_image_features(clip_eval, approx_clip), dim=-1
        )

        score = (feat_target * feat_approx).sum().item()
        if score > best_score:
            best_score  = score
            best_pooled, best_seq = pooled_k, seq_k

    return best_pooled, best_seq, best_score


# ---------------------------------------------------------------------------
# 1. Text decoding: find nearest training prompt to predicted p*
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_pooled_to_text(
    pred_pooled: torch.Tensor,          # (2048,) — averaged across seeds
    vocab_prompts: List[str],
    models: dict,
    device: torch.device,
    top_k: int = 3,
) -> List[tuple]:
    """
    Find the top-k nearest training prompts to the predicted pooled embedding
    by cosine similarity in SD3.5's pooled conditioning space (2048-dim).

    Returns list of (prompt_str, cosine_similarity) sorted descending.
    """
    query = F.normalize(pred_pooled.float().unsqueeze(0), dim=-1)  # (1, 2048)

    sims = []
    for prompt in vocab_prompts:
        _, pooled = get_sd3_text_embedding(
            [prompt],
            models["tokenizer"],   models["text_encoder"],
            models["tokenizer_2"], models["text_encoder_2"],
            models["tokenizer_3"], models["text_encoder_3"],
            device=device,
        )
        pooled_n = F.normalize(pooled.float(), dim=-1)  # (1, 2048)
        sim = (query * pooled_n).sum().item()
        sims.append((prompt, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


# ---------------------------------------------------------------------------
# 2. Side-by-side comparison grid
# ---------------------------------------------------------------------------

def _load_pil_font(font_size: int, bold: bool = False):
    """Load a scalable PIL font with graceful fallback."""
    candidates = []
    if bold:
        candidates.extend(["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arialbd.ttf"])
    candidates.extend(["DejaVuSans.ttf", "Arial.ttf"])

    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, size=font_size)
        except OSError:
            continue

    # Pillow fallback (small bitmap font if scalable fonts are unavailable).
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
        return ImageFont.load_default()


def _text_wh(draw: ImageDraw.ImageDraw, text: str, font) -> tuple:
    """Return rendered text (width, height) across Pillow versions."""
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    return draw.textsize(text, font=font)


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list:
    """Greedy word-wrap for PIL text drawing."""
    words = text.split()
    if not words:
        return [text]

    lines = [words[0]]
    for word in words[1:]:
        trial = f"{lines[-1]} {word}"
        w, _ = _text_wh(draw, trial, font)
        if w <= max_width:
            lines[-1] = trial
        else:
            lines.append(word)
    return lines


def _add_text_label(img_tensor: torch.Tensor, label: str, font_size: int = 36) -> torch.Tensor:
    """Burn a text label into the top of a (3, H, W) [0,1] float tensor."""
    pil = transforms.ToPILImage()(img_tensor.clamp(0, 1))
    draw = ImageDraw.Draw(pil)
    font = _load_pil_font(font_size, bold=True)

    x_pad = 8
    y_pad = max(6, int(0.28 * font_size))
    line_gap = max(2, int(0.12 * font_size))
    max_text_w = max(16, pil.width - 2 * x_pad)

    lines = _wrap_text(draw, label, font, max_text_w)
    line_heights = [_text_wh(draw, line, font)[1] for line in lines]
    text_h = sum(line_heights) + line_gap * (len(lines) - 1)
    strip_h = text_h + 2 * y_pad

    draw.rectangle([0, 0, pil.width, strip_h], fill=(20, 20, 20))
    y = y_pad
    for line, h in zip(lines, line_heights):
        draw.text((x_pad, y), line, fill=(255, 255, 255), font=font)
        y += h + line_gap

    return transforms.ToTensor()(pil)


def plot_comparison_grid(
    imgs_and:   torch.Tensor,   # (N, 3, H, W) [0,1]
    imgs_pstar: torch.Tensor,   # (N, 3, H, W)
    imgs_mono:  torch.Tensor,   # (N, 3, H, W)
    decoded_text: List[tuple],  # top-k (prompt, sim) from decode_pooled_to_text
    c1: str,
    c2: str,
    out_path: Path,
    n_display: int = 4,
    mono_prompt: str = "",
):
    """
    Save a 3-row comparison grid:
      Row 0 — SuperDiff AND
      Row 1 — SD3.5(p*)  [with decoded nearest prompts annotated below]
      Row 2 — SD3.5 monolithic

    Each row shows n_display images side by side.
    Decoded text is printed below the p* row as a caption block.
    """
    N = min(n_display, imgs_and.shape[0])
    mono_prompt_label = mono_prompt or f"{c1} and {c2}"

    row_label_font_size = 36

    def label_row(imgs, label):
        labelled = [_add_text_label(imgs[i], label, font_size=row_label_font_size) for i in range(N)]
        return torch.stack(labelled)   # (N, 3, H, W)

    row_and   = label_row(imgs_and,   f"SuperDiff AND  ({c1}  +  {c2})")
    row_pstar = label_row(imgs_pstar, "SD3.5 ( p* — inverted )")
    row_mono  = label_row(imgs_mono,  f"SD3.5 monolithic  \"{mono_prompt_label}\"")

    combined = torch.cat([row_and, row_pstar, row_mono], dim=0)  # (3N, 3, H, W)
    grid = make_grid(combined, nrow=N, padding=4, normalize=False)

    # Convert to PIL to add a caption strip at the bottom
    grid_pil = transforms.ToPILImage()(grid)
    caption_font_size = 26
    caption_font = _load_pil_font(caption_font_size, bold=False)
    caption_lines = ["Nearest training prompts to p* (cosine similarity in SD3.5 pooled space):"]
    for rank, (prompt, sim) in enumerate(decoded_text, 1):
        caption_lines.append(f"  {rank}. \"{prompt}\"  —  {sim:.4f}")

    measure_img = Image.new("RGB", (1, 1))
    measure_draw = ImageDraw.Draw(measure_img)
    wrapped_caption_lines = []
    for line in caption_lines:
        wrapped_caption_lines.extend(
            _wrap_text(measure_draw, line, caption_font, max(16, grid_pil.width - 16))
        )

    _, line_h = _text_wh(measure_draw, "Ag", caption_font)
    line_gap = max(2, int(0.18 * caption_font_size))
    line_step = line_h + line_gap
    caption_h = line_step * len(wrapped_caption_lines) + 10
    canvas = Image.new("RGB", (grid_pil.width, grid_pil.height + caption_h), color=(30, 30, 30))
    canvas.paste(grid_pil, (0, 0))

    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(wrapped_caption_lines):
        draw.text(
            (8, grid_pil.height + 5 + i * line_step),
            line,
            fill=(220, 220, 220),
            font=caption_font,
        )

    canvas.save(str(out_path))


# ---------------------------------------------------------------------------
# Grid manifest helpers
# ---------------------------------------------------------------------------

def _write_manifest(path: Path, lines: list) -> None:
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


def save_comparison_grid_manifest(
    out_path: Path,
    c1: str,
    c2: str,
    seeds: list,
    pstar_source: str,
    n_display: int = 4,
    pstar_prompts: list = None,
    mono_prompt: str = "",
) -> None:
    """Write a _manifest.txt for comparison_grid.png.

    Layout: 3 rows × min(n_display, len(seeds)) columns
      Row 0  — SuperDiff AND
      Row 1  — SD3.5 (p*)
      Row 2  — SD3.5 monolithic
    Each column corresponds to a seed (left-to-right = seeds[0..n_display-1]).
    """
    from datetime import datetime

    n_cols = min(n_display, len(seeds))
    manifest_path = out_path.with_name(out_path.stem + "_manifest.txt")
    mono_prompt_label = mono_prompt or f"{c1} and {c2}"

    row_info = [
        ("Row 0", "SuperDiff AND",        f'"{c1}" ∧ "{c2}"'),
        ("Row 1", f"SD3.5 (p* — {pstar_source})", None),
        ("Row 2", "SD3.5 monolithic",     f'"{mono_prompt_label}"'),
    ]

    lines = [
        f"Grid manifest for : {out_path.name}",
        f"Generated         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Layout            : 3 rows × {n_cols} columns",
        f"Concept pair      : \"{c1}\" AND \"{c2}\"",
        f"p* source         : {pstar_source}",
        f"Seeds (col order) : {seeds[:n_cols]}",
        "",
    ]

    for row_idx, (row_tag, row_label, row_prompt) in enumerate(row_info):
        lines.append(f"{row_tag} — {row_label}")
        if row_prompt:
            lines.append(f"  Prompt  : {row_prompt}")
        for col, seed in enumerate(seeds[:n_cols]):
            seed_note = ""
            if row_idx == 1 and pstar_prompts and col < len(pstar_prompts):
                seed_note = f"  →  p* prompt: \"{pstar_prompts[col]}\""
            lines.append(f"  Col {col}   : seed {seed}{seed_note}")
        lines.append("")

    _write_manifest(manifest_path, lines)


def save_single_grid_manifest(
    out_path: Path,
    row_label: str,
    seeds: list,
    nrow: int = 4,
    per_seed_prompts: list = None,
) -> None:
    """Write a _manifest.txt for a single-condition save_image grid.

    torchvision.save_image with nrow=4 arranges N images as ceil(N/4) rows × 4 cols.
    Each cell (row r, col c) = image index r*nrow + c = seeds[r*nrow + c].
    """
    from datetime import datetime
    import math

    n = len(seeds)
    n_rows = math.ceil(n / nrow)
    n_cols = min(n, nrow)
    manifest_path = out_path.with_name(out_path.stem + "_manifest.txt")

    lines = [
        f"Grid manifest for : {out_path.name}",
        f"Generated         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Layout            : {n_rows} row(s) × {n_cols} columns  (nrow={nrow})",
        f"Condition         : {row_label}",
        f"Seeds             : {seeds}",
        "",
    ]

    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * nrow + c
            if idx >= n:
                break
            seed = seeds[idx]
            prompt_note = ""
            if per_seed_prompts and idx < len(per_seed_prompts):
                prompt_note = f"  →  prompt: \"{per_seed_prompts[idx]}\""
            lines.append(f"Row {r}, Col {c}  :  seed {seed}{prompt_note}")
        lines.append("")

    _write_manifest(manifest_path, lines)


# ---------------------------------------------------------------------------
# Grid-asset export for plot_gap_analysis.py (plots 27–28)
# ---------------------------------------------------------------------------

_GRID_EXPORT_ORDER = [
    "prompt_a",
    "prompt_b",
    "monolithic",
    "monolithic_naive",
    "monolithic_natural",
    "poe",
    "superdiff_fm_ode",
    "pstar_vlm",
    "pstar_z2t",
]

_GRID_EXPORT_LABELS = {
    "prompt_a":         "SD3.5 A",
    "prompt_b":         "SD3.5 B",
    "monolithic":       "SD3.5 A∧B",
    "monolithic_naive": "SD3.5 A∧B (naive)",
    "monolithic_natural": "SD3.5 A∧B (natural)",
    "poe":              "PoE A×B",
    "superdiff_fm_ode": "SuperDiff A∧B",
    "pstar_vlm":        "p* VLM",
    "pstar_z2t":        "p* Z2T",
}


def export_pair_grid_assets(
    pair_dir: Path,
    c1: str,
    c2: str,
    seed: int,
    decoded_images: dict,
    trackers: dict,
    pair_index: int = None,
    projection_method: str = "mds",
    source_prompts: dict = None,
) -> None:
    """
    Export per-pair assets consumed by plot_gap_analysis.py plots 27–28:
      - single-seed decoded images per condition
      - 2D trajectory projection per condition (seed-matched)
      - minimal metadata (pair labels, prompt key, source prompts)
    """
    assets_dir = pair_dir / "grid_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_json = pair_dir / "grid_assets.json"

    existing = {}
    if out_json.exists():
        try:
            existing = json.loads(out_json.read_text())
        except Exception:
            existing = {}

    image_paths = dict(existing.get("decoded_image_paths", {}))
    for cond in _GRID_EXPORT_ORDER:
        img = decoded_images.get(cond)
        if img is None:
            continue
        out_img = assets_dir / f"{cond}.png"
        save_image(img, out_img, nrow=1, normalize=False)
        image_paths[cond] = str(out_img.relative_to(pair_dir))

    traj_payload = None
    trajectory_flat_paths = dict(existing.get("trajectory_flat_paths", {}))
    ordered_trackers = {
        cond: trackers[cond]
        for cond in _GRID_EXPORT_ORDER
        if cond in trackers and trackers[cond] is not None
    }
    for cond, tracker in ordered_trackers.items():
        flat = tracker.trajectories[:, 0].reshape(tracker.trajectories.shape[0], -1)
        flat_np = flat.numpy().astype(np.float16)
        out_flat = assets_dir / f"trajectory_flat_{cond}.npy"
        np.save(out_flat, flat_np)
        trajectory_flat_paths[cond] = str(out_flat.relative_to(pair_dir))

    if ordered_trackers:
        try:
            projected, _, n_steps = project_trajectories(
                ordered_trackers, method=projection_method
            )
            traj_payload = {
                "projection_method": projection_method,
                "n_steps": int(n_steps),
                "projected": {
                    cond: projected[cond].tolist() for cond in ordered_trackers.keys()
                },
                "labels": {
                    cond: _GRID_EXPORT_LABELS.get(cond, cond)
                    for cond in ordered_trackers.keys()
                },
            }
        except Exception as exc:
            print(f"  Warning: failed to export trajectory projection for grid assets: {exc}")

    merged_source_prompts = dict(existing.get("source_prompts", {}))
    merged_source_prompts.update(source_prompts or {})

    payload = {
        "pair": [c1, c2],
        "pair_index": (
            int(pair_index) if pair_index is not None else existing.get("pair_index")
        ),
        "seed": int(seed),
        "prompt_key_map": {"A": c1, "B": c2},
        "decoded_image_paths": image_paths,
        "trajectory_flat_paths": trajectory_flat_paths,
        "projection_method": projection_method,
        "trajectory_projection": traj_payload,
        "source_prompts": merged_source_prompts,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {out_json}")


# ---------------------------------------------------------------------------
# 3. Trajectory MDS / PCA visualisation
# ---------------------------------------------------------------------------

def _collect_trajectory_array(tracker: LatentTrajectoryCollector) -> np.ndarray:
    """Flatten trajectories to (T+1, D) numpy array."""
    traj = tracker.trajectories  # (T+1, B, C, H, W)
    T1, B, C, H, W = traj.shape
    return traj[:, 0].reshape(T1, -1).numpy().astype(np.float32)  # (T+1, D)


def plot_trajectory_mds(
    tracker_and:   LatentTrajectoryCollector,
    tracker_pstar: LatentTrajectoryCollector,
    tracker_mono:  LatentTrajectoryCollector,
    c1: str,
    c2: str,
    out_path: Path,
    method: str = "mds",
    pstar_label: str = "SD3.5 (p* — Inverter)",
    tracker_c1: LatentTrajectoryCollector = None,
    tracker_c2: LatentTrajectoryCollector = None,
):
    """
    Jointly project the latent trajectories of:
      - SuperDiff AND
      - SD3.5(p*)
      - SD3.5 monolithic
      - SD3.5(c1) and SD3.5(c2) single-concept baselines (optional)
    into 2D via PCA or MDS and plot time-coloured curves.

    All three start from the same x_T (shared noise), so the origin is common.

    Parameters
    ----------
    pstar_label : str
        Human-readable label for the p* source, e.g. "SD3.5 (p* — VLM/BLIP-2)",
        "SD3.5 (p* — PEZ)", "SD3.5 (p* — Z2T)", "SD3.5 (p* — Inverter)".
    """
    if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
        print("  [skip] trajectory MDS — matplotlib or scikit-learn not available")
        return

    traj_specs = [
        ("SuperDiff AND", tracker_and),
        (pstar_label, tracker_pstar),
        (f'SD3.5 monolithic "{c1} and {c2}"', tracker_mono),
    ]
    if tracker_c1 is not None:
        traj_specs.append((f'SD3.5 solo "{c1}"', tracker_c1))
    if tracker_c2 is not None:
        traj_specs.append((f'SD3.5 solo "{c2}"', tracker_c2))

    traj_arrays = [_collect_trajectory_array(tk) for _, tk in traj_specs]
    T1 = traj_arrays[0].shape[0]
    if any(arr.shape[0] != T1 for arr in traj_arrays[1:]):
        print("  [skip] trajectory MDS — trajectory length mismatch")
        return

    stacked = np.vstack(traj_arrays)  # (N_curves*(T+1), D)

    # Dimensionality reduction
    if method == "pca":
        proj = PCA(n_components=2).fit_transform(stacked)
        axis_label = "PC"
    else:  # mds
        from sklearn.metrics import pairwise_distances
        dist = pairwise_distances(stacked, metric="euclidean")
        mds = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=42, normalized_stress="auto")
        proj = mds.fit_transform(dist)
        axis_label = "MDS"

    proj_specs = []
    start = 0
    for (label, _), arr in zip(traj_specs, traj_arrays):
        end = start + arr.shape[0]
        proj_specs.append((label, proj[start:end]))
        start = end

    fig, ax = plt.subplots(figsize=(14, 11))
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=T1 - 1)
    line_styles = ["-", "--", "-.", ":", "--", "-."]
    end_markers = ["o", "s", "^", "D", "P", "X"]

    def draw_curve(pts, linestyle="-", marker="o"):
        for t in range(len(pts) - 1):
            seg = np.array([[pts[t], pts[t + 1]]])
            lc = LineCollection(
                seg,
                colors=[cmap(norm(t))],
                linewidths=2.5,
                linestyles=linestyle,
                alpha=0.85,
            )
            ax.add_collection(lc)
        ax.scatter(
            *pts[-1], s=180, zorder=5, marker=marker,
            edgecolors="k", linewidths=1.0, color=cmap(norm(T1 - 1)),
        )

    curve_meta = []
    for idx, (label, pts) in enumerate(proj_specs):
        linestyle = line_styles[idx % len(line_styles)]
        marker = end_markers[idx % len(end_markers)]
        draw_curve(pts, linestyle=linestyle, marker=marker)
        symbol = chr(ord("A") + idx)
        curve_meta.append((label, pts[-1], linestyle, marker, symbol))

    # Terminal-state symbols (A, B, C, ...) are drawn directly on endpoints.
    # Full labels are kept in the legend to avoid clutter from callout boxes/arrows.
    for _, pt, _, _, symbol in curve_meta:
        ax.text(
            pt[0], pt[1], symbol,
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="black", zorder=8,
        )

    # Shared origin marker (all start from same x_T)
    shared_start = proj_specs[0][1][0]
    ax.plot(*shared_start, "ko", markersize=11, zorder=6)
    ax.annotate("$x_T$ (shared)", shared_start, xytext=(10, -18),
                textcoords="offset points", fontsize=10, color="black")

    # Colourbar for time
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Denoising step", pad=0.02)

    ax.autoscale()
    ax.margins(0.15)   # extra breathing room so labels don't clip
    ax.set_xlabel(f"{axis_label} 1", fontsize=13)
    ax.set_ylabel(f"{axis_label} 2", fontsize=13)
    include_singles = tracker_c1 is not None or tracker_c2 is not None
    suffix = " + single concepts" if include_singles else ""
    ax.set_title(
        f"Latent Trajectory: SuperDiff AND  vs  {pstar_label}  vs  Monolithic{suffix}\n"
        f"({c1}  +  {c2})",
        fontsize=13,
    )
    ax.grid(True, alpha=0.25)

    # Custom legend (same time-colour scale; identity encoded by line style/marker)
    from matplotlib.lines import Line2D
    legend_color = cmap(norm(T1 - 1))
    legend_elements = []
    for lbl, _, linestyle, marker, symbol in curve_meta:
        legend_elements.append(
            Line2D(
                [0], [0],
                color=legend_color,
                lw=2.5,
                linestyle=linestyle,
                marker=marker,
                markersize=7,
                markerfacecolor=legend_color,
                markeredgecolor="k",
                label=f"{symbol}: {lbl}",
            )
        )
    legend_elements.append(
        Line2D([0], [0], marker="o", color="k", lw=0, markersize=9, label="Shared start $x_T$")
    )
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=10,
        framealpha=0.9,
        edgecolor="0.7",
    )

    plt.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# PEZ: gradient-based hard prompt recovery (Wen et al., NeurIPS 2023)
# ---------------------------------------------------------------------------
# Optimises continuous token embeddings against CLIP-L image similarity,
# using a straight-through estimator (STE) to project to nearest vocab tokens
# each step.  After convergence the discrete ids are decoded to a text string,
# which is then re-encoded with SD3.5's full triple text encoder.
# ---------------------------------------------------------------------------

def _clip_encode_soft_embeds(
    clip_model,
    soft_embeds: torch.Tensor,   # (n_tokens, D_tok) — float32, on device
    device: torch.device,
) -> torch.Tensor:
    """
    Run CLIP text encoder with injected soft token embeddings (bypass lookup).
    Sequence layout:  [BOS] [soft × n_tokens] [EOS] [PAD × …]
    Returns normalised (1, D_feat) pooled text feature.
    """
    text_model = clip_model.text_model
    emb_layer  = text_model.embeddings
    cfg        = text_model.config
    max_len    = cfg.max_position_embeddings   # 77

    n_tok = min(soft_embeds.shape[0], max_len - 2)
    n_pad = max_len - n_tok - 2

    bos_emb = emb_layer.token_embedding(
        torch.tensor([cfg.bos_token_id], device=device)).float()      # (1, D_tok)
    eos_emb = emb_layer.token_embedding(
        torch.tensor([cfg.eos_token_id], device=device)).float()      # (1, D_tok)

    parts = [bos_emb, soft_embeds[:n_tok].float()]
    parts.append(eos_emb)
    if n_pad > 0:
        pad_emb = emb_layer.token_embedding(
            torch.zeros(n_pad, dtype=torch.long, device=device)).float()
        parts.append(pad_emb)

    seq = torch.cat(parts, dim=0).unsqueeze(0)   # (1, max_len, D_tok)

    pos_ids = torch.arange(max_len, device=device).unsqueeze(0)
    hidden  = seq + emb_layer.position_embedding(pos_ids).float()

    seq_len = 1 + n_tok + 1
    # Build causal mask manually — _build_causal_attention_mask was removed in
    # transformers >= 4.37.  Standard additive causal mask: upper-triangular -inf,
    # lower-triangular + diagonal = 0, shape (1, 1, max_len, max_len).
    causal_mask = torch.triu(
        torch.full((1, 1, max_len, max_len), float("-inf"),
                   dtype=hidden.dtype, device=device),
        diagonal=1,
    )

    # Padding mask: 0 for real tokens, -inf for padding
    ext_mask = torch.zeros(1, 1, 1, max_len, device=device)
    if n_pad > 0:
        ext_mask[0, 0, 0, seq_len:] = -1e4

    out = text_model.encoder(
        inputs_embeds=hidden,
        attention_mask=ext_mask,
        causal_attention_mask=causal_mask,
    )
    normed = text_model.final_layer_norm(out.last_hidden_state)
    pooled = normed[:, seq_len - 1, :]            # EOS position
    text_feat = clip_model.text_projection(pooled)
    return F.normalize(text_feat, dim=-1)


@torch.no_grad()
def _clip_image_feat(img_01: torch.Tensor, clip_model, device: torch.device) -> torch.Tensor:
    """CLIP-L image feature, normalised. (1, D_feat)."""
    clip_in  = images_to_clip_input(img_01).to(device)
    feat     = _clip_image_features(clip_model, clip_in)
    return F.normalize(feat, dim=-1)


def pez_invert_image(
    img_01:        torch.Tensor,   # (1, 3, H, W) [0,1]
    clip_model,                     # CLIPModel — CLIP-L backbone
    clip_tokenizer,
    n_tokens:  int   = 16,
    n_iters:   int   = 300,
    lr:        float = 0.4,
    device:    torch.device = None,
) -> str:
    """
    PEZ hard-prompt optimisation.
    Optimises `n_tokens` soft embeddings in CLIP-L token-embedding space to
    maximise cosine similarity to the target CLIP image embedding, then decodes
    the projected discrete tokens to a text string.
    """
    device = device or torch.device("cuda")

    img_feat = _clip_image_feat(img_01, clip_model, device)   # (1, D_feat) — no grad

    tok_emb  = clip_model.text_model.embeddings.token_embedding.weight.detach().float()
    # Vocab size may be large — keep on GPU for fast nearest-neighbour
    tok_emb  = tok_emb.to(device)

    # Initialise soft embeds from random tokens
    init_ids = torch.randint(0, tok_emb.shape[0], (n_tokens,), device=device)
    soft     = tok_emb[init_ids].clone().requires_grad_(True)

    optim = torch.optim.Adam([soft], lr=lr)

    for _ in range(n_iters):
        optim.zero_grad()

        # Nearest vocab token for each soft embed (STE: grad flows through soft)
        with torch.no_grad():
            normed_soft  = F.normalize(soft, dim=-1)
            normed_vocab = F.normalize(tok_emb, dim=-1)
            nearest_ids  = (normed_soft @ normed_vocab.T).argmax(dim=-1)
            hard         = tok_emb[nearest_ids]

        ste = soft + (hard - soft).detach()   # forward = hard, backward = soft

        text_feat = _clip_encode_soft_embeds(clip_model, ste, device)
        loss      = -(text_feat * img_feat).sum()
        loss.backward()
        optim.step()

    with torch.no_grad():
        normed_soft  = F.normalize(soft.detach(), dim=-1)
        normed_vocab = F.normalize(tok_emb, dim=-1)
        final_ids    = (normed_soft @ normed_vocab.T).argmax(dim=-1)

    prompt = clip_tokenizer.decode(final_ids.cpu().tolist(), skip_special_tokens=True)
    return prompt.strip() or "a photo"


# ---------------------------------------------------------------------------
# Zero2Text-style: training-free CLIP alignment via ridge regression
# (Kim et al., arXiv 2602.01757, Feb 2026)
# ---------------------------------------------------------------------------
# This implements the core algorithmic idea:
#   1. Build a pool of M candidate prompts (template expansion + pair concepts).
#   2. Compute their CLIP-L text embeddings E ∈ R^{M×D}.
#   3. Ridge regression:  λ* = (EEᵀ + αI)⁻¹ E z*   where z* = CLIP image embed.
#   4. Pseudo-target:     z_align = Eᵀλ* (normalised).
#   5. Pick the candidate nearest to z_align.
#   6. Repeat for n_iters rounds, seeding the template pool with the previous
#      best candidate.
#
# Swap steps 1 & 6 for an actual LLM call when the authors' code is released.
# ---------------------------------------------------------------------------

_Z2T_TEMPLATES = [
    "{c1} and {c2}",
    "a photo of {c1} and {c2}",
    "{c1} next to {c2}",
    "{c1} beside {c2}",
    "{c1} with {c2}",
    "a {c1} and a {c2} together",
    "{c1} and {c2} in the same scene",
    "a scene containing {c1} and {c2}",
    "{c1} alongside {c2}",
    "an image of {c1} and {c2}",
    "realistic photo of {c1} and {c2}",
    "{c1} near {c2}",
    "{c1} and {c2}, photorealistic",
    "a high quality image of {c1} and {c2}",
    "{c1} together with {c2}",
    "a picture of {c1} and {c2}",
    "{c2} and {c1}",
    "photo of {c2} next to {c1}",
    "{c2} beside {c1}",
    "{c2} with {c1}",
]


_Z2T_STOPWORDS = {
    "a", "an", "the", "and", "or", "with", "without", "of", "on", "in", "at",
    "to", "from", "by", "for", "near", "next", "beside", "alongside",
    "photo", "image", "picture", "scene", "realistic", "high", "quality",
    "is", "are", "was", "were", "be", "been",
}


_Z2T_COLOR_TABLE = {
    "white":     (0.95, 0.95, 0.95),
    "black":     (0.05, 0.05, 0.05),
    "gray":      (0.55, 0.55, 0.55),
    "red":       (0.85, 0.20, 0.20),
    "orange":    (0.92, 0.55, 0.20),
    "yellow":    (0.90, 0.85, 0.20),
    "green":     (0.20, 0.70, 0.25),
    "blue":      (0.20, 0.35, 0.90),
    "purple":    (0.58, 0.30, 0.78),
    "pink":      (0.92, 0.60, 0.75),
    "brown":     (0.45, 0.30, 0.20),
    "cyan":      (0.10, 0.75, 0.80),
    "teal":      (0.10, 0.62, 0.58),
    "turquoise": (0.20, 0.80, 0.75),
}


_Z2T_OBJECT_LEXICON = [
    "cat", "dog", "bird", "owl", "horse", "person", "man", "woman", "child",
    "book", "notebook", "newspaper", "magazine",
    "car", "truck", "bus", "train", "airplane", "boat", "bicycle", "motorcycle",
    "umbrella", "hat", "glasses", "shirt", "jacket", "dress", "shoe", "boot",
    "chair", "table", "bench", "bed", "sofa", "lamp", "clock", "mirror",
    "laptop", "computer", "keyboard", "phone", "camera", "television", "monitor",
    "bottle", "cup", "mug", "plate", "bowl", "spoon", "fork", "knife",
    "apple", "banana", "orange", "pizza", "cake", "sandwich",
    "tree", "flower", "plant", "grass", "mountain", "river", "ocean", "beach",
    "road", "bridge", "building", "house", "window", "door", "stairs",
    "backpack", "handbag", "suitcase", "box", "gift",
]


def _z2t_tokenize(prompt: str) -> List[str]:
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", prompt.lower())
    return [tok for tok in cleaned.split() if tok]


def _z2t_subject(prompt: str) -> str:
    toks = _z2t_tokenize(prompt)
    for tok in reversed(toks):
        if tok not in _Z2T_STOPWORDS:
            return tok
    return "object"


def _z2t_attributes(prompt: str, subject: str) -> List[str]:
    toks = _z2t_tokenize(prompt)
    attrs = []
    for tok in toks:
        if tok == subject or tok in _Z2T_STOPWORDS:
            continue
        if len(tok) <= 2 or tok.isdigit():
            continue
        attrs.append(tok)
    # preserve order while deduplicating
    return list(dict.fromkeys(attrs))


def _z2t_dominant_color_name(img_01: torch.Tensor) -> str:
    """Estimate a coarse dominant color from non-background pixels."""
    px = img_01.detach().float().squeeze(0).permute(1, 2, 0).reshape(-1, 3).cpu()
    # Drop near-white background so subject color dominates.
    fg_mask = (px < 0.93).any(dim=1)
    if int(fg_mask.sum()) > 64:
        px = px[fg_mask]
    mean_rgb = px.mean(dim=0)

    best_name = "teal"
    best_d2 = float("inf")
    for name, rgb in _Z2T_COLOR_TABLE.items():
        ref = torch.tensor(rgb, dtype=mean_rgb.dtype)
        d2 = float(((mean_rgb - ref) ** 2).sum())
        if d2 < best_d2:
            best_d2 = d2
            best_name = name
    return best_name


def _z2t_phrase_with_article(noun_phrase: str) -> str:
    """Convert 'owl' -> 'an owl', 'book' -> 'a book'."""
    phrase = noun_phrase.strip()
    if not phrase:
        return "an object"
    first = phrase.split()[0].lower()
    article = "an" if first[:1] in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {phrase}"


@torch.no_grad()
def _z2t_mine_subjects(
    img_01: torch.Tensor,
    clip_model,
    clip_tokenizer,
    device: torch.device,
    top_k: int = 3,
) -> List[str]:
    """
    Image-only subject mining via CLIP zero-shot retrieval over a fixed lexicon.
    """
    prompts = [f"a photo of {_z2t_phrase_with_article(noun)}" for noun in _Z2T_OBJECT_LEXICON]
    tokens = clip_tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=77
    ).to(device)
    text_feats = F.normalize(_clip_text_features(clip_model, **tokens), dim=-1)  # (M, D)
    z_target = _clip_image_feat(img_01, clip_model, device)  # (1, D)

    scores = (text_feats @ z_target.T).squeeze(-1)  # (M,)
    k = min(max(1, top_k), scores.shape[0])
    top_idx = torch.topk(scores, k=k).indices.tolist()

    subjects = []
    for idx in top_idx:
        noun = _Z2T_OBJECT_LEXICON[idx]
        if noun not in subjects:
            subjects.append(noun)
    return subjects or ["object"]


def _z2t_build_candidate_pool_pair_conditioned(
    img_01: torch.Tensor,
    c1: str,
    c2: str,
) -> List[str]:
    """Build a candidate pool with both pairwise and single-subject templates."""
    pool = [t.format(c1=c1, c2=c2) for t in _Z2T_TEMPLATES]

    subj1 = _z2t_subject(c1)
    subj2 = _z2t_subject(c2)
    attrs1 = _z2t_attributes(c1, subj1)
    attrs2 = _z2t_attributes(c2, subj2)
    color = _z2t_dominant_color_name(img_01)

    # Generic image-descriptive candidates (not tied to "c1 and c2").
    pool += [
        "a single subject on a plain white background",
        "a close-up portrait of an animal on white background",
    ]

    # If both prompts refer to the same entity class (e.g., cat + cat), enable
    # blended single-subject candidates to avoid forced two-object phrasing.
    if subj1 == subj2:
        subject = subj1
        merged_attrs = list(dict.fromkeys(attrs1 + attrs2))
        attr_phrase = " ".join(merged_attrs[:4]).strip()

        pool += [
            f"a {color} {subject}",
            f"a realistic photo of a {color} {subject}",
            f"a close-up portrait of a {color} {subject}",
            f"a {subject} with mixed attributes",
            f"a {subject} with blended features",
        ]

        if attr_phrase:
            pool += [
                f"a {attr_phrase} {subject}",
                f"a realistic photo of a {attr_phrase} {subject}",
                f"a {color} {attr_phrase} {subject}",
                f"a {subject} with {attr_phrase} features",
            ]
    else:
        # Keep a few two-subject variants, but include color-aware phrasing.
        pool += [
            f"a {color} scene with {subj1} and {subj2}",
            f"a realistic photo with {subj1} and {subj2}",
        ]

    return list(dict.fromkeys(pool))


def _z2t_build_candidate_pool_image_only(
    img_01: torch.Tensor,
    mined_subjects: List[str],
) -> List[str]:
    """
    Build an image-only candidate pool (no c1/c2 prompt priors).
    """
    color = _z2t_dominant_color_name(img_01)
    subjects = [s.strip() for s in mined_subjects if s.strip()]
    if not subjects:
        subjects = ["object"]

    pool = [
        "a photo",
        "a realistic photo",
        "an image of a scene",
        "a single subject on a plain white background",
        "a close-up photo on a plain background",
    ]

    for subj in subjects:
        subj_phrase = _z2t_phrase_with_article(subj)
        pool += [
            f"{subj_phrase}",
            f"a photo of {subj_phrase}",
            f"a close-up photo of {subj_phrase}",
            f"a realistic photo of {subj_phrase}",
            f"a {color} {subj}",
            f"a photo of a {color} {subj}",
        ]

    if len(subjects) >= 2:
        s1, s2 = subjects[0], subjects[1]
        s1a = _z2t_phrase_with_article(s1)
        s2a = _z2t_phrase_with_article(s2)
        pool += [
            f"{s1a} and {s2a}",
            f"a photo of {s1a} and {s2a}",
            f"{s1a} next to {s2a}",
            f"{s1a} on {s2a}",
            f"{s2a} on {s1a}",
            f"a {color} scene with {s1a} and {s2a}",
        ]

    return list(dict.fromkeys(pool))


@torch.no_grad()
def z2t_invert_image(
    img_01:       torch.Tensor,    # (1, 3, H, W) [0,1]
    clip_model,                     # CLIPModel — CLIP-L
    clip_tokenizer,
    c1:           str = "",
    c2:           str = "",
    pool_mode:    str = "image_only",
    n_iters:      int   = 5,
    ridge_alpha:  float = 0.01,
    device:       torch.device = None,
) -> str:
    """
    Zero2Text-style training-free prompt recovery via recursive ridge regression.
    Returns the best text string from the candidate pool.
    """
    device = device or torch.device("cuda")

    z_target = _clip_image_feat(img_01, clip_model, device)  # (1, D) — target

    def embed_prompts(prompts):
        tokens = clip_tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=77,
        ).to(device)
        feats = _clip_text_features(clip_model, **tokens)
        return F.normalize(feats, dim=-1)   # (M, D)

    if pool_mode not in {"image_only", "pair_conditioned"}:
        raise ValueError(
            f"Unknown Z2T pool mode: {pool_mode!r}. "
            "Expected one of {'image_only', 'pair_conditioned'}."
        )
    if pool_mode == "pair_conditioned" and (not c1.strip() or not c2.strip()):
        raise ValueError("pair_conditioned Z2T mode requires non-empty c1 and c2.")

    mined_subjects = []
    if pool_mode == "image_only":
        mined_subjects = _z2t_mine_subjects(
            img_01, clip_model=clip_model, clip_tokenizer=clip_tokenizer, device=device
        )

    best_prompt = "a photo" if pool_mode == "image_only" else f"{c1} and {c2}"

    for it in range(n_iters):
        # Expand template pool; seed with current best in later iterations.
        if pool_mode == "image_only":
            pool = _z2t_build_candidate_pool_image_only(img_01, mined_subjects=mined_subjects)
        else:
            pool = _z2t_build_candidate_pool_pair_conditioned(img_01, c1, c2)
        if it > 0:
            pool += [best_prompt]
        pool = list(dict.fromkeys(pool))   # deduplicate, preserve order

        E = embed_prompts(pool)    # (M, D)

        # Ridge regression: λ* = (EEᵀ + αI)⁻¹ E z*ᵀ
        M    = E.shape[0]
        EEt  = E @ E.T                                   # (M, M)
        reg  = ridge_alpha * torch.eye(M, device=device, dtype=EEt.dtype)
        lam  = torch.linalg.solve(EEt + reg, E @ z_target.T)  # (M, 1)

        # Pseudo-target in CLIP space
        z_align = F.normalize((E.T @ lam).T, dim=-1)    # (1, D)

        # Hybrid ranking:
        # - z_align score keeps the Zero2Text ridge objective
        # - direct image-text score discourages overly generic templates
        sims_align = (E @ z_align.T).squeeze(-1)          # (M,)
        sims_target = (E @ z_target.T).squeeze(-1)        # (M,)
        scores = 0.7 * sims_align + 0.3 * sims_target
        best_idx    = scores.argmax().item()
        best_prompt = pool[best_idx]

    return best_prompt


# ---------------------------------------------------------------------------
# VLM captioning: BLIP-2 image → text → SD3.5 (p* VLM)
# ---------------------------------------------------------------------------
# Captions the AND result with BLIP-2, then uses that caption as a prompt
# for SD3.5.  This tests whether the VLM's natural-language description of
# the AND output is sufficient to reproduce it through SD3.5.
# Default model: Salesforce/blip2-opt-2.7b  (~5 GB, no auth required).
# ---------------------------------------------------------------------------

def load_blip2(model_id: str, device: torch.device):
    """Load BLIP-2 processor and model. Returns (processor, model)."""
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    print(f"  Loading BLIP-2 ({model_id}) ...")
    blip_dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = Blip2Processor.from_pretrained(model_id)
    model     = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=blip_dtype,
    ).to(device).eval()
    return processor, model


@torch.no_grad()
def vlm_caption_image(
    img_01:         torch.Tensor,   # (1, 3, H, W) float32 [0,1]
    blip2_proc,
    blip2_model,
    device:         torch.device,
    max_new_tokens: int = 60,
) -> str:
    """
    Caption img_01 with BLIP-2 and return the text string.
    The image is converted to PIL before passing to the processor.
    """
    from torchvision import transforms as T
    pil = T.ToPILImage()(img_01.squeeze(0).detach().cpu().clamp(0, 1))
    inputs = blip2_proc(images=pil, return_tensors="pt")
    inputs = {k: v.to(device=device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=blip2_model.dtype)
    out_ids = blip2_model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = blip2_proc.decode(out_ids[0], skip_special_tokens=True).strip()
    return caption or "a photo"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Measure composability gap")
    p.add_argument("--ckpt",       default=None,
                   help="Path to trained CLIP-inverter checkpoint.  Required only when "
                        "--pstar-source inverter (or omitted).  Other sources skip the load.")
    p.add_argument("--model-id",   default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--output-dir", default="",
                   help="Root output directory.  If empty (default), a timestamped "
                        "directory is created under experiments/inversion/gap_analysis/.")
    p.add_argument("--data-dir",   default="experiments/inversion/training_data",
                   help="Path to training data directory (for text-decoding vocabulary)")
    p.add_argument("--regime", choices=["small", "medium", "large"], default=None,
                   help=(
                       "Preset scale for pairs and seeds:\n"
                       "  small  —  4 pairs ×  4 seeds =  16 records  (quick iteration)\n"
                       "  medium —  4 pairs × 16 seeds =  64 records  (default scale)\n"
                       "  large  — 24 pairs × 24 seeds = 576 records  (ECCV paper)\n"
                       "Explicit --seeds and --pairs-* flags override regime defaults."
                   ))
    p.add_argument("--seeds",      type=int, nargs="+", default=None,
                   help="Seed list override. Defaults: small=[42,1,2,3], medium=0..15, "
                        "large=[42,1..23] "
                        "(or 0..7 when --regime is not set).")
    p.add_argument(
        "--grid-seed",
        type=int,
        default=None,
        help=(
            "Seed value used when exporting per-pair grid assets for "
            "plot_gap_analysis.py --plot grid. "
            "Must be one of --seeds. Default: first seed in --seeds."
        ),
    )
    p.add_argument(
        "--monolithic-baseline",
        choices=["naive", "natural"],
        default="naive",
        help=(
            "Which monolithic prompt variant should populate backward-compatible "
            "keys/exports (`d_T_mono`, `d_t_mono`, `gap_and_mono`, "
            "`sd35_monolithic.png`, grid_assets['monolithic']). "
            "'naive' uses \"c1 and c2\"; 'natural' uses pair-specific naturalized prompts."
        ),
    )
    p.add_argument("--steps",      type=int,   default=50)
    p.add_argument("--guidance",   type=float, default=4.5)
    p.add_argument(
        "--superdiff-kappa-mode",
        choices=["dynamic", "fixed"],
        default="dynamic",
        help=(
            "How SuperDiff AND mixes concepts. "
            "'dynamic' uses timestep-adaptive kappa (default). "
            "'fixed' uses a constant kappa value for all steps. "
            "Note: this script currently forces dynamic mode to stay aligned "
            "with trajectory_dynamics_experiment.py grid behavior."
        ),
    )
    p.add_argument(
        "--superdiff-fixed-kappa",
        type=float,
        default=0.5,
        help=(
            "Constant kappa used when --superdiff-kappa-mode fixed. "
            "Use 0.5 for equal concept blending."
        ),
    )
    p.add_argument("--image-size", type=int,   default=512)
    p.add_argument("--dtype",      default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument(
        "--latent-gap-batch-size",
        type=int,
        default=4,
        help=(
            "Mini-batch size used for VAE latent-gap encoding. "
            "Lower this (e.g., 1-2) if large-regime runs OOM during metric computation."
        ),
    )
    p.add_argument("--clip-model-id", default="openai/clip-vit-large-patch14")
    p.add_argument("--projection", default="mds", choices=["pca", "mds"],
                   help="Dimensionality reduction method for trajectory visualisation")
    p.add_argument("--k-samples", type=int, default=1,
                   help="Best-of-K inversion: sample K p* candidates via MC dropout "
                        "and pick the best by single-NFE CLIP score. "
                        "k=1 (default) disables and uses a single deterministic forward pass.")
    p.add_argument("--poe", action="store_true",
                   help="Also run PoE baseline (score addition of c1 and c2) and save its outputs.")
    # --- PEZ flags ---
    p.add_argument("--pez", action="store_true",
                   help="Also run PEZ hard-prompt optimisation as a second p* source.")
    p.add_argument("--pez-tokens", type=int, default=16,
                   help="Number of tokens to optimise in PEZ (default 16).")
    p.add_argument("--pez-iters", type=int, default=300,
                   help="Gradient steps for PEZ optimisation (default 300).")
    p.add_argument("--pez-lr", type=float, default=0.4,
                   help="Adam lr for PEZ (default 0.4).")
    # --- Zero2Text flags ---
    p.add_argument("--z2t", action="store_true",
                   help="Also run legacy Zero2Text-style ridge-regression inversion (approximate).")
    p.add_argument("--z2t-iters", type=int, default=5,
                   help="Recursive alignment rounds for Z2T (default 5).")
    p.add_argument("--z2t-alpha", type=float, default=0.01,
                   help="Ridge regularisation coefficient for Z2T (default 0.01).")
    p.add_argument(
        "--z2t-pool-mode",
        choices=["image_only", "pair_conditioned"],
        default="image_only",
        help=(
            "Candidate pool policy for Z2T. "
            "image_only (default) uses only CLIP-mined subjects from the AND image; "
            "pair_conditioned keeps the legacy c1/c2 template prior."
        ),
    )
    # --- VLM flags ---
    p.add_argument("--vlm", action="store_true",
                   help="Also run BLIP-2 captioning as a fourth p* source.")
    p.add_argument("--vlm-model-id", default="Salesforce/blip2-opt-2.7b",
                   help="HuggingFace model ID for the BLIP-2 captioner.")
    p.add_argument("--vlm-max-tokens", type=int, default=60,
                   help="Maximum new tokens for BLIP-2 generation (default 60).")
    p.add_argument(
        "--vlm-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help=(
            "Device for BLIP-2 captioning. "
            "'auto' uses CUDA when available; 'cpu' lowers GPU memory pressure."
        ),
    )
    # --- Convenience shorthand + merge mode ---
    p.add_argument(
        "--pstar-source",
        choices=["inverter", "pez", "vlm", "z2t", "all"],
        default=None,
        help=(
            "Shorthand for selecting a single p* source.  Maps to the "
            "individual flags above.  Use with --merge to accumulate multiple "
            "sources into the same output JSON files across separate runs:\n"
            "  inverter  — trained CLIP inverter  (default, always on)\n"
            "  pez       — discrete token optimisation  (--pez)\n"
            "  vlm       — BLIP-2 VLM captioning  (--vlm)\n"
            "  z2t       — Zero2Text ridge regression  (--z2t)\n"
            "  all       — VLM only (inverter / PEZ / Z2T excluded)"
        ),
    )
    p.add_argument(
        "--merge",
        action="store_true",
        help=(
            "When set, merge the new p* column(s) into the existing JSON "
            "files in --output-dir rather than overwriting them.  Useful "
            "when running --pstar-source multiple times to build up all "
            "sources incrementally."
        ),
    )
    p.add_argument(
        "--anchor",
        choices=["seed", "mean"],
        default="seed",
        help=(
            "Distance anchor used for d_T_* and d_t_* measurements.\n"
            "  seed  (default) — per-seed paired comparison: each condition is\n"
            "         compared to the AND latent from the *same* starting noise x_T.\n"
            "         Controls for shared initial randomness; recommended primary metric.\n"
            "  mean  — sensitivity check: compare every condition to the *average*\n"
            "         AND latent z_AND_avg = mean_s(z_AND(s)).  This adds within-AND\n"
            "         variance (d_within_and) to every measurement.  Distances are\n"
            "         systematically larger; useful as a reviewer robustness check.\n"
            "         Saved under keys d_T_*_meananchor (including p* when present)."
        ),
    )
    return p.parse_args()


def _merge_json_column(path: Path, new_records: list, match_keys: tuple, new_cols: list):
    """
    Merge `new_cols` from `new_records` into the JSON list at `path`.
    Records are matched by the fields in `match_keys` (e.g. ("pair", "seed")).
    Existing records that don't have a match are left unchanged.
    New records that don't have a match are appended (shouldn't happen in practice).
    """
    if not path.exists():
        # Nothing to merge into — just write the new records as-is
        path.write_text(json.dumps(new_records, indent=2))
        return

    existing = json.loads(path.read_text())

    # Build lookup: match_key_values → index in existing list
    lookup = {}
    for idx, rec in enumerate(existing):
        key = tuple(rec.get(k) for k in match_keys)
        lookup[key] = idx

    for new_rec in new_records:
        key = tuple(new_rec.get(k) for k in match_keys)
        if key in lookup:
            for col in new_cols:
                if col in new_rec:
                    existing[lookup[key]][col] = new_rec[col]
        else:
            existing.append(new_rec)   # unexpected new record — append

    path.write_text(json.dumps(existing, indent=2))


def _merge_gap_json(path: Path, new_results: list, new_gap_keys: list):
    """
    Merge new gap keys from new_results into the all_pairs_gap.json list at `path`.
    Records matched by pair slug.
    """
    if not path.exists():
        path.write_text(json.dumps(new_results, indent=2))
        return

    existing = json.loads(path.read_text())
    lookup   = {r["slug"]: i for i, r in enumerate(existing)}

    for new_rec in new_results:
        slug = new_rec.get("slug")
        if slug in lookup:
            for gk in new_gap_keys:
                if gk in new_rec:
                    existing[lookup[slug]][gk] = new_rec[gk]
            # Also merge prompt lists if present
            for extra_key in (
                "pez_prompts",
                "z2t_prompts",
                "z2t_pool_mode",
                "vlm_captions",
                "decoded_p_star",
            ):
                if extra_key in new_rec:
                    existing[lookup[slug]][extra_key] = new_rec[extra_key]
        else:
            existing.append(new_rec)

    path.write_text(json.dumps(existing, indent=2))


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if args.dtype == "float16" else torch.bfloat16

    if args.vlm_device == "auto":
        vlm_device = device
    elif args.vlm_device == "cuda":
        if torch.cuda.is_available():
            vlm_device = torch.device("cuda")
        else:
            print("Warning: --vlm-device cuda requested but CUDA is unavailable; falling back to CPU.")
            vlm_device = torch.device("cpu")
    else:
        vlm_device = torch.device("cpu")

    # Keep SuperDiff config aligned with trajectory_dynamics_experiment.py grid runs.
    # That script uses the dynamic-kappa fm_ode path by default.
    if args.superdiff_kappa_mode != "dynamic":
        print(
            "Aligning SuperDiff config with trajectory_dynamics_experiment.py: "
            f"overriding --superdiff-kappa-mode {args.superdiff_kappa_mode!r} -> 'dynamic'."
        )
        args.superdiff_kappa_mode = "dynamic"

    # ---- Resolve --regime into test pairs and seeds ----
    if args.regime is not None:
        test_pairs = REGIME_PAIRS[args.regime]
        if args.seeds is None:
            args.seeds = REGIME_SEEDS[args.regime]
        print(f"Regime: {args.regime}  "
              f"({len(test_pairs)} pairs × {len(args.seeds)} seeds = "
              f"{len(test_pairs) * len(args.seeds)} records)")
    else:
        test_pairs = TEST_PAIRS
        if args.seeds is None:
            args.seeds = list(range(8))   # legacy default

    # ---- Resolve grid seed used by plot_gap_analysis.py --plot grid export ----
    if args.grid_seed is None:
        args._grid_seed = int(args.seeds[0])
        args._grid_seed_idx = 0
    else:
        if args.grid_seed not in args.seeds:
            raise ValueError(
                f"--grid-seed {args.grid_seed} is not in --seeds {args.seeds}. "
                "Add it to --seeds or choose one of the listed values."
            )
        args._grid_seed = int(args.grid_seed)
        args._grid_seed_idx = args.seeds.index(args.grid_seed)

    print(
        f"Grid export seed: {args._grid_seed} "
        f"(index {args._grid_seed_idx} in --seeds)"
    )
    print(f"Monolithic baseline mode: {args.monolithic_baseline}")

    # ---- Resolve --pstar-source shorthand into individual flags ----
    if args.pstar_source is not None:
        src = args.pstar_source
        # Project policy: --pstar-source all runs only VLM.
        if src == "all":
            args.pez = False
            args.vlm = True
            args.z2t = False
            args._skip_inverter = True
        else:
            # "inverter" = default run (no extra flags needed)
            # Any other single source disables the trained inverter to save time.
            args.pez = (src == "pez")
            args.vlm = (src == "vlm")
            args.z2t = (src == "z2t")
            args._skip_inverter = (src != "inverter")
    else:
        args._skip_inverter = False

    # ---- Resolve output root (auto-timestamp if not explicitly set) ----
    if not args.output_dir:
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        regime_tag  = f"{args.regime}_" if args.regime else ""
        args.output_dir = f"experiments/inversion/gap_analysis/{regime_tag}{timestamp}"
        print(f"Output directory (auto): {args.output_dir}")

    out_root    = Path(args.output_dir)
    latent_size = args.image_size // 8

    print(
        "SuperDiff kappa mode: "
        f"{args.superdiff_kappa_mode}"
        + (
            f" (kappa={args.superdiff_fixed_kappa:.3f})"
            if args.superdiff_kappa_mode == "fixed"
            else ""
        )
    )

    # ---- Load SD3.5 ----
    print("Loading SD3.5 ...")
    models = get_sd3_models(model_id=args.model_id, dtype=dtype, device=device)

    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )

    # ---- Load inverter (skipped when --pstar-source is not inverter/all) ----
    inverter   = None
    preprocess = None
    if not args._skip_inverter:
        if args.ckpt is None:
            raise ValueError(
                "--ckpt is required when running the CLIP inverter.  "
                "Pass --pstar-source pez/vlm/z2t to skip it, or supply --ckpt <path>."
            )
        print(f"Loading inverter from {args.ckpt} ...")
        inverter   = load_inverter(args.ckpt, clip_model_id=args.clip_model_id, device=device)
        inverter   = inverter.eval().to(dtype=torch.float32)
        preprocess = make_clip_preprocessor(device)

    # ---- Load CLIP for evaluation (separate from inverter backbone) ----
    print("Loading CLIP for evaluation ...")
    from transformers import CLIPModel
    clip_eval = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval()

    # ---- Load CLIP tokenizer for PEZ / Z2T ----
    clip_tokenizer = None
    if args.pez or args.z2t:
        from transformers import CLIPTokenizer
        clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_id)
        print(f"  CLIP tokenizer loaded (vocab size {clip_tokenizer.vocab_size})")
        if args.z2t:
            print(f"  Z2T pool mode: {args.z2t_pool_mode}")

    # ---- Load BLIP-2 for VLM captioning ----
    blip2_proc, blip2_model = None, None
    if args.vlm:
        print(f"  BLIP-2 device: {vlm_device}")
        blip2_proc, blip2_model = load_blip2(args.vlm_model_id, vlm_device)

    # ---- Load LPIPS if available ----
    lpips_fn = None
    if LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net="vgg")  # kept on CPU; images moved to CPU at call time

    # ---- Pre-compute unconditional conditioning ----
    with torch.no_grad():
        uncond_embeds, uncond_pooled = get_sd3_text_embedding(
            [""],
            models["tokenizer"],   models["text_encoder"],
            models["tokenizer_2"], models["text_encoder_2"],
            models["tokenizer_3"], models["text_encoder_3"],
            device=device,
        )

    # ---- Load vocabulary for text decoding ----
    vocab_prompts = []
    index_path = Path(args.data_dir) / "dataset_index.json"
    if index_path.exists():
        with open(index_path) as f:
            vocab_prompts = [entry["prompt"] for entry in json.load(f)]
        print(f"Loaded {len(vocab_prompts)} vocab prompts from {index_path}")
    else:
        print(f"Warning: {index_path} not found — text decoding will be skipped")

    all_pair_results = []
    all_seed_records = []
    all_traj_records = []
    all_within_and_records = []

    for pair_idx, (c1, c2) in enumerate(test_pairs):
        pair_group = _pair_group(c1, c2)
        mono_prompt_naive, mono_prompt_natural = _pair_monolithic_prompts(c1, c2)
        mono_prompt_active = (
            mono_prompt_natural
            if args.monolithic_baseline == "natural"
            else mono_prompt_naive
        )
        pair_slug  = f"{c1.replace(' ', '_')}_{c2.replace(' ', '_')}"
        pair_dir   = out_root / "pairs" / pair_slug
        img_dir    = pair_dir / "images"
        traj_dir   = pair_dir / "trajectories"
        pair_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)
        traj_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Pair: '{c1}' AND '{c2}'")
        print(f"Group: {pair_group}")
        print(f"Monolithic (naive):   \"{mono_prompt_naive}\"")
        print(f"Monolithic (natural): \"{mono_prompt_natural}\"")
        print(
            f"Monolithic (active/{args.monolithic_baseline}): "
            f"\"{mono_prompt_active}\""
        )
        print(f"{'='*60}")

        images_and   = []
        images_pstar = []
        images_mono  = []  # active baseline (backward-compatible key)
        images_mono_naive = []
        images_mono_natural = []
        images_poe   = [] if args.poe else None

        trackers_and   = []
        trackers_pstar = []
        trackers_mono  = []  # active baseline (backward-compatible key)
        trackers_mono_naive = []
        trackers_mono_natural = []
        trackers_poe   = [] if args.poe else None
        pred_pooled_list = []   # accumulate across seeds for text decoding

        # Optional additional p* sources
        images_pstar_pez = [] if args.pez else None
        trackers_pstar_pez = [] if args.pez else None
        pez_prompts = [] if args.pez else None
        images_pstar_z2t = [] if args.z2t else None
        trackers_pstar_z2t = [] if args.z2t else None
        z2t_prompts = [] if args.z2t else None
        images_pstar_vlm = [] if args.vlm else None
        trackers_pstar_vlm = [] if args.vlm else None
        vlm_captions = [] if args.vlm else None

        images_c1        = []
        images_c2        = []
        trackers_c1      = []
        trackers_c2      = []
        per_seed_records = []

        for seed in args.seeds:
            gen = torch.Generator(device=device).manual_seed(seed)
            init_latents = torch.randn(
                1, 16, latent_size, latent_size,
                device=device, dtype=dtype, generator=gen,
            )

            # ---- Step 1: SuperDiff-AND ----
            with torch.no_grad():
                lat_and, tracker_and, *_ = superdiff_fm_ode_sd3(
                    latents=init_latents.clone(),
                    obj_prompt=c1,
                    bg_prompt=c2,
                    scheduler=scheduler,
                    transformer=models["transformer"],
                    tokenizer=models["tokenizer"],
                    text_encoder=models["text_encoder"],
                    tokenizer_2=models["tokenizer_2"],
                    text_encoder_2=models["text_encoder_2"],
                    tokenizer_3=models["tokenizer_3"],
                    text_encoder_3=models["text_encoder_3"],
                    guidance_scale=args.guidance,
                    num_inference_steps=args.steps,
                    batch_size=1,
                    device=device,
                    dtype=dtype,
                    kappa_mode=args.superdiff_kappa_mode,
                    fixed_kappa=args.superdiff_fixed_kappa,
                )
                img_and = decode_latents(models["vae"], lat_and)  # (1, 3, H, W)
                # Single explicit source for all inversion methods (inverter/PEZ/Z2T/VLM).
                source_img_for_inversion = img_and.clone()

            images_and.append(img_and.cpu())
            trackers_and.append(tracker_and)

            # ---- Step 2: Invert SuperDiff-AND image (trained CLIP inverter) ----
            # Skipped when --pstar-source {pez|vlm|z2t} to save time on subsequent runs.
            if not args._skip_inverter:
                img_clip = preprocess(source_img_for_inversion)  # (1, 3, 224, 224), float32

                candidates = inverter.sample_k(img_clip, k=args.k_samples)

                pred_pooled, pred_seq, best_score = select_best_p_star(
                    candidates,
                    x_T=init_latents.clone(),
                    target_img=source_img_for_inversion,
                    transformer=models["transformer"],
                    scheduler=scheduler,
                    uncond_embeds=uncond_embeds,
                    uncond_pooled=uncond_pooled,
                    guidance_scale=args.guidance,
                    vae=models["vae"],
                    clip_eval=clip_eval,
                    device=device,
                    dtype=dtype,
                )
                if best_score is not None:
                    print(f"    best-of-{args.k_samples} score: {best_score:.4f}")
                pred_pooled_list.append(pred_pooled.squeeze(0).float().cpu())

                # ---- Step 3: SD3.5(p*) ----
                with torch.no_grad():
                    lat_pstar, tracker_pstar = sample_sd3_with_precomputed_cond(
                        latents=init_latents.clone(),
                        cond_embeds=pred_seq.to(dtype=dtype),
                        cond_pooled=pred_pooled.to(dtype=dtype),
                        uncond_embeds=uncond_embeds,
                        uncond_pooled=uncond_pooled,
                        scheduler=scheduler,
                        transformer=models["transformer"],
                        guidance_scale=args.guidance,
                        num_inference_steps=args.steps,
                        device=device,
                        dtype=dtype,
                    )
                    img_pstar = decode_latents(models["vae"], lat_pstar)

                images_pstar.append(img_pstar.cpu())
                trackers_pstar.append(tracker_pstar)
            else:
                # Placeholder so list lengths stay consistent across seeds
                images_pstar.append(None)
                trackers_pstar.append(None)

            # ---- Step 3b: PEZ hard-prompt inversion ----
            if args.pez:
                pez_prompt = pez_invert_image(
                    source_img_for_inversion,
                    clip_model=clip_eval,
                    clip_tokenizer=clip_tokenizer,
                    n_tokens=args.pez_tokens,
                    n_iters=args.pez_iters,
                    lr=args.pez_lr,
                    device=device,
                )
                pez_prompts.append(pez_prompt)
                print(f"    PEZ prompt: \"{pez_prompt}\"")
                with torch.no_grad():
                    lat_pez, tracker_pez = sample_sd3_with_trajectory_tracking(
                        latents=init_latents.clone(),
                        prompt=pez_prompt,
                        scheduler=scheduler,
                        transformer=models["transformer"],
                        tokenizer=models["tokenizer"],
                        text_encoder=models["text_encoder"],
                        tokenizer_2=models["tokenizer_2"],
                        text_encoder_2=models["text_encoder_2"],
                        tokenizer_3=models["tokenizer_3"],
                        text_encoder_3=models["text_encoder_3"],
                        guidance_scale=args.guidance,
                        num_inference_steps=args.steps,
                        batch_size=1,
                        device=device,
                        dtype=dtype,
                    )
                    img_pez = decode_latents(models["vae"], lat_pez)
                images_pstar_pez.append(img_pez.cpu())
                trackers_pstar_pez.append(tracker_pez)

            # ---- Step 3c: Zero2Text ridge-regression inversion ----
            if args.z2t:
                z2t_prompt = z2t_invert_image(
                    source_img_for_inversion,
                    clip_model=clip_eval,
                    clip_tokenizer=clip_tokenizer,
                    c1=c1,
                    c2=c2,
                    pool_mode=args.z2t_pool_mode,
                    n_iters=args.z2t_iters,
                    ridge_alpha=args.z2t_alpha,
                    device=device,
                )
                z2t_prompts.append(z2t_prompt)
                print(f"    Z2T prompt:  \"{z2t_prompt}\"")
                with torch.no_grad():
                    lat_z2t, tracker_z2t = sample_sd3_with_trajectory_tracking(
                        latents=init_latents.clone(),
                        prompt=z2t_prompt,
                        scheduler=scheduler,
                        transformer=models["transformer"],
                        tokenizer=models["tokenizer"],
                        text_encoder=models["text_encoder"],
                        tokenizer_2=models["tokenizer_2"],
                        text_encoder_2=models["text_encoder_2"],
                        tokenizer_3=models["tokenizer_3"],
                        text_encoder_3=models["text_encoder_3"],
                        guidance_scale=args.guidance,
                        num_inference_steps=args.steps,
                        batch_size=1,
                        device=device,
                        dtype=dtype,
                    )
                    img_z2t = decode_latents(models["vae"], lat_z2t)
                images_pstar_z2t.append(img_z2t.cpu())
                trackers_pstar_z2t.append(tracker_z2t)

            # ---- Step 3d: VLM captioning (BLIP-2) ----
            if args.vlm:
                vlm_caption = vlm_caption_image(
                    source_img_for_inversion,
                    blip2_proc=blip2_proc,
                    blip2_model=blip2_model,
                    device=vlm_device,
                    max_new_tokens=args.vlm_max_tokens,
                )
                vlm_captions.append(vlm_caption)
                print(f"    VLM caption: \"{vlm_caption}\"")
                with torch.no_grad():
                    lat_vlm, tracker_vlm = sample_sd3_with_trajectory_tracking(
                        latents=init_latents.clone(),
                        prompt=vlm_caption,
                        scheduler=scheduler,
                        transformer=models["transformer"],
                        tokenizer=models["tokenizer"],
                        text_encoder=models["text_encoder"],
                        tokenizer_2=models["tokenizer_2"],
                        text_encoder_2=models["text_encoder_2"],
                        tokenizer_3=models["tokenizer_3"],
                        text_encoder_3=models["text_encoder_3"],
                        guidance_scale=args.guidance,
                        num_inference_steps=args.steps,
                        batch_size=1,
                        device=device,
                        dtype=dtype,
                    )
                    img_vlm = decode_latents(models["vae"], lat_vlm)
                images_pstar_vlm.append(img_vlm.cpu())
                trackers_pstar_vlm.append(tracker_vlm)

            # ---- Step 4: Monolithic baselines (naive and naturalized) ----
            with torch.no_grad():
                lat_mono_naive, tracker_mono_naive = sample_sd3_with_trajectory_tracking(
                    latents=init_latents.clone(),
                    prompt=mono_prompt_naive,
                    scheduler=scheduler,
                    transformer=models["transformer"],
                    tokenizer=models["tokenizer"],
                    text_encoder=models["text_encoder"],
                    tokenizer_2=models["tokenizer_2"],
                    text_encoder_2=models["text_encoder_2"],
                    tokenizer_3=models["tokenizer_3"],
                    text_encoder_3=models["text_encoder_3"],
                    guidance_scale=args.guidance,
                    num_inference_steps=args.steps,
                    batch_size=1,
                    device=device,
                    dtype=dtype,
                )
                img_mono_naive = decode_latents(models["vae"], lat_mono_naive)

                if mono_prompt_natural == mono_prompt_naive:
                    lat_mono_natural = lat_mono_naive
                    tracker_mono_natural = tracker_mono_naive
                    img_mono_natural = img_mono_naive
                else:
                    lat_mono_natural, tracker_mono_natural = sample_sd3_with_trajectory_tracking(
                        latents=init_latents.clone(),
                        prompt=mono_prompt_natural,
                        scheduler=scheduler,
                        transformer=models["transformer"],
                        tokenizer=models["tokenizer"],
                        text_encoder=models["text_encoder"],
                        tokenizer_2=models["tokenizer_2"],
                        text_encoder_2=models["text_encoder_2"],
                        tokenizer_3=models["tokenizer_3"],
                        text_encoder_3=models["text_encoder_3"],
                        guidance_scale=args.guidance,
                        num_inference_steps=args.steps,
                        batch_size=1,
                        device=device,
                        dtype=dtype,
                    )
                    img_mono_natural = decode_latents(models["vae"], lat_mono_natural)

            if args.monolithic_baseline == "natural":
                img_mono = img_mono_natural
                tracker_mono = tracker_mono_natural
            else:
                img_mono = img_mono_naive
                tracker_mono = tracker_mono_naive

            images_mono.append(img_mono.cpu())
            images_mono_naive.append(img_mono_naive.cpu())
            images_mono_natural.append(img_mono_natural.cpu())
            trackers_mono.append(tracker_mono)
            trackers_mono_naive.append(tracker_mono_naive)
            trackers_mono_natural.append(tracker_mono_natural)

            # ---- Step 4b: PoE baseline (score addition) ----
            if args.poe:
                with torch.no_grad():
                    lat_poe, tracker_poe = poe_sd3_with_trajectory_tracking(
                        latents=init_latents.clone(),
                        prompt_a=c1,
                        prompt_b=c2,
                        scheduler=scheduler,
                        transformer=models["transformer"],
                        tokenizer=models["tokenizer"],
                        text_encoder=models["text_encoder"],
                        tokenizer_2=models["tokenizer_2"],
                        text_encoder_2=models["text_encoder_2"],
                        tokenizer_3=models["tokenizer_3"],
                        text_encoder_3=models["text_encoder_3"],
                        guidance_scale=args.guidance,
                        num_inference_steps=args.steps,
                        batch_size=1,
                        device=device,
                        dtype=dtype,
                    )
                    img_poe = decode_latents(models["vae"], lat_poe)
                images_poe.append(img_poe.cpu())
                trackers_poe.append(tracker_poe)

            # ---- c1-only and c2-only baselines (same initial noise) ----
            with torch.no_grad():
                lat_c1, tracker_c1 = sample_sd3_with_trajectory_tracking(
                    latents=init_latents.clone(),
                    prompt=c1,
                    scheduler=scheduler,
                    transformer=models["transformer"],
                    tokenizer=models["tokenizer"],
                    text_encoder=models["text_encoder"],
                    tokenizer_2=models["tokenizer_2"],
                    text_encoder_2=models["text_encoder_2"],
                    tokenizer_3=models["tokenizer_3"],
                    text_encoder_3=models["text_encoder_3"],
                    guidance_scale=args.guidance,
                    num_inference_steps=args.steps,
                    batch_size=1,
                    device=device,
                    dtype=dtype,
                )
                img_c1 = decode_latents(models["vae"], lat_c1)

                lat_c2, tracker_c2 = sample_sd3_with_trajectory_tracking(
                    latents=init_latents.clone(),
                    prompt=c2,
                    scheduler=scheduler,
                    transformer=models["transformer"],
                    tokenizer=models["tokenizer"],
                    text_encoder=models["text_encoder"],
                    tokenizer_2=models["tokenizer_2"],
                    text_encoder_2=models["text_encoder_2"],
                    tokenizer_3=models["tokenizer_3"],
                    text_encoder_3=models["text_encoder_3"],
                    guidance_scale=args.guidance,
                    num_inference_steps=args.steps,
                    batch_size=1,
                    device=device,
                    dtype=dtype,
                )
                img_c2 = decode_latents(models["vae"], lat_c2)

            images_c1.append(img_c1.cpu())
            images_c2.append(img_c2.cpu())
            trackers_c1.append(tracker_c1)
            trackers_c2.append(tracker_c2)

            # ---- Per-seed terminal latent distances (anchor = AND) ----
            # Use raw tracker latents — no VAE re-encode error.
            # Per-element MSE is dimension-normalised and scale-comparable.
            z_and  = tracker_and.trajectories[-1].float()   # (1, 16, H, W)
            z_mono = tracker_mono.trajectories[-1].float()
            z_mono_naive = tracker_mono_naive.trajectories[-1].float()
            z_mono_natural = tracker_mono_natural.trajectories[-1].float()
            z_c1_t = tracker_c1.trajectories[-1].float()
            z_c2_t = tracker_c2.trajectories[-1].float()
            rec = {
                "pair":     f"{c1} + {c2}",
                "c1":       c1,
                "c2":       c2,
                "pair_group": pair_group,
                "seed":     seed,
                "monolithic_baseline": args.monolithic_baseline,
                "mono_prompt_active": mono_prompt_active,
                "mono_prompt_naive": mono_prompt_naive,
                "mono_prompt_natural": mono_prompt_natural,
                "d_T_mono": float(((z_mono  - z_and) ** 2).mean()),  # backward compat: active
                "d_T_mono_naive": float(((z_mono_naive  - z_and) ** 2).mean()),
                "d_T_mono_natural": float(((z_mono_natural - z_and) ** 2).mean()),
                "d_T_c1":   float(((z_c1_t  - z_and) ** 2).mean()),
                "d_T_c2":   float(((z_c2_t  - z_and) ** 2).mean()),
            }
            if args.poe:
                z_poe_t = trackers_poe[-1].trajectories[-1].float()
                rec["d_T_poe"] = float(((z_poe_t - z_and) ** 2).mean())
            if not args._skip_inverter:
                z_pstar_t = tracker_pstar.trajectories[-1].float()
                d_inv = float(((z_pstar_t - z_and) ** 2).mean())
                rec["d_T_pstar"]     = d_inv   # backward compat
                rec["d_T_pstar_inv"] = d_inv   # explicit symmetric name
            if args.pez:
                z_pez = trackers_pstar_pez[-1].trajectories[-1].float()
                rec["d_T_pstar_pez"] = float(((z_pez - z_and) ** 2).mean())
            if args.z2t:
                z_z2t = trackers_pstar_z2t[-1].trajectories[-1].float()
                rec["d_T_pstar_z2t"] = float(((z_z2t - z_and) ** 2).mean())
            if args.vlm:
                z_vlm = trackers_pstar_vlm[-1].trajectories[-1].float()
                rec["d_T_pstar_vlm"] = float(((z_vlm - z_and) ** 2).mean())
            per_seed_records.append(rec)

            print(f"  seed {seed} done")

        # ---- Mean-anchor sensitivity check (--anchor mean) ----
        # z_AND_avg = mean_s(z_AND(s)).  Each condition is compared to this
        # single average latent instead of the per-seed AND latent.
        # This adds d_within_and_mean_centered to every measurement, so
        # distances are systematically larger — useful as a robustness check.
        # Keys use the suffix _meananchor to distinguish from the primary metric.
        if args.anchor == "mean":
            and_stack = torch.stack(
                [t.trajectories[-1].float() for t in trackers_and], dim=0
            )   # (N_seeds, 1, C, H, W)
            z_and_avg = and_stack.mean(dim=0)   # (1, C, H, W)

            mono_stack  = torch.stack(
                [t.trajectories[-1].float() for t in trackers_mono], dim=0)
            mono_naive_stack  = torch.stack(
                [t.trajectories[-1].float() for t in trackers_mono_naive], dim=0)
            mono_natural_stack  = torch.stack(
                [t.trajectories[-1].float() for t in trackers_mono_natural], dim=0)
            poe_stack = torch.stack(
                [t.trajectories[-1].float() for t in trackers_poe], dim=0) if args.poe else None
            c1_stack    = torch.stack(
                [t.trajectories[-1].float() for t in trackers_c1], dim=0)
            c2_stack    = torch.stack(
                [t.trajectories[-1].float() for t in trackers_c2], dim=0)
            pstar_stack = torch.stack(
                [t.trajectories[-1].float() for t in trackers_pstar], dim=0
            ) if not args._skip_inverter else None
            pez_stack = torch.stack(
                [t.trajectories[-1].float() for t in trackers_pstar_pez], dim=0
            ) if args.pez else None
            z2t_stack = torch.stack(
                [t.trajectories[-1].float() for t in trackers_pstar_z2t], dim=0
            ) if args.z2t else None
            vlm_stack = torch.stack(
                [t.trajectories[-1].float() for t in trackers_pstar_vlm], dim=0
            ) if args.vlm else None

            for si, seed in enumerate(args.seeds):
                # Find the per_seed_record we just appended for this seed
                rec = next(r for r in per_seed_records
                           if r["pair"] == f"{c1} + {c2}" and r["seed"] == seed)
                rec["d_T_mono_meananchor"] = float(  # backward compat: active
                    ((mono_stack[si] - z_and_avg) ** 2).mean())
                rec["d_T_mono_naive_meananchor"] = float(
                    ((mono_naive_stack[si] - z_and_avg) ** 2).mean())
                rec["d_T_mono_natural_meananchor"] = float(
                    ((mono_natural_stack[si] - z_and_avg) ** 2).mean())
                if args.poe:
                    rec["d_T_poe_meananchor"] = float(
                        ((poe_stack[si] - z_and_avg) ** 2).mean())
                rec["d_T_c1_meananchor"]   = float(
                    ((c1_stack[si]   - z_and_avg) ** 2).mean())
                rec["d_T_c2_meananchor"]   = float(
                    ((c2_stack[si]   - z_and_avg) ** 2).mean())
                if pstar_stack is not None:
                    d_inv_mean = float(((pstar_stack[si] - z_and_avg) ** 2).mean())
                    rec["d_T_pstar_meananchor"] = d_inv_mean
                    rec["d_T_pstar_inv_meananchor"] = d_inv_mean
                if pez_stack is not None:
                    rec["d_T_pstar_pez_meananchor"] = float(
                        ((pez_stack[si] - z_and_avg) ** 2).mean())
                if z2t_stack is not None:
                    rec["d_T_pstar_z2t_meananchor"] = float(
                        ((z2t_stack[si] - z_and_avg) ** 2).mean())
                if vlm_stack is not None:
                    rec["d_T_pstar_vlm_meananchor"] = float(
                        ((vlm_stack[si] - z_and_avg) ** 2).mean())

        # ---- Within-AND cross-seed pairwise distances (noise floor) ----
        # For each unordered pair of seeds (s_i, s_j), compute
        #   d_within_and = mean( (z_AND[s_i] - z_AND[s_j])^2 )
        # This is the stochastic baseline: how much do AND outputs vary
        # across seeds starting from different x_T?  If the per-condition
        # gaps (d_T_mono, d_T_c1, …) are well above this, the pairing used
        # in the main metric is not masking a noise-floor effect.
        print("  Computing within-AND pairwise distances (noise floor) ...")
        for i, seed_i in enumerate(args.seeds):
            for j, seed_j in enumerate(args.seeds):
                if j <= i:
                    continue
                z_i = trackers_and[i].trajectories[-1].float()
                z_j = trackers_and[j].trajectories[-1].float()
                all_within_and_records.append({
                    "pair":         f"{c1} + {c2}",
                    "pair_group":   pair_group,
                    "seed_a":       seed_i,
                    "seed_b":       seed_j,
                    "d_within_and": float(((z_i - z_j) ** 2).mean()),
                })

        # ---- Per-step trajectory distances (CPU, no GPU needed) ----
        # trackers already hold trajectories as CPU tensors from store_step().
        # Shape: (num_steps+1, 1, C, H, W) — iterate over all T+1 states.
        print("  Computing per-step trajectory distances ...")
        for seed_idx, seed in enumerate(args.seeds):
            ta  = trackers_and[seed_idx]
            tm  = trackers_mono[seed_idx]
            tm_naive = trackers_mono_naive[seed_idx]
            tm_natural = trackers_mono_natural[seed_idx]
            tp_oe = trackers_poe[seed_idx] if args.poe else None
            tc1 = trackers_c1[seed_idx]
            tc2 = trackers_c2[seed_idx]
            tp     = trackers_pstar[seed_idx]      # None when _skip_inverter
            tp_pez = trackers_pstar_pez[seed_idx] if args.pez else None
            tp_z2t = trackers_pstar_z2t[seed_idx] if args.z2t else None
            tp_vlm = trackers_pstar_vlm[seed_idx] if args.vlm else None
            n_steps = ta.trajectories.shape[0]   # T+1
            for step in range(n_steps):
                z_and  = ta.trajectories[step].float()
                z_mono = tm.trajectories[step].float()
                z_mono_naive = tm_naive.trajectories[step].float()
                z_mono_natural = tm_natural.trajectories[step].float()
                z_c1_t = tc1.trajectories[step].float()
                z_c2_t = tc2.trajectories[step].float()
                sigma  = float(ta.sigmas[min(step, len(ta.sigmas) - 1)])
                trec = {
                    "pair":     f"{c1} + {c2}",
                    "c1":       c1,
                    "c2":       c2,
                    "pair_group": pair_group,
                    "seed":     seed,
                    "step":     step,
                    "sigma":    sigma,
                    "monolithic_baseline": args.monolithic_baseline,
                    "d_t_mono": float(((z_mono  - z_and) ** 2).mean()),  # backward compat: active
                    "d_t_mono_naive": float(((z_mono_naive  - z_and) ** 2).mean()),
                    "d_t_mono_natural": float(((z_mono_natural - z_and) ** 2).mean()),
                    "d_t_c1":   float(((z_c1_t  - z_and) ** 2).mean()),
                    "d_t_c2":   float(((z_c2_t  - z_and) ** 2).mean()),
                }
                if tp_oe is not None:
                    z_poe_s = tp_oe.trajectories[step].float()
                    trec["d_t_poe"] = float(((z_poe_s - z_and) ** 2).mean())
                if tp is not None:
                    z_pstar_s = tp.trajectories[step].float()
                    d_t_inv = float(((z_pstar_s - z_and) ** 2).mean())
                    trec["d_t_pstar"]     = d_t_inv   # backward compat
                    trec["d_t_pstar_inv"] = d_t_inv   # explicit symmetric name
                if args.pez:
                    z_pez_s = tp_pez.trajectories[step].float()
                    trec["d_t_pstar_pez"] = float(((z_pez_s - z_and) ** 2).mean())
                if args.z2t:
                    z_z2t_s = tp_z2t.trajectories[step].float()
                    trec["d_t_pstar_z2t"] = float(((z_z2t_s - z_and) ** 2).mean())
                if args.vlm:
                    z_vlm_s = tp_vlm.trajectories[step].float()
                    trec["d_t_pstar_vlm"] = float(((z_vlm_s - z_and) ** 2).mean())
                all_traj_records.append(trec)

        # Stack across seeds
        imgs_and   = torch.cat(images_and,  dim=0)  # (N, 3, H, W)
        imgs_pstar = torch.cat([x for x in images_pstar if x is not None], dim=0) \
                     if not args._skip_inverter else None
        imgs_mono  = torch.cat(images_mono, dim=0)  # active baseline
        imgs_mono_naive = torch.cat(images_mono_naive, dim=0)
        imgs_mono_natural = torch.cat(images_mono_natural, dim=0)
        imgs_poe   = torch.cat(images_poe,  dim=0) if args.poe else None
        imgs_c1    = torch.cat(images_c1,   dim=0)
        imgs_c2    = torch.cat(images_c2,   dim=0)

        # ---- Save image grids + manifests ----
        save_image(imgs_and,  img_dir / "superdiff_and.png",   nrow=4, normalize=False)
        save_single_grid_manifest(img_dir / "superdiff_and.png",
            f'SuperDiff AND — "{c1}" ∧ "{c2}"', args.seeds)
        if imgs_pstar is not None:
            save_image(imgs_pstar, img_dir / "sd35_pstar.png", nrow=4, normalize=False)
            save_single_grid_manifest(img_dir / "sd35_pstar.png",
                f'SD3.5 (p* — inverter) — "{c1}" ∧ "{c2}"', args.seeds)
        save_image(imgs_mono, img_dir / "sd35_monolithic.png", nrow=4, normalize=False)
        save_single_grid_manifest(img_dir / "sd35_monolithic.png",
            f'SD3.5 monolithic ({args.monolithic_baseline}, active) — "{mono_prompt_active}"', args.seeds)
        save_image(imgs_mono_naive, img_dir / "sd35_monolithic_naive.png", nrow=4, normalize=False)
        save_single_grid_manifest(img_dir / "sd35_monolithic_naive.png",
            f'SD3.5 monolithic (naive) — "{mono_prompt_naive}"', args.seeds)
        save_image(imgs_mono_natural, img_dir / "sd35_monolithic_natural.png", nrow=4, normalize=False)
        save_single_grid_manifest(img_dir / "sd35_monolithic_natural.png",
            f'SD3.5 monolithic (natural) — "{mono_prompt_natural}"', args.seeds)
        if imgs_poe is not None:
            save_image(imgs_poe, img_dir / "sd35_poe.png", nrow=4, normalize=False)
            save_single_grid_manifest(img_dir / "sd35_poe.png",
                f'SD3.5 PoE — "{c1}" × "{c2}"', args.seeds)
        save_image(imgs_c1,   img_dir / "sd35_c1_only.png",   nrow=4, normalize=False)
        save_single_grid_manifest(img_dir / "sd35_c1_only.png",
            f'SD3.5 solo — "{c1}"', args.seeds)
        save_image(imgs_c2,   img_dir / "sd35_c2_only.png",   nrow=4, normalize=False)
        save_single_grid_manifest(img_dir / "sd35_c2_only.png",
            f'SD3.5 solo — "{c2}"', args.seeds)

        if args.pez:
            imgs_pstar_pez_all = torch.cat(images_pstar_pez, dim=0)
            save_image(imgs_pstar_pez_all, img_dir / "sd35_pstar_pez.png", nrow=4, normalize=False)
            save_single_grid_manifest(img_dir / "sd35_pstar_pez.png",
                f'SD3.5 (p* — PEZ) — "{c1}" ∧ "{c2}"', args.seeds,
                per_seed_prompts=pez_prompts)
        if args.z2t:
            imgs_pstar_z2t_all = torch.cat(images_pstar_z2t, dim=0)
            save_image(imgs_pstar_z2t_all, img_dir / "sd35_pstar_z2t.png", nrow=4, normalize=False)
            save_single_grid_manifest(img_dir / "sd35_pstar_z2t.png",
                f'SD3.5 (p* — Z2T/{args.z2t_pool_mode}) — "{c1}" ∧ "{c2}"', args.seeds,
                per_seed_prompts=z2t_prompts)
        if args.vlm:
            imgs_pstar_vlm_all = torch.cat(images_pstar_vlm, dim=0)
            save_image(imgs_pstar_vlm_all, img_dir / "sd35_pstar_vlm.png", nrow=4, normalize=False)
            save_single_grid_manifest(img_dir / "sd35_pstar_vlm.png",
                f'SD3.5 (p* — VLM/BLIP-2) — "{c1}" ∧ "{c2}"', args.seeds,
                per_seed_prompts=vlm_captions)

        # ---- Export single-seed assets for cross-pair grid plots (27–28) ----
        # Use the configured grid seed so all conditions remain seed-matched.
        grid_seed_idx = args._grid_seed_idx
        grid_seed = args._grid_seed

        decoded_for_grid = {
            "prompt_a":         images_c1[grid_seed_idx],
            "prompt_b":         images_c2[grid_seed_idx],
            "monolithic":       images_mono[grid_seed_idx],  # active
            "monolithic_naive": images_mono_naive[grid_seed_idx],
            "monolithic_natural": images_mono_natural[grid_seed_idx],
            "superdiff_fm_ode": images_and[grid_seed_idx],
        }
        trackers_for_grid = {
            "prompt_a":         trackers_c1[grid_seed_idx],
            "prompt_b":         trackers_c2[grid_seed_idx],
            "monolithic":       trackers_mono[grid_seed_idx],  # active
            "monolithic_naive": trackers_mono_naive[grid_seed_idx],
            "monolithic_natural": trackers_mono_natural[grid_seed_idx],
            "superdiff_fm_ode": trackers_and[grid_seed_idx],
        }
        if args.poe and images_poe is not None and trackers_poe is not None:
            decoded_for_grid["poe"] = images_poe[grid_seed_idx]
            trackers_for_grid["poe"] = trackers_poe[grid_seed_idx]
        if args.vlm and images_pstar_vlm is not None and trackers_pstar_vlm is not None:
            decoded_for_grid["pstar_vlm"] = images_pstar_vlm[grid_seed_idx]
            trackers_for_grid["pstar_vlm"] = trackers_pstar_vlm[grid_seed_idx]
        if args.z2t and images_pstar_z2t is not None and trackers_pstar_z2t is not None:
            decoded_for_grid["pstar_z2t"] = images_pstar_z2t[grid_seed_idx]
            trackers_for_grid["pstar_z2t"] = trackers_pstar_z2t[grid_seed_idx]

        source_prompts = {}
        source_prompts["monolithic_baseline"] = args.monolithic_baseline
        source_prompts["monolithic"] = mono_prompt_active
        source_prompts["monolithic_naive"] = mono_prompt_naive
        source_prompts["monolithic_natural"] = mono_prompt_natural
        if args.vlm and vlm_captions and grid_seed_idx < len(vlm_captions):
            source_prompts["pstar_vlm"] = vlm_captions[grid_seed_idx]
        if args.z2t and z2t_prompts and grid_seed_idx < len(z2t_prompts):
            source_prompts["pstar_z2t"] = z2t_prompts[grid_seed_idx]

        export_pair_grid_assets(
            pair_dir=pair_dir,
            c1=c1,
            c2=c2,
            seed=grid_seed,
            decoded_images=decoded_for_grid,
            trackers=trackers_for_grid,
            pair_index=pair_idx,
            projection_method=args.projection,
            source_prompts=source_prompts,
        )

        # ---- Step 5: Gap metrics ----
        print(f"  Computing image gap: AND vs monolithic (active/{args.monolithic_baseline}) ...")
        gap_and_mono = compute_image_gap(imgs_and, imgs_mono, clip_eval, lpips_fn, device)
        print("  Computing image gap: AND vs monolithic (naive) ...")
        gap_and_mono_naive = compute_image_gap(
            imgs_and, imgs_mono_naive, clip_eval, lpips_fn, device
        )
        print("  Computing image gap: AND vs monolithic (natural) ...")
        gap_and_mono_natural = compute_image_gap(
            imgs_and, imgs_mono_natural, clip_eval, lpips_fn, device
        )

        metrics = {
            "pair":         (c1, c2),
            "pair_index":   int(pair_idx),
            "slug":         pair_slug,
            "pair_group":   pair_group,
            "n_seeds":      len(args.seeds),
            "monolithic_baseline": args.monolithic_baseline,
            "mono_prompt_active": mono_prompt_active,
            "mono_prompt_naive": mono_prompt_naive,
            "mono_prompt_natural": mono_prompt_natural,
            "gap_and_mono": gap_and_mono,  # backward compat (active baseline)
            "gap_and_mono_naive": gap_and_mono_naive,
            "gap_and_mono_natural": gap_and_mono_natural,
        }

        if imgs_poe is not None:
            print("  Computing image gap: AND vs PoE ...")
            gap_and_poe = compute_image_gap(imgs_and, imgs_poe, clip_eval, lpips_fn, device)
            traj_gap_poe_list = [
                compute_trajectory_gap(ta, tp_oe)
                for ta, tp_oe in zip(trackers_and, trackers_poe)
            ]
            traj_gap_poe = {
                "traj_mse_mean": float(sum(t["traj_mse_mean"] for t in traj_gap_poe_list) / len(traj_gap_poe_list)),
                "traj_cos_mean": float(sum(t["traj_cos_mean"] for t in traj_gap_poe_list) / len(traj_gap_poe_list)),
            }
            lat_gap_poe = compute_latent_gap(
                models["vae"], imgs_and, imgs_poe, device, dtype,
                latent_batch_size=args.latent_gap_batch_size,
            )
            metrics["gap_and_poe"] = {**gap_and_poe, **lat_gap_poe, **traj_gap_poe}
            print(f"    AND vs PoE:  CLIP={gap_and_poe['clip_cos']:.4f}")

        if not args._skip_inverter:
            print("  Computing image gap: AND vs p* ...")
            gap_and_pstar = compute_image_gap(imgs_and, imgs_pstar, clip_eval, lpips_fn, device)

            print("  Computing trajectory gap: AND vs p* ...")
            traj_gap_list = [
                compute_trajectory_gap(ta, tp)
                for ta, tp in zip(trackers_and, trackers_pstar)
            ]
            traj_gap = {
                "traj_mse_mean": float(sum(t["traj_mse_mean"] for t in traj_gap_list) / len(traj_gap_list)),
                "traj_cos_mean": float(sum(t["traj_cos_mean"] for t in traj_gap_list) / len(traj_gap_list)),
            }

            print("  Computing VAE latent gap ...")
            lat_gap = compute_latent_gap(
                models["vae"], imgs_and, imgs_pstar, device, dtype,
                latent_batch_size=args.latent_gap_batch_size,
            )

            metrics["gap_and_pstar"]     = {**gap_and_pstar, **lat_gap, **traj_gap}  # backward compat
            metrics["gap_and_pstar_inv"] = {**gap_and_pstar, **lat_gap, **traj_gap}

        if args.pez:
            print("  Computing image gap: AND vs PEZ ...")
            gap_and_pstar_pez = compute_image_gap(imgs_and, imgs_pstar_pez_all, clip_eval, lpips_fn, device)
            metrics["gap_and_pstar_pez"] = gap_and_pstar_pez
            metrics["pez_prompts"] = pez_prompts
            print(f"    AND vs PEZ:  CLIP={gap_and_pstar_pez['clip_cos']:.4f}")

        if args.z2t:
            print("  Computing image gap: AND vs Z2T ...")
            gap_and_pstar_z2t = compute_image_gap(imgs_and, imgs_pstar_z2t_all, clip_eval, lpips_fn, device)
            metrics["gap_and_pstar_z2t"] = gap_and_pstar_z2t
            metrics["z2t_prompts"] = z2t_prompts
            metrics["z2t_pool_mode"] = args.z2t_pool_mode
            metrics["z2t_source"] = "superdiff_and"
            print(f"    AND vs Z2T:  CLIP={gap_and_pstar_z2t['clip_cos']:.4f}")

        if args.vlm:
            print("  Computing image gap: AND vs VLM ...")
            gap_and_pstar_vlm = compute_image_gap(imgs_and, imgs_pstar_vlm_all, clip_eval, lpips_fn, device)
            metrics["gap_and_pstar_vlm"] = gap_and_pstar_vlm
            metrics["vlm_captions"] = vlm_captions
            print(f"    AND vs VLM:  CLIP={gap_and_pstar_vlm['clip_cos']:.4f}")

        with open(pair_dir / "gap_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        all_pair_results.append(metrics)
        all_seed_records.extend(per_seed_records)

        # ---- Visualisation 1: text decoding of p* ----
        decoded_text = []
        if vocab_prompts and not args._skip_inverter:
            print("  Decoding p* to nearest training prompts ...")
            avg_pooled = torch.stack(pred_pooled_list).mean(dim=0).to(device)
            decoded_text = decode_pooled_to_text(
                avg_pooled, vocab_prompts, models, device, top_k=3
            )
            print(f"  Nearest prompts to p*:")
            for rank, (prompt, sim) in enumerate(decoded_text, 1):
                print(f"    {rank}. \"{prompt}\"  (cos={sim:.4f})")
            metrics["decoded_p_star"] = [
                {"prompt": p, "cos_sim": round(s, 4)} for p, s in decoded_text
            ]
            with open(pair_dir / "gap_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        # ---- Visualisation 2: side-by-side comparison grid ----
        if not args._skip_inverter:
            print(
                "  Saving comparison grid "
                f"(active monolithic: {args.monolithic_baseline}) ..."
            )
            plot_comparison_grid(
                imgs_and, imgs_pstar, imgs_mono,
                decoded_text=decoded_text,
                c1=c1, c2=c2,
                out_path=img_dir / "comparison_grid.png",
                n_display=4,
                mono_prompt=mono_prompt_active,
            )
            save_comparison_grid_manifest(
                img_dir / "comparison_grid.png",
                c1=c1, c2=c2,
                seeds=args.seeds,
                pstar_source="inverter",
                n_display=4,
                mono_prompt=mono_prompt_active,
            )
            if mono_prompt_natural != mono_prompt_naive:
                alt_mode = "naive" if args.monolithic_baseline == "natural" else "natural"
                alt_prompt = mono_prompt_naive if alt_mode == "naive" else mono_prompt_natural
                alt_imgs = imgs_mono_naive if alt_mode == "naive" else imgs_mono_natural
                print(f"  Saving comparison grid ({alt_mode} monolithic) ...")
                plot_comparison_grid(
                    imgs_and, imgs_pstar, alt_imgs,
                    decoded_text=decoded_text,
                    c1=c1, c2=c2,
                    out_path=img_dir / f"comparison_grid_mono_{alt_mode}.png",
                    n_display=4,
                    mono_prompt=alt_prompt,
                )
                save_comparison_grid_manifest(
                    img_dir / f"comparison_grid_mono_{alt_mode}.png",
                    c1=c1, c2=c2,
                    seeds=args.seeds,
                    pstar_source="inverter",
                    n_display=4,
                    mono_prompt=alt_prompt,
                )

        # ---- Visualisation 3: trajectory MDS/PCA (seed 0, one plot per active p* source) ----
        _traj_sources = []
        if not args._skip_inverter:
            _traj_sources.append((trackers_pstar[0],     "SD3.5 (p* — Inverter)", ""))
        if args.poe and trackers_poe:
            _traj_sources.append((trackers_poe[0], "SD3.5 (PoE)", "_poe"))
        if args.vlm and trackers_pstar_vlm:
            _traj_sources.append((trackers_pstar_vlm[0], "SD3.5 (p* — VLM/BLIP-2)", "_vlm"))
        if args.pez and trackers_pstar_pez:
            _traj_sources.append((trackers_pstar_pez[0], "SD3.5 (p* — PEZ)", "_pez"))
        if args.z2t and trackers_pstar_z2t:
            _traj_sources.append((trackers_pstar_z2t[0], "SD3.5 (p* — Z2T)", "_z2t"))

        for _tp, _label, _suffix in _traj_sources:
            print(f"  Plotting trajectory {args.projection.upper()} [{_label}] (seed 0) ...")
            plot_trajectory_mds(
                tracker_and=trackers_and[0],
                tracker_pstar=_tp,
                tracker_mono=trackers_mono[0],
                c1=c1, c2=c2,
                out_path=traj_dir / f"trajectory_{args.projection}{_suffix}.png",
                method=args.projection,
                pstar_label=_label,
                tracker_c1=trackers_c1[0],
                tracker_c2=trackers_c2[0],
            )

        print(f"\n  Results for '{c1}' AND '{c2}':")
        if not args._skip_inverter:
            print(f"    AND vs p*:        CLIP={gap_and_pstar['clip_cos']:.4f}  "
                  f"lat_mse={lat_gap['lat_mse']:.4f}  "
                  f"traj_mse={traj_gap['traj_mse_mean']:.4f}  "
                  f"traj_cos={traj_gap['traj_cos_mean']:.4f}")
            if "lpips" in gap_and_pstar:
                print(f"    LPIPS (AND/p*):  {gap_and_pstar['lpips']:.4f}")
        print(
            f"    AND vs mono (active/{args.monolithic_baseline}): "
            f"CLIP={gap_and_mono['clip_cos']:.4f}"
        )
        print(f"    AND vs mono (naive):   CLIP={gap_and_mono_naive['clip_cos']:.4f}")
        print(f"    AND vs mono (natural): CLIP={gap_and_mono_natural['clip_cos']:.4f}")
        if imgs_poe is not None:
            print(f"    AND vs PoE:       CLIP={gap_and_poe['clip_cos']:.4f}  "
                  f"lat_mse={lat_gap_poe['lat_mse']:.4f}  "
                  f"traj_mse={traj_gap_poe['traj_mse_mean']:.4f}  "
                  f"traj_cos={traj_gap_poe['traj_cos_mean']:.4f}")

    # -------------------------------------------------------------------------
    # Determine which new pstar columns were actually written this run
    # -------------------------------------------------------------------------
    new_term_cols = [
        "pair_group",
        "monolithic_baseline",
        "mono_prompt_active",
        "mono_prompt_naive",
        "mono_prompt_natural",
        "d_T_mono",
        "d_T_mono_naive",
        "d_T_mono_natural",
        "d_T_c1",
        "d_T_c2",
    ]
    if args.anchor == "mean":
        new_term_cols += [
            "d_T_mono_meananchor",
            "d_T_mono_naive_meananchor",
            "d_T_mono_natural_meananchor",
            "d_T_c1_meananchor",
            "d_T_c2_meananchor",
        ]
    new_traj_cols = [
        "pair_group",
        "monolithic_baseline",
        "d_t_mono",
        "d_t_mono_naive",
        "d_t_mono_natural",
        "d_t_c1",
        "d_t_c2",
    ]
    new_gap_keys  = [
        "pair_group",
        "monolithic_baseline",
        "mono_prompt_active",
        "mono_prompt_naive",
        "mono_prompt_natural",
        "gap_and_mono",
        "gap_and_mono_naive",
        "gap_and_mono_natural",
    ]

    if not args._skip_inverter:
        new_term_cols += ["d_T_pstar", "d_T_pstar_inv"]
        if args.anchor == "mean":
            new_term_cols += ["d_T_pstar_meananchor", "d_T_pstar_inv_meananchor"]
        new_traj_cols += ["d_t_pstar", "d_t_pstar_inv"]
        new_gap_keys  += ["gap_and_pstar", "gap_and_pstar_inv"]
    if args.poe:
        new_term_cols.append("d_T_poe")
        if args.anchor == "mean":
            new_term_cols.append("d_T_poe_meananchor")
        new_traj_cols.append("d_t_poe")
        new_gap_keys.append("gap_and_poe")
    if args.pez:
        new_term_cols.append("d_T_pstar_pez")
        if args.anchor == "mean":
            new_term_cols.append("d_T_pstar_pez_meananchor")
        new_traj_cols.append("d_t_pstar_pez")
        new_gap_keys.append("gap_and_pstar_pez")
    if args.z2t:
        new_term_cols.append("d_T_pstar_z2t")
        if args.anchor == "mean":
            new_term_cols.append("d_T_pstar_z2t_meananchor")
        new_traj_cols.append("d_t_pstar_z2t")
        new_gap_keys.append("gap_and_pstar_z2t")
    if args.vlm:
        new_term_cols.append("d_T_pstar_vlm")
        if args.anchor == "mean":
            new_term_cols.append("d_T_pstar_vlm_meananchor")
        new_traj_cols.append("d_t_pstar_vlm")
        new_gap_keys.append("gap_and_pstar_vlm")

    # -------------------------------------------------------------------------
    # Write / merge output files
    # -------------------------------------------------------------------------
    metrics_dir = out_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary_path      = metrics_dir / "all_pairs_gap.json"
    seed_records_path = metrics_dir / "per_seed_distances.json"
    traj_path         = metrics_dir / "trajectory_distances.json"
    within_and_path   = metrics_dir / "within_and_distances.json"

    if args.merge and (summary_path.exists() or seed_records_path.exists()):
        print(f"\nMerging new columns into existing JSON files in {metrics_dir} ...")
        _merge_gap_json(summary_path, all_pair_results, new_gap_keys)
        _merge_json_column(seed_records_path, all_seed_records,
                           match_keys=("pair", "seed"), new_cols=new_term_cols)
        _merge_json_column(traj_path, all_traj_records,
                           match_keys=("pair", "seed", "step"), new_cols=new_traj_cols)
        # within_and is source-agnostic — only write if not yet present
        if not within_and_path.exists():
            within_and_path.write_text(json.dumps(all_within_and_records, indent=2))
        mode = "merged"
    else:
        with open(summary_path, "w") as f:
            json.dump(all_pair_results, f, indent=2)
        with open(seed_records_path, "w") as f:
            json.dump(all_seed_records, f, indent=2)
        with open(traj_path, "w") as f:
            json.dump(all_traj_records, f, indent=2)
        with open(within_and_path, "w") as f:
            json.dump(all_within_and_records, f, indent=2)
        mode = "written"

    print(f"\nAll results {mode} to {out_root}/")
    print(f"  metrics/per_seed_distances.json   → {seed_records_path}")
    print(f"  metrics/trajectory_distances.json → {traj_path}")
    print(f"  metrics/within_and_distances.json → {within_and_path}")
    print(f"  pairs/*/grid_assets.json          → per-pair assets for plot_gap_analysis.py plots 27–28")
    print(f"  grid export seed                  → {args._grid_seed}")
    if new_term_cols:
        print(f"  New columns: {new_term_cols}")
    print(f"\nTo visualize:")
    print(f"  python scripts/plot_gap_analysis.py --data-dir {args.output_dir}")


if __name__ == "__main__":
    main()
