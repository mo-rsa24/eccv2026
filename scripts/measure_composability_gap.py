"""
Phase 3: Measure the composability gap between SuperDiff-AND and p*.

For each held-out test pair (c₁, c₂):
  1. Generate images from SuperDiff-AND(c₁, c₂) with trajectory tracking
  2. Invert each image via a selected p* source (e.g. SD-IPC, inverter)
  3. Generate images from p* with the same initial noise + trajectory tracking
  4. Generate images from the model-family monolithic prompt baseline
  5. Compute gap metrics at three levels:
       - Image:      CLIP cosine similarity, LPIPS
       - Latent:     MSE in VAE latent space at the final step
       - Trajectory: step-wise MSE and cosine similarity

Results are saved as gap_metrics.json + image grids per pair.

Usage
-----
# Training-free quick run (no checkpoint needed; auto-timestamped output dir):
conda run -n superdiff python scripts/measure_composability_gap.py \
    --pstar-source sdipc --regime small \
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
    --pstar-source sdipc --regime small \
    --output-dir experiments/inversion/gap_analysis/my_run \
    --steps 50 --guidance 4.5

# Choose which seed is used for plot_gap_analysis.py --plot grid assets
# while running the trusted training-free p* path (currently SD-IPC-only):
conda run -n superdiff python scripts/measure_composability_gap.py \
    --pstar-source all --regime small --seeds 0 1 2 42 \
    --grid-seed 42

# SD 1.4 fair-comparison smoke test (2 seeds, true SD-IPC path):
conda run -n jaxstack python scripts/measure_composability_gap.py \
    --model-family sd14 \
    --pstar-source sdipc \
    --poe \
    --regime small \
    --seeds 42 43 \
    --grid-seed 42 \
    --steps 50 --guidance 7.5
"""

import argparse
import json
import math
import os
import re
import shutil
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

try:
    from taxonomy_manifest import (
        LARGE_REGIME_PAIRS as TAXONOMY_LARGE_REGIME_PAIRS,
        REPRESENTATIVE_PAIRS as TAXONOMY_REPRESENTATIVE_PAIRS,
        get_pair_taxonomy_from_slug,
        get_pair_taxonomy_record,
        pair_slug as taxonomy_pair_slug,
        taxonomy_manifest_payload,
    )
except ImportError:
    from scripts.taxonomy_manifest import (
        LARGE_REGIME_PAIRS as TAXONOMY_LARGE_REGIME_PAIRS,
        REPRESENTATIVE_PAIRS as TAXONOMY_REPRESENTATIVE_PAIRS,
        get_pair_taxonomy_from_slug,
        get_pair_taxonomy_record,
        pair_slug as taxonomy_pair_slug,
        taxonomy_manifest_payload,
    )

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from notebooks.dynamics import get_latents
from notebooks.utils import get_sd_models, get_sd3_models, get_sd3_text_embedding
from notebooks.composition_experiments import (
    LatentTrajectoryCollector,
    sample_sd3_with_trajectory_tracking,
    get_vel_sd3,
)
from scripts.trajectory_dynamics_experiment import (
    poe_sd_with_trajectory_tracking,
    sample_sd1_with_trajectory_tracking,
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


MAX_LEN_CLIP = 77
MAX_LEN_T5 = 256
DEFAULT_MODEL_ID_BY_FAMILY = {
    "sd35": "stabilityai/stable-diffusion-3.5-medium",
    "sd14": "CompVis/stable-diffusion-v1-4",
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CO3_BASE_DIRS = (
    PROJECT_ROOT / "experiments" / "eccv2026" / "taxonomy_qualitative",
    Path("/datasets/mmolefe/eccv2026/experiments/eccv2026/taxonomy_qualitative"),
)
ACTIVE_LOGICAL_ANCHOR_KEY = "poe"
ACTIVE_LOGICAL_ANCHOR_LABEL = "PoE"
CO3_TRAJECTORY_SUPPORT = "endpoint_only"


# ---------------------------------------------------------------------------
# Test pairs (held out from training)
# ---------------------------------------------------------------------------

_PAIRS_CORE = list(TAXONOMY_REPRESENTATIVE_PAIRS)
_PAIRS_LARGE = list(TAXONOMY_LARGE_REGIME_PAIRS)

assert len(_PAIRS_CORE) == 4, f"Expected 4 core pairs, got {len(_PAIRS_CORE)}"
assert len(_PAIRS_LARGE) == 24, f"Expected 24 taxonomy pairs, got {len(_PAIRS_LARGE)}"

# Pair-specific naturalized monolithic prompts for fair SD3.5 baseline comparison.
_PAIR_MONO_NATURAL_PROMPT = {
    ("a butterfly", "a flower meadow"): "a butterfly in a flower meadow",
    ("a camel", "a desert landscape"): "a camel in a desert landscape",
    ("a dolphin", "an ocean wave"): "a dolphin leaping through an ocean wave",
    ("a lion", "a savanna at sunset"): "a lion in a savanna at sunset",
    ("a dog", "oil painting style"): "an oil painting of a dog",
    ("a lighthouse", "watercolour style"): "a watercolour painting of a lighthouse",
    ("a bicycle", "sketch style"): "a sketch of a bicycle",
    ("a desk lamp", "a glacier"): "a desk lamp beside a glacier",
    ("a bathtub", "a streetlamp"): "a bathtub beside a streetlamp",
    ("a lab microscope", "a hay bale"): "a lab microscope beside a hay bale",
    ("a black grand piano", "a white vase"): "a black grand piano beside a white vase",
    ("a typewriter", "a cactus"): "a typewriter beside a cactus",
    ("a cat", "a dog"): "a cat and a dog side by side",
    ("a cat", "a bear"): "a cat and a bear side by side",
    ("a tiger", "a lion"): "a tiger and a lion side by side",
    ("a cat", "an owl"): "a cat and an owl",
    ("a bird", "a book"): "a bird and a book",
    ("a man with black hair and black shirt", "a red umbrella"):
        "a man with black hair and black shirt holding a red umbrella",
    ("a red bmw", "a white canopy truck"): "a red bmw and a white canopy truck",
    ("a red sports car", "a car with green tyres"): "a red sports car with green tyres",
    ("a glass teapot", "a teapot with golden handles"): "a glass teapot with golden handles",
    ("a blue backpack", "a backpack with orange zippers"): "a blue backpack with orange zippers",
    ("a lighthouse", "an ocean with stormy waves"): "a lighthouse by an ocean with stormy waves",
    ("a fox", "a snow-covered pine forest"): "a fox in a snow-covered pine forest",
}

assert len(_PAIR_MONO_NATURAL_PROMPT) == 24, (
    f"Expected natural prompts for 24 pairs, got {len(_PAIR_MONO_NATURAL_PROMPT)}"
)


def _pair_group(c1: str, c2: str) -> str:
    meta = get_pair_taxonomy_record(c1, c2)
    if meta is not None:
        return meta["taxonomy_group_key"]
    if (c1, c2) in set(_PAIRS_CORE):
        return "representative"
    return "other"


def _pair_monolithic_prompts(c1: str, c2: str) -> tuple:
    naive = f"{c1} and {c2}"
    natural = _PAIR_MONO_NATURAL_PROMPT.get((c1, c2), naive)
    return naive, natural


def _resolve_cli_pair_tokens(pair_tokens: list[str]) -> list[tuple[str, str]]:
    resolved_pairs: list[tuple[str, str]] = []
    for token in pair_tokens:
        meta = get_pair_taxonomy_from_slug(token)
        if meta is not None:
            resolved_pairs.append(tuple(meta["pair"]))
            continue

        parts = token.split("+", 1)
        if len(parts) == 2:
            c1, c2 = parts[0].strip(), parts[1].strip()
            if c1 and c2:
                resolved_pairs.append((c1, c2))
                continue

        raise ValueError(
            "Unrecognized --pairs entry "
            f"{token!r}. Use a taxonomy slug like "
            "'a_penguin_a_desert_landscape' or a literal pair formatted as "
            "'prompt_a+prompt_b'."
        )
    return resolved_pairs


REGIME_PAIRS = {
    "tiny":   _PAIRS_CORE[:1],   # 1 pair ×  1 seed =  1 record   (end-to-end smoke test)
    "small":  _PAIRS_CORE,
    "medium": _PAIRS_CORE,
    "large":  _PAIRS_LARGE,   # 24 pairs total
}

REGIME_SEEDS = {
    # "small":  list(range(8)),    #  4 pairs ×  8 seeds =  32 records
    # "small":  list(range(4)),    #  4 pairs ×  4 seeds =  16 records
    "tiny":   [42],               #  1 pair  ×  1 seed  =   1 record  (fast code test)
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
    if shift_factor is None:
        shift_factor = 0.0
    images = vae.decode(latents / vae.config.scaling_factor + shift_factor, return_dict=False)[0]
    return ((images / 2 + 0.5).clamp(0, 1)).float()


def model_display_name(model_family: str) -> str:
    if model_family == "sd14":
        return "SD 1.4"
    if model_family == "sdxl":
        return "SDXL"
    return "SD3.5"


def model_file_prefix(model_family: str) -> str:
    if model_family == "sd14":
        return "sd14"
    if model_family == "sdxl":
        return "sdxl"
    return "sd35"


def model_condition_label(model_family: str, label: str) -> str:
    return f"{model_display_name(model_family)} {label}"


def grid_export_labels(model_family: str) -> dict:
    family = model_display_name(model_family)
    return {
        "prompt_a":           f"{family} A",
        "prompt_b":           f"{family} B",
        "monolithic":         f"{family} A∧B",
        "monolithic_naive":   f"{family} A∧B (naive)",
        "monolithic_natural": f"{family} A∧B (natural)",
        "poe":                "PoE",
        "co3":                "CO3",
        "pstar_sdipc":        "PoE p*",
        "pstar_co3_sdipc":    "CO3 p*",
        "pstar_inv":          "p* CLIP inverter",
        "superdiff_fm_ode":   "SuperDiff A∧B",
        "pstar_z2t":          "p* Z2T",
    }


def default_co3_filename(model_family: str) -> str:
    return "co3_sd14.png" if model_family == "sd14" else "co3.png"


def _legacy_taxonomy_slug(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace(",", "")
        .replace("'", "")
        .replace("/", "")
    )


def _legacy_taxonomy_pair_slug(c1: str, c2: str) -> str:
    return f"{_legacy_taxonomy_slug(c1)}__x__{_legacy_taxonomy_slug(c2)}"


def resolve_co3_grid_image(
    c1: str,
    c2: str,
    *,
    taxonomy_group_key: str | None,
    pair_slug: str | None,
    co3_filename: str,
    co3_base: str = "",
) -> Path | None:
    """Resolve an external CO3 decoded image for grid export.

    The gap-analysis taxonomy uses manifest-style pair slugs, while older CO3
    qualitative runs wrote images under legacy ``groupX/.../__x__/`` paths.
    Search both layouts so representative-pair figures can attach CO3 images
    without changing the gap metrics or trajectory payloads.
    """
    candidate_bases = []
    if co3_base:
        candidate_bases.append(Path(co3_base))
    candidate_bases.extend(DEFAULT_CO3_BASE_DIRS)

    legacy_pair_slug = _legacy_taxonomy_pair_slug(c1, c2)
    seen: set[Path] = set()
    for base_dir in candidate_bases:
        if base_dir in seen or not base_dir.exists():
            continue
        seen.add(base_dir)

        if taxonomy_group_key and pair_slug:
            manifest_path = base_dir / taxonomy_group_key / pair_slug / co3_filename
            if manifest_path.exists():
                return manifest_path

        legacy_matches = sorted(base_dir.glob(f"*/{legacy_pair_slug}/{co3_filename}"))
        if legacy_matches:
            return legacy_matches[0]
    return None


def load_external_rgb_tensor(image_path: Path, image_size: int | None = None) -> torch.Tensor:
    """Load an external RGB image as a (1, 3, H, W) float tensor in [0, 1]."""
    img = Image.open(image_path).convert("RGB")
    if image_size is not None:
        img = img.resize((image_size, image_size), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0).float()


def make_shared_init_latents(
    *,
    seed: int,
    model_family: str,
    scheduler,
    models: dict,
    device: torch.device,
    dtype: torch.dtype,
    num_inference_steps: int,
    latent_size: int,
) -> tuple[torch.Tensor, float]:
    """Build the shared-noise latent used for all conditions of one pair-seed record."""
    if model_family == "sd14":
        init_latents = get_latents(
            scheduler,
            z_channels=models["unet"].config.in_channels,
            device=device,
            dtype=dtype,
            num_inference_steps=num_inference_steps,
            batch_size=1,
            latent_width=latent_size,
            latent_height=latent_size,
            seed=seed,
        )
        euler_sigma = float(getattr(scheduler, "init_noise_sigma", 1.0))
    else:
        gen = torch.Generator(device=device).manual_seed(seed)
        init_latents = torch.randn(
            1, 16, latent_size, latent_size,
            device=device, dtype=dtype, generator=gen,
        )
        euler_sigma = 1.0
    return init_latents, euler_sigma


def make_batched_init_latents(
    *,
    seeds,
    model_family: str,
    scheduler,
    models: dict,
    device: torch.device,
    dtype: torch.dtype,
    num_inference_steps: int,
    latent_size: int,
) -> tuple[torch.Tensor, float]:
    """Stack per-seed init latents into a single (N, C, H, W) batch tensor."""
    latents_list, sigma = [], None
    for seed in seeds:
        lat, sig = make_shared_init_latents(
            seed=seed,
            model_family=model_family,
            scheduler=scheduler,
            models=models,
            device=device,
            dtype=dtype,
            num_inference_steps=num_inference_steps,
            latent_size=latent_size,
        )
        latents_list.append(lat)  # each (1, C, H, W)
        sigma = sig
    return torch.cat(latents_list, dim=0), sigma  # (N, C, H, W), float


def _slice_tracker(tracker: "LatentTrajectoryCollector", i: int) -> "LatentTrajectoryCollector":
    """Return a single-sample view of a batched LatentTrajectoryCollector (zero-copy)."""
    s = LatentTrajectoryCollector.__new__(LatentTrajectoryCollector)
    s.num_steps = tracker.num_steps
    s.batch_size = 1
    s.shape = tracker.shape
    s.trajectories = tracker.trajectories[:, i : i + 1]  # (T+1, 1, C, H, W)
    s.velocities   = tracker.velocities[:, i : i + 1]    # (T,   1, C, H, W)
    s.sigmas       = tracker.sigmas                        # shared (T+1,)
    s.timesteps    = tracker.timesteps                     # shared (T,)
    return s


def _cat_trackers(trackers: list) -> "LatentTrajectoryCollector":
    """Concatenate a list of LatentTrajectoryCollectors along the batch dimension."""
    t0 = trackers[0]
    s = LatentTrajectoryCollector.__new__(LatentTrajectoryCollector)
    s.num_steps  = t0.num_steps
    s.batch_size = sum(t.batch_size for t in trackers)
    s.shape      = t0.shape
    s.trajectories = torch.cat([t.trajectories for t in trackers], dim=1)
    s.velocities   = torch.cat([t.velocities   for t in trackers], dim=1)
    s.sigmas    = t0.sigmas
    s.timesteps = t0.timesteps
    return s


@torch.no_grad()
def sample_sd1_with_precomputed_cond(
    latents: torch.Tensor,
    cond_emb: torch.Tensor,
    scheduler,
    unet,
    tokenizer,
    text_encoder,
    *,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    batch_size: int = 1,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
    model_id: str,
    euler_init_noise_sigma: float = 1.0,
) -> tuple:
    """DDIM rerun for SD 1.x with a precomputed (B, 77, 768) conditioning sequence."""
    from diffusers import DDIMScheduler
    import inspect as _inspect

    ddim = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    ddim.set_timesteps(num_inference_steps)

    latents = (latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    cond_emb = cond_emb.to(device=device, dtype=dtype)

    uncond_tokens = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_emb = text_encoder(uncond_tokens.input_ids.to(device))[0].to(dtype=dtype)

    tracker = LatentTrajectoryCollector(
        num_inference_steps,
        batch_size,
        latents.shape[1],
        latents.shape[2],
        latents.shape[3],
    )

    extra_step_kwargs = {}
    if "eta" in _inspect.signature(ddim.step).parameters:
        extra_step_kwargs["eta"] = 0.0

    from tqdm.auto import tqdm as _tqdm
    for i, t in _tqdm(enumerate(ddim.timesteps), total=num_inference_steps,
                      desc=f"SDIPC B={batch_size}", leave=False):
        latent_model_input = ddim.scale_model_input(latents, t)
        noise_pred_uncond = unet(latent_model_input, t, encoder_hidden_states=uncond_emb).sample
        noise_pred_cond = unet(latent_model_input, t, encoder_hidden_states=cond_emb).sample
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        tracker.store_step(i, latents, noise_pred, float(i) / num_inference_steps, t.item())
        latents = ddim.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    tracker.store_final(latents)
    return latents, tracker


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
    model_family: str = "sd35",
):
    """
    Save a 3-row comparison grid:
      Row 0 — SuperDiff AND
      Row 1 — model-family(p*)  [with decoded nearest prompts annotated below]
      Row 2 — model-family monolithic

    Each row shows n_display images side by side.
    Decoded text is printed below the p* row as a caption block.
    """
    N = min(n_display, imgs_and.shape[0])
    mono_prompt_label = mono_prompt or f"{c1} and {c2}"
    family = model_display_name(model_family)

    row_label_font_size = 36

    def label_row(imgs, label):
        labelled = [_add_text_label(imgs[i], label, font_size=row_label_font_size) for i in range(N)]
        return torch.stack(labelled)   # (N, 3, H, W)

    row_and   = label_row(imgs_and,   f"SuperDiff AND  ({c1}  +  {c2})")
    row_pstar = label_row(imgs_pstar, f"{family} ( p* — CLIP inverter )")
    row_mono  = label_row(imgs_mono,  f"{family} monolithic  \"{mono_prompt_label}\"")

    combined = torch.cat([row_and, row_pstar, row_mono], dim=0)  # (3N, 3, H, W)
    grid = make_grid(combined, nrow=N, padding=4, normalize=False)

    # Convert to PIL to add a caption strip at the bottom
    grid_pil = transforms.ToPILImage()(grid)
    caption_font_size = 26
    caption_font = _load_pil_font(caption_font_size, bold=False)
    caption_lines = [f"Nearest training prompts to p* (cosine similarity in {family} pooled space):"]
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
    model_family: str = "sd35",
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
    family = model_display_name(model_family)

    row_info = [
        ("Row 0", "SuperDiff AND",        f'"{c1}" ∧ "{c2}"'),
        ("Row 1", f"{family} (p* — {pstar_source})", None),
        ("Row 2", f"{family} monolithic", f'"{mono_prompt_label}"'),
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
# Decoded conditions grid (seed-matched, all available conditions)
# ---------------------------------------------------------------------------

def plot_decoded_conditions_grid(
    pair_dir: Path,
    decoded_images: dict,
    external_image_paths: dict,
    c1: str,
    c2: str,
    seed: int,
    model_family: str = "sd35",
    font_size: int = 28,
) -> "Path | None":
    """
    Save a single labeled decoded-image grid showing every available condition.

    Conditions are shown in _GRID_EXPORT_ORDER (minus the rarely-used
    monolithic_naive / monolithic_natural duplicates).  CO3 is included as an
    image panel even though it has no denoising trajectory.

    Terminology reminder
    --------------------
    monolithic        — backward-compatible alias; points to the active
                        baseline selected by --monolithic-baseline.
    monolithic_naive  — prompt "c1 and c2"   (simple concatenation).
    monolithic_natural— pair-specific naturalized prompt (e.g. "a fox in a
                        snow-covered pine forest").
    poe               — Product of Experts (active logical-composition anchor).
    pstar_sdipc       — SD 1.x/SD3.5 rerun from the PoE decoded image via
                        the closed-form SD-IPC projection.
    co3               — external CO3 decoded image (endpoint only — no trajectory).
    pstar_co3_sdipc   — SD-IPC rerun from the CO3 decoded image.

    Saved to: <pair_dir>/images/decoded_conditions_grid.png
    """
    # Omit duplicates that are already covered by "monolithic" slot.
    skip = {"monolithic_naive", "monolithic_natural"}
    order = [c for c in _GRID_EXPORT_ORDER if c not in skip]
    labels = grid_export_labels(model_family)

    available_panels = []
    for cond in order:
        img = decoded_images.get(cond)
        if img is None:
            src_path = (external_image_paths or {}).get(cond)
            if src_path and Path(src_path).exists():
                img = load_external_rgb_tensor(Path(src_path)).squeeze(0)
        if img is None:
            continue
        img_t = img.squeeze(0) if (torch.is_tensor(img) and img.ndim == 4) else img
        label = labels.get(cond, cond)
        labeled = _add_text_label(img_t.float().clamp(0, 1), label, font_size=font_size)
        available_panels.append(labeled)

    if not available_panels:
        print("  [skip] decoded conditions grid — no images available")
        return None

    n = len(available_panels)
    grid = make_grid(torch.stack(available_panels), nrow=n, padding=6, normalize=False)
    img_dir = pair_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_path = img_dir / "decoded_conditions_grid.png"
    save_image(grid, out_path, normalize=False)
    print(f"  Saved decoded conditions grid ({n} conditions): {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Unified trajectory figure — all conditions with CO3 endpoint-only note
# ---------------------------------------------------------------------------

def plot_unified_trajectory_all_conditions(
    trackers: dict,
    c1: str,
    c2: str,
    out_path: Path,
    method: str = "mds",
    co3_is_endpoint_only: bool = False,
    model_family: str = "sd35",
    logical_anchor_label: str = ACTIVE_LOGICAL_ANCHOR_LABEL,
) -> None:
    """
    Jointly project all available denoising trajectories to 2D and plot
    time-gradient curves for every condition in one figure.

    CO3 is an external image and has no denoising trajectory
    (CO3_TRAJECTORY_SUPPORT = 'endpoint_only').  When co3_is_endpoint_only
    is True, this is noted explicitly in the figure legend and title rather
    than silently omitting CO3.  The pstar_co3_sdipc condition DOES have a
    full trajectory and IS included when a tracker is present.

    Parameters
    ----------
    trackers : dict
        {cond_key: LatentTrajectoryCollector} for all conditions that have
        full denoising trajectories.  CO3 itself should NOT be included here
        (only pstar_co3_sdipc, which is the SD-IPC rerun from the CO3 image).
    co3_is_endpoint_only : bool
        When True, add a legend entry and title note explaining that CO3 is
        an external image without a trajectory.
    """
    if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
        print("  [skip] unified trajectory — matplotlib or scikit-learn not available")
        return
    if not trackers:
        print("  [skip] unified trajectory — no trackers available")
        return

    labels = grid_export_labels(model_family)
    traj_specs = [
        (cond, labels.get(cond, cond), tracker)
        for cond, tracker in trackers.items()
    ]

    traj_arrays = [_collect_trajectory_array(tk) for _, _, tk in traj_specs]
    T1 = traj_arrays[0].shape[0]
    if any(arr.shape[0] != T1 for arr in traj_arrays[1:]):
        print("  [skip] unified trajectory — trajectory length mismatch across conditions")
        return

    stacked = np.vstack(traj_arrays)

    if method == "pca":
        proj = PCA(n_components=2).fit_transform(stacked)
        axis_label = "PC"
    else:
        from sklearn.metrics import pairwise_distances
        dist = pairwise_distances(stacked, metric="euclidean")
        mds = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=42, normalized_stress="auto")
        proj = mds.fit_transform(dist)
        axis_label = "MDS"

    proj_specs = []
    start = 0
    for (cond, label, _), arr in zip(traj_specs, traj_arrays):
        end = start + arr.shape[0]
        proj_specs.append((cond, label, proj[start:end]))
        start = end

    fig, ax = plt.subplots(figsize=(15, 12))
    cmap = plt.get_cmap("viridis")
    norm_c = Normalize(vmin=0, vmax=T1 - 1)
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), "--", "-.", ":"]
    end_markers = ["o", "s", "^", "D", "P", "X", "*", "v"]

    def _draw_curve(pts, linestyle="-", marker="o"):
        for t_idx in range(len(pts) - 1):
            seg = np.array([[pts[t_idx], pts[t_idx + 1]]])
            lc = LineCollection(seg, colors=[cmap(norm_c(t_idx))], linewidths=2.5,
                                linestyles=linestyle, alpha=0.85)
            ax.add_collection(lc)
        ax.scatter(*pts[-1], s=180, zorder=5, marker=marker,
                   edgecolors="k", linewidths=1.0, color=cmap(norm_c(T1 - 1)))

    curve_meta = []
    for idx, (cond, label, pts) in enumerate(proj_specs):
        ls = line_styles[idx % len(line_styles)]
        mk = end_markers[idx % len(end_markers)]
        _draw_curve(pts, linestyle=ls, marker=mk)
        symbol = chr(ord("A") + idx)
        curve_meta.append((cond, label, pts[-1], ls, mk, symbol))

    for _, _, pt, _, _, symbol in curve_meta:
        ax.text(pt[0], pt[1], symbol, ha="center", va="center",
                fontsize=10, fontweight="bold", color="black", zorder=8)

    shared_start = proj_specs[0][2][0]
    ax.plot(*shared_start, "ko", markersize=11, zorder=6)
    ax.annotate("$x_T$ (shared)", shared_start, xytext=(10, -18),
                textcoords="offset points", fontsize=10, color="black")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_c)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Denoising step", pad=0.02)

    ax.autoscale()
    ax.margins(0.15)
    ax.set_xlabel(f"{axis_label} 1", fontsize=13)
    ax.set_ylabel(f"{axis_label} 2", fontsize=13)

    co3_note = (
        "\nCO3 omitted from trajectories — external image, endpoint-only "
        f"(CO3_TRAJECTORY_SUPPORT='{CO3_TRAJECTORY_SUPPORT}')"
        if co3_is_endpoint_only else ""
    )
    ax.set_title(
        f"Unified Latent Trajectory — All Conditions{co3_note}\n"
        f"Pair: {c1} × {c2}  |  logical anchor: {logical_anchor_label}",
        fontsize=12,
    )
    ax.grid(True, alpha=0.25)

    from matplotlib.lines import Line2D
    legend_color = cmap(norm_c(T1 - 1))
    legend_elements = []
    for _, label, _, ls, mk, symbol in curve_meta:
        legend_elements.append(
            Line2D([0], [0], color=legend_color, lw=2.5, linestyle=ls,
                   marker=mk, markersize=7, markerfacecolor=legend_color,
                   markeredgecolor="k", label=f"{symbol}: {label}")
        )
    legend_elements.append(
        Line2D([0], [0], marker="o", color="k", lw=0, markersize=9,
               label="Shared start $x_T$")
    )
    if co3_is_endpoint_only:
        legend_elements.append(
            Line2D([0], [0], color="gray", lw=0, marker="", linestyle="none",
                   label="CO3: endpoint only — no denoising trajectory")
        )
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=10,
              framealpha=0.9, edgecolor="0.7")

    plt.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved unified trajectory ({len(trackers)} conditions): {out_path}")


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
    "co3",
    "pstar_sdipc",
    "pstar_co3_sdipc",
    "pstar_inv",
    "superdiff_fm_ode",
    "pstar_z2t",
]

_GRID_EXPORT_LABELS = {
    "prompt_a":         "SD3.5 A",
    "prompt_b":         "SD3.5 B",
    "monolithic":       "SD3.5 A∧B",
    "monolithic_naive": "SD3.5 A∧B (naive)",
    "monolithic_natural": "SD3.5 A∧B (natural)",
    "poe":              "PoE",
    "co3":              "CO3",
    "pstar_sdipc":      "PoE p*",
    "pstar_co3_sdipc":  "CO3 p*",
    "pstar_inv":        "p* CLIP inverter",
    "superdiff_fm_ode": "SuperDiff A∧B",
    "pstar_z2t":        "p* Z2T",
}


def export_pair_grid_assets(
    pair_dir: Path,
    c1: str,
    c2: str,
    seed: int,
    decoded_images: dict,
    trackers: dict,
    external_image_paths: dict | None = None,
    pair_index: int = None,
    projection_method: str = "mds",
    source_prompts: dict = None,
    taxonomy_group_key: str | None = None,
    taxonomy_group_label: str | None = None,
    pair_slug: str | None = None,
    is_representative_pair: bool | None = None,
    model_family: str = "sd35",
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
    for cond, src_path in (external_image_paths or {}).items():
        if cond not in _GRID_EXPORT_ORDER or not src_path:
            continue
        src = Path(src_path)
        if not src.exists():
            continue
        out_img = assets_dir / f"{cond}{src.suffix or '.png'}"
        shutil.copy2(src, out_img)
        image_paths[cond] = str(out_img.relative_to(pair_dir))

    traj_payload = None
    label_map = grid_export_labels(model_family)
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
                    cond: label_map.get(cond, cond)
                    for cond in ordered_trackers.keys()
                },
            }
        except Exception as exc:
            print(f"  Warning: failed to export trajectory projection for grid assets: {exc}")

    merged_source_prompts = dict(existing.get("source_prompts", {}))
    merged_source_prompts.update(source_prompts or {})

    payload = {
        "pair": [c1, c2],
        "pair_slug": pair_slug or existing.get("pair_slug"),
        "pair_index": (
            int(pair_index) if pair_index is not None else existing.get("pair_index")
        ),
        "taxonomy_group_key": taxonomy_group_key or existing.get("taxonomy_group_key"),
        "taxonomy_group_label": taxonomy_group_label or existing.get("taxonomy_group_label"),
        "is_representative_pair": (
            bool(is_representative_pair)
            if is_representative_pair is not None
            else existing.get("is_representative_pair")
        ),
        "seed": int(seed),
        "prompt_key_map": {"A": c1, "B": c2},
        "condition_labels": label_map,
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
    pstar_label: str = "SD3.5 (p* — CLIP inverter)",
    tracker_c1: LatentTrajectoryCollector = None,
    tracker_c2: LatentTrajectoryCollector = None,
    model_family: str = "sd35",
    logical_anchor_label: str = ACTIVE_LOGICAL_ANCHOR_LABEL,
):
    """
    Jointly project the latent trajectories of:
      - the active logical anchor
      - model-family(p*)
      - model-family monolithic
      - model-family(c1) and model-family(c2) single-concept baselines (optional)
    into 2D via PCA or MDS and plot time-coloured curves.

    All three start from the same x_T (shared noise), so the origin is common.

    Parameters
    ----------
    pstar_label : str
        Human-readable label for the p* source, e.g. "SD3.5 (p* — SD-IPC)",
        "SD3.5 (p* — PEZ)", "SD3.5 (p* — Z2T)", "SD3.5 (p* — CLIP inverter)".
    """
    if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
        print("  [skip] trajectory MDS — matplotlib or scikit-learn not available")
        return
    family = model_display_name(model_family)

    traj_specs = [
        (logical_anchor_label, tracker_and),
        (pstar_label, tracker_pstar),
        (f'{family} monolithic "{c1} and {c2}"', tracker_mono),
    ]
    if tracker_c1 is not None:
        traj_specs.append((f'{family} solo "{c1}"', tracker_c1))
    if tracker_c2 is not None:
        traj_specs.append((f'{family} solo "{c2}"', tracker_c2))

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
        f"Latent Trajectory: {logical_anchor_label} vs {pstar_label} vs Monolithic{suffix}\n"
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
# SD-IPC: closed-form CLIP image -> SD3.5 conditioning
# ---------------------------------------------------------------------------

def build_sdipc_runtime(
    models,
    clip_model,
    clip_model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    model_family: str = "sd35",
):
    """Prepare SD-IPC projection matrices and cached null conditioning."""
    from transformers import CLIPProcessor

    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

    with torch.no_grad():
        inv_text = torch.linalg.pinv(
            clip_model.text_projection.weight.float(),
            atol=0.3,
        )
        visual_proj = clip_model.visual_projection.weight.float()

        ids_l_null = models["tokenizer"](
            "",
            padding="max_length",
            max_length=MAX_LEN_CLIP,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        out_l_null = models["text_encoder"](
            input_ids=ids_l_null,
            output_hidden_states=True,
        )
        # SD 1.4 UNet cross-attention was trained on last_hidden_state (post
        # final_layer_norm, norm ≈ 27/token).  hidden_states[-2] is the
        # penultimate layer BEFORE layer norm and can have norms >200, which
        # creates a scale explosion in the CFG guidance at the BOS position,
        # collapsing SD-IPC reruns to pure noise.
        # SD 3.5 intentionally uses the penultimate CLIP-L layer (SD3 paper).
        if model_family == "sd14":
            null_clip_l_seq = out_l_null.last_hidden_state.to(device=device, dtype=dtype)
        else:
            null_clip_l_seq = out_l_null.hidden_states[-2].to(device=device, dtype=dtype)

        runtime = {
            "clip_processor": clip_processor,
            "visual_proj": visual_proj,
            "inv_text": inv_text,
            "null_clip_l_seq": null_clip_l_seq,
        }

        if model_family == "sd35":
            ids_g_null = models["tokenizer_2"](
                "",
                padding="max_length",
                max_length=MAX_LEN_CLIP,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            out_g_null = models["text_encoder_2"](
                input_ids=ids_g_null,
                output_hidden_states=True,
            )
            null_clip_g_seq = out_g_null.hidden_states[-2].to(device=device, dtype=dtype)
            null_clip_g_pooled = out_g_null.text_embeds.to(device=device, dtype=dtype)

            ids_t5_null = models["tokenizer_3"](
                "",
                padding="max_length",
                max_length=MAX_LEN_T5,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            null_t5_seq = models["text_encoder_3"](
                input_ids=ids_t5_null
            ).last_hidden_state.to(device=device, dtype=dtype)

            runtime.update({
                "null_clip_g_seq": null_clip_g_seq,
                "null_clip_g_pooled": null_clip_g_pooled,
                "null_t5_seq": null_t5_seq,
            })

    return runtime


@torch.no_grad()
def sdipc_project_image(
    img_01: torch.Tensor,
    clip_model,
    clip_processor,
    visual_proj: torch.Tensor,
    inv_text: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Project image(s) into CLIP-L text space using the SD-IPC closed form.

    Accepts img_01 of shape (1, 3, H, W) or (N, 3, H, W).
    Returns (1, D) or (N, D) respectively.
    """
    from torchvision import transforms as T

    pils = [T.ToPILImage()(img_01[i].detach().cpu().clamp(0, 1)) for i in range(img_01.shape[0])]
    pixel_values = clip_processor(images=pils, return_tensors="pt").pixel_values.to(device)
    pooler_out = clip_model.vision_model(pixel_values=pixel_values).pooler_output.float()
    joint = pooler_out @ visual_proj.to(device).T
    text_space = joint @ inv_text.to(device).T
    text_space = text_space / text_space.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return 27.5 * text_space


@torch.no_grad()
def sdipc_to_sd3_cond(
    proj_vec: torch.Tensor,
    runtime: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    """Convert an SD-IPC projection into SD3.5 prompt_embeds + pooled projections."""
    pv = proj_vec.to(device=device, dtype=dtype)

    seq_l = torch.zeros_like(runtime["null_clip_l_seq"])
    seq_l[:, 0] = runtime["null_clip_l_seq"][:, 0]
    seq_l[:, 1:] = pv.unsqueeze(1)

    seq_l_pad = torch.nn.functional.pad(seq_l, (0, 4096 - 768))
    seq_g_pad = torch.nn.functional.pad(runtime["null_clip_g_seq"], (0, 4096 - 1280))
    clip_seq = torch.cat([seq_l_pad, seq_g_pad], dim=1)
    prompt_embeds = torch.cat([clip_seq, runtime["null_t5_seq"]], dim=1)

    pooled_projections = torch.cat([pv, runtime["null_clip_g_pooled"]], dim=-1)
    return prompt_embeds, pooled_projections


@torch.no_grad()
def sdipc_to_sd14_cond(
    proj_vec: torch.Tensor,
    runtime: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert SD-IPC projection(s) into SD 1.4 cross-attention sequence(s).

    proj_vec: (1, D) or (N, D) — returns (1, 77, 768) or (N, 77, 768).
    """
    pv = proj_vec.to(device=device, dtype=dtype)   # (N, D)
    N = pv.shape[0]
    seq = runtime["null_clip_l_seq"].expand(N, -1, -1).clone()  # (N, 77, 768)
    seq[:, 0] = runtime["null_clip_l_seq"][:, 0].expand(N, -1)
    seq[:, 1:] = pv.unsqueeze(1)  # (N, 1, D) → broadcast to (N, 76, D)
    return seq


@torch.no_grad()
def run_sdipc_rerun(
    *,
    source_img: torch.Tensor,
    init_latents: torch.Tensor,
    model_family: str,
    models: dict,
    scheduler,
    sdipc_runtime: dict,
    clip_eval,
    device: torch.device,
    dtype: torch.dtype,
    guidance: float,
    steps: int,
    model_id: str,
    sd14_euler_sigma: float,
    uncond_embeds: torch.Tensor | None = None,
    uncond_pooled: torch.Tensor | None = None,
) -> tuple[torch.Tensor, LatentTrajectoryCollector]:
    """Run the SD-IPC closed-form rerun from a decoded source image."""
    proj_sdipc = sdipc_project_image(
        source_img,
        clip_model=clip_eval,
        clip_processor=sdipc_runtime["clip_processor"],
        visual_proj=sdipc_runtime["visual_proj"],
        inv_text=sdipc_runtime["inv_text"],
        device=device,
    )
    if model_family == "sd14":
        cond_sdipc = sdipc_to_sd14_cond(
            proj_sdipc,
            runtime=sdipc_runtime,
            device=device,
            dtype=dtype,
        )
        lat_sdipc, tracker_sdipc = sample_sd1_with_precomputed_cond(
            latents=init_latents,
            cond_emb=cond_sdipc,
            scheduler=scheduler,
            unet=models["unet"],
            tokenizer=models["tokenizer"],
            text_encoder=models["text_encoder"],
            guidance_scale=guidance,
            num_inference_steps=steps,
            batch_size=1,
            device=device,
            dtype=dtype,
            model_id=model_id,
            euler_init_noise_sigma=sd14_euler_sigma,
        )
    else:
        cond_sdipc, pooled_sdipc = sdipc_to_sd3_cond(
            proj_sdipc,
            runtime=sdipc_runtime,
            device=device,
            dtype=dtype,
        )
        lat_sdipc, tracker_sdipc = sample_sd3_with_precomputed_cond(
            latents=init_latents,
            cond_embeds=cond_sdipc,
            cond_pooled=pooled_sdipc,
            uncond_embeds=uncond_embeds,
            uncond_pooled=uncond_pooled,
            scheduler=scheduler,
            transformer=models["transformer"],
            guidance_scale=guidance,
            num_inference_steps=steps,
            device=device,
            dtype=dtype,
        )
    return lat_sdipc, tracker_sdipc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Measure composability gap")
    p.add_argument(
        "--model-family",
        choices=["sd35", "sd14"],
        default="sd35",
        help=(
            "Backend family used for AND / p* / monolithic / PoE evaluation. "
            "'sd35' keeps the current SD3.5 flow-matching path; "
            "'sd14' switches to the original SD 1.4 latent-diffusion path."
        ),
    )
    p.add_argument("--ckpt",       default=None,
                   help="Path to trained CLIP-inverter checkpoint.  Required only when "
                        "--pstar-source inverter (or omitted).  Other sources skip the load.")
    p.add_argument(
        "--model-id",
        default=None,
        help=(
            "Model identifier override. Defaults to "
            f"{DEFAULT_MODEL_ID_BY_FAMILY['sd35']} for --model-family sd35 and "
            f"{DEFAULT_MODEL_ID_BY_FAMILY['sd14']} for --model-family sd14."
        ),
    )
    p.add_argument("--output-dir", default="",
                   help="Root output directory.  If empty (default), a timestamped "
                        "directory is created under experiments/inversion/gap_analysis/.")
    p.add_argument("--data-dir",   default="experiments/inversion/training_data",
                   help="Path to training data directory (for text-decoding vocabulary)")
    p.add_argument("--regime", choices=["tiny", "small", "medium", "large"], default=None,
                   help=(
                       "Preset scale for pairs and seeds:\n"
                       "  tiny   —  1 pair  ×  1 seed  =   1 record  (end-to-end smoke test)\n"
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
        "--pairs",
        nargs="+",
        default=None,
        help=(
            "Explicit pair override. Each entry may be either a taxonomy pair slug "
            "(for example a_penguin_a_desert_landscape) or a literal pair formatted "
            "as 'prompt_a+prompt_b'. Overrides --regime defaults."
        ),
    )
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
            "`<family>_monolithic.png`, grid_assets['monolithic']). "
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
    p.add_argument("--pair-start", type=int, default=None,
                   help="First pair index (inclusive) to process. Use with --pair-end for "
                        "multi-GPU parallelism across a --regime run.")
    p.add_argument("--pair-end",   type=int, default=None,
                   help="Last pair index (exclusive) to process.")
    p.add_argument("--seed-batch-size", type=int, default=8,
                   help="Number of seeds to process in each batched UNet call. "
                        "Reduce if CUDA OOM. Default 8 uses ~15 GB on SD1.4 fp16.")
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
    p.add_argument(
        "--co3-base",
        default="",
        help=(
            "Optional base directory to search for external CO3 decoded images that "
            "should be attached to pairs/*/grid_assets.json for plot 27. "
            "If unset, the repo-local and legacy /datasets taxonomy_qualitative "
            "directories are searched automatically."
        ),
    )
    p.add_argument(
        "--co3-filename",
        default="",
        help=(
            "Filename to look for inside CO3 pair directories. Defaults to "
            "co3_sd14.png for --model-family sd14 and co3.png otherwise."
        ),
    )
    p.add_argument(
        "--run-co3",
        action="store_true",
        help=(
            "Run CO3 in-process (SD 1.4) using the same shared x_T as all other "
            "conditions. Requires --model-family sd14. CO3 is endpoint-only "
            "(no denoising trajectory), but its image is generated from the correct "
            "starting noise for a fair comparison. Combine with --sdipc to also run "
            "the SD-IPC rerun from the CO3 output."
        ),
    )
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
    # --- SD-IPC flags ---
    p.add_argument("--sdipc", action="store_true",
                   help="Also run SD-IPC closed-form projection as a training-free p* source.")
    # Deprecated VLM flags kept as aliases so old commands still parse.
    p.add_argument("--vlm", action="store_true",
                   help="Deprecated alias for --sdipc. VLM captioning is decommissioned.")
    p.add_argument("--vlm-model-id", default="Salesforce/blip2-opt-2.7b",
                   help=argparse.SUPPRESS)
    p.add_argument("--vlm-max-tokens", type=int, default=60,
                   help=argparse.SUPPRESS)
    p.add_argument(
        "--vlm-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help=argparse.SUPPRESS,
    )
    # --- Convenience shorthand + merge mode ---
    p.add_argument(
        "--pstar-source",
        choices=["inverter", "pez", "sdipc", "z2t", "all", "vlm"],
        default=None,
        help=(
            "Shorthand for selecting a single p* source.  Maps to the "
            "individual flags above.  Use with --merge to accumulate multiple "
            "sources into the same output JSON files across separate runs:\n"
            "  inverter  — trained CLIP inverter  (default, always on)\n"
            "  pez       — discrete token optimisation  (--pez)\n"
            "  sdipc     — closed-form SD-IPC projection  (--sdipc)\n"
            "  z2t       — Zero2Text ridge regression  (--z2t)\n"
            "  all       — SD-IPC only (inverter / PEZ / Z2T excluded)\n"
            "  vlm       — deprecated alias for sdipc"
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
    p.add_argument(
        "--paper-only",
        action="store_true",
        help=(
            "Restrict to paper-pipeline conditions only: c1, c2, monolithic (active baseline), "
            "poe, and pstar_sdipc (SD-IPC closed-form, SD 1.4 only). "
            "Skips monolithic_naive, monolithic_natural, CO3, pstar_inv, pstar_z2t, pstar_pez, "
            "and decoded_conditions_grid.png. "
            "Defaults to --model-family sd14 (required for pstar_sdipc); "
            "override with explicit --model-family sd35 if needed. "
            "Sets --sdipc automatically; ignores --pez, --z2t, --run-co3."
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
                "decoded_p_star",
            ):
                if extra_key in new_rec:
                    existing[lookup[slug]][extra_key] = new_rec[extra_key]
        else:
            existing.append(new_rec)

    path.write_text(json.dumps(existing, indent=2))


def main():
    args   = parse_args()

    # ---- Paper-only defaults to SD 1.4 (pstar_sdipc is SD 1.4 only) ----
    if args.paper_only and args.model_family == "sd35":
        print("[PAPER-ONLY] Switching to SD 1.4 (pstar_sdipc requires SD 1.4)")
        args.model_family = "sd14"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if args.dtype == "float16" else torch.bfloat16
    args.model_id = args.model_id or DEFAULT_MODEL_ID_BY_FAMILY[args.model_family]
    args.co3_filename = args.co3_filename or default_co3_filename(args.model_family)
    family_label = model_display_name(args.model_family)
    family_prefix = model_file_prefix(args.model_family)

    if args.vlm:
        print("Warning: --vlm is deprecated and now maps to SD-IPC. BLIP-2 captioning is no longer used.")
        args.sdipc = True
        args.vlm = False

    if not args.poe:
        print(
            f"Enabling --poe automatically: Phase 1 reachability now uses {ACTIVE_LOGICAL_ANCHOR_LABEL} "
            "as the active logical-composition anchor."
        )
        args.poe = True

    # Keep the SD3.5 SuperDiff config aligned with trajectory_dynamics_experiment.py grid runs.
    # That script uses the dynamic-kappa fm_ode path by default.
    if args.model_family == "sd35" and args.superdiff_kappa_mode != "dynamic":
        print(
            "Aligning SuperDiff config with trajectory_dynamics_experiment.py: "
            f"overriding --superdiff-kappa-mode {args.superdiff_kappa_mode!r} -> 'dynamic'."
        )
        args.superdiff_kappa_mode = "dynamic"

    # ---- Resolve explicit pair override / --regime into test pairs and seeds ----
    if args.pairs is not None:
        test_pairs = _resolve_cli_pair_tokens(args.pairs)
        if args.seeds is None:
            args.seeds = list(range(8))   # legacy default
        print(f"Explicit pairs: {len(test_pairs)}  "
              f"({len(test_pairs)} pairs × {len(args.seeds)} seeds = "
              f"{len(test_pairs) * len(args.seeds)} records)")
    elif args.regime is not None:
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

    # ---- Optional pair-slice for multi-GPU parallelism ----
    # Use --pair-start / --pair-end to assign disjoint subsets to each GPU.
    # Example: 24 pairs across 4 GPUs → --pair-start 0 --pair-end 6 on GPU 0, etc.
    if args.pair_start is not None or args.pair_end is not None:
        ps = args.pair_start or 0
        pe = args.pair_end   or len(test_pairs)
        test_pairs = test_pairs[ps:pe]
        print(f"Pair slice [{ps}:{pe}] → {len(test_pairs)} pairs on this worker.")

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
    print(f"Active logical anchor: {ACTIVE_LOGICAL_ANCHOR_LABEL}")

    # ---- Resolve --pstar-source shorthand into individual flags ----
    if args.pstar_source is not None:
        src = args.pstar_source
        if src == "vlm":
            print("Warning: --pstar-source vlm is deprecated and now maps to --pstar-source sdipc.")
            src = "sdipc"
        # Project policy: --pstar-source all runs only SD-IPC.
        if src == "all":
            args.pez = False
            args.sdipc = True
            args.z2t = False
            args._skip_inverter = True
        else:
            # "inverter" = default run (no extra flags needed)
            # Any other single source disables the trained inverter to save time.
            args.pez = (src == "pez")
            args.sdipc = (src == "sdipc")
            args.z2t = (src == "z2t")
            args._skip_inverter = (src != "inverter")
    else:
        args._skip_inverter = False

    if args.model_family == "sd14" and not args._skip_inverter:
        raise ValueError(
            "--model-family sd14 does not support the current trained CLIP inverter path. "
            "Use --pstar-source sdipc, --pstar-source pez, or --pstar-source z2t."
        )

    # ---- Paper-only mode: restrict to core paper conditions ----
    if args.paper_only:
        print("\n[PAPER-ONLY MODE] Restricting to paper-pipeline conditions:")
        print(f"  • Model: {args.model_family.upper()}")
        print("  • c1 (Concept A)")
        print("  • c2 (Concept B)")
        print("  • monolithic (active baseline)")
        print("  • poe (logical anchor)")
        print("  • pstar_sdipc (p* with SD-IPC closed-form)")
        print("\nSkipping: naive/natural baselines, CO3, pstar variants (inverter/PEZ/Z2T), decoded_conditions_grid.png\n")
        args.pez = False
        args.z2t = False
        args.run_co3 = False
        args.sdipc = True
        args._skip_inverter = True
        args._paper_only = True
    else:
        args._paper_only = False

    # ---- Resolve output root (auto-timestamp if not explicitly set) ----
    if not args.output_dir:
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        regime_tag  = f"{args.model_family}_{args.regime}_" if args.regime else f"{args.model_family}_"
        args.output_dir = f"experiments/inversion/gap_analysis/{regime_tag}{timestamp}"
        print(f"Output directory (auto): {args.output_dir}")

    out_root    = Path(args.output_dir)
    latent_size = args.image_size // 8

    if args.model_family == "sd35":
        print(
            "SuperDiff kappa mode: "
            f"{args.superdiff_kappa_mode}"
            + (
                f" (kappa={args.superdiff_fixed_kappa:.3f})"
                if args.superdiff_kappa_mode == "fixed"
                else ""
            )
        )
    else:
        print("SuperDiff backend: SD 1.4 author stochastic mode")
    print(f"Model family: {args.model_family} ({family_label})")
    print(f"Model id: {args.model_id}")

    scheduler = None
    if args.model_family == "sd14":
        print(f"Loading {family_label} ...")
        models = get_sd_models(model_id=args.model_id, dtype=dtype, device=device)
        from diffusers import EulerDiscreteScheduler
        scheduler = EulerDiscreteScheduler.from_pretrained(
            args.model_id, subfolder="scheduler"
        )
        uncond_embeds = None
        uncond_pooled = None
    else:
        print(f"Loading {family_label} ...")
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
                "Pass --pstar-source sdipc/pez/z2t to skip it, or supply --ckpt <path>."
            )
        print(f"Loading inverter from {args.ckpt} ...")
        inverter   = load_inverter(args.ckpt, clip_model_id=args.clip_model_id, device=device)
        inverter   = inverter.eval().to(dtype=torch.float32)
        preprocess = make_clip_preprocessor(device)

    # ---- Load CLIP for evaluation (separate from inverter backbone) ----
    print("Loading CLIP for evaluation ...")
    from transformers import CLIPModel
    clip_eval = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval()

    sdipc_runtime = None
    if args.sdipc:
        print("Preparing SD-IPC runtime ...")
        sdipc_runtime = build_sdipc_runtime(
            models=models,
            clip_model=clip_eval,
            clip_model_id=args.clip_model_id,
            device=device,
            dtype=dtype,
            model_family=args.model_family,
        )

    # ---- Load CLIP tokenizer for PEZ / Z2T ----
    clip_tokenizer = None
    if args.pez or args.z2t:
        from transformers import CLIPTokenizer
        clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_id)
        print(f"  CLIP tokenizer loaded (vocab size {clip_tokenizer.vocab_size})")
        if args.z2t:
            print(f"  Z2T pool mode: {args.z2t_pool_mode}")

    # ---- Load LPIPS if available ----
    lpips_fn = None
    if LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net="vgg")  # kept on CPU; images moved to CPU at call time

    # ---- Load CO3 in-process model (SD 1.4 only) ----
    co3_model = None
    if args.run_co3:
        if args.model_family != "sd14":
            raise ValueError("--run-co3 requires --model-family sd14")
        import sys as _sys
        _sys.path.insert(0, str(PROJECT_ROOT / "compositions" / "co3"))
        from composers.Co3 import Co3 as _Co3Class
        from composers.config.co3_config import Co3Config as _Co3Config
        _c1_0, _c2_0 = test_pairs[0]
        _co3_init_cfg = _Co3Config(
            prompt=f"{_c1_0}+{_c2_0}",
            prompt_orig=f"{_c1_0} and {_c2_0}",
            sd_version="1.4",
            resolution_h=args.image_size,
            resolution_w=args.image_size,
            n_timesteps=args.steps,
            guidance_scale=7.5,  # CO3 default (0.8) is tuned for SDXL; SD 1.4 needs standard CFG
            output_path=str(out_root / "co3_tmp"),
            output_path_all=str(out_root / "co3_tmp"),
        )
        co3_model = _Co3Class(_co3_init_cfg)
        print(f"CO3 model loaded (SD 1.4, {args.image_size}×{args.image_size}, {args.steps} steps)")

    # ---- Pre-compute unconditional conditioning ----
    if args.model_family == "sd35":
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
    if args.model_family == "sd35":
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
        taxonomy_meta = get_pair_taxonomy_record(c1, c2) or {}
        pair_group = _pair_group(c1, c2)
        taxonomy_group_key = taxonomy_meta.get("taxonomy_group_key", pair_group)
        taxonomy_group_label = taxonomy_meta.get("taxonomy_group_label", pair_group)
        is_representative_pair = bool(taxonomy_meta.get("is_representative_pair", False))
        mono_prompt_naive, mono_prompt_natural = _pair_monolithic_prompts(c1, c2)
        mono_prompt_active = (
            mono_prompt_natural
            if args.monolithic_baseline == "natural"
            else mono_prompt_naive
        )
        pair_slug  = taxonomy_meta.get("pair_slug", taxonomy_pair_slug(c1, c2))
        pair_dir   = out_root / "pairs" / pair_slug
        img_dir    = pair_dir / "images"
        traj_dir   = pair_dir / "trajectories"
        pair_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)
        traj_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Pair: '{c1}' AND '{c2}'")
        print(f"Group: {taxonomy_group_label} [{taxonomy_group_key}]")
        print(f"Monolithic (naive):   \"{mono_prompt_naive}\"")
        print(f"Monolithic (natural): \"{mono_prompt_natural}\"")
        print(
            f"Monolithic (active/{args.monolithic_baseline}): "
            f"\"{mono_prompt_active}\""
        )
        print(f"{'='*60}")

        # Legacy *_and collections are kept for backward-compatible filenames and
        # JSON key namespaces (`gap_and_*`, `d_within_and`, ...), but from here on
        # they store the active logical anchor trajectory/image, which is PoE.
        images_and   = []
        images_pstar = []
        images_mono  = []  # active baseline (backward-compatible key)
        images_mono_naive = [] if not args._paper_only else None
        images_mono_natural = [] if not args._paper_only else None

        trackers_and   = []
        trackers_pstar = []
        trackers_mono  = []  # active baseline (backward-compatible key)
        trackers_mono_naive = [] if not args._paper_only else None
        trackers_mono_natural = [] if not args._paper_only else None
        pred_pooled_list = []   # accumulate across seeds for text decoding

        # Optional additional p* sources
        images_pstar_pez = [] if args.pez else None
        trackers_pstar_pez = [] if args.pez else None
        pez_prompts = [] if args.pez else None
        images_pstar_z2t = [] if args.z2t else None
        trackers_pstar_z2t = [] if args.z2t else None
        z2t_prompts = [] if args.z2t else None
        images_pstar_sdipc = [] if args.sdipc else None
        trackers_pstar_sdipc = [] if args.sdipc else None

        images_co3 = [] if (co3_model is not None and not args._paper_only) else None
        # CO3 p* (SD-IPC rerun from the in-process CO3 image, per seed)
        # Only populated when --run-co3 and --sdipc are both active.
        images_pstar_co3_sdipc = [] if (args.sdipc and co3_model is not None and not args._paper_only) else None
        trackers_pstar_co3_sdipc = [] if (args.sdipc and co3_model is not None and not args._paper_only) else None

        # Update CO3 prompts for this pair (re-uses the loaded model across pairs)
        if co3_model is not None:
            co3_model.config.prompt = f"{c1}+{c2}"
            co3_model.config.prompt_orig = f"{c1} and {c2}"
            co3_model.prepare_prompts(co3_model.config)
            co3_model.prepare_embeds()

        images_c1        = []
        images_c2        = []
        trackers_c1      = []
        trackers_c2      = []
        per_seed_records = []

        # ---- SD 1.4 seed-level batch pre-computation ----
        # When incompatible features (inverter, PEZ, Z2T, CO3) are all disabled,
        # batch all N seeds for one pair into a single batch_size=N UNet call.
        # This reduces UNet invocations from 11*N per pair to 11 (N-wide batches).
        _batched = (
            args.model_family == "sd14"
            and args._skip_inverter
            and not args.pez
            and not args.z2t
            and co3_model is None
        )
        _b = {}  # populated below when _batched is True
        if _batched:
            _N   = len(args.seeds)
            _SBS = args.seed_batch_size
            _chunks = [args.seeds[i : i + _SBS] for i in range(0, _N, _SBS)]
            print(
                f"  [batch] {_N} seeds → {len(_chunks)} chunk(s) of ≤{_SBS} "
                f"(--seed-batch-size {_SBS})"
            )
            # Accumulators: lists of tensors / trackers, one entry per chunk.
            _acc: dict = {k: [] for k in [
                "init",
                "lat_and", "tr_and", "img_and",
                "lat_sdipc", "tr_sdipc", "img_sdipc",
                "lat_mono",  "tr_mono",  "img_mono",
                "lat_c1",    "tr_c1",    "img_c1",
                "lat_c2",    "tr_c2",    "img_c2",
            ]}
            _acc["sigma"] = None

            for _ci, _chunk_seeds in enumerate(_chunks):
                _Nc = len(_chunk_seeds)
                print(f"    chunk {_ci + 1}/{len(_chunks)}: seeds {_chunk_seeds} (batch_size={_Nc})")
                _chunk_init, _chunk_sigma = make_batched_init_latents(
                    seeds=_chunk_seeds,
                    model_family=args.model_family,
                    scheduler=scheduler,
                    models=models,
                    device=device,
                    dtype=dtype,
                    num_inference_steps=args.steps,
                    latent_size=latent_size,
                )
                _acc["init"].append(_chunk_init)
                _acc["sigma"] = _chunk_sigma

                with torch.no_grad():
                    _cl_and, _ct_and = poe_sd_with_trajectory_tracking(
                        latents=_chunk_init.clone(),
                        prompt_a=c1,
                        prompt_b=c2,
                        scheduler=scheduler,
                        unet=models["unet"],
                        tokenizer=models["tokenizer"],
                        text_encoder=models["text_encoder"],
                        guidance_scale=args.guidance,
                        num_inference_steps=args.steps,
                        batch_size=_Nc,
                        device=device,
                        dtype=dtype,
                        model_id=args.model_id,
                        euler_init_noise_sigma=_chunk_sigma,
                        height=args.image_size,
                        width=args.image_size,
                    )
                    _acc["lat_and"].append(_cl_and)
                    _acc["tr_and"].append(_ct_and)
                    _ci_and = decode_latents(models["vae"], _cl_and)
                    _acc["img_and"].append(_ci_and)

                    if args.sdipc:
                        _proj = sdipc_project_image(
                            _ci_and,
                            clip_model=clip_eval,
                            clip_processor=sdipc_runtime["clip_processor"],
                            visual_proj=sdipc_runtime["visual_proj"],
                            inv_text=sdipc_runtime["inv_text"],
                            device=device,
                        )
                        _cond_sdipc = sdipc_to_sd14_cond(
                            _proj, runtime=sdipc_runtime, device=device, dtype=dtype
                        )
                        _cl_s, _ct_s = sample_sd1_with_precomputed_cond(
                            latents=_chunk_init.clone(),
                            cond_emb=_cond_sdipc,
                            scheduler=scheduler,
                            unet=models["unet"],
                            tokenizer=models["tokenizer"],
                            text_encoder=models["text_encoder"],
                            guidance_scale=args.guidance,
                            num_inference_steps=args.steps,
                            batch_size=_Nc,
                            device=device,
                            dtype=dtype,
                            model_id=args.model_id,
                            euler_init_noise_sigma=_chunk_sigma,
                        )
                        _acc["lat_sdipc"].append(_cl_s)
                        _acc["tr_sdipc"].append(_ct_s)
                        _acc["img_sdipc"].append(decode_latents(models["vae"], _cl_s))

                    _cl_mono, _ct_mono = sample_sd1_with_trajectory_tracking(
                        latents=_chunk_init.clone(),
                        prompt=mono_prompt_naive,
                        scheduler=scheduler,
                        unet=models["unet"],
                        tokenizer=models["tokenizer"],
                        text_encoder=models["text_encoder"],
                        guidance_scale=args.guidance,
                        num_inference_steps=args.steps,
                        batch_size=_Nc,
                        device=device,
                        dtype=dtype,
                        model_id=args.model_id,
                        euler_init_noise_sigma=_chunk_sigma,
                        height=args.image_size,
                        width=args.image_size,
                    )
                    _acc["lat_mono"].append(_cl_mono)
                    _acc["tr_mono"].append(_ct_mono)
                    _acc["img_mono"].append(decode_latents(models["vae"], _cl_mono))

                    _cl_c1, _ct_c1 = sample_sd1_with_trajectory_tracking(
                        latents=_chunk_init.clone(),
                        prompt=c1,
                        scheduler=scheduler,
                        unet=models["unet"],
                        tokenizer=models["tokenizer"],
                        text_encoder=models["text_encoder"],
                        guidance_scale=args.guidance,
                        num_inference_steps=args.steps,
                        batch_size=_Nc,
                        device=device,
                        dtype=dtype,
                        model_id=args.model_id,
                        euler_init_noise_sigma=_chunk_sigma,
                        height=args.image_size,
                        width=args.image_size,
                    )
                    _acc["lat_c1"].append(_cl_c1)
                    _acc["tr_c1"].append(_ct_c1)
                    _acc["img_c1"].append(decode_latents(models["vae"], _cl_c1))

                    _cl_c2, _ct_c2 = sample_sd1_with_trajectory_tracking(
                        latents=_chunk_init.clone(),
                        prompt=c2,
                        scheduler=scheduler,
                        unet=models["unet"],
                        tokenizer=models["tokenizer"],
                        text_encoder=models["text_encoder"],
                        guidance_scale=args.guidance,
                        num_inference_steps=args.steps,
                        batch_size=_Nc,
                        device=device,
                        dtype=dtype,
                        model_id=args.model_id,
                        euler_init_noise_sigma=_chunk_sigma,
                        height=args.image_size,
                        width=args.image_size,
                    )
                    _acc["lat_c2"].append(_cl_c2)
                    _acc["tr_c2"].append(_ct_c2)
                    _acc["img_c2"].append(decode_latents(models["vae"], _cl_c2))

            # Concatenate chunks into final _b dict (same shape as before: (N, ...))
            for _k in ["init", "lat_and", "img_and", "lat_mono", "img_mono",
                        "lat_c1", "img_c1", "lat_c2", "img_c2"]:
                _b[_k] = torch.cat(_acc[_k], dim=0)
            for _k in ["tr_and", "tr_mono", "tr_c1", "tr_c2"]:
                _b[_k] = _cat_trackers(_acc[_k])
            _b["sigma"] = _acc["sigma"]
            if args.sdipc:
                _b["lat_sdipc"] = torch.cat(_acc["lat_sdipc"], dim=0)
                _b["img_sdipc"] = torch.cat(_acc["img_sdipc"], dim=0)
                _b["tr_sdipc"]  = _cat_trackers(_acc["tr_sdipc"])

        for _seed_idx, seed in enumerate(args.seeds):
            if _batched:
                # -- Batch path: slice all pre-computed tensors for this seed --
                _i = _seed_idx
                init_latents        = _b["init"][_i : _i + 1]
                sd14_euler_sigma    = _b["sigma"]
                lat_and             = _b["lat_and"][_i : _i + 1]
                tracker_and         = _slice_tracker(_b["tr_and"], _i)
                img_and             = _b["img_and"][_i : _i + 1]
                source_img_for_inversion = img_and.clone()
                images_and.append(img_and.cpu())
                trackers_and.append(tracker_and)
                images_pstar.append(None)   # _skip_inverter always True in batch mode
                trackers_pstar.append(None)
                if args.sdipc:
                    images_pstar_sdipc.append(_b["img_sdipc"][_i : _i + 1].cpu())
                    trackers_pstar_sdipc.append(_slice_tracker(_b["tr_sdipc"], _i))
                lat_mono_naive       = _b["lat_mono"][_i : _i + 1]
                img_mono_naive       = _b["img_mono"][_i : _i + 1]
                tracker_mono_naive   = _slice_tracker(_b["tr_mono"], _i)
                lat_mono_natural     = lat_mono_naive
                img_mono_natural     = img_mono_naive
                tracker_mono_natural = tracker_mono_naive
                img_mono    = img_mono_natural if args.monolithic_baseline == "natural" else img_mono_naive
                tracker_mono = tracker_mono_natural if args.monolithic_baseline == "natural" else tracker_mono_naive
                images_mono.append(img_mono.cpu())
                trackers_mono.append(tracker_mono)
                lat_c1      = _b["lat_c1"][_i : _i + 1]
                img_c1      = _b["img_c1"][_i : _i + 1]
                tracker_c1  = _slice_tracker(_b["tr_c1"], _i)
                lat_c2      = _b["lat_c2"][_i : _i + 1]
                img_c2      = _b["img_c2"][_i : _i + 1]
                tracker_c2  = _slice_tracker(_b["tr_c2"], _i)
                images_c1.append(img_c1.cpu())
                images_c2.append(img_c2.cpu())
                trackers_c1.append(tracker_c1)
                trackers_c2.append(tracker_c2)
            else:
                init_latents, sd14_euler_sigma = make_shared_init_latents(
                    seed=seed,
                    model_family=args.model_family,
                    scheduler=scheduler,
                    models=models,
                    device=device,
                    dtype=dtype,
                    num_inference_steps=args.steps,
                    latent_size=latent_size,
                )

                # ---- Step 1: Active logical anchor (PoE) ----
                with torch.no_grad():
                    if args.model_family == "sd14":
                        lat_and, tracker_and = poe_sd_with_trajectory_tracking(
                            latents=init_latents.clone(),
                            prompt_a=c1,
                            prompt_b=c2,
                            scheduler=scheduler,
                            unet=models["unet"],
                            tokenizer=models["tokenizer"],
                            text_encoder=models["text_encoder"],
                            guidance_scale=args.guidance,
                            num_inference_steps=args.steps,
                            batch_size=1,
                            device=device,
                            dtype=dtype,
                            model_id=args.model_id,
                            euler_init_noise_sigma=sd14_euler_sigma,
                            height=args.image_size,
                            width=args.image_size,
                        )
                    else:
                        lat_and, tracker_and = poe_sd3_with_trajectory_tracking(
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
                    img_and = decode_latents(models["vae"], lat_and)  # (1, 3, H, W)
                    # All active p* reruns are anchored on the PoE output image.
                    source_img_for_inversion = img_and.clone()

                images_and.append(img_and.cpu())
                trackers_and.append(tracker_and)

                # ---- Step 2: Invert active logical-anchor image (trained CLIP inverter) ----
                # Skipped when --pstar-source {sdipc|pez|z2t} to save time on subsequent runs.
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
                        if args.model_family == "sd14":
                            lat_pez, tracker_pez = sample_sd1_with_trajectory_tracking(
                                latents=init_latents.clone(),
                                prompt=pez_prompt,
                                scheduler=scheduler,
                                unet=models["unet"],
                                tokenizer=models["tokenizer"],
                                text_encoder=models["text_encoder"],
                                guidance_scale=args.guidance,
                                num_inference_steps=args.steps,
                                batch_size=1,
                                device=device,
                                dtype=dtype,
                                model_id=args.model_id,
                                euler_init_noise_sigma=sd14_euler_sigma,
                                height=args.image_size,
                                width=args.image_size,
                            )
                        else:
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
                        if args.model_family == "sd14":
                            lat_z2t, tracker_z2t = sample_sd1_with_trajectory_tracking(
                                latents=init_latents.clone(),
                                prompt=z2t_prompt,
                                scheduler=scheduler,
                                unet=models["unet"],
                                tokenizer=models["tokenizer"],
                                text_encoder=models["text_encoder"],
                                guidance_scale=args.guidance,
                                num_inference_steps=args.steps,
                                batch_size=1,
                                device=device,
                                dtype=dtype,
                                model_id=args.model_id,
                                euler_init_noise_sigma=sd14_euler_sigma,
                                height=args.image_size,
                                width=args.image_size,
                            )
                        else:
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

                # ---- Step 3d: SD-IPC closed-form projection from the PoE anchor image ----
                if args.sdipc:
                    with torch.no_grad():
                        lat_sdipc, tracker_sdipc = run_sdipc_rerun(
                            source_img=source_img_for_inversion,
                            init_latents=init_latents.clone(),
                            model_family=args.model_family,
                            models=models,
                            scheduler=scheduler,
                            sdipc_runtime=sdipc_runtime,
                            clip_eval=clip_eval,
                            device=device,
                            dtype=dtype,
                            guidance=args.guidance,
                            steps=args.steps,
                            model_id=args.model_id,
                            sd14_euler_sigma=sd14_euler_sigma,
                            uncond_embeds=uncond_embeds,
                            uncond_pooled=uncond_pooled,
                        )
                        img_sdipc = decode_latents(models["vae"], lat_sdipc)
                    images_pstar_sdipc.append(img_sdipc.cpu())
                    trackers_pstar_sdipc.append(tracker_sdipc)

                # ---- Step 3e: CO3 in-process (shared x_T, endpoint-only) ----
                if co3_model is not None and images_co3 is not None:
                    # Pass unit-normal x_T to CO3 (undo Euler sigma scaling).
                    # CO3 uses DDIMScheduler with init_noise_sigma=1.0, so unit-normal
                    # is the correct scale. All other DDIM-based methods also divide by
                    # euler_sigma internally, ensuring a shared underlying x_T.
                    co3_init = (init_latents / sd14_euler_sigma).to(
                        device=co3_model.unet.device, dtype=co3_model.weight_dtype
                    )
                    pil_co3_list = co3_model.run_sampling(latents=co3_init)
                    img_co3 = transforms.ToTensor()(pil_co3_list[0]).unsqueeze(0).float()
                    images_co3.append(img_co3.cpu())
                    print(f"    CO3 (in-process, seed {seed}): done")

                    # ---- Step 3f: SD-IPC rerun from CO3 image (per-seed p* for CO3) ----
                    if args.sdipc:
                        with torch.no_grad():
                            lat_co3_sdipc_seed, tracker_co3_sdipc_seed = run_sdipc_rerun(
                                source_img=img_co3.to(device=device),
                                init_latents=init_latents.clone(),
                                model_family=args.model_family,
                                models=models,
                                scheduler=scheduler,
                                sdipc_runtime=sdipc_runtime,
                                clip_eval=clip_eval,
                                device=device,
                                dtype=dtype,
                                guidance=args.guidance,
                                steps=args.steps,
                                model_id=args.model_id,
                                sd14_euler_sigma=sd14_euler_sigma,
                                uncond_embeds=uncond_embeds,
                                uncond_pooled=uncond_pooled,
                            )
                            img_co3_sdipc_seed = decode_latents(models["vae"], lat_co3_sdipc_seed)
                        images_pstar_co3_sdipc.append(img_co3_sdipc_seed.cpu())
                        trackers_pstar_co3_sdipc.append(tracker_co3_sdipc_seed)
                        print(f"    CO3 p* SD-IPC (in-process, seed {seed}): done")

                # ---- Step 4: Monolithic baselines (naive and naturalized) ----
                # In paper-only mode, only compute the active baseline; skip the variant.
                with torch.no_grad():
                    if args.model_family == "sd14":
                        lat_mono_naive, tracker_mono_naive = sample_sd1_with_trajectory_tracking(
                            latents=init_latents.clone(),
                            prompt=mono_prompt_naive,
                            scheduler=scheduler,
                            unet=models["unet"],
                            tokenizer=models["tokenizer"],
                            text_encoder=models["text_encoder"],
                            guidance_scale=args.guidance,
                            num_inference_steps=args.steps,
                            batch_size=1,
                            device=device,
                            dtype=dtype,
                            model_id=args.model_id,
                            euler_init_noise_sigma=sd14_euler_sigma,
                            height=args.image_size,
                            width=args.image_size,
                        )
                    else:
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

                    # Only compute natural variant if needed (not paper-only, or if it's the active baseline)
                    lat_mono_natural = None
                    tracker_mono_natural = None
                    img_mono_natural = None
                    if not args._paper_only or args.monolithic_baseline == "natural":
                        if mono_prompt_natural == mono_prompt_naive:
                            lat_mono_natural = lat_mono_naive
                            tracker_mono_natural = tracker_mono_naive
                            img_mono_natural = img_mono_naive
                        else:
                            if args.model_family == "sd14":
                                lat_mono_natural, tracker_mono_natural = sample_sd1_with_trajectory_tracking(
                                    latents=init_latents.clone(),
                                    prompt=mono_prompt_natural,
                                    scheduler=scheduler,
                                    unet=models["unet"],
                                    tokenizer=models["tokenizer"],
                                    text_encoder=models["text_encoder"],
                                    guidance_scale=args.guidance,
                                    num_inference_steps=args.steps,
                                    batch_size=1,
                                    device=device,
                                    dtype=dtype,
                                    model_id=args.model_id,
                                    euler_init_noise_sigma=sd14_euler_sigma,
                                    height=args.image_size,
                                    width=args.image_size,
                                )
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
                if images_mono_naive is not None:
                    images_mono_naive.append(img_mono_naive.cpu())
                if images_mono_natural is not None:
                    images_mono_natural.append(img_mono_natural.cpu())
                trackers_mono.append(tracker_mono)
                if trackers_mono_naive is not None:
                    trackers_mono_naive.append(tracker_mono_naive)
                if trackers_mono_natural is not None:
                    trackers_mono_natural.append(tracker_mono_natural)

                # ---- c1-only and c2-only baselines (same initial noise) ----
                with torch.no_grad():
                    if args.model_family == "sd14":
                        lat_c1, tracker_c1 = sample_sd1_with_trajectory_tracking(
                            latents=init_latents.clone(),
                            prompt=c1,
                            scheduler=scheduler,
                            unet=models["unet"],
                            tokenizer=models["tokenizer"],
                            text_encoder=models["text_encoder"],
                            guidance_scale=args.guidance,
                            num_inference_steps=args.steps,
                            batch_size=1,
                            device=device,
                            dtype=dtype,
                            model_id=args.model_id,
                            euler_init_noise_sigma=sd14_euler_sigma,
                            height=args.image_size,
                            width=args.image_size,
                        )
                    else:
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

                    if args.model_family == "sd14":
                        lat_c2, tracker_c2 = sample_sd1_with_trajectory_tracking(
                            latents=init_latents.clone(),
                            prompt=c2,
                            scheduler=scheduler,
                            unet=models["unet"],
                            tokenizer=models["tokenizer"],
                            text_encoder=models["text_encoder"],
                            guidance_scale=args.guidance,
                            num_inference_steps=args.steps,
                            batch_size=1,
                            device=device,
                            dtype=dtype,
                            model_id=args.model_id,
                            euler_init_noise_sigma=sd14_euler_sigma,
                            height=args.image_size,
                            width=args.image_size,
                        )
                    else:
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

            # ---- Per-seed terminal latent distances (anchor = active logical target) ----
            # Use raw tracker latents — no VAE re-encode error.
            # Per-element MSE is dimension-normalised and scale-comparable.
            z_and  = tracker_and.trajectories[-1].float()   # (1, 16, H, W)
            z_mono = tracker_mono.trajectories[-1].float()
            z_mono_naive = tracker_mono_naive.trajectories[-1].float() if trackers_mono_naive is not None else None
            z_mono_natural = tracker_mono_natural.trajectories[-1].float() if trackers_mono_natural is not None else None
            z_c1_t = tracker_c1.trajectories[-1].float()
            z_c2_t = tracker_c2.trajectories[-1].float()
            rec = {
                "pair":     f"{c1} + {c2}",
                "c1":       c1,
                "c2":       c2,
                "model_family": args.model_family,
                "model_id": args.model_id,
                "pair_group": pair_group,
                "pair_slug": pair_slug,
                "taxonomy_group_key": taxonomy_group_key,
                "taxonomy_group_label": taxonomy_group_label,
                "is_representative_pair": is_representative_pair,
                "logical_anchor": ACTIVE_LOGICAL_ANCHOR_KEY,
                "logical_anchor_label": ACTIVE_LOGICAL_ANCHOR_LABEL,
                "seed":     seed,
                "monolithic_baseline": args.monolithic_baseline,
                "mono_prompt_active": mono_prompt_active,
                "mono_prompt_naive": mono_prompt_naive,
                "mono_prompt_natural": mono_prompt_natural,
                "d_T_poe":  0.0,
                "d_T_mono": float(((z_mono  - z_and) ** 2).mean()),  # backward compat: active
                "d_T_mono_to_poe": float(((z_mono  - z_and) ** 2).mean()),
                "d_T_c1":   float(((z_c1_t  - z_and) ** 2).mean()),
                "d_T_c2":   float(((z_c2_t  - z_and) ** 2).mean()),
            }
            # Add naive/natural mono distances only if they were computed
            if z_mono_naive is not None:
                rec["d_T_mono_naive"] = float(((z_mono_naive  - z_and) ** 2).mean())
            if z_mono_natural is not None:
                rec["d_T_mono_natural"] = float(((z_mono_natural - z_and) ** 2).mean())
            if not args._skip_inverter:
                z_pstar_t = tracker_pstar.trajectories[-1].float()
                d_inv = float(((z_pstar_t - z_and) ** 2).mean())
                rec["d_T_pstar"]     = d_inv   # backward compat
                rec["d_T_pstar_inv"] = d_inv   # explicit symmetric name
                rec["d_T_pstar_inv_to_mono"] = float(((z_pstar_t - z_mono) ** 2).mean())
            if args.pez:
                z_pez = trackers_pstar_pez[-1].trajectories[-1].float()
                rec["d_T_pstar_pez"] = float(((z_pez - z_and) ** 2).mean())
                rec["d_T_pstar_pez_to_mono"] = float(((z_pez - z_mono) ** 2).mean())
            if args.z2t:
                z_z2t = trackers_pstar_z2t[-1].trajectories[-1].float()
                rec["d_T_pstar_z2t"] = float(((z_z2t - z_and) ** 2).mean())
                rec["d_T_pstar_z2t_to_mono"] = float(((z_z2t - z_mono) ** 2).mean())
            if args.sdipc:
                z_sdipc = trackers_pstar_sdipc[-1].trajectories[-1].float()
                rec["d_T_pstar_sdipc"] = float(((z_sdipc - z_and) ** 2).mean())
                rec["d_T_pstar_sdipc_to_mono"] = float(((z_sdipc - z_mono) ** 2).mean())
            if images_co3 is not None:
                # CO3 is endpoint-only: VAE-encode the decoded image back to latent
                # space so d_T_co3 is measured in the same (unscaled latent) metric
                # as d_T_mono, d_T_c1, etc.  The VAE round-trip introduces the same
                # encode/decode error as for every other condition that uses
                # decode_latents(), so the comparison remains valid.
                with torch.no_grad():
                    img_co3_t = img_co3.to(device=device, dtype=dtype)
                    if img_co3_t.ndim == 3:
                        img_co3_t = img_co3_t.unsqueeze(0)
                    z_co3 = (
                        models["vae"].encode(img_co3_t * 2.0 - 1.0).latent_dist.mean
                        * models["vae"].config.scaling_factor
                    ).float()
                rec["d_T_co3"] = float(((z_co3 - z_and) ** 2).mean())
            if trackers_pstar_co3_sdipc is not None:
                z_co3_sdipc = trackers_pstar_co3_sdipc[-1].trajectories[-1].float()
                rec["d_T_pstar_co3_sdipc"] = float(((z_co3_sdipc - z_and) ** 2).mean())
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
                [t.trajectories[-1].float() for t in trackers_mono_naive], dim=0
            ) if trackers_mono_naive is not None else None
            mono_natural_stack  = torch.stack(
                [t.trajectories[-1].float() for t in trackers_mono_natural], dim=0
            ) if trackers_mono_natural is not None else None
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
            sdipc_stack = torch.stack(
                [t.trajectories[-1].float() for t in trackers_pstar_sdipc], dim=0
            ) if args.sdipc else None

            for si, seed in enumerate(args.seeds):
                # Find the per_seed_record we just appended for this seed
                rec = next(r for r in per_seed_records
                           if r["pair"] == f"{c1} + {c2}" and r["seed"] == seed)
                rec["d_T_mono_meananchor"] = float(  # backward compat: active
                    ((mono_stack[si] - z_and_avg) ** 2).mean())
                if mono_naive_stack is not None:
                    rec["d_T_mono_naive_meananchor"] = float(
                        ((mono_naive_stack[si] - z_and_avg) ** 2).mean())
                if mono_natural_stack is not None:
                    rec["d_T_mono_natural_meananchor"] = float(
                        ((mono_natural_stack[si] - z_and_avg) ** 2).mean())
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
                if sdipc_stack is not None:
                    rec["d_T_pstar_sdipc_meananchor"] = float(
                        ((sdipc_stack[si] - z_and_avg) ** 2).mean())

        # ---- Within-anchor cross-seed pairwise distances (noise floor) ----
        # For each unordered pair of seeds (s_i, s_j), compute
        #   d_within_and = mean( (z_AND[s_i] - z_AND[s_j])^2 )
        # This is the stochastic baseline: how much do active-anchor outputs vary
        # across seeds starting from different x_T?  If the per-condition
        # gaps (d_T_mono, d_T_c1, …) are well above this, the pairing used
        # in the main metric is not masking a noise-floor effect.
        print(f"  Computing within-{ACTIVE_LOGICAL_ANCHOR_LABEL} pairwise distances (noise floor) ...")
        for i, seed_i in enumerate(args.seeds):
            for j, seed_j in enumerate(args.seeds):
                if j <= i:
                    continue
                z_i = trackers_and[i].trajectories[-1].float()
                z_j = trackers_and[j].trajectories[-1].float()
                all_within_and_records.append({
                    "pair":         f"{c1} + {c2}",
                    "model_family": args.model_family,
                    "model_id": args.model_id,
                    "pair_group":   pair_group,
                    "pair_slug":    pair_slug,
                    "taxonomy_group_key": taxonomy_group_key,
                    "taxonomy_group_label": taxonomy_group_label,
                    "is_representative_pair": is_representative_pair,
                    "logical_anchor": ACTIVE_LOGICAL_ANCHOR_KEY,
                    "logical_anchor_label": ACTIVE_LOGICAL_ANCHOR_LABEL,
                    "seed_a":       seed_i,
                    "seed_b":       seed_j,
                    "d_within_and": float(((z_i - z_j) ** 2).mean()),
                    "d_within_poe": float(((z_i - z_j) ** 2).mean()),
                })

        # ---- Per-step trajectory distances (CPU, no GPU needed) ----
        # trackers already hold trajectories as CPU tensors from store_step().
        # Shape: (num_steps+1, 1, C, H, W) — iterate over all T+1 states.
        print("  Computing per-step trajectory distances ...")
        for seed_idx, seed in enumerate(args.seeds):
            ta  = trackers_and[seed_idx]
            tm  = trackers_mono[seed_idx]
            tm_naive = trackers_mono_naive[seed_idx] if trackers_mono_naive is not None else None
            tm_natural = trackers_mono_natural[seed_idx] if trackers_mono_natural is not None else None
            tc1 = trackers_c1[seed_idx]
            tc2 = trackers_c2[seed_idx]
            tp     = trackers_pstar[seed_idx]      # None when _skip_inverter
            tp_pez = trackers_pstar_pez[seed_idx] if args.pez else None
            tp_z2t = trackers_pstar_z2t[seed_idx] if args.z2t else None
            tp_sdipc = trackers_pstar_sdipc[seed_idx] if args.sdipc else None
            tp_co3_sdipc = (
                trackers_pstar_co3_sdipc[seed_idx]
                if trackers_pstar_co3_sdipc is not None
                else None
            )
            n_steps = ta.trajectories.shape[0]   # T+1
            # Validate that all active trackers have the same step count so that
            # per-step indexing is correctly aligned across conditions.
            _active_trackers = {
                "and": ta, "mono": tm, "c1": tc1, "c2": tc2,
                **({} if tp is None else {"pstar_inv": tp}),
                **({} if tp_pez is None else {"pstar_pez": tp_pez}),
                **({} if tp_z2t is None else {"pstar_z2t": tp_z2t}),
                **({} if tp_sdipc is None else {"pstar_sdipc": tp_sdipc}),
                **({} if tp_co3_sdipc is None else {"pstar_co3_sdipc": tp_co3_sdipc}),
            }
            _step_counts = {name: t.trajectories.shape[0] for name, t in _active_trackers.items()}
            if len(set(_step_counts.values())) != 1:
                raise AssertionError(
                    f"Trackers have different step counts for seed {seed}, pair {c1}+{c2}: "
                    f"{_step_counts}. All conditions must use the same scheduler."
                )
            for step in range(n_steps):
                z_and  = ta.trajectories[step].float()
                z_mono = tm.trajectories[step].float()
                z_mono_naive = tm_naive.trajectories[step].float() if tm_naive is not None else None
                z_mono_natural = tm_natural.trajectories[step].float() if tm_natural is not None else None
                z_c1_t = tc1.trajectories[step].float()
                z_c2_t = tc2.trajectories[step].float()
                sigma  = float(ta.sigmas[min(step, len(ta.sigmas) - 1)])
                trec = {
                    "pair":     f"{c1} + {c2}",
                    "c1":       c1,
                    "c2":       c2,
                    "model_family": args.model_family,
                    "model_id": args.model_id,
                    "pair_group": pair_group,
                    "pair_slug": pair_slug,
                    "taxonomy_group_key": taxonomy_group_key,
                    "taxonomy_group_label": taxonomy_group_label,
                    "is_representative_pair": is_representative_pair,
                    "logical_anchor": ACTIVE_LOGICAL_ANCHOR_KEY,
                    "logical_anchor_label": ACTIVE_LOGICAL_ANCHOR_LABEL,
                    "seed":     seed,
                    "step":     step,
                    "sigma":    sigma,
                    "monolithic_baseline": args.monolithic_baseline,
                    "d_t_poe": 0.0,
                    "d_t_mono": float(((z_mono  - z_and) ** 2).mean()),  # backward compat: active
                    "d_t_mono_to_poe": float(((z_mono  - z_and) ** 2).mean()),
                    "d_t_c1":   float(((z_c1_t  - z_and) ** 2).mean()),
                    "d_t_c2":   float(((z_c2_t  - z_and) ** 2).mean()),
                }
                # Add naive/natural mono distances only if they were computed
                if z_mono_naive is not None:
                    trec["d_t_mono_naive"] = float(((z_mono_naive  - z_and) ** 2).mean())
                if z_mono_natural is not None:
                    trec["d_t_mono_natural"] = float(((z_mono_natural - z_and) ** 2).mean())
                if tp is not None:
                    z_pstar_s = tp.trajectories[step].float()
                    d_t_inv = float(((z_pstar_s - z_and) ** 2).mean())
                    trec["d_t_pstar"]     = d_t_inv   # backward compat
                    trec["d_t_pstar_inv"] = d_t_inv   # explicit symmetric name
                    trec["d_t_pstar_inv_to_mono"] = float(((z_pstar_s - z_mono) ** 2).mean())
                if args.pez:
                    z_pez_s = tp_pez.trajectories[step].float()
                    trec["d_t_pstar_pez"] = float(((z_pez_s - z_and) ** 2).mean())
                    trec["d_t_pstar_pez_to_mono"] = float(((z_pez_s - z_mono) ** 2).mean())
                if args.z2t:
                    z_z2t_s = tp_z2t.trajectories[step].float()
                    trec["d_t_pstar_z2t"] = float(((z_z2t_s - z_and) ** 2).mean())
                    trec["d_t_pstar_z2t_to_mono"] = float(((z_z2t_s - z_mono) ** 2).mean())
                if args.sdipc:
                    z_sdipc_s = tp_sdipc.trajectories[step].float()
                    trec["d_t_pstar_sdipc"] = float(((z_sdipc_s - z_and) ** 2).mean())
                    trec["d_t_pstar_sdipc_to_mono"] = float(((z_sdipc_s - z_mono) ** 2).mean())
                if tp_co3_sdipc is not None:
                    z_co3_sdipc_s = tp_co3_sdipc.trajectories[step].float()
                    trec["d_t_pstar_co3_sdipc"] = float(((z_co3_sdipc_s - z_and) ** 2).mean())
                all_traj_records.append(trec)

        # Stack across seeds
        imgs_and   = torch.cat(images_and,  dim=0)  # (N, 3, H, W)
        imgs_pstar = torch.cat([x for x in images_pstar if x is not None], dim=0) \
                     if not args._skip_inverter else None
        imgs_mono  = torch.cat(images_mono, dim=0)  # active baseline
        imgs_mono_naive = torch.cat(images_mono_naive, dim=0) if images_mono_naive is not None else None
        imgs_mono_natural = torch.cat(images_mono_natural, dim=0) if images_mono_natural is not None else None
        imgs_c1    = torch.cat(images_c1,   dim=0)
        imgs_c2    = torch.cat(images_c2,   dim=0)

        # ---- Save image grids + manifests ----
        # Skip legacy superdiff_and.png in paper-only mode
        if not args._paper_only:
            save_image(imgs_and,  img_dir / "superdiff_and.png",   nrow=4, normalize=False)
            save_single_grid_manifest(img_dir / "superdiff_and.png",
                f'{ACTIVE_LOGICAL_ANCHOR_LABEL} logical anchor (legacy filename) — "{c1}" × "{c2}"', args.seeds)

        poe_grid = img_dir / f"{family_prefix}_poe.png"
        save_image(imgs_and, poe_grid, nrow=4, normalize=False)
        save_single_grid_manifest(
            poe_grid,
            f'{family_label} {ACTIVE_LOGICAL_ANCHOR_LABEL} — "{c1}" × "{c2}"',
            args.seeds,
        )
        if imgs_pstar is not None:
            pstar_grid = img_dir / f"{family_prefix}_pstar.png"
            save_image(imgs_pstar, pstar_grid, nrow=4, normalize=False)
            save_single_grid_manifest(
                pstar_grid,
                f'{family_label} (p* — CLIP inverter) — "{c1}" ∧ "{c2}"',
                args.seeds,
            )
        mono_grid = img_dir / f"{family_prefix}_monolithic.png"
        save_image(imgs_mono, mono_grid, nrow=4, normalize=False)
        save_single_grid_manifest(
            mono_grid,
            f'{family_label} monolithic ({args.monolithic_baseline}, active) — "{mono_prompt_active}"',
            args.seeds,
        )
        # Skip naive/natural variant grids in paper-only mode
        if not args._paper_only:
            if imgs_mono_naive is not None:
                mono_naive_grid = img_dir / f"{family_prefix}_monolithic_naive.png"
                save_image(imgs_mono_naive, mono_naive_grid, nrow=4, normalize=False)
                save_single_grid_manifest(
                    mono_naive_grid,
                    f'{family_label} monolithic (naive) — "{mono_prompt_naive}"',
                    args.seeds,
                )
            if imgs_mono_natural is not None:
                mono_natural_grid = img_dir / f"{family_prefix}_monolithic_natural.png"
                save_image(imgs_mono_natural, mono_natural_grid, nrow=4, normalize=False)
                save_single_grid_manifest(
                    mono_natural_grid,
                    f'{family_label} monolithic (natural) — "{mono_prompt_natural}"',
                    args.seeds,
                )
        c1_grid = img_dir / f"{family_prefix}_c1_only.png"
        save_image(imgs_c1, c1_grid, nrow=4, normalize=False)
        save_single_grid_manifest(c1_grid, f'{family_label} solo — "{c1}"', args.seeds)
        c2_grid = img_dir / f"{family_prefix}_c2_only.png"
        save_image(imgs_c2, c2_grid, nrow=4, normalize=False)
        save_single_grid_manifest(c2_grid, f'{family_label} solo — "{c2}"', args.seeds)

        if args.pez:
            imgs_pstar_pez_all = torch.cat(images_pstar_pez, dim=0)
            pez_grid = img_dir / f"{family_prefix}_pstar_pez.png"
            save_image(imgs_pstar_pez_all, pez_grid, nrow=4, normalize=False)
            save_single_grid_manifest(pez_grid,
                f'{family_label} (p* — PEZ) — "{c1}" ∧ "{c2}"', args.seeds,
                per_seed_prompts=pez_prompts)
        if args.z2t:
            imgs_pstar_z2t_all = torch.cat(images_pstar_z2t, dim=0)
            z2t_grid = img_dir / f"{family_prefix}_pstar_z2t.png"
            save_image(imgs_pstar_z2t_all, z2t_grid, nrow=4, normalize=False)
            save_single_grid_manifest(z2t_grid,
                f'{family_label} (p* — Z2T/{args.z2t_pool_mode}) — "{c1}" ∧ "{c2}"', args.seeds,
                per_seed_prompts=z2t_prompts)
        if args.sdipc:
            imgs_pstar_sdipc_all = torch.cat(images_pstar_sdipc, dim=0)
            sdipc_grid = img_dir / f"{family_prefix}_pstar_sdipc.png"
            save_image(imgs_pstar_sdipc_all, sdipc_grid, nrow=4, normalize=False)
            save_single_grid_manifest(sdipc_grid,
                f'{family_label} (PoE p* — SD-IPC) — "{c1}" × "{c2}"', args.seeds)

        # ---- Export single-seed assets for cross-pair grid plots (27–28) ----
        # Use the configured grid seed so all conditions remain seed-matched.
        grid_seed_idx = args._grid_seed_idx
        grid_seed = args._grid_seed

        decoded_for_grid = {
            "prompt_a":         images_c1[grid_seed_idx],
            "prompt_b":         images_c2[grid_seed_idx],
            "monolithic":       images_mono[grid_seed_idx],  # active
            "poe":              images_and[grid_seed_idx],
        }
        trackers_for_grid = {
            "prompt_a":         trackers_c1[grid_seed_idx],
            "prompt_b":         trackers_c2[grid_seed_idx],
            "monolithic":       trackers_mono[grid_seed_idx],  # active
            "poe":              trackers_and[grid_seed_idx],
        }
        # Add naive/natural variants only if they were computed (not paper-only mode)
        if images_mono_naive is not None:
            decoded_for_grid["monolithic_naive"] = images_mono_naive[grid_seed_idx]
            trackers_for_grid["monolithic_naive"] = trackers_mono_naive[grid_seed_idx]
        if images_mono_natural is not None:
            decoded_for_grid["monolithic_natural"] = images_mono_natural[grid_seed_idx]
            trackers_for_grid["monolithic_natural"] = trackers_mono_natural[grid_seed_idx]
        if not args._skip_inverter and images_pstar is not None and trackers_pstar is not None:
            decoded_for_grid["pstar_inv"] = images_pstar[grid_seed_idx]
            trackers_for_grid["pstar_inv"] = trackers_pstar[grid_seed_idx]
        if args.sdipc and images_pstar_sdipc is not None and trackers_pstar_sdipc is not None:
            decoded_for_grid["pstar_sdipc"] = images_pstar_sdipc[grid_seed_idx]
            trackers_for_grid["pstar_sdipc"] = trackers_pstar_sdipc[grid_seed_idx]
        if args.z2t and images_pstar_z2t is not None and trackers_pstar_z2t is not None:
            decoded_for_grid["pstar_z2t"] = images_pstar_z2t[grid_seed_idx]
            trackers_for_grid["pstar_z2t"] = trackers_pstar_z2t[grid_seed_idx]

        source_prompts = {}
        source_prompts["logical_anchor"] = ACTIVE_LOGICAL_ANCHOR_KEY
        source_prompts["monolithic_baseline"] = args.monolithic_baseline
        source_prompts["poe"] = "Active logical anchor (PoE)"
        source_prompts["monolithic"] = mono_prompt_active
        if not args._paper_only:
            source_prompts["monolithic_naive"] = mono_prompt_naive
            source_prompts["monolithic_natural"] = mono_prompt_natural
        if not args._skip_inverter:
            source_prompts["pstar_inv"] = "CLIP-inverter rerun"
        if args.sdipc:
            source_prompts["pstar_sdipc"] = "SD-IPC closed-form rerun from PoE"
        if args.z2t and z2t_prompts and grid_seed_idx < len(z2t_prompts):
            source_prompts["pstar_z2t"] = z2t_prompts[grid_seed_idx]

        external_grid_images = {}

        # ---- CO3 grid image: in-process (preferred) or disk fallback ----
        # Skip CO3 in paper-only mode
        _co3_source_for_sdipc = None
        _run_co3_sdipc = False
        if not args._paper_only and images_co3 is not None and images_co3:
            # In-process CO3 ran with shared x_T — use directly.
            decoded_for_grid["co3"] = images_co3[grid_seed_idx]
            source_prompts["co3"] = f"CO3 in-process (shared x_T, seed {grid_seed})"
            _co3_source_for_sdipc = decoded_for_grid["co3"].to(device)
            _run_co3_sdipc = args.sdipc
        elif not args._paper_only:
            # Fallback: try to load a pre-generated CO3 image from disk.
            co3_grid_image = resolve_co3_grid_image(
                c1,
                c2,
                taxonomy_group_key=taxonomy_group_key,
                pair_slug=pair_slug,
                co3_filename=args.co3_filename,
                co3_base=args.co3_base,
            )
            if co3_grid_image is not None:
                external_grid_images["co3"] = co3_grid_image
                source_prompts["co3"] = f"External CO3 image ({co3_grid_image.name})"
                if args.sdipc:
                    _co3_source_for_sdipc = load_external_rgb_tensor(
                        co3_grid_image, image_size=args.image_size,
                    ).to(device=device)
                    _run_co3_sdipc = True

        if _run_co3_sdipc and _co3_source_for_sdipc is not None:
            grid_init_latents, grid_sd14_euler_sigma = make_shared_init_latents(
                seed=grid_seed,
                model_family=args.model_family,
                scheduler=scheduler,
                models=models,
                device=device,
                dtype=dtype,
                num_inference_steps=args.steps,
                latent_size=latent_size,
            )
            with torch.no_grad():
                lat_co3_sdipc, _tracker_co3_sdipc = run_sdipc_rerun(
                    source_img=_co3_source_for_sdipc,
                    init_latents=grid_init_latents.clone(),
                    model_family=args.model_family,
                    models=models,
                    scheduler=scheduler,
                    sdipc_runtime=sdipc_runtime,
                    clip_eval=clip_eval,
                    device=device,
                    dtype=dtype,
                    guidance=args.guidance,
                    steps=args.steps,
                    model_id=args.model_id,
                    sd14_euler_sigma=grid_sd14_euler_sigma,
                    uncond_embeds=uncond_embeds,
                    uncond_pooled=uncond_pooled,
                )
                img_co3_sdipc = decode_latents(models["vae"], lat_co3_sdipc)
            decoded_for_grid["pstar_co3_sdipc"] = img_co3_sdipc.cpu()
            # Assign the SD-IPC rerun trajectory to the "co3" key so that CO3
            # participates in trajectory plots with a full denoising path starting
            # from the same x_T as all other conditions (not endpoint-only).
            trackers_for_grid["co3"] = _tracker_co3_sdipc
            source_prompts["pstar_co3_sdipc"] = "SD-IPC closed-form rerun from CO3 image"

        export_pair_grid_assets(
            pair_dir=pair_dir,
            c1=c1,
            c2=c2,
            seed=grid_seed,
            decoded_images=decoded_for_grid,
            trackers=trackers_for_grid,
            external_image_paths=external_grid_images,
            pair_index=pair_idx,
            projection_method=args.projection,
            source_prompts=source_prompts,
            taxonomy_group_key=taxonomy_group_key,
            taxonomy_group_label=taxonomy_group_label,
            pair_slug=pair_slug,
            is_representative_pair=is_representative_pair,
            model_family=args.model_family,
        )

        # ---- Decoded conditions grid (all available conditions, one figure) ----
        # Skip in paper-only mode
        if not args._paper_only:
            plot_decoded_conditions_grid(
                pair_dir=pair_dir,
                decoded_images=decoded_for_grid,
                external_image_paths={k: str(v) for k, v in external_grid_images.items()},
                c1=c1,
                c2=c2,
                seed=grid_seed,
                model_family=args.model_family,
            )

        # ---- Unified trajectory figure (all conditions with CO3 endpoint-only note) ----
        # CO3 itself has CO3_TRAJECTORY_SUPPORT = "endpoint_only" — it is shown in the
        # decoded conditions grid above but explicitly excluded from the trajectory plot.
        # pstar_co3_sdipc has a full trajectory (SD-IPC rerun from CO3) and IS included.
        co3_present = "co3" in external_grid_images or "co3" in decoded_for_grid
        if trackers_for_grid:
            unified_traj_path = (
                pair_dir / "trajectories"
                / f"trajectory_{args.projection}_unified_all_conditions.png"
            )
            plot_unified_trajectory_all_conditions(
                trackers=trackers_for_grid,
                c1=c1,
                c2=c2,
                out_path=unified_traj_path,
                method=args.projection,
                # CO3 is endpoint-only when it was loaded externally but has no
                # denoising trajectory. When args.sdipc ran pstar_co3_sdipc, the
                # tracker was stored under "co3" above, so it IS in trackers_for_grid
                # and co3_is_endpoint_only becomes False.
                co3_is_endpoint_only=co3_present and "co3" not in trackers_for_grid,
                model_family=args.model_family,
                logical_anchor_label=ACTIVE_LOGICAL_ANCHOR_LABEL,
            )

        # ---- Step 5: Gap metrics ----
        print(
            f"  Computing image gap: {ACTIVE_LOGICAL_ANCHOR_LABEL} vs monolithic "
            f"(active/{args.monolithic_baseline}) ..."
        )
        gap_and_mono = compute_image_gap(imgs_and, imgs_mono, clip_eval, lpips_fn, device)

        # Compute naive/natural gaps only if they were generated
        gap_and_mono_naive = None
        if imgs_mono_naive is not None:
            print(f"  Computing image gap: {ACTIVE_LOGICAL_ANCHOR_LABEL} vs monolithic (naive) ...")
            gap_and_mono_naive = compute_image_gap(
                imgs_and, imgs_mono_naive, clip_eval, lpips_fn, device
            )

        gap_and_mono_natural = None
        if imgs_mono_natural is not None:
            print(f"  Computing image gap: {ACTIVE_LOGICAL_ANCHOR_LABEL} vs monolithic (natural) ...")
            gap_and_mono_natural = compute_image_gap(
                imgs_and, imgs_mono_natural, clip_eval, lpips_fn, device
            )

        metrics = {
            "pair":         (c1, c2),
            "pair_index":   int(pair_idx),
            "slug":         pair_slug,
            "model_family": args.model_family,
            "model_id":     args.model_id,
            "pair_group":   pair_group,
            "taxonomy_group_key": taxonomy_group_key,
            "taxonomy_group_label": taxonomy_group_label,
            "is_representative_pair": is_representative_pair,
            "logical_anchor": ACTIVE_LOGICAL_ANCHOR_KEY,
            "logical_anchor_label": ACTIVE_LOGICAL_ANCHOR_LABEL,
            "co3_support": CO3_TRAJECTORY_SUPPORT,
            "n_seeds":      len(args.seeds),
            "monolithic_baseline": args.monolithic_baseline,
            "mono_prompt_active": mono_prompt_active,
            "mono_prompt_naive": mono_prompt_naive,
            "mono_prompt_natural": mono_prompt_natural,
            "gap_and_mono": gap_and_mono,  # backward compat (active baseline)
        }
        if gap_and_mono_naive is not None:
            metrics["gap_and_mono_naive"] = gap_and_mono_naive
        if gap_and_mono_natural is not None:
            metrics["gap_and_mono_natural"] = gap_and_mono_natural

        if not args._skip_inverter:
            print(f"  Computing image gap: {ACTIVE_LOGICAL_ANCHOR_LABEL} vs p* ...")
            gap_and_pstar = compute_image_gap(imgs_and, imgs_pstar, clip_eval, lpips_fn, device)

            print(f"  Computing trajectory gap: {ACTIVE_LOGICAL_ANCHOR_LABEL} vs p* ...")
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
            print(f"  Computing image gap: {ACTIVE_LOGICAL_ANCHOR_LABEL} vs PEZ ...")
            gap_and_pstar_pez = compute_image_gap(imgs_and, imgs_pstar_pez_all, clip_eval, lpips_fn, device)
            metrics["gap_and_pstar_pez"] = gap_and_pstar_pez
            metrics["pez_prompts"] = pez_prompts
            print(f"    {ACTIVE_LOGICAL_ANCHOR_LABEL} vs PEZ:  CLIP={gap_and_pstar_pez['clip_cos']:.4f}")

        if args.z2t:
            print(f"  Computing image gap: {ACTIVE_LOGICAL_ANCHOR_LABEL} vs Z2T ...")
            gap_and_pstar_z2t = compute_image_gap(imgs_and, imgs_pstar_z2t_all, clip_eval, lpips_fn, device)
            metrics["gap_and_pstar_z2t"] = gap_and_pstar_z2t
            metrics["z2t_prompts"] = z2t_prompts
            metrics["z2t_pool_mode"] = args.z2t_pool_mode
            metrics["z2t_source"] = ACTIVE_LOGICAL_ANCHOR_KEY
            print(f"    {ACTIVE_LOGICAL_ANCHOR_LABEL} vs Z2T:  CLIP={gap_and_pstar_z2t['clip_cos']:.4f}")

        if args.sdipc:
            print(f"  Computing image gap: {ACTIVE_LOGICAL_ANCHOR_LABEL} vs SD-IPC (PoE p*) ...")
            gap_and_pstar_sdipc = compute_image_gap(imgs_and, imgs_pstar_sdipc_all, clip_eval, lpips_fn, device)
            metrics["gap_and_pstar_sdipc"] = gap_and_pstar_sdipc
            print(f"    {ACTIVE_LOGICAL_ANCHOR_LABEL} vs PoE p* (SD-IPC):  CLIP={gap_and_pstar_sdipc['clip_cos']:.4f}")

            # PoE p* (SD-IPC) vs monolithic — key comparison:
            # If gap_pstar_sdipc_mono_naive is LOW: the recovered p* looks like naive
            #   "A and B" text → natural language CAN represent the PoE composition.
            # If gap_pstar_sdipc_mono_naive is HIGH: p* captures something that simple
            #   concatenation cannot → confirms a composability gap w.r.t. monolithic.
            if imgs_mono_naive is not None:
                print(f"  Computing image gap: PoE p* (SD-IPC) vs monolithic (naive) ...")
                gap_pstar_sdipc_mono_naive = compute_image_gap(
                    imgs_pstar_sdipc_all, imgs_mono_naive, clip_eval, lpips_fn, device
                )
                metrics["gap_pstar_sdipc_mono_naive"] = gap_pstar_sdipc_mono_naive
            else:
                gap_pstar_sdipc_mono_naive = None

            if imgs_mono_natural is not None:
                print(f"  Computing image gap: PoE p* (SD-IPC) vs monolithic (natural) ...")
                gap_pstar_sdipc_mono_natural = compute_image_gap(
                    imgs_pstar_sdipc_all, imgs_mono_natural, clip_eval, lpips_fn, device
                )
                metrics["gap_pstar_sdipc_mono_natural"] = gap_pstar_sdipc_mono_natural
            else:
                gap_pstar_sdipc_mono_natural = None

            if gap_pstar_sdipc_mono_naive is not None or gap_pstar_sdipc_mono_natural is not None:
                msg = "    PoE p* vs mono:"
                if gap_pstar_sdipc_mono_naive is not None:
                    msg += f"  naive: CLIP={gap_pstar_sdipc_mono_naive['clip_cos']:.4f}"
                if gap_pstar_sdipc_mono_natural is not None:
                    msg += f"  | natural: CLIP={gap_pstar_sdipc_mono_natural['clip_cos']:.4f}"
                print(msg)

            # CO3 p* gap metrics (single-seed: grid_seed only, since pstar_co3_sdipc
            # is generated once per pair at grid_seed).
            # gap_pstar_co3_sdipc_mono_naive : CO3 p* vs naive "A and B" — does CO3
            #   capture something beyond simple text concatenation?
            # gap_and_pstar_co3_sdipc        : PoE vs CO3 p* — cross-method alignment:
            #   how close is the logical composition to the CO3-grounded text vector?
            _img_pstar_co3 = decoded_for_grid.get("pstar_co3_sdipc")
            if _img_pstar_co3 is not None and imgs_mono_naive is not None:
                # Single-image tensors: ensure (1, 3, H, W) for compute_image_gap.
                _img_pstar_co3_b = _img_pstar_co3 if _img_pstar_co3.dim() == 4 else _img_pstar_co3.unsqueeze(0)
                _img_mono_naive_b = imgs_mono_naive[grid_seed_idx].unsqueeze(0)
                _img_and_b = imgs_and[grid_seed_idx].unsqueeze(0)
                print(f"  Computing image gap: CO3 p* (SD-IPC) vs monolithic (naive) ...")
                gap_pstar_co3_sdipc_mono_naive = compute_image_gap(
                    _img_pstar_co3_b, _img_mono_naive_b, clip_eval, lpips_fn, device
                )
                print(f"  Computing image gap: {ACTIVE_LOGICAL_ANCHOR_LABEL} vs CO3 p* (cross-method) ...")
                gap_and_pstar_co3_sdipc = compute_image_gap(
                    _img_and_b, _img_pstar_co3_b, clip_eval, lpips_fn, device
                )
                metrics["gap_pstar_co3_sdipc_mono_naive"] = gap_pstar_co3_sdipc_mono_naive
                metrics["gap_and_pstar_co3_sdipc"]        = gap_and_pstar_co3_sdipc
                print(
                    f"    CO3 p* vs mono (naive): CLIP={gap_pstar_co3_sdipc_mono_naive['clip_cos']:.4f}  "
                    f"| {ACTIVE_LOGICAL_ANCHOR_LABEL} vs CO3 p*: CLIP={gap_and_pstar_co3_sdipc['clip_cos']:.4f}"
                )

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
                model_family=args.model_family,
            )
            save_comparison_grid_manifest(
                img_dir / "comparison_grid.png",
                c1=c1, c2=c2,
                seeds=args.seeds,
                pstar_source="inverter",
                n_display=4,
                mono_prompt=mono_prompt_active,
                model_family=args.model_family,
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
                    model_family=args.model_family,
                )
                save_comparison_grid_manifest(
                    img_dir / f"comparison_grid_mono_{alt_mode}.png",
                    c1=c1, c2=c2,
                    seeds=args.seeds,
                    pstar_source="inverter",
                    n_display=4,
                    mono_prompt=alt_prompt,
                    model_family=args.model_family,
                )

        # ---- Visualisation 3: trajectory MDS/PCA (seed 0, one plot per active p* source) ----
        _traj_sources = []
        if not args._skip_inverter:
            _traj_sources.append((trackers_pstar[0],     f"{family_label} (p* — CLIP inverter)", ""))
        if args.sdipc and trackers_pstar_sdipc:
            _traj_sources.append((trackers_pstar_sdipc[0], f"{family_label} (PoE p* — SD-IPC)", "_sdipc"))
        if args.pez and trackers_pstar_pez:
            _traj_sources.append((trackers_pstar_pez[0], f"{family_label} (p* — PEZ)", "_pez"))
        if args.z2t and trackers_pstar_z2t:
            _traj_sources.append((trackers_pstar_z2t[0], f"{family_label} (p* — Z2T)", "_z2t"))

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
                model_family=args.model_family,
                logical_anchor_label=ACTIVE_LOGICAL_ANCHOR_LABEL,
            )

        print(f"\n  Results for '{c1}' AND '{c2}':")
        if not args._skip_inverter:
            print(f"    {ACTIVE_LOGICAL_ANCHOR_LABEL} vs p*:        CLIP={gap_and_pstar['clip_cos']:.4f}  "
                  f"lat_mse={lat_gap['lat_mse']:.4f}  "
                  f"traj_mse={traj_gap['traj_mse_mean']:.4f}  "
                  f"traj_cos={traj_gap['traj_cos_mean']:.4f}")
            if "lpips" in gap_and_pstar:
                print(f"    LPIPS ({ACTIVE_LOGICAL_ANCHOR_LABEL}/p*):  {gap_and_pstar['lpips']:.4f}")
        print(
            f"    {ACTIVE_LOGICAL_ANCHOR_LABEL} vs mono (active/{args.monolithic_baseline}): "
            f"CLIP={gap_and_mono['clip_cos']:.4f}"
        )
        if gap_and_mono_naive is not None:
            print(f"    {ACTIVE_LOGICAL_ANCHOR_LABEL} vs mono (naive):   CLIP={gap_and_mono_naive['clip_cos']:.4f}")
        if gap_and_mono_natural is not None:
            print(f"    {ACTIVE_LOGICAL_ANCHOR_LABEL} vs mono (natural): CLIP={gap_and_mono_natural['clip_cos']:.4f}")
        if args.sdipc and "gap_pstar_sdipc_mono_naive" in metrics:
            print(
                f"    PoE p* vs mono (naive):   CLIP={metrics['gap_pstar_sdipc_mono_naive']['clip_cos']:.4f}  "
                f"[gap = {gap_and_pstar_sdipc['clip_cos'] - metrics['gap_pstar_sdipc_mono_naive']['clip_cos']:+.4f} vs PoE↔p*]"
            )

    # -------------------------------------------------------------------------
    # Determine which new pstar columns were actually written this run
    # -------------------------------------------------------------------------
    new_term_cols = [
        "model_family",
        "model_id",
        "pair_group",
        "pair_slug",
        "taxonomy_group_key",
        "taxonomy_group_label",
        "is_representative_pair",
        "logical_anchor",
        "logical_anchor_label",
        "monolithic_baseline",
        "mono_prompt_active",
        "mono_prompt_naive",
        "mono_prompt_natural",
        "d_T_poe",
        "d_T_mono",
        "d_T_mono_to_poe",
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
        "model_family",
        "model_id",
        "pair_group",
        "pair_slug",
        "taxonomy_group_key",
        "taxonomy_group_label",
        "is_representative_pair",
        "logical_anchor",
        "logical_anchor_label",
        "monolithic_baseline",
        "d_t_poe",
        "d_t_mono",
        "d_t_mono_to_poe",
        "d_t_mono_naive",
        "d_t_mono_natural",
        "d_t_c1",
        "d_t_c2",
    ]
    new_gap_keys  = [
        "model_family",
        "model_id",
        "pair_group",
        "taxonomy_group_key",
        "taxonomy_group_label",
        "is_representative_pair",
        "logical_anchor",
        "logical_anchor_label",
        "co3_support",
        "monolithic_baseline",
        "mono_prompt_active",
        "mono_prompt_naive",
        "mono_prompt_natural",
        "gap_and_mono",
        "gap_and_mono_naive",
        "gap_and_mono_natural",
    ]

    if not args._skip_inverter:
        new_term_cols += ["d_T_pstar", "d_T_pstar_inv", "d_T_pstar_inv_to_mono"]
        if args.anchor == "mean":
            new_term_cols += ["d_T_pstar_meananchor", "d_T_pstar_inv_meananchor"]
        new_traj_cols += ["d_t_pstar", "d_t_pstar_inv", "d_t_pstar_inv_to_mono"]
        new_gap_keys  += ["gap_and_pstar", "gap_and_pstar_inv"]
    if args.pez:
        new_term_cols += ["d_T_pstar_pez", "d_T_pstar_pez_to_mono"]
        if args.anchor == "mean":
            new_term_cols.append("d_T_pstar_pez_meananchor")
        new_traj_cols += ["d_t_pstar_pez", "d_t_pstar_pez_to_mono"]
        new_gap_keys.append("gap_and_pstar_pez")
    if args.z2t:
        new_term_cols += ["d_T_pstar_z2t", "d_T_pstar_z2t_to_mono"]
        if args.anchor == "mean":
            new_term_cols.append("d_T_pstar_z2t_meananchor")
        new_traj_cols += ["d_t_pstar_z2t", "d_t_pstar_z2t_to_mono"]
        new_gap_keys.append("gap_and_pstar_z2t")
    if args.sdipc:
        new_term_cols += ["d_T_pstar_sdipc", "d_T_pstar_sdipc_to_mono"]
        if args.anchor == "mean":
            new_term_cols.append("d_T_pstar_sdipc_meananchor")
        new_traj_cols += ["d_t_pstar_sdipc", "d_t_pstar_sdipc_to_mono"]
        new_gap_keys.extend([
            "gap_and_pstar_sdipc",
            "gap_pstar_sdipc_mono_naive",
            "gap_pstar_sdipc_mono_natural",
            "gap_pstar_co3_sdipc_mono_naive",
            "gap_and_pstar_co3_sdipc",
        ])

    # -------------------------------------------------------------------------
    # Write / merge output files
    # -------------------------------------------------------------------------
    metrics_dir = out_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    taxonomy_manifest_path = metrics_dir / "taxonomy_manifest.json"
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
        # within_and is source-agnostic, but rewrite it to keep taxonomy metadata
        # and avoid stale rows from older runs without the enriched schema.
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

    taxonomy_manifest_path.write_text(
        json.dumps(
            {
                **taxonomy_manifest_payload(),
                "model_family": args.model_family,
                "model_id": args.model_id,
                "regime": args.regime,
                "seeds": list(args.seeds),
                "grid_seed": int(args._grid_seed),
                "monolithic_baseline": args.monolithic_baseline,
                "logical_anchor": ACTIVE_LOGICAL_ANCHOR_KEY,
                "logical_anchor_label": ACTIVE_LOGICAL_ANCHOR_LABEL,
                "legacy_anchor_namespace": "gap_and_* / d_T_* / d_t_* remain backward-compatible aliases for the active logical anchor",
                "co3_support": CO3_TRAJECTORY_SUPPORT,
            },
            indent=2,
        )
    )

    print(f"\nAll results {mode} to {out_root}/")
    print(f"  metrics/taxonomy_manifest.json    → {taxonomy_manifest_path}")
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
