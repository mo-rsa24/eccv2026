#!/usr/bin/env python3
"""Generate a standalone SDXL spatial-relations PoE figure.

This script is intentionally outside the canonical taxonomy and paper figure
pipeline. It renders a single qualitative grid:

Rows:
  1. Plain co-occurrence baseline
  2. Left/right of the photo
  3. Left/right of the frame
  4. Left/right of the couch
  5. CLIP-style "A photo of ..." phrasing

Columns:
  A, B, A∧B, PoE, [optional] CO3

Each row reuses the same initial latent noise across all generated columns.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "eccv2026" / "spatial_relations_poe"
CO3_REPO = PROJECT_ROOT / "compositions" / "co3"

BASE_CONDITION_ORDER = ["prompt_a", "prompt_b", "monolithic", "poe"]
CONDITION_LABELS = {
    "prompt_a": "A",
    "prompt_b": "B",
    "monolithic": "A∧B",
    "poe": "PoE",
    "co3": "CO3",
}
CONDITION_COLORS = {
    "prompt_a": "#C84C5B",
    "prompt_b": "#2B6F97",
    "monolithic": "#3B8D5B",
    "poe": "#D9872B",
    "co3": "#5B8E7D",
}

HEADER_BBOX_ALPHA = 0.96
ROW_LABEL_BBOX = {
    "boxstyle": "round,pad=0.28",
    "facecolor": "#F6F8FB",
    "edgecolor": "#D7DEE8",
    "linewidth": 0.9,
    "alpha": 0.98,
}

CO3_DEFAULTS = dict(
    guidance_scale=0.8,
    n_timesteps=50,
    num_ts_to_correct=6,
    num_latent_corrector_steps=5,
    num_resampling_steps=3,
    corrector_algo="co3-hybrid",
    modulate_comp_weights=True,
    beta=0.9,
    lmda=0.8,
    negative_prompt="",
)
CO3_SD_VERSION = "xl"


@dataclass
class RowSpec:
    row: int
    name: str
    label: str
    prompt_a: str
    prompt_b: str
    semantic_prompt: str


class LatentTrajectoryCollector:
    """Minimal trajectory collector copied from the notebook helper.

    Kept local so this script does not import the full analysis notebook module,
    which pulls in optional plotting dependencies such as seaborn.
    """

    def __init__(self, num_steps: int, batch_size: int, z_channels: int, latent_height: int, latent_width: int):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.shape = (z_channels, latent_height, latent_width)
        self.trajectories = torch.zeros(
            (num_steps + 1, batch_size, z_channels, latent_height, latent_width)
        )
        self.velocities = torch.zeros(
            (num_steps, batch_size, z_channels, latent_height, latent_width)
        )
        self.sigmas = torch.zeros(num_steps + 1)
        self.timesteps = torch.zeros(num_steps)

    def store_step(
        self,
        step: int,
        latents: torch.Tensor,
        velocity: torch.Tensor | None,
        sigma: float,
        timestep: float | None,
    ) -> None:
        self.trajectories[step] = latents.detach().cpu()
        if velocity is not None:
            self.velocities[step] = velocity.detach().cpu()
        self.sigmas[step] = sigma
        if timestep is not None:
            self.timesteps[step] = timestep

    def store_final(self, latents: torch.Tensor) -> None:
        self.trajectories[-1] = latents.detach().cpu()


DEFAULT_ROWS = [
    {
        "name": "plain_cooccurrence",
        "label": "Row 1  Plain co-occurrence",
        "prompt_a": "a dog",
        "prompt_b": "a cat",
    },
    {
        "name": "photo_reference",
        "label": "Row 2  Left/right of the photo",
        "prompt_a": "A dog on the left of the photo",
        "prompt_b": "A cat on the right of the photo",
    },
    {
        "name": "frame_reference",
        "label": "Row 3  Left/right of the frame",
        "prompt_a": "A dog on the left of the frame",
        "prompt_b": "A cat on the right of the frame",
    },
    {
        "name": "couch_reference",
        "label": "Row 4  Left/right of the couch",
        "prompt_a": "A dog on the left of the couch",
        "prompt_b": "A cat on the right of the couch",
    },
    {
        "name": "clip_photo_prefix",
        "label": "Row 5  CLIP-style photo prefix",
        "prompt_a": "A photo of a dog on the left of the photo",
        "prompt_b": "A photo of a cat on the right of the photo",
    },
]


def lower_first(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]


def slugify(text: str) -> str:
    clean = text.lower().strip()
    for old, new in (
        (" ", "_"),
        ("/", "_"),
        ("'", ""),
        (",", ""),
        (".", ""),
        ("(", ""),
        (")", ""),
    ):
        clean = clean.replace(old, new)
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_")


def default_semantic_prompt(prompt_a: str, prompt_b: str) -> str:
    return f"{prompt_a} and {lower_first(prompt_b)}"


def load_row_specs(path: Path | None) -> list[RowSpec]:
    if path is None:
        raw_rows = DEFAULT_ROWS
    else:
        raw_rows = json.loads(path.read_text())

    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("Row specs must be a non-empty JSON list.")

    row_specs: list[RowSpec] = []
    for idx, raw in enumerate(raw_rows, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Row spec {idx} is not a JSON object.")
        prompt_a = str(raw["prompt_a"]).strip()
        prompt_b = str(raw["prompt_b"]).strip()
        if not prompt_a or not prompt_b:
            raise ValueError(f"Row spec {idx} must define non-empty prompt_a and prompt_b.")
        row_specs.append(
            RowSpec(
                row=idx,
                name=str(raw.get("name") or f"row_{idx:02d}"),
                label=str(raw.get("label") or f"Row {idx}"),
                prompt_a=prompt_a,
                prompt_b=prompt_b,
                semantic_prompt=str(raw.get("semantic_prompt") or default_semantic_prompt(prompt_a, prompt_b)).strip(),
            )
        )
    return row_specs


def get_condition_order(with_co3: bool) -> list[str]:
    return BASE_CONDITION_ORDER + (["co3"] if with_co3 else [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a standalone SDXL spatial-relations PoE figure."
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="SDXL model ID.")
    parser.add_argument("--seed", type=int, default=42, help="Shared base seed for all rows.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--height", type=int, default=1024, help="Render height.")
    parser.add_argument("--width", type=int, default=1024, help="Render width.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory for timestamped outputs.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Optional negative prompt used as the CFG negative branch.",
    )
    parser.add_argument(
        "--row-specs-json",
        type=Path,
        default=None,
        help="Optional JSON file overriding the default row specs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Torch device, e.g. "auto", "cuda", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Saved figure DPI.",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=2.55,
        help="Per-image cell width/height in matplotlib inches.",
    )
    parser.add_argument(
        "--with-co3",
        action="store_true",
        help="Append a 5th CO3 column. If --run-co3 is not set, row-local co3.png files will be loaded.",
    )
    parser.add_argument(
        "--run-co3",
        action="store_true",
        help="Generate CO3 in-process for each row. Implies --with-co3.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def get_prompt_conditioning(
    prompt: str,
    batch_size: int,
    tokenizer: Any,
    text_encoder: Any,
    device: torch.device,
    height: int = 512,
    width: int = 512,
    tokenizer_2: Any = None,
    text_encoder_2: Any = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    from notebooks.utils import get_text_embedding

    prompt_batch = [prompt] * batch_size

    if tokenizer_2 is None or text_encoder_2 is None:
        prompt_embeds = get_text_embedding(prompt_batch, tokenizer, text_encoder, device)
        return prompt_embeds, None

    prompt_embeds, pooled_prompt_embeds = get_text_embedding(
        prompt_batch,
        tokenizer,
        text_encoder,
        device,
        tokenizer_2=tokenizer_2,
        text_encoder_2=text_encoder_2,
        return_pooled=True,
    )

    add_time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]],
        device=device,
        dtype=prompt_embeds.dtype,
    ).repeat(batch_size, 1)

    added_cond_kwargs = {
        "text_embeds": pooled_prompt_embeds.to(device=device, dtype=prompt_embeds.dtype),
        "time_ids": add_time_ids,
    }
    return prompt_embeds, added_cond_kwargs


def _load_runtime_modules() -> dict[str, Any]:
    try:
        from diffusers import DDIMScheduler, EulerDiscreteScheduler
        from notebooks.dynamics import get_latents
        from notebooks.utils import get_image, get_sd_models
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing runtime dependencies for SDXL generation. "
            "Install the repo's diffusion stack in the active environment."
        ) from exc

    return {
        "DDIMScheduler": DDIMScheduler,
        "EulerDiscreteScheduler": EulerDiscreteScheduler,
        "LatentTrajectoryCollector": LatentTrajectoryCollector,
        "get_prompt_conditioning": get_prompt_conditioning,
        "get_latents": get_latents,
        "get_image": get_image,
        "get_sd_models": get_sd_models,
    }


def _load_co3_runtime_modules() -> dict[str, Any]:
    try:
        if str(CO3_REPO) not in sys.path:
            sys.path.insert(0, str(CO3_REPO))
        from composers.Co3 import Co3
        from composers.config import Co3Config
        from composers.utils_custom import seed_everything
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CO3 dependencies are unavailable. Run from the CO3-capable environment or omit --with-co3/--run-co3."
        ) from exc

    return {
        "Co3": Co3,
        "Co3Config": Co3Config,
        "seed_everything": seed_everything,
    }


def _load_models(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    runtime = _load_runtime_modules()
    models = runtime["get_sd_models"](model_id=model_id, dtype=dtype, device=device)
    if not models.get("is_sdxl", False):
        raise ValueError(f"Model {model_id} is not recognized as SDXL by notebooks.utils.get_sd_models.")
    scheduler = runtime["EulerDiscreteScheduler"].from_pretrained(model_id, subfolder="scheduler")
    return (
        models["tokenizer"],
        models["tokenizer_2"],
        models["text_encoder"],
        models["text_encoder_2"],
        models["unet"],
        models["vae"],
        scheduler,
    )


def _prepare_negative_branch(
    negative_prompt: str | None,
    batch_size: int,
    tokenizer: Any,
    tokenizer_2: Any,
    text_encoder: Any,
    text_encoder_2: Any,
    device: torch.device,
    *,
    height: int,
    width: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    runtime = _load_runtime_modules()
    negative_text = negative_prompt or ""
    return runtime["get_prompt_conditioning"](
        negative_text,
        batch_size=batch_size,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        tokenizer_2=tokenizer_2,
        text_encoder_2=text_encoder_2,
        device=device,
        height=height,
        width=width,
    )


@torch.no_grad()
def run_cfg_with_tracking(
    latents: torch.Tensor,
    prompt: str,
    scheduler: Any,
    unet: Any,
    tokenizer: Any,
    text_encoder: Any,
    tokenizer_2: Any,
    text_encoder_2: Any,
    *,
    guidance_scale: float,
    num_inference_steps: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    model_id: str,
    euler_init_noise_sigma: float,
    height: int,
    width: int,
    negative_prompt: str | None,
) -> tuple[torch.Tensor, Any]:
    runtime = _load_runtime_modules()
    ddim = runtime["DDIMScheduler"].from_pretrained(model_id, subfolder="scheduler")
    ddim.set_timesteps(num_inference_steps)

    latents = (latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    cond_emb, cond_kwargs = runtime["get_prompt_conditioning"](
        prompt,
        batch_size=batch_size,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        tokenizer_2=tokenizer_2,
        text_encoder_2=text_encoder_2,
        device=device,
        height=height,
        width=width,
    )
    neg_emb, neg_kwargs = _prepare_negative_branch(
        negative_prompt,
        batch_size,
        tokenizer,
        tokenizer_2,
        text_encoder,
        text_encoder_2,
        device,
        height=height,
        width=width,
    )

    tracker = runtime["LatentTrajectoryCollector"](
        num_inference_steps,
        batch_size,
        latents.shape[1],
        latents.shape[2],
        latents.shape[3],
    )

    extra_step_kwargs: dict[str, Any] = {}
    if "eta" in inspect.signature(ddim.step).parameters:
        extra_step_kwargs["eta"] = 0.0

    for i, t in enumerate(ddim.timesteps):
        latent_model_input = ddim.scale_model_input(latents, t)
        noise_neg = unet(
            latent_model_input,
            t,
            encoder_hidden_states=neg_emb,
            added_cond_kwargs=neg_kwargs,
            timestep_cond=None,
        ).sample
        noise_cond = unet(
            latent_model_input,
            t,
            encoder_hidden_states=cond_emb,
            added_cond_kwargs=cond_kwargs,
            timestep_cond=None,
        ).sample
        noise_pred = noise_neg + guidance_scale * (noise_cond - noise_neg)
        tracker.store_step(i, latents, noise_pred, float(i) / num_inference_steps, float(t))
        latents = ddim.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    tracker.store_final(latents)
    return latents, tracker


@torch.no_grad()
def run_poe_with_tracking(
    latents: torch.Tensor,
    prompt_a: str,
    prompt_b: str,
    scheduler: Any,
    unet: Any,
    tokenizer: Any,
    text_encoder: Any,
    tokenizer_2: Any,
    text_encoder_2: Any,
    *,
    guidance_scale: float,
    num_inference_steps: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    model_id: str,
    euler_init_noise_sigma: float,
    height: int,
    width: int,
    negative_prompt: str | None,
) -> tuple[torch.Tensor, Any]:
    runtime = _load_runtime_modules()
    ddim = runtime["DDIMScheduler"].from_pretrained(model_id, subfolder="scheduler")
    ddim.set_timesteps(num_inference_steps)

    latents = (latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    neg_emb, neg_kwargs = _prepare_negative_branch(
        negative_prompt,
        batch_size,
        tokenizer,
        tokenizer_2,
        text_encoder,
        text_encoder_2,
        device,
        height=height,
        width=width,
    )
    a_emb, a_kwargs = runtime["get_prompt_conditioning"](
        prompt_a,
        batch_size=batch_size,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        tokenizer_2=tokenizer_2,
        text_encoder_2=text_encoder_2,
        device=device,
        height=height,
        width=width,
    )
    b_emb, b_kwargs = runtime["get_prompt_conditioning"](
        prompt_b,
        batch_size=batch_size,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        tokenizer_2=tokenizer_2,
        text_encoder_2=text_encoder_2,
        device=device,
        height=height,
        width=width,
    )

    tracker = runtime["LatentTrajectoryCollector"](
        num_inference_steps,
        batch_size,
        latents.shape[1],
        latents.shape[2],
        latents.shape[3],
    )

    extra_step_kwargs: dict[str, Any] = {}
    if "eta" in inspect.signature(ddim.step).parameters:
        extra_step_kwargs["eta"] = 0.0

    for i, t in enumerate(ddim.timesteps):
        latent_model_input = ddim.scale_model_input(latents, t)
        noise_neg = unet(
            latent_model_input,
            t,
            encoder_hidden_states=neg_emb,
            added_cond_kwargs=neg_kwargs,
            timestep_cond=None,
        ).sample
        noise_a = unet(
            latent_model_input,
            t,
            encoder_hidden_states=a_emb,
            added_cond_kwargs=a_kwargs,
            timestep_cond=None,
        ).sample
        noise_b = unet(
            latent_model_input,
            t,
            encoder_hidden_states=b_emb,
            added_cond_kwargs=b_kwargs,
            timestep_cond=None,
        ).sample

        noise_pred = (
            noise_neg
            + guidance_scale * (noise_a - noise_neg)
            + guidance_scale * (noise_b - noise_neg)
        )
        tracker.store_step(i, latents, noise_pred, float(i) / num_inference_steps, float(t))
        latents = ddim.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    tracker.store_final(latents)
    return latents, tracker


def fit_with_padding(image: Image.Image, target_size: int, fill_color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    contained = ImageOps.contain(image.convert("RGB"), (target_size, target_size), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), color=fill_color)
    x = (target_size - contained.width) // 2
    y = (target_size - contained.height) // 2
    canvas.paste(contained, (x, y))
    return canvas


def condition_prompt_map(row_spec: RowSpec) -> dict[str, str]:
    prompt_map = {
        "prompt_a": row_spec.prompt_a,
        "prompt_b": row_spec.prompt_b,
        "monolithic": row_spec.semantic_prompt,
        "poe": f"{row_spec.prompt_a} | {row_spec.prompt_b}",
    }
    prompt_map["co3"] = f"{row_spec.prompt_a} + {row_spec.prompt_b}"
    return prompt_map


def _make_co3_config(
    prompt_a: str,
    prompt_b: str,
    out_dir: Path,
    seed: int,
    *,
    height: int,
    width: int,
) -> Any:
    runtime = _load_co3_runtime_modules()
    prompt_orig = f"{prompt_a} and {lower_first(prompt_b)}"
    prompt = f"{prompt_a}+{prompt_b}+{prompt_orig}"
    return runtime["Co3Config"](
        prompt=prompt,
        prompt_orig=prompt_orig,
        seeds=[seed],
        output_path=str(out_dir),
        output_path_all=str(out_dir),
        sd_version=CO3_SD_VERSION,
        resolution_h=height,
        resolution_w=width,
        **CO3_DEFAULTS,
    )


def _load_co3_model(
    row_spec: RowSpec,
    out_dir: Path,
    seed: int,
    *,
    height: int,
    width: int,
) -> tuple[Any, Any]:
    runtime = _load_co3_runtime_modules()
    config = _make_co3_config(
        row_spec.prompt_a,
        row_spec.prompt_b,
        out_dir,
        seed,
        height=height,
        width=width,
    )
    return runtime["Co3"](config), config


def _run_co3_for_row(
    co3_model: Any,
    row_spec: RowSpec,
    out_dir: Path,
    seed: int,
    *,
    height: int,
    width: int,
) -> Image.Image:
    runtime = _load_co3_runtime_modules()
    config = _make_co3_config(
        row_spec.prompt_a,
        row_spec.prompt_b,
        out_dir,
        seed,
        height=height,
        width=width,
    )
    co3_model.config = config
    co3_model.config.latent_corrector_ts = (
        co3_model.scheduler.timesteps[: config.num_ts_to_correct]
        if config.num_ts_to_correct >= 0
        else []
    )
    co3_model.prepare_prompts(config)
    co3_model.prepare_embeds()
    runtime["seed_everything"](seed)
    co3_model.config.seed = seed
    co3_model.config.output_dir = str(out_dir)
    images = co3_model.run_sampling()
    return images[0]


def wrap_prompt_text(text: str, width: int) -> str:
    return "\n".join(
        textwrap.wrap(
            text,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def draw_grid(
    row_specs: list[RowSpec],
    row_image_paths: list[dict[str, Path]],
    output_path: Path,
    *,
    condition_order: list[str],
    dpi: int,
    cell_size: float,
) -> None:
    n_rows = len(row_specs)
    n_cols = len(condition_order)
    wrapped_prompt_maps = [
        {cond_key: wrap_prompt_text(prompt, width=28) for cond_key, prompt in condition_prompt_map(row_spec).items()}
        for row_spec in row_specs
    ]
    max_caption_lines = max(
        max(prompt.count("\n") + 1 for prompt in wrapped_prompt_map.values())
        for wrapped_prompt_map in wrapped_prompt_maps
    )
    caption_height_ratio = max(0.33, 0.16 * max_caption_lines)
    fig_width = 2.6 + n_cols * cell_size
    fig_height = 1.15 + n_rows * (cell_size + 0.62 + 0.24 * max_caption_lines)
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(
        n_rows + 1,
        n_cols + 1,
        width_ratios=[1.9] + [1.0] * n_cols,
        height_ratios=[0.28] + [1.0] * n_rows,
        left=0.03,
        right=0.99,
        top=0.985,
        bottom=0.025,
        wspace=0.06,
        hspace=0.16,
    )

    corner_ax = fig.add_subplot(grid[0, 0])
    corner_ax.axis("off")
    corner_ax.text(
        0.0,
        0.45,
        "Spatial prompt phrasing",
        ha="left",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        color="#25303B",
    )

    for col_idx, cond_key in enumerate(condition_order, start=1):
        ax = fig.add_subplot(grid[0, col_idx])
        ax.axis("off")
        color = CONDITION_COLORS[cond_key]
        title = ax.text(
            0.5,
            0.45,
            CONDITION_LABELS[cond_key],
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=color,
        )
        title.set_bbox(
            {
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 1.15,
                "alpha": HEADER_BBOX_ALPHA,
            }
        )

    for row_idx, (row_spec, row_paths) in enumerate(zip(row_specs, row_image_paths, strict=True), start=1):
        label_ax = fig.add_subplot(grid[row_idx, 0])
        label_ax.axis("off")
        label_ax.text(
            0.02,
            0.5,
            row_spec.label,
            ha="left",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            color="#2A2F36",
            bbox=ROW_LABEL_BBOX,
            wrap=True,
        )

        prompt_map = wrapped_prompt_maps[row_idx - 1]
        for col_idx, cond_key in enumerate(condition_order, start=1):
            cell_grid = grid[row_idx, col_idx].subgridspec(
                2,
                1,
                height_ratios=[1.0, caption_height_ratio],
                hspace=0.03,
            )
            image_ax = fig.add_subplot(cell_grid[0, 0])
            text_ax = fig.add_subplot(cell_grid[1, 0])
            image = plt.imread(row_paths[cond_key])
            image_ax.imshow(image)
            image_ax.set_xticks([])
            image_ax.set_yticks([])
            image_ax.set_facecolor("white")
            for spine in image_ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.2)
                spine.set_color(CONDITION_COLORS[cond_key])

            text_ax.axis("off")
            text_ax.set_facecolor("white")
            text_ax.text(
                0.5,
                0.94,
                prompt_map[cond_key],
                ha="center",
                va="top",
                fontsize=8.2,
                color="#313842",
                linespacing=1.1,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_row_preview(
    row_dir: Path,
    row_images: dict[str, Image.Image],
    row_spec: RowSpec,
    *,
    condition_order: list[str],
    dpi: int,
) -> Path:
    preview_path = row_dir / "row_grid.png"
    prompt_map = {cond_key: wrap_prompt_text(prompt, width=28) for cond_key, prompt in condition_prompt_map(row_spec).items()}
    max_caption_lines = max(prompt_map[cond_key].count("\n") + 1 for cond_key in condition_order)
    caption_height_ratio = max(0.34, 0.16 * max_caption_lines)
    fig = plt.figure(figsize=(2.6 * len(condition_order), 4.9))
    outer = fig.add_gridspec(1, len(condition_order), wspace=0.14)
    for idx, cond_key in enumerate(condition_order):
        cell = outer[0, idx].subgridspec(2, 1, height_ratios=[1.0, caption_height_ratio], hspace=0.04)
        ax = fig.add_subplot(cell[0, 0])
        text_ax = fig.add_subplot(cell[1, 0])
        ax.imshow(np.asarray(row_images[cond_key]))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            CONDITION_LABELS[cond_key],
            fontsize=9.0,
            fontweight="bold",
            color=CONDITION_COLORS[cond_key],
            pad=6,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.8)
            spine.set_color(CONDITION_COLORS[cond_key])
        text_ax.axis("off")
        text_ax.text(
            0.5,
            0.94,
            prompt_map[cond_key],
            ha="center",
            va="top",
            fontsize=8.1,
            color="#313842",
            linespacing=1.1,
        )
    fig.suptitle(row_spec.label, fontsize=11.0, fontweight="bold", y=0.99)
    fig.savefig(preview_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return preview_path


def row_output_dir(run_dir: Path, row_spec: RowSpec) -> Path:
    return run_dir / f"row_{row_spec.row:02d}_{slugify(row_spec.name)}"


def run_row(
    row_spec: RowSpec,
    run_dir: Path,
    models_tuple: tuple[Any, Any, Any, Any, Any, Any, Any],
    *,
    model_id: str,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    negative_prompt: str | None,
    with_co3: bool,
    co3_model: Any | None,
    dpi: int,
) -> dict[str, Any]:
    runtime = _load_runtime_modules()
    tokenizer, tokenizer_2, text_encoder, text_encoder_2, unet, vae, scheduler = models_tuple
    row_dir = row_output_dir(run_dir, row_spec)
    row_dir.mkdir(parents=True, exist_ok=True)

    x_t = runtime["get_latents"](
        scheduler,
        z_channels=4,
        device=device,
        dtype=dtype,
        num_inference_steps=num_inference_steps,
        batch_size=1,
        latent_width=width // 8,
        latent_height=height // 8,
        seed=seed,
    )
    euler_sigma = float(getattr(scheduler, "init_noise_sigma", 1.0))

    latents_a, tracker_a = run_cfg_with_tracking(
        x_t.clone(),
        row_spec.prompt_a,
        scheduler,
        unet,
        tokenizer,
        text_encoder,
        tokenizer_2,
        text_encoder_2,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        batch_size=1,
        device=device,
        dtype=dtype,
        model_id=model_id,
        euler_init_noise_sigma=euler_sigma,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
    )
    latents_b, tracker_b = run_cfg_with_tracking(
        x_t.clone(),
        row_spec.prompt_b,
        scheduler,
        unet,
        tokenizer,
        text_encoder,
        tokenizer_2,
        text_encoder_2,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        batch_size=1,
        device=device,
        dtype=dtype,
        model_id=model_id,
        euler_init_noise_sigma=euler_sigma,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
    )
    latents_mono, tracker_mono = run_cfg_with_tracking(
        x_t.clone(),
        row_spec.semantic_prompt,
        scheduler,
        unet,
        tokenizer,
        text_encoder,
        tokenizer_2,
        text_encoder_2,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        batch_size=1,
        device=device,
        dtype=dtype,
        model_id=model_id,
        euler_init_noise_sigma=euler_sigma,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
    )
    latents_poe, tracker_poe = run_poe_with_tracking(
        x_t.clone(),
        row_spec.prompt_a,
        row_spec.prompt_b,
        scheduler,
        unet,
        tokenizer,
        text_encoder,
        tokenizer_2,
        text_encoder_2,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        batch_size=1,
        device=device,
        dtype=dtype,
        model_id=model_id,
        euler_init_noise_sigma=euler_sigma,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
    )

    row_images = {
        "prompt_a": runtime["get_image"](vae, latents_a, nrow=1, ncol=1),
        "prompt_b": runtime["get_image"](vae, latents_b, nrow=1, ncol=1),
        "monolithic": runtime["get_image"](vae, latents_mono, nrow=1, ncol=1),
        "poe": runtime["get_image"](vae, latents_poe, nrow=1, ncol=1),
    }
    if with_co3:
        co3_image_path = row_dir / "co3.png"
        if co3_model is not None:
            row_images["co3"] = _run_co3_for_row(
                co3_model,
                row_spec,
                row_dir,
                seed,
                height=height,
                width=width,
            )
        elif co3_image_path.exists():
            row_images["co3"] = Image.open(co3_image_path).convert("RGB")
        else:
            raise FileNotFoundError(
                f"{co3_image_path} not found. Use --run-co3 to generate it or place a row-local co3.png before rerunning."
            )

    output_paths: dict[str, str] = {}
    filename_map = {
        "prompt_a": "prompt_a.png",
        "prompt_b": "prompt_b.png",
        "monolithic": "monolithic.png",
        "poe": "poe.png",
    }
    if with_co3:
        filename_map["co3"] = "co3.png"
    for cond_key, filename in filename_map.items():
        image_path = row_dir / filename
        row_images[cond_key].save(image_path)
        output_paths[cond_key] = str(image_path)

    condition_order = get_condition_order(with_co3)
    preview_path = save_row_preview(
        row_dir,
        row_images,
        row_spec,
        condition_order=condition_order,
        dpi=dpi,
    )

    row_manifest = {
        "row": row_spec.row,
        "name": row_spec.name,
        "label": row_spec.label,
        "seed": int(seed),
        "model_family": "sdxl",
        "model_id": model_id,
        "negative_prompt": negative_prompt,
        "prompt_a": row_spec.prompt_a,
        "prompt_b": row_spec.prompt_b,
        "semantic_prompt": row_spec.semantic_prompt,
        "poe_prompt": f"{row_spec.prompt_a} | {row_spec.prompt_b}",
        "height": int(height),
        "width": int(width),
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "latent_shape": list(x_t.shape),
        "outputs": {
            **output_paths,
            "row_grid": str(preview_path),
        },
        "trajectory_steps": {
            "prompt_a": int(tracker_a.trajectories.shape[0]),
            "prompt_b": int(tracker_b.trajectories.shape[0]),
            "monolithic": int(tracker_mono.trajectories.shape[0]),
            "poe": int(tracker_poe.trajectories.shape[0]),
        },
    }
    if with_co3:
        row_manifest["trajectory_steps"]["co3"] = None
    (row_dir / "row_manifest.json").write_text(json.dumps(row_manifest, indent=2))
    return row_manifest


def build_run_manifest(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    row_specs: list[RowSpec],
    row_manifests: list[dict[str, Any]],
    figure_path: Path,
) -> dict[str, Any]:
    condition_order = get_condition_order(args.with_co3)
    return {
        "created_at": datetime.now().isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "model_family": "sdxl",
        "model_id": args.model_id,
        "seed": int(args.seed),
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "height": int(args.height),
        "width": int(args.width),
        "device": str(args.device),
        "layout": {
            "rows": len(row_specs),
            "columns": len(condition_order),
            "column_order": condition_order,
            "column_labels": CONDITION_LABELS,
            "condition_colors": CONDITION_COLORS,
        },
        "row_specs": [asdict(spec) for spec in row_specs],
        "rows": row_manifests,
        "figure_path": str(figure_path),
        "run_dir": str(run_dir),
    }


def main() -> None:
    args = parse_args()
    if args.run_co3:
        args.with_co3 = True
    row_specs = load_row_specs(args.row_specs_json)
    run_dir = args.output_dir / f"spatial_relations_poe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("=" * 88)
    print("Standalone SDXL spatial-relations PoE grid")
    print(f"Rows        : {len(row_specs)}")
    print(f"Columns     : {len(get_condition_order(args.with_co3))}")
    print(f"Model       : {args.model_id}")
    print(f"Seed        : {args.seed}")
    print(f"Steps       : {args.num_inference_steps}")
    print(f"Scale       : {args.guidance_scale}")
    print(f"Device      : {device}")
    print(f"CO3         : {'enabled' if args.with_co3 else 'disabled'}")
    if args.with_co3:
        print(f"CO3 source  : {'in-process generation' if args.run_co3 else 'row-local co3.png'}")
    print(f"Output dir  : {run_dir}")
    print("=" * 88)

    models_tuple = _load_models(args.model_id, device, dtype)
    co3_model = None
    if args.run_co3:
        bootstrap_row_dir = row_output_dir(run_dir, row_specs[0])
        bootstrap_row_dir.mkdir(parents=True, exist_ok=True)
        co3_model, _ = _load_co3_model(
            row_specs[0],
            bootstrap_row_dir,
            args.seed,
            height=args.height,
            width=args.width,
        )
    row_manifests: list[dict[str, Any]] = []
    row_image_paths: list[dict[str, Path]] = []

    try:
        for row_spec in tqdm(row_specs, desc="Rows"):
            print(f"\n[ROW {row_spec.row:02d}] {row_spec.label}")
            print(f"  A         : {row_spec.prompt_a}")
            print(f"  B         : {row_spec.prompt_b}")
            print(f"  A∧B       : {row_spec.semantic_prompt}")
            row_manifest = run_row(
                row_spec,
                run_dir,
                models_tuple,
                model_id=args.model_id,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                height=args.height,
                width=args.width,
                device=device,
                dtype=dtype,
                negative_prompt=args.negative_prompt,
                with_co3=args.with_co3,
                co3_model=co3_model,
                dpi=args.dpi,
            )
            row_manifests.append(row_manifest)
            row_image_paths.append(
                {
                    cond_key: Path(row_manifest["outputs"][cond_key])
                    for cond_key in get_condition_order(args.with_co3)
                }
            )
    finally:
        del models_tuple
        if device.type == "cuda":
            torch.cuda.empty_cache()

    figure_path = run_dir / "spatial_relations_grid.png"
    draw_grid(
        row_specs,
        row_image_paths,
        figure_path,
        condition_order=get_condition_order(args.with_co3),
        dpi=args.dpi,
        cell_size=args.cell_size,
    )

    run_manifest = build_run_manifest(
        args=args,
        run_dir=run_dir,
        row_specs=row_specs,
        row_manifests=row_manifests,
        figure_path=figure_path,
    )
    (run_dir / "manifest.json").write_text(json.dumps(run_manifest, indent=2))

    print(f"\nSaved figure  : {figure_path}")
    print(f"Saved manifest: {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
