#!/usr/bin/env python3
"""Diagnose the SDXL joint-vs-marginal denoising gap from shared initial noise.

This standalone research script compares four SDXL denoising conditions from the
same seed and scheduler:

  - solo A
  - solo B
  - monolithic joint prompt (oracle joint condition)
  - PoE score addition baseline

It records guided denoiser outputs in noise-prediction space and computes:

    marginal_mean_t    = 0.5 * (pred_A_t + pred_B_t)
    delta_joint_marg_t = pred_joint_t - marginal_mean_t
    delta_joint_poe_t  = pred_joint_t - pred_poe_t

Outputs include:
  - run_config.json
  - joint_marginal_gap.json
  - delta_norm_vs_step.png
  - delta_cosine_vs_step.png
  - delta_heatmaps_joint_marg.png
  - delta_heatmaps_joint_poe.png
  - decoded_endpoints.png
  - raw selected-step delta tensors under tensors/
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "eccv2026" / "joint_marginal_gap_sdxl"
CONDITION_ORDER = ["solo_a", "solo_b", "monolithic", "poe"]
CONDITION_LABELS = {
    "solo_a": "A",
    "solo_b": "B",
    "monolithic": "A∧B",
    "poe": "PoE",
}
CONDITION_COLORS = {
    "solo_a": "#C84C5B",
    "solo_b": "#2B6F97",
    "monolithic": "#3B8D5B",
    "poe": "#D9872B",
}
DELTA_STYLES = {
    "joint_marginal": {"label": r"$\delta_t$ = joint - marginal mean", "color": "#7B4EA3"},
    "joint_poe": {"label": r"joint - PoE", "color": "#D14E3F"},
}


@dataclass
class StepRecord:
    step: int
    timestep: int
    noise_pred: torch.Tensor
    noise_uncond: torch.Tensor


def lower_first(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]


def default_joint_prompt(prompt_a: str, prompt_b: str) -> str:
    return f"{prompt_a} and {lower_first(prompt_b)}"


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the SDXL joint-vs-marginal denoising gap from shared noise."
    )
    parser.add_argument("--prompt-a", required=True, help="Prompt for concept A.")
    parser.add_argument("--prompt-b", required=True, help="Prompt for concept B.")
    parser.add_argument(
        "--joint-prompt",
        default=None,
        help='Monolithic oracle joint prompt. Defaults to "prompt_a and prompt_b".',
    )
    parser.add_argument("--seed", type=int, default=42, help="Shared initial-noise seed.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--height", type=int, default=1024, help="Render height.")
    parser.add_argument("--width", type=int, default=1024, help="Render width.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="SDXL model ID.")
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Optional negative prompt used as the CFG negative branch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Torch device, e.g. "auto", "cuda", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory for timestamped outputs.",
    )
    parser.add_argument(
        "--selected-steps",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit step indices for selected-step heatmaps.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Saved figure DPI.")
    parser.add_argument(
        "--poe-mode",
        type=str,
        choices=["energy_add", "avg_cfg"],
        default="energy_add",
        help=(
            "PoE score-combination mode. "
            "'energy_add' (default): additive energy PoE (Composable Diffusion, Liu et al. 2022). "
            "'avg_cfg': averaged conditional corrections with single unconditional anchor."
        ),
    )
    return parser.parse_args()


def _load_runtime_modules() -> dict[str, Any]:
    try:
        from diffusers import DDIMScheduler, EulerDiscreteScheduler
        from notebooks.composition_experiments import get_prompt_conditioning
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
        "get_prompt_conditioning": get_prompt_conditioning,
        "get_latents": get_latents,
        "get_image": get_image,
        "get_sd_models": get_sd_models,
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
    scheduler = runtime["DDIMScheduler"].from_pretrained(model_id, subfolder="scheduler")
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
def run_cfg_collect_predictions(
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
    height: int,
    width: int,
    negative_prompt: str | None,
) -> tuple[torch.Tensor, list[StepRecord]]:
    runtime = _load_runtime_modules()
    ddim = scheduler  # set_timesteps already called inside get_latents()

    latents = latents.to(device=device, dtype=dtype)
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

    records: list[StepRecord] = []
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
        records.append(
            StepRecord(
                step=i,
                timestep=int(t.item()),
                noise_pred=noise_pred.detach().float().cpu(),
                noise_uncond=noise_neg.detach().float().cpu(),
            )
        )
        latents = ddim.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    return latents, records


@torch.no_grad()
def run_poe_collect_predictions(
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
    height: int,
    width: int,
    negative_prompt: str | None,
    poe_mode: str = "energy_add",
) -> tuple[torch.Tensor, list[StepRecord]]:
    runtime = _load_runtime_modules()
    ddim = scheduler  # set_timesteps already called inside get_latents()

    latents = latents.to(device=device, dtype=dtype)
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

    records: list[StepRecord] = []
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

        if poe_mode == "energy_add":
            # Composable Diffusion energy-addition PoE (Liu et al. 2022):
            #   s_PoE = s_neg + gs*(s_A - s_neg) + gs*(s_B - s_neg)
            #         = s_neg*(1 - 2*gs) + gs*s_A + gs*s_B
            noise_pred = (
                noise_neg
                + guidance_scale * (noise_a - noise_neg)
                + guidance_scale * (noise_b - noise_neg)
            )
        elif poe_mode == "avg_cfg":
            # Average-CFG PoE: one unconditional anchor, averaged conditional corrections.
            #   s_PoE = s_neg + gs*(0.5*(s_A + s_B) - s_neg)
            noise_pred = noise_neg + guidance_scale * (
                0.5 * (noise_a + noise_b) - noise_neg
            )
        else:
            raise ValueError(f"Unknown poe_mode={poe_mode!r}. Choose 'energy_add' or 'avg_cfg'.")
        records.append(
            StepRecord(
                step=i,
                timestep=int(t.item()),
                noise_pred=noise_pred.detach().float().cpu(),
                noise_uncond=noise_neg.detach().float().cpu(),
            )
        )
        latents = ddim.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    return latents, records


def _norm_l2(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.reshape(-1)).item())


def _rms(tensor: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(torch.square(tensor))).item())


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float | None:
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    denom = torch.linalg.vector_norm(a_flat) * torch.linalg.vector_norm(b_flat)
    denom_value = float(denom.item())
    if denom_value <= eps:
        return None
    return float(torch.dot(a_flat, b_flat).item() / denom_value)


def _selected_steps(num_steps: int, explicit: list[int] | None) -> list[int]:
    last = num_steps - 1
    if explicit is not None:
        values = sorted(set(explicit))
    else:
        values = sorted(set([0, int(round(0.15 * last)), last // 2, last]))
    if not values:
        raise ValueError("selected_steps resolved to an empty list.")
    bad = [step for step in values if step < 0 or step > last]
    if bad:
        raise ValueError(
            f"Selected steps {bad} are out of range for num_inference_steps={num_steps}. "
            f"Valid step indices are 0..{last}."
        )
    return values


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _tensor_to_pixel_l2(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.squeeze(0).float().numpy()
    return np.sqrt(np.sum(np.square(arr), axis=0))


def save_selected_tensors(
    tensor_map: dict[str, list[torch.Tensor]],
    selected_steps: list[int],
    tensors_dir: Path,
    run_dir: Path,
) -> dict[str, dict[str, str]]:
    tensors_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, dict[str, str]] = {}
    for delta_name, tensors in tensor_map.items():
        delta_dir = tensors_dir / delta_name
        delta_dir.mkdir(parents=True, exist_ok=True)
        saved[delta_name] = {}
        for step in selected_steps:
            out_path = delta_dir / f"step_{step:03d}.npy"
            np.save(out_path, tensors[step].numpy())
            saved[delta_name][str(step)] = _relative(out_path, run_dir)
    return saved


def plot_delta_norms(step_rows: list[dict[str, Any]], save_path: Path) -> None:
    steps = [row["step"] for row in step_rows]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.4), sharey=False)

    panels = [
        # (axis, title, ylabel, key_jm, key_jp)
        (axes[0], "ε-space (noise prediction)", "L2 norm",
         "joint_minus_marginal_l2", "joint_minus_poe_l2"),
        (axes[1], "x₀-space  |δ| = (√(1−ᾱ)/√ᾱ) · |δ_ε|", "L2 norm (x₀ units)",
         "joint_minus_marginal_l2_x0", "joint_minus_poe_l2_x0"),
    ]
    for ax, title, ylabel, key_jm, key_jp in panels:
        for delta_name, key in (("joint_marginal", key_jm), ("joint_poe", key_jp)):
            style = DELTA_STYLES[delta_name]
            ax.plot(steps, [row[key] for row in step_rows],
                    label=style["label"], color=style["color"], lw=2.4)
        ax.set_xlabel("Denoising step")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(framealpha=0.9, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Joint-oracle discrepancy magnitude", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_delta_cosines(step_rows: list[dict[str, Any]], save_path: Path) -> None:
    steps = [row["step"] for row in step_rows]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    cosine_specs = (
        ("joint_marginal", "cosine_joint_marginal_vs_poe", "cos(delta_joint_marg, PoE)"),
        ("joint_poe", "cosine_joint_poe_vs_poe", "cos(joint-PoE, PoE)"),
        ("joint_poe", "cosine_joint_poe_vs_poe_residual", "cos(joint-PoE, PoE-marg)"),
    )
    for delta_name, key, label in cosine_specs:
        style = DELTA_STYLES[delta_name]
        series = [math.nan if row[key] is None else row[key] for row in step_rows]
        ax.plot(steps, series, label=label, color=style["color"], lw=2.2, alpha=0.9)

    ax.axhline(0.0, color="#444444", lw=1.0, ls="--", alpha=0.55)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Joint discrepancy direction alignment")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmaps(
    delta_tensors: list[torch.Tensor],
    selected_steps: list[int],
    step_rows: list[dict[str, Any]],
    *,
    title: str,
    save_path: Path,
) -> None:
    maps = [_tensor_to_pixel_l2(delta_tensors[step]) for step in selected_steps]
    vmax = max(float(np.max(m)) for m in maps)
    vmax = vmax if vmax > 0 else 1.0

    fig, axes = plt.subplots(
        1,
        len(selected_steps),
        figsize=(3.7 * len(selected_steps), 3.8),
        squeeze=False,
    )
    for ax, step, heatmap in zip(axes[0], selected_steps, maps, strict=True):
        row = step_rows[step]
        im = ax.imshow(heatmap, cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(f"step {step}\nt={row['timestep']}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title + "\nper-pixel channel L2, shared color scale", fontsize=12)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("L2 over channels", rotation=90)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_decoded_endpoints(images: dict[str, Any], prompts: dict[str, str], save_path: Path) -> None:
    fig, axes = plt.subplots(1, len(CONDITION_ORDER), figsize=(13.8, 4.6))
    for ax, cond in zip(axes, CONDITION_ORDER, strict=True):
        ax.imshow(np.asarray(images[cond]))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(CONDITION_LABELS[cond], color=CONDITION_COLORS[cond], fontsize=11, fontweight="bold")
        ax.set_xlabel(prompts[cond], fontsize=8.5, labelpad=8)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.8)
            spine.set_color(CONDITION_COLORS[cond])

    fig.suptitle("Decoded endpoints from shared initial noise", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _x0_snr_factor(alphas_cumprod: torch.Tensor, timestep: int) -> float:
    """Return sqrt(1 - ᾱ_t) / sqrt(ᾱ_t), the scalar that converts an ε-space
    delta to the equivalent x₀-space delta:
        δ_{x0,t} = -(√(1-ᾱ_t)/√ᾱ_t) · δ_{ε,t}
    Since x_t is shared across conditions it cancels, so the sign does not
    matter for norms. The magnitude is what we record.
    """
    abar = float(alphas_cumprod[timestep].item())
    abar = max(abar, 1e-8)
    return math.sqrt(max(1.0 - abar, 0.0)) / math.sqrt(abar)


def build_metrics(
    prompt_a: str,
    prompt_b: str,
    joint_prompt: str,
    seed: int,
    model_id: str,
    negative_prompt: str | None,
    guidance_scale: float,
    num_inference_steps: int,
    height: int,
    width: int,
    selected_steps: list[int],
    records: dict[str, list[StepRecord]],
    selected_tensor_paths: dict[str, dict[str, str]],
    run_dir: Path,
    scheduler: Any,
) -> tuple[dict[str, Any], dict[str, list[torch.Tensor]]]:
    alphas_cumprod = scheduler.alphas_cumprod  # shape (num_train_timesteps,), on CPU

    step_rows: list[dict[str, Any]] = []
    delta_tensors = {
        "joint_marginal": [],
        "joint_poe": [],
    }

    for step in range(num_inference_steps):
        pred_a = records["solo_a"][step].noise_pred
        pred_b = records["solo_b"][step].noise_pred
        pred_joint = records["monolithic"][step].noise_pred
        pred_poe = records["poe"][step].noise_pred

        marginal_mean = 0.5 * (pred_a + pred_b)
        delta_joint_marg = pred_joint - marginal_mean
        delta_joint_poe = pred_joint - pred_poe
        poe_residual = pred_poe - marginal_mean

        delta_tensors["joint_marginal"].append(delta_joint_marg)
        delta_tensors["joint_poe"].append(delta_joint_poe)

        timestep = int(records["monolithic"][step].timestep)
        snr_factor = _x0_snr_factor(alphas_cumprod, timestep)

        l2_jm = _norm_l2(delta_joint_marg)
        l2_jp = _norm_l2(delta_joint_poe)
        rms_jm = _rms(delta_joint_marg)
        rms_jp = _rms(delta_joint_poe)

        step_rows.append(
            {
                "step": step,
                "timestep": timestep,
                "snr_factor": snr_factor,
                # ε-prediction space
                "joint_minus_marginal_l2": l2_jm,
                "joint_minus_marginal_rms": rms_jm,
                "joint_minus_poe_l2": l2_jp,
                "joint_minus_poe_rms": rms_jp,
                # x₀-prediction space (exact rescaling, no extra UNet calls)
                # |δ_{x0}| = (√(1-ᾱ)/√ᾱ) · |δ_ε|
                "joint_minus_marginal_l2_x0": snr_factor * l2_jm,
                "joint_minus_marginal_rms_x0": snr_factor * rms_jm,
                "joint_minus_poe_l2_x0": snr_factor * l2_jp,
                "joint_minus_poe_rms_x0": snr_factor * rms_jp,
                "cosine_joint_marginal_vs_poe": _cosine(delta_joint_marg, pred_poe),
                "cosine_joint_poe_vs_poe": _cosine(delta_joint_poe, pred_poe),
                "cosine_joint_poe_vs_poe_residual": _cosine(delta_joint_poe, poe_residual),
            }
        )

    metrics = {
        "created_at": datetime.now().isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "prediction_space": "guided_noise_prediction",
        "model_family": "sdxl",
        "model_id": model_id,
        "seed": int(seed),
        "negative_prompt": negative_prompt,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "joint_prompt": joint_prompt,
        "poe_prompt": f"{prompt_a} | {prompt_b}",
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "height": int(height),
        "width": int(width),
        "selected_steps": selected_steps,
        "selected_step_tensor_paths": selected_tensor_paths,
        "plots": {
            "delta_norm_vs_step": _relative(run_dir / "delta_norm_vs_step.png", run_dir),
            "delta_cosine_vs_step": _relative(run_dir / "delta_cosine_vs_step.png", run_dir),
            "delta_heatmaps_joint_marg": _relative(run_dir / "delta_heatmaps_joint_marg.png", run_dir),
            "delta_heatmaps_joint_poe": _relative(run_dir / "delta_heatmaps_joint_poe.png", run_dir),
            "decoded_endpoints": _relative(run_dir / "decoded_endpoints.png", run_dir),
        },
        "steps": step_rows,
    }
    return metrics, delta_tensors


def main() -> None:
    args = parse_args()
    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError(f"height={args.height} and width={args.width} must both be divisible by 8 for SDXL latents.")

    joint_prompt = args.joint_prompt or default_joint_prompt(args.prompt_a, args.prompt_b)
    selected_steps = _selected_steps(args.num_inference_steps, args.selected_steps)
    run_dir = args.output_dir / f"joint_marginal_gap_sdxl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tensors_dir = run_dir / "tensors"

    device = resolve_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("=" * 88)
    print("Joint-marginal oracle diagnostic for SDXL")
    print(f"Prompt A     : {args.prompt_a}")
    print(f"Prompt B     : {args.prompt_b}")
    print(f"Joint prompt : {joint_prompt}")
    print(f"Model        : {args.model_id}")
    print(f"Seed         : {args.seed}")
    print(f"Steps        : {args.num_inference_steps}")
    print(f"Guidance     : {args.guidance_scale}")
    print(f"Device       : {device}")
    print(f"Selected     : {selected_steps}")
    print(f"Output dir   : {run_dir}")
    print("=" * 88)

    config = {
        "created_at": datetime.now().isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "model_family": "sdxl",
        "model_id": args.model_id,
        "prompt_a": args.prompt_a,
        "prompt_b": args.prompt_b,
        "joint_prompt": joint_prompt,
        "negative_prompt": args.negative_prompt,
        "seed": int(args.seed),
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "height": int(args.height),
        "width": int(args.width),
        "device": str(device),
        "selected_steps": selected_steps,
        "prediction_space": "guided_noise_prediction",
        "scheduler_class": "DDIMScheduler",
        "poe_mode": args.poe_mode,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    tokenizer, tokenizer_2, text_encoder, text_encoder_2, unet, vae, scheduler = _load_models(
        args.model_id,
        device,
        dtype,
    )
    runtime = _load_runtime_modules()
    x_t = runtime["get_latents"](
        scheduler,
        z_channels=4,
        device=device,
        dtype=dtype,
        num_inference_steps=args.num_inference_steps,
        batch_size=1,
        latent_width=args.width // 8,
        latent_height=args.height // 8,
        seed=args.seed,
    )
    print("\nCollecting denoising predictions from shared initial noise...")
    latents_a, records_a = run_cfg_collect_predictions(
        x_t.clone(),
        args.prompt_a,
        scheduler,
        unet,
        tokenizer,
        text_encoder,
        tokenizer_2,
        text_encoder_2,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        batch_size=1,
        device=device,
        dtype=dtype,
        height=args.height,
        width=args.width,
        negative_prompt=args.negative_prompt,
    )
    latents_b, records_b = run_cfg_collect_predictions(
        x_t.clone(),
        args.prompt_b,
        scheduler,
        unet,
        tokenizer,
        text_encoder,
        tokenizer_2,
        text_encoder_2,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        batch_size=1,
        device=device,
        dtype=dtype,
        height=args.height,
        width=args.width,
        negative_prompt=args.negative_prompt,
    )
    latents_joint, records_joint = run_cfg_collect_predictions(
        x_t.clone(),
        joint_prompt,
        scheduler,
        unet,
        tokenizer,
        text_encoder,
        tokenizer_2,
        text_encoder_2,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        batch_size=1,
        device=device,
        dtype=dtype,
        height=args.height,
        width=args.width,
        negative_prompt=args.negative_prompt,
    )
    latents_poe, records_poe = run_poe_collect_predictions(
        x_t.clone(),
        args.prompt_a,
        args.prompt_b,
        scheduler,
        unet,
        tokenizer,
        text_encoder,
        tokenizer_2,
        text_encoder_2,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        batch_size=1,
        device=device,
        dtype=dtype,
        height=args.height,
        width=args.width,
        negative_prompt=args.negative_prompt,
        poe_mode=args.poe_mode,
    )

    records = {
        "solo_a": records_a,
        "solo_b": records_b,
        "monolithic": records_joint,
        "poe": records_poe,
    }

    placeholder_tensor_paths = {name: {} for name in ("joint_marginal", "joint_poe")}
    metrics, delta_tensors = build_metrics(
        args.prompt_a,
        args.prompt_b,
        joint_prompt,
        args.seed,
        args.model_id,
        args.negative_prompt,
        args.guidance_scale,
        args.num_inference_steps,
        args.height,
        args.width,
        selected_steps,
        records,
        placeholder_tensor_paths,
        run_dir,
        scheduler,
    )
    selected_tensor_paths = save_selected_tensors(delta_tensors, selected_steps, tensors_dir, run_dir)
    metrics["selected_step_tensor_paths"] = selected_tensor_paths
    metrics["poe_mode"] = args.poe_mode

    print("Rendering plots...")
    plot_delta_norms(metrics["steps"], run_dir / "delta_norm_vs_step.png")
    plot_delta_cosines(metrics["steps"], run_dir / "delta_cosine_vs_step.png")
    plot_delta_heatmaps(
        delta_tensors["joint_marginal"],
        selected_steps,
        metrics["steps"],
        title="Joint minus marginal mean discrepancy",
        save_path=run_dir / "delta_heatmaps_joint_marg.png",
    )
    plot_delta_heatmaps(
        delta_tensors["joint_poe"],
        selected_steps,
        metrics["steps"],
        title="Joint minus PoE discrepancy",
        save_path=run_dir / "delta_heatmaps_joint_poe.png",
    )

    images = {
        "solo_a": runtime["get_image"](vae, latents_a, nrow=1, ncol=1),
        "solo_b": runtime["get_image"](vae, latents_b, nrow=1, ncol=1),
        "monolithic": runtime["get_image"](vae, latents_joint, nrow=1, ncol=1),
        "poe": runtime["get_image"](vae, latents_poe, nrow=1, ncol=1),
    }
    plot_decoded_endpoints(
        images,
        {
            "solo_a": args.prompt_a,
            "solo_b": args.prompt_b,
            "monolithic": joint_prompt,
            "poe": f"{args.prompt_a} | {args.prompt_b}",
        },
        run_dir / "decoded_endpoints.png",
    )

    (run_dir / "joint_marginal_gap.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved diagnostic artifacts to {run_dir}")


if __name__ == "__main__":
    main()
