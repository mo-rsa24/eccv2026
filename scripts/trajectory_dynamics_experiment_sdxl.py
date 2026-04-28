"""
Trajectory Dynamics Experiment — SDXL

Adapted from trajectory_dynamics_experiment.py to support SDXL models.
Compares reverse diffusion trajectories under different conditioning:
  1. Prompt A (e.g., "a dog")
  2. Prompt B (e.g., "a cat")
  3. CLIP Monolithic AND (e.g., "a dog and a cat")
  4. PoE AND (prompt A ∧ prompt B)

All conditions start from the exact same initial Gaussian noise x_T.
Trajectories are recorded at every timestep, jointly projected via PCA,
and visualized as time-gradient-colored curves in 2D.
"""

import numpy as np
import torch
from diffusers import DDIMScheduler
import inspect as _inspect
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from notebooks.utils import get_sd_models
from notebooks.composition_experiments import LatentTrajectoryCollector, get_prompt_conditioning


@torch.no_grad()
def _encode_sdxl(texts, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device, height, width):
    """
    Encode prompts for SDXL using both text encoders.
    Returns concatenated embeddings plus SDXL added conditioning kwargs.
    """
    prompt_embeds, added_cond_kwargs = get_prompt_conditioning(
        texts[0],
        batch_size=len(texts),
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=device,
        height=height,
        width=width,
        tokenizer_2=tokenizer_2,
        text_encoder_2=text_encoder_2,
    )
    return prompt_embeds, added_cond_kwargs


def _guided_eps(eps_cond: torch.Tensor, eps_uncond: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


def _poe_guided_eps(
    eps_a: torch.Tensor,
    eps_b: torch.Tensor,
    eps_uncond: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    return eps_uncond + guidance_scale * (eps_a - eps_uncond) + guidance_scale * (eps_b - eps_uncond)


def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1)


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = _flatten_batch(a).float()
    b_flat = _flatten_batch(b).float()
    cos = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=1).mean()
    return float((1.0 - cos).item())


@torch.no_grad()
def run_sdxl_reference_gap_with_tracking(
    latents,
    prompt_a,
    prompt_b,
    monolithic_prompt,
    scheduler,
    unet,
    tokenizer,
    text_encoder,
    tokenizer_2,
    text_encoder_2,
    guidance_scale=7.5,
    num_inference_steps=50,
    batch_size=1,
    device=torch.device("cuda"),
    dtype=torch.float16,
    model_id=None,
    euler_init_noise_sigma=1.0,
    height=1024,
    width=1024,
    interaction_gap_mode="reference_and_poe",
):
    """
    Joint mono-vs-PoE SDXL run with theorem-aligned interaction-gap logging.

    Returns a dict with final latents, trackers, per-step scalar diagnostics, and
    representative spatial maps derived from the reference and PoE residuals.
    """

    if model_id is not None:
        ddim = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    else:
        ddim = DDIMScheduler.from_config(scheduler.config)
    ddim.set_timesteps(num_inference_steps)

    latents_mono = (latents / euler_init_noise_sigma).to(dtype=dtype, device=device)
    latents_poe = latents_mono.clone()

    uncond_emb, uncond_kwargs = _encode_sdxl([""] * batch_size, tokenizer, tokenizer_2,
                                             text_encoder, text_encoder_2, device, height, width)
    a_emb, a_kwargs = _encode_sdxl([prompt_a] * batch_size, tokenizer, tokenizer_2,
                                   text_encoder, text_encoder_2, device, height, width)
    b_emb, b_kwargs = _encode_sdxl([prompt_b] * batch_size, tokenizer, tokenizer_2,
                                   text_encoder, text_encoder_2, device, height, width)
    mono_emb, mono_kwargs = _encode_sdxl([monolithic_prompt] * batch_size, tokenizer, tokenizer_2,
                                         text_encoder, text_encoder_2, device, height, width)

    tracker_mono = LatentTrajectoryCollector(
        num_inference_steps,
        batch_size,
        latents_mono.shape[1],
        latents_mono.shape[2],
        latents_mono.shape[3],
    )
    tracker_poe = LatentTrajectoryCollector(
        num_inference_steps,
        batch_size,
        latents_poe.shape[1],
        latents_poe.shape[2],
        latents_poe.shape[3],
    )

    extra_step_kwargs = {}
    if "eta" in _inspect.signature(ddim.step).parameters:
        extra_step_kwargs["eta"] = 0.0

    step_rows = []
    delta_ref_l2_t = []
    delta_poe_l2_t = []
    traj_mse_mono_poe_t = []
    traj_cosdist_mono_poe_t = []
    peak_ref_value = -1.0
    peak_poe_value = -1.0
    peak_ref_step = 0
    peak_poe_step = 0
    peak_ref_absmap = None
    peak_poe_absmap = None
    onset_ref_step = None

    from tqdm.auto import tqdm as _tqdm

    for i, t in _tqdm(enumerate(ddim.timesteps), total=num_inference_steps,
                      desc=f"Interaction gap SDXL B={batch_size}", leave=False):
        latent_mono_input = ddim.scale_model_input(latents_mono, t)
        latent_poe_input = ddim.scale_model_input(latents_poe, t)

        with torch.no_grad():
            noise_mono_all = unet(
                latent_mono_input.repeat(4, 1, 1, 1),
                t,
                encoder_hidden_states=torch.cat([uncond_emb, a_emb, b_emb, mono_emb], dim=0),
                added_cond_kwargs={
                    "text_embeds": torch.cat(
                        [
                            uncond_kwargs["text_embeds"],
                            a_kwargs["text_embeds"],
                            b_kwargs["text_embeds"],
                            mono_kwargs["text_embeds"],
                        ],
                        dim=0,
                    ),
                    "time_ids": torch.cat(
                        [
                            uncond_kwargs["time_ids"],
                            a_kwargs["time_ids"],
                            b_kwargs["time_ids"],
                            mono_kwargs["time_ids"],
                        ],
                        dim=0,
                    ),
                },
                timestep_cond=None,
            ).sample
            eps_uncond_on_mono, eps_a_on_mono, eps_b_on_mono, eps_mono_on_mono = noise_mono_all.chunk(4)

            noise_poe_all = unet(
                latent_poe_input.repeat(4, 1, 1, 1),
                t,
                encoder_hidden_states=torch.cat([uncond_emb, a_emb, b_emb, mono_emb], dim=0),
                added_cond_kwargs={
                    "text_embeds": torch.cat(
                        [
                            uncond_kwargs["text_embeds"],
                            a_kwargs["text_embeds"],
                            b_kwargs["text_embeds"],
                            mono_kwargs["text_embeds"],
                        ],
                        dim=0,
                    ),
                    "time_ids": torch.cat(
                        [
                            uncond_kwargs["time_ids"],
                            a_kwargs["time_ids"],
                            b_kwargs["time_ids"],
                            mono_kwargs["time_ids"],
                        ],
                        dim=0,
                    ),
                },
                timestep_cond=None,
            ).sample
            eps_uncond_on_poe, eps_a_on_poe, eps_b_on_poe, eps_mono_on_poe = noise_poe_all.chunk(4)

        eps_ref_on_mono = _guided_eps(eps_mono_on_mono, eps_uncond_on_mono, guidance_scale)
        eps_poe_on_mono = _poe_guided_eps(eps_a_on_mono, eps_b_on_mono, eps_uncond_on_mono, guidance_scale)
        delta_ref_eps = eps_ref_on_mono - eps_poe_on_mono

        eps_ref_on_poe = _guided_eps(eps_mono_on_poe, eps_uncond_on_poe, guidance_scale)
        eps_poe_on_poe = _poe_guided_eps(eps_a_on_poe, eps_b_on_poe, eps_uncond_on_poe, guidance_scale)
        delta_poe_eps = eps_ref_on_poe - eps_poe_on_poe

        tracker_mono.store_step(i, latents_mono, eps_ref_on_mono, float(i) / num_inference_steps, t.item())
        tracker_poe.store_step(i, latents_poe, eps_poe_on_poe, float(i) / num_inference_steps, t.item())

        latents_mono = ddim.step(eps_ref_on_mono, t, latents_mono, **extra_step_kwargs).prev_sample
        latents_poe = ddim.step(eps_poe_on_poe, t, latents_poe, **extra_step_kwargs).prev_sample

        delta_ref_l2 = float(torch.linalg.vector_norm(delta_ref_eps, dim=(1, 2, 3)).pow(2).mean().item())
        delta_ref_mean_abs = float(delta_ref_eps.abs().mean().item())
        delta_poe_l2 = float(torch.linalg.vector_norm(delta_poe_eps, dim=(1, 2, 3)).pow(2).mean().item())
        delta_poe_mean_abs = float(delta_poe_eps.abs().mean().item())
        traj_mse = float(((tracker_mono.trajectories[i].to(dtype=torch.float32) - tracker_poe.trajectories[i].to(dtype=torch.float32)) ** 2).mean().item())
        traj_cosdist = _cosine_distance(eps_ref_on_mono, eps_poe_on_poe)

        delta_ref_l2_t.append(delta_ref_l2)
        delta_poe_l2_t.append(delta_poe_l2)
        traj_mse_mono_poe_t.append(traj_mse)
        traj_cosdist_mono_poe_t.append(traj_cosdist)

        ref_absmap = delta_ref_eps.abs().mean(dim=1)[0].detach().cpu().numpy().astype(np.float32)
        poe_absmap = delta_poe_eps.abs().mean(dim=1)[0].detach().cpu().numpy().astype(np.float32)
        if delta_ref_l2 > peak_ref_value:
            peak_ref_value = delta_ref_l2
            peak_ref_step = i
            peak_ref_absmap = ref_absmap
        if delta_poe_l2 > peak_poe_value:
            peak_poe_value = delta_poe_l2
            peak_poe_step = i
            peak_poe_absmap = poe_absmap
        if onset_ref_step is None and delta_ref_l2 > 1e-8:
            onset_ref_step = i

        row = {
            "step": int(i),
            "timestep": int(t.item()),
            "step_frac": float(i) / float(max(1, num_inference_steps - 1)),
            "delta_ref_eps_l2": delta_ref_l2,
            "delta_ref_eps_mean_abs": delta_ref_mean_abs,
            "traj_mse_mono_poe": traj_mse,
            "traj_cosdist_mono_poe": traj_cosdist,
        }
        if interaction_gap_mode == "reference_and_poe":
            row["delta_poe_eps_l2"] = delta_poe_l2
            row["delta_poe_eps_mean_abs"] = delta_poe_mean_abs
        step_rows.append(row)

    tracker_mono.store_final(latents_mono)
    tracker_poe.store_final(latents_poe)

    return {
        "latents_mono": latents_mono,
        "latents_poe": latents_poe,
        "tracker_mono": tracker_mono,
        "tracker_poe": tracker_poe,
        "timeseries": step_rows,
        "delta_ref_l2_t": np.asarray(delta_ref_l2_t, dtype=np.float32),
        "delta_poe_l2_t": np.asarray(delta_poe_l2_t, dtype=np.float32),
        "traj_mse_mono_poe_t": np.asarray(traj_mse_mono_poe_t, dtype=np.float32),
        "traj_cosdist_mono_poe_t": np.asarray(traj_cosdist_mono_poe_t, dtype=np.float32),
        "summary": {
            "gamma_ref_uniform": float(np.mean(delta_ref_l2_t)) if delta_ref_l2_t else 0.0,
            "gamma_poe_uniform": float(np.mean(delta_poe_l2_t)) if delta_poe_l2_t else 0.0,
            "delta_ref_peak_value": float(peak_ref_value),
            "delta_ref_peak_step": int(peak_ref_step),
            "delta_ref_onset_step": int(onset_ref_step if onset_ref_step is not None else num_inference_steps - 1),
            "delta_poe_peak_value": float(peak_poe_value),
            "delta_poe_peak_step": int(peak_poe_step),
            "traj_mse_terminal": float(traj_mse_mono_poe_t[-1]) if traj_mse_mono_poe_t else 0.0,
            "traj_cosdist_mean": float(np.mean(traj_cosdist_mono_poe_t)) if traj_cosdist_mono_poe_t else 0.0,
        },
        "peak_ref_absmap": peak_ref_absmap,
        "peak_poe_absmap": peak_poe_absmap,
    }


def poe_sdxl_with_trajectory_tracking(
    latents,
    prompt_a,
    prompt_b,
    scheduler,
    unet,
    tokenizer,
    text_encoder,
    tokenizer_2,
    text_encoder_2,
    guidance_scale=7.5,
    num_inference_steps=50,
    batch_size=1,
    device=torch.device("cuda"),
    dtype=torch.float16,
    model_id=None,
    euler_init_noise_sigma=1.0,
    height=1024,
    width=1024,
):
    """
    PoE with trajectory tracking for SDXL.

    Implements faithful Product of Experts composition:
        noise_pred = uncond + guidance_scale * (cond_a - uncond)
                           + guidance_scale * (cond_b - uncond)

    SDXL specifics:
    - Uses dual text encoders (CLIP-L + CLIP-G) -> concatenated embeddings
    - Uses DDIMScheduler for deterministic stepping (eta=0)
    - Initial latents are scaled by Euler's init_noise_sigma, need to denormalize
    """

    # Build DDIMScheduler for deterministic stepping
    if model_id is not None:
        ddim = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    else:
        ddim = DDIMScheduler.from_config(scheduler.config)
    ddim.set_timesteps(num_inference_steps)

    # Denormalize Euler-scaled latents (x * sigma_max) back to N(0,I) for DDIM
    latents = (latents / euler_init_noise_sigma).to(dtype=dtype, device=device)

    # Encode prompts with SDXL dual text encoders
    uncond_emb, uncond_kwargs = _encode_sdxl([""] * batch_size, tokenizer, tokenizer_2,
                                             text_encoder, text_encoder_2, device, height, width)
    a_emb, a_kwargs = _encode_sdxl([prompt_a] * batch_size, tokenizer, tokenizer_2,
                                   text_encoder, text_encoder_2, device, height, width)
    b_emb, b_kwargs = _encode_sdxl([prompt_b] * batch_size, tokenizer, tokenizer_2,
                                   text_encoder, text_encoder_2, device, height, width)

    tracker = LatentTrajectoryCollector(
        num_inference_steps,
        batch_size,
        latents.shape[1],
        latents.shape[2],
        latents.shape[3],
    )

    extra_step_kwargs = {}
    if "eta" in _inspect.signature(ddim.step).parameters:
        extra_step_kwargs["eta"] = 0.0  # deterministic DDIM

    from tqdm.auto import tqdm as _tqdm

    for i, t in _tqdm(enumerate(ddim.timesteps), total=num_inference_steps,
                      desc=f"PoE SDXL B={batch_size}", leave=False):
        # scale_model_input: identity for DDIM but matches pipeline for correctness
        latent_model_input = ddim.scale_model_input(latents, t)

        # Three batched UNet calls — faithful PoE (Product of Experts):
        #   noise_pred = uncond + g*(cond_a - uncond) + g*(cond_b - uncond)
        with torch.no_grad():
            noise_uncond = unet(
                latent_model_input,
                t,
                encoder_hidden_states=uncond_emb,
                added_cond_kwargs=uncond_kwargs,
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
            noise_uncond
            + guidance_scale * (noise_a - noise_uncond)
            + guidance_scale * (noise_b - noise_uncond)
        )

        tracker.store_step(i, latents, noise_pred, float(i) / num_inference_steps, t.item())

        # Deterministic DDIM step
        latents = ddim.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    tracker.store_final(latents)
    return latents, tracker


def poe_sdxl_monolithic_with_trajectory_tracking(
    latents,
    monolithic_prompt,
    scheduler,
    unet,
    tokenizer,
    text_encoder,
    tokenizer_2,
    text_encoder_2,
    guidance_scale=7.5,
    num_inference_steps=50,
    batch_size=1,
    device=torch.device("cuda"),
    dtype=torch.float16,
    model_id=None,
    euler_init_noise_sigma=1.0,
    height=1024,
    width=1024,
):
    """
    Monolithic (single prompt AND) with trajectory tracking for SDXL.

    Baseline: condition on "prompt_a AND prompt_b" directly.
    """

    if model_id is not None:
        ddim = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    else:
        ddim = DDIMScheduler.from_config(scheduler.config)
    ddim.set_timesteps(num_inference_steps)

    latents = (latents / euler_init_noise_sigma).to(dtype=dtype, device=device)

    uncond_emb, uncond_kwargs = _encode_sdxl([""] * batch_size, tokenizer, tokenizer_2,
                                             text_encoder, text_encoder_2, device, height, width)
    mono_emb, mono_kwargs = _encode_sdxl([monolithic_prompt] * batch_size, tokenizer, tokenizer_2,
                                         text_encoder, text_encoder_2, device, height, width)

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
                      desc=f"Monolithic SDXL B={batch_size}", leave=False):
        latent_model_input = ddim.scale_model_input(latents, t)

        with torch.no_grad():
            noise_uncond = unet(
                latent_model_input,
                t,
                encoder_hidden_states=uncond_emb,
                added_cond_kwargs=uncond_kwargs,
                timestep_cond=None,
            ).sample
            noise_mono = unet(
                latent_model_input,
                t,
                encoder_hidden_states=mono_emb,
                added_cond_kwargs=mono_kwargs,
                timestep_cond=None,
            ).sample

        noise_pred = noise_uncond + guidance_scale * (noise_mono - noise_uncond)

        tracker.store_step(i, latents, noise_pred, float(i) / num_inference_steps, t.item())
        latents = ddim.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    tracker.store_final(latents)
    return latents, tracker


# ============================================================================
# Example usage / test functions
# ============================================================================

if __name__ == "__main__":
    import argparse
    from notebooks.dynamics import get_latents
    from diffusers import EulerDiscreteScheduler

    parser = argparse.ArgumentParser(
        description="Generate SDXL PoE trajectories (test)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="SDXL model ID"
    )
    parser.add_argument(
        "--prompt-a",
        type=str,
        default="a dog",
        help="First prompt"
    )
    parser.add_argument(
        "--prompt-b",
        type=str,
        default="a cat",
        help="Second prompt"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for test results"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading SDXL models from {args.model_id}...")
    models = get_sd_models(model_id=args.model_id, dtype=dtype, device=device)

    print("Creating initial noise...")
    euler = EulerDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    x_T = get_latents(
        euler,
        z_channels=4,
        device=device,
        dtype=dtype,
        num_inference_steps=args.num_inference_steps,
        batch_size=1,
        latent_width=128,
        latent_height=128,
        seed=args.seed,
    )
    euler_sigma = float(getattr(euler, "init_noise_sigma", 1.0))

    print(f"Running PoE with '{args.prompt_a}' AND '{args.prompt_b}'...")
    latents_poe, tracker_poe = poe_sdxl_with_trajectory_tracking(
        x_T.clone(),
        args.prompt_a,
        args.prompt_b,
        euler,
        models["unet"],
        models["tokenizer"],
        models["text_encoder"],
        models["tokenizer_2"],
        models["text_encoder_2"],
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        batch_size=1,
        device=device,
        dtype=dtype,
        model_id=args.model_id,
        euler_init_noise_sigma=euler_sigma,
    )

    print(f"Running Monolithic with '{args.prompt_a} and {args.prompt_b}'...")
    monolithic_prompt = f"{args.prompt_a} and {args.prompt_b}"
    latents_mono, tracker_mono = poe_sdxl_monolithic_with_trajectory_tracking(
        x_T.clone(),
        monolithic_prompt,
        euler,
        models["unet"],
        models["tokenizer"],
        models["text_encoder"],
        models["tokenizer_2"],
        models["text_encoder_2"],
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        batch_size=1,
        device=device,
        dtype=dtype,
        model_id=args.model_id,
        euler_init_noise_sigma=euler_sigma,
    )

    print("✓ PoE trajectory tracking complete")
    print(f"  PoE trajectory shape: {tracker_poe.trajectories.shape}")
    print(f"  Mono trajectory shape: {tracker_mono.trajectories.shape}")

    # Compute trajectory divergence
    poe_traj = tracker_poe.trajectories.float()
    mono_traj = tracker_mono.trajectories.float()
    d_t = ((poe_traj - mono_traj) ** 2).mean(dim=(1, 2, 3, 4))
    print(f"  Trajectory L2 distance (PoE vs Mono): {d_t[-1].item():.6f}")
