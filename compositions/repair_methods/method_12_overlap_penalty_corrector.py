"""
Method 12 — Localized Overlap Penalty Corrector
===============================================

This method steers the latent away from *contested support* rather than
forcing the two marginal experts to agree globally. It uses a Tweedie-space
occupancy energy plus a preservation term:

    q_A(i) = sigmoid(gamma * ||mu_A(i) - mu_unc(i)||)
    q_B(i) = sigmoid(gamma * ||mu_B(i) - mu_unc(i)||)

    E_overlap = mean_i q_A(i) q_B(i)
    E_pres    = relu(tau - mass(q_A)) + relu(tau - mass(q_B))

The latent is corrected directly:

    x_t <- x_t - eta * grad_x [ lambda_overlap(t) * E_overlap
                              + lambda_pres      * E_pres ]

The denoising step then proceeds from the corrected latent using a vanilla
PoE / SuperDiff-style composition:

    vf = v_unc + gs * [(v_A - v_unc) + (v_B - v_unc)]

This is intentionally a regularizer, not a claim of exact PMI recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ._base import Vel, predict_x0, scheduler_step  # predict_x0 used for x0_disagree diagnostic


@dataclass
class OverlapPenaltyConfig:
    guidance_scale: float = 7.5
    gamma: float = 6.0
    # Penalty strength: increased from 0.45/0.10 — previous values yielded
    # correction_grad_norm ≈ 0.027 × 0.05 ≈ 0.001, negligible vs latent norms O(1-10)
    lambda_overlap_start: float = 2.0
    lambda_overlap_end: float = 0.50
    lambda_pres: float = 0.20
    preservation_tau: float = 0.12
    # Step size: increased from 0.05 — previous effective latent correction was ~1e-4
    step_size: float = 0.30
    # Activation window: FIXED — was 0.70→0.25 which started correction AFTER layout
    # was committed.  Now 0.05→0.60 covers the critical early layout-formation phase.
    apply_from_frac: float = 0.05
    apply_to_frac: float = 0.60
    max_grad_norm: float = 1.0
    use_tweedie_space: bool = True
    occupancy_downsample: int = 2
    occupancy_bias_quantile: float = 0.70
    occupancy_scale_floor: float = 1e-4
    contention_threshold: float = 0.55


def _poe_velocity(
    vel_c1: Vel,
    vel_c2: Vel,
    vel_uncond: Vel,
    guidance_scale: float,
) -> Vel:
    d1 = vel_c1 - vel_uncond
    d2 = vel_c2 - vel_uncond
    return vel_uncond + guidance_scale * (d1 + d2)


def _occupancy_source(
    latents: Vel,
    sigma: torch.Tensor,
    vel: Vel,
    vel_uncond: Vel,
    cfg: OverlapPenaltyConfig,
) -> Vel:
    if cfg.use_tweedie_space:
        x0 = predict_x0(latents, vel, sigma)
        x0_unc = predict_x0(latents, vel_uncond, sigma)
        return x0 - x0_unc
    return vel - vel_uncond


def _norm_map(source: Vel, cfg: OverlapPenaltyConfig) -> torch.Tensor:
    norm_map = source.float().norm(dim=1, keepdim=True)
    if cfg.occupancy_downsample and cfg.occupancy_downsample > 1:
        k = int(cfg.occupancy_downsample)
        norm_map = F.avg_pool2d(norm_map, kernel_size=k, stride=k, ceil_mode=False)
    return norm_map


def _occupancy(
    latents: Vel,
    sigma: torch.Tensor,
    vel: Vel,
    vel_uncond: Vel,
    cfg: OverlapPenaltyConfig,
) -> torch.Tensor:
    source = _occupancy_source(latents, sigma, vel, vel_uncond, cfg)
    norm_map = _norm_map(source, cfg)
    flat = norm_map.flatten(1)
    bias = torch.quantile(flat, cfg.occupancy_bias_quantile, dim=1, keepdim=True)
    scale = (flat.std(dim=1, keepdim=True) + cfg.occupancy_scale_floor)
    logits = (flat - bias) / scale
    logits = logits.reshape_as(norm_map)
    return torch.sigmoid(cfg.gamma * logits)


def _compute_energies(
    latents: Vel,
    sigma: torch.Tensor,
    vel_c1: Vel,
    vel_c2: Vel,
    vel_uncond: Vel,
    cfg: OverlapPenaltyConfig,
) -> Tuple[torch.Tensor, dict]:
    q_a = _occupancy(latents, sigma, vel_c1, vel_uncond, cfg)
    q_b = _occupancy(latents, sigma, vel_c2, vel_uncond, cfg)
    overlap_map = q_a * q_b

    overlap_energy = overlap_map.flatten(1).mean(dim=1)
    mass_a = q_a.flatten(1).mean(dim=1)
    mass_b = q_b.flatten(1).mean(dim=1)
    preservation = F.relu(cfg.preservation_tau - mass_a) + F.relu(cfg.preservation_tau - mass_b)
    contention_fraction = (overlap_map > cfg.contention_threshold).float().flatten(1).mean(dim=1)

    # Directional diagnostics: cosine similarity and dominance ratio of score deltas.
    # cos_delta1_delta2 > 0 → constructive interference (concepts point same way, e.g.
    # cat+dog both want the centre → chimera). < 0 → destructive interference.
    d1 = vel_c1 - vel_uncond
    d2 = vel_c2 - vel_uncond
    d1_flat = d1.flatten(1).float()
    d2_flat = d2.flatten(1).float()
    cos_d1_d2  = F.cosine_similarity(d1_flat, d2_flat, dim=1)         # [B]
    norm_ratio  = d1_flat.norm(dim=1) / (d2_flat.norm(dim=1) + 1e-8)  # [B]

    diagnostics = {
        "overlap_energy":       overlap_energy,
        "preservation_penalty": preservation,
        "occupancy_mass_a":     mass_a,
        "occupancy_mass_b":     mass_b,
        "contention_fraction":  contention_fraction,
        "cos_delta1_delta2":    cos_d1_d2,
        "dominance_ratio":      norm_ratio,
    }
    return overlap_energy, diagnostics


def _scheduled_overlap_weight(t_frac: float, cfg: OverlapPenaltyConfig) -> float:
    lo = min(cfg.apply_from_frac, cfg.apply_to_frac)
    hi = max(cfg.apply_from_frac, cfg.apply_to_frac)
    if t_frac < lo or t_frac > hi:
        return 0.0
    progress = 0.0 if hi == lo else (t_frac - lo) / (hi - lo)
    return cfg.lambda_overlap_start + progress * (cfg.lambda_overlap_end - cfg.lambda_overlap_start)


def _energy_grad(
    latents: Vel,
    t,
    sigma: torch.Tensor,
    vel_fn,
    embeddings_c1,
    embeddings_c2,
    embeddings_uncond,
    cond_kwargs_c1,
    cond_kwargs_c2,
    cond_kwargs_uncond,
    cfg: OverlapPenaltyConfig,
    lambda_overlap_t: float,
) -> Tuple[Vel, dict]:
    if lambda_overlap_t == 0.0 and cfg.lambda_pres == 0.0:
        zero = torch.zeros_like(latents)
        return zero, {
            "overlap_energy": torch.zeros(latents.shape[0], device=latents.device),
            "preservation_penalty": torch.zeros(latents.shape[0], device=latents.device),
            "occupancy_mass_a": torch.zeros(latents.shape[0], device=latents.device),
            "occupancy_mass_b": torch.zeros(latents.shape[0], device=latents.device),
            "contention_fraction": torch.zeros(latents.shape[0], device=latents.device),
        }

    x = latents.detach().requires_grad_(True)
    vel_c1 = vel_fn(x, t, sigma, embeddings_c1, cond_kwargs_c1)
    vel_c2 = vel_fn(x, t, sigma, embeddings_c2, cond_kwargs_c2)
    vel_uncond = vel_fn(x, t, sigma, embeddings_uncond, cond_kwargs_uncond)

    overlap_energy, diag = _compute_energies(x, sigma, vel_c1, vel_c2, vel_uncond, cfg)
    total = lambda_overlap_t * overlap_energy + cfg.lambda_pres * diag["preservation_penalty"]
    grad = torch.autograd.grad(total.mean(), x)[0].detach()
    return grad, {k: v.detach() for k, v in diag.items()}


def run_overlap_penalty_corrector(
    latents: Vel,
    vel_fn,
    embeddings_c1,
    embeddings_c2,
    embeddings_uncond,
    scheduler,
    cfg: Optional[OverlapPenaltyConfig] = None,
    cond_kwargs_c1: Optional[dict] = None,
    cond_kwargs_c2: Optional[dict] = None,
    cond_kwargs_uncond: Optional[dict] = None,
    device=None,
) -> Tuple[Vel, list]:
    if cfg is None:
        cfg = OverlapPenaltyConfig()

    infos = []
    n_steps = len(scheduler.timesteps)

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma = scheduler.sigmas[i]
        t_frac = i / max(n_steps - 1, 1)

        lambda_overlap_t = _scheduled_overlap_weight(t_frac, cfg)
        grad, diag = _energy_grad(
            latents=latents,
            t=t,
            sigma=sigma,
            vel_fn=vel_fn,
            embeddings_c1=embeddings_c1,
            embeddings_c2=embeddings_c2,
            embeddings_uncond=embeddings_uncond,
            cond_kwargs_c1=cond_kwargs_c1,
            cond_kwargs_c2=cond_kwargs_c2,
            cond_kwargs_uncond=cond_kwargs_uncond,
            cfg=cfg,
            lambda_overlap_t=lambda_overlap_t,
        )

        grad_norm = grad.flatten(1).float().norm(dim=1)
        if cfg.max_grad_norm > 0:
            scale = (cfg.max_grad_norm / grad_norm.clamp(min=1e-8)).clamp(max=1.0)
            grad = grad * scale[:, None, None, None].to(grad.dtype)

        latents_corr = latents - cfg.step_size * grad.to(latents.dtype)

        vel_c1 = vel_fn(latents_corr, t, sigma, embeddings_c1, cond_kwargs_c1)
        vel_c2 = vel_fn(latents_corr, t, sigma, embeddings_c2, cond_kwargs_c2)
        vel_uncond = vel_fn(latents_corr, t, sigma, embeddings_uncond, cond_kwargs_uncond)
        vf = _poe_velocity(vel_c1, vel_c2, vel_uncond, cfg.guidance_scale)

        # x0_disagree_norm: ||x̂₀¹ - x̂₀²|| at the corrected latent.
        # Should INCREASE over trajectory if correction is working (concepts separating).
        x0_1_corr = predict_x0(latents_corr, vel_c1, sigma)
        x0_2_corr = predict_x0(latents_corr, vel_c2, sigma)
        x0_disagree = (x0_1_corr - x0_2_corr).flatten(1).float().norm(dim=1).detach()

        infos.append(
            {
                "t_frac":               torch.tensor(t_frac),
                "lambda_overlap_t":     torch.tensor(lambda_overlap_t),
                "lambda_pres":          torch.tensor(cfg.lambda_pres),
                "overlap_energy":       diag["overlap_energy"],
                "preservation_penalty": diag["preservation_penalty"],
                "occupancy_mass_a":     diag["occupancy_mass_a"],
                "occupancy_mass_b":     diag["occupancy_mass_b"],
                "contention_fraction":  diag["contention_fraction"],
                "cos_delta1_delta2":    diag["cos_delta1_delta2"].detach(),
                "dominance_ratio":      diag["dominance_ratio"].detach(),
                "x0_disagree_norm":     x0_disagree,
                "correction_grad_norm": grad.flatten(1).float().norm(dim=1).detach(),
            }
        )

        latents = scheduler_step(scheduler, vf, t, latents_corr, sigma, dsigma)

    return latents, infos
