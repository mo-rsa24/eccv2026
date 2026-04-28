"""
Method 11 — Tweedie-PoE Corrector
==================================

Theoretical grounding
---------------------
Standard PoE computes the composed score as:

    s_PoE = s_1 + s_2 - s_∅

This misses the interaction term r_t:

    s_joint ≈ s_PoE + r_t

Via the Tweedie / MMSE identity (s_i = (x̂₀⁽ⁱ⁾ - x_t) / σ²), correcting
the score is equivalent to correcting the clean estimate x̂₀:

    x̂₀_corr = x̂₀_PoE + Δx₀
    s_corr   = (x̂₀_corr - x_t) / σ²

This method constructs Δx₀ entirely from *marginal* experts (s_1, s_2, s_∅)
— no joint prompt c_mul is used at inference time.  Three complementary
correction terms are combined with a timestep-aware schedule:

A) Spatial divergence push (α_t) — pushes x̂₀¹ and x̂₀² APART at spatially
   contested regions, preventing chimera collapse (Group 4):

       overlap_weight = (||x̂₀¹||_C · ||x̂₀²||_C) / max(same)  ∈ [0,1]
       Δx₀_diverge = +overlap_weight ⊙ (x̂₀¹ − x̂₀²)

   NOTE: sign is + (divergence), not − (which was the old "agreement" term
   that incorrectly pushed toward a chimera average).
   Active only when t_frac ≥ spatial_phase_gate (no signal at pure noise).

B) Conflict projection  (β_t)  — remove components of each x̂₀ estimate that
   oppose the other expert, preventing dominance (Group 4):

       x̂₀¹_proj = x̂₀¹ − proj_{x̂₀²}(x̂₀¹)   when dot(x̂₀¹, x̂₀²) < 0
       Δx₀_proj  = (x̂₀¹_proj + x̂₀²_proj) − x̂₀_PoE

C) Spatial repulsion  (γ_t)  — penalise overlap between per-concept x̂₀
   magnitudes in the spatial domain, encouraging slot separation (Group 4):

       mask_1 = softmax( |x̂₀¹|_channel )   ∈ [0,1]^{H×W}
       mask_2 = softmax( |x̂₀²|_channel )   ∈ [0,1]^{H×W}
       Δx₀_spatial = overlap(mask_1, mask_2) · (x̂₀¹ - x̂₀²)
   Active only when t_frac ≥ spatial_phase_gate.

Timestep schedule (empirically motivated by gap diagnostics):
  • Early (t_frac 0.2–0.4):  α large  — spatial divergence push during layout
  • Mid   (0.35–0.55):       γ large  — spatial repulsion / slot assignment
  • Late  (t_frac > 0.7):    β large  — fine-grained conflict removal

Final composed velocity (apply_outer_cfg=True, default):
    x̂₀_PoE  = x̂₀¹ + x̂₀² − x̂₀∅
    x̂₀_corr = x̂₀_PoE + α_t·Δx₀_diverge + β_t·Δx₀_proj + γ_t·Δx₀_spatial
    vel_poe_corr = (x_t − x̂₀_corr) / σ
    vf = vel_uncond + gs * (vel_poe_corr − vel_uncond)   [CFG guidance]

Note: apply_outer_cfg=False (legacy) skips the CFG wrapper.  This is INCORRECT
because vel_poe_corr is an unguided flow velocity — without CFG the output is
geometric noise.  The default True is the correct behaviour.

Note: For SD1/2 epsilon-prediction the velocity convention differs.
The scheduler_step in _base.py handles both.

References
----------
- Meng & Ermon 2021 "SDEdit: Image Synthesis and Editing..." (Tweedie connection)
- Yu et al. 2020 "Gradient Surgery for Multi-Task Learning" (PCGrad — method B)
- Liu et al. 2022 "Compositional Visual Generation with Composable Diffusion Models"
- The gap diagnostic data: cosine_joint_poe_vs_poe_residual ≈ -0.99 at t=981
  confirms PoE error is large and structured at early timesteps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ._base import (
    Vel,
    predict_x0,
    score_deltas,
    cosine_similarity_deltas,
    scheduler_step,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TweediePoeConfig:
    # Base guidance scale (applied to the final corrected velocity)
    guidance_scale: float = 7.5

    # ---- Correction strengths (peak values) ----
    # A) Spatial divergence push: pushes x̂₀¹ and x̂₀² APART at contested regions
    #    (replaces old "agreement-seeking" which had the wrong sign and drove chimeras)
    alpha_max: float = 0.35   # peak weight for divergence push
    # B) Conflict projection: removes opposing components of x̂₀ estimates
    beta_max: float = 0.3     # peak weight for projection correction
    # C) Spatial repulsion: encourages spatial slot separation
    gamma_max: float = 0.25   # peak weight for spatial correction

    # ---- Timestep schedule peaks (in t_frac ∈ [0,1], 0=noise, 1=clean) ----
    # Each correction is a Gaussian bump centred at the peak, with given width.
    alpha_peak: float = 0.20  # layout-formation window (shifted from 0.15)
    alpha_width: float = 0.20

    beta_peak: float = 0.80   # late: fine conflict removal
    beta_width: float = 0.20

    gamma_peak: float = 0.35  # mid-early: spatial layout (shifted from 0.45)
    gamma_width: float = 0.20

    # ---- Phase gate: spatial corrections (A and C) inactive below this t_frac ----
    # At pure noise (t_frac ≈ 0) x̂₀ is near-Gaussian and carries no spatial
    # information — applying corrections there is meaningless.
    spatial_phase_gate: float = 0.20

    # ---- Outer CFG wrapper ----
    # True (default): vf = vel_uncond + gs*(vel_poe_corr − vel_uncond)
    # False (legacy): vf = vel_poe_corr  directly (no guidance — produces noise)
    # NOTE: apply_outer_cfg=False is WRONG for epsilon/velocity prediction models.
    # vel_poe_corr = (x_t - x̂₀_corr)/σ is an unguided flow velocity; without
    # CFG wrapping there is no guidance signal and the output is geometric noise.
    apply_outer_cfg: bool = True

    # ---- Projection (B): only project when experts conflict ----
    proj_only_on_conflict: bool = True
    proj_eps: float = 1e-8

    # ---- Spatial (C): channel-norm softmax temperature ----
    spatial_temperature: float = 1.0   # lower = sharper spatial masks

    # ---- Safety: clip Δx₀ norm to at most this fraction of ||x̂₀_PoE|| ----
    delta_clip_frac: float = 0.5

    # ---- Vanilla PoE fallback ----
    # If True, when all correction strengths are 0 the method reduces to plain PoE
    poe_fallback: bool = True


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def _gaussian_bump(t_frac: float, peak: float, width: float) -> float:
    """Scalar Gaussian bump schedule evaluated at t_frac."""
    return math.exp(-0.5 * ((t_frac - peak) / (width + 1e-8)) ** 2)


def _correction_weights(t_frac: float, cfg: TweediePoeConfig) -> Tuple[float, float, float]:
    """Return (α_t, β_t, γ_t) at the current fractional timestep."""
    alpha = cfg.alpha_max * _gaussian_bump(t_frac, cfg.alpha_peak, cfg.alpha_width)
    beta  = cfg.beta_max  * _gaussian_bump(t_frac, cfg.beta_peak,  cfg.beta_width)
    gamma = cfg.gamma_max * _gaussian_bump(t_frac, cfg.gamma_peak, cfg.gamma_width)
    return alpha, beta, gamma


# ---------------------------------------------------------------------------
# Correction A — Spatial divergence push
# ---------------------------------------------------------------------------

def _delta_divergence(x0_1: Vel, x0_2: Vel, t_frac: float,
                      cfg: "TweediePoeConfig") -> Vel:
    """
    Δx₀_diverge = +overlap_weight ⊙ (x̂₀¹ - x̂₀²)

    Pushes the two Tweedie predictions APART at pixels where both concepts
    are simultaneously claiming spatial territory (high channel-norm overlap).

    Returns zeros when t_frac < cfg.spatial_phase_gate because at pure-noise
    steps x̂₀ is near-Gaussian and its channel norms carry no layout signal.

    Note: the old "agreement" correction was -(x̂₀¹ - x̂₀²), which drove both
    predictions toward their mean and produced chimeras.  The sign is now +.
    """
    if t_frac < cfg.spatial_phase_gate:
        return torch.zeros_like(x0_1)
    n1 = x0_1.float().norm(dim=1, keepdim=True)          # [B,1,H,W]
    n2 = x0_2.float().norm(dim=1, keepdim=True)          # [B,1,H,W]
    raw_overlap = n1 * n2
    max_overlap = (raw_overlap.flatten(1)
                              .max(dim=1).values
                              .view(-1, 1, 1, 1) + 1e-8)
    overlap_weight = raw_overlap / max_overlap            # [B,1,H,W] ∈ [0,1]
    return (overlap_weight * (x0_1 - x0_2)).to(x0_1.dtype)


# ---------------------------------------------------------------------------
# Correction B — Conflict projection (PCGrad in x̂₀-space)
# ---------------------------------------------------------------------------

def _delta_projection(
    x0_1: Vel,
    x0_2: Vel,
    x0_poe: Vel,
    cfg: TweediePoeConfig,
) -> Vel:
    """
    Remove from x̂₀¹ the component that opposes x̂₀², and vice versa.

    Δx₀_proj = (x̂₀¹_proj + x̂₀²_proj) - x̂₀_PoE
    """
    B, C, H, W = x0_1.shape
    x0_1f = x0_1.reshape(B, -1).float()
    x0_2f = x0_2.reshape(B, -1).float()

    dot   = (x0_1f * x0_2f).sum(dim=1, keepdim=True)          # [B, 1]
    n2_sq = (x0_2f ** 2).sum(dim=1, keepdim=True) + cfg.proj_eps
    n1_sq = (x0_1f ** 2).sum(dim=1, keepdim=True) + cfg.proj_eps

    if cfg.proj_only_on_conflict:
        conflict = (dot < 0).float()                            # [B, 1]
        x0_1_proj = x0_1f - conflict * (dot / n2_sq) * x0_2f
        x0_2_proj = x0_2f - conflict * (dot / n1_sq) * x0_1f
    else:
        x0_1_proj = x0_1f - (dot / n2_sq) * x0_2f
        x0_2_proj = x0_2f - (dot / n1_sq) * x0_1f

    x0_1_proj = x0_1_proj.reshape(B, C, H, W).to(x0_1.dtype)
    x0_2_proj = x0_2_proj.reshape(B, C, H, W).to(x0_2.dtype)

    x0_combined = x0_1_proj + x0_2_proj   # analogous to PoE without uncond
    return x0_combined - x0_poe


# ---------------------------------------------------------------------------
# Correction C — Spatial repulsion
# ---------------------------------------------------------------------------

def _spatial_activation_mask(x0: Vel, temperature: float) -> torch.Tensor:
    """
    Compute a per-spatial-position activation from the channel-norm of x̂₀.

    Returns mask ∈ [0,1]^{B×1×H×W} (softmax over spatial positions).
    """
    # Channel L2 norm at each spatial position: [B, H, W]
    norm_map = x0.float().norm(dim=1, keepdim=True)          # [B, 1, H, W]
    B, _, H, W = norm_map.shape
    flat = norm_map.reshape(B, H * W) / (temperature + 1e-8)
    soft = F.softmax(flat, dim=1).reshape(B, 1, H, W)
    return soft


def _delta_spatial(x0_1: Vel, x0_2: Vel, t_frac: float = 1.0,
                   phase_gate: float = 0.0) -> Vel:
    """
    Δx₀_spatial = overlap(mask_1, mask_2) · (x̂₀¹ - x̂₀²)

    Where overlap is a scalar ∈ [0,1] measuring how much the two spatial
    activation masks coincide (soft IoU proxy).  High overlap → large push
    to separate the two denoised images in space.

    Returns zeros when t_frac < phase_gate (no spatial signal at pure noise).
    """
    if t_frac < phase_gate:
        return torch.zeros_like(x0_1)
    mask_1 = _spatial_activation_mask(x0_1, temperature=1.0)   # [B, 1, H, W]
    mask_2 = _spatial_activation_mask(x0_2, temperature=1.0)

    # Soft overlap: dot product of normalised masks, per batch element → [B]
    overlap = (mask_1 * mask_2).flatten(1).sum(dim=1)           # [B], in [0,1]

    overlap_b = overlap[:, None, None, None]                    # broadcast
    return overlap_b * (x0_1 - x0_2)


# ---------------------------------------------------------------------------
# Main composition function
# ---------------------------------------------------------------------------

def compose_tweedie_poe(
    vel_c1:    Vel,
    vel_c2:    Vel,
    vel_uncond: Vel,
    latents:   Vel,
    sigma:     torch.Tensor,
    t_frac:    float,
    cfg: Optional[TweediePoeConfig] = None,
) -> Tuple[Vel, dict]:
    """
    Compute Tweedie-corrected PoE velocity.

    Parameters
    ----------
    vel_c1, vel_c2, vel_uncond : [B, C, H, W] velocity / epsilon fields
    latents : [B, C, H, W] current noisy latents x_t
    sigma   : scalar or [B] noise level σ_t
    t_frac  : float ∈ [0,1], fraction of denoising completed (0=noise, 1=clean)
    cfg     : TweediePoeConfig

    Returns
    -------
    vf   : [B, C, H, W] corrected velocity
    info : dict with diagnostics
    """
    if cfg is None:
        cfg = TweediePoeConfig()

    # ---- Tweedie x̂₀ estimates ----
    x0_1   = predict_x0(latents, vel_c1,    sigma)
    x0_2   = predict_x0(latents, vel_c2,    sigma)
    x0_unc = predict_x0(latents, vel_uncond, sigma)

    # ---- PoE in x̂₀-space ----
    x0_poe = x0_1 + x0_2 - x0_unc

    # ---- Correction weights at this timestep ----
    alpha_t, beta_t, gamma_t = _correction_weights(t_frac, cfg)

    # ---- Build Δx₀ ----
    dx0_div     = _delta_divergence(x0_1, x0_2, t_frac, cfg)        # A
    dx0_proj    = _delta_projection(x0_1, x0_2, x0_poe, cfg)        # B
    dx0_spatial = _delta_spatial(x0_1, x0_2, t_frac,               # C
                                  cfg.spatial_phase_gate)

    delta_x0 = alpha_t * dx0_div + beta_t * dx0_proj + gamma_t * dx0_spatial

    # ---- Safety clip: ||Δx₀|| ≤ clip_frac * ||x̂₀_PoE|| ----
    if cfg.delta_clip_frac > 0:
        poe_norm   = x0_poe.flatten(1).float().norm(dim=1, keepdim=True)   # [B,1]
        delta_norm = delta_x0.flatten(1).float().norm(dim=1, keepdim=True) # [B,1]
        max_norm   = cfg.delta_clip_frac * poe_norm + 1e-8
        scale      = (max_norm / delta_norm.clamp(min=1e-8)).clamp(max=1.0)
        delta_x0   = delta_x0 * scale.reshape(delta_x0.shape[0], 1, 1, 1).to(delta_x0.dtype)

    # ---- Corrected x̂₀ ----
    x0_corr = x0_poe + delta_x0

    # ---- Convert corrected x̂₀ back to velocity  (v = (x_t - x̂₀) / σ) ----
    # This is the inverse of predict_x0: v = (x_t - x̂₀) / σ
    sigma_b = sigma if sigma.ndim == 0 else sigma.reshape(sigma.shape[0], 1, 1, 1)
    vel_poe_corr = (latents - x0_corr) / (sigma_b + 1e-8)

    # ---- Compose final velocity ----
    # apply_outer_cfg=True (default): wrap with CFG guidance — required.
    # apply_outer_cfg=False (legacy): skips CFG; produces unguided noise — wrong.
    if cfg.apply_outer_cfg:
        vf = vel_uncond + cfg.guidance_scale * (vel_poe_corr - vel_uncond)
    else:
        vf = vel_poe_corr

    # ---- Diagnostics ----
    x0_disagree_norm = (x0_1 - x0_2).flatten(1).float().norm(dim=1).mean().item()
    delta_norm_val   = delta_x0.flatten(1).float().norm(dim=1).mean().item()
    poe_norm_val     = x0_poe.flatten(1).float().norm(dim=1).mean().item()

    # Cosine similarity between Δx₀ and the PoE residual (x̂₀_PoE - avg_marginal)
    x0_avg   = (x0_1 + x0_2) / 2.0
    poe_res  = x0_poe - x0_avg
    cos_delta_poe_res = F.cosine_similarity(
        delta_x0.flatten(1).float(),
        poe_res.flatten(1).float(),
        dim=1,
    ).mean().item()

    # Cosine similarity between score deltas Δ₁ and Δ₂ — distinguishes
    # constructive interference (positive, e.g. cat+dog both claim centre)
    # from destructive interference (negative, concepts pull in opposite dirs).
    d1 = vel_c1 - vel_uncond
    d2 = vel_c2 - vel_uncond
    cos_d1_d2 = cosine_similarity_deltas(d1, d2).mean().item()

    info = {
        "alpha_t":            torch.tensor(alpha_t),
        "beta_t":             torch.tensor(beta_t),
        "gamma_t":            torch.tensor(gamma_t),
        "x0_disagree_norm":   torch.tensor(x0_disagree_norm),
        "delta_norm":         torch.tensor(delta_norm_val),
        "poe_norm":           torch.tensor(poe_norm_val),
        "cos_delta_poe_res":  torch.tensor(cos_delta_poe_res),
        "cos_d1_d2":          torch.tensor(cos_d1_d2),
    }
    return vf, info


# ---------------------------------------------------------------------------
# Full denoising loop
# ---------------------------------------------------------------------------

def run_tweedie_poe_corrector(
    latents:           Vel,
    vel_fn,            # callable(latents, t, sigma, emb, kwargs) -> Vel
    embeddings_c1,
    embeddings_c2,
    embeddings_uncond,
    scheduler,
    cfg: Optional[TweediePoeConfig] = None,
    cond_kwargs_c1:    Optional[dict] = None,
    cond_kwargs_c2:    Optional[dict] = None,
    cond_kwargs_uncond: Optional[dict] = None,
    device=None,
) -> Tuple[Vel, list]:
    """
    Full denoising loop using the Tweedie-PoE corrector.

    vel_fn signature:
        vel_fn(latents, t, sigma, embeddings, added_cond_kwargs) -> Vel

    Returns (final_latents, list_of_per_step_info_dicts).
    """
    if cfg is None:
        cfg = TweediePoeConfig()

    N = len(scheduler.timesteps)
    infos = []

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma  = scheduler.sigmas[i]
        t_frac = i / max(N - 1, 1)    # 0 = pure noise, 1 = clean

        vel_c1    = vel_fn(latents, t, sigma, embeddings_c1,    cond_kwargs_c1)
        vel_c2    = vel_fn(latents, t, sigma, embeddings_c2,    cond_kwargs_c2)
        vel_uncond = vel_fn(latents, t, sigma, embeddings_uncond, cond_kwargs_uncond)

        vf, info = compose_tweedie_poe(
            vel_c1, vel_c2, vel_uncond,
            latents, sigma, t_frac, cfg,
        )
        infos.append(info)

        latents = scheduler_step(scheduler, vf, t, latents, sigma, dsigma)

    return latents, infos
