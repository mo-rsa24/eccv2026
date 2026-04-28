"""
Method 13 — Phase-Aware Spatial Routing
=========================================

Motivation
----------
The missing interaction term in PoE,

    Δ(x_t) = ∇_{x_t} log PMI(c1, c2 ; x_t)
            = ∇_{x_t} log [ p(c1,c2|x_t) / (p(c1|x_t)·p(c2|x_t)) ],

is intractable without a model of the joint distribution p(x_t|c1,c2).
Rather than approximating Δ(x_t), this method enforces the condition under
which PoE is *exact*: spatial disjointness of the two concepts.

When supp(c1) ∩ supp(c2) = ∅ in image space, the joint factorises:
    p(x|c1,c2) ∝ p(x|c1) · p(x|c2)   [PoE is exact]

We enforce this by routing each spatial position's Tweedie prediction to
whichever concept claims it more strongly, using a temperature-scheduled
competitive softmax over the channel-norm maps of x̂₀¹ and x̂₀².

Algorithm (three phases)
-------------------------
Phase 1  t_frac ∈ [0,        phase1_end]:  Soft router, T = temp_start
Phase 2  t_frac ∈ [phase1_end, phase2_end]: Router with T decaying linearly
Phase 3  t_frac ∈ [phase2_end, 1.0]:        Standard PoE + optional PCGrad

The spatial router
------------------
Given Tweedie predictions x̂₀¹ and x̂₀² (both [B,C,H,W]):

    n1[b,h,w] = ||x̂₀¹[b,:,h,w]||₂   (channel L2 norm)
    n2[b,h,w] = ||x̂₀²[b,:,h,w]||₂

    stack = cat([n1, n2], dim=1)       [B,2,H,W]
    w     = softmax(stack/T, dim=1)    [B,2,H,W], softmax over concept axis
    w1,w2 = w[:,0:1], w[:,1:2]         w1+w2=1 per pixel (guaranteed)

    x̂₀_comp = w1⊙x̂₀¹ + w2⊙x̂₀²       convex combination, no background subtraction

    vel_comp = (x_t - x̂₀_comp) / σ
    vf       = vel_uncond + gs * (vel_comp - vel_uncond)

Key design choices
------------------
1.  Softmax over dim=1 (concept axis), NOT over spatial dims.  This gives
    per-pixel winner-take-all between concepts (what we want).  Softmax over
    spatial dims would instead be an attention map (a different operation).

2.  No background subtraction.  The PoE formula x̂₀_PoE = x̂₀¹+x̂₀²-x̂₀∅
    subtracts the unconditional to remove baseline content.  Here, the
    weights already enforce a unit-sum constraint (w1+w2=1), so subtracting
    x̂₀∅ would double-subtract background content.  The CFG wrapper handles
    the unconditional baseline.

3.  Temperature schedule.  At T=1 (high noise, t_frac≈0) both n1,n2 are
    near-equal Gaussian → w1=w2=0.5, soft blending — correct because no
    spatial commitment should be made yet.  As T→0 the softmax hardens into
    a binary assignment, committing spatial territory.

4.  Phase 3 fallback.  Once masks are near-binary, standard PoE is
    equivalent to hard masking by the evolved velocity field.  PCGrad
    projection removes any residual cross-concept interference in Δ₁,Δ₂.

Diagnostics tracked per step
-----------------------------
  phase         : 1, 2, or 3
  temperature   : T at this step
  spatial_iou   : soft IoU of w1,w2 (lower is better — less overlap)
  cos_d1_d2     : cosine similarity of Δ₁,Δ₂ (constructive vs destructive)
  x0_disagree   : ||x̂₀¹ - x̂₀²|| (concept separation; should grow early)
  w1_mean       : mean of w1 over spatial dims (≈0.5 early, diverges later)
  w2_mean       : mean of w2 over spatial dims
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ._base import (
    Vel,
    predict_x0,
    score_deltas,
    cosine_similarity_deltas,
    x0_disagreement,
    attention_overlap_iou,
    scheduler_step,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SpatialRoutingConfig:
    guidance_scale: float = 7.5

    # Phase boundaries (t_frac ∈ [0,1], 0=pure noise, 1=clean)
    phase1_end: float = 0.25   # soft routing → hardening
    phase2_end: float = 0.70   # hardening   → plain PoE + PCGrad

    # Softmax temperature schedule over [phase1_end, phase2_end]
    temp_start: float = 1.0    # high T = soft (exploratory)
    temp_end:   float = 0.2    # low  T = hard (committed)

    # Phase 3: PCGrad conflict projection on score deltas
    pcgrad_phase3:    bool  = True
    pcgrad_strength:  float = 0.3   # 0 = unchanged, 1 = fully projected

    norm_eps: float = 1e-6


# ---------------------------------------------------------------------------
# Temperature schedule
# ---------------------------------------------------------------------------

def _interpolate_temperature(t_frac: float, cfg: SpatialRoutingConfig) -> float:
    """
    Linear interpolation of softmax temperature.
      t_frac ≤ phase1_end   → temp_start  (constant soft phase)
      t_frac ≥ phase2_end   → temp_end    (constant hard; phase 3 uses PoE)
      otherwise             → linear decay
    """
    if t_frac <= cfg.phase1_end:
        return cfg.temp_start
    if t_frac >= cfg.phase2_end:
        return cfg.temp_end
    progress = (t_frac - cfg.phase1_end) / (cfg.phase2_end - cfg.phase1_end)
    return cfg.temp_start + progress * (cfg.temp_end - cfg.temp_start)


# ---------------------------------------------------------------------------
# Spatial router
# ---------------------------------------------------------------------------

def _spatial_router(
    x0_1: Vel,
    x0_2: Vel,
    temperature: float,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute soft spatial assignment weights via competitive softmax.

    For each spatial position (h, w), the softmax decides which concept
    (c1 or c2) claims that pixel more strongly, based on the channel L2
    norm of the respective Tweedie prediction.

    Parameters
    ----------
    x0_1, x0_2 : [B, C, H, W]  Tweedie predictions
    temperature : float > 0     lower = sharper / more binary assignment

    Returns
    -------
    w1, w2 : [B, 1, H, W]  soft assignment weights, w1 + w2 = 1 per pixel
    """
    n1 = x0_1.float().norm(dim=1, keepdim=True)   # [B,1,H,W]
    n2 = x0_2.float().norm(dim=1, keepdim=True)   # [B,1,H,W]

    stack = torch.cat([n1, n2], dim=1)             # [B,2,H,W]
    # Softmax over dim=1 (concept axis) — per-pixel winner-take-all between c1 and c2
    w = F.softmax(stack / (temperature + eps), dim=1)   # [B,2,H,W]
    w1 = w[:, 0:1, :, :]    # [B,1,H,W]
    w2 = w[:, 1:2, :, :]    # [B,1,H,W]
    return w1, w2


# ---------------------------------------------------------------------------
# PCGrad projection for Phase 3
# ---------------------------------------------------------------------------

def _pcgrad_projection(
    d1: Vel,
    d2: Vel,
    strength: float,
    eps: float = 1e-8,
) -> Tuple[Vel, Vel]:
    """
    PCGrad: remove from each score delta the component opposing the other,
    blended by `strength` ∈ [0, 1].  Projection only applied when dot < 0.

    Returns d1_out, d2_out with reduced cross-concept interference.
    """
    B = d1.shape[0]
    d1f = d1.reshape(B, -1).float()
    d2f = d2.reshape(B, -1).float()

    dot   = (d1f * d2f).sum(dim=1, keepdim=True)           # [B,1]
    n2_sq = (d2f * d2f).sum(dim=1, keepdim=True) + eps
    n1_sq = (d1f * d1f).sum(dim=1, keepdim=True) + eps

    conflict = (dot < 0).float()                            # [B,1] — 1 if opposing

    d1_proj = d1f - conflict * (dot / n2_sq) * d2f
    d2_proj = d2f - conflict * (dot / n1_sq) * d1f

    # Blend: strength=0 → unchanged, strength=1 → fully projected
    d1_out = ((1.0 - strength) * d1f + strength * d1_proj).reshape_as(d1).to(d1.dtype)
    d2_out = ((1.0 - strength) * d2f + strength * d2_proj).reshape_as(d2).to(d2.dtype)
    return d1_out, d2_out


# ---------------------------------------------------------------------------
# Main composition function (single step)
# ---------------------------------------------------------------------------

def compose_spatial_routing(
    vel_c1:    Vel,
    vel_c2:    Vel,
    vel_uncond: Vel,
    latents:   Vel,
    sigma:     torch.Tensor,
    t_frac:    float,
    cfg: Optional[SpatialRoutingConfig] = None,
) -> Tuple[Vel, dict]:
    """
    Single-step Phase-Aware Spatial Routing composition.

    Parameters
    ----------
    vel_c1, vel_c2, vel_uncond : [B, C, H, W] velocity / epsilon fields
    latents  : [B, C, H, W] current noisy latents x_t
    sigma    : scalar or [B] noise level σ_t
    t_frac   : float ∈ [0,1], fraction of denoising completed (0=noise, 1=clean)
    cfg      : SpatialRoutingConfig

    Returns
    -------
    vf   : [B, C, H, W] composed velocity
    info : dict of per-step diagnostics
    """
    if cfg is None:
        cfg = SpatialRoutingConfig()

    # ---- Tweedie predictions ----
    x0_1 = predict_x0(latents, vel_c1, sigma)      # [B,C,H,W]
    x0_2 = predict_x0(latents, vel_c2, sigma)

    # ---- Score deltas (used for phase 3 and diagnostics) ----
    d1, d2 = score_deltas(vel_c1, vel_c2, vel_uncond)
    cos_d1_d2 = cosine_similarity_deltas(d1, d2)   # [B]
    x0_dis    = x0_disagreement(x0_1, x0_2)        # [B]

    # Safe sigma broadcast shape
    sigma_b = sigma if sigma.ndim == 0 else sigma.reshape(-1, 1, 1, 1)

    if t_frac < cfg.phase2_end:
        # ---- Phase 1 or 2: spatial routing ----
        T    = _interpolate_temperature(t_frac, cfg)
        w1, w2 = _spatial_router(x0_1, x0_2, T, cfg.norm_eps)

        # Convex combination — no background subtraction (see module docstring)
        x0_comp  = w1 * x0_1 + w2 * x0_2          # [B,C,H,W]

        vel_comp = (latents - x0_comp) / (sigma_b + cfg.norm_eps)
        vf = vel_uncond + cfg.guidance_scale * (vel_comp - vel_uncond)

        # Soft spatial IoU: measures how much the two weight maps overlap.
        # Lower is better (concepts claiming different territories).
        # Computed on the soft weights (hard weights are always complementary → IoU≡0).
        spatial_iou = attention_overlap_iou(
            w1.squeeze(1), w2.squeeze(1)
        )  # [B]

        phase = 1 if t_frac < cfg.phase1_end else 2

        info = {
            "phase":        torch.tensor(float(phase)),
            "temperature":  torch.tensor(T),
            "spatial_iou":  spatial_iou.detach(),
            "cos_d1_d2":    cos_d1_d2.detach(),
            "x0_disagree":  x0_dis.detach(),
            "w1_mean":      w1.mean().detach(),
            "w2_mean":      w2.mean().detach(),
        }

    else:
        # ---- Phase 3: standard PoE + optional PCGrad ----
        if cfg.pcgrad_phase3:
            d1, d2 = _pcgrad_projection(d1, d2, cfg.pcgrad_strength)

        vf = vel_uncond + cfg.guidance_scale * (d1 + d2)

        info = {
            "phase":        torch.tensor(3.0),
            "temperature":  torch.tensor(cfg.temp_end),
            "spatial_iou":  torch.zeros(latents.shape[0], device=latents.device),
            "cos_d1_d2":    cos_d1_d2.detach(),
            "x0_disagree":  x0_dis.detach(),
            "w1_mean":      torch.tensor(0.5),
            "w2_mean":      torch.tensor(0.5),
        }

    return vf, info


# ---------------------------------------------------------------------------
# Full denoising loop
# ---------------------------------------------------------------------------

def run_spatial_routing(
    latents:             Vel,
    vel_fn,              # callable(latents, t, sigma, embeddings, cond_kwargs) -> Vel
    embeddings_c1,
    embeddings_c2,
    embeddings_uncond,
    scheduler,
    cfg: Optional[SpatialRoutingConfig] = None,
    cond_kwargs_c1:      Optional[dict] = None,
    cond_kwargs_c2:      Optional[dict] = None,
    cond_kwargs_uncond:  Optional[dict] = None,
    device=None,
) -> Tuple[Vel, list]:
    """
    Full denoising loop using Phase-Aware Spatial Routing (Method 13).

    Interface is identical to all other run_* functions in this package.
    Returns (final_latents, list_of_per_step_info_dicts).
    """
    if cfg is None:
        cfg = SpatialRoutingConfig()

    N = len(scheduler.timesteps)
    infos = []

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma  = scheduler.sigmas[i]
        t_frac = i / max(N - 1, 1)   # 0 = pure noise, 1 = clean

        vel_c1     = vel_fn(latents, t, sigma, embeddings_c1,     cond_kwargs_c1)
        vel_c2     = vel_fn(latents, t, sigma, embeddings_c2,     cond_kwargs_c2)
        vel_uncond = vel_fn(latents, t, sigma, embeddings_uncond, cond_kwargs_uncond)

        vf, info = compose_spatial_routing(
            vel_c1, vel_c2, vel_uncond,
            latents, sigma, t_frac, cfg,
        )
        infos.append(info)

        latents = scheduler_step(scheduler, vf, t, latents, sigma, dsigma)

    return latents, infos
