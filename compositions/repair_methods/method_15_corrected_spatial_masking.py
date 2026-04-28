"""
Method 15 — Corrected Spatial Masking (Tweedie-Space Disentanglement)
======================================================================

Motivation
----------
Standard PoE misses the interaction term:

    s_joint ≈ s_PoE + ∇_{x_t} log PMI(c1, c2 ; x_t)

The interaction term is zero when supp(c1) ∩ supp(c2) = ∅, i.e. when the two
concepts occupy disjoint spatial regions.  If we can enforce spatial disjointness
in the Tweedie (clean-image) prediction space, PoE becomes exact.

This method implements a corrected version of the spatial masking idea, fixing
four concrete errors in the naïve version:

Fix 1 — Correct masking criterion
-----------------------------------
WRONG (naïve):   M1[j] = 1 if ||x̂₀^(1,j) - x_t^(j)|| > ||x̂₀^(2,j) - x_t^(j)||
  Problem: x̂₀ is in clean space, x_t has noise variance O(σ²) → signal buried.

CORRECT (this):  M1[j] = 1 if ||x̂₀^(1,j) - x̄₀^(j)||² > ||x̂₀^(2,j) - x̄₀^(j)||²
  where x̄₀ = (x̂₀^(1) + x̂₀^(2)) / 2   (Tweedie-space mean)

This compares how far each concept's Tweedie prediction deviates from their
shared mean at each spatial position — operating entirely in clean image space,
and symmetric between the two concepts.  High deviation = strong spatial claim.

Soft version (default): instead of a hard threshold, use a normalised ratio:
    r[j] = ||x̂₀^(1,j) - x̄₀^(j)||² / (||x̂₀^(1,j) - x̄₀^(j)||² + ||x̂₀^(2,j) - x̄₀^(j)||² + ε)
    M1 = r,  M2 = 1 - r   (soft, smooth masks; sum to 1 per pixel)

A temperature parameter τ sharpens toward hard masks as denoising progresses.

Fix 2 — Timestep gating
------------------------
Masks are only applied when t_frac ≥ phase_gate (default 0.20).  At pure-noise
steps the Tweedie prediction is near-Gaussian and carries no spatial signal.
Before the gate, fall back to plain PoE.

Additionally, the masking strength is linearly ramped from 0 (at phase_gate) to
1 (at phase_peak, default 0.50) and then held constant.  This prevents abrupt
transitions that corrupt the denoising trajectory.

Fix 3 — Background region handling
-------------------------------------
WRONG (naïve): background pixels → x̂₀^(∅) (unconditional Tweedie mean)
  Problem: at high noise x̂₀^(∅) is the dataset mean — structurally incoherent.

CORRECT (this): background is handled by softmax blending, not a third map.
  M1 + M2 = 1 per pixel (by construction).  There is no separate background.
  The unconditional Tweedie prediction is used only in the CFG wrapper:
      vf = vel_unc + gs * (vel_comp - vel_unc)
  where vel_comp is derived from the masked composite x̂₀_comp.

Fix 4 — Concept leakage mitigation
-------------------------------------
The marginal score ε_θ(x_t, c1, t) may already encode c2 if both concepts
co-occurred in training data.  Without retraining, the practical mitigation is
a gradient conflict filter applied after masking: remove from vel_c1 (in the
M1-assigned region) any component that points in the direction of vel_c2 - vel_unc.
This is a per-pixel PCGrad step in velocity space, applied only at mask boundaries.

Algorithm (single step)
-----------------------
1.  Compute x̂₀^(1), x̂₀^(2) from Tweedie formula
2.  Compute x̄₀ = (x̂₀^(1) + x̂₀^(2)) / 2
3.  Compute soft masks M1, M2 (Fix 1)
4.  Blend:  x̂₀_comp = M1 ⊙ x̂₀^(1) + M2 ⊙ x̂₀^(2)
5.  Apply leakage filter (Fix 4) — per-pixel conflict projection at boundary
6.  Convert to velocity and apply CFG (Fix 3)
7.  Apply timestep gate (Fix 2): blend with PoE if t_frac < phase_peak

Diagnostics tracked per step
-----------------------------
  mask_sharpness  : mean(max(M1, M2)) ∈ [0.5, 1.0]  — 0.5=soft, 1.0=binary
  spatial_iou     : soft IoU of M1, M2 (lower = better separation)
  x0_disagree     : ||x̂₀^(1) - x̂₀^(2)|| (concept separation)
  cos_d1_d2       : cos(Δ1, Δ2) (constructive vs destructive interference)
  blend_weight    : how much masking vs PoE fallback is applied
  leakage_removed : norm of conflict component removed by Fix 4
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
class CorrectedSpatialMaskingConfig:
    guidance_scale: float = 7.5

    # Fix 2 — timestep gating
    phase_gate:  float = 0.20   # masks inactive below this t_frac
    phase_peak:  float = 0.50   # full mask strength from here onward

    # Fix 1 — mask sharpening temperature
    # τ_start at phase_gate → τ_end at t_frac=1.0
    # Lower τ = sharper (more binary) mask assignment
    tau_start: float = 2.0   # soft at layout formation
    tau_end:   float = 0.3   # sharp at fine-detail stage

    # Fix 4 — leakage filter
    leakage_filter: bool = True
    leakage_strength: float = 0.5  # 0=off, 1=fully projected

    # Safety clip on mask-composite correction vs PoE
    delta_clip_frac: float = 0.6

    norm_eps: float = 1e-8


# ---------------------------------------------------------------------------
# Fix 1: Correct soft masks
# ---------------------------------------------------------------------------

def _tweedie_masks(
    x0_1: Vel,
    x0_2: Vel,
    tau: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute soft spatial assignment masks M1, M2 in Tweedie-space.

    For each spatial position j:
        x̄₀[j] = (x̂₀^(1,j) + x̂₀^(2,j)) / 2   (concept mean)
        d1[j]  = ||x̂₀^(1,j) - x̄₀[j]||²
        d2[j]  = ||x̂₀^(2,j) - x̄₀[j]||²

    Soft assignment via temperature-scaled softmax:
        logit1 = d1 / τ
        logit2 = d2 / τ
        [M1, M2] = softmax([logit1, logit2], dim=1)

    M1 + M2 = 1 per pixel.  High d1 relative to d2 → M1 close to 1 → concept 1 owns pixel.

    Parameters
    ----------
    x0_1, x0_2 : [B, C, H, W]
    tau : float > 0  (lower = sharper/more binary)

    Returns
    -------
    M1, M2 : [B, 1, H, W]  soft masks, sum to 1 per pixel
    """
    x0_mean = (x0_1 + x0_2) / 2.0   # [B,C,H,W]

    # Per-pixel squared distance in channel space (sum over channel dim)
    d1 = (x0_1 - x0_mean).float().pow(2).sum(dim=1, keepdim=True)   # [B,1,H,W]
    d2 = (x0_2 - x0_mean).float().pow(2).sum(dim=1, keepdim=True)

    stack = torch.cat([d1, d2], dim=1)              # [B,2,H,W]
    w = F.softmax(stack / (tau + eps), dim=1)        # softmax over concept axis
    M1 = w[:, 0:1]
    M2 = w[:, 1:2]
    return M1, M2


# ---------------------------------------------------------------------------
# Fix 4: Leakage filter (per-pixel PCGrad in velocity space)
# ---------------------------------------------------------------------------

def _leakage_filter(
    vel_c1: Vel,
    vel_c2: Vel,
    vel_uncond: Vel,
    M1: torch.Tensor,
    M2: torch.Tensor,
    strength: float,
    eps: float = 1e-8,
) -> Tuple[Vel, Vel, torch.Tensor]:
    """
    Remove from vel_c1 (in M1-assigned regions) the component aligned with
    the c2 score delta, and vice versa.

    This mitigates score leakage: if ε_θ(x_t, c1) encodes c2 due to training
    co-occurrence, the component (vel_c1 - vel_unc) that aligns with (vel_c2 - vel_unc)
    is a leakage artifact.  We subtract a fraction `strength` of it.

    Only operates at mask boundaries (where neither M1 nor M2 is dominant).
    """
    d1 = (vel_c1 - vel_uncond).float()   # [B,C,H,W]
    d2 = (vel_c2 - vel_uncond).float()

    # Per-pixel dot product (over channel dimension)
    dot = (d1 * d2).sum(dim=1, keepdim=True)     # [B,1,H,W]
    n2_sq = (d2 * d2).sum(dim=1, keepdim=True) + eps
    n1_sq = (d1 * d1).sum(dim=1, keepdim=True) + eps

    # Project out only where there is conflict (dot < 0 means opposing — skip)
    # For leakage we care about POSITIVE dot (both pointing same way → blending)
    leakage_mask = (dot > 0).float()  # [B,1,H,W]

    # At mask boundaries: neither concept dominates — 0.5 < M1 < 1, similarly M2
    # Boundary weight: soft mask entropy proxy → high when M1 ≈ M2 ≈ 0.5
    boundary = 1.0 - (M1 - M2).abs()   # [B,1,H,W], in [0,1]; 1 at boundary, 0 at pure assignments

    proj1 = leakage_mask * boundary * (dot / n2_sq) * d2   # component of d1 along d2
    proj2 = leakage_mask * boundary * (dot / n1_sq) * d1   # component of d2 along d1

    d1_filtered = d1 - strength * proj1
    d2_filtered = d2 - strength * proj2

    leakage_removed = (strength * proj1).flatten(1).float().norm(dim=1).mean()

    vel_c1_out = (vel_uncond + d1_filtered).to(vel_c1.dtype)
    vel_c2_out = (vel_uncond + d2_filtered).to(vel_c2.dtype)
    return vel_c1_out, vel_c2_out, leakage_removed


# ---------------------------------------------------------------------------
# Temperature schedule
# ---------------------------------------------------------------------------

def _mask_tau(t_frac: float, cfg: CorrectedSpatialMaskingConfig) -> float:
    """Linear interpolation of mask temperature from tau_start to tau_end."""
    progress = (t_frac - cfg.phase_gate) / max(1.0 - cfg.phase_gate, 1e-8)
    progress = max(0.0, min(1.0, progress))
    return cfg.tau_start + progress * (cfg.tau_end - cfg.tau_start)


def _blend_weight(t_frac: float, cfg: CorrectedSpatialMaskingConfig) -> float:
    """
    How much to use masked composite vs PoE fallback.
    0 = pure PoE, 1 = pure masked composite.
    Linearly ramps from 0 (at phase_gate) to 1 (at phase_peak).
    """
    if t_frac < cfg.phase_gate:
        return 0.0
    if t_frac >= cfg.phase_peak:
        return 1.0
    return (t_frac - cfg.phase_gate) / (cfg.phase_peak - cfg.phase_gate)


# ---------------------------------------------------------------------------
# Main composition function (single step)
# ---------------------------------------------------------------------------

def compose_corrected_spatial_masking(
    vel_c1:     Vel,
    vel_c2:     Vel,
    vel_uncond: Vel,
    latents:    Vel,
    sigma:      torch.Tensor,
    t_frac:     float,
    cfg: Optional[CorrectedSpatialMaskingConfig] = None,
) -> Tuple[Vel, dict]:
    """
    Single-step Corrected Spatial Masking composition.

    Parameters
    ----------
    vel_c1, vel_c2, vel_uncond : [B, C, H, W]
    latents  : [B, C, H, W]
    sigma    : scalar noise level
    t_frac   : float in [0,1]
    cfg      : CorrectedSpatialMaskingConfig

    Returns
    -------
    vf   : [B, C, H, W] composed velocity
    info : dict of per-step diagnostics
    """
    if cfg is None:
        cfg = CorrectedSpatialMaskingConfig()

    # ---- Tweedie x̂₀ estimates ----
    x0_1   = predict_x0(latents, vel_c1,    sigma)
    x0_2   = predict_x0(latents, vel_c2,    sigma)
    x0_unc = predict_x0(latents, vel_uncond, sigma)

    # ---- PoE x̂₀ (fallback) ----
    x0_poe = x0_1 + x0_2 - x0_unc

    # ---- Diagnostics (always computed) ----
    d1, d2    = score_deltas(vel_c1, vel_c2, vel_uncond)
    cos_d1_d2 = cosine_similarity_deltas(d1, d2)
    x0_dis    = x0_disagreement(x0_1, x0_2)

    bw = _blend_weight(t_frac, cfg)

    if bw == 0.0:
        # Pure PoE fallback (Fix 2 — before phase gate)
        sigma_b  = sigma if sigma.ndim == 0 else sigma.reshape(-1, 1, 1, 1)
        vel_poe  = (latents - x0_poe) / (sigma_b + cfg.norm_eps)
        vf = vel_uncond + cfg.guidance_scale * (vel_poe - vel_uncond)

        info = {
            "mask_sharpness":  torch.tensor(0.5),
            "spatial_iou":     torch.tensor(1.0),
            "x0_disagree":     x0_dis.detach(),
            "cos_d1_d2":       cos_d1_d2.detach(),
            "blend_weight":    torch.tensor(0.0),
            "leakage_removed": torch.tensor(0.0),
        }
        return vf, info

    # ---- Fix 4: Apply leakage filter before masking ----
    leakage_removed = torch.tensor(0.0)
    if cfg.leakage_filter:
        vel_c1, vel_c2, leakage_removed = _leakage_filter(
            vel_c1, vel_c2, vel_uncond,
            torch.ones_like(latents[:, :1]),   # placeholder — computed after mask
            torch.ones_like(latents[:, :1]),
            cfg.leakage_strength, cfg.norm_eps,
        )
        # Recompute x̂₀ with filtered velocities
        x0_1 = predict_x0(latents, vel_c1, sigma)
        x0_2 = predict_x0(latents, vel_c2, sigma)

    # ---- Fix 1: Correct soft masks (Tweedie-space) ----
    tau = _mask_tau(t_frac, cfg)
    M1, M2 = _tweedie_masks(x0_1, x0_2, tau, cfg.norm_eps)

    # ---- Now apply leakage filter with actual masks ----
    if cfg.leakage_filter:
        vel_c1_f, vel_c2_f, leakage_removed = _leakage_filter(
            vel_c1, vel_c2, vel_uncond,
            M1, M2,
            cfg.leakage_strength, cfg.norm_eps,
        )
        x0_1 = predict_x0(latents, vel_c1_f, sigma)
        x0_2 = predict_x0(latents, vel_c2_f, sigma)

    # ---- Fix 3: Mask-weighted composite (no x̂₀_unc background region) ----
    x0_comp = M1 * x0_1 + M2 * x0_2   # M1 + M2 = 1 per pixel

    # ---- Blend masked composite with PoE fallback ----
    x0_blended = bw * x0_comp + (1.0 - bw) * x0_poe

    # ---- Safety clip ----
    if cfg.delta_clip_frac > 0:
        delta      = x0_blended - x0_poe
        poe_norm   = x0_poe.flatten(1).float().norm(dim=1, keepdim=True)
        delta_norm = delta.flatten(1).float().norm(dim=1, keepdim=True)
        max_norm   = cfg.delta_clip_frac * poe_norm + cfg.norm_eps
        scale      = (max_norm / delta_norm.clamp(min=cfg.norm_eps)).clamp(max=1.0)
        scale_b    = scale.reshape(delta.shape[0], 1, 1, 1).to(delta.dtype)
        x0_blended = x0_poe + delta * scale_b

    # ---- Convert to velocity and apply CFG (Fix 3) ----
    sigma_b  = sigma if sigma.ndim == 0 else sigma.reshape(-1, 1, 1, 1)
    vel_comp = (latents - x0_blended) / (sigma_b + cfg.norm_eps)
    vf = vel_uncond + cfg.guidance_scale * (vel_comp - vel_uncond)

    # ---- Diagnostics ----
    mask_sharpness = M1.squeeze(1).clamp(0, 1).amax(dim=(-2, -1)).mean()
    spatial_iou = attention_overlap_iou(M1.squeeze(1), M2.squeeze(1))

    info = {
        "mask_sharpness":  mask_sharpness.detach(),
        "spatial_iou":     spatial_iou.detach(),
        "x0_disagree":     x0_dis.detach(),
        "cos_d1_d2":       cos_d1_d2.detach(),
        "blend_weight":    torch.tensor(float(bw)),
        "leakage_removed": leakage_removed.detach() if isinstance(leakage_removed, torch.Tensor) else torch.tensor(float(leakage_removed)),
        "tau":             torch.tensor(float(tau)),
    }
    return vf, info


# ---------------------------------------------------------------------------
# Full denoising loop
# ---------------------------------------------------------------------------

def run_corrected_spatial_masking(
    latents:             Vel,
    vel_fn,
    embeddings_c1,
    embeddings_c2,
    embeddings_uncond,
    scheduler,
    cfg: Optional[CorrectedSpatialMaskingConfig] = None,
    cond_kwargs_c1:      Optional[dict] = None,
    cond_kwargs_c2:      Optional[dict] = None,
    cond_kwargs_uncond:  Optional[dict] = None,
    device=None,
) -> Tuple[Vel, list]:
    """
    Full denoising loop using Corrected Spatial Masking (Method 15).

    Returns (final_latents, list_of_per_step_info_dicts).
    """
    if cfg is None:
        cfg = CorrectedSpatialMaskingConfig()

    N = len(scheduler.timesteps)
    infos = []

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma  = scheduler.sigmas[i]
        t_frac = i / max(N - 1, 1)

        vel_c1     = vel_fn(latents, t, sigma, embeddings_c1,     cond_kwargs_c1)
        vel_c2     = vel_fn(latents, t, sigma, embeddings_c2,     cond_kwargs_c2)
        vel_uncond = vel_fn(latents, t, sigma, embeddings_uncond, cond_kwargs_uncond)

        vf, info = compose_corrected_spatial_masking(
            vel_c1, vel_c2, vel_uncond,
            latents, sigma, t_frac, cfg,
        )
        infos.append(info)

        latents = scheduler_step(scheduler, vf, t, latents, sigma, dsigma)

    return latents, infos
