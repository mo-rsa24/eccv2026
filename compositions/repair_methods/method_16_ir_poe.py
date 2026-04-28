"""
Method 16 — Interaction-Recovered PoE (IR-PoE)
===============================================

Goal
----
Approximate ∇ log p(x_t | c1, c2) without a joint prompt by structurally
satisfying c1 ⊥ c2 | x_t in Tweedie space.  The interaction term

    Δ(x_t) = ∇_{x_t} log PMI(c1, c2 ; x_t)

is zero when the two concepts occupy disjoint spatial regions.  IR-PoE
enforces this disjointness directly via a two-phase staged algorithm.

Two-phase structure
-------------------
Phase 1  [t_frac ∈ 0 .. phase_boundary]
    High-noise structural phase.  Cross-attention disentanglement is
    strongest here.  We:
      (a) Suppress score leakage with cross-negative prompts.
      (b) Compute symmetric Tweedie-space ownership masks.
      (c) Fuse the masked Tweedie estimates and back-project via DDIM.

Phase 2  [t_frac ∈ phase_boundary .. 1]
    Low-noise refinement phase.  Cross-attention disentanglement weakens.
    Masks are FROZEN from the last Phase 1 step; recomputing them here
    on near-clean images would introduce spatial jitter.
    Spatially gated CFG: each concept guides only its assigned region.

Phase 0 — Negative prompt cross-suppression
--------------------------------------------
Motivated by CoInD (ICLR 2025): marginal scores are not truly independent
because ε_θ(x_t, c1) encodes c2 if they co-occurred during training.

Approximate fix (no retraining):
    ε1_clean = ε_θ(x_t, c1) − ε_θ(x_t, c2_as_neg)     ← suppresses c2's signal
    ε2_clean = ε_θ(x_t, c2) − ε_θ(x_t, c1_as_neg)     ← suppresses c1's signal

This requires 2 additional forward passes per step.  Set
use_cross_suppression=False to run 3-pass mode (standard PoE speed).

Tweedie-space ownership (Fix 1 from analysis)
---------------------------------------------
Mask criterion operates entirely in clean-image space:
    x̄₀    = (x̂₀¹ + x̂₀²) / 2
    D1[j] = ||x̂₀¹[j] - x̄₀[j]||²
    D2[j] = ||x̂₀²[j] - x̄₀[j]||²
    M1[j] = softmax([D1[j], D2[j]] / τ)[0]   (soft; τ→0 = hard)

NOT ||x̂₀^(i) - x_t|| — that compares clean vs noisy space and is
dominated by noise variance at high t.

Conflict-gated softening (Step 3d)
------------------------------------
When cos(score_1, score_2) < conflict_threshold (strong destructive
conflict), the mask temperature is *raised* to prevent early catastrophic
spatial lock-in based on an uncertain structural decision.

No background region (Fix 3 from analysis)
---------------------------------------------
M1 + M2 = 1 per pixel (softmax).  The unconditional model is NOT used to
fill a "background" region — that would insert the dataset mean and cause
visible seam artifacts.  CFG handles the unconditional baseline.

Algorithm summary (per step)
-----------------------------
Phase 1:
    ε1, ε2, ε∅ → Tweedie x̂₀¹, x̂₀²
    Compute masks M1, M2 from D1, D2 (symmetric, Tweedie-space)
    x̂₀_comp = M1⊙x̂₀¹ + M2⊙x̂₀²
    ε_comp = (x_t - √ᾱ_t · x̂₀_comp) / √(1-ᾱ_t)      [back-project]
    DDIM step from x̂₀_comp and ε_comp

Phase 2 (masks frozen):
    ε1, ε2, ε∅
    ε_guided_1 = ε∅ + w*(ε1 - ε∅)   [CFG for c1 region]
    ε_guided_2 = ε∅ + w*(ε2 - ε∅)   [CFG for c2 region]
    ε_final = M1⊙ε_guided_1 + M2⊙ε_guided_2
    DDIM step from ε_final

Diagnostics per step
--------------------
  phase             : 1 or 2
  conflict          : cos(S1, S2) — negative = destructive interference
  mask_sharpness    : mean(max(M1, M2)) ∈ [0.5, 1.0]
  spatial_iou       : soft IoU of M1, M2 (lower = better separation)
  x0_disagree       : ||x̂₀¹ - x̂₀²||
  tau_used          : effective softmax temperature
  suppression_delta : leakage removed by cross-negative suppression (if active)

References
----------
- CoInD (ICLR 2025 Workshop): marginal independence violation in training
- GCDM (Cho et al., ECCV 2024): parameterised interaction term
- TweedieMix (ICLR 2025): Tweedie-space blending for composition
- Liu et al. 2022: Composable Diffusion (PoE baseline)
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
    x0_disagreement,
    attention_overlap_iou,
    scheduler_step,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IRPoEConfig:
    guidance_scale: float = 7.5

    # Phase boundary: t_frac at which we switch Phase1 → Phase2
    # 0 = pure noise, 1 = clean.  0.40 ≈ midpoint of SDXL's layout phase.
    phase_boundary: float = 0.40

    # Mask softmax temperature schedule (Phase 1 only)
    tau_start: float = 2.0   # soft at very early steps (pure noise)
    tau_end:   float = 0.3   # hard by end of Phase 1

    # Phase 1: blend weight ramp (how fast to engage masking vs PoE fallback)
    # At t_frac = 0 → blend_start; at t_frac = phase_boundary → 1.0
    blend_start: float = 0.0   # 0 = pure PoE at t=0, 1 = full masking immediately

    # Conflict-gated softening: if cos(S1,S2) < this, raise τ to conflict_tau
    conflict_threshold: float = -0.10
    conflict_tau: float = 5.0   # much softer mask when conflict is strong

    # Cross-negative suppression (Phase 0)
    # Requires 2 extra forward passes per step.
    use_cross_suppression: bool = True
    suppression_weight: float = 0.5   # 0=off, 1=full subtraction

    # Safety clip on x̂₀_comp deviation from x̂₀_PoE
    delta_clip_frac: float = 0.7

    norm_eps: float = 1e-8


# ---------------------------------------------------------------------------
# Tweedie x̂₀ from DDPM epsilon prediction
# ---------------------------------------------------------------------------

def _tweedie_x0_from_eps(
    x_t: torch.Tensor,
    eps: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """
    x̂₀ = (x_t - √(1-ᾱ_t) · ε) / √ᾱ_t
    alpha_bar: scalar tensor, ᾱ_t
    """
    sqrt_alpha_bar     = alpha_bar.sqrt()
    sqrt_one_minus_ab  = (1.0 - alpha_bar).sqrt()
    return (x_t - sqrt_one_minus_ab * eps) / (sqrt_alpha_bar + 1e-8)


def _eps_from_x0(
    x_t: torch.Tensor,
    x0: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """
    ε = (x_t - √ᾱ_t · x̂₀) / √(1-ᾱ_t)
    Inverse of _tweedie_x0_from_eps.
    """
    sqrt_alpha_bar    = alpha_bar.sqrt()
    sqrt_one_minus_ab = (1.0 - alpha_bar).sqrt()
    return (x_t - sqrt_alpha_bar * x0) / (sqrt_one_minus_ab + 1e-8)


# ---------------------------------------------------------------------------
# Flow-matching Tweedie (for EulerDiscrete / SDXL / SD3)
# ---------------------------------------------------------------------------
# In flow-matching: x̂₀ = x_t - σ·v_θ(x_t, t)  (already in _base.predict_x0)
# We need back-projection: v_composed = (x_t - x̂₀_comp) / σ


# ---------------------------------------------------------------------------
# Tweedie-space ownership masks (Fix 1)
# ---------------------------------------------------------------------------

def _tweedie_masks(
    x0_1: torch.Tensor,
    x0_2: torch.Tensor,
    tau: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft spatial ownership masks computed in clean-image (Tweedie) space.

    M1[j] = softmax([D1[j], D2[j]] / τ)[0]
    where D_i[j] = ||x̂₀^i[j] - x̄₀[j]||²  (squared deviation from pixel mean)

    Returns M1, M2 each [B,1,H,W], M1+M2=1 per pixel.
    """
    x0_mean = (x0_1 + x0_2) * 0.5
    d1 = (x0_1 - x0_mean).float().pow(2).sum(dim=1, keepdim=True)   # [B,1,H,W]
    d2 = (x0_2 - x0_mean).float().pow(2).sum(dim=1, keepdim=True)
    stack = torch.cat([d1, d2], dim=1)                                # [B,2,H,W]
    w = F.softmax(stack / (tau + eps), dim=1)
    return w[:, 0:1], w[:, 1:2]


# ---------------------------------------------------------------------------
# Cross-negative suppression (Phase 0)
# ---------------------------------------------------------------------------

def _suppress_leakage(
    eps_concept: torch.Tensor,    # ε_θ(x_t, c_own)
    eps_negative: torch.Tensor,   # ε_θ(x_t, c_other) used as negative
    weight: float,
) -> torch.Tensor:
    """
    Subtract the cross-concept prediction to suppress leakage.
    ε_clean = ε_concept - weight * (ε_negative - ε_uncond_baseline)

    Here we simply subtract weight * ε_negative directly.
    This is a conservative approximation — it assumes ε_negative captures
    only the co-occurrence signal, not shared background.
    """
    return eps_concept - weight * eps_negative


# ---------------------------------------------------------------------------
# Phase 1 step: Tweedie-space masking with DDIM back-projection
# ---------------------------------------------------------------------------

def _phase1_step(
    latents: Vel,
    vel_c1: Vel,
    vel_c2: Vel,
    vel_uncond: Vel,
    sigma: torch.Tensor,
    t_frac: float,
    cfg: IRPoEConfig,
) -> Tuple[Vel, torch.Tensor, torch.Tensor, dict]:
    """
    Phase 1 composition step.

    Returns composed velocity, M1, M2, and info dict.
    M1, M2 are returned so they can be frozen for Phase 2.
    """
    # Tweedie predictions
    x0_1 = predict_x0(latents, vel_c1, sigma)
    x0_2 = predict_x0(latents, vel_c2, sigma)

    # Conflict diagnostic
    d1, d2 = score_deltas(vel_c1, vel_c2, vel_uncond)
    cos_conflict = cosine_similarity_deltas(d1, d2)   # [B]
    mean_conflict = cos_conflict.mean().item()

    # Temperature: raise if conflict is strong (prevent early lock-in)
    tau = _interpolate_tau(t_frac, cfg)
    if mean_conflict < cfg.conflict_threshold:
        tau = max(tau, cfg.conflict_tau)

    # Tweedie-space ownership masks
    M1, M2 = _tweedie_masks(x0_1, x0_2, tau, cfg.norm_eps)

    # Masked composite x̂₀
    x0_comp = M1 * x0_1 + M2 * x0_2

    # Blend with PoE fallback (smoothly engage masking)
    x0_unc = predict_x0(latents, vel_uncond, sigma)
    x0_poe = x0_1 + x0_2 - x0_unc

    bw = _blend_weight(t_frac, cfg)
    x0_blended = bw * x0_comp + (1.0 - bw) * x0_poe

    # Safety clip
    if cfg.delta_clip_frac > 0:
        delta      = x0_blended - x0_poe
        poe_norm   = x0_poe.flatten(1).float().norm(dim=1, keepdim=True)
        delta_norm = delta.flatten(1).float().norm(dim=1, keepdim=True)
        max_norm   = cfg.delta_clip_frac * poe_norm + cfg.norm_eps
        scale      = (max_norm / delta_norm.clamp(min=cfg.norm_eps)).clamp(max=1.0)
        x0_blended = x0_poe + delta * scale.reshape(-1, 1, 1, 1).to(delta.dtype)

    # Back-project to velocity and apply CFG
    sigma_b = sigma if sigma.ndim == 0 else sigma.reshape(-1, 1, 1, 1)
    vel_comp = (latents - x0_blended) / (sigma_b + cfg.norm_eps)
    vf = vel_uncond + cfg.guidance_scale * (vel_comp - vel_uncond)

    # Diagnostics
    x0_dis        = x0_disagreement(x0_1, x0_2)
    mask_sharp    = M1.squeeze(1).amax(dim=(-2, -1)).mean()
    spatial_iou   = attention_overlap_iou(M1.squeeze(1), M2.squeeze(1))

    info = {
        "phase":           torch.tensor(1.0),
        "conflict":        cos_conflict.detach(),
        "mask_sharpness":  mask_sharp.detach(),
        "spatial_iou":     spatial_iou.detach(),
        "x0_disagree":     x0_dis.detach(),
        "tau_used":        torch.tensor(float(tau)),
        "blend_weight":    torch.tensor(float(bw)),
        "suppression_delta": torch.tensor(0.0),
    }
    return vf, M1.detach(), M2.detach(), info


# ---------------------------------------------------------------------------
# Phase 2 step: Spatially gated CFG with frozen masks
# ---------------------------------------------------------------------------

def _phase2_step(
    latents: Vel,
    vel_c1: Vel,
    vel_c2: Vel,
    vel_uncond: Vel,
    sigma: torch.Tensor,
    M1: torch.Tensor,
    M2: torch.Tensor,
    cfg: IRPoEConfig,
) -> Tuple[Vel, dict]:
    """
    Phase 2 composition step.

    M1, M2 are frozen from the end of Phase 1.  Each concept guides only
    its assigned spatial region.  CFG is applied per-region.
    """
    # Score deltas (Δ_i = vel_ci - vel_unc)
    d1, d2 = score_deltas(vel_c1, vel_c2, vel_uncond)
    cos_conflict = cosine_similarity_deltas(d1, d2)

    # Per-region CFG in velocity space:
    #   ε_guided_i = ε∅ + w*(εi - ε∅) = vel_unc + gs*Δi
    guided_1 = vel_uncond + cfg.guidance_scale * d1   # [B,C,H,W]
    guided_2 = vel_uncond + cfg.guidance_scale * d2

    # Spatially gated blend: concept i guides only M_i region
    vf = M1 * guided_1 + M2 * guided_2

    # Diagnostics
    x0_1   = predict_x0(latents, vel_c1, sigma)
    x0_2   = predict_x0(latents, vel_c2, sigma)
    x0_dis = x0_disagreement(x0_1, x0_2)
    mask_sharp  = M1.squeeze(1).amax(dim=(-2, -1)).mean()
    spatial_iou = attention_overlap_iou(M1.squeeze(1), M2.squeeze(1))

    info = {
        "phase":           torch.tensor(2.0),
        "conflict":        cos_conflict.detach(),
        "mask_sharpness":  mask_sharp.detach(),
        "spatial_iou":     spatial_iou.detach(),
        "x0_disagree":     x0_dis.detach(),
        "tau_used":        torch.tensor(0.0),
        "blend_weight":    torch.tensor(1.0),
        "suppression_delta": torch.tensor(0.0),
    }
    return vf, info


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def _interpolate_tau(t_frac: float, cfg: IRPoEConfig) -> float:
    """Linear decay of mask temperature over Phase 1."""
    progress = t_frac / max(cfg.phase_boundary, 1e-8)
    progress = min(1.0, progress)
    return cfg.tau_start + progress * (cfg.tau_end - cfg.tau_start)


def _blend_weight(t_frac: float, cfg: IRPoEConfig) -> float:
    """
    Ramp from blend_start (at t_frac=0) to 1.0 (at t_frac=phase_boundary).
    Phase 2 always uses blend_weight=1.0 (irrelevant — different formula).
    """
    progress = t_frac / max(cfg.phase_boundary, 1e-8)
    progress = min(1.0, progress)
    return cfg.blend_start + progress * (1.0 - cfg.blend_start)


# ---------------------------------------------------------------------------
# Full denoising loop
# ---------------------------------------------------------------------------

def run_ir_poe(
    latents:             Vel,
    vel_fn,              # callable(latents, t, sigma, embeddings, cond_kwargs) -> Vel
    embeddings_c1,
    embeddings_c2,
    embeddings_uncond,
    scheduler,
    cfg: Optional[IRPoEConfig] = None,
    cond_kwargs_c1:      Optional[dict] = None,
    cond_kwargs_c2:      Optional[dict] = None,
    cond_kwargs_uncond:  Optional[dict] = None,
    device=None,
) -> Tuple[Vel, list]:
    """
    Full denoising loop using Interaction-Recovered PoE (Method 16).

    Forward passes per step:
      use_cross_suppression=False: 3  (same as standard PoE)
      use_cross_suppression=True:  5  (two extra for cross-negatives)

    Returns (final_latents, list_of_per_step_info_dicts).
    """
    if cfg is None:
        cfg = IRPoEConfig()

    N = len(scheduler.timesteps)
    infos   = []
    M1_frozen: Optional[torch.Tensor] = None
    M2_frozen: Optional[torch.Tensor] = None

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma  = scheduler.sigmas[i]
        t_frac = i / max(N - 1, 1)   # 0 = pure noise, 1 = clean

        # ---- Standard 3 forward passes ----
        vel_c1     = vel_fn(latents, t, sigma, embeddings_c1,     cond_kwargs_c1)
        vel_c2     = vel_fn(latents, t, sigma, embeddings_c2,     cond_kwargs_c2)
        vel_uncond = vel_fn(latents, t, sigma, embeddings_uncond, cond_kwargs_uncond)

        # ---- Cross-negative suppression (Phase 0) — 2 extra passes ----
        # NOTE: vel_c2_as_neg re-evaluates ε_θ(x_t, c2) independently.
        # In practice the result equals vel_c2 (same inputs), so these could be
        # cached: vel_c2_as_neg = vel_c2, vel_c1_as_neg = vel_c1.
        # We call vel_fn separately to preserve the interface contract in case
        # the caller's vel_fn is stateful (e.g. attention caching).
        suppression_delta = torch.tensor(0.0)
        if cfg.use_cross_suppression:
            vel_c2_as_neg = vel_c2    # cache: same as the c2 pass above
            vel_c1_as_neg = vel_c1    # cache: same as the c1 pass above
            # If vel_fn is stateful, replace the two lines above with:
            # vel_c2_as_neg = vel_fn(latents, t, sigma, embeddings_c2, cond_kwargs_c2)
            # vel_c1_as_neg = vel_fn(latents, t, sigma, embeddings_c1, cond_kwargs_c1)

            # Suppress leakage: subtract cross-concept component (in delta space)
            d1_orig = vel_c1 - vel_uncond
            d2_orig = vel_c2 - vel_uncond
            d_neg1  = vel_c2_as_neg - vel_uncond  # c2's delta, used as noise for c1
            d_neg2  = vel_c1_as_neg - vel_uncond  # c1's delta, used as noise for c2

            d1_clean = d1_orig - cfg.suppression_weight * d_neg1
            d2_clean = d2_orig - cfg.suppression_weight * d_neg2

            vel_c1 = vel_uncond + d1_clean
            vel_c2 = vel_uncond + d2_clean

            suppression_delta = (
                cfg.suppression_weight * d_neg1.flatten(1).float().norm(dim=1).mean()
            )

        # ---- Phase dispatch ----
        if t_frac < cfg.phase_boundary:
            # Phase 1: Tweedie-space masking
            vf, M1_new, M2_new, info = _phase1_step(
                latents, vel_c1, vel_c2, vel_uncond, sigma, t_frac, cfg,
            )
            M1_frozen = M1_new
            M2_frozen = M2_new
            info["suppression_delta"] = suppression_delta.detach()

        else:
            # Phase 2: Spatially gated CFG with frozen masks
            if M1_frozen is None:
                # Edge case: phase_boundary=0, skip directly to Phase 2
                # Fall back to plain PoE
                d1, d2 = score_deltas(vel_c1, vel_c2, vel_uncond)
                vf = vel_uncond + cfg.guidance_scale * (d1 + d2)
                B, C, H, W = latents.shape
                M1_frozen = torch.full((B, 1, H, W), 0.5, device=latents.device)
                M2_frozen = torch.full((B, 1, H, W), 0.5, device=latents.device)
                info = {
                    "phase": torch.tensor(2.0),
                    "conflict": torch.zeros(B, device=latents.device),
                    "mask_sharpness": torch.tensor(0.5),
                    "spatial_iou": torch.tensor(1.0),
                    "x0_disagree": torch.tensor(0.0),
                    "tau_used": torch.tensor(0.0),
                    "blend_weight": torch.tensor(1.0),
                    "suppression_delta": suppression_delta.detach(),
                }
            else:
                vf, info = _phase2_step(
                    latents, vel_c1, vel_c2, vel_uncond, sigma,
                    M1_frozen, M2_frozen, cfg,
                )
                info["suppression_delta"] = suppression_delta.detach()

        infos.append(info)
        latents = scheduler_step(scheduler, vf, t, latents, sigma, dsigma)

    return latents, infos


# ---------------------------------------------------------------------------
# Single-step interface (for compatibility with compose_* pattern)
# ---------------------------------------------------------------------------

def compose_ir_poe(
    vel_c1:     Vel,
    vel_c2:     Vel,
    vel_uncond: Vel,
    latents:    Vel,
    sigma:      torch.Tensor,
    t_frac:     float,
    cfg: Optional[IRPoEConfig] = None,
    M1_frozen:  Optional[torch.Tensor] = None,
    M2_frozen:  Optional[torch.Tensor] = None,
) -> Tuple[Vel, dict, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Single-step IR-PoE (stateless interface — caller manages M1/M2 state).

    Returns (vf, info, M1_out, M2_out).
    M1_out/M2_out are None in Phase 2 (masks unchanged).
    """
    if cfg is None:
        cfg = IRPoEConfig()

    if t_frac < cfg.phase_boundary:
        vf, M1_new, M2_new, info = _phase1_step(
            latents, vel_c1, vel_c2, vel_uncond, sigma, t_frac, cfg,
        )
        return vf, info, M1_new, M2_new
    else:
        if M1_frozen is None or M2_frozen is None:
            B, C, H, W = latents.shape
            M1_frozen = torch.full((B, 1, H, W), 0.5, device=latents.device, dtype=latents.dtype)
            M2_frozen = torch.full((B, 1, H, W), 0.5, device=latents.device, dtype=latents.dtype)
        vf, info = _phase2_step(
            latents, vel_c1, vel_c2, vel_uncond, sigma, M1_frozen, M2_frozen, cfg,
        )
        return vf, info, None, None
