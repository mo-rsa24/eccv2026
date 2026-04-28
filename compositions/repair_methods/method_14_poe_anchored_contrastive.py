"""
Method 14 — PoE-Anchored Contrastive Tweedie
=============================================

Motivation
----------
CO3 achieves high-quality compositional generation using the contrastive formula:

    x̂₀_comp = w_0 · x̂₀(ε_multi) + w_1 · x̂₀(ε_c1) + w_2 · x̂₀(ε_c2)

with sum-zero weights (w_0=2, w_1=w_2=-1): the joint prompt prediction acts as
a spatial anchor and the individual concept predictions are *subtracted* away,
removing the mode-collision bias that pure PoE accumulates.

The problem: CO3 requires a joint prompt "a cat and a dog" to produce ε_multi,
which explicitly encodes spatial co-occurrence.  Our goal is to replicate this
behaviour without providing a joint prompt.

Key observation
---------------
PoE already constructs a pseudo-joint prediction at each step:

    x̂₀_PoE = x̂₀_c1 + x̂₀_c2 − x̂₀_unc

This is a coarse approximation of the joint Tweedie prediction.  We treat it as
the anchor and apply CO3's contrastive subtraction:

    x̂₀_comp = w_0 · x̂₀_PoE + w_1 · x̂₀_c1 + w_2 · x̂₀_c2

with w_0 > 0, w_1 = w_2 < 0 (sum-zero constraint: w_0 + w_1 + w_2 = 0).

Expanding:
    x̂₀_comp = w_0 · (x̂₀_c1 + x̂₀_c2 − x̂₀_unc)
             + w_1 · x̂₀_c1
             + w_2 · x̂₀_c2
           = (w_0 + w_1) · x̂₀_c1 + (w_0 + w_2) · x̂₀_c2 − w_0 · x̂₀_unc

With symmetric weights w_1 = w_2 = −w_0/2:
    x̂₀_comp = (w_0/2) · x̂₀_c1 + (w_0/2) · x̂₀_c2 − w_0 · x̂₀_unc
             = w_0 · (x̂₀_c1/2 + x̂₀_c2/2 − x̂₀_unc)
             = w_0 · x̂₀_PoE / 2   [just a scaling — degenerate!]

This means standard symmetric sum-zero weights collapse to a scaled PoE.  The
non-trivial variant arises when weights are *asymmetric* or when the anchor
receives extra weight beyond the sum-zero constraint.

Better formulation (matches CO3 intent):
    w_0 = anchor_weight  (e.g. 1.5)
    w_1 = w_2 = −(anchor_weight − 1) / 2   [so w_0 + w_1 + w_2 = 1, not 0]

This preserves the scale while still subtracting the mode-collision directions
from the per-concept predictions.

Why this should help:
    If x̂₀_PoE already contains good spatial layout (cat left, dog right), and
    x̂₀_c1 predicts "cat in centre" (mode-collapsed), then:
        x̂₀_comp = w_0 · x̂₀_PoE − |w_1| · (x̂₀_c1 + x̂₀_c2)
    subtracts the centred/collapsed components, leaving the spread-out parts of
    x̂₀_PoE intact.  The same mechanism CO3 uses with its joint prompt.

Algorithm
---------
Standard denoising loop; per step:

1.  Compute x̂₀_c1, x̂₀_c2, x̂₀_unc from Tweedie formula
2.  Compute x̂₀_PoE = x̂₀_c1 + x̂₀_c2 − x̂₀_unc
3.  Compute contrastive composite:
        x̂₀_comp = anchor_w · x̂₀_PoE − concept_w · (x̂₀_c1 + x̂₀_c2)
    where concept_w = (anchor_w − 1) / 2  (sum-one constraint)
4.  Convert back to velocity: vel_comp = (x_t − x̂₀_comp) / σ
5.  Apply CFG: vf = vel_unc + gs · (vel_comp − vel_unc)

The anchor_weight can be scheduled: start near 1.0 (plain PoE) at pure-noise
steps and ramp up after layout forms (t_frac > phase_start), increasing the
contrastive subtraction as spatial structure becomes reliable.

Diagnostics tracked per step
-----------------------------
  poe_norm          : ||x̂₀_PoE|| (sanity)
  comp_norm         : ||x̂₀_comp||
  x0_disagree       : ||x̂₀_c1 − x̂₀_c2|| (concept separation)
  cos_d1_d2         : cos(Δ₁, Δ₂) (constructive vs destructive interference)
  anchor_w          : effective anchor weight at this step
  contrastive_delta : ||x̂₀_comp − x̂₀_PoE|| (how much the subtraction changed)
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
    scheduler_step,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PoeAnchoredContrastiveConfig:
    guidance_scale: float = 7.5

    # Anchor weight for x̂₀_PoE.  At anchor_w=1.0 this reduces to plain PoE.
    # At anchor_w > 1 the contrastive subtraction is active.
    # Default 1.5 ≈ CO3's "2" relative to the sum-one scaling.
    anchor_weight_start: float = 1.0    # at t_frac=0 (pure noise): near-PoE
    anchor_weight_peak:  float = 1.8    # peak around layout-formation
    anchor_weight_end:   float = 1.2    # taper off at fine-detail stage

    # t_frac thresholds for the weight schedule
    phase_ramp_start: float = 0.15   # start increasing anchor weight
    phase_peak:       float = 0.40   # peak
    phase_ramp_end:   float = 0.75   # taper back toward 1.0

    # Safety clip on contrastive delta
    delta_clip_frac: float = 0.8   # ||Δx̂₀|| ≤ clip_frac * ||x̂₀_PoE||

    norm_eps: float = 1e-8


# ---------------------------------------------------------------------------
# Weight schedule
# ---------------------------------------------------------------------------

def _anchor_weight(t_frac: float, cfg: PoeAnchoredContrastiveConfig) -> float:
    """
    Triangular schedule for anchor_weight:
      t_frac ≤ phase_ramp_start → anchor_weight_start
      t_frac in [ramp_start, peak] → linear ramp up to anchor_weight_peak
      t_frac in [peak, ramp_end] → linear ramp down to anchor_weight_end
      t_frac ≥ ramp_end → anchor_weight_end
    """
    if t_frac <= cfg.phase_ramp_start:
        return cfg.anchor_weight_start
    if t_frac <= cfg.phase_peak:
        progress = (t_frac - cfg.phase_ramp_start) / (cfg.phase_peak - cfg.phase_ramp_start + 1e-8)
        return cfg.anchor_weight_start + progress * (cfg.anchor_weight_peak - cfg.anchor_weight_start)
    if t_frac <= cfg.phase_ramp_end:
        progress = (t_frac - cfg.phase_peak) / (cfg.phase_ramp_end - cfg.phase_peak + 1e-8)
        return cfg.anchor_weight_peak + progress * (cfg.anchor_weight_end - cfg.anchor_weight_peak)
    return cfg.anchor_weight_end


# ---------------------------------------------------------------------------
# Main composition function (single step)
# ---------------------------------------------------------------------------

def compose_poe_anchored_contrastive(
    vel_c1:     Vel,
    vel_c2:     Vel,
    vel_uncond: Vel,
    latents:    Vel,
    sigma:      torch.Tensor,
    t_frac:     float,
    cfg: Optional[PoeAnchoredContrastiveConfig] = None,
) -> Tuple[Vel, dict]:
    """
    Single-step PoE-Anchored Contrastive Tweedie composition.

    Parameters
    ----------
    vel_c1, vel_c2, vel_uncond : [B, C, H, W] velocity fields
    latents : [B, C, H, W] current noisy latents x_t
    sigma   : scalar or [B] noise level σ_t
    t_frac  : float in [0,1], fraction of denoising completed (0=noise, 1=clean)
    cfg     : PoeAnchoredContrastiveConfig

    Returns
    -------
    vf   : [B, C, H, W] composed velocity
    info : dict of per-step diagnostics
    """
    if cfg is None:
        cfg = PoeAnchoredContrastiveConfig()

    # ---- Tweedie x̂₀ estimates ----
    x0_c1  = predict_x0(latents, vel_c1,    sigma)   # [B,C,H,W]
    x0_c2  = predict_x0(latents, vel_c2,    sigma)
    x0_unc = predict_x0(latents, vel_uncond, sigma)

    # ---- PoE anchor in x̂₀-space ----
    x0_poe = x0_c1 + x0_c2 - x0_unc   # standard PoE — pseudo-joint anchor

    # ---- Contrastive weighting ----
    # x̂₀_comp = anchor_w · x̂₀_PoE − concept_w · (x̂₀_c1 + x̂₀_c2)
    # where concept_w = (anchor_w − 1) / 2  ← sum-one preservation
    aw = _anchor_weight(t_frac, cfg)
    cw = (aw - 1.0) / 2.0   # concept subtraction weight

    x0_comp = aw * x0_poe - cw * (x0_c1 + x0_c2)

    # ---- Safety clip on contrastive delta ----
    delta    = x0_comp - x0_poe
    if cfg.delta_clip_frac > 0:
        poe_norm   = x0_poe.flatten(1).float().norm(dim=1, keepdim=True)
        delta_norm = delta.flatten(1).float().norm(dim=1, keepdim=True)
        max_norm   = cfg.delta_clip_frac * poe_norm + cfg.norm_eps
        scale      = (max_norm / delta_norm.clamp(min=cfg.norm_eps)).clamp(max=1.0)
        scale_b    = scale.reshape(delta.shape[0], 1, 1, 1).to(delta.dtype)
        x0_comp    = x0_poe + delta * scale_b

    # ---- Convert x̂₀_comp back to velocity and apply CFG ----
    sigma_b  = sigma if sigma.ndim == 0 else sigma.reshape(-1, 1, 1, 1)
    vel_comp = (latents - x0_comp) / (sigma_b + cfg.norm_eps)
    vf = vel_uncond + cfg.guidance_scale * (vel_comp - vel_uncond)

    # ---- Diagnostics ----
    d1, d2    = score_deltas(vel_c1, vel_c2, vel_uncond)
    cos_d1_d2 = cosine_similarity_deltas(d1, d2)
    x0_dis    = x0_disagreement(x0_c1, x0_c2)

    poe_norm_val  = x0_poe.flatten(1).float().norm(dim=1).mean()
    comp_norm_val = x0_comp.flatten(1).float().norm(dim=1).mean()
    cont_delta    = (x0_comp - x0_poe).flatten(1).float().norm(dim=1).mean()

    info = {
        "anchor_w":           torch.tensor(float(aw)),
        "concept_w":          torch.tensor(float(cw)),
        "poe_norm":           poe_norm_val.detach(),
        "comp_norm":          comp_norm_val.detach(),
        "x0_disagree":        x0_dis.detach(),
        "cos_d1_d2":          cos_d1_d2.detach(),
        "contrastive_delta":  cont_delta.detach(),
    }
    return vf, info


# ---------------------------------------------------------------------------
# Full denoising loop
# ---------------------------------------------------------------------------

def run_poe_anchored_contrastive(
    latents:             Vel,
    vel_fn,              # callable(latents, t, sigma, embeddings, cond_kwargs) -> Vel
    embeddings_c1,
    embeddings_c2,
    embeddings_uncond,
    scheduler,
    cfg: Optional[PoeAnchoredContrastiveConfig] = None,
    cond_kwargs_c1:      Optional[dict] = None,
    cond_kwargs_c2:      Optional[dict] = None,
    cond_kwargs_uncond:  Optional[dict] = None,
    device=None,
) -> Tuple[Vel, list]:
    """
    Full denoising loop using PoE-Anchored Contrastive Tweedie (Method 14).

    Interface identical to all other run_* functions in this package.
    Returns (final_latents, list_of_per_step_info_dicts).
    """
    if cfg is None:
        cfg = PoeAnchoredContrastiveConfig()

    N = len(scheduler.timesteps)
    infos = []

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma  = scheduler.sigmas[i]
        t_frac = i / max(N - 1, 1)   # 0 = pure noise, 1 = clean

        vel_c1     = vel_fn(latents, t, sigma, embeddings_c1,     cond_kwargs_c1)
        vel_c2     = vel_fn(latents, t, sigma, embeddings_c2,     cond_kwargs_c2)
        vel_uncond = vel_fn(latents, t, sigma, embeddings_uncond, cond_kwargs_uncond)

        vf, info = compose_poe_anchored_contrastive(
            vel_c1, vel_c2, vel_uncond,
            latents, sigma, t_frac, cfg,
        )
        infos.append(info)

        latents = scheduler_step(scheduler, vf, t, latents, sigma, dsigma)

    return latents, infos
