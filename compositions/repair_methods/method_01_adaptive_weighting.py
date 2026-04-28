"""
Method 01 — Adaptive Timestep- and State-Dependent Expert Weighting
====================================================================

Theoretical grounding
---------------------
Standard PoE (and SuperDiff) use fixed or Proposition-6-determined weights
that do not respond to *how much the two experts currently agree*.  The
missing interaction term r_t is large precisely when experts disagree
sharply.  This method modulates the guidance weights w_i(t, x_t) using
three inference-time diagnostics:

  1. Expert disagreement angle:  cos(theta) = (Delta_1 . Delta_2) / (||Delta_1|| ||Delta_2||)
  2. Predicted x0 inconsistency:  D_t = ||x0_hat_1 - x0_hat_2||
  3. Dominance ratio:             R_t = ||Delta_1|| / ||Delta_2||

Composed velocity:
    vf = su  +  w1(t, xt) * Delta_1  +  w2(t, xt) * Delta_2

where w_i are shrunk from a base schedule when disagreement / dominance
are large:
    w_i <- base_w(t) / (1 + k_disagree * D_t)   (disagree shrinkage)
    w_2 <- w_2 * (1 / R_t).clamp(min_ratio, max_ratio)  (dominance balance)

References
----------
- Sadat et al. 2024 "Analysis of CFG Weight Schedulers" (arXiv:2404.13040)
- The document §"Dynamic timestep- and state-dependent expert weighting"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import math
import torch

from ._base import (
    Vel, score_deltas, cosine_similarity_deltas,
    x0_disagreement, predict_x0, dominance_ratio,
    delta_norms, scheduler_step,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveWeightingConfig:
    # Base guidance scale (mirrors SuperDiff guidance_scale)
    guidance_scale: float = 7.5

    # How aggressively to shrink weights when x0 predictions disagree
    # w_i <- base / (1 + k_disagree * D_t)
    k_disagree: float = 1.0

    # Damping applied when angle between deltas is negative (conflict)
    k_conflict: float = 0.3

    # Dominance balancing: clamp R_t to [min_ratio, max_ratio] then rescale w2
    balance_dominance: bool = True
    min_ratio: float = 0.5    # never let one concept be <50% as strong
    max_ratio: float = 2.0    # never let one concept be >2x as strong

    # Base weight schedule: "constant", "linear_increase", "cosine_increase"
    # These control how the base weight grows as noise decreases (t_frac -> 1)
    schedule: str = "constant"
    w_lo: float = 7.5    # weight at pure noise (t_frac=0)
    w_hi: float = 7.5    # weight at clean (t_frac=1)  [only used if schedule != constant]


# ---------------------------------------------------------------------------
# Per-step composition
# ---------------------------------------------------------------------------

def _base_weight(t_frac: float, cfg: AdaptiveWeightingConfig) -> float:
    """Base guidance weight as function of fractional timestep."""
    if cfg.schedule == "constant":
        return cfg.guidance_scale
    elif cfg.schedule == "linear_increase":
        return cfg.w_lo + t_frac * (cfg.w_hi - cfg.w_lo)
    elif cfg.schedule == "cosine_increase":
        return cfg.w_lo + (cfg.w_hi - cfg.w_lo) * (1 - math.cos(math.pi * t_frac)) / 2
    else:
        raise ValueError(f"Unknown schedule: {cfg.schedule!r}")


def compose_adaptive_weighting(
    vel_c1: Vel,
    vel_c2: Vel,
    vel_uncond: Vel,
    latents: Vel,
    sigma: torch.Tensor,
    t_frac: float,                  # fractional timestep in [0,1]
    cfg: Optional[AdaptiveWeightingConfig] = None,
) -> Tuple[Vel, dict]:
    """
    Compute composed velocity with adaptive per-concept weights.

    Parameters
    ----------
    vel_c1, vel_c2, vel_uncond : [B, C, H, W] velocity fields
    latents : [B, C, H, W] current latent state x_t
    sigma   : scalar or [B] noise level
    t_frac  : float in [0,1], fraction of denoising complete
    cfg     : AdaptiveWeightingConfig

    Returns
    -------
    vf   : [B, C, H, W] composed velocity
    info : dict with diagnostics (cos_theta, D_t, R_t, w1, w2)
    """
    if cfg is None:
        cfg = AdaptiveWeightingConfig()

    B = vel_c1.shape[0]
    d1, d2 = score_deltas(vel_c1, vel_c2, vel_uncond)

    # ---- Diagnostics ----
    cos_theta = cosine_similarity_deltas(d1, d2)          # [B]  in [-1, 1]
    x0_1 = predict_x0(latents, vel_c1, sigma)
    x0_2 = predict_x0(latents, vel_c2, sigma)
    D_t = x0_disagreement(x0_1, x0_2)                    # [B]  >= 0
    R_t = dominance_ratio(d1, d2)                         # [B]  >= 0
    n1, n2 = delta_norms(d1, d2)

    # ---- Base weight ----
    base_w = _base_weight(t_frac, cfg)

    # ---- Disagreement-driven shrinkage ----
    # Both experts shrunk equally when they disagree strongly on x0
    shrink = 1.0 / (1.0 + cfg.k_disagree * D_t)          # [B]  in (0, 1]
    w1 = base_w * shrink                                  # [B]
    w2 = base_w * shrink                                  # [B]

    # ---- Conflict gating: extra damping when angle is negative ----
    conflict_mask = (cos_theta < 0).float()               # [B]
    conflict_damp = 1.0 / (1.0 + cfg.k_conflict * conflict_mask)
    w1 = w1 * conflict_damp
    w2 = w2 * conflict_damp

    # ---- Dominance balancing: equalise norms late in trajectory ----
    if cfg.balance_dominance:
        # R_t = n1/n2.  If R_t >> 1, concept-1 dominates; boost w2 and/or shrink w1.
        R_t_clamped = R_t.clamp(cfg.min_ratio, cfg.max_ratio)
        # Rescale so the geometric mean of effective norms is preserved:
        #   w1 * n1 / sqrt(R_t), w2 * n2 * sqrt(R_t)
        sqrt_R = R_t_clamped.sqrt()
        w1 = w1 / sqrt_R
        w2 = w2 * sqrt_R

    # ---- Compose ----
    w1_b = w1[:, None, None, None]
    w2_b = w2[:, None, None, None]
    vf = vel_uncond + w1_b * d1 + w2_b * d2

    info = {
        "cos_theta": cos_theta.detach(),
        "D_t": D_t.detach(),
        "R_t": R_t.detach(),
        "w1": w1.detach(),
        "w2": w2.detach(),
    }
    return vf, info


# ---------------------------------------------------------------------------
# Full denoising loop
# ---------------------------------------------------------------------------

def run_adaptive_weighting(
    latents: Vel,
    vel_fn,          # callable(latents, t, sigma, emb, kwargs) -> Vel
    embeddings_c1,
    embeddings_c2,
    embeddings_uncond,
    scheduler,
    cfg: Optional[AdaptiveWeightingConfig] = None,
    cond_kwargs_c1: Optional[dict] = None,
    cond_kwargs_c2: Optional[dict] = None,
    cond_kwargs_uncond: Optional[dict] = None,
    device=None,
) -> Tuple[Vel, list]:
    """
    Full denoising loop using adaptive weighting composition.

    vel_fn signature:
        vel_fn(latents, t, sigma, embeddings, added_cond_kwargs) -> Vel

    Returns (final_latents, list_of_per_step_info_dicts)
    """
    if cfg is None:
        cfg = AdaptiveWeightingConfig()

    N = len(scheduler.timesteps)
    infos = []

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma = scheduler.sigmas[i]
        t_frac = i / max(N - 1, 1)   # 0 at start (pure noise), 1 at end (clean)

        vel_c1 = vel_fn(latents, t, sigma, embeddings_c1, cond_kwargs_c1)
        vel_c2 = vel_fn(latents, t, sigma, embeddings_c2, cond_kwargs_c2)
        vel_uncond = vel_fn(latents, t, sigma, embeddings_uncond, cond_kwargs_uncond)

        vf, info = compose_adaptive_weighting(
            vel_c1, vel_c2, vel_uncond, latents, sigma, t_frac, cfg
        )
        infos.append(info)

        latents = scheduler_step(scheduler, vf, t, latents, sigma, dsigma)

    return latents, infos
