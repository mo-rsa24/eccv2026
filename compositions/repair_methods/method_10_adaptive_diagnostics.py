"""
Method 10 — Adaptive Schedule Driven by Expert Disagreement / Attention Overlap
================================================================================

Theoretical grounding
---------------------
Methods 01–09 apply fixed correction strategies uniformly across all
timesteps.  Method 10 is a *meta-controller*: it monitors three key
failure diagnostics at every step and activates different correction
sub-routines only when the corresponding failure signal is detected.

This avoids injecting unnecessary bias into Groups 1/2 (easy compositions)
while still repairing Groups 3/4 (hard regimes) at the right timestep.

Diagnostic signals
------------------
  D_t = ||x0_hat_1 - x0_hat_2||               (x0 disagreement)
  O_t = IoU(A_1, A_2)                          (attention map overlap)
  R_t = ||Delta_1|| / (||Delta_2|| + eps)      (dominance ratio)

Adaptive rules (executed in order; earlier rules take priority)
---------------------------------------------------------------
  Early timesteps (t_frac < early_threshold):
    - If D_t > disagree_thresh:
        -> Reduce both weights (avoid premature commitment)
        -> Increase exploration noise
    - Activate only mild gradient surgery

  Mid timesteps (early_threshold <= t_frac < late_threshold):
    - If O_t > overlap_thresh:
        -> Activate mask separation (Method 04 logic)
    - Always apply gradient surgery if cos < 0

  Late timesteps (t_frac >= late_threshold):
    - If R_t > dominance_thresh or R_t < 1/dominance_thresh:
        -> Apply dominance balancing (equalise weights by norm ratio)
    - If O_t > overlap_thresh:
        -> Apply retention: boost the weaker concept's delta

This method can be combined with any base composition (SuperDiff's
Proposition 6 kappa, vanilla PoE, or any Method 01-09) as an outer layer.

References
----------
- The document §"Adaptive schedules driven by expert disagreement /
  attention overlap / x0 inconsistency"
- A-STAR (arXiv:2306.14544) — attention retention
- Li et al. 2024 "Attention Overlap" (arXiv:2410.20972) — overlap = entity drop
- Sadat et al. 2024 "CFG Weight Schedulers" (arXiv:2404.13040)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable
import torch
import torch.nn.functional as F

from ._base import (
    Vel, score_deltas, cosine_similarity_deltas,
    x0_disagreement, predict_x0, delta_norms,
    dominance_ratio, attention_overlap_iou, scheduler_step,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveDiagnosticsConfig:
    # Base guidance scale
    guidance_scale: float = 7.5

    # Timestep thresholds (fraction of denoising complete)
    early_threshold: float = 0.30   # [0, early) = early regime
    late_threshold:  float = 0.70   # [late, 1] = late regime

    # Diagnostic thresholds
    disagree_thresh:   float = 0.5    # D_t threshold to trigger early correction
    overlap_thresh:    float = 0.3    # O_t threshold to trigger separation/retention
    dominance_thresh:  float = 2.0    # R_t threshold (or 1/R_t) for balancing

    # --- Early regime corrections ---
    # Weight shrinkage when D_t > disagree_thresh
    early_shrink_factor: float = 0.7
    # Extra noise multiplier for exploration
    early_explore_noise: float = 0.1

    # --- Mid regime corrections ---
    # Overlap separation strength (mask-gated PoE)
    mid_sep_strength: float = 0.4

    # --- Late regime corrections ---
    # Dominance balancing: rescale weaker concept's weight
    late_balance: bool = True
    # Retention: boost weaker concept's weight when overlap is high
    late_retention_boost: float = 1.5

    # Gradient surgery: always apply in mid+late when cos < 0
    use_gradient_surgery: bool = True
    surgery_eps: float = 1e-8

    # Whether to log the active regime at each step
    verbose: bool = False


# ---------------------------------------------------------------------------
# Diagnostic computation
# ---------------------------------------------------------------------------

def compute_diagnostics(
    vel_c1: Vel,
    vel_c2: Vel,
    vel_uncond: Vel,
    latents: Vel,
    sigma: torch.Tensor,
    M1: Optional[torch.Tensor] = None,
    M2: Optional[torch.Tensor] = None,
) -> dict:
    """
    Compute all three diagnostic signals.

    Returns dict with keys: D_t, O_t, R_t, cos_theta
    All tensors are [B] scalars.
    """
    d1, d2 = score_deltas(vel_c1, vel_c2, vel_uncond)

    x0_1 = predict_x0(latents, vel_c1, sigma)
    x0_2 = predict_x0(latents, vel_c2, sigma)
    D_t = x0_disagreement(x0_1, x0_2)          # [B]

    cos = cosine_similarity_deltas(d1, d2)      # [B]
    R_t = dominance_ratio(d1, d2)               # [B]

    O_t = None
    if M1 is not None and M2 is not None:
        O_t = attention_overlap_iou(M1, M2)     # [B]

    return {"D_t": D_t, "O_t": O_t, "R_t": R_t, "cos_theta": cos}


# ---------------------------------------------------------------------------
# Regime-specific correction functions
# ---------------------------------------------------------------------------

def _pcgrad_project(d1: Vel, d2: Vel, eps: float = 1e-8) -> Tuple[Vel, Vel]:
    """Inline PCGrad projection (subset of Method 02 logic)."""
    B = d1.shape[0]
    d1f = d1.reshape(B, -1).float()
    d2f = d2.reshape(B, -1).float()
    dot = (d1f * d2f).sum(dim=1, keepdim=True)
    conflict = (dot.squeeze(1) < 0)
    n2_sq = (d2f ** 2).sum(dim=1, keepdim=True) + eps
    n1_sq = (d1f ** 2).sum(dim=1, keepdim=True) + eps
    mask = conflict[:, None].float()
    d1_proj = d1f - mask * (dot / n2_sq) * d2f
    d2_proj = d2f - mask * (dot / n1_sq) * d1f
    return (
        d1_proj.reshape_as(d1).to(d1.dtype),
        d2_proj.reshape_as(d2).to(d2.dtype),
        conflict,
    )


def _mask_separate(
    d1: Vel,
    d2: Vel,
    M1: torch.Tensor,
    M2: torch.Tensor,
    sep_strength: float,
) -> Tuple[Vel, Vel]:
    """Inline mask-gated separation (subset of Method 04)."""
    overlap = M1 * M2
    M1_s = (M1 - sep_strength * overlap).clamp(0, 1)
    M2_s = (M2 - sep_strength * overlap).clamp(0, 1)
    M1_b = M1_s.unsqueeze(1).expand_as(d1)
    M2_b = M2_s.unsqueeze(1).expand_as(d2)
    return M1_b * d1, M2_b * d2


def _balance_by_norm(
    d1: Vel,
    d2: Vel,
    R_t: torch.Tensor,
    dominance_thresh: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    When one concept dominates (R_t >> 1 or << 1), equalise by scaling weights.
    Returns (w1, w2) per-batch scale factors.
    """
    B = d1.shape[0]
    w1 = torch.ones(B, device=d1.device, dtype=torch.float32)
    w2 = torch.ones(B, device=d2.device, dtype=torch.float32)

    dom_1 = (R_t > dominance_thresh)
    dom_2 = (R_t < 1.0 / dominance_thresh)

    # Concept 1 dominates: reduce w1 or boost w2
    if dom_1.any():
        w1[dom_1] = 1.0 / R_t[dom_1].clamp(min=1.0)
        w2[dom_1] = R_t[dom_1].clamp(max=dominance_thresh)

    # Concept 2 dominates: reduce w2 or boost w1
    if dom_2.any():
        inv_R = 1.0 / R_t[dom_2].clamp(min=1e-4)
        w2[dom_2] = 1.0 / inv_R.clamp(min=1.0)
        w1[dom_2] = inv_R.clamp(max=dominance_thresh)

    return w1, w2


# ---------------------------------------------------------------------------
# Main composition function (one step)
# ---------------------------------------------------------------------------

def compose_adaptive_diagnostics(
    vel_c1: Vel,
    vel_c2: Vel,
    vel_uncond: Vel,
    latents: Vel,
    sigma: torch.Tensor,
    dsigma: torch.Tensor,
    t_frac: float,
    cfg: Optional[AdaptiveDiagnosticsConfig] = None,
    M1: Optional[torch.Tensor] = None,
    M2: Optional[torch.Tensor] = None,
) -> Tuple[Vel, float, dict]:
    """
    Diagnostic-driven adaptive composition.

    Returns
    -------
    vf           : [B, C, H, W] composed velocity
    noise_scale  : float extra noise multiplier for the SDE step
    info         : dict with diagnostics and active regime
    """
    if cfg is None:
        cfg = AdaptiveDiagnosticsConfig()

    d1, d2 = score_deltas(vel_c1, vel_c2, vel_uncond)
    diags = compute_diagnostics(vel_c1, vel_c2, vel_uncond, latents, sigma, M1, M2)

    D_t = diags["D_t"]              # [B]
    O_t = diags["O_t"]              # [B] or None
    R_t = diags["R_t"]              # [B]
    cos = diags["cos_theta"]        # [B]

    # Base weights
    B = d1.shape[0]
    w1 = torch.ones(B, device=d1.device, dtype=torch.float32)
    w2 = torch.ones(B, device=d1.device, dtype=torch.float32)
    noise_scale = 1.0

    regime = "unknown"

    # ---- Early regime ----
    if t_frac < cfg.early_threshold:
        regime = "early"
        # Shrink weights when disagreement is high
        high_disagree = (D_t > cfg.disagree_thresh)
        if high_disagree.any():
            w1[high_disagree] *= cfg.early_shrink_factor
            w2[high_disagree] *= cfg.early_shrink_factor
            noise_scale = 1.0 + cfg.early_explore_noise

        # Mild gradient surgery
        if cfg.use_gradient_surgery:
            d1, d2, _ = _pcgrad_project(d1, d2, cfg.surgery_eps)

    # ---- Mid regime ----
    elif t_frac < cfg.late_threshold:
        regime = "mid"

        # Always apply gradient surgery when conflict exists
        if cfg.use_gradient_surgery:
            d1, d2, _ = _pcgrad_project(d1, d2, cfg.surgery_eps)

        # Activate mask separation when overlap is high
        if O_t is not None:
            high_overlap = (O_t > cfg.overlap_thresh)
            if high_overlap.any() and M1 is not None and M2 is not None:
                # Apply separation only to high-overlap batch elements
                d1_sep, d2_sep = _mask_separate(d1, d2, M1, M2, cfg.mid_sep_strength)
                mask_b = high_overlap[:, None, None, None].float()
                d1 = mask_b * d1_sep + (1 - mask_b) * d1
                d2 = mask_b * d2_sep + (1 - mask_b) * d2

    # ---- Late regime ----
    else:
        regime = "late"

        # Always apply gradient surgery
        if cfg.use_gradient_surgery:
            d1, d2, _ = _pcgrad_project(d1, d2, cfg.surgery_eps)

        # Dominance balancing
        if cfg.late_balance:
            w1_bal, w2_bal = _balance_by_norm(d1, d2, R_t, cfg.dominance_thresh)
            w1 = w1 * w1_bal
            w2 = w2 * w2_bal

        # Retention: boost weaker concept when overlap is high
        if O_t is not None:
            high_overlap = (O_t > cfg.overlap_thresh)
            if high_overlap.any():
                # Identify weaker concept (lower norm -> boost)
                n1, n2 = delta_norms(d1, d2)
                c1_weaker = (n1 < n2) & high_overlap
                c2_weaker = (n2 < n1) & high_overlap
                w1[c1_weaker] *= cfg.late_retention_boost
                w2[c2_weaker] *= cfg.late_retention_boost

    # ---- Compose ----
    w1_b = w1[:, None, None, None]
    w2_b = w2[:, None, None, None]
    vf = vel_uncond + cfg.guidance_scale * (w1_b * d1 + w2_b * d2)

    info = {
        "regime": regime,
        "D_t": D_t.detach(),
        "O_t": O_t.detach() if O_t is not None else None,
        "R_t": R_t.detach(),
        "cos_theta": cos.detach(),
        "w1": w1.detach(),
        "w2": w2.detach(),
        "noise_scale": noise_scale,
    }

    if cfg.verbose:
        print(
            f"  t_frac={t_frac:.2f} | regime={regime}"
            f" | D_t={D_t.mean():.3f} | R_t={R_t.mean():.3f}"
            f" | O_t={O_t.mean():.3f}" if O_t is not None else ""
        )

    return vf, noise_scale, info


# ---------------------------------------------------------------------------
# Full denoising loop
# ---------------------------------------------------------------------------

def run_adaptive_diagnostics(
    latents: Vel,
    vel_fn,
    embeddings_c1,
    embeddings_c2,
    embeddings_uncond,
    scheduler,
    cfg: Optional[AdaptiveDiagnosticsConfig] = None,
    cond_kwargs_c1: Optional[dict] = None,
    cond_kwargs_c2: Optional[dict] = None,
    cond_kwargs_uncond: Optional[dict] = None,
    mask_fn: Optional[Callable] = None,   # callable(latents, t, sigma) -> (M1, M2)
) -> Tuple[Vel, list]:
    """
    Full denoising loop with diagnostic-driven adaptive composition.

    mask_fn (optional): returns spatial masks from attention or delta norms.
    """
    if cfg is None:
        cfg = AdaptiveDiagnosticsConfig()

    N = len(scheduler.timesteps)
    infos = []

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma  = scheduler.sigmas[i]
        t_frac = i / max(N - 1, 1)

        vel_c1    = vel_fn(latents, t, sigma, embeddings_c1, cond_kwargs_c1)
        vel_c2    = vel_fn(latents, t, sigma, embeddings_c2, cond_kwargs_c2)
        vel_uncond = vel_fn(latents, t, sigma, embeddings_uncond, cond_kwargs_uncond)

        # Optional mask extraction
        M1, M2 = None, None
        if mask_fn is not None:
            M1, M2 = mask_fn(latents, t, sigma)

        vf, noise_scale, info = compose_adaptive_diagnostics(
            vel_c1, vel_c2, vel_uncond,
            latents, sigma, dsigma, t_frac,
            cfg, M1, M2,
        )
        infos.append(info)

        latents = scheduler_step(scheduler, vf, t, latents, sigma, dsigma, noise_scale=noise_scale)

    return latents, infos
