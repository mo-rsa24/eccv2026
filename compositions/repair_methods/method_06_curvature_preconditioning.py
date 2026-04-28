"""
Method 06 — Second-Order / Curvature-Aware Preconditioning
===========================================================

Theoretical grounding
---------------------
Standard PoE composition implicitly uses Euclidean geometry in latent
space: it assumes all directions of the score delta are equally important.
When the score field is stiff or highly curved in certain directions,
a naive sum of score deltas can overshoot into hybrid regions or collapse.

A preconditioned update replaces the Euclidean step with:

    x_{t-Δt} <- x_t  +  η · P_t · (su + Σ_i w_i · Delta_i)  +  noise

where P_t is a symmetric positive definite (SPD) preconditioner estimated
from the score network.

Three levels of approximation (in decreasing accuracy, increasing speed):

  Level 1 — Diagonal empirical covariance of Delta_i (cheapest):
      P_t = diag(1 / (var(Delta_1, Delta_2) + eps))
      Variance computed element-wise over the two expert deltas.
      Dampens directions where experts disagree greatly.

  Level 2 — Rank-1 correction based on conflict direction (medium):
      Find the direction of maximal conflict:
          d_conflict = (Delta_1 - Delta_2) / ||Delta_1 - Delta_2||
      Reduce step size along d_conflict:
          P_t = I - (1 - alpha) * d_conflict * d_conflict^T
      This is a rank-1 spectral dampening of the conflict subspace.

  Level 3 — Jacobian-vector product estimate (most accurate, slow):
      Estimate the diagonal Hessian of the score network via finite
      differences in the forward pass:
          diag(J_s) ~ (s(x + eps*e_i) - s(x - eps*e_i)) / (2*eps)  per dim i
      In practice, approximate with a single random JVP direction.

This implementation provides Levels 1 and 2 (Level 3 requires gradient
access to the denoising network and is left as an extension).

References
----------
- Karras et al. 2022 "EDM" (arXiv:2206.00364) §3 — EDM preconditioning
- The document §"Second-order or curvature-aware corrections"
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from ._base import Vel, score_deltas, delta_norms, scheduler_step


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CurvaturePrecondConfig:
    # Guidance scale applied to the preconditioned update
    guidance_scale: float = 7.5

    # Preconditioning level: 1 (diagonal covariance), 2 (rank-1 conflict)
    level: int = 1

    # Level 1: clamp preconditioner values to [damp_min, damp_max]
    # to prevent exploding/vanishing steps
    damp_min: float = 0.1
    damp_max: float = 10.0

    # Level 2: alpha controls how much the conflict direction is dampened
    # alpha=0 -> conflict direction fully removed; alpha=1 -> no effect (identity)
    alpha: float = 0.3

    # Whether to re-scale the update to match the original norm
    # (preconditioning changes direction; this option also changes magnitude)
    preserve_norm: bool = False

    eps: float = 1e-8


# ---------------------------------------------------------------------------
# Level 1: Diagonal covariance preconditioner
# ---------------------------------------------------------------------------

def _diagonal_preconditioner(
    d1: Vel,
    d2: Vel,
    cfg: CurvaturePrecondConfig,
) -> torch.Tensor:
    """
    Diagonal preconditioner based on element-wise variance of [d1, d2].

    Var_elem = 0.5 * [(d1 - mean)^2 + (d2 - mean)^2]
    P_diag   = 1 / (sqrt(Var_elem) + eps)
    Clamped to [damp_min, damp_max].

    Shape: [B, C, H, W] (same as input)
    """
    mean = 0.5 * (d1 + d2)
    var  = 0.5 * ((d1 - mean) ** 2 + (d2 - mean) ** 2)   # [B, C, H, W]
    p_diag = 1.0 / (var.float().sqrt() + cfg.eps)
    p_diag = p_diag.clamp(cfg.damp_min, cfg.damp_max)
    return p_diag.to(d1.dtype)


# ---------------------------------------------------------------------------
# Level 2: Rank-1 conflict-direction dampening
# ---------------------------------------------------------------------------

def _rank1_conflict_preconditioner(
    d1: Vel,
    d2: Vel,
    g_nom: torch.Tensor,    # [B, D] nominal update (flat)
    cfg: CurvaturePrecondConfig,
) -> torch.Tensor:          # [B, D] preconditioned update (flat)
    """
    P_t = I - (1 - alpha) * d_c * d_c^T  applied to g_nom.

    d_c = normalised conflict direction = (d1 - d2) / ||d1 - d2||

    P_t @ g_nom = g_nom - (1-alpha) * (d_c . g_nom) * d_c
    """
    B, D = g_nom.shape
    d1_flat = d1.reshape(B, D).float()
    d2_flat = d2.reshape(B, D).float()

    diff = d1_flat - d2_flat
    diff_norm = diff.norm(dim=1, keepdim=True) + cfg.eps
    d_conflict = diff / diff_norm                              # [B, D]

    proj = (g_nom.float() * d_conflict).sum(dim=1, keepdim=True)  # [B, 1]
    g_prec = g_nom.float() - (1 - cfg.alpha) * proj * d_conflict   # [B, D]

    return g_prec.to(g_nom.dtype)


# ---------------------------------------------------------------------------
# Main composition function
# ---------------------------------------------------------------------------

def compose_curvature_precond(
    vel_c1: Vel,
    vel_c2: Vel,
    vel_uncond: Vel,
    cfg: Optional[CurvaturePrecondConfig] = None,
) -> Tuple[Vel, dict]:
    """
    Curvature-preconditioned composition.

    Returns
    -------
    vf   : [B, C, H, W] preconditioned composed velocity
    info : dict with diagnostics
    """
    if cfg is None:
        cfg = CurvaturePrecondConfig()

    B, C, H, W = vel_c1.shape
    D = C * H * W

    d1, d2 = score_deltas(vel_c1, vel_c2, vel_uncond)

    # Nominal update
    g_nom = vel_uncond + cfg.guidance_scale * (d1 + d2)

    if cfg.level == 1:
        # Diagonal element-wise preconditioning
        P_diag = _diagonal_preconditioner(d1, d2, cfg)  # [B, C, H, W]
        g_prec = P_diag * g_nom

        if cfg.preserve_norm:
            nom_norm = g_nom.reshape(B, -1).float().norm(dim=1)
            prec_norm = g_prec.reshape(B, -1).float().norm(dim=1) + cfg.eps
            scale = nom_norm / prec_norm
            g_prec = g_prec * scale[:, None, None, None]

        info = {
            "P_diag_mean": P_diag.mean().item(),
            "P_diag_max":  P_diag.amax(dim=(1, 2, 3)).detach(),
        }

    elif cfg.level == 2:
        # Rank-1 conflict-direction dampening
        g_flat = g_nom.reshape(B, D)
        g_prec_flat = _rank1_conflict_preconditioner(d1, d2, g_flat, cfg)

        if cfg.preserve_norm:
            nom_norm = g_flat.float().norm(dim=1)
            prec_norm = g_prec_flat.float().norm(dim=1) + cfg.eps
            scale = nom_norm / prec_norm
            g_prec_flat = g_prec_flat * scale[:, None]

        g_prec = g_prec_flat.reshape(B, C, H, W).to(vel_c1.dtype)

        # Measure how much the conflict direction was suppressed
        d1_flat = d1.reshape(B, D).float()
        d2_flat = d2.reshape(B, D).float()
        diff = d1_flat - d2_flat
        diff_norm = diff.norm(dim=1) + cfg.eps
        d_conflict = diff / diff_norm[:, None]
        proj_before = (g_flat.float() * d_conflict).sum(dim=1)
        proj_after  = (g_prec_flat.float() * d_conflict).sum(dim=1)

        info = {
            "conflict_proj_before": proj_before.detach(),
            "conflict_proj_after":  proj_after.detach(),
        }

    else:
        raise ValueError(f"Unsupported preconditioning level: {cfg.level}. Use 1 or 2.")

    return g_prec, info


# ---------------------------------------------------------------------------
# Full denoising loop
# ---------------------------------------------------------------------------

def run_curvature_preconditioning(
    latents: Vel,
    vel_fn,
    embeddings_c1,
    embeddings_c2,
    embeddings_uncond,
    scheduler,
    cfg: Optional[CurvaturePrecondConfig] = None,
    cond_kwargs_c1: Optional[dict] = None,
    cond_kwargs_c2: Optional[dict] = None,
    cond_kwargs_uncond: Optional[dict] = None,
) -> Tuple[Vel, list]:
    """Full denoising loop with curvature-aware preconditioning."""
    if cfg is None:
        cfg = CurvaturePrecondConfig()

    infos = []
    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma = scheduler.sigmas[i]

        vel_c1 = vel_fn(latents, t, sigma, embeddings_c1, cond_kwargs_c1)
        vel_c2 = vel_fn(latents, t, sigma, embeddings_c2, cond_kwargs_c2)
        vel_uncond = vel_fn(latents, t, sigma, embeddings_uncond, cond_kwargs_uncond)

        vf, info = compose_curvature_precond(vel_c1, vel_c2, vel_uncond, cfg)
        infos.append(info)

        latents = scheduler_step(scheduler, vf, t, latents, sigma, dsigma)

    return latents, infos
