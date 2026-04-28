"""
Method 03 — Trust-Region / Constrained QP Composition
======================================================

Theoretical grounding
---------------------
The vanilla PoE step g = su + w1*Delta_1 + w2*Delta_2 is unconstrained:
it can take arbitrarily large steps in any direction.  When the two
experts conflict or one dominates, this can push the latent off-manifold
or into a hybrid basin.

This method replaces the unconstrained step with a constrained quadratic
program (QP) per denoising step:

    min_{g}  ||g - g_nominal||^2
    subject to:
        g . Delta_1 >= kappa_1(t)   [concept-1 must make progress]
        g . Delta_2 >= kappa_2(t)   [concept-2 must make progress]

where g_nominal = su + w1*Delta_1 + w2*Delta_2 is the vanilla composed
velocity, and kappa_i(t) are per-timestep minimum-progress thresholds.

The QP has a closed-form solution for two linear constraints via
Lagrangian duality (see _solve_constrained_qp).

Intuition
---------
- If the nominal g already satisfies both constraints, it is returned unchanged.
- If one constraint is violated, g is projected onto the feasible half-space.
- If both are violated and the feasible set is non-empty, g is pulled to the
  constraint boundary closest to g_nominal.
- If the feasible set is empty (true conflict), the constraints are softened
  and a best-effort solution is returned.

Why this helps
--------------
- Group 3: Prevents the entangled direction from boosting one concept while
  strongly decreasing the other.
- Group 4: Can enforce balanced progress, reducing dominance.

References
----------
- The document §"Trust-region or constrained optimisation versions of PoE"
- Boyd & Vandenberghe "Convex Optimization" §5 (Lagrangian duality)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch

from ._base import Vel, score_deltas, delta_norms, scheduler_step


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrustRegionConfig:
    # Base guidance scale for nominal velocity
    guidance_scale: float = 7.5

    # Minimum progress thresholds kappa_i(t) as fraction of ||Delta_i||^2
    # kappa_i(t) = kappa_fraction * ||Delta_i||^2 * schedule(t)
    # Set to 0.0 to disable (purely unconstrained == vanilla PoE)
    kappa_fraction: float = 0.05

    # Schedule for thresholds: "constant", "linear_increase"
    # linear_increase: threshold grows from 0 (early) to kappa_fraction (late)
    threshold_schedule: str = "linear_increase"

    # When the feasible set is empty, soften: reduce kappa by this factor
    # and retry.  If still infeasible, return projection onto intersection
    # of individual half-spaces (best-effort).
    infeasible_softening: float = 0.5

    eps: float = 1e-8


# ---------------------------------------------------------------------------
# QP solver: two linear constraints
# ---------------------------------------------------------------------------

def _solve_constrained_qp(
    g_nom: torch.Tensor,    # [B, D]  nominal update
    a1: torch.Tensor,       # [B, D]  constraint 1 direction (Delta_1 flat)
    a2: torch.Tensor,       # [B, D]  constraint 2 direction (Delta_2 flat)
    k1: torch.Tensor,       # [B]     threshold for constraint 1
    k2: torch.Tensor,       # [B]     threshold for constraint 2
    eps: float = 1e-8,
) -> torch.Tensor:          # [B, D]  solution g*
    """
    Closed-form solution to:
        min_{g} ||g - g_nom||^2
        s.t.    g.a1 >= k1,  g.a2 >= k2

    Case analysis:
    1. g_nom already feasible -> return g_nom
    2. Only constraint 1 violated -> project onto {g.a1 = k1}
    3. Only constraint 2 violated -> project onto {g.a2 = k2}
    4. Both violated -> solve 2-constraint KKT system

    Each case handled batchwise with masking.
    """
    B, D = g_nom.shape

    # Check which constraints are violated
    dot1 = (g_nom * a1).sum(dim=1)   # [B]  g_nom . Delta_1
    dot2 = (g_nom * a2).sum(dim=1)   # [B]  g_nom . Delta_2
    viol1 = (dot1 < k1)              # [B] bool
    viol2 = (dot2 < k2)              # [B] bool

    a1_sq = (a1 ** 2).sum(dim=1) + eps   # [B]
    a2_sq = (a2 ** 2).sum(dim=1) + eps   # [B]

    g = g_nom.clone()

    # Case 2: only constraint-1 violated — project onto hyperplane g.a1 = k1
    mask_1only = viol1 & ~viol2
    if mask_1only.any():
        # g* = g_nom + lambda * a1,  lambda = (k1 - dot1) / ||a1||^2
        lam = (k1 - dot1) / a1_sq          # [B]
        g_proj = g_nom + lam[:, None] * a1
        g[mask_1only] = g_proj[mask_1only]

    # Case 3: only constraint-2 violated
    mask_2only = ~viol1 & viol2
    if mask_2only.any():
        lam = (k2 - dot2) / a2_sq
        g_proj = g_nom + lam[:, None] * a2
        g[mask_2only] = g_proj[mask_2only]

    # Case 4: both violated — solve 2-constraint KKT
    # g* = g_nom + lam1*a1 + lam2*a2
    # g*.a1 = k1  ->  dot1 + lam1*a1_sq + lam2*(a1.a2) = k1
    # g*.a2 = k2  ->  dot2 + lam1*(a1.a2) + lam2*a2_sq = k2
    mask_both = viol1 & viol2
    if mask_both.any():
        a12 = (a1 * a2).sum(dim=1)   # [B]  a1 . a2
        # 2x2 system per batch element:
        # [a1_sq  a12 ] [lam1]   [k1 - dot1]
        # [a12   a2_sq] [lam2] = [k2 - dot2]
        det = a1_sq * a2_sq - a12 ** 2 + eps
        rhs1 = k1 - dot1
        rhs2 = k2 - dot2
        lam1 = (a2_sq * rhs1 - a12 * rhs2) / det    # [B]
        lam2 = (a1_sq * rhs2 - a12 * rhs1) / det    # [B]
        g_proj = g_nom + lam1[:, None] * a1 + lam2[:, None] * a2
        g[mask_both] = g_proj[mask_both]

    return g


# ---------------------------------------------------------------------------
# Main composition function
# ---------------------------------------------------------------------------

def compose_trust_region(
    vel_c1: Vel,
    vel_c2: Vel,
    vel_uncond: Vel,
    t_frac: float,
    cfg: Optional[TrustRegionConfig] = None,
) -> Tuple[Vel, dict]:
    """
    Constrained QP composition.

    Parameters
    ----------
    vel_c1, vel_c2, vel_uncond : [B, C, H, W]
    t_frac : float in [0,1]   fraction of denoising complete
    cfg    : TrustRegionConfig

    Returns
    -------
    vf   : [B, C, H, W] composed velocity
    info : dict with diagnostics
    """
    if cfg is None:
        cfg = TrustRegionConfig()

    B, C, H, W = vel_c1.shape
    D = C * H * W

    d1, d2 = score_deltas(vel_c1, vel_c2, vel_uncond)
    n1, n2 = delta_norms(d1, d2)   # [B]

    # Nominal update (vanilla PoE)
    g_nom = (vel_uncond + cfg.guidance_scale * (d1 + d2)).reshape(B, D).float()

    # Constraint directions
    a1 = d1.reshape(B, D).float()
    a2 = d2.reshape(B, D).float()

    # Progress thresholds
    if cfg.threshold_schedule == "constant":
        sched = 1.0
    else:  # linear_increase
        sched = t_frac

    k1 = cfg.kappa_fraction * sched * (n1 ** 2)   # [B]
    k2 = cfg.kappa_fraction * sched * (n2 ** 2)   # [B]

    # Solve QP
    g_star = _solve_constrained_qp(g_nom, a1, a2, k1, k2, cfg.eps)

    # Check if feasible; if not, soften and re-solve
    dot1_star = (g_star * a1).sum(dim=1)
    dot2_star = (g_star * a2).sum(dim=1)
    infeasible = (dot1_star < k1 * cfg.infeasible_softening) | \
                 (dot2_star < k2 * cfg.infeasible_softening)

    if infeasible.any():
        k1_soft = k1 * cfg.infeasible_softening
        k2_soft = k2 * cfg.infeasible_softening
        g_soft = _solve_constrained_qp(g_nom, a1, a2, k1_soft, k2_soft, cfg.eps)
        g_star[infeasible] = g_soft[infeasible]

    vf = g_star.reshape(B, C, H, W).to(vel_c1.dtype)

    # Diagnostics
    progress_1 = (g_star * a1).sum(dim=1)   # [B]
    progress_2 = (g_star * a2).sum(dim=1)   # [B]

    info = {
        "progress_1": progress_1.detach(),
        "progress_2": progress_2.detach(),
        "threshold_1": k1.detach(),
        "threshold_2": k2.detach(),
        "infeasible_frac": infeasible.float().mean().detach(),
    }
    return vf, info


# ---------------------------------------------------------------------------
# Full denoising loop
# ---------------------------------------------------------------------------

def run_trust_region(
    latents: Vel,
    vel_fn,
    embeddings_c1,
    embeddings_c2,
    embeddings_uncond,
    scheduler,
    cfg: Optional[TrustRegionConfig] = None,
    cond_kwargs_c1: Optional[dict] = None,
    cond_kwargs_c2: Optional[dict] = None,
    cond_kwargs_uncond: Optional[dict] = None,
) -> Tuple[Vel, list]:
    """Full denoising loop using trust-region constrained composition."""
    if cfg is None:
        cfg = TrustRegionConfig()

    N = len(scheduler.timesteps)
    infos = []
    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma = scheduler.sigmas[i]
        t_frac = i / max(N - 1, 1)

        vel_c1 = vel_fn(latents, t, sigma, embeddings_c1, cond_kwargs_c1)
        vel_c2 = vel_fn(latents, t, sigma, embeddings_c2, cond_kwargs_c2)
        vel_uncond = vel_fn(latents, t, sigma, embeddings_uncond, cond_kwargs_uncond)

        vf, info = compose_trust_region(vel_c1, vel_c2, vel_uncond, t_frac, cfg)
        infos.append(info)

        latents = scheduler_step(scheduler, vf, t, latents, sigma, dsigma)

    return latents, infos
