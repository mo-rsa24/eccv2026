"""
Method 09 — Energy-Based Corrections from Auxiliary Failure Probes
==================================================================

Theoretical grounding
---------------------
When PoE fails, it fails in characteristic ways:
  - Suppression: one concept is absent (entity missing)
  - Dominance: one concept overwhelms the other
  - Hybridisation: the two concepts merge into a chimera object

Rather than modifying the score composition rule, this method adds an
explicit "failure energy" E_failure(x_t, t) whose gradient steers the
latent away from these failure modes:

    vf = su + Delta_1 + Delta_2 - lambda(t) * ∇_{x_t} E_failure(x_t, t)

E_failure is a sum of penalty terms:
  1. Suppression penalty: -log(||Delta_i|| / max_norm)  if ||Delta_i|| is low
  2. Dominance penalty:   (||Delta_1|| - ||Delta_2||)^2 / (||Delta_1|| + ||Delta_2||)^2
  3. Hybrid penalty:      max(0, cos(Delta_1, Delta_2))  (positive cosine = same direction = hybrid)
  4. Attention overlap:   IoU(M1, M2)  (if masks available)

The gradient ∇_{x_t} E_failure is computed via torch.autograd.grad,
which requires the score network to be differentiable with respect to x_t.
This is standard for guidance (DPS-style), but requires enabling gradients
for the UNet/transformer forward pass.

When autograd is unavailable, we provide a finite-difference approximation.

Risk
----
The energy terms are designed to detect generic failure modes rather than
encode a specific joint semantic configuration.  The dominance and hybrid
penalties are "negative constraints" (what to avoid), not "positive targets"
(what to achieve), which keeps them closer to logical correction than
semantic composition.

References
----------
- Chung et al. 2022 "Diffusion Posterior Sampling" (arXiv:2209.14687)
  — DPS-style guidance via energy gradients through the denoising network
- The document §"Energy-based corrections from auxiliary probes"
- Li et al. 2024 "Attention Overlap" (arXiv:2410.20972) — overlap penalty
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from ._base import (
    Vel, score_deltas, cosine_similarity_deltas,
    delta_norms, dominance_ratio, attention_overlap_iou, scheduler_step,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ProbeEnergyConfig:
    # Guidance scale for baseline composition
    guidance_scale: float = 7.5

    # Strength of failure energy gradient
    lambda_energy: float = 0.5

    # Schedule for lambda: "constant", "linear_increase" (ramp up over time)
    lambda_schedule: str = "linear_increase"

    # Which failure penalties to include
    use_suppression_penalty: bool = True
    use_dominance_penalty:   bool = True
    use_hybrid_penalty:      bool = True
    use_overlap_penalty:     bool = True   # requires attention masks

    # Suppression penalty: fire when ||Delta_i|| < suppression_threshold * max_norm
    suppression_threshold: float = 0.2

    # Gradient computation method: "autograd" or "finite_diff"
    grad_method: str = "finite_diff"
    fd_eps: float = 0.01     # finite-difference epsilon

    eps: float = 1e-8


# ---------------------------------------------------------------------------
# Energy terms (analytical, from delta statistics)
# ---------------------------------------------------------------------------

def _suppression_energy(
    d1: Vel,
    d2: Vel,
    cfg: ProbeEnergyConfig,
) -> torch.Tensor:
    """
    Suppression energy: penalise when either concept delta is very weak.
    E_supp = max(0, thresh * max_n - n1) + max(0, thresh * max_n - n2)
    Shape: [B]
    """
    n1, n2 = delta_norms(d1, d2)
    max_n = torch.maximum(n1, n2)
    thresh = cfg.suppression_threshold * max_n
    E = F.relu(thresh - n1) + F.relu(thresh - n2)
    return E


def _dominance_energy(
    d1: Vel,
    d2: Vel,
    cfg: ProbeEnergyConfig,
) -> torch.Tensor:
    """
    Dominance energy: penalise imbalance between concept norms.
    E_dom = (n1 - n2)^2 / (n1 + n2 + eps)^2
    Shape: [B]
    """
    n1, n2 = delta_norms(d1, d2)
    E = ((n1 - n2) ** 2) / ((n1 + n2 + cfg.eps) ** 2)
    return E


def _hybrid_energy(
    d1: Vel,
    d2: Vel,
    cfg: ProbeEnergyConfig,
) -> torch.Tensor:
    """
    Hybrid energy: penalise positive cosine similarity (same direction = chimera).
    E_hyb = max(0, cos(Delta_1, Delta_2))
    Shape: [B]
    """
    cos = cosine_similarity_deltas(d1, d2)
    return F.relu(cos)


def _overlap_energy(
    M1: torch.Tensor,
    M2: torch.Tensor,
) -> torch.Tensor:
    """
    Attention overlap energy: IoU between spatial masks.
    Shape: [B]
    """
    return attention_overlap_iou(M1, M2)


def compute_failure_energy(
    latents: Vel,
    d1: Vel,
    d2: Vel,
    cfg: ProbeEnergyConfig,
    M1: Optional[torch.Tensor] = None,
    M2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute total failure energy E_failure(x_t) -> [B] (sum over active terms).
    """
    E = torch.zeros(latents.shape[0], device=latents.device, dtype=torch.float32)

    if cfg.use_suppression_penalty:
        E = E + _suppression_energy(d1, d2, cfg)

    if cfg.use_dominance_penalty:
        E = E + _dominance_energy(d1, d2, cfg)

    if cfg.use_hybrid_penalty:
        E = E + _hybrid_energy(d1, d2, cfg)

    if cfg.use_overlap_penalty and M1 is not None and M2 is not None:
        E = E + _overlap_energy(M1, M2)

    return E   # [B]


# ---------------------------------------------------------------------------
# Gradient of E_failure w.r.t. x_t
# ---------------------------------------------------------------------------

def _energy_grad_finite_diff(
    latents: Vel,
    vel_fn_for_energy,    # callable(x) -> (d1, d2) using fixed t/sigma/embeddings
    cfg: ProbeEnergyConfig,
    M1: Optional[torch.Tensor] = None,
    M2: Optional[torch.Tensor] = None,
) -> Vel:
    """
    Finite-difference approximation of ∇_{x_t} E_failure.

    For each spatial location, estimate gradient by adding a small perturbation
    in a random direction and computing the energy difference.

    Uses a single random direction (Rademacher estimator) for efficiency.
    """
    eps = cfg.fd_eps
    perturb = torch.sign(torch.randn_like(latents))   # Rademacher ±1

    x_plus  = latents + eps * perturb
    x_minus = latents - eps * perturb

    d1_p, d2_p = vel_fn_for_energy(x_plus)
    d1_m, d2_m = vel_fn_for_energy(x_minus)

    E_plus  = compute_failure_energy(x_plus,  d1_p, d2_p, cfg, M1, M2).mean()
    E_minus = compute_failure_energy(x_minus, d1_m, d2_m, cfg, M1, M2).mean()

    # dE/dx ~ (E_plus - E_minus) / (2*eps) * perturb
    grad = ((E_plus - E_minus) / (2 * eps)) * perturb
    return grad


def _energy_grad_autograd(
    latents: Vel,
    vel_fn_for_energy,
    cfg: ProbeEnergyConfig,
    M1: Optional[torch.Tensor] = None,
    M2: Optional[torch.Tensor] = None,
) -> Vel:
    """
    Exact gradient via torch.autograd.
    Requires the velocity network to be differentiable w.r.t. x_t.
    """
    x = latents.detach().requires_grad_(True)
    d1, d2 = vel_fn_for_energy(x)
    E = compute_failure_energy(x, d1, d2, cfg, M1, M2).mean()
    grad = torch.autograd.grad(E, x)[0]
    return grad.detach()


# ---------------------------------------------------------------------------
# Main composition function
# ---------------------------------------------------------------------------

def compose_probe_energy(
    vel_c1: Vel,
    vel_c2: Vel,
    vel_uncond: Vel,
    latents: Vel,
    t_frac: float,
    vel_fn_for_energy=None,    # callable(x) -> (d1, d2); required for grad
    cfg: Optional[ProbeEnergyConfig] = None,
    M1: Optional[torch.Tensor] = None,
    M2: Optional[torch.Tensor] = None,
) -> Tuple[Vel, dict]:
    """
    Energy-probe-corrected composition.

    Parameters
    ----------
    vel_c1, vel_c2, vel_uncond : [B, C, H, W]
    latents  : [B, C, H, W] current x_t
    t_frac   : float in [0, 1]
    vel_fn_for_energy : callable x -> (d1, d2) for gradient computation
    cfg      : ProbeEnergyConfig
    M1, M2   : optional [B, H, W] attention masks for overlap penalty

    Returns
    -------
    vf   : [B, C, H, W]
    info : dict
    """
    if cfg is None:
        cfg = ProbeEnergyConfig()

    d1, d2 = score_deltas(vel_c1, vel_c2, vel_uncond)

    # Baseline PoE velocity
    vf_baseline = vel_uncond + cfg.guidance_scale * (d1 + d2)

    # Current failure energy (for logging)
    E_current = compute_failure_energy(latents, d1, d2, cfg, M1, M2)

    # Lambda schedule
    if cfg.lambda_schedule == "linear_increase":
        lam = cfg.lambda_energy * t_frac
    else:
        lam = cfg.lambda_energy

    # Compute energy gradient
    E_grad = None
    if lam > 0 and vel_fn_for_energy is not None:
        if cfg.grad_method == "autograd":
            E_grad = _energy_grad_autograd(latents, vel_fn_for_energy, cfg, M1, M2)
        else:  # finite_diff
            E_grad = _energy_grad_finite_diff(latents, vel_fn_for_energy, cfg, M1, M2)

    if E_grad is not None:
        vf = vf_baseline - lam * E_grad
    else:
        vf = vf_baseline

    info = {
        "E_failure": E_current.detach(),
        "lambda_used": lam,
        "E_grad_norm": E_grad.flatten(1).norm(dim=1).detach() if E_grad is not None
                       else torch.zeros(latents.shape[0], device=latents.device),
    }
    return vf, info


# ---------------------------------------------------------------------------
# Full denoising loop
# ---------------------------------------------------------------------------

def run_probe_energy(
    latents: Vel,
    vel_fn,
    embeddings_c1,
    embeddings_c2,
    embeddings_uncond,
    scheduler,
    cfg: Optional[ProbeEnergyConfig] = None,
    cond_kwargs_c1: Optional[dict] = None,
    cond_kwargs_c2: Optional[dict] = None,
    cond_kwargs_uncond: Optional[dict] = None,
) -> Tuple[Vel, list]:
    """Full denoising loop with probe-energy failure correction."""
    if cfg is None:
        cfg = ProbeEnergyConfig()

    N = len(scheduler.timesteps)
    infos = []

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma  = scheduler.sigmas[i]
        t_frac = i / max(N - 1, 1)

        vel_c1    = vel_fn(latents, t, sigma, embeddings_c1, cond_kwargs_c1)
        vel_c2    = vel_fn(latents, t, sigma, embeddings_c2, cond_kwargs_c2)
        vel_uncond = vel_fn(latents, t, sigma, embeddings_uncond, cond_kwargs_uncond)

        # Build a closure over current t/sigma for FD gradient
        def _vel_fn_closure(x_in):
            v1 = vel_fn(x_in, t, sigma, embeddings_c1, cond_kwargs_c1)
            v2 = vel_fn(x_in, t, sigma, embeddings_c2, cond_kwargs_c2)
            vu = vel_fn(x_in, t, sigma, embeddings_uncond, cond_kwargs_uncond)
            return score_deltas(v1, v2, vu)

        vf, info = compose_probe_energy(
            vel_c1, vel_c2, vel_uncond,
            latents, t_frac,
            vel_fn_for_energy=_vel_fn_closure,
            cfg=cfg,
        )
        infos.append(info)

        latents = scheduler_step(scheduler, vf, t, latents, sigma, dsigma)

    return latents, infos
