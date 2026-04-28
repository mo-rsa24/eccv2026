"""
_base.py — Shared utilities for all PoE repair method implementations.

Every repair sampler in this package:
  - Accepts the same interface: (latents, vel_c1, vel_c2, vel_uncond, sigma, dsigma, t_idx, T)
  - Returns a composed velocity field vf of the same shape as the inputs
  - Operates on flow-matching velocity fields (not epsilon predictions)

Notation (consistent with the document):
  su   = vel_uncond           unconditional score proxy
  s1   = vel_c1               concept-1 conditional score proxy
  s2   = vel_c2               concept-2 conditional score proxy
  d1   = s1 - su = Delta_1    concept-1 score delta
  d2   = s2 - su = Delta_2    concept-2 score delta
  t_frac  in [0,1]            fractional timestep (0=pure noise, 1=clean)

All methods are inference-time only — no retraining required unless noted.
"""

from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Vel = torch.Tensor   # shape [B, C, H, W]


# ---------------------------------------------------------------------------
# Shared diagnostics
# ---------------------------------------------------------------------------

def score_deltas(vel_c1: Vel, vel_c2: Vel, vel_uncond: Vel) -> Tuple[Vel, Vel]:
    """Return (Delta_1, Delta_2) — per-concept score deltas."""
    return vel_c1 - vel_uncond, vel_c2 - vel_uncond


def cosine_similarity_deltas(d1: Vel, d2: Vel, eps: float = 1e-8) -> torch.Tensor:
    """
    cos(theta) between Delta_1 and Delta_2, per batch element.
    Shape: [B]
    """
    d1_flat = d1.flatten(1).float()
    d2_flat = d2.flatten(1).float()
    return F.cosine_similarity(d1_flat, d2_flat, dim=1, eps=eps)


def delta_norms(d1: Vel, d2: Vel) -> Tuple[torch.Tensor, torch.Tensor]:
    """L2 norms of deltas, per batch element. Shape: ([B], [B])."""
    n1 = d1.flatten(1).float().norm(dim=1)
    n2 = d2.flatten(1).float().norm(dim=1)
    return n1, n2


def dominance_ratio(d1: Vel, d2: Vel, eps: float = 1e-8) -> torch.Tensor:
    """R_t = ||Delta_1|| / (||Delta_2|| + eps). Shape: [B]."""
    n1, n2 = delta_norms(d1, d2)
    return n1 / (n2 + eps)


def x0_disagreement(x0_1: Vel, x0_2: Vel) -> torch.Tensor:
    """D_t = ||x0_hat_1 - x0_hat_2||_2 per batch element. Shape: [B]."""
    return (x0_1 - x0_2).flatten(1).float().norm(dim=1)


def predict_x0(latents: Vel, vel: Vel, sigma: torch.Tensor) -> Vel:
    """
    Tweedie / MMSE x0 estimate for flow-matching models.
    x0_hat = x_t - sigma * v_theta(x_t, t)
    (Derived from x_t = x0 + sigma * noise, v = (x_t - x0) / sigma => x0 = x_t - sigma*v)
    """
    return latents - sigma * vel


def attention_overlap_iou(A1: torch.Tensor, A2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Soft IoU between two spatial attention maps A1, A2.
    A1, A2: [..., H, W] — values in [0,1] (after softmax/normalization).
    Returns scalar IoU per batch element.
    """
    intersection = (A1 * A2).flatten(-2).sum(-1)
    union = (A1 + A2 - A1 * A2).flatten(-2).sum(-1)
    return intersection / (union + eps)


# ---------------------------------------------------------------------------
# Vanilla SuperDiff baseline (reference implementation)
# ---------------------------------------------------------------------------

def vanilla_superdiff(
    vel_c1: Vel,
    vel_c2: Vel,
    vel_uncond: Vel,
    kappa: torch.Tensor,     # [B] — Proposition 6 composition weight
    guidance_scale: float,
) -> Vel:
    """
    Baseline SuperDiff AND composition (Proposition 6).
    vf = su + gs * [(d2) + kappa * (d1 - d2)]
       = su + gs * [d2 + kappa*(vel_c1 - vel_c2)]
    """
    d1, d2 = score_deltas(vel_c1, vel_c2, vel_uncond)
    kappa_b = kappa[:, None, None, None]
    return vel_uncond + guidance_scale * (d2 + kappa_b * (d1 - d2))


# ---------------------------------------------------------------------------
# Scheduler-agnostic step
# ---------------------------------------------------------------------------

def scheduler_step(
    scheduler,
    model_output: Vel,
    t,
    latents: Vel,
    sigma: torch.Tensor,
    dsigma: torch.Tensor,
    noise_scale: float = 1.0,
) -> Vel:
    """
    Apply one denoising step in a scheduler-agnostic way.

    - For EulerDiscreteScheduler (SD1/2, epsilon prediction):
        uses scheduler.step(model_output, t, latents).prev_sample
    - For FlowMatchEulerDiscreteScheduler (SD3, velocity prediction):
        uses the manual SDE update:
        x = x + 2*dsigma*vf + noise_scale * sqrt(2*|dsigma|*sigma) * eps

    The `model_output` should be whatever the composition returns:
      - epsilon (noise) for SD1/2
      - velocity for SD3 flow-matching
    """
    cls_name = type(scheduler).__name__
    if "FlowMatch" in cls_name:
        # Flow-matching SDE step
        noise = torch.sqrt(2 * torch.abs(dsigma) * sigma) * torch.randn_like(latents)
        return latents + 2 * dsigma * model_output + noise_scale * noise
    else:
        # Standard diffusers scheduler step (handles epsilon/v-prediction, ODE)
        return scheduler.step(model_output, t, latents).prev_sample


# ---------------------------------------------------------------------------
# Common schedule helpers
# ---------------------------------------------------------------------------

def linear_schedule(t_frac: float, lo: float, hi: float) -> float:
    """Linearly interpolate from lo (t_frac=0, pure noise) to hi (t_frac=1, clean)."""
    return lo + t_frac * (hi - lo)


def cosine_schedule(t_frac: float, lo: float, hi: float) -> float:
    """Cosine ease-in from lo to hi."""
    import math
    return lo + (hi - lo) * (1 - math.cos(math.pi * t_frac)) / 2
