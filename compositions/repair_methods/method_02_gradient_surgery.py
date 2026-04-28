"""
Method 02 — Conflict-Aware Gradient Surgery (PCGrad-style Projection)
======================================================================

Theoretical grounding
---------------------
When the score deltas Delta_1 and Delta_2 point in conflicting directions
(Delta_1 . Delta_2 < 0), their sum pushes the latent along a direction
that helps one concept at the expense of the other.  This is a direct
analog of gradient conflict in multi-task learning.

PCGrad (Yu et al. 2020, NeurIPS) resolves gradient conflict by projecting
each gradient onto the normal plane of the other when they conflict:

    If Delta_1 . Delta_2 < 0:
        Delta_1 <- Delta_1 - (Delta_1 . Delta_2 / ||Delta_2||^2) * Delta_2
        Delta_2 <- Delta_2 - (Delta_2 . Delta_1 / ||Delta_1||^2) * Delta_1

After projection the composed velocity is:
    vf = su + w1 * Delta_1_proj + w2 * Delta_2_proj

The method can be applied globally (on the full latent vector) or
patchwise (per spatial patch in the latent grid) for more local control.

Why this helps
--------------
- Group 3 (entanglement): Prevents one expert from undoing the other in
  shared directions, reducing partial suppression.
- Group 4 (slot competition): Reduces the dominant expert from erasing
  evidence of the other (though discrete slot assignment needs masking too).

References
----------
- Yu et al. 2020 "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)
- The document §"Conflict-aware gradient surgery: projection / orthogonalisation"
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch

from ._base import Vel, score_deltas, cosine_similarity_deltas, delta_norms, scheduler_step


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GradientSurgeryConfig:
    # Guidance scale applied to projected deltas
    guidance_scale: float = 7.5

    # Whether to project only when there is conflict (dot < 0)
    # If False, always orthogonalise (stronger but may be too conservative)
    only_on_conflict: bool = True

    # Apply surgery patchwise (per latent spatial position)?
    # Patchwise is more targeted but slower.
    patchwise: bool = False

    # After projection, optionally re-scale each delta to its original norm
    # (preserves magnitude, only changes direction)
    renormalize: bool = False

    # Additional damping factor applied when conflict is detected
    # vf components are multiplied by (1 - conflict_damp) when conflicting
    conflict_damp: float = 0.0   # 0.0 = no extra damping beyond projection

    eps: float = 1e-8


# ---------------------------------------------------------------------------
# Core projection
# ---------------------------------------------------------------------------

def _project_pcgrad(
    d1_flat: torch.Tensor,   # [B, D]
    d2_flat: torch.Tensor,   # [B, D]
    cfg: GradientSurgeryConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PCGrad projection in flat (batch, spatial) space.

    Returns (d1_proj, d2_proj, conflict_mask) where conflict_mask is [B] bool.
    """
    dot = (d1_flat * d2_flat).sum(dim=1, keepdim=True)        # [B, 1]
    n2_sq = (d2_flat ** 2).sum(dim=1, keepdim=True) + cfg.eps # [B, 1]
    n1_sq = (d1_flat ** 2).sum(dim=1, keepdim=True) + cfg.eps # [B, 1]

    conflict_mask = (dot.squeeze(1) < 0)                       # [B]

    if cfg.only_on_conflict:
        # Only project conflicting batch elements
        proj_1_onto_2 = (dot / n2_sq) * d2_flat               # [B, D]
        proj_2_onto_1 = (dot / n1_sq) * d1_flat               # [B, D]

        # For conflicting elements: subtract projection; for others: keep
        mask_b = conflict_mask[:, None].float()                # [B, 1]
        d1_proj = d1_flat - mask_b * proj_1_onto_2
        d2_proj = d2_flat - mask_b * proj_2_onto_1
    else:
        # Always project (orthogonalise)
        d1_proj = d1_flat - (dot / n2_sq) * d2_flat
        d2_proj = d2_flat - (dot / n1_sq) * d1_flat

    # Optional renormalisation: restore original norms
    if cfg.renormalize:
        n1_orig = d1_flat.norm(dim=1, keepdim=True)
        n2_orig = d2_flat.norm(dim=1, keepdim=True)
        d1_proj = d1_proj / (d1_proj.norm(dim=1, keepdim=True) + cfg.eps) * n1_orig
        d2_proj = d2_proj / (d2_proj.norm(dim=1, keepdim=True) + cfg.eps) * n2_orig

    return d1_proj, d2_proj, conflict_mask


def _apply_conflict_damp(
    d1_proj: torch.Tensor,
    d2_proj: torch.Tensor,
    conflict_mask: torch.Tensor,
    damp: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply additional scalar damping to conflicting batch elements."""
    if damp == 0.0:
        return d1_proj, d2_proj
    mask_b = conflict_mask[:, None].float()
    factor = 1.0 - damp * mask_b
    return d1_proj * factor, d2_proj * factor


# ---------------------------------------------------------------------------
# Global (full-latent) surgery
# ---------------------------------------------------------------------------

def _global_surgery(
    d1: Vel,
    d2: Vel,
    cfg: GradientSurgeryConfig,
) -> Tuple[Vel, Vel, torch.Tensor]:
    """Apply PCGrad to d1, d2 treated as flat vectors per batch element."""
    B, C, H, W = d1.shape
    d1_flat = d1.reshape(B, -1).float()
    d2_flat = d2.reshape(B, -1).float()

    d1_proj, d2_proj, conflict_mask = _project_pcgrad(d1_flat, d2_flat, cfg)
    d1_proj, d2_proj = _apply_conflict_damp(d1_proj, d2_proj, conflict_mask, cfg.conflict_damp)

    return (
        d1_proj.reshape(B, C, H, W).to(d1.dtype),
        d2_proj.reshape(B, C, H, W).to(d2.dtype),
        conflict_mask,
    )


# ---------------------------------------------------------------------------
# Patchwise surgery
# ---------------------------------------------------------------------------

def _patchwise_surgery(
    d1: Vel,
    d2: Vel,
    cfg: GradientSurgeryConfig,
) -> Tuple[Vel, Vel, torch.Tensor]:
    """
    Apply PCGrad independently per spatial position (H*W patches).

    Operates on each [B, C] vector at each spatial location.
    Returns projected d1, d2 of same shape; conflict_mask [B, H, W].
    """
    B, C, H, W = d1.shape
    # Reshape to [B*H*W, C] for batched projection
    d1_hw = d1.permute(0, 2, 3, 1).reshape(B * H * W, C).float()
    d2_hw = d2.permute(0, 2, 3, 1).reshape(B * H * W, C).float()

    d1_proj, d2_proj, conflict_mask_hw = _project_pcgrad(d1_hw, d2_hw, cfg)
    d1_proj, d2_proj = _apply_conflict_damp(
        d1_proj, d2_proj, conflict_mask_hw, cfg.conflict_damp
    )

    d1_out = d1_proj.reshape(B, H, W, C).permute(0, 3, 1, 2).to(d1.dtype)
    d2_out = d2_proj.reshape(B, H, W, C).permute(0, 3, 1, 2).to(d2.dtype)
    conflict_mask_out = conflict_mask_hw.reshape(B, H, W)  # [B, H, W]

    # Summarise to [B]: fraction of conflicting patches
    conflict_summary = conflict_mask_out.float().mean(dim=(1, 2))  # [B]
    return d1_out, d2_out, conflict_summary


# ---------------------------------------------------------------------------
# Main composition function
# ---------------------------------------------------------------------------

def compose_gradient_surgery(
    vel_c1: Vel,
    vel_c2: Vel,
    vel_uncond: Vel,
    cfg: Optional[GradientSurgeryConfig] = None,
) -> Tuple[Vel, dict]:
    """
    Compute composed velocity with PCGrad-style gradient surgery.

    Returns
    -------
    vf   : [B, C, H, W] composed velocity
    info : dict with diagnostics
    """
    if cfg is None:
        cfg = GradientSurgeryConfig()

    d1, d2 = score_deltas(vel_c1, vel_c2, vel_uncond)

    cos_before = cosine_similarity_deltas(d1, d2)

    if cfg.patchwise:
        d1_proj, d2_proj, conflict_frac = _patchwise_surgery(d1, d2, cfg)
    else:
        d1_proj, d2_proj, conflict_mask = _global_surgery(d1, d2, cfg)
        conflict_frac = conflict_mask.float()

    cos_after = cosine_similarity_deltas(d1_proj, d2_proj)

    vf = vel_uncond + cfg.guidance_scale * (d1_proj + d2_proj)

    info = {
        "cos_before": cos_before.detach(),
        "cos_after": cos_after.detach(),
        "conflict_frac": conflict_frac.detach(),
    }
    return vf, info


# ---------------------------------------------------------------------------
# Full denoising loop
# ---------------------------------------------------------------------------

def run_gradient_surgery(
    latents: Vel,
    vel_fn,
    embeddings_c1,
    embeddings_c2,
    embeddings_uncond,
    scheduler,
    cfg: Optional[GradientSurgeryConfig] = None,
    cond_kwargs_c1: Optional[dict] = None,
    cond_kwargs_c2: Optional[dict] = None,
    cond_kwargs_uncond: Optional[dict] = None,
) -> Tuple[Vel, list]:
    """Full denoising loop using gradient surgery composition."""
    if cfg is None:
        cfg = GradientSurgeryConfig()

    infos = []
    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma = scheduler.sigmas[i]

        vel_c1 = vel_fn(latents, t, sigma, embeddings_c1, cond_kwargs_c1)
        vel_c2 = vel_fn(latents, t, sigma, embeddings_c2, cond_kwargs_c2)
        vel_uncond = vel_fn(latents, t, sigma, embeddings_uncond, cond_kwargs_uncond)

        vf, info = compose_gradient_surgery(vel_c1, vel_c2, vel_uncond, cfg)
        infos.append(info)

        latents = scheduler_step(scheduler, vf, t, latents, sigma, dsigma)

    return latents, infos
