"""
Method 05 — Residual Interaction-Term Estimator (r̂_t)
======================================================

Theoretical grounding
---------------------
The exact interaction term missing from PoE is:

    r_t(x_t, c1, c2) = ∇_{x_t} log p_t(x_t | c1, c2) - [s1 + s2 - su]

This cannot be accessed at test time without a joint-conditioned model.
Instead, we *learn* a lightweight approximation r̂_t that predicts the
correction from signals already available during inference:

    r̂_t = f_θ([Delta_1, Delta_2, Delta_1 ⊙ Delta_2, cos(θ)], t)

Crucially:
  - f_θ does NOT consume any joint prompt text embedding
  - Trained on a broad prompt distribution (pair-agnostic)
  - At test time only receives expert-visible signals

Architecture: small MLP operating on *statistics* of the score deltas
(not the full latent, which would be prohibitively large).  The output is
an *additive correction in the latent-delta space*, meaning we predict
the correction direction and magnitude, then add it to the composed velocity.

Alternative (full-latent) version uses a lightweight U-Net correction at
cost of significantly more compute; the statistics-based MLP is preferred.

Training (offline, not done here)
----------------------------------
The target r̂_t is computed as:
    r_target = s_joint - (s1 + s2 - su)
where s_joint is obtained from a joint-prompt forward pass on training pairs.
Training loss: MSE(r̂_t(features), r_target).

At test time only the MLP inference is needed (no joint model).

Risk
----
If the MLP is trained on joint-prompt data keyed by semantic pair type,
it can silently encode a joint prior.  Keep it pair-agnostic and evaluate
with the group-leakage test described in the ablation plan.

References
----------
- The document §"Residual interaction-term estimation as a lightweight corrector"
- r̂_t = f_θ([Delta_1, Delta_2, Delta_1 ⊙ Delta_2, cos θ], t)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Vel, score_deltas, cosine_similarity_deltas, delta_norms, scheduler_step


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ResidualAdapterConfig:
    # Guidance scale for baseline composition
    guidance_scale: float = 7.5

    # Scaling factor for the residual correction (tunable at test time)
    residual_scale: float = 1.0

    # MLP hidden dimensions
    hidden_dims: Tuple[int, ...] = (256, 256, 128)

    # Dropout rate (for MC-dropout diversity, matching inverter design)
    dropout: float = 0.1

    # Whether the adapter is active (False = vanilla PoE, for ablation)
    active: bool = True


# ---------------------------------------------------------------------------
# Statistics feature extraction
# ---------------------------------------------------------------------------

def _extract_features(
    d1: Vel,
    d2: Vel,
    t_frac: float,
) -> torch.Tensor:
    """
    Extract scalar/low-dimensional statistics from score deltas.

    Features (per batch element):
      - ||Delta_1||, ||Delta_2||                     (2 scalars)
      - cos(theta) = Delta_1 . Delta_2 / (||d1|| ||d2||)  (1)
      - ||Delta_1 - Delta_2|| / (||Delta_1|| + ||Delta_2|| + eps) (1, normalised difference)
      - ||Delta_1 + Delta_2|| / (||Delta_1|| + ||Delta_2|| + eps) (1, normalised sum)
      - ||Delta_1 ⊙ Delta_2|| (Hadamard product norm)             (1)
      - t_frac                                                     (1)

    Total feature dim: 7 per batch element.

    For a full-latent adapter, the features would be flattened deltas —
    prohibitively large for typical latent sizes (4 x 64 x 64 = 16384).
    Statistics are fast and generalisable.

    Returns: [B, 7]
    """
    B = d1.shape[0]
    eps = 1e-8

    d1_flat = d1.reshape(B, -1).float()
    d2_flat = d2.reshape(B, -1).float()

    n1 = d1_flat.norm(dim=1)            # [B]
    n2 = d2_flat.norm(dim=1)            # [B]
    n_sum = n1 + n2 + eps

    cos_theta = cosine_similarity_deltas(d1, d2)   # [B]

    diff_norm = (d1_flat - d2_flat).norm(dim=1) / n_sum
    sum_norm  = (d1_flat + d2_flat).norm(dim=1) / n_sum
    hadamard_norm = (d1_flat * d2_flat).norm(dim=1)

    t_vec = torch.full((B,), t_frac, device=d1.device, dtype=torch.float32)

    features = torch.stack([
        n1, n2, cos_theta, diff_norm, sum_norm, hadamard_norm, t_vec
    ], dim=1)   # [B, 7]

    return features


# ---------------------------------------------------------------------------
# MLP architecture
# ---------------------------------------------------------------------------

class ResidualAdapterMLP(nn.Module):
    """
    Lightweight MLP that maps scalar features of the score deltas to a
    *correction vector* in the score-delta statistics space.

    Input:  [B, 7]  (statistics of Delta_1, Delta_2 + t_frac)
    Output: [B, 4]  (correction coefficients for a_1, a_2, a_12, a_0)

    The correction is then expanded back to latent space as:
        r̂ = coeff_1 * d1 + coeff_2 * d2
            + coeff_12 * (d1 ⊙ d2) / (||d1 ⊙ d2|| + eps)
            + coeff_0 * (d1 + d2) / (||d1 + d2|| + eps)

    This keeps the correction in the span of the available directions,
    without requiring the MLP to directly output a 16384-dimensional vector.
    """

    FEATURE_DIM = 7
    OUTPUT_DIM = 4    # coefficients for [d1, d2, hadamard, sum]

    def __init__(self, cfg: ResidualAdapterConfig):
        super().__init__()
        dims = [self.FEATURE_DIM] + list(cfg.hidden_dims) + [self.OUTPUT_DIM]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
                layers.append(nn.Dropout(cfg.dropout))
        self.net = nn.Sequential(*layers)

        # Zero-init output layer so the residual starts as identity (no-op)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, 7] -> coefficients: [B, 4]"""
        return self.net(features)


# ---------------------------------------------------------------------------
# Main composition function
# ---------------------------------------------------------------------------

def compose_residual_adapter(
    vel_c1: Vel,
    vel_c2: Vel,
    vel_uncond: Vel,
    t_frac: float,
    adapter: Optional[ResidualAdapterMLP] = None,
    cfg: Optional[ResidualAdapterConfig] = None,
) -> Tuple[Vel, dict]:
    """
    Residual-corrected PoE composition.

    Parameters
    ----------
    vel_c1, vel_c2, vel_uncond : [B, C, H, W]
    t_frac  : float in [0, 1]
    adapter : ResidualAdapterMLP (None = use zero residual, i.e. vanilla PoE)
    cfg     : ResidualAdapterConfig

    Returns
    -------
    vf   : [B, C, H, W] composed velocity
    info : dict with diagnostics
    """
    if cfg is None:
        cfg = ResidualAdapterConfig()

    B, C, H, W = vel_c1.shape
    D = C * H * W
    eps = 1e-8

    d1, d2 = score_deltas(vel_c1, vel_c2, vel_uncond)

    # Vanilla PoE baseline
    vf_baseline = vel_uncond + cfg.guidance_scale * (d1 + d2)

    if adapter is None or not cfg.active:
        return vf_baseline, {"residual_norm": torch.zeros(B, device=d1.device)}

    # Feature extraction
    features = _extract_features(d1, d2, t_frac)   # [B, 7]

    # MLP forward
    adapter.eval()
    with torch.no_grad():
        coeffs = adapter(features)                  # [B, 4]

    # Expand coefficients to latent-space correction
    d1_flat = d1.reshape(B, -1).float()
    d2_flat = d2.reshape(B, -1).float()

    hadamard = d1_flat * d2_flat
    hadamard = hadamard / (hadamard.norm(dim=1, keepdim=True) + eps)

    d_sum_flat = d1_flat + d2_flat
    d_sum_flat = d_sum_flat / (d_sum_flat.norm(dim=1, keepdim=True) + eps)

    # r̂ = c1*d1 + c2*d2 + c12*hadamard + c0*d_sum
    c1  = coeffs[:, 0:1]   # [B, 1]
    c2  = coeffs[:, 1:2]
    c12 = coeffs[:, 2:3]
    c0  = coeffs[:, 3:4]

    residual_flat = (
        c1  * d1_flat  +
        c2  * d2_flat  +
        c12 * hadamard +
        c0  * d_sum_flat
    )  # [B, D]

    residual = residual_flat.reshape(B, C, H, W).to(d1.dtype)
    residual_norm = residual_flat.norm(dim=1)   # [B]

    vf = vf_baseline + cfg.residual_scale * residual

    info = {
        "residual_norm": residual_norm.detach(),
        "coeffs": coeffs.detach(),
    }
    return vf, info


# ---------------------------------------------------------------------------
# Training utility: compute residual targets from joint-prompt forward pass
# (offline — not used at test time)
# ---------------------------------------------------------------------------

def compute_residual_targets(
    vel_c1: Vel,
    vel_c2: Vel,
    vel_uncond: Vel,
    vel_joint: Vel,   # from a joint-prompt forward pass (training only)
) -> Vel:
    """
    Compute r_target = s_joint - (s1 + s2 - su).

    Used only during offline training of the adapter MLP.
    vel_joint must come from a SEPARATE single-shot "c1 and c2" forward pass
    (training data generation, not test time).
    """
    d1, d2 = score_deltas(vel_c1, vel_c2, vel_uncond)
    poe_score = vel_uncond + d1 + d2
    return vel_joint - poe_score


# ---------------------------------------------------------------------------
# Full denoising loop
# ---------------------------------------------------------------------------

def run_residual_adapter(
    latents: Vel,
    vel_fn,
    embeddings_c1,
    embeddings_c2,
    embeddings_uncond,
    scheduler,
    adapter: Optional[ResidualAdapterMLP] = None,
    cfg: Optional[ResidualAdapterConfig] = None,
    cond_kwargs_c1: Optional[dict] = None,
    cond_kwargs_c2: Optional[dict] = None,
    cond_kwargs_uncond: Optional[dict] = None,
) -> Tuple[Vel, list]:
    """Full denoising loop with optional residual adapter correction."""
    if cfg is None:
        cfg = ResidualAdapterConfig()

    N = len(scheduler.timesteps)
    infos = []
    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma = scheduler.sigmas[i]
        t_frac = i / max(N - 1, 1)

        vel_c1 = vel_fn(latents, t, sigma, embeddings_c1, cond_kwargs_c1)
        vel_c2 = vel_fn(latents, t, sigma, embeddings_c2, cond_kwargs_c2)
        vel_uncond = vel_fn(latents, t, sigma, embeddings_uncond, cond_kwargs_uncond)

        vf, info = compose_residual_adapter(
            vel_c1, vel_c2, vel_uncond, t_frac, adapter, cfg
        )
        infos.append(info)

        latents = scheduler_step(scheduler, vf, t, latents, sigma, dsigma)

    return latents, infos
