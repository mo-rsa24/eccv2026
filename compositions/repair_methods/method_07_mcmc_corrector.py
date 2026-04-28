"""
Method 07 — Particle-Based MCMC / Annealed Corrector
=====================================================

Theoretical grounding
---------------------
The fundamental issue exposed by Du et al. (2023) is that reverse diffusion
with the summed composed score does NOT sample from the composed noisy
marginal:

    q̃_{prod,t}(x_t)  ≠  Π_i q̃_{i,t}(x_t)

A single denoising trajectory is therefore biased when the target is a
product/mixture composition.  MCMC correctors after each predictor step
can reduce this bias by taking a few Langevin/HMC steps targeting an
approximation of the correct composed distribution at that noise level.

This implementation provides two complementary strategies:

  Strategy A — Annealed Langevin corrector (after each predictor step):
      For K_corr steps at noise level sigma:
          x <- x + step_size * s_comp(x, t) + sqrt(2 * step_size) * noise
      where s_comp = su + w1*d1 + w2*d2.
      This is standard MALA/overdamped Langevin targeting the composed
      density at the current noise level.

  Strategy B — Sequential Monte Carlo (SMC) over diffusion paths:
      Maintain P particles.  At each timestep:
        1. Propagate each particle via the standard composed predictor step.
        2. Compute importance weights from a composition-specific
           log-density estimate (using the Itô estimator from SuperDiff's
           Theorem 1, which the codebase already tracks).
        3. Resample particles by effective sample size (ESS) threshold.
      This is the reverse-diffusion SMC of Wu et al. (2025).

Strategy A is cheaper (no particle duplication) but less principled.
Strategy B is more principled but uses P times more memory and compute.

References
----------
- Du et al. 2023 "Reduce, Reuse, Recycle" (ICML 2023) §3–4
- Wu et al. 2025 "Reverse Diffusion Sequential Monte Carlo Samplers"
  (NeurIPS 2025)
- The document §"Particle-based correction during denoising"
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch

from ._base import Vel, score_deltas, scheduler_step


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MCMCCorrectorConfig:
    # Guidance scale
    guidance_scale: float = 7.5

    # Strategy: "langevin" (A) or "smc" (B)
    strategy: str = "langevin"

    # --- Langevin (Strategy A) ---
    # Number of Langevin corrector steps per denoising step
    n_corrector_steps: int = 3
    # Step size for Langevin corrector (as fraction of dsigma magnitude)
    langevin_step_scale: float = 0.1
    # Only run corrector at timesteps where t_frac > this threshold
    # (corrector is most useful in mid-to-late denoising)
    corrector_start_frac: float = 0.2

    # --- SMC (Strategy B) ---
    # Number of particles (copies of the latent)
    n_particles: int = 4
    # Resample when ESS < ess_threshold * n_particles
    ess_threshold: float = 0.5
    # How many Langevin steps to run per SMC resampling event
    smc_mh_steps: int = 2


# ---------------------------------------------------------------------------
# Log-density approximation
# ---------------------------------------------------------------------------

def _log_density_estimate(
    latents: Vel,
    vel_c1: Vel,
    vel_c2: Vel,
    vel_uncond: Vel,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Approximate log p_comp(x_t) using the Itô density estimator.

    Based on SuperDiff Theorem 1 (Eq. 13), the log-likelihood update is:
        d log q^i ≈ -|dσ|/σ · ||v^i||^2  (leading term, ignoring path integral)

    Here we use the composed score magnitude as a proxy for the composed
    log-density at the current state.  This is a rough approximation
    but allows weight computation without access to true likelihoods.

    Returns [B] approximate log-weight (unnormalised).
    """
    d1, d2 = score_deltas(vel_c1, vel_c2, vel_uncond)
    s_comp = vel_uncond + d1 + d2   # composed score

    # Log-density proxy: negative squared norm of composed score delta
    # (lower ‖s_comp - su‖ -> less extreme guidance -> more stable composition)
    delta_comp = s_comp - vel_uncond
    log_w = -(delta_comp ** 2).flatten(1).sum(1) / (2 * sigma.clamp(min=1e-4))
    return log_w   # [B]


# ---------------------------------------------------------------------------
# Strategy A: Annealed Langevin corrector
# ---------------------------------------------------------------------------

def _langevin_corrector(
    latents: Vel,
    vel_fn,
    embeddings_c1,
    embeddings_c2,
    embeddings_uncond,
    t,
    sigma: torch.Tensor,
    cfg: MCMCCorrectorConfig,
    cond_kwargs_c1,
    cond_kwargs_c2,
    cond_kwargs_uncond,
) -> Vel:
    """
    Run K Langevin steps targeting composed density at current sigma.

    Each step:
        x <- x + step_size * s_comp(x, t)  +  sqrt(2*step_size) * eps
    """
    step_size = cfg.langevin_step_scale * sigma.abs().item()

    for _ in range(cfg.n_corrector_steps):
        vel_c1 = vel_fn(latents, t, sigma, embeddings_c1, cond_kwargs_c1)
        vel_c2 = vel_fn(latents, t, sigma, embeddings_c2, cond_kwargs_c2)
        vel_u  = vel_fn(latents, t, sigma, embeddings_uncond, cond_kwargs_uncond)

        d1, d2 = score_deltas(vel_c1, vel_c2, vel_u)
        s_comp = vel_u + cfg.guidance_scale * (d1 + d2)

        noise = torch.randn_like(latents) * (2 * step_size) ** 0.5
        latents = latents + step_size * s_comp + noise

    return latents


# ---------------------------------------------------------------------------
# Strategy B: SMC over diffusion paths
# ---------------------------------------------------------------------------

def _effective_sample_size(log_weights: torch.Tensor) -> torch.Tensor:
    """
    ESS = (Σ w_i)^2 / Σ w_i^2  (unnormalised form).
    log_weights: [P]  unnormalised log weights
    Returns scalar ESS.
    """
    log_weights = log_weights - log_weights.max()
    w = log_weights.exp()
    w = w / (w.sum() + 1e-8)
    ess = 1.0 / (w ** 2).sum()
    return ess


def _systematic_resample(particles: List[Vel], log_weights: torch.Tensor) -> List[Vel]:
    """
    Systematic resampling.
    particles: list of P latents, each [B, C, H, W]
    log_weights: [P]
    Returns resampled list of P latents.
    """
    P = len(particles)
    log_w = log_weights - log_weights.max()
    w = log_w.exp()
    w = w / (w.sum() + 1e-8)

    # Cumulative weights
    cumw = w.cumsum(0)

    # Systematic sampling points
    u = torch.rand(1, device=w.device) / P
    positions = u + torch.arange(P, device=w.device, dtype=w.dtype) / P

    indices = torch.searchsorted(cumw, positions).clamp(0, P - 1)
    return [particles[idx.item()] for idx in indices]


# ---------------------------------------------------------------------------
# Main composition functions
# ---------------------------------------------------------------------------

def run_mcmc_corrector(
    latents: Vel,
    vel_fn,
    embeddings_c1,
    embeddings_c2,
    embeddings_uncond,
    scheduler,
    cfg: Optional[MCMCCorrectorConfig] = None,
    cond_kwargs_c1: Optional[dict] = None,
    cond_kwargs_c2: Optional[dict] = None,
    cond_kwargs_uncond: Optional[dict] = None,
) -> Tuple[Vel, list]:
    """
    Full denoising loop with MCMC corrector (Strategy A or B).

    Strategy A (langevin): standard denoising + Langevin corrector steps.
    Strategy B (smc):      SMC with P particles over diffusion paths.
    """
    if cfg is None:
        cfg = MCMCCorrectorConfig()

    if cfg.strategy == "langevin":
        return _run_langevin(
            latents, vel_fn,
            embeddings_c1, embeddings_c2, embeddings_uncond,
            scheduler, cfg,
            cond_kwargs_c1, cond_kwargs_c2, cond_kwargs_uncond,
        )
    elif cfg.strategy == "smc":
        return _run_smc(
            latents, vel_fn,
            embeddings_c1, embeddings_c2, embeddings_uncond,
            scheduler, cfg,
            cond_kwargs_c1, cond_kwargs_c2, cond_kwargs_uncond,
        )
    else:
        raise ValueError(f"Unknown strategy: {cfg.strategy!r}. Use 'langevin' or 'smc'.")


def _run_langevin(
    latents, vel_fn,
    embeddings_c1, embeddings_c2, embeddings_uncond,
    scheduler, cfg,
    cond_kwargs_c1, cond_kwargs_c2, cond_kwargs_uncond,
):
    N = len(scheduler.timesteps)
    infos = []

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma  = scheduler.sigmas[i]
        t_frac = i / max(N - 1, 1)

        vel_c1    = vel_fn(latents, t, sigma, embeddings_c1, cond_kwargs_c1)
        vel_c2    = vel_fn(latents, t, sigma, embeddings_c2, cond_kwargs_c2)
        vel_uncond = vel_fn(latents, t, sigma, embeddings_uncond, cond_kwargs_uncond)

        d1, d2 = score_deltas(vel_c1, vel_c2, vel_uncond)
        vf = vel_uncond + cfg.guidance_scale * (d1 + d2)

        # Predictor step
        latents = scheduler_step(scheduler, vf, t, latents, sigma, dsigma)

        # Corrector (only applied after a threshold fraction of steps)
        corrector_applied = False
        if t_frac >= cfg.corrector_start_frac and cfg.n_corrector_steps > 0:
            latents = _langevin_corrector(
                latents, vel_fn,
                embeddings_c1, embeddings_c2, embeddings_uncond,
                t, sigma, cfg,
                cond_kwargs_c1, cond_kwargs_c2, cond_kwargs_uncond,
            )
            corrector_applied = True

        infos.append({
            "t_frac": t_frac,
            "corrector_applied": corrector_applied,
        })

    return latents, infos


def _run_smc(
    latents, vel_fn,
    embeddings_c1, embeddings_c2, embeddings_uncond,
    scheduler, cfg,
    cond_kwargs_c1, cond_kwargs_c2, cond_kwargs_uncond,
):
    """
    SMC over diffusion paths.
    Maintains P independent latent particles; resamples by ESS.
    Returns the particle with the highest final log-weight.
    """
    P = cfg.n_particles
    # Initialise P particles as copies of the starting latent
    particles = [latents.clone() for _ in range(P)]
    log_weights = torch.zeros(P, device=latents.device, dtype=torch.float32)

    infos = []
    N = len(scheduler.timesteps)

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma  = scheduler.sigmas[i]

        new_particles = []
        step_log_weights = []

        for p_idx, x_p in enumerate(particles):
            vel_c1    = vel_fn(x_p, t, sigma, embeddings_c1, cond_kwargs_c1)
            vel_c2    = vel_fn(x_p, t, sigma, embeddings_c2, cond_kwargs_c2)
            vel_uncond = vel_fn(x_p, t, sigma, embeddings_uncond, cond_kwargs_uncond)

            d1, d2 = score_deltas(vel_c1, vel_c2, vel_uncond)
            vf = vel_uncond + cfg.guidance_scale * (d1 + d2)

            # Predictor step for this particle
            x_next = scheduler_step(scheduler, vf, t, x_p, sigma, dsigma)

            # Incremental log-weight (importance weight)
            log_w_inc = _log_density_estimate(
                x_next, vel_c1, vel_c2, vel_uncond, sigma
            ).mean()   # scalar for this particle

            new_particles.append(x_next)
            step_log_weights.append(log_w_inc)

        # Update cumulative log-weights
        step_lw = torch.stack(step_log_weights)   # [P]
        log_weights = log_weights + step_lw

        # ESS-based resampling
        ess = _effective_sample_size(log_weights)
        resampled = False
        if ess < cfg.ess_threshold * P:
            new_particles = _systematic_resample(new_particles, log_weights)
            log_weights = torch.zeros(P, device=latents.device, dtype=torch.float32)
            resampled = True

            # Optional: MH correction steps on resampled particles
            if cfg.smc_mh_steps > 0:
                cfg_lan = MCMCCorrectorConfig(
                    guidance_scale=cfg.guidance_scale,
                    strategy="langevin",
                    n_corrector_steps=cfg.smc_mh_steps,
                    langevin_step_scale=0.05,
                )
                new_particles = [
                    _langevin_corrector(
                        x_p, vel_fn,
                        embeddings_c1, embeddings_c2, embeddings_uncond,
                        t, sigma, cfg_lan,
                        cond_kwargs_c1, cond_kwargs_c2, cond_kwargs_uncond,
                    )
                    for x_p in new_particles
                ]

        particles = new_particles
        infos.append({"ess": ess.item(), "resampled": resampled})

    # Return highest-weight particle
    best_idx = log_weights.argmax().item()
    return particles[best_idx], infos
