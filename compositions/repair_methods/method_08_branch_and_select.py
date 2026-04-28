"""
Method 08 — Mixture-of-Joints Branch-and-Select Sampling
=========================================================

Theoretical grounding
---------------------
Group 4 failures (slot competition, identity collision) require a discrete
choice: which spatial region belongs to which concept, which mode wins.
A single additive score field averages incompatible hypotheses into a hybrid.

Branch-and-select maintains K hypotheses ("branches"), each corresponding
to a candidate spatial assignment of concepts to slots.  At each timestep:

  1. Each branch is propagated with its own masked composition rule:
         s_comp^(a) = su + Delta_1^(a) + Delta_2^(a)
     where Delta_i^(a) is Delta_i masked to the region assigned to concept i
     in assignment a.

  2. Branches are scored by a non-joint scoring function:
         score(x, a) = presence_1(x) + presence_2(x) - lambda * hybrid(x)
     using only single-concept presence proxies (no joint prompt).

  3. The top-K branches are retained (pruning).

  4. New branches are proposed by perturbing the current mask assignment
     (swap regions, dilate/erode, add random perturbation).

  5. At the end of denoising, the best-scoring branch is returned.

Presence proxies (no joint model):
  - ||Delta_i||_F (Frobenius norm) as a proxy for how strongly concept i
    is currently "pulling" the latent.
  - Optionally: per-concept CLIP similarity (requires CLIP encoder).

Hybrid penalty:
  - Cosine similarity between Delta_1 and Delta_2 (high similarity ->
    both concepts are entangled in the same direction -> hybrid risk).
  - Overlap IoU between mask_c1 and mask_c2.

References
----------
- The document §"Mixture-of-joints and branch-and-select sampling"
- Li et al. 2024 "Attention Overlap Is Responsible for Entity Missing"
  (arXiv:2410.20972) — motivation for mask-based slot assignment
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable
import torch
import torch.nn.functional as F

from ._base import Vel, score_deltas, cosine_similarity_deltas, attention_overlap_iou, scheduler_step


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BranchAndSelectConfig:
    # Guidance scale
    guidance_scale: float = 7.5

    # Number of branches to maintain
    n_branches: int = 4

    # Number of candidate proposals per branch per step
    n_proposals: int = 2

    # Hybrid penalty weight
    lambda_hybrid: float = 0.5

    # Re-score and prune every N timesteps (1 = every step, more = cheaper)
    prune_every: int = 5

    # Initial mask assignment strategy: "random", "equal_split", "norm_based"
    init_strategy: str = "norm_based"

    # Mask perturbation strategies for proposals
    # Options: "swap", "erode", "dilate", "random_noise"
    proposal_strategies: List[str] = field(
        default_factory=lambda: ["swap", "random_noise"]
    )

    eps: float = 1e-8


# ---------------------------------------------------------------------------
# Mask assignment utilities
# ---------------------------------------------------------------------------

def _norm_based_masks(
    d1: Vel,
    d2: Vel,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initial masks based on relative norm of concept deltas at each spatial location.
    M1[b, h, w] = ||d1[:, :, h, w]|| / (||d1[:, :, h, w]|| + ||d2[:, :, h, w]|| + eps)
    Returns M1, M2 in [0,1]^{B, H, W} with M1 + M2 ~ 1.
    """
    n1 = d1.float().norm(dim=1)   # [B, H, W]
    n2 = d2.float().norm(dim=1)
    total = n1 + n2 + eps
    M1 = n1 / total
    M2 = n2 / total
    return M1, M2


def _equal_split_masks(B: int, H: int, W: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left half -> concept 1, right half -> concept 2."""
    M1 = torch.zeros(B, H, W, device=device, dtype=dtype)
    M2 = torch.zeros(B, H, W, device=device, dtype=dtype)
    mid = W // 2
    M1[:, :, :mid] = 1.0
    M2[:, :, mid:] = 1.0
    return M1, M2


def _random_masks(B: int, H: int, W: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random spatial split."""
    M1 = torch.rand(B, H, W, device=device, dtype=dtype)
    M2 = 1.0 - M1
    return M1, M2


# ---------------------------------------------------------------------------
# Mask proposal generators
# ---------------------------------------------------------------------------

def _propose_swap(M1: torch.Tensor, M2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Swap: concept 1 takes concept 2's region and vice versa."""
    return M2.clone(), M1.clone()


def _propose_random_noise(
    M1: torch.Tensor,
    M2: torch.Tensor,
    noise_std: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add Gaussian noise to masks and renormalise."""
    M1_new = (M1 + noise_std * torch.randn_like(M1)).clamp(0.01, 0.99)
    M2_new = (M2 + noise_std * torch.randn_like(M2)).clamp(0.01, 0.99)
    total = M1_new + M2_new + 1e-8
    return M1_new / total, M2_new / total


def _propose_erode(M1: torch.Tensor, M2: torch.Tensor, kernel: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Erode M1 (shrink concept-1 region, expand concept-2 region)."""
    pad = kernel // 2
    M1_e = -F.max_pool2d(-M1.unsqueeze(1), kernel, stride=1, padding=pad).squeeze(1)
    M2_e = 1.0 - M1_e
    return M1_e.clamp(0, 1), M2_e.clamp(0, 1)


def _propose_dilate(M1: torch.Tensor, M2: torch.Tensor, kernel: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dilate M1 (expand concept-1 region)."""
    pad = kernel // 2
    M1_d = F.max_pool2d(M1.unsqueeze(1), kernel, stride=1, padding=pad).squeeze(1)
    M2_d = 1.0 - M1_d
    return M1_d.clamp(0, 1), M2_d.clamp(0, 1)


_PROPOSAL_FNS = {
    "swap":         _propose_swap,
    "random_noise": _propose_random_noise,
    "erode":        _propose_erode,
    "dilate":       _propose_dilate,
}


# ---------------------------------------------------------------------------
# Branch scoring
# ---------------------------------------------------------------------------

def _score_branch(
    x: Vel,
    d1: Vel,
    d2: Vel,
    M1: torch.Tensor,
    M2: torch.Tensor,
    cfg: BranchAndSelectConfig,
) -> torch.Tensor:
    """
    Score a branch by:
        presence_1 + presence_2  - lambda_hybrid * (hybrid_cos + overlap_iou)

    All terms are mean over batch; returns scalar.
    """
    # Presence proxies: how strongly is concept i active globally?
    pres_1 = d1.float().norm(dim=(1, 2, 3)).mean()    # scalar
    pres_2 = d2.float().norm(dim=(1, 2, 3)).mean()

    # Hybrid penalty 1: cos(Delta_1, Delta_2) — high = entangled
    cos = cosine_similarity_deltas(d1, d2).mean()      # scalar

    # Hybrid penalty 2: spatial overlap IoU
    iou = attention_overlap_iou(M1, M2).mean()         # scalar

    score = pres_1 + pres_2 - cfg.lambda_hybrid * (cos.clamp(0) + iou)
    return score


# ---------------------------------------------------------------------------
# Composed velocity for a given mask assignment
# ---------------------------------------------------------------------------

def _compose_with_masks(
    d1: Vel,
    d2: Vel,
    vel_uncond: Vel,
    M1: torch.Tensor,
    M2: torch.Tensor,
    guidance_scale: float,
) -> Vel:
    """Apply mask-gated composition."""
    M1_b = M1.unsqueeze(1).expand_as(d1)
    M2_b = M2.unsqueeze(1).expand_as(d2)
    return vel_uncond + guidance_scale * (M1_b * d1 + M2_b * d2)


# ---------------------------------------------------------------------------
# Full denoising loop
# ---------------------------------------------------------------------------

def run_branch_and_select(
    latents: Vel,
    vel_fn,
    embeddings_c1,
    embeddings_c2,
    embeddings_uncond,
    scheduler,
    cfg: Optional[BranchAndSelectConfig] = None,
    cond_kwargs_c1: Optional[dict] = None,
    cond_kwargs_c2: Optional[dict] = None,
    cond_kwargs_uncond: Optional[dict] = None,
) -> Tuple[Vel, list]:
    """
    Branch-and-select denoising.

    Maintains K branches (latent + mask assignment).
    Returns the best-scoring branch at the end.
    """
    if cfg is None:
        cfg = BranchAndSelectConfig()

    B, C, H, W = latents.shape
    N = len(scheduler.timesteps)

    # --- Initialise branches ---
    # Each branch: (latent [B,C,H,W], M1 [B,H,W], M2 [B,H,W])
    def _init_masks(d1, d2):
        if cfg.init_strategy == "norm_based":
            return _norm_based_masks(d1, d2)
        elif cfg.init_strategy == "equal_split":
            return _equal_split_masks(B, H, W, latents.device, latents.dtype)
        else:
            return _random_masks(B, H, W, latents.device, latents.dtype)

    # Bootstrap: get initial deltas for mask init
    t0 = scheduler.timesteps[0]
    s0 = scheduler.sigmas[0]
    vel_c1_init = vel_fn(latents, t0, s0, embeddings_c1, cond_kwargs_c1)
    vel_c2_init = vel_fn(latents, t0, s0, embeddings_c2, cond_kwargs_c2)
    vel_u_init  = vel_fn(latents, t0, s0, embeddings_uncond, cond_kwargs_uncond)
    d1_init, d2_init = score_deltas(vel_c1_init, vel_c2_init, vel_u_init)

    M1_init, M2_init = _init_masks(d1_init, d2_init)

    branches = []
    for _ in range(cfg.n_branches):
        # Add slight perturbation for diversity
        M1_p, M2_p = _propose_random_noise(M1_init, M2_init, noise_std=0.15)
        branches.append((latents.clone(), M1_p.clone(), M2_p.clone()))

    infos = []

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma  = scheduler.sigmas[i]

        # Step each branch
        new_branches = []
        branch_scores = []

        for (x_b, M1_b, M2_b) in branches:
            vel_c1    = vel_fn(x_b, t, sigma, embeddings_c1, cond_kwargs_c1)
            vel_c2    = vel_fn(x_b, t, sigma, embeddings_c2, cond_kwargs_c2)
            vel_uncond = vel_fn(x_b, t, sigma, embeddings_uncond, cond_kwargs_uncond)
            d1, d2 = score_deltas(vel_c1, vel_c2, vel_uncond)

            # Generate proposals for this branch
            candidates = [(x_b, M1_b, M2_b, d1, d2, vel_uncond)]   # include current

            for strategy in cfg.proposal_strategies[:cfg.n_proposals]:
                fn = _PROPOSAL_FNS.get(strategy, _propose_random_noise)
                M1_prop, M2_prop = fn(M1_b, M2_b)
                candidates.append((x_b, M1_prop, M2_prop, d1, d2, vel_uncond))

            # Step and score each candidate
            for (x_c, M1_c, M2_c, d1_c, d2_c, vu) in candidates:
                vf = _compose_with_masks(d1_c, d2_c, vu, M1_c, M2_c, cfg.guidance_scale)
                x_next = scheduler_step(scheduler, vf, t, x_c, sigma, dsigma)

                sc = _score_branch(x_next, d1_c, d2_c, M1_c, M2_c, cfg)
                new_branches.append((x_next, M1_c, M2_c))
                branch_scores.append(sc.item())

        # Prune: keep top-K branches
        if i % cfg.prune_every == 0 or i == N - 1:
            ranked = sorted(
                zip(branch_scores, new_branches),
                key=lambda x: x[0],
                reverse=True,
            )
            branches = [b for _, b in ranked[:cfg.n_branches]]
        else:
            # No pruning: take first cfg.n_branches candidates
            branches = new_branches[:cfg.n_branches]

        infos.append({
            "best_score": max(branch_scores),
            "worst_score": min(branch_scores),
            "n_candidates": len(new_branches),
        })

    # Select best final branch
    # Re-score with final velocities
    final_scores = []
    final_latents_list = []
    for (x_b, M1_b, M2_b) in branches:
        t_last = scheduler.timesteps[-1]
        s_last = scheduler.sigmas[-1]
        vel_c1 = vel_fn(x_b, t_last, s_last, embeddings_c1, cond_kwargs_c1)
        vel_c2 = vel_fn(x_b, t_last, s_last, embeddings_c2, cond_kwargs_c2)
        vel_u  = vel_fn(x_b, t_last, s_last, embeddings_uncond, cond_kwargs_uncond)
        d1, d2 = score_deltas(vel_c1, vel_c2, vel_u)
        sc = _score_branch(x_b, d1, d2, M1_b, M2_b, cfg)
        final_scores.append(sc.item())
        final_latents_list.append(x_b)

    best_idx = max(range(len(final_scores)), key=lambda i: final_scores[i])
    return final_latents_list[best_idx], infos
