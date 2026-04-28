"""
Method 04 — Region/Slot-Aware Mask-Gated PoE
=============================================

Theoretical grounding
---------------------
Group 4 failures (slot competition, entity missing) arise because PoE
applies each concept's score delta globally, causing cross-attention maps
of the two concepts to overlap.  Attention overlap empirically predicts
entity dropping (Li et al. 2024, arXiv:2410.20972).

This method allocates spatial regions (latent patches) to each concept
by deriving soft binary masks from the cross-attention maps of each
single-concept forward pass, then applies the score deltas only within
the concept's assigned region:

    M_i(x_t, t) in [0,1]^{H x W}   — soft spatial mask for concept i

    vf = su  +  w1(t) * (M1 ⊙ Delta_1)  +  w2(t) * (M2 ⊙ Delta_2)

with an optional overlap penalty that suppresses shared regions:
    overlap = M1 * M2
    M1 <- clamp01(M1 - sep_strength * overlap)
    M2 <- clamp01(M2 - sep_strength * overlap)

Mask extraction
---------------
Because SD3.5 is a transformer, cross-attention maps are extracted by
registering forward hooks on the attention layers and accumulating
token-to-spatial attention weights.  We expose:
  - register_attention_hooks(unet) -> hook_handle
  - get_attention_maps(hook_handle) -> {layer_id: [B, heads, spatial, tokens]}
  - remove_attention_hooks(hook_handle)

If cross-attention maps are unavailable (e.g., no hook access), the
method falls back to using the norm of each score delta as a proxy:
    M_i(x, t) ~ softmax(||Delta_i||_channel)   (spatial norm map)

References
----------
- Li et al. 2024 "Attention Overlap Is Responsible for Entity Missing"
  (arXiv:2410.20972)
- Chefer et al. 2023 "A-STAR: Test-time Attention Segregation and Retention"
  (arXiv:2306.14544)
- Hertz et al. 2022 "Prompt-to-Prompt" (arXiv:2208.01626)
- The document §"Region-aware or slot-aware composition using attention-derived masks"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn.functional as F

from ._base import Vel, score_deltas, attention_overlap_iou, scheduler_step


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MaskGatedPoEConfig:
    # Base guidance scale
    guidance_scale: float = 7.5

    # Overlap suppression strength (0 = no suppression, 1 = full suppression)
    # M_i <- clamp(M_i - sep_strength * overlap, 0, 1)
    sep_strength: float = 0.5

    # Schedule for sep_strength: "constant", "linear_increase"
    sep_schedule: str = "constant"    # increase separation strength over time

    # Temperature for softmax when converting attention to mask
    mask_temperature: float = 1.0

    # Minimum mask value (prevents full dropout of a concept in any region)
    mask_min: float = 0.0

    # Whether to use the delta-norm fallback when attention maps are unavailable
    use_delta_norm_fallback: bool = True

    # Layer indices to aggregate attention from (None = all layers)
    attention_layers: Optional[List[int]] = None


# ---------------------------------------------------------------------------
# Attention hook infrastructure
# ---------------------------------------------------------------------------

class AttentionMapCollector:
    """
    Registers forward hooks on transformer cross-attention layers to
    collect token-to-spatial attention weights.

    Usage:
        collector = AttentionMapCollector(transformer)
        collector.register()
        # run forward pass
        maps = collector.get_maps()   # {layer_idx: Tensor [B, heads, spatial, tokens]}
        collector.clear()
        collector.remove()
    """

    def __init__(self, model, layer_indices: Optional[List[int]] = None):
        self.model = model
        self.layer_indices = layer_indices
        self._handles = []
        self._maps: Dict[int, torch.Tensor] = {}
        self._layer_count = 0

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # output is typically (attn_output, attn_weights) or just attn_output
            # Try to capture attention_weights if returned
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights = output[1]   # [B, heads, tgt_seq, src_seq]
                # Accumulate (average over heads)
                if layer_idx in self._maps:
                    self._maps[layer_idx] = self._maps[layer_idx] + attn_weights.detach()
                else:
                    self._maps[layer_idx] = attn_weights.detach().clone()
        return hook

    def register(self):
        """Register hooks on all (or specified) cross-attention layers."""
        self._layer_count = 0
        for name, module in self.model.named_modules():
            if "attn" in name.lower() and hasattr(module, "to_q"):
                if self.layer_indices is None or self._layer_count in self.layer_indices:
                    h = module.register_forward_hook(self._make_hook(self._layer_count))
                    self._handles.append(h)
                self._layer_count += 1

    def get_maps(self) -> Dict[int, torch.Tensor]:
        return dict(self._maps)

    def clear(self):
        self._maps.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Mask extraction utilities
# ---------------------------------------------------------------------------

def _delta_norm_to_mask(delta: Vel, temperature: float = 1.0) -> torch.Tensor:
    """
    Fallback mask: spatial softmax of channel-norm of score delta.
    delta: [B, C, H, W]
    Returns: [B, H, W] soft mask in [0,1]
    """
    norm_map = delta.float().norm(dim=1)                    # [B, H, W]
    B, H, W = norm_map.shape
    flat = norm_map.reshape(B, -1) / temperature
    mask = F.softmax(flat, dim=-1).reshape(B, H, W)
    return mask


def attention_maps_to_mask(
    attn_maps: Dict[int, torch.Tensor],
    token_idx: int,
    spatial_h: int,
    spatial_w: int,
    temperature: float = 1.0,
    layer_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Aggregate collected attention maps into a single spatial mask.

    attn_maps: {layer_idx: [B, heads, spatial, tokens]}
    token_idx: which token to use (e.g., index of the concept token)
    spatial_h, spatial_w: expected spatial dimensions of the latent

    Returns: [B, H, W] mask in [0, 1]
    """
    layers = layer_indices if layer_indices is not None else list(attn_maps.keys())
    accumulated = None
    for idx in layers:
        if idx not in attn_maps:
            continue
        am = attn_maps[idx]          # [B, heads, spatial, tokens]
        # Take the token of interest, average over heads
        concept_attn = am[..., token_idx].mean(dim=1)   # [B, spatial]
        if accumulated is None:
            accumulated = concept_attn
        else:
            accumulated = accumulated + concept_attn

    if accumulated is None:
        return None

    B, S = accumulated.shape
    # Resize to spatial_h x spatial_w
    h = int(S ** 0.5)
    if h * h != S:
        # Non-square spatial: pad or just return flat
        h = spatial_h
        w = spatial_w
    else:
        w = h

    spatial_map = accumulated.reshape(B, 1, h, w)
    spatial_map = F.interpolate(spatial_map, size=(spatial_h, spatial_w), mode="bilinear", align_corners=False)
    spatial_map = spatial_map.squeeze(1)              # [B, H, W]

    # Normalise to [0,1]
    spatial_map = spatial_map / (spatial_map.amax(dim=(1, 2), keepdim=True) + 1e-8)

    # Apply softmax-like sharpening
    flat = spatial_map.reshape(B, -1) / temperature
    mask = F.softmax(flat, dim=-1).reshape(B, *spatial_map.shape[1:])
    return mask


# ---------------------------------------------------------------------------
# Main composition function
# ---------------------------------------------------------------------------

def compose_mask_gated(
    vel_c1: Vel,
    vel_c2: Vel,
    vel_uncond: Vel,
    t_frac: float,
    mask_c1: Optional[torch.Tensor] = None,   # [B, H, W] or None
    mask_c2: Optional[torch.Tensor] = None,   # [B, H, W] or None
    cfg: Optional[MaskGatedPoEConfig] = None,
) -> Tuple[Vel, dict]:
    """
    Mask-gated composition.

    If mask_c1 / mask_c2 are None, falls back to delta-norm masks.

    Parameters
    ----------
    vel_c1, vel_c2, vel_uncond : [B, C, H, W]
    t_frac  : float in [0,1]
    mask_c1 : [B, H, W] or None
    mask_c2 : [B, H, W] or None
    cfg     : MaskGatedPoEConfig

    Returns
    -------
    vf   : [B, C, H, W] composed velocity
    info : dict with diagnostics
    """
    if cfg is None:
        cfg = MaskGatedPoEConfig()

    B, C, H, W = vel_c1.shape
    d1, d2 = score_deltas(vel_c1, vel_c2, vel_uncond)

    # Build masks
    if mask_c1 is None or not cfg.use_delta_norm_fallback is False:
        if mask_c1 is None:
            mask_c1 = _delta_norm_to_mask(d1, cfg.mask_temperature)  # [B, H, W]
        if mask_c2 is None:
            mask_c2 = _delta_norm_to_mask(d2, cfg.mask_temperature)

    mask_c1 = mask_c1.to(d1.device, d1.dtype)
    mask_c2 = mask_c2.to(d2.device, d2.dtype)

    # Separation strength schedule
    if cfg.sep_schedule == "linear_increase":
        sep = cfg.sep_strength * t_frac
    else:
        sep = cfg.sep_strength

    # Overlap suppression
    overlap = mask_c1 * mask_c2
    iou = attention_overlap_iou(mask_c1, mask_c2)   # [B]

    mask_c1_sep = (mask_c1 - sep * overlap).clamp(cfg.mask_min, 1.0)
    mask_c2_sep = (mask_c2 - sep * overlap).clamp(cfg.mask_min, 1.0)

    # Broadcast masks over channels: [B, H, W] -> [B, C, H, W]
    M1 = mask_c1_sep.unsqueeze(1).expand_as(d1)
    M2 = mask_c2_sep.unsqueeze(1).expand_as(d2)

    # Masked deltas
    d1_local = M1 * d1
    d2_local = M2 * d2

    vf = vel_uncond + cfg.guidance_scale * (d1_local + d2_local)

    info = {
        "overlap_iou": iou.detach(),
        "mask_c1_mean": mask_c1_sep.mean(dim=(1, 2)).detach(),
        "mask_c2_mean": mask_c2_sep.mean(dim=(1, 2)).detach(),
        "sep_strength_used": sep,
    }
    return vf, info


# ---------------------------------------------------------------------------
# Full denoising loop
# ---------------------------------------------------------------------------

def run_mask_gated_poe(
    latents: Vel,
    vel_fn,
    embeddings_c1,
    embeddings_c2,
    embeddings_uncond,
    scheduler,
    cfg: Optional[MaskGatedPoEConfig] = None,
    cond_kwargs_c1: Optional[dict] = None,
    cond_kwargs_c2: Optional[dict] = None,
    cond_kwargs_uncond: Optional[dict] = None,
    attn_collector: Optional[AttentionMapCollector] = None,
    token_idx_c1: int = 1,
    token_idx_c2: int = 1,
) -> Tuple[Vel, list]:
    """
    Full denoising loop using mask-gated PoE composition.

    If attn_collector is provided and registered, attention maps are used
    for masking; otherwise delta-norm fallback is used.
    """
    if cfg is None:
        cfg = MaskGatedPoEConfig()

    N = len(scheduler.timesteps)
    infos = []

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma = scheduler.sigmas[i]
        t_frac = i / max(N - 1, 1)

        # Run concept-1 forward pass and collect attention
        mask_c1 = None
        mask_c2 = None

        if attn_collector is not None:
            attn_collector.clear()

        vel_c1 = vel_fn(latents, t, sigma, embeddings_c1, cond_kwargs_c1)

        if attn_collector is not None:
            maps_c1 = attn_collector.get_maps()
            attn_collector.clear()
            B, C, H, W = latents.shape
            mask_c1 = attention_maps_to_mask(
                maps_c1, token_idx_c1, H, W,
                temperature=cfg.mask_temperature,
                layer_indices=cfg.attention_layers,
            )

        vel_c2 = vel_fn(latents, t, sigma, embeddings_c2, cond_kwargs_c2)

        if attn_collector is not None:
            maps_c2 = attn_collector.get_maps()
            attn_collector.clear()
            B, C, H, W = latents.shape
            mask_c2 = attention_maps_to_mask(
                maps_c2, token_idx_c2, H, W,
                temperature=cfg.mask_temperature,
                layer_indices=cfg.attention_layers,
            )

        vel_uncond = vel_fn(latents, t, sigma, embeddings_uncond, cond_kwargs_uncond)

        vf, info = compose_mask_gated(
            vel_c1, vel_c2, vel_uncond, t_frac, mask_c1, mask_c2, cfg
        )
        infos.append(info)

        latents = scheduler_step(scheduler, vf, t, latents, sigma, dsigma)

    return latents, infos
