"""
Shared utilities for cross-attention hook infrastructure and score metrics.
Used by visualize_scores.py and visualize_attention_maps.py.

Cross-attention hooking is adapted from:
  compositions/Attend-and-Excite/utils/ptp_utils.py
with the CrossAttention import fixed for current diffusers versions.
"""

import abc
import types
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Diffusers compatibility shim — class was renamed between versions
# ---------------------------------------------------------------------------
try:
    from diffusers.models.attention_processor import Attention as _AttnClass
except ImportError:
    try:
        from diffusers.models.cross_attention import CrossAttention as _AttnClass
    except ImportError:
        _AttnClass = object  # fallback; hooks won't type-check but will still run

try:
    from diffusers.models.attention_processor import AttnProcessor2_0 as _DefaultProcessor
except ImportError:
    try:
        from diffusers.models.attention_processor import AttnProcessor as _DefaultProcessor
    except ImportError:
        _DefaultProcessor = None


# ---------------------------------------------------------------------------
# Color / label constants (four distributions, no SuperDiff AND)
# ---------------------------------------------------------------------------

CONDITION_COLORS = {
    "solo_a":     "#e63946",
    "solo_b":     "#457b9d",
    "monolithic": "#2a9d8f",
    "poe":        "#9b5de5",
    # PoE split passes (for attention maps)
    "poe_a":      "#c77dff",
    "poe_b":      "#7b2d8b",
    # uncond reference
    "uncond":     "#aaaaaa",
}

CONDITION_LABELS = {
    "solo_a":     "P(A)",
    "solo_b":     "P(B)",
    "monolithic": "P(A∧B)",
    "poe":        "PoE",
    "poe_a":      "PoE pass-A",
    "poe_b":      "PoE pass-B",
    "uncond":     "Uncond",
}


# ---------------------------------------------------------------------------
# Attention Store (adapted from ptp_utils.py)
# ---------------------------------------------------------------------------

class SDXLAttentionStore:
    """Stores cross-attention maps from a single UNet forward pass."""

    @staticmethod
    def _empty_store():
        return {
            "down_cross": [], "mid_cross": [], "up_cross": [],
            "down_self":  [], "mid_self":  [], "up_self":  [],
        }

    def forward(self, attn_probs: torch.Tensor, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # Only store maps at manageable resolutions (≤ 32×32 = 1024 tokens)
        if attn_probs.shape[1] <= 32 ** 2:
            self.step_store[key].append(attn_probs.detach().cpu())

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self._empty_store()

    def get_attention(self):
        # If between_steps() has not been called yet, fall back to step_store
        if self.attention_store:
            return self.attention_store
        return self.step_store

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        self.step_store = self._empty_store()
        self.attention_store = {}
        self.num_att_layers = -1

    def __call__(self, attn_probs: torch.Tensor, is_cross: bool, place_in_unet: str):
        self.forward(attn_probs, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self._empty_store()
        self.attention_store = {}


# ---------------------------------------------------------------------------
# Attention Processor (adapted from ptp_utils.py)
# ---------------------------------------------------------------------------

class SDXLAttnProcessor:
    """
    Replaces a UNet attention processor to intercept attention probabilities.

    Matches the AttnProcessor2_0 layout used in diffusers >= 0.20 (SDXL):
    query/key/value are reshaped to (batch, heads, seq, head_dim) before
    attention is computed. We explicitly materialise softmax(QK^T/√d) so
    we can store the per-head attention maps, then reconstruct hidden_states
    identically to AttnProcessor2_0.
    """

    def __init__(self, attnstore: SDXLAttentionStore, place_in_unet: str):
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kwargs):
        import math

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        is_cross = encoder_hidden_states is not None

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        ctx = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if encoder_hidden_states is not None and attn.norm_cross:
            ctx = attn.norm_encoder_hidden_states(ctx)
        key   = attn.to_k(ctx)
        value = attn.to_v(ctx)

        inner_dim = key.shape[-1]
        head_dim  = inner_dim // attn.heads

        # Reshape to (batch, heads, seq, head_dim)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if getattr(attn, 'norm_q', None) is not None:
            query = attn.norm_q(query)
        if getattr(attn, 'norm_k', None) is not None:
            key   = attn.norm_k(key)

        if is_cross:
            # Explicitly materialise attention probs for cross-attention (float32 for stability).
            # Cross-attention query_len = spatial tokens (≤ 32^2 = 1024 at res ≤ 32),
            # key_len = text tokens (77). Memory cost: batch*heads*1024*77*4 ≈ small.
            scale = 1.0 / math.sqrt(head_dim)
            attn_weight = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
            if attention_mask is not None:
                attn_weight = attn_weight + attention_mask.float()
            attention_probs = torch.softmax(attn_weight, dim=-1).to(query.dtype)
            # Store in the format expected by aggregate_cross_attn:
            # (heads*batch, query_len, key_len) — same as old ptp_utils format
            attn_probs_flat = attention_probs.reshape(batch_size * attn.heads,
                                                       attention_probs.shape[2],
                                                       attention_probs.shape[3])
            self.attnstore(attn_probs_flat.detach(), is_cross, self.place_in_unet)
            # Reconstruct output
            hidden_states = torch.matmul(attention_probs, value)
        else:
            # Self-attention: use fused SDPA to avoid materialising (seq, seq) matrix
            # (self-attention seq_len can be up to 16384 for high-res layers → OOM in float32)
            # We still call attnstore with dummy to increment cur_att_layer counter.
            hidden_states = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            # Do not call attnstore for self-attention — we only need cross-attention maps.
            # between_steps() is flushed manually in capture_attention_at_step().

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ---------------------------------------------------------------------------
# Hook registration helpers
# ---------------------------------------------------------------------------

def register_hooks(unet, store: SDXLAttentionStore) -> None:
    """Register SDXLAttnProcessor on all attention layers of a bare UNet."""
    attn_procs = {}
    cross_att_count = 0

    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = SDXLAttnProcessor(attnstore=store, place_in_unet=place_in_unet)

    unet.set_attn_processor(attn_procs)
    store.num_att_layers = cross_att_count


def restore_processors(unet) -> None:
    """Restore UNet to its default attention processors (AttnProcessor2_0)."""
    if _DefaultProcessor is None:
        # Fallback: reset to empty dict which triggers diffusers default
        unet.set_attn_processor({})
        return
    unet.set_attn_processor(
        {k: _DefaultProcessor() for k in unet.attn_processors.keys()}
    )


# ---------------------------------------------------------------------------
# Core capture primitive
# ---------------------------------------------------------------------------

@torch.no_grad()
def capture_attention_at_step(
    unet,
    latent_model_input: torch.Tensor,
    t: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    added_cond_kwargs: Optional[dict] = None,
) -> SDXLAttentionStore:
    """
    Run ONE UNet forward pass with attention hooks active.

    Registers a fresh SDXLAttentionStore, runs the forward pass,
    immediately restores default processors, and returns the populated store.
    Always restores processors even if the forward pass raises.

    Returns the populated SDXLAttentionStore.
    """
    store = SDXLAttentionStore()
    register_hooks(unet, store)
    try:
        kwargs = {}
        if added_cond_kwargs is not None:
            kwargs["added_cond_kwargs"] = added_cond_kwargs
        unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        ).sample
    finally:
        restore_processors(unet)
    # Flush step_store → attention_store (between_steps only auto-fires when
    # cur_att_layer wraps around, which requires exactly num_att_layers calls;
    # for a single forward pass this may not happen, so flush manually.)
    if not store.attention_store:
        store.between_steps()
    return store


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_cross_attn(
    store: SDXLAttentionStore,
    res: int = 16,
    from_where: Tuple[str, ...] = ("down", "mid", "up"),
) -> torch.Tensor:
    """
    Aggregate cross-attention maps across layers at a given spatial resolution.

    Returns shape: (res, res, seq_len), averaged over all matching layers/heads.
    If an exact spatial resolution is not available, square maps from the
    requested UNet regions are resized to `res x res` before averaging.
    """
    out = []
    attn_maps = store.get_attention()

    for location in from_where:
        key = f"{location}_cross"
        if key not in attn_maps:
            continue
        for item in attn_maps[key]:
            num_pixels = item.shape[1]
            side = int(round(num_pixels ** 0.5))
            if side * side != num_pixels:
                continue

            # item shape: (heads*batch, num_pixels, seq_len)
            cross_maps = item.reshape(-1, side, side, item.shape[-1])
            if side != res:
                cross_maps = cross_maps.permute(0, 3, 1, 2)
                cross_maps = F.interpolate(
                    cross_maps.float(),
                    size=(res, res),
                    mode="bilinear",
                    align_corners=False,
                )
                cross_maps = cross_maps.permute(0, 2, 3, 1).to(item.dtype)
            out.append(cross_maps)

    if not out:
        return torch.zeros(res, res, 1)

    out = torch.cat(out, dim=0)        # (total_heads, res, res, seq_len)
    out = out.mean(dim=0)              # (res, res, seq_len)
    return out


def _normalize_token_groups(token_groups) -> List[List[int]]:
    """Coerce token positions into a list-of-lists phrase/group representation."""
    if token_groups is None:
        return []
    if len(token_groups) == 0:
        return []
    first = token_groups[0]
    if isinstance(first, (list, tuple)):
        return [[int(idx) for idx in group if int(idx) >= 0] for group in token_groups]
    return [[int(idx)] for idx in token_groups if int(idx) >= 0]


def token_attn_map(
    store: SDXLAttentionStore,
    token_groups,
    res: int = 16,
) -> torch.Tensor:
    """
    Extract and average spatial attention maps for specific token groups.

    Args:
        token_groups: either a flat list of 1-based token positions or a list of
            token spans/groups, one per concept phrase.
        res: spatial resolution to extract

    Returns shape: (len(token_groups), res, res)
    """
    full_map = aggregate_cross_attn(store, res=res)   # (res, res, seq_len)
    groups = _normalize_token_groups(token_groups)
    maps = []
    for group in groups:
        valid = [idx for idx in group if 0 < idx < full_map.shape[2]]
        if valid:
            maps.append(full_map[:, :, valid].mean(dim=2))
        else:
            maps.append(torch.zeros(res, res, dtype=full_map.dtype))
    if not maps:
        return torch.zeros(0, res, res, dtype=full_map.dtype)
    return torch.stack(maps, dim=0)   # (n_tokens, res, res)


def get_token_groups(tokenizer, prompt: str, phrases: List[str]) -> List[List[int]]:
    """
    Return 1-based CLIP token spans for the first occurrence of each phrase.

    The CLIP tokenizer places BOS at position 0; content tokens start at 1.
    Subword-tokenised phrases may span multiple tokens — we return the full span.

    Args:
        tokenizer: CLIP tokenizer (tokenizer_1 for SDXL)
        prompt: the text prompt
        phrases: list of words/phrases to locate

    Returns: list of token-position groups, one per phrase.
    """
    prompt_ids = tokenizer.encode(prompt)  # includes BOS/EOS
    content_ids = prompt_ids[1:-1]

    groups: List[List[int]] = []
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase:
            groups.append([])
            continue

        phrase_ids = tokenizer.encode(phrase, add_special_tokens=False)
        if not phrase_ids:
            groups.append([])
            continue

        found: List[int] = []
        max_start = len(content_ids) - len(phrase_ids)
        for start in range(max_start + 1):
            if content_ids[start:start + len(phrase_ids)] == phrase_ids:
                found = list(range(start + 1, start + 1 + len(phrase_ids)))
                break
        groups.append(found)
    return groups


def get_token_indices(tokenizer, prompt: str, words: List[str]) -> List[int]:
    """
    Backwards-compatible wrapper returning the first token position per phrase.
    """
    groups = get_token_groups(tokenizer, prompt, words)
    return [group[0] if group else 0 for group in groups]


# ---------------------------------------------------------------------------
# Score metric helpers
# ---------------------------------------------------------------------------

def score_rms(noise_pred: torch.Tensor) -> float:
    """Root-mean-square of the noise prediction (scale-independent magnitude)."""
    return float(noise_pred.float().pow(2).mean().sqrt())


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors, treating them as flat vectors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return float(F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item())


def score_deviation(
    noise_cond: torch.Tensor, noise_uncond: torch.Tensor
) -> torch.Tensor:
    """CFG direction vector: delta = noise_cond - noise_uncond."""
    return noise_cond.float() - noise_uncond.float()


# ---------------------------------------------------------------------------
# Attention mass (for competition metric)
# ---------------------------------------------------------------------------

def attention_mass(
    store: SDXLAttentionStore,
    token_groups,
    res: int = 16,
) -> float:
    """
    Mean spatial attention to one or more token groups, averaged over group maps.
    Returns a scalar in [0, 1] (attention probability mass).
    """
    maps = token_attn_map(store, token_groups, res=res)   # (n, res, res)
    if maps.numel() == 0:
        return 0.0
    return float(maps.mean().item())
