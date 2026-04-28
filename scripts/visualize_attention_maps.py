"""
Cross-Attention Map Visualization for Composability Research
=============================================================

Visualizes what the UNet cross-attention is attending to at each denoising
timestep for the four distributions under study:

  P(A)        — solo concept A
  P(B)        — solo concept B
  P(A∧B)      — monolithic "A and B" prompt
  PoE         — Product of Experts (separate A pass + separate B pass)

Key architectural insight: PoE has NO joint cross-attention pass. A and B
tokens never compete in the same attention layer. This is the mechanistic
explanation for why PoE differs from the monolithic P(A∧B).

Produces five plots:
  1. token_maps.png          — spatial attention heatmaps × conditions × timesteps
  2. attention_timeline.png  — filmstrip of monolithic A∧B attention across all steps
  3. attention_competition.png — concept attention mass (A and B) vs timestep
  4. attention_disagreement.png — D_t = cosine distance(map_A, map_B) + mass ratio
  5. group_compare.png       — Group 1 (co-occurrence) vs Group 4 (adversarial)

Usage:
  python scripts/visualize_attention_maps.py \\
    --prompt-a "a butterfly" --prompt-b "a flower meadow" \\
    --tokens-a "butterfly" --tokens-b "flower" "meadow" \\
    --num-steps 50 --snapshot-steps 5 25 45 \\
    --output-dir results/attention_viz/butterfly_flower \\
    --plots all
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import inspect as _inspect

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from notebooks.utils import get_sd_models
from notebooks.dynamics import get_latents
from notebooks.composition_experiments import get_prompt_conditioning
from scripts.plots.attention_utils import (
    CONDITION_COLORS, CONDITION_LABELS,
    SDXLAttentionStore,
    register_hooks, restore_processors,
    capture_attention_at_step,
    aggregate_cross_attn, token_attn_map, get_token_groups,
    attention_mass,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepAttentionSnapshot:
    step_index: int
    timestep: int
    cross_attn_map: np.ndarray           # (attn_res, attn_res, seq_len)
    latent: Optional[np.ndarray] = None  # (C, H, W) float32 CPU — stored for VAE decode


@dataclass
class ConditionAttentionData:
    """All attention data collected for one denoising condition."""
    condition: str
    # Full spatial maps at snapshot steps only
    snapshots: List[StepAttentionSnapshot] = field(default_factory=list)
    # Scalar attention mass for A- and B-concept tokens at EVERY step
    concept_a_mass_per_step: List[float] = field(default_factory=list)
    concept_b_mass_per_step: List[float] = field(default_factory=list)
    timesteps: List[int] = field(default_factory=list)
    # Aggregated cross-attention maps at EVERY step (for D_t computation)
    # Shape per entry: (attn_res, attn_res, seq_len) float32
    full_maps_per_step: List[np.ndarray] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Encoding helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode(texts, tokenizer, tokenizer_2, text_encoder, text_encoder_2,
            device, height, width):
    emb, cond_kwargs = get_prompt_conditioning(
        texts[0],
        batch_size=len(texts),
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=device,
        height=height,
        width=width,
        tokenizer_2=tokenizer_2,
        text_encoder_2=text_encoder_2,
    )
    return emb, cond_kwargs


# ---------------------------------------------------------------------------
# Core denoising loop with attention capture
# ---------------------------------------------------------------------------

def run_denoising_with_attention_capture(
    latents_init: torch.Tensor,
    prompt_a: str,
    prompt_b: str,
    monolithic_prompt: str,
    ddim,
    unet,
    tokenizer,
    tokenizer_2,
    text_encoder,
    text_encoder_2,
    guidance_scale: float,
    num_steps: int,
    snapshot_steps: List[int],
    token_indices_a: List[int],
    token_indices_b: List[int],
    token_indices_a_mono: List[int],
    token_indices_b_mono: List[int],
    attn_res: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    vae=None,
) -> Dict[str, ConditionAttentionData]:
    """
    Runs four separate denoising loops (solo_a, solo_b, monolithic, poe).
    At EVERY step: capture attention for competition scalars.
    At snapshot_steps: also save full (attn_res, attn_res, seq_len) maps.

    PoE has two separate attention stores per step (pass-A and pass-B).
    All conditions share the same x_T from latents_init.

    Returns dict with keys: "solo_a", "solo_b", "monolithic", "poe_a", "poe_b"
    """
    uncond_emb, uncond_kw = _encode([""] * 1,               tokenizer, tokenizer_2,
                                    text_encoder, text_encoder_2, device, height, width)
    a_emb,     a_kw      = _encode([prompt_a] * 1,          tokenizer, tokenizer_2,
                                    text_encoder, text_encoder_2, device, height, width)
    b_emb,     b_kw      = _encode([prompt_b] * 1,          tokenizer, tokenizer_2,
                                    text_encoder, text_encoder_2, device, height, width)
    mono_emb,  mono_kw   = _encode([monolithic_prompt] * 1, tokenizer, tokenizer_2,
                                    text_encoder, text_encoder_2, device, height, width)

    extra_eta = {}
    if "eta" in _inspect.signature(ddim.step).parameters:
        extra_eta["eta"] = 0.0

    snapshot_set = set(snapshot_steps)
    cond_names = ["solo_a", "solo_b", "monolithic", "poe_a", "poe_b"]
    results: Dict[str, ConditionAttentionData] = {
        c: ConditionAttentionData(condition=c) for c in cond_names
    }

    def _collect_from_store(
        store: SDXLAttentionStore,
        cond_key: str,
        step_i: int,
        t_val: int,
        save_full_map: bool,
        current_latent: Optional[torch.Tensor] = None,
    ):
        """Read scalars (always) and optionally full map + latent from a captured store."""
        cd = results[cond_key]
        if cond_key == "monolithic":
            indices_a = token_indices_a_mono
            indices_b = token_indices_b_mono
        else:
            indices_a = token_indices_a
            indices_b = token_indices_b
        mass_a = attention_mass(store, indices_a, res=attn_res)
        mass_b = attention_mass(store, indices_b, res=attn_res)
        cd.concept_a_mass_per_step.append(mass_a)
        cd.concept_b_mass_per_step.append(mass_b)
        cd.timesteps.append(t_val)
        # Always store aggregated map for poe passes (needed for D_t)
        if cond_key in ("poe_a", "poe_b"):
            full_map_step = aggregate_cross_attn(store, res=attn_res).numpy()
            cd.full_maps_per_step.append(full_map_step)
        if save_full_map:
            full_map = aggregate_cross_attn(store, res=attn_res).numpy()
            lat_cpu = (current_latent.float().cpu().numpy()
                       if current_latent is not None else None)
            cd.snapshots.append(StepAttentionSnapshot(
                step_index=step_i,
                timestep=t_val,
                cross_attn_map=full_map,
                latent=lat_cpu,
            ))

    # -----------------------------------------------------------------------
    # Loop factory — runs one denoising loop, capturing attention at each step
    # -----------------------------------------------------------------------

    def run_loop(
        cond_keys: List[str],
        emb_list: List[torch.Tensor],
        kw_list: List[Optional[dict]],
        is_poe: bool,
    ):
        """
        cond_keys: condition name(s) to store into (1 for solo/mono, 2 for poe)
        emb_list/kw_list: embeddings per condition key
        is_poe: if True, composite score is PoE; each emb gets its own capture
        """
        latents = latents_init.clone().to(dtype=dtype, device=device)

        for i, t in enumerate(ddim.timesteps):
            lmi = ddim.scale_model_input(latents, t)
            save_map = (i in snapshot_set)

            # Capture attention for each conditioning pass
            for ck, emb, kw in zip(cond_keys, emb_list, kw_list):
                store = capture_attention_at_step(unet, lmi, t, emb,
                                                  added_cond_kwargs=kw)
                _collect_from_store(store, ck, i, int(t.item()), save_map,
                                    current_latent=latents if save_map else None)

            # Actual denoising step (no hooks — just score computation)
            with torch.no_grad():
                n_unc = unet(lmi, t, encoder_hidden_states=uncond_emb,
                             added_cond_kwargs=uncond_kw).sample
                if is_poe:
                    n_a = unet(lmi, t, encoder_hidden_states=emb_list[0],
                               added_cond_kwargs=kw_list[0]).sample
                    n_b = unet(lmi, t, encoder_hidden_states=emb_list[1],
                               added_cond_kwargs=kw_list[1]).sample
                    n_pred = (n_unc
                              + guidance_scale * (n_a - n_unc)
                              + guidance_scale * (n_b - n_unc))
                else:
                    n_cond = unet(lmi, t, encoder_hidden_states=emb_list[0],
                                  added_cond_kwargs=kw_list[0]).sample
                    n_pred = n_unc + guidance_scale * (n_cond - n_unc)

            latents = ddim.step(n_pred, t, latents, **extra_eta).prev_sample

    print("  solo_a ...")
    run_loop(["solo_a"], [a_emb],   [a_kw],   is_poe=False)
    print("  solo_b ...")
    run_loop(["solo_b"], [b_emb],   [b_kw],   is_poe=False)
    print("  monolithic ...")
    run_loop(["monolithic"], [mono_emb], [mono_kw], is_poe=False)
    print("  poe (pass-A and pass-B per step) ...")
    run_loop(["poe_a", "poe_b"], [a_emb, b_emb], [a_kw, b_kw], is_poe=True)

    return results


# ---------------------------------------------------------------------------
# Attention map normalisation helper
# ---------------------------------------------------------------------------

def _norm_map(arr: np.ndarray) -> np.ndarray:
    """Normalise a 2D array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _snapshot_map_for_group(
    snap: Optional[StepAttentionSnapshot],
    token_group: List[int],
    attn_res: int,
) -> np.ndarray:
    """Average the attention map over a phrase token span from a stored snapshot."""
    if snap is None or not token_group:
        return np.zeros((attn_res, attn_res), dtype=np.float32)
    valid = [idx for idx in token_group if 0 < idx < snap.cross_attn_map.shape[2]]
    if not valid:
        return np.zeros((attn_res, attn_res), dtype=np.float32)
    return snap.cross_attn_map[:, :, valid].mean(axis=2)


def _safe_retention(value: float, baseline: float) -> float:
    """Ratio against the solo baseline, guarded against divide-by-zero."""
    if baseline <= 1e-8:
        return float("nan")
    return value / baseline


def _format_metric(label: str, value: float) -> str:
    if np.isnan(value):
        return f"{label}=n/a"
    return f"{label}={value:.2f}x"


# ---------------------------------------------------------------------------
# VAE decode + composite helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_latent(vae, latent_np: np.ndarray, device, dtype) -> np.ndarray:
    """
    Decode a single latent to an RGB image via the VAE.

    Accepts either:
      - (C, H, W)
      - (1, C, H, W)

    SDXL VAE is kept in fp32 for numerical stability during decode.
    """
    # Always decode in fp32 — SDXL VAE can produce NaNs in fp16
    vae.to(device=device, dtype=torch.float32)
    lat = torch.from_numpy(latent_np).to(device=device, dtype=torch.float32)
    if lat.ndim == 3:
        lat = lat.unsqueeze(0)
    elif lat.ndim != 4:
        raise ValueError(f"Expected latent with 3 or 4 dims, got shape {tuple(lat.shape)}")
    lat = torch.nan_to_num(lat, nan=0.0, posinf=0.0, neginf=0.0)
    shift_factor = getattr(vae.config, "shift_factor", None) or 0.0
    decoded = vae.decode(lat / vae.config.scaling_factor + shift_factor).sample
    decoded = torch.nan_to_num(decoded, nan=0.0, posinf=1.0, neginf=-1.0)
    if decoded.ndim != 4 or decoded.shape[0] != 1:
        raise ValueError(f"Expected decoded latent batch of shape (1, C, H, W), got {tuple(decoded.shape)}")
    img = decoded[0].float().cpu().numpy()
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=-1.0)
    img = np.clip((img * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    return np.transpose(img, (1, 2, 0))   # (H, W, 3)


def composite_attn_over_image(
    image_rgb: np.ndarray,    # (H, W, 3) uint8
    attn_map: np.ndarray,     # (res, res) float in [0,1]
    alpha: float = 0.55,
    cmap_name: str = "hot",
) -> np.ndarray:
    """
    Overlay a normalised attention heatmap on top of an RGB image.
    The heatmap is upsampled to match image size, then alpha-composited.
    Returns uint8 (H, W, 3).
    """
    from PIL import Image as _PILImage

    H, W = image_rgb.shape[:2]

    # Upsample attention map to image resolution
    attn_pil = _PILImage.fromarray((attn_map * 255).astype(np.uint8), mode="L")
    attn_up = np.array(attn_pil.resize((W, H), _PILImage.BILINEAR)) / 255.0  # (H,W) float

    # Apply colormap
    cmap = plt.get_cmap(cmap_name)
    heat_rgba = (cmap(attn_up) * 255).astype(np.uint8)   # (H, W, 4)
    heat_rgb  = heat_rgba[:, :, :3]

    # Alpha composite: out = alpha * heat + (1-alpha) * image
    out = (alpha * heat_rgb.astype(float) +
           (1 - alpha) * image_rgb.astype(float))
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Plot 1 — Token Attention Map Grid
# ---------------------------------------------------------------------------

def plot_token_attention_maps(
    attn_data: Dict[str, ConditionAttentionData],
    token_groups_a: List[List[int]],
    token_groups_b: List[List[int]],
    token_groups_a_mono: List[List[int]],
    token_groups_b_mono: List[List[int]],
    token_names_a: List[str],
    token_names_b: List[str],
    snapshot_steps: List[int],
    attn_res: int,
    save_path: Path,
    vae=None,
    device=None,
    dtype=None,
):
    """
    Grid: rows = (condition × token), columns = snapshot timestep.

    Row order:
      P(A) tokens_a
      P(B) tokens_b
      P(A∧B) tokens_a, tokens_b
      PoE pass-A tokens_a
      PoE pass-B tokens_b
    """
    # Build row specs: (cond_key, token_group, row_label)
    row_specs = []
    for name, group in zip(token_names_a, token_groups_a):
        row_specs.append(("solo_a", group, f"P(A): {name}", CONDITION_COLORS["solo_a"]))
    for name, group in zip(token_names_b, token_groups_b):
        row_specs.append(("solo_b", group, f"P(B): {name}", CONDITION_COLORS["solo_b"]))
    for name, group in zip(token_names_a, token_groups_a_mono):
        row_specs.append(("monolithic", group, f"P(A∧B): {name}", CONDITION_COLORS["monolithic"]))
    for name, group in zip(token_names_b, token_groups_b_mono):
        row_specs.append(("monolithic", group, f"P(A∧B): {name}", CONDITION_COLORS["monolithic"]))
    for name, group in zip(token_names_a, token_groups_a):
        row_specs.append(("poe_a", group, f"PoE pass-A: {name}", CONDITION_COLORS["poe_a"]))
    for name, group in zip(token_names_b, token_groups_b):
        row_specs.append(("poe_b", group, f"PoE pass-B: {name}", CONDITION_COLORS["poe_b"]))

    nrows = len(row_specs)
    ncols = len(snapshot_steps)

    # Build snapshot index lookup: step_index → snapshot position
    snap_lookup: Dict[str, Dict[int, StepAttentionSnapshot]] = {}
    for cond_key, cdata in attn_data.items():
        snap_lookup[cond_key] = {s.step_index: s for s in cdata.snapshots}

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3 * ncols + 1, 1.8 * nrows),
                             squeeze=False)

    # Column labels: timestep labels
    all_timesteps = attn_data["solo_a"].timesteps
    t_labels = []
    for si in snapshot_steps:
        if si < len(all_timesteps):
            frac = 1.0 - si / max(len(all_timesteps) - 1, 1)
            t_labels.append(f"t≈{frac:.0%}T\n(step {si})")
        else:
            t_labels.append(f"step {si}")

    for col_j, (si, t_label) in enumerate(zip(snapshot_steps, t_labels)):
        axes[0, col_j].set_title(t_label, fontsize=9, pad=4)

    for row_i, (cond_key, token_group, row_label, row_color) in enumerate(row_specs):
        for col_j, si in enumerate(snapshot_steps):
            ax = axes[row_i, col_j]
            snap = snap_lookup.get(cond_key, {}).get(si)
            raw_map = _snapshot_map_for_group(snap, token_group, attn_res)
            if np.any(raw_map):
                normed  = _norm_map(raw_map)
                if vae is not None and snap.latent is not None:
                    img_rgb  = decode_latent(vae, snap.latent, device, dtype)
                    composite = composite_attn_over_image(img_rgb, normed)
                    ax.imshow(composite, aspect="equal", interpolation="bilinear")
                else:
                    ax.imshow(normed, cmap="hot", vmin=0, vmax=1,
                              aspect="equal", interpolation="nearest")
            else:
                ax.imshow(np.zeros((attn_res, attn_res)), cmap="hot",
                          vmin=0, vmax=1, aspect="equal")

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(row_color)
                spine.set_linewidth(1.8)

        # Row label on the left
        axes[row_i, 0].set_ylabel(row_label, fontsize=8, rotation=0,
                                   ha="right", va="center", labelpad=4,
                                   color=row_color)

    fig.suptitle("Cross-Attention Maps per Condition and Timestep",
                 fontsize=11, y=1.01)

    # Shared colorbar
    cbar_ax = fig.add_axes([1.01, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="hot", norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Norm. attention")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 2 — Attention Timeline for P(A∧B)
# ---------------------------------------------------------------------------

def plot_attention_timeline(
    mono_data: ConditionAttentionData,
    token_groups_a_mono: List[List[int]],
    token_groups_b_mono: List[List[int]],
    token_names_a: List[str],
    token_names_b: List[str],
    attn_res: int,
    save_path: Path,
    vae=None,
    device=None,
    dtype=None,
):
    """
    Filmstrip showing how A-token and B-token attention evolves across all
    snapshot steps in the monolithic P(A∧B) pass.
    """
    snap_by_step = {s.step_index: s for s in mono_data.snapshots}
    sorted_snaps = sorted(snap_by_step.values(), key=lambda s: s.step_index)

    if not sorted_snaps:
        print("  Warning: no snapshots for monolithic — skipping timeline plot.")
        return

    # Tokens to show: A tokens then B tokens
    all_token_groups  = token_groups_a_mono + token_groups_b_mono
    all_token_names   = token_names_a + token_names_b

    nrows = len(all_token_groups)
    ncols = len(sorted_snaps)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 1.2, nrows * 1.5 + 0.6),
                             squeeze=False)

    for col_j, snap in enumerate(sorted_snaps):
        frac = 1.0 - snap.step_index / max(len(mono_data.timesteps) - 1, 1)
        col_title = f"{frac:.0%}T"
        axes[0, col_j].set_title(col_title, fontsize=7, pad=3)

        for row_i, (token_group, tok_name) in enumerate(zip(all_token_groups, all_token_names)):
            ax = axes[row_i, col_j]
            raw = _snapshot_map_for_group(snap, token_group, attn_res)
            if np.any(raw):
                normed = _norm_map(raw)
                if vae is not None and snap.latent is not None:
                    img_rgb   = decode_latent(vae, snap.latent, device, dtype)
                    composite = composite_attn_over_image(img_rgb, normed)
                    ax.imshow(composite, aspect="equal", interpolation="bilinear")
                else:
                    ax.imshow(normed, cmap="hot", vmin=0, vmax=1,
                              aspect="equal", interpolation="nearest")
            else:
                ax.imshow(np.zeros((attn_res, attn_res)), cmap="hot",
                          vmin=0, vmax=1, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])

    # Row labels
    for row_i, tok_name in enumerate(all_token_names):
        color = (CONDITION_COLORS["solo_a"] if row_i < len(token_names_a)
                 else CONDITION_COLORS["solo_b"])
        axes[row_i, 0].set_ylabel(tok_name, fontsize=9, rotation=0,
                                   ha="right", va="center", labelpad=4,
                                   color=color)

    fig.suptitle("P(A∧B) Attention Timeline: how token attention evolves across denoising",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 3 — Attention Competition Metric
# ---------------------------------------------------------------------------

def plot_attention_competition(
    attn_data: Dict[str, ConditionAttentionData],
    token_names_a: List[str],
    token_names_b: List[str],
    save_path: Path,
):
    """
    2-row subplot showing concept attention mass vs denoising step.

    Key insight: in P(A∧B), mass_A and mass_B compete (joint softmax).
    In PoE, each pass has independent attention — no competition.
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    label_a = " + ".join(token_names_a)
    label_b = " + ".join(token_names_b)

    # Conditions to plot: solo_a (has mass_a), solo_b (has mass_b),
    # monolithic (has both), poe_a (has mass_a), poe_b (has mass_b)
    plot_specs = [
        # (cond_key, which_mass, ax, label_suffix)
        ("solo_a",     "a", ax_top, "P(A)"),
        ("monolithic", "a", ax_top, "P(A∧B)"),
        ("poe_a",      "a", ax_top, "PoE pass-A"),
        ("solo_b",     "b", ax_bot, "P(B)"),
        ("monolithic", "b", ax_bot, "P(A∧B)"),
        ("poe_b",      "b", ax_bot, "PoE pass-B"),
    ]

    line_styles = {
        "solo_a":     {"ls": "-",  "lw": 1.8},
        "solo_b":     {"ls": "-",  "lw": 1.8},
        "monolithic": {"ls": "-",  "lw": 2.2},
        "poe_a":      {"ls": "--", "lw": 1.6},
        "poe_b":      {"ls": "--", "lw": 1.6},
    }

    for cond_key, which_mass, ax, label_suffix in plot_specs:
        cdata = attn_data.get(cond_key)
        if cdata is None:
            continue
        steps = list(range(len(cdata.timesteps)))
        mass = (cdata.concept_a_mass_per_step if which_mass == "a"
                else cdata.concept_b_mass_per_step)
        color_key = cond_key.replace("_a", "").replace("_b", "")
        # Use slightly different shades for poe_a vs poe_b
        color = CONDITION_COLORS.get(cond_key, CONDITION_COLORS.get(color_key, "#888888"))
        style = line_styles.get(cond_key, {})
        ax.plot(steps, mass, color=color, label=label_suffix,
                lw=style.get("lw", 1.5), ls=style.get("ls", "-"))

    ax_top.set_ylabel(f"Attention mass: {label_a}\n(concept A)", fontsize=9)
    ax_bot.set_ylabel(f"Attention mass: {label_b}\n(concept B)", fontsize=9)
    ax_bot.set_xlabel("Denoising step (0 = start from noise)")

    for ax in (ax_top, ax_bot):
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, framealpha=0.8, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Concept Attention Mass vs Denoising Step\n"
        "P(A∧B): attention competes (joint softmax) — PoE: each pass is independent",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 3b — Attention-Space Disagreement D_t
# ---------------------------------------------------------------------------

def plot_attention_disagreement(
    attn_data: Dict[str, ConditionAttentionData],
    save_path: Path,
):
    """
    Plot D_t = 1 - cosine_similarity(attn_A(x_t), attn_B(x_t)) over denoising steps.

    Both maps are captured at the same latent state x_t.
    - Low D_t → A and B are attending to the same spatial regions (competition / overlap).
    - High D_t → A and B have spatially separated (PoE has "committed" to a layout).

    Also plots:
    - The monolithic analogue: cosine distance between A-token map and B-token map
      within the same joint attention pass (true competition signal).
    - Attention mass ratio A/(A+B) per step for PoE and monolithic (flat = no trade-off).
    """
    import torch.nn.functional as F

    poe_a_data = attn_data.get("poe_a")
    poe_b_data = attn_data.get("poe_b")
    mono_data  = attn_data.get("monolithic")

    if poe_a_data is None or poe_b_data is None:
        print("  Warning: poe_a or poe_b data missing — skipping disagreement plot.")
        return

    n_steps = min(len(poe_a_data.full_maps_per_step),
                  len(poe_b_data.full_maps_per_step))
    if n_steps == 0:
        print("  Warning: no full_maps_per_step stored — skipping disagreement plot.")
        return

    steps = list(range(n_steps))

    # D_t for PoE: cosine distance between the two independent pass maps
    # Flatten (res, res, seq_len) → 1D vector for cosine similarity
    poe_dt = []
    for ma, mb in zip(poe_a_data.full_maps_per_step[:n_steps],
                      poe_b_data.full_maps_per_step[:n_steps]):
        va = torch.from_numpy(ma).float().flatten()
        vb = torch.from_numpy(mb).float().flatten()
        sim = float(F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item())
        poe_dt.append(1.0 - sim)

    # Attention mass ratio A/(A+B) — flat in PoE means no layout trade-off
    def _ratio(mass_a_list, mass_b_list, n):
        ratios = []
        for a, b in zip(mass_a_list[:n], mass_b_list[:n]):
            total = a + b
            ratios.append(a / total if total > 1e-8 else 0.5)
        return ratios

    poe_ratio   = _ratio(poe_a_data.concept_a_mass_per_step,
                         poe_b_data.concept_b_mass_per_step, n_steps)
    mono_ratio  = (_ratio(mono_data.concept_a_mass_per_step,
                          mono_data.concept_b_mass_per_step,
                          min(n_steps, len(mono_data.concept_a_mass_per_step)))
                   if mono_data else None)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # Top: D_t
    ax_top.plot(steps, poe_dt,
                color=CONDITION_COLORS.get("poe_a", "#e377c2"),
                lw=2.0, label="PoE: D_t (cosine distance A vs B)")
    ax_top.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax_top.set_ylabel("D_t = 1 − cos(map_A, map_B)", fontsize=9)
    ax_top.set_ylim(bottom=0)
    ax_top.legend(fontsize=8, framealpha=0.8)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Annotate: low D_t = maps similar (both concepts fight for same region)
    #           high D_t = maps differ (spatial separation achieved)
    ax_top.annotate("low → overlap\n(no separation)", xy=(n_steps * 0.05, 0.02),
                    fontsize=7, color="gray", va="bottom")
    ax_top.annotate("high → separated", xy=(n_steps * 0.05, max(poe_dt) * 0.85),
                    fontsize=7, color="gray", va="top")

    # Bottom: mass ratio
    ax_bot.plot(steps, poe_ratio,
                color=CONDITION_COLORS.get("poe_a", "#e377c2"),
                lw=2.0, ls="--", label="PoE: mass_A / (mass_A + mass_B)")
    if mono_ratio is not None:
        mono_steps = list(range(len(mono_ratio)))
        ax_bot.plot(mono_steps, mono_ratio,
                    color=CONDITION_COLORS.get("monolithic", "#2ca02c"),
                    lw=2.0, label="P(A∧B): mass_A / (mass_A + mass_B)")
    ax_bot.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax_bot.set_ylim(0, 1)
    ax_bot.set_ylabel("Attention mass ratio  A/(A+B)", fontsize=9)
    ax_bot.set_xlabel("Denoising step (0 = start from noise)")
    ax_bot.legend(fontsize=8, framealpha=0.8)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    fig.suptitle(
        "Attention-Space Disagreement (D_t) and Mass Ratio over Trajectory\n"
        "PoE: no token competition — D_t measures spontaneous spatial separation",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 4 — Group Comparison (Group 1 co-occurrence vs Group 4 adversarial)
# ---------------------------------------------------------------------------

def plot_group_comparison(
    pairs: List[Tuple[str, str, str, str, str]],
    # list of (prompt_a, prompt_b, token_a, token_b, group_label)
    models: dict,
    ddim,
    seed: int,
    guidance_scale: float,
    num_steps: int,
    attn_res: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    mid_step: int,
    save_path: Path,
):
    """
    2×4 grid: rows = two concept pairs (e.g. Group 1, Group 4),
    columns = P(A), P(B), P(A∧B), PoE (mean of pass-A and pass-B).

    Each cell shows the mean spatial attention to the relevant concept tokens
    at mid_step.
    """
    from diffusers import EulerDiscreteScheduler

    euler = EulerDiscreteScheduler.from_config(ddim.config)
    euler.set_timesteps(num_steps)
    euler_sigma = float(getattr(euler, "init_noise_sigma", 1.0))

    unet = models["unet"]
    tokenizer   = models["tokenizer"]
    tokenizer_2 = models["tokenizer_2"]
    te1 = models["text_encoder"]
    te2 = models["text_encoder_2"]

    nrows = len(pairs)
    ncols = 4
    col_labels = ["P(A)", "P(B)", "P(A∧B)", "PoE"]
    col_colors = [CONDITION_COLORS["solo_a"], CONDITION_COLORS["solo_b"],
                  CONDITION_COLORS["monolithic"], CONDITION_COLORS["poe"]]

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5 * ncols, 3 * nrows + 0.6),
                             squeeze=False)

    for row_i, (prompt_a, prompt_b, token_a, token_b, group_label) in enumerate(pairs):
        mono_prompt = f"{prompt_a} and {prompt_b}"
        tok_groups_a = get_token_groups(tokenizer, prompt_a, [token_a])
        tok_groups_b = get_token_groups(tokenizer, prompt_b, [token_b])
        tok_groups_a_in_mono = get_token_groups(tokenizer, mono_prompt, [token_a])
        tok_groups_b_in_mono = get_token_groups(tokenizer, mono_prompt, [token_b])

        # Generate x_T
        x_T_raw = get_latents(euler, z_channels=unet.config.in_channels,
                               device=device, dtype=dtype,
                               num_inference_steps=num_steps, batch_size=1,
                               latent_width=width // 8, latent_height=height // 8,
                               seed=seed)
        x_T = (x_T_raw / euler_sigma).to(dtype=dtype, device=device)

        uncond_emb, uncond_kw = _encode([""] * 1,         tokenizer, tokenizer_2,
                                         te1, te2, device, height, width)
        a_emb,     a_kw      = _encode([prompt_a] * 1,    tokenizer, tokenizer_2,
                                         te1, te2, device, height, width)
        b_emb,     b_kw      = _encode([prompt_b] * 1,    tokenizer, tokenizer_2,
                                         te1, te2, device, height, width)
        mono_emb,  mono_kw   = _encode([mono_prompt] * 1, tokenizer, tokenizer_2,
                                         te1, te2, device, height, width)

        extra_eta = {}
        if "eta" in _inspect.signature(ddim.step).parameters:
            extra_eta["eta"] = 0.0

        print(f"  Group comparison [{group_label}]: {prompt_a} ∧ {prompt_b}")

        # Run each condition up to mid_step, capture attention at that step
        def _run_to_step_and_capture(emb, kw, token_groups, latents):
            lats = latents.clone()
            for i, t in enumerate(ddim.timesteps):
                lmi = ddim.scale_model_input(lats, t)
                if i == mid_step:
                    store = capture_attention_at_step(unet, lmi, t, emb,
                                                      added_cond_kwargs=kw)
                    raw = token_attn_map(store, token_groups, res=attn_res)
                    if raw.numel() == 0:
                        return np.zeros((attn_res, attn_res), dtype=np.float32), 0.0
                    mean_map = raw.mean(dim=0).numpy()   # (res, res)
                    return mean_map, float(raw.mean().item())
                with torch.no_grad():
                    n_unc = unet(lmi, t, encoder_hidden_states=uncond_emb,
                                 added_cond_kwargs=uncond_kw).sample
                    n_c   = unet(lmi, t, encoder_hidden_states=emb,
                                 added_cond_kwargs=kw).sample
                    n_pred = n_unc + guidance_scale * (n_c - n_unc)
                lats = ddim.step(n_pred, t, lats, **extra_eta).prev_sample
            return np.zeros((attn_res, attn_res), dtype=np.float32), 0.0

        map_solo_a, mass_solo_a = _run_to_step_and_capture(a_emb,    a_kw,    tok_groups_a,         x_T)
        map_solo_b, mass_solo_b = _run_to_step_and_capture(b_emb,    b_kw,    tok_groups_b,         x_T)
        map_mono_a, mass_mono_a = _run_to_step_and_capture(mono_emb, mono_kw, tok_groups_a_in_mono, x_T)
        map_mono_b, mass_mono_b = _run_to_step_and_capture(mono_emb, mono_kw, tok_groups_b_in_mono, x_T)
        map_poe_a,  mass_poe_a  = _run_to_step_and_capture(a_emb,    a_kw,    tok_groups_a,         x_T)
        map_poe_b,  mass_poe_b  = _run_to_step_and_capture(b_emb,    b_kw,    tok_groups_b,         x_T)

        map_mono = (map_mono_a + map_mono_b) / 2.0
        map_poe     = (map_poe_a + map_poe_b) / 2.0

        mono_ret_a = _safe_retention(mass_mono_a, mass_solo_a)
        mono_ret_b = _safe_retention(mass_mono_b, mass_solo_b)
        poe_ret_a = _safe_retention(mass_poe_a, mass_solo_a)
        poe_ret_b = _safe_retention(mass_poe_b, mass_solo_b)
        mono_ret_mean = np.nanmean([mono_ret_a, mono_ret_b])
        poe_ret_mean = np.nanmean([poe_ret_a, poe_ret_b])
        mono_suppression = 1.0 - mono_ret_mean if not np.isnan(mono_ret_mean) else float("nan")
        poe_suppression = 1.0 - poe_ret_mean if not np.isnan(poe_ret_mean) else float("nan")

        cell_maps = [map_solo_a, map_solo_b, map_mono, map_poe]
        cell_annotations = [
            [f"A={mass_solo_a:.3f}"],
            [f"B={mass_solo_b:.3f}"],
            [
                _format_metric("A", mono_ret_a),
                _format_metric("B", mono_ret_b),
                f"supp={mono_suppression:.2f}" if not np.isnan(mono_suppression) else "supp=n/a",
            ],
            [
                _format_metric("A", poe_ret_a),
                _format_metric("B", poe_ret_b),
                f"supp={poe_suppression:.2f}" if not np.isnan(poe_suppression) else "supp=n/a",
            ],
        ]

        for col_j, (cmap, col_label, col_color, annotation_lines) in enumerate(
            zip(cell_maps, col_labels, col_colors, cell_annotations)
        ):
            ax = axes[row_i, col_j]
            normed = _norm_map(cmap)
            ax.imshow(normed, cmap="hot", vmin=0, vmax=1,
                      aspect="equal", interpolation="nearest")

            ax.text(
                0.04, 0.04, "\n".join(annotation_lines), transform=ax.transAxes,
                fontsize=7, color="white", va="bottom",
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 2, "edgecolor": "none"},
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row_i == 0:
                ax.set_title(col_label, fontsize=10, color=col_color, pad=4)
            for spine in ax.spines.values():
                spine.set_edgecolor(col_color)
                spine.set_linewidth(1.5)

        axes[row_i, 0].set_ylabel(group_label, fontsize=9, rotation=0,
                                   ha="right", va="center", labelpad=6)

    fig.suptitle(
        f"Cross-Attention Comparison at t≈mid (step {mid_step})\n"
        "Co-occurrence (G1) vs Adversarial (G4): "
        "retention of concept-token attention relative to solo baselines",
        fontsize=10,
    )

    # Shared colorbar
    cbar_ax = fig.add_axes([1.01, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="hot",
                                norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Norm. attention")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

ALL_PLOTS = {"token_maps", "timeline", "competition", "disagreement", "group_compare"}


def build_timeline_snapshot_steps(num_steps: int) -> List[int]:
    """Evenly spaced timeline steps, always including the final denoising step."""
    if num_steps <= 1:
        return [0]
    stride = max(1, num_steps // 10)
    steps = list(range(0, num_steps, stride))
    last_step = num_steps - 1
    if steps[-1] != last_step:
        steps.append(last_step)
    return sorted(set(steps))


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize cross-attention maps for compositional diffusion"
    )
    p.add_argument("--prompt-a",  required=True)
    p.add_argument("--prompt-b",  required=True)
    p.add_argument("--tokens-a",  nargs="+", default=None,
                   help="Key tokens from prompt_a (default: last word)")
    p.add_argument("--tokens-b",  nargs="+", default=None,
                   help="Key tokens from prompt_b (default: last word)")
    p.add_argument("--model-id",  default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--num-steps", type=int,   default=50)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--snapshot-steps", type=int, nargs="+", default=[5, 25, 45],
                   help="Denoising step indices at which to capture full attention maps")
    p.add_argument("--attn-res",  type=int,   default=16,
                   help="Spatial resolution for attention maps (default 16 = 16×16)")
    p.add_argument("--output-dir", default="results/attention_viz")
    p.add_argument("--plots", nargs="+", default=["all"],
                   help=f"Which plots: all or any of {sorted(ALL_PLOTS)}")
    p.add_argument("--group-compare-a", nargs=4,
                   metavar=("PROMPT_A", "PROMPT_B", "TOKEN_A", "TOKEN_B"),
                   default=["a butterfly", "a flower meadow", "butterfly", "flower"],
                   help="Group 1 pair for comparison plot")
    p.add_argument("--group-compare-b", nargs=4,
                   metavar=("PROMPT_A", "PROMPT_B", "TOKEN_A", "TOKEN_B"),
                   default=["a cat", "a dog", "cat", "dog"],
                   help="Group 4 pair for comparison plot")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width",  type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-image-underlay", action="store_true",
                   help="Skip VAE decode; show plain heatmaps instead of composited images")
    return p.parse_args()


def resolve_plots(plots_arg: List[str]) -> set:
    if "all" in plots_arg:
        return set(ALL_PLOTS)
    unknown = set(plots_arg) - ALL_PLOTS
    if unknown:
        print(f"Warning: unknown plot names {unknown}; ignoring")
    return set(plots_arg) & ALL_PLOTS


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32

    # Resolve token names
    token_names_a = args.tokens_a or [args.prompt_a.split()[-1]]
    token_names_b = args.tokens_b or [args.prompt_b.split()[-1]]

    print(f"Loading SDXL models from {args.model_id} ...")
    models = get_sd_models(model_id=args.model_id, dtype=dtype, device=device)

    from diffusers import DDIMScheduler, EulerDiscreteScheduler
    ddim = DDIMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    ddim.set_timesteps(args.num_steps)

    # Resolve token indices
    tokenizer = models["tokenizer"]
    token_groups_a = get_token_groups(tokenizer, args.prompt_a, token_names_a)
    token_groups_b = get_token_groups(tokenizer, args.prompt_b, token_names_b)
    monolithic_prompt = f"{args.prompt_a} and {args.prompt_b}"
    token_groups_a_mono = get_token_groups(tokenizer, monolithic_prompt, token_names_a)
    token_groups_b_mono = get_token_groups(tokenizer, monolithic_prompt, token_names_b)
    print(f"Token groups A ({token_names_a}): {token_groups_a}")
    print(f"Token groups B ({token_names_b}): {token_groups_b}")
    print(f"Token groups A in monolithic prompt: {token_groups_a_mono}")
    print(f"Token groups B in monolithic prompt: {token_groups_b_mono}")

    # Build x_T
    euler = EulerDiscreteScheduler.from_config(ddim.config)
    euler.set_timesteps(args.num_steps)
    euler_sigma = float(getattr(euler, "init_noise_sigma", 1.0))
    x_T_raw = get_latents(
        euler,
        z_channels=models["unet"].config.in_channels,
        device=device, dtype=dtype,
        num_inference_steps=args.num_steps,
        batch_size=1,
        latent_width=args.width // 8,
        latent_height=args.height // 8,
        seed=args.seed,
    )
    latents_init = (x_T_raw / euler_sigma).to(dtype=dtype, device=device)

    plots_to_run = resolve_plots(args.plots)
    print(f"Plots to generate: {sorted(plots_to_run)}")
    print(f"Pair: '{args.prompt_a}'  ∧  '{args.prompt_b}'")

    # Determine snapshot steps — for timeline we want dense coverage
    snapshot_steps = list(args.snapshot_steps)
    if "timeline" in plots_to_run:
        timeline_steps = build_timeline_snapshot_steps(args.num_steps)
        snapshot_steps = sorted(set(snapshot_steps) | set(timeline_steps))

    vae = None if args.no_image_underlay else models.get("vae")

    print("Running denoising with attention capture ...")
    attn_data = run_denoising_with_attention_capture(
        latents_init=latents_init,
        prompt_a=args.prompt_a,
        prompt_b=args.prompt_b,
        monolithic_prompt=monolithic_prompt,
        ddim=ddim,
        unet=models["unet"],
        tokenizer=tokenizer,
        tokenizer_2=models["tokenizer_2"],
        text_encoder=models["text_encoder"],
        text_encoder_2=models["text_encoder_2"],
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        snapshot_steps=snapshot_steps,
        token_indices_a=token_groups_a,
        token_indices_b=token_groups_b,
        token_indices_a_mono=token_groups_a_mono,
        token_indices_b_mono=token_groups_b_mono,
        attn_res=args.attn_res,
        height=args.height,
        width=args.width,
        device=device,
        dtype=dtype,
        vae=vae,
    )

    if "token_maps" in plots_to_run:
        plot_token_attention_maps(
            attn_data=attn_data,
            token_groups_a=token_groups_a,
            token_groups_b=token_groups_b,
            token_groups_a_mono=token_groups_a_mono,
            token_groups_b_mono=token_groups_b_mono,
            token_names_a=token_names_a,
            token_names_b=token_names_b,
            snapshot_steps=list(args.snapshot_steps),   # use original subset
            attn_res=args.attn_res,
            save_path=output_dir / "token_maps.png",
            vae=vae, device=device, dtype=dtype,
        )

    if "timeline" in plots_to_run:
        plot_attention_timeline(
            mono_data=attn_data["monolithic"],
            token_groups_a_mono=token_groups_a_mono,
            token_groups_b_mono=token_groups_b_mono,
            token_names_a=token_names_a,
            token_names_b=token_names_b,
            attn_res=args.attn_res,
            save_path=output_dir / "attention_timeline.png",
            vae=vae, device=device, dtype=dtype,
        )

    if "competition" in plots_to_run:
        plot_attention_competition(
            attn_data=attn_data,
            token_names_a=token_names_a,
            token_names_b=token_names_b,
            save_path=output_dir / "attention_competition.png",
        )

    if "disagreement" in plots_to_run:
        plot_attention_disagreement(
            attn_data=attn_data,
            save_path=output_dir / "attention_disagreement.png",
        )

    if "group_compare" in plots_to_run:
        gc_a = args.group_compare_a
        gc_b = args.group_compare_b
        pairs = [
            (gc_a[0], gc_a[1], gc_a[2], gc_a[3], "Group 1: co-occurrence"),
            (gc_b[0], gc_b[1], gc_b[2], gc_b[3], "Group 4: adversarial"),
        ]
        plot_group_comparison(
            pairs=pairs,
            models=models,
            ddim=ddim,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            attn_res=args.attn_res,
            height=args.height,
            width=args.width,
            device=device,
            dtype=dtype,
            mid_step=args.num_steps // 2,
            save_path=output_dir / "group_compare.png",
        )

    print(f"\nAll done. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
