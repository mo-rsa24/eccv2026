"""
Score-Space Visualization for Composability Research
=====================================================

Visualizes what the score/noise-prediction functions are doing at each
denoising timestep for the four distributions under study:

  P(A)        — solo concept A
  P(B)        — solo concept B
  P(A∧B)      — monolithic "A and B" prompt
  PoE         — Product of Experts: uncond + gs*(s_A-uncond) + gs*(s_B-uncond)

Produces five plots:
  1. score_magnitude.png   — ||s||_rms vs timestep per condition
  2. score_alignment.png   — cos(s_X, s_Y) for all condition pairs
  3. deviation_alignment.png — cos(δ_A, δ_B) concept conflict metric
  4. pca_basins.png        — terminal latent basins as PCA point clouds + KDE
  5. score_vectorfield.png — score vectors at t* projected to 2D via PCA

Usage:
  python scripts/visualize_scores.py \\
    --prompt-a "a butterfly" --prompt-b "a flower meadow" \\
    --num-seeds 5 --num-steps 50 --guidance-scale 7.5 --seed 42 \\
    --output-dir results/score_viz/butterfly_flower \\
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
import matplotlib.patches as mpatches

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
    score_rms, cosine_sim, score_deviation,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

CONDITIONS = ["solo_a", "solo_b", "monolithic", "poe"]

@dataclass
class ScoreRecord:
    step: int
    timestep: int
    cond: str
    noise_pred: torch.Tensor   # (1, 4, H, W) on CPU float32
    noise_uncond: torch.Tensor # same shape


@dataclass
class StepStats:
    """Aggregated stats at a single step across seeds."""
    step: int
    timestep: int
    # keyed by condition
    magnitude_mean: Dict[str, float] = field(default_factory=dict)
    magnitude_std:  Dict[str, float] = field(default_factory=dict)
    # keyed by "condA_vs_condB"
    cosine_mean: Dict[str, float] = field(default_factory=dict)
    cosine_std:  Dict[str, float] = field(default_factory=dict)
    # CFG deviation alignment
    dev_align_mean: Dict[str, float] = field(default_factory=dict)
    dev_align_std:  Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Encoding helper (reuses trajectory_dynamics_experiment_sdxl._encode_sdxl)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode(texts, tokenizer, tokenizer_2, text_encoder, text_encoder_2,
            device, height, width):
    """Encode a list of prompts with SDXL dual text encoders."""
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
# Core data collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_scores_single_run(
    prompt_a: str,
    prompt_b: str,
    seed: int,
    models: dict,
    ddim,
    guidance_scale: float,
    num_steps: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Dict[str, List[ScoreRecord]], Dict[str, torch.Tensor]]:
    """
    Run all four denoising loops from the same x_T and collect per-step score records.

    Returns:
        records_by_cond: dict mapping condition name → list of ScoreRecord (one per step)
        terminal_latents: dict mapping condition name → final latent (1,C,H,W) CPU float32
    """
    unet        = models["unet"]
    tokenizer   = models["tokenizer"]
    tokenizer_2 = models["tokenizer_2"]
    te1         = models["text_encoder"]
    te2         = models["text_encoder_2"]

    from diffusers import DDIMScheduler, EulerDiscreteScheduler

    euler = EulerDiscreteScheduler.from_config(ddim.config)
    euler.set_timesteps(num_steps)
    euler_sigma = float(getattr(euler, "init_noise_sigma", 1.0))

    # Generate shared x_T
    x_T_raw = get_latents(
        euler,
        z_channels=unet.config.in_channels,
        device=device,
        dtype=dtype,
        num_inference_steps=num_steps,
        batch_size=1,
        latent_width=width // 8,
        latent_height=height // 8,
        seed=seed,
    )
    x_T = (x_T_raw / euler_sigma).to(dtype=dtype, device=device)

    # Encode all prompts once
    monolithic_prompt = f"{prompt_a} and {prompt_b}"
    uncond_emb, uncond_kw = _encode([""] * 1,               tokenizer, tokenizer_2, te1, te2, device, height, width)
    a_emb,     a_kw      = _encode([prompt_a] * 1,          tokenizer, tokenizer_2, te1, te2, device, height, width)
    b_emb,     b_kw      = _encode([prompt_b] * 1,          tokenizer, tokenizer_2, te1, te2, device, height, width)
    mono_emb,  mono_kw   = _encode([monolithic_prompt] * 1, tokenizer, tokenizer_2, te1, te2, device, height, width)

    extra_eta = {}
    if "eta" in _inspect.signature(ddim.step).parameters:
        extra_eta["eta"] = 0.0

    def _run_loop(cond_emb, cond_kw, is_poe=False,
                  poe_a_emb=None, poe_a_kw=None,
                  poe_b_emb=None, poe_b_kw=None):
        latents = x_T.clone()
        records = []
        for i, t in enumerate(ddim.timesteps):
            lmi = ddim.scale_model_input(latents, t)
            n_unc = unet(lmi, t, encoder_hidden_states=uncond_emb,
                         added_cond_kwargs=uncond_kw).sample
            if is_poe:
                n_a = unet(lmi, t, encoder_hidden_states=poe_a_emb,
                           added_cond_kwargs=poe_a_kw).sample
                n_b = unet(lmi, t, encoder_hidden_states=poe_b_emb,
                           added_cond_kwargs=poe_b_kw).sample
                n_pred = n_unc + guidance_scale * (n_a - n_unc) + guidance_scale * (n_b - n_unc)
            else:
                n_cond = unet(lmi, t, encoder_hidden_states=cond_emb,
                              added_cond_kwargs=cond_kw).sample
                n_pred = n_unc + guidance_scale * (n_cond - n_unc)

            records.append(ScoreRecord(
                step=i,
                timestep=int(t.item()),
                cond="",  # set by caller
                noise_pred=n_pred.float().cpu(),
                noise_uncond=n_unc.float().cpu(),
            ))
            latents = ddim.step(n_pred, t, latents, **extra_eta).prev_sample
        return records, latents.float().cpu()

    print(f"  seed={seed}: solo_a ... ", end="", flush=True)
    rec_a,    lat_a    = _run_loop(a_emb,    a_kw)
    print("solo_b ... ", end="", flush=True)
    rec_b,    lat_b    = _run_loop(b_emb,    b_kw)
    print("monolithic ... ", end="", flush=True)
    rec_mono, lat_mono = _run_loop(mono_emb, mono_kw)
    print("poe ... done")
    rec_poe,  lat_poe  = _run_loop(None, None, is_poe=True,
                                   poe_a_emb=a_emb, poe_a_kw=a_kw,
                                   poe_b_emb=b_emb, poe_b_kw=b_kw)

    for r in rec_a:    r.cond = "solo_a"
    for r in rec_b:    r.cond = "solo_b"
    for r in rec_mono: r.cond = "monolithic"
    for r in rec_poe:  r.cond = "poe"

    return (
        {"solo_a": rec_a, "solo_b": rec_b, "monolithic": rec_mono, "poe": rec_poe},
        {"solo_a": lat_a, "solo_b": lat_b, "monolithic": lat_mono, "poe":  lat_poe},
    )


# ---------------------------------------------------------------------------
# Statistics aggregation
# ---------------------------------------------------------------------------

PAIR_KEYS = [
    ("solo_a",     "solo_b"),
    ("solo_a",     "monolithic"),
    ("solo_b",     "monolithic"),
    ("solo_a",     "poe"),
    ("solo_b",     "poe"),
    ("monolithic", "poe"),
]

PAIR_STYLES = {
    ("solo_a", "solo_b"):        {"color": "#ff6b6b", "ls": "-",  "lw": 2.0},
    ("monolithic", "poe"):       {"color": "#2a9d8f", "ls": "-",  "lw": 2.5},
    ("solo_a", "monolithic"):    {"color": "#e63946", "ls": "--", "lw": 1.2},
    ("solo_b", "monolithic"):    {"color": "#457b9d", "ls": "--", "lw": 1.2},
    ("solo_a", "poe"):           {"color": "#c77dff", "ls": ":",  "lw": 1.2},
    ("solo_b", "poe"):           {"color": "#7b2d8b", "ls": ":",  "lw": 1.2},
}


def _pair_label(a, b):
    return f"{CONDITION_LABELS[a]} vs {CONDITION_LABELS[b]}"


def aggregate_step_stats(
    all_records_by_seed: List[Dict[str, List[ScoreRecord]]],
) -> List[StepStats]:
    """
    Given records from N seeds, compute mean ± std of metrics at each step.
    """
    n_steps = len(all_records_by_seed[0]["solo_a"])
    timesteps = [r.timestep for r in all_records_by_seed[0]["solo_a"]]
    stats_list = []

    for i in range(n_steps):
        ss = StepStats(step=i, timestep=timesteps[i])

        # Magnitude
        for cond in CONDITIONS:
            mags = [score_rms(seed_recs[cond][i].noise_pred)
                    for seed_recs in all_records_by_seed]
            ss.magnitude_mean[cond] = float(np.mean(mags))
            ss.magnitude_std[cond]  = float(np.std(mags))

        # Cosine similarity between condition pairs
        for ca, cb in PAIR_KEYS:
            sims = [cosine_sim(seed_recs[ca][i].noise_pred,
                               seed_recs[cb][i].noise_pred)
                    for seed_recs in all_records_by_seed]
            key = f"{ca}_vs_{cb}"
            ss.cosine_mean[key] = float(np.mean(sims))
            ss.cosine_std[key]  = float(np.std(sims))

        # CFG deviation alignment: δ = noise_cond - noise_uncond
        # Key pair: δ_A vs δ_B, δ_A vs δ_mono, δ_B vs δ_mono
        dev_pairs = [
            ("solo_a",     "solo_b"),
            ("solo_a",     "monolithic"),
            ("solo_b",     "monolithic"),
        ]
        for ca, cb in dev_pairs:
            sims = [
                cosine_sim(
                    score_deviation(seed_recs[ca][i].noise_pred, seed_recs[ca][i].noise_uncond),
                    score_deviation(seed_recs[cb][i].noise_pred, seed_recs[cb][i].noise_uncond),
                )
                for seed_recs in all_records_by_seed
            ]
            key = f"dev_{ca}_vs_{cb}"
            ss.dev_align_mean[key] = float(np.mean(sims))
            ss.dev_align_std[key]  = float(np.std(sims))

        stats_list.append(ss)

    return stats_list


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _step_axis(stats: List[StepStats]):
    """Return step indices and formatted timestep labels."""
    steps = [s.step for s in stats]
    return steps


# ---------------------------------------------------------------------------
# Plot 1 — Score Magnitude vs Timestep
# ---------------------------------------------------------------------------

def plot_score_magnitude(stats: List[StepStats], save_path: Path):
    steps = _step_axis(stats)
    fig, ax = plt.subplots(figsize=(8, 4))

    for cond in CONDITIONS:
        means = [s.magnitude_mean[cond] for s in stats]
        stds  = [s.magnitude_std[cond]  for s in stats]
        color = CONDITION_COLORS[cond]
        label = CONDITION_LABELS[cond]
        ax.plot(steps, means, color=color, label=label, lw=2)
        ax.fill_between(steps,
                        [m - sd for m, sd in zip(means, stds)],
                        [m + sd for m, sd in zip(means, stds)],
                        color=color, alpha=0.15)

    ax.set_xlabel("Denoising step (0 = start from noise)")
    ax.set_ylabel(r"Score RMS  $\sqrt{\langle s^2 \rangle}$")
    ax.set_title("Score Magnitude vs Denoising Step")
    ax.legend(framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Plot 2 — Score Alignment (cosine similarity between condition pairs)
# ---------------------------------------------------------------------------

def plot_score_alignment(stats: List[StepStats], save_path: Path):
    steps = _step_axis(stats)
    fig, ax = plt.subplots(figsize=(8, 4))

    for ca, cb in PAIR_KEYS:
        key = f"{ca}_vs_{cb}"
        means = [s.cosine_mean[key] for s in stats]
        stds  = [s.cosine_std[key]  for s in stats]
        style = PAIR_STYLES[(ca, cb)]
        label = _pair_label(ca, cb)
        ax.plot(steps, means, label=label,
                color=style["color"], ls=style["ls"], lw=style["lw"])
        ax.fill_between(steps,
                        [m - sd for m, sd in zip(means, stds)],
                        [m + sd for m, sd in zip(means, stds)],
                        color=style["color"], alpha=0.12)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Score Alignment Between Conditions")
    ax.legend(fontsize=8, framealpha=0.8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Plot 3 — Deviation Alignment (concept conflict metric)
# ---------------------------------------------------------------------------

def plot_deviation_alignment(stats: List[StepStats], save_path: Path):
    steps = _step_axis(stats)
    fig, ax = plt.subplots(figsize=(8, 4))

    dev_style = {
        "dev_solo_a_vs_solo_b":        {"color": "#ff6b6b", "ls": "-",  "lw": 2.5,
                                        "label": r"cos($\delta_A$, $\delta_B$)  [concept conflict]"},
        "dev_solo_a_vs_monolithic":    {"color": "#e63946", "ls": "--", "lw": 1.5,
                                        "label": r"cos($\delta_A$, $\delta_{A\wedge B}$)"},
        "dev_solo_b_vs_monolithic":    {"color": "#457b9d", "ls": "--", "lw": 1.5,
                                        "label": r"cos($\delta_B$, $\delta_{A\wedge B}$)"},
    }

    for key, style in dev_style.items():
        means = [s.dev_align_mean[key] for s in stats]
        stds  = [s.dev_align_std[key]  for s in stats]
        ax.plot(steps, means, label=style["label"],
                color=style["color"], ls=style["ls"], lw=style["lw"])
        ax.fill_between(steps,
                        [m - sd for m, sd in zip(means, stds)],
                        [m + sd for m, sd in zip(means, stds)],
                        color=style["color"], alpha=0.12)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Cosine similarity of CFG direction vectors")
    ax.set_title("Concept Conflict: CFG Direction Alignment\n"
                 "Negative = concepts compete in score space")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Plot 4 — PCA Basin Point Cloud
# ---------------------------------------------------------------------------

def plot_pca_basins(
    terminal_latents_by_seed: List[Dict[str, torch.Tensor]],
    save_path: Path,
):
    """
    Project all terminal latents (all seeds, all conditions) to 2D via joint PCA.
    Scatter one color per condition + optional KDE contours.
    """
    from sklearn.decomposition import PCA
    try:
        from scipy.stats import gaussian_kde
        _have_kde = True
    except ImportError:
        _have_kde = False

    # Stack all latents: shape (n_seeds * n_conditions, D)
    all_flat = []
    labels = []
    for seed_lats in terminal_latents_by_seed:
        for cond in CONDITIONS:
            flat = seed_lats[cond].flatten().numpy().astype(np.float32)
            all_flat.append(flat)
            labels.append(cond)

    all_flat = np.stack(all_flat, axis=0)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(all_flat)    # (N, 2)

    # Split by condition
    pts_by_cond = {c: [] for c in CONDITIONS}
    for i, (cond, p) in enumerate(zip(labels, proj)):
        pts_by_cond[cond].append(p)

    fig, ax = plt.subplots(figsize=(7, 6))
    patches = []

    for cond in CONDITIONS:
        pts = np.array(pts_by_cond[cond])   # (n_seeds, 2)
        color = CONDITION_COLORS[cond]
        label = CONDITION_LABELS[cond]

        ax.scatter(pts[:, 0], pts[:, 1], c=color, s=60, alpha=0.85,
                   zorder=3, edgecolors="white", linewidths=0.5)

        # KDE contours
        if _have_kde and len(pts) >= 5:
            try:
                kde = gaussian_kde(pts.T, bw_method="scott")
                xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
                ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
                margin_x = (xmax - xmin) * 0.3 + 1e-6
                margin_y = (ymax - ymin) * 0.3 + 1e-6
                xx, yy = np.mgrid[xmin-margin_x:xmax+margin_x:60j,
                                   ymin-margin_y:ymax+margin_y:60j]
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(xx, yy, zz, levels=3, colors=[color],
                           alpha=0.6, linewidths=1.0)
            except Exception:
                pass

        patches.append(mpatches.Patch(color=color, label=label))

    var_ratio = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% var)")
    ax.set_title("Terminal Latent Basins (PCA)\n"
                 "Non-overlap = composability gap")
    ax.legend(handles=patches, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Plot 5 — Score Vector Field at t*
# ---------------------------------------------------------------------------

def plot_score_vectorfield(
    all_records_by_seed: List[Dict[str, List[ScoreRecord]]],
    t_star_index: int,
    save_path: Path,
):
    """
    At step t_star, project the mean score vectors for all conditions +
    uncond into 2D via joint PCA and draw as arrows from the origin.
    """
    from sklearn.decomposition import PCA

    # Average noise_pred over seeds at t_star for each condition
    mean_preds = {}
    for cond in CONDITIONS:
        preds = [seed_recs[cond][t_star_index].noise_pred
                 for seed_recs in all_records_by_seed]
        mean_preds[cond] = torch.stack(preds, dim=0).mean(dim=0)

    # Also grab uncond (same for all conditions; use solo_a's uncond)
    uncond_preds = [seed_recs["solo_a"][t_star_index].noise_uncond
                    for seed_recs in all_records_by_seed]
    mean_preds["uncond"] = torch.stack(uncond_preds, dim=0).mean(dim=0)

    cond_order = CONDITIONS + ["uncond"]
    vecs = np.stack([mean_preds[c].flatten().numpy().astype(np.float32)
                     for c in cond_order], axis=0)   # (5, D)

    pca = PCA(n_components=2)
    proj = pca.fit_transform(vecs)   # (5, 2)

    # Compute raw magnitudes before normalisation
    magnitudes = [score_rms(mean_preds[c]) for c in cond_order]

    # Normalise to unit length for direction clarity
    norms = np.linalg.norm(proj, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    proj_unit = proj / norms

    fig, ax = plt.subplots(figsize=(6, 6))

    for i, cond in enumerate(cond_order):
        color = CONDITION_COLORS.get(cond, "#888888")
        label = f"{CONDITION_LABELS.get(cond, cond)}\n(‖s‖={magnitudes[i]:.3f})"
        dx, dy = proj_unit[i, 0], proj_unit[i, 1]
        ax.quiver(0, 0, dx, dy, angles="xy", scale_units="xy", scale=1,
                  color=color, width=0.012, headwidth=4, headlength=5,
                  label=label, alpha=0.9)
        # Label at tip
        ax.text(dx * 1.08, dy * 1.08, CONDITION_LABELS.get(cond, cond),
                color=color, fontsize=9, ha="center", va="center")

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.4)
    ax.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.4)

    t_val = all_records_by_seed[0]["solo_a"][t_star_index].timestep
    var_ratio = pca.explained_variance_ratio_
    ax.set_title(f"Score Vectors at t={t_val} (PCA, step {t_star_index})\n"
                 f"PC1={var_ratio[0]*100:.1f}% PC2={var_ratio[1]*100:.1f}%\n"
                 "Arrows normalised to unit length; magnitudes in labels")
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

ALL_PLOTS = {"magnitude", "alignment", "deviation", "pca", "vectorfield"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize score-space dynamics for compositional diffusion"
    )
    p.add_argument("--prompt-a", required=True, help="Concept A prompt")
    p.add_argument("--prompt-b", required=True, help="Concept B prompt")
    p.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--num-seeds",  type=int,   default=5)
    p.add_argument("--num-steps",  type=int,   default=50)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--output-dir", default="results/score_viz")
    p.add_argument("--plots",      nargs="+",  default=["all"],
                   help=f"Which plots: all or any of {sorted(ALL_PLOTS)}")
    p.add_argument("--height",     type=int,   default=1024)
    p.add_argument("--width",      type=int,   default=1024)
    p.add_argument("--device",     default="cuda")
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

    print(f"Loading SDXL models from {args.model_id} ...")
    models = get_sd_models(model_id=args.model_id, dtype=dtype, device=device)

    from diffusers import DDIMScheduler, EulerDiscreteScheduler
    ddim = DDIMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    ddim.set_timesteps(args.num_steps)

    plots_to_run = resolve_plots(args.plots)
    print(f"Plots to generate: {sorted(plots_to_run)}")
    print(f"Running {args.num_seeds} seed(s): {args.seed} ... {args.seed + args.num_seeds - 1}")
    print(f"Pair: '{args.prompt_a}'  ∧  '{args.prompt_b}'")

    all_records_by_seed: List[Dict[str, List[ScoreRecord]]] = []
    terminal_latents_by_seed: List[Dict[str, torch.Tensor]] = []

    for seed_offset in range(args.num_seeds):
        seed = args.seed + seed_offset
        print(f"Seed {seed} ({seed_offset+1}/{args.num_seeds}):")
        recs, lats = collect_scores_single_run(
            prompt_a=args.prompt_a,
            prompt_b=args.prompt_b,
            seed=seed,
            models=models,
            ddim=ddim,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            height=args.height,
            width=args.width,
            device=device,
            dtype=dtype,
        )
        all_records_by_seed.append(recs)
        terminal_latents_by_seed.append(lats)

    # Aggregate step statistics
    print("Aggregating statistics ...")
    stats = aggregate_step_stats(all_records_by_seed)

    t_star = args.num_steps // 2

    if "magnitude" in plots_to_run:
        plot_score_magnitude(stats, output_dir / "score_magnitude.png")

    if "alignment" in plots_to_run:
        plot_score_alignment(stats, output_dir / "score_alignment.png")

    if "deviation" in plots_to_run:
        plot_deviation_alignment(stats, output_dir / "deviation_alignment.png")

    if "pca" in plots_to_run:
        plot_pca_basins(terminal_latents_by_seed, output_dir / "pca_basins.png")

    if "vectorfield" in plots_to_run:
        plot_score_vectorfield(all_records_by_seed, t_star,
                               output_dir / "score_vectorfield.png")

    print(f"\nAll done. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
