"""
Gap analysis — p* plots 11–19 and 25.

  Gap validity  (data: within_and_distances.json)
    11  Within-AND noise floor — pairwise AND distances vs gap conditions (×ratio)
                                 Confirms the gap is structural, not stochastic

  p* progressive abstraction  (data: per_seed_distances.json; mirrors 00–10)
    12  p* strip by pair    — raw seed dots per p* source + mono, 4 pair panels
    13  p* grouped bars     — mean ± std per pair, bars = p* sources + mono
    14  p* histogram        — distance histogram per pair, bars by p* source
    15  p* KDE pooled       — smooth of all conditions (p* solid, baselines dashed)
                              Key visual: overlap with mono ≈ gap closed
    16  p* temporal         — per-step MSE for all p* sources + mono, pooled, mean+CI

  Closing the gap — synthesis  (data: per_seed_distances.json + all_pairs_gap.json)
    17  p* terminal strip   — terminal distance from AND: all p* sources vs baselines
    18  CLIP comparison     — per-pair CLIP sim AND↔p* vs AND↔mono (all p* sources)
    19  JS² divergence      — JS²(P_p*, P_mono) per pair; lower = p* closer to mono
    25  Combined 17+15      — side-by-side strip (left) and KDE (right), equal panel size
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

try:
    from scipy.spatial.distance import jensenshannon
except ImportError:
    jensenshannon = None

from .utils import (
    TERM_COLOR, TERM_LABEL, TRAJ_COLOR, TRAJ_LABEL,
    TERM_CONDITIONS, TRAJ_PSTAR_PRIORITY, PSTAR_GAP_KEY,
    LABEL_D_T_MSE, LABEL_D_t_MSE, LABEL_JS2,
    SCIPY_OK, stats,
    cap_pairs, get_present_poe, get_present_pstar, pair_color_map, short_pair,
    kde_pmf, traj_stats,
    hide_top_right, save_fig,
    load_all_pairs_gap, load_within_and,
)


# ===========================================================================
# Plot 11 — Within-AND noise-floor validation.
#
#   Answers the supervisor's concern: are the per-condition gaps (d_T_mono,
#   d_T_c1, d_T_c2, d_T_pstar) genuinely above the stochastic variation of
#   AND itself?
#
#   Left column  — within-AND pairwise distances:
#       d_within_and(s,s') = mean( (z_AND[s] - z_AND[s'])² )  for all s≠s'
#       These represent how much AND outputs vary across different starting
#       noises x_T — i.e. the irreducible stochastic noise floor.
#   Right columns — per-condition gaps (same pooling as plot 03/17).
#
#   Annotations show mean value and ×ratio relative to the noise floor.
#   The grey dashed line + band (mean ± 1 std of within-AND) runs across
#   all columns as a visual reference.
#
#   Requires within_and_distances.json (produced by measure_composability_gap.py).
# ===========================================================================

def plot_11(df_term, df_traj, out_dir, data_dir=None, **kw):
    if data_dir is None:
        print("  Skipping plot 11: data_dir not provided.")
        return

    raw = load_within_and(data_dir)
    if raw is None:
        print("  Skipping plot 11: within_and_distances.json not found in data_dir.")
        print("  Re-run measure_composability_gap.py to generate it.")
        return

    df_within = pd.DataFrame(raw)
    within_vals = df_within["d_within_and"].values

    # Core conditions only: mono AND, solo c₁, solo c₂ anchored to SuperDiff AND.
    # p* stays in plots 17–19 which already skip gracefully when it is absent.
    gap_conds = list(TERM_CONDITIONS) + get_present_poe(df_term)

    n_cols   = 1 + len(gap_conds)   # within-AND + gap conditions
    rng      = np.random.default_rng(42)
    WITHIN_COLOR = "#999999"

    # y-axis ceiling
    all_vals = np.concatenate([within_vals] + [df_term[c].values for c in gap_conds])
    y_top    = max(all_vals.max() * 1.22, 0.30)

    fig, ax = plt.subplots(figsize=(2.8 * n_cols, 5.8))

    # ---- Within-AND column (x = 0) ----
    x       = 0.0
    jitter  = rng.uniform(-0.18, 0.18, len(within_vals))
    ax.scatter(x + jitter, within_vals, color=WITHIN_COLOR, s=38, alpha=0.48,
               linewidths=0.8, edgecolors="white", zorder=3)
    ax.plot([x - 0.28, x + 0.28], [within_vals.mean(), within_vals.mean()],
            color=WITHIN_COLOR, lw=5.0, solid_capstyle="butt", zorder=4)
    ax.text(x, within_vals.mean() + y_top * 0.025,
            f"{within_vals.mean():.3f}", ha="center", va="bottom",
            fontsize=9, color=WITHIN_COLOR, fontweight="bold")

    # Grey reference band (mean ± 1 std) spanning the full width
    w_mean, w_std = within_vals.mean(), within_vals.std()
    ax.axhspan(w_mean - w_std, w_mean + w_std,
               alpha=0.07, color=WITHIN_COLOR, zorder=1)
    ax.axhline(w_mean, color=WITHIN_COLOR, lw=1.2, ls="--", alpha=0.55, zorder=2)

    # ---- Per-condition gap columns ----
    for xi, cond in enumerate(gap_conds, start=1):
        vals   = df_term[cond].values
        x      = float(xi)
        c      = TERM_COLOR[cond]
        jitter = rng.uniform(-0.18, 0.18, len(vals))

        ax.scatter(x + jitter, vals, color=c, s=44, alpha=0.52,
                   linewidths=1.0, edgecolors="white", zorder=3)
        ax.plot([x - 0.28, x + 0.28], [vals.mean(), vals.mean()],
                color=c, lw=5.0, solid_capstyle="butt", zorder=4)

        ratio = vals.mean() / w_mean if w_mean > 0 else float("nan")
        ax.text(x, vals.mean() + y_top * 0.025,
                f"{vals.mean():.3f}\n×{ratio:.1f}",
                ha="center", va="bottom", fontsize=9, color=c, fontweight="bold")

    # Axes
    x_labels = ["within-AND\n(noise floor)"] + [TERM_LABEL[c] for c in gap_conds]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlim(-0.6, n_cols - 0.4)
    ax.set_ylim(bottom=0, top=y_top)
    ax.set_ylabel(r"Per-element MSE  $d_T$ or $d_{\mathrm{within}}$", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    hide_top_right(ax)

    n_within = len(df_within)
    n_seeds  = df_term["seed"].nunique()
    n_pairs  = df_term["pair"].nunique()
    ax.set_title(
        "Plot 11 — Gap Validation: Per-Condition Distances vs Within-AND Noise Floor\n"
        f"Grey = pairwise $\\left\\|z_T^{{\\mathrm{{AND}}}}(s)-z_T^{{\\mathrm{{AND}}}}(s')\\right\\|_2^2$  "
        f"({n_within} seed-pairs from {n_pairs} concept pairs × {n_seeds} seeds).  "
        f"Dashed line + band = within-AND mean ± 1 std.\n"
        "×N = gap/noise-floor ratio.  "
        "Ratios >> 1 confirm the composability gap is structural, not stochastic.",
        fontsize=11,
    )

    fig.tight_layout()
    save_fig(fig, out_dir / "plot_11_within_and_noise_floor.png")


# ===========================================================================
# Plot 12 — p* raw seed strip by concept pair  (foundation, mirrors plot 00)
#
#   Shows terminal latent distance from AND for all p* sources + monolithic,
#   one panel per concept pair.  Each dot = one seed, bold tick = mean.
#   x-axis: p* sources (priority order, left) | mono (right, dashed separator)
#   Gracefully skipped if no d_T_pstar_* column present.
# ===========================================================================

def plot_12(df_term, df_traj, out_dir, **kw):
    present_pstar = get_present_pstar(df_term)
    if not present_pstar:
        print("  Skipping plot 12: no d_T_pstar_* column found.")
        print("  Re-run measure_composability_gap.py with at least one p* source.")
        return

    pairs     = cap_pairs(sorted(df_term["pair"].unique()), kw.get("max_display_pairs"))
    rng       = np.random.default_rng(42)
    all_conds = present_pstar + ["d_T_mono"]
    x_pos     = {c: float(i) for i, c in enumerate(all_conds)}

    all_vals = df_term[all_conds].values
    y_top    = max(float(all_vals.max()) * 1.12, 0.30)
    n_panels = len(pairs)

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, pair in zip(axes, pairs):
        sub = df_term[df_term["pair"] == pair]

        for cond in all_conds:
            vals   = sub[cond].values
            x      = x_pos[cond]
            jitter = rng.uniform(-0.13, 0.13, len(vals))
            c      = TERM_COLOR[cond]

            ax.scatter(x + jitter, vals, color=c, s=55, alpha=0.72,
                       linewidths=1.2, edgecolors="white", zorder=3)
            ax.plot([x - 0.24, x + 0.24], [vals.mean(), vals.mean()],
                    color=c, lw=4.5, solid_capstyle="butt", zorder=4)
            ax.text(x, vals.mean() + y_top * 0.025,
                    f"{vals.mean():.3f}", ha="center", va="bottom",
                    fontsize=8, color=c)

        # Dashed vertical separator: p* sources | mono baseline
        sep_x = len(present_pstar) - 0.5
        ax.axvline(sep_x, color="#CCCCCC", lw=1.0, ls="--", zorder=1)

        ax.set_title(short_pair(pair), fontsize=11)
        ax.set_xticks([x_pos[c] for c in all_conds])
        ax.set_xticklabels([TERM_LABEL[c] for c in all_conds],
                           fontsize=8, rotation=20, ha="right")
        ax.set_xlim(-0.5, len(all_conds) - 0.5)
        ax.set_ylim(bottom=0, top=y_top)
        ax.grid(axis="y", alpha=0.25)
        hide_top_right(ax)

    axes[0].set_ylabel(LABEL_D_T_MSE, fontsize=11)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=TERM_COLOR[c], markersize=9,
               markeredgecolor="white", markeredgewidth=1.2,
               label=TERM_LABEL[c])
        for c in all_conds
    ]
    fig.legend(handles=legend_handles, title="Condition",
               loc="upper right", fontsize=9, title_fontsize=10,
               frameon=True, bbox_to_anchor=(1.0, 0.93))

    pstar_names = ", ".join(TERM_LABEL[c] for c in present_pstar)
    n_seeds     = df_term["seed"].nunique()
    fig.suptitle(
        "Plot 12 — p* Closing the Gap: Raw Seed Strip  (per Concept Pair)\n"
        f"[{pstar_names}]  vs Monolithic  ·  Bold tick = mean  ·  "
        f"Each dot = one seed  (N = {n_seeds} per condition)",
        fontsize=12,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_12_pstar_strip_by_pair.png")


# ===========================================================================
# Plot 13 — p* grouped bar: mean ± std per pair, bars = p* sources + mono
#            (aggregation layer, mirrors plot 01)
# ===========================================================================

def plot_13(df_term, df_traj, out_dir, **kw):
    present_pstar = get_present_pstar(df_term)
    if not present_pstar:
        print("  Skipping plot 13: no d_T_pstar_* column found.")
        return

    pairs     = sorted(df_term["pair"].unique())
    n_pairs   = len(pairs)
    x         = np.arange(n_pairs)
    all_conds = present_pstar + ["d_T_mono"]
    n_bars    = len(all_conds)
    bar_w     = 0.80 / n_bars
    offsets   = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_w

    fig_w = max(10, 2.5 * n_pairs)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    for i, cond in enumerate(all_conds):
        means = [df_term[df_term["pair"] == p][cond].mean() for p in pairs]
        stds  = [df_term[df_term["pair"] == p][cond].std()  for p in pairs]
        ax.bar(x + offsets[i], means, bar_w * 0.88,
               yerr=stds, capsize=4,
               error_kw={"lw": 1.5, "ecolor": "#444"},
               color=TERM_COLOR[cond], label=TERM_LABEL[cond],
               alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([short_pair(p) for p in pairs],
                       fontsize=10, rotation=45, ha="right")
    ax.set_ylabel(LABEL_D_T_MSE, fontsize=11)
    pstar_names = " / ".join(TERM_LABEL[c] for c in present_pstar)
    n_seeds     = df_term["seed"].nunique()
    ax.set_title(
        f"Plot 13 — p* Closing the Gap: Mean ± Std per Concept Pair\n"
        f"[{pstar_names}]  vs Monolithic  ·  "
        f"Bars grouped by concept pair  ·  Error bars = ±1 std  ({n_seeds} seeds per pair)",
        fontsize=11,
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    hide_top_right(ax)
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_13_pstar_bars.png")


# ===========================================================================
# Plot 14 — p* distance histogram: one panel per concept pair, bars by p* source
#            (distribution layer, mirrors plot 02)
#
#   x-axis = distance bins (shared across all panels & conditions)
#   Each bin is split into n_cond sub-bars (one per p* source + mono)
#   Reveals the shape of the distance distribution for each inversion method.
# ===========================================================================

def plot_14(df_term, df_traj, out_dir, **kw):
    present_pstar = get_present_pstar(df_term)
    if not present_pstar:
        print("  Skipping plot 14: no d_T_pstar_* column found.")
        return

    pairs     = cap_pairs(sorted(df_term["pair"].unique()), kw.get("max_display_pairs"))
    n_pairs   = len(pairs)
    all_conds = present_pstar + ["d_T_mono"]
    n_conds   = len(all_conds)

    all_vals  = df_term[all_conds].values.flatten()
    bin_edges = np.linspace(all_vals.min() * 0.85, all_vals.max() * 1.08, 10)  # 9 bins
    bin_ctrs  = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_w     = bin_edges[1] - bin_edges[0]
    bar_w     = bin_w * 0.92 / n_conds
    c_offset  = (np.arange(n_conds) - (n_conds - 1) / 2) * bar_w

    fig, axes = plt.subplots(1, n_pairs, figsize=(4.5 * n_pairs, 4.5),
                             sharey=True, sharex=True)
    if n_pairs == 1:
        axes = [axes]

    for ax, pair in zip(axes, pairs):
        sub = df_term[df_term["pair"] == pair]
        for ci, cond in enumerate(all_conds):
            vals      = sub[cond].values
            counts, _ = np.histogram(vals, bins=bin_edges)
            ax.bar(bin_ctrs + c_offset[ci], counts, bar_w,
                   color=TERM_COLOR[cond], label=TERM_LABEL[cond],
                   alpha=0.85, edgecolor="white", linewidth=0.6)

        ax.set_title(short_pair(pair), fontsize=11)
        ax.set_xlabel(r"$d_T$  (per-element MSE)", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        hide_top_right(ax)

    axes[0].set_ylabel("Count  (seeds)", fontsize=11)

    legend_handles = [
        Patch(facecolor=TERM_COLOR[c], label=TERM_LABEL[c], alpha=0.85)
        for c in all_conds
    ]
    axes[-1].legend(handles=legend_handles, fontsize=8,
                    loc="upper right", title="p* source / cond")

    pstar_names = " / ".join(TERM_LABEL[c] for c in present_pstar)
    n_seeds     = df_term["seed"].nunique()
    fig.suptitle(
        f"Plot 14 — p* Distance Histogram  (colour = p* source)\n"
        f"[{pstar_names}]  vs Monolithic  ·  "
        f"One panel per concept pair  ·  9 bins  ·  {n_seeds} seeds per pair",
        fontsize=12,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_14_pstar_histogram.png")


# ===========================================================================
# Plot 15 — p* pooled KDE: smooth of terminal distance for ALL conditions
#            including all p* sources + all baselines (mono, c1, c2).
#
#   Key visual for "AND unreachable by text":
#     - p* KDEs (solid) should overlap the mono KDE if gap is closed
#     - Persistent separation between p* and AND suggests AND lies outside
#       the text-conditioned latent manifold of SD3.5
#   (mirrors plot 04, adds all p* sources as solid curves)
# ===========================================================================

def plot_15(df_term, df_traj, out_dir, **kw):
    if not SCIPY_OK:
        print("  Skipping plot 15 (scipy required).")
        return

    present_pstar = get_present_pstar(df_term)
    if not present_pstar:
        print("  Skipping plot 15: no d_T_pstar_* column found.")
        return

    pstar_conds    = present_pstar
    baseline_conds = TERM_CONDITIONS + get_present_poe(df_term)
    all_conds      = pstar_conds + baseline_conds

    all_vals = df_term[all_conds].values.flatten()
    x_grid   = np.linspace(all_vals.min() * 0.80, all_vals.max() * 1.10, 700)
    n        = len(df_term)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # p* sources — solid lines + light fill
    for cond in pstar_conds:
        vals = df_term[cond].values
        dens = stats.gaussian_kde(vals, bw_method="scott")(x_grid)
        c    = TERM_COLOR[cond]
        ax.plot(x_grid, dens, color=c, label=TERM_LABEL[cond], lw=2.5, ls="-")
        ax.fill_between(x_grid, dens, alpha=0.12, color=c)

    # Baseline conditions — dashed lines + lighter fill
    for cond in baseline_conds:
        vals = df_term[cond].values
        dens = stats.gaussian_kde(vals, bw_method="scott")(x_grid)
        c    = TERM_COLOR[cond]
        ax.plot(x_grid, dens, color=c, label=TERM_LABEL[cond],
                lw=2.0, ls="--", alpha=0.80)
        ax.fill_between(x_grid, dens, alpha=0.07, color=c)

    ax.set_xlabel("Terminal distance to SuperDiff AND (per-element MSE)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Plot 15 — Terminal Distance Distribution Across Conditions  (N = {n} per condition)",
        fontsize=12,
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.22)
    hide_top_right(ax)
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_15_pstar_kde_pooled.png")


# ===========================================================================
# Plot 16 — p* temporal trajectory: per-step MSE for all p* sources + mono,
#            all pairs + seeds pooled.  Mean (solid) + ±1 std band + 95% CI.
#            (mirrors plot 10, temporal layer of the p* sequence)
#
#   Solid lines = p* sources (auto-detected from trajectory data)
#   Dashed line = mono (reference)
#   Band layers: light outer = ±1 std,  darker inner = 95 % CI
# ===========================================================================

def plot_16(df_term, df_traj, out_dir, **kw):
    # Detect which trajectory p* columns are present
    present_traj_pstar = [c for c in TRAJ_PSTAR_PRIORITY if c in df_traj.columns]
    if "d_t_pstar_inv" in present_traj_pstar and "d_t_pstar" in present_traj_pstar:
        present_traj_pstar.remove("d_t_pstar")

    if not present_traj_pstar:
        print("  Skipping plot 16: no d_t_pstar_* trajectory column found.")
        print("  Re-run measure_composability_gap.py with trajectory tracking enabled.")
        return

    steps      = sorted(df_traj["step"].unique())
    n_per_step = int(df_traj[df_traj["step"] == steps[0]].shape[0])
    n_pairs    = df_traj["pair"].nunique()
    n_seeds    = df_traj["seed"].nunique()

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # p* sources — solid lines with std band + CI band
    for traj_col in present_traj_pstar:
        c     = TRAJ_COLOR.get(traj_col, "#999999")
        label = TRAJ_LABEL.get(traj_col, traj_col)
        means, stds = traj_stats(df_traj, traj_col, steps)
        ci95        = 1.96 * stds / np.sqrt(n_per_step)

        ax.plot(steps, means, color=c, lw=2.5, ls="-", label=label)
        ax.fill_between(steps, means - stds,  means + stds,
                        alpha=0.10, color=c)
        ax.fill_between(steps, means - ci95,  means + ci95,
                        alpha=0.25, color=c)

    # Baselines — dashed references with CI band
    baseline_traj_conds = ["d_t_mono"] + (["d_t_poe"] if "d_t_poe" in df_traj.columns else [])
    for bcond in baseline_traj_conds:
        means_b, stds_b = traj_stats(df_traj, bcond, steps)
        ci95_b          = 1.96 * stds_b / np.sqrt(n_per_step)
        c_b             = TRAJ_COLOR[bcond]
        ax.plot(steps, means_b, color=c_b, lw=2.0, ls="--",
                label=TRAJ_LABEL[bcond], alpha=0.85)
        ax.fill_between(steps, means_b - ci95_b, means_b + ci95_b,
                        alpha=0.12, color=c_b)

    ax.set_xlabel("Denoising step", fontsize=12)
    ax.set_ylabel(LABEL_D_t_MSE, fontsize=12)
    pstar_names = " / ".join(TRAJ_LABEL.get(c, c) for c in present_traj_pstar)
    ax.set_title(
        "Plot 16 — p* Temporal Trajectory: Distance from AND over Denoising Steps\n"
        f"Solid = p* [{pstar_names}]  ·  Dashed = baseline(s)\n"
        f"All pairs + seeds pooled  (N = {n_pairs} × {n_seeds} = {n_pairs * n_seeds} per cond)  ·  "
        "Band = ±1 std  ·  inner band = 95 % CI",
        fontsize=11,
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.22)
    hide_top_right(ax)
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_16_pstar_temporal.png")


# ===========================================================================
# Plot 17 — p* terminal strip: terminal distance from AND for ALL conditions
#            including p* (the inverter output).
#            Gracefully skipped if d_T_pstar is absent — re-run measure script.
# ===========================================================================

def plot_17(df_term, df_traj, out_dir, **kw):
    present_pstar = get_present_pstar(df_term)
    if not present_pstar:
        print("  Skipping plot 17: no d_T_pstar_* column found in per_seed_distances.json.")
        print("  Re-run measure_composability_gap.py to generate at least one p* source.")
        return

    # p* columns first (in priority order), then baselines
    baseline_conds = ["d_T_mono", "d_T_c1", "d_T_c2"] + get_present_poe(df_term)
    all_conds = present_pstar + baseline_conds
    rng       = np.random.default_rng(42)
    x_pos     = {c: float(i) for i, c in enumerate(all_conds)}

    all_vals = df_term[all_conds].values
    y_top    = max(all_vals.max() * 1.12, 0.60)

    # Width scales with the number of conditions
    fig_w = max(9, 1.8 * len(all_conds))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    for cond in all_conds:
        vals   = df_term[cond].values
        x      = x_pos[cond]
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        c      = TERM_COLOR[cond]

        ax.scatter(x + jitter, vals, color=c, s=50, alpha=0.55,
                   linewidths=1.1, edgecolors="white", zorder=3)
        ax.plot([x - 0.30, x + 0.30], [vals.mean(), vals.mean()],
                color=c, lw=5.0, solid_capstyle="butt", zorder=4)
        ax.text(x, vals.mean() + y_top * 0.03, f"{vals.mean():.3f}",
                ha="center", va="bottom", fontsize=9, color=c)

    # Vertical separator between p* columns and baselines
    if present_pstar:
        sep_x = len(present_pstar) - 0.5
        ax.axvline(sep_x, color="#CCCCCC", lw=1.0, ls="--", zorder=1)

    ax.set_xticks([x_pos[c] for c in all_conds])
    ax.set_xticklabels([TERM_LABEL[c] for c in all_conds], fontsize=10, rotation=15, ha="right")
    ax.set_xlim(-0.6, len(all_conds) - 0.4)
    ax.set_ylim(bottom=0, top=y_top)
    ax.set_ylabel("Terminal distance to SuperDiff AND (per-element MSE)", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    hide_top_right(ax)

    ax.set_title(
        "Plot 17 — Terminal Distance by Condition",
        fontsize=12,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=TERM_COLOR[c], markersize=9,
               markeredgecolor="white", markeredgewidth=1.2,
               label=TERM_LABEL[c])
        for c in all_conds
    ]
    ax.legend(handles=legend_handles, title="Condition",
              fontsize=9, title_fontsize=10, frameon=True,
              loc="upper right")

    fig.tight_layout()
    save_fig(fig, out_dir / "plot_17_pstar_strip.png")


# ===========================================================================
# Plot 25 — Combined view of plot 17 (left) and plot 15 (right).
#            Both subplots are rendered with equal panel sizes.
# ===========================================================================

def plot_25(df_term, df_traj, out_dir, **kw):
    if not SCIPY_OK:
        print("  Skipping plot 25 (scipy required for KDE panel).")
        return

    present_pstar = get_present_pstar(df_term)
    if not present_pstar:
        print("  Skipping plot 25: no d_T_pstar_* column found.")
        print("  Re-run measure_composability_gap.py to generate at least one p* source.")
        return

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 1]}
    )

    # ---- Left panel: plot 17 (terminal strip) ----
    strip_conds = present_pstar + ["d_T_mono", "d_T_c1", "d_T_c2"] + get_present_poe(df_term)
    rng = np.random.default_rng(42)
    x_pos = {c: float(i) for i, c in enumerate(strip_conds)}
    y_top = max(df_term[strip_conds].values.max() * 1.12, 0.60)

    for cond in strip_conds:
        vals = df_term[cond].values
        x = x_pos[cond]
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        c = TERM_COLOR[cond]

        ax_left.scatter(x + jitter, vals, color=c, s=50, alpha=0.55,
                        linewidths=1.1, edgecolors="white", zorder=3)
        ax_left.plot([x - 0.30, x + 0.30], [vals.mean(), vals.mean()],
                     color=c, lw=5.0, solid_capstyle="butt", zorder=4)
        ax_left.text(x, vals.mean() + y_top * 0.03, f"{vals.mean():.3f}",
                     ha="center", va="bottom", fontsize=9, color=c)

    sep_x = len(present_pstar) - 0.5
    ax_left.axvline(sep_x, color="#CCCCCC", lw=1.0, ls="--", zorder=1)
    ax_left.set_xticks([x_pos[c] for c in strip_conds])
    ax_left.set_xticklabels([TERM_LABEL[c] for c in strip_conds],
                            fontsize=10, rotation=15, ha="right")
    ax_left.set_xlim(-0.6, len(strip_conds) - 0.4)
    ax_left.set_ylim(bottom=0, top=y_top)
    ax_left.set_ylabel("Terminal distance to SuperDiff AND (per-element MSE)", fontsize=11)
    ax_left.set_title("Plot 17 — Terminal Distance by Condition", fontsize=12)
    ax_left.grid(axis="y", alpha=0.25)
    hide_top_right(ax_left)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=TERM_COLOR[c], markersize=8,
               markeredgecolor="white", markeredgewidth=1.2,
               label=TERM_LABEL[c])
        for c in strip_conds
    ]
    ax_left.legend(handles=legend_handles, title="Condition",
                   fontsize=8, title_fontsize=9, frameon=True,
                   loc="upper right")

    # ---- Right panel: plot 15 (pooled KDE) ----
    baseline_conds = TERM_CONDITIONS + get_present_poe(df_term)
    kde_conds = present_pstar + baseline_conds
    all_vals = df_term[kde_conds].values.flatten()
    x_grid = np.linspace(all_vals.min() * 0.80, all_vals.max() * 1.10, 700)
    n = len(df_term)

    for cond in present_pstar:
        vals = df_term[cond].values
        dens = stats.gaussian_kde(vals, bw_method="scott")(x_grid)
        c = TERM_COLOR[cond]
        ax_right.plot(x_grid, dens, color=c, label=TERM_LABEL[cond], lw=2.5, ls="-")
        ax_right.fill_between(x_grid, dens, alpha=0.12, color=c)

    for cond in baseline_conds:
        vals = df_term[cond].values
        dens = stats.gaussian_kde(vals, bw_method="scott")(x_grid)
        c = TERM_COLOR[cond]
        ax_right.plot(x_grid, dens, color=c, label=TERM_LABEL[cond],
                      lw=2.0, ls="--", alpha=0.80)
        ax_right.fill_between(x_grid, dens, alpha=0.07, color=c)

    ax_right.set_xlabel("Terminal distance to SuperDiff AND (per-element MSE)", fontsize=11)
    ax_right.set_ylabel("Density", fontsize=11)
    ax_right.set_title(
        f"Plot 15 — Terminal Distance Distribution Across Conditions  (N = {n} per condition)",
        fontsize=12,
    )
    ax_right.legend(fontsize=9, loc="upper right")
    ax_right.grid(alpha=0.22)
    hide_top_right(ax_right)

    fig.suptitle(
        "Plot 25 — Terminal Composability Gap (p* vs Baselines): "
        "Seed-Level Strip (Left) and Pooled KDE (Right)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, out_dir / "plot_17_and_15.png")


# ===========================================================================
# Plot 18 — CLIP similarity comparison: AND↔p* vs AND↔mono, per concept pair.
#            Uses all_pairs_gap.json — no re-run needed.
# ===========================================================================

def plot_18(df_term, df_traj, out_dir, data_dir=None, **kw):
    if data_dir is None:
        print("  Skipping plot 18: data_dir not provided.")
        return

    gap_data = load_all_pairs_gap(
        data_dir,
        monolithic_baseline=kw.get("monolithic_baseline", "auto"),
    )
    if gap_data is None:
        print("  Skipping plot 18: all_pairs_gap.json not found in data_dir.")
        return

    pairs = [f"{r['pair'][0]} + {r['pair'][1]}" for r in gap_data]

    # --- Discover which pstar sources are present in the JSON ---
    # Check the first record to find gap_and_pstar_* keys; suppress legacy alias
    # when the explicit _inv key is present.
    present_gap_keys = []
    sample = gap_data[0] if gap_data else {}
    for pstar_col, gap_key in PSTAR_GAP_KEY.items():
        if gap_key in sample:
            present_gap_keys.append((pstar_col, gap_key))
    # Suppress legacy "d_T_pstar" / "gap_and_pstar" when _inv variant exists
    key_names = [gk for _, gk in present_gap_keys]
    if "gap_and_pstar_inv" in key_names and "gap_and_pstar" in key_names:
        present_gap_keys = [(pc, gk) for pc, gk in present_gap_keys if gk != "gap_and_pstar"]

    if not present_gap_keys:
        print("  Skipping plot 18: no gap_and_pstar_* keys found in all_pairs_gap.json.")
        return

    # Always include monolithic as the reference bar
    clip_mono = [r["gap_and_mono"]["clip_cos"] for r in gap_data]

    n_pstar  = len(present_gap_keys)
    n_pairs  = len(pairs)
    # Total bars per pair-group = n_pstar p* sources + 1 monolithic
    n_bars   = n_pstar + 1
    bar_w    = 0.80 / n_bars
    offsets  = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_w
    x_base   = np.arange(n_pairs)

    fig_w = max(10, 2.5 * n_pairs)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    # Draw p* bars
    all_clip_vals = list(clip_mono)
    for bi, (pstar_col, gap_key) in enumerate(present_gap_keys):
        clip_vals = [r[gap_key]["clip_cos"] for r in gap_data]
        all_clip_vals.extend(clip_vals)
        c    = TERM_COLOR[pstar_col]
        bars = ax.bar(x_base + offsets[bi], clip_vals, bar_w * 0.90,
                      color=c, alpha=0.85,
                      label=f"AND ↔ {TERM_LABEL[pstar_col]}")
        for bar, h in zip(bars, clip_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        ax.axhline(np.mean(clip_vals), color=c, lw=1.2, ls="--", alpha=0.55)

    # Draw monolithic bar (rightmost in each group)
    mono_offset = offsets[n_pstar]
    bars_m = ax.bar(x_base + mono_offset, clip_mono, bar_w * 0.90,
                    color=TERM_COLOR["d_T_mono"], alpha=0.85,
                    label="AND ↔ Monolithic")
    for bar, h in zip(bars_m, clip_mono):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(np.mean(clip_mono), color=TERM_COLOR["d_T_mono"],
               lw=1.2, ls="--", alpha=0.55)

    ax.set_xticks(x_base)
    ax.set_xticklabels([short_pair(p) for p in pairs],
                       fontsize=10, rotation=45, ha="right")
    ax.set_ylabel("CLIP cosine similarity to AND  (↑ better)", fontsize=11)
    ax.set_ylim(bottom=max(0, min(all_clip_vals) * 0.90),
                top=min(1.0, max(all_clip_vals) * 1.08))
    pstar_names = " / ".join(TERM_LABEL[pc] for pc, _ in present_gap_keys)
    ax.set_title(
        f"Plot 18 — CLIP Similarity to AND: [{pstar_names}]  vs Monolithic\n"
        "Higher = generated image closer to SuperDiff-AND in CLIP space.\n"
        "Dashed lines = per-source mean.",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    hide_top_right(ax)
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_18_clip_comparison.png")


# ===========================================================================
# Plot 19 — JS² divergence: p* vs mono terminal distance distributions.
#            Lower JS² = p* and mono arrive at more similar latent neighbourhoods.
#            Gracefully skipped if d_T_pstar is absent.
# ===========================================================================

def plot_19(df_term, df_traj, out_dir, **kw):
    if not SCIPY_OK:
        print("  Skipping plot 19 (scipy required).")
        return
    if jensenshannon is None:
        print("  Skipping plot 19 (scipy.spatial.distance.jensenshannon not available).")
        return

    present_pstar = get_present_pstar(df_term)
    if not present_pstar:
        print("  Skipping plot 19: no d_T_pstar_* column found.")
        return

    pairs  = sorted(df_term["pair"].unique())
    n_pairs = len(pairs)
    n_pstar = len(present_pstar)

    # Shared x-grid over all values for KDE comparability
    all_vals = np.concatenate(
        [df_term[c].values for c in present_pstar] + [df_term["d_T_mono"].values]
    )
    x_grid = np.linspace(all_vals.min() * 0.70, all_vals.max() * 1.15, 1000)

    # Compute JS²(P_pstar_col, P_mono) per pair per source
    jsd_by_source = {}
    for pstar_col in present_pstar:
        jsd_vals = []
        for pair in pairs:
            sub    = df_term[df_term["pair"] == pair]
            p_star = kde_pmf(sub[pstar_col].values,     x_grid)
            p_mono = kde_pmf(sub["d_T_mono"].values,    x_grid)
            jsd_vals.append(jensenshannon(p_star, p_mono) ** 2)
        jsd_by_source[pstar_col] = jsd_vals

    # --- Grouped bar chart ---
    bar_w   = 0.80 / n_pstar
    offsets = (np.arange(n_pstar) - (n_pstar - 1) / 2) * bar_w
    x_base  = np.arange(n_pairs)

    fig_w = max(9, 2.2 * n_pairs)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    for pi, pstar_col in enumerate(present_pstar):
        c    = TERM_COLOR[pstar_col]
        jsd  = jsd_by_source[pstar_col]
        xs   = x_base + offsets[pi]
        bars = ax.bar(xs, jsd, bar_w * 0.88,
                      color=c, alpha=0.85, edgecolor="white",
                      label=TERM_LABEL[pstar_col])
        for bar, h in zip(bars, jsd):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8, color=c)
        # Mean dashed line per source
        mean_jsd = float(np.mean(jsd))
        ax.axhline(mean_jsd, color=c, lw=1.2, ls="--", alpha=0.60)

    ax.set_xticks(x_base)
    ax.set_xticklabels([short_pair(p) for p in pairs],
                       fontsize=10, rotation=45, ha="right")
    ax.set_ylabel(LABEL_JS2, fontsize=11)
    pstar_names = " / ".join(TERM_LABEL[c] for c in present_pstar)
    ax.set_title(
        f"Plot 19 — Gap Closure: JS² divergence of [{pstar_names}] vs Monolithic\n"
        "One bar per p* source per concept pair.  "
        "Lower = p* distribution closer to monolithic SD3.5.",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    hide_top_right(ax)

    fig.tight_layout()
    save_fig(fig, out_dir / "plot_19_js_pstar_vs_mono.png")


# ===========================================================================
# Plots 27–28 — Multi-pair grid reconstructions with p* VLM (legacy Z2T optional).
#
# Data source:
#   pairs/<pair_slug>/grid_assets.json
#   (exported by measure_composability_gap.py)
# ===========================================================================

_GRID_COL_ORDER = [
    "prompt_a",
    "prompt_b",
    "monolithic",
    "poe",
    "superdiff_fm_ode",
    "pstar_vlm",
    "pstar_z2t",
]

_GRID_COL_HEADER = {
    "prompt_a":         "SD3.5 A",
    "prompt_b":         "SD3.5 B",
    "monolithic":       "SD3.5 A∧B",
    "poe":              "PoE A×B",
    "superdiff_fm_ode": "SuperDiff A∧B",
    "pstar_vlm":        "p* VLM",
    "pstar_z2t":        "p* Z2T",
}


def _resolve_grid_columns_from_filter(pstar_filter):
    """Resolve which grid columns to render from the global p* source filter.

    pstar_filter=None (auto/all) keeps all grid columns.
    pstar_filter=frozenset() removes p* grid columns.
    Otherwise, only mapped p* grid columns are kept.
    """
    if pstar_filter is None:
        return list(_GRID_COL_ORDER)

    pstar_grid_map = {
        "d_T_pstar_vlm": "pstar_vlm",
        "d_T_pstar_z2t": "pstar_z2t",
    }
    allowed_pstar_grid = {
        pstar_grid_map[c] for c in pstar_filter if c in pstar_grid_map
    }
    return [
        c for c in _GRID_COL_ORDER
        if (not c.startswith("pstar_")) or (c in allowed_pstar_grid)
    ]


def _filter_available_grid_columns(assets: list, grid_cols: list, *, mode: str) -> list:
    """Drop columns that are absent from all loaded grid assets.

    This keeps figure layouts clean for runs that did not export optional
    methods like PoE or legacy Z2T.
    """
    def _asset_has_col(asset: dict, col_key: str) -> bool:
        if mode == "decoded":
            path_map = asset.get("decoded_image_paths", {})
            return bool(path_map.get(col_key))

        if mode == "trajectory":
            flat_map = asset.get("trajectory_flat_paths", {})
            if flat_map.get(col_key):
                return True
            traj = asset.get("trajectory_projection", {})
            projected = traj.get("projected", {}) if isinstance(traj, dict) else {}
            return col_key in projected

        raise ValueError(f"Unknown grid column filter mode: {mode!r}")

    kept = [col_key for col_key in grid_cols if any(_asset_has_col(asset, col_key) for asset in assets)]
    dropped = [col_key for col_key in grid_cols if col_key not in kept]
    if dropped:
        dropped_labels = ", ".join(_GRID_COL_HEADER.get(col_key, col_key) for col_key in dropped)
        print(f"  Hiding unavailable grid columns: {dropped_labels}")
    return kept


def _apply_monolithic_baseline_to_grid_asset(payload: dict, mode: str) -> dict:
    """Optionally remap 'monolithic' grid asset keys to naive/natural variants."""
    if mode not in {"naive", "natural"}:
        return payload
    src_key = f"monolithic_{mode}"

    for map_key in ("decoded_image_paths", "trajectory_flat_paths"):
        path_map = payload.get(map_key)
        if isinstance(path_map, dict) and src_key in path_map:
            path_map["monolithic"] = path_map[src_key]

    traj_proj = payload.get("trajectory_projection")
    if isinstance(traj_proj, dict):
        projected = traj_proj.get("projected")
        if isinstance(projected, dict) and src_key in projected:
            projected["monolithic"] = projected[src_key]
        labels = traj_proj.get("labels")
        if isinstance(labels, dict) and src_key in labels:
            labels["monolithic"] = labels[src_key]

    src_prompts = payload.get("source_prompts")
    if isinstance(src_prompts, dict) and src_key in src_prompts:
        src_prompts["monolithic"] = src_prompts[src_key]

    return payload


def _load_grid_assets(
    data_dir: Path,
    max_pairs: int = 4,
    monolithic_baseline: str = "auto",
) -> list:
    """Load per-pair grid assets exported by measure_composability_gap.py."""
    pairs_root = Path(data_dir) / "pairs"
    if not pairs_root.exists():
        return []

    assets = []
    for pair_dir in sorted([p for p in pairs_root.iterdir() if p.is_dir()]):
        asset_path = pair_dir / "grid_assets.json"
        if not asset_path.exists():
            continue
        try:
            payload = json.loads(asset_path.read_text())
        except Exception as exc:
            print(f"  Warning: could not parse {asset_path}: {exc}")
            continue
        payload = _apply_monolithic_baseline_to_grid_asset(payload, monolithic_baseline)
        payload["_pair_dir"] = str(pair_dir)
        assets.append(payload)

    def _sort_key(asset):
        idx = asset.get("pair_index")
        if isinstance(idx, int):
            return (0, idx, "")
        return (1, 0, " + ".join(asset.get("pair", [])))

    assets.sort(key=_sort_key)
    if max_pairs is not None:
        assets = assets[:max_pairs]
    return assets


def _project_flat_trajectories(flat_by_cond: dict, method: str = "mds"):
    """
    Jointly project pre-flattened trajectories to 2D.
    flat_by_cond: {cond -> (T+1, D) float array}
    """
    if not flat_by_cond:
        return None, None

    cond_names = list(flat_by_cond.keys())
    n_steps = int(next(iter(flat_by_cond.values())).shape[0])

    max_dim = max(arr.shape[1] for arr in flat_by_cond.values())
    stacked_parts = []
    for cond in cond_names:
        arr = flat_by_cond[cond].astype(np.float32, copy=False)
        if arr.shape[1] < max_dim:
            pad = np.zeros((arr.shape[0], max_dim - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        stacked_parts.append(arr)
    stacked = np.vstack(stacked_parts)

    if method == "pca":
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            print("  Skipping plot 28: scikit-learn missing (PCA projection unavailable).")
            return None, None
        proj = PCA(n_components=2).fit_transform(stacked)
    else:
        try:
            from sklearn.manifold import MDS
            from sklearn.metrics import pairwise_distances
        except ImportError:
            print("  Skipping plot 28: scikit-learn missing (MDS projection unavailable).")
            return None, None
        dist = pairwise_distances(stacked, metric="euclidean")
        proj = MDS(
            n_components=2,
            random_state=42,
            dissimilarity="precomputed",
            normalized_stress="auto",
        ).fit_transform(dist)

    projected = {}
    start = 0
    for cond in cond_names:
        end = start + n_steps
        projected[cond] = proj[start:end]
        start = end
    return projected, n_steps


def _plot_manifold_grid_from_projected(
    pair_payload: list,
    output_path: Path,
    projection_method: str = "mds",
    max_plots: int = 4,
    cell_size: float = 5.8,
) -> None:
    """Render a 2-column manifold grid from preprojected trajectories."""
    selected = pair_payload[:max_plots]
    if not selected:
        print("  Skipping plot 28: empty projected payload.")
        return

    n_plots = len(selected)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_size * n_cols, cell_size * n_rows),
        squeeze=False,
    )

    shared_n_steps = max(int(item["n_steps"]) for item in selected)
    norm = Normalize(vmin=0, vmax=max(shared_n_steps - 1, 1))
    cmap_name = "viridis"
    end_markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    end_colors = [
        "#e63946", "#457b9d", "#2a9d8f", "#e9c46a",
        "#f4a261", "#8d99ae", "#ef476f", "#118ab2",
    ]

    for idx, item in enumerate(selected):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row_idx][col_idx]

        projected = item["projected"]
        labels = item["labels"]
        prompt_key_map = item.get("prompt_key_map", {})
        cond_names = list(projected.keys())
        if not cond_names:
            ax.axis("off")
            continue

        for cond_idx, cond in enumerate(cond_names):
            pts = projected[cond]
            if len(pts) >= 2:
                points = pts.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(
                    segments,
                    cmap=cm.get_cmap(cmap_name),
                    norm=norm,
                    linewidths=2.0,
                    alpha=0.9,
                )
                lc.set_array(np.arange(len(segments)))
                ax.add_collection(lc)

            marker = end_markers[cond_idx % len(end_markers)]
            ep_color = end_colors[cond_idx % len(end_colors)]
            ax.plot(
                pts[-1, 0], pts[-1, 1],
                marker=marker, color=ep_color, linestyle="none",
                markersize=9, markeredgecolor="black", markeredgewidth=0.9,
                zorder=6,
            )

        origin = projected[cond_names[0]][0]
        ax.plot(origin[0], origin[1], "ko", markersize=5, zorder=5)
        ax.annotate(
            r"$x_T$",
            xy=(origin[0], origin[1]),
            fontsize=10,
            fontweight="bold",
            textcoords="offset points",
            xytext=(-11, -10),
        )

        ax_prefix = "MDS" if projection_method == "mds" else "PC"
        ax.autoscale()
        ax.set_xlabel(f"{ax_prefix} 1", fontsize=11)
        ax.set_ylabel(f"{ax_prefix} 2", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=10)

        if prompt_key_map:
            key_text = "\n".join([f"{k} = {v}" for k, v in prompt_key_map.items()])
            ax.set_title(key_text, fontsize=12, pad=6)
        else:
            ax.set_title(f"Pair {idx + 1}", fontsize=12, pad=6)

        legend_handles = []
        for cond_idx, cond in enumerate(cond_names):
            marker = end_markers[cond_idx % len(end_markers)]
            ep_color = end_colors[cond_idx % len(end_colors)]
            legend_handles.append(
                Line2D(
                    [0], [0],
                    color=cm.get_cmap(cmap_name)(0.5), lw=2.0,
                    marker=marker, markerfacecolor=ep_color, markeredgecolor="black",
                    markersize=7,
                    label=labels.get(cond, cond),
                )
            )
        ax.legend(handles=legend_handles, loc="best", fontsize=9, framealpha=0.9)

    for idx in range(n_plots, n_rows * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx][col_idx].axis("off")

    sm = cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    fig.tight_layout(rect=(0.0, 0.0, 0.89, 0.98))
    cax = fig.add_axes([0.91, 0.14, 0.018, 0.74])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Time (steps)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


def plot_27(df_term, df_traj, out_dir, data_dir=None, **kw):
    """Decoded image grid with extra p* columns (VLM; legacy Z2T optional)."""
    if data_dir is None:
        print("  Skipping plot 27: data_dir not provided.")
        return

    max_pairs = kw.get("max_display_pairs")
    max_pairs = 4 if max_pairs is None else max_pairs
    assets = _load_grid_assets(
        Path(data_dir),
        max_pairs=max_pairs,
        monolithic_baseline=kw.get("monolithic_baseline", "auto"),
    )
    if not assets:
        print("  Skipping plot 27: no grid_assets.json found under pairs/*.")
        print("  Re-run measure_composability_gap.py (updated version) to export grid assets.")
        return

    grid_cols = _resolve_grid_columns_from_filter(kw.get("pstar_filter"))
    grid_cols = _filter_available_grid_columns(assets, grid_cols, mode="decoded")
    if not grid_cols:
        print("  Skipping plot 27: no grid columns selected after p* filtering.")
        return

    n_rows = len(assets)
    n_cols = len(grid_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.4 * n_cols, 3.3 * n_rows),
        squeeze=False,
    )

    for row_idx, asset in enumerate(assets):
        pair_dir = Path(asset["_pair_dir"])
        img_map = asset.get("decoded_image_paths", {})
        prompt_key_map = asset.get("prompt_key_map", {})

        for col_idx, col_key in enumerate(grid_cols):
            ax = axes[row_idx][col_idx]
            ax.axis("off")

            rel_path = img_map.get(col_key)
            if rel_path:
                img_path = pair_dir / rel_path
                if img_path.exists():
                    ax.imshow(Image.open(img_path).convert("RGB"))
                else:
                    ax.set_facecolor("#eeeeee")
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=11, color="#888888")
            else:
                ax.set_facecolor("#eeeeee")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11, color="#888888")

            if row_idx == 0:
                ax.set_title(_GRID_COL_HEADER[col_key], fontsize=14, fontweight="bold", pad=7)

        if prompt_key_map:
            key_text = "\n".join([f"{k} = {v}" for k, v in prompt_key_map.items()])
            axes[row_idx][0].text(
                0.0, -0.08, key_text,
                transform=axes[row_idx][0].transAxes,
                ha="left", va="top",
                fontsize=11, fontweight="bold",
                linespacing=1.17, color="#111111", clip_on=False,
            )

    has_pstar = any(c.startswith("pstar_") for c in grid_cols)
    fig.suptitle(
        "Plot 27 — Decoded Image Grid"
        + (" with p* Sources (VLM; legacy Z2T optional)" if has_pstar else ""),
        fontsize=14,
    )
    fig.tight_layout(pad=0.5, rect=(0.0, 0.0, 1.0, 0.96))
    fig.subplots_adjust(hspace=0.50)
    save_fig(fig, out_dir / "decoded_images_grid.png")


def plot_28(df_term, df_traj, out_dir, data_dir=None, **kw):
    """Trajectory manifold grid with extra p* trajectories (VLM; legacy Z2T optional)."""
    if data_dir is None:
        print("  Skipping plot 28: data_dir not provided.")
        return

    max_pairs = kw.get("max_display_pairs")
    max_pairs = 4 if max_pairs is None else max_pairs
    assets = _load_grid_assets(
        Path(data_dir),
        max_pairs=max_pairs,
        monolithic_baseline=kw.get("monolithic_baseline", "auto"),
    )
    if not assets:
        print("  Skipping plot 28: no grid_assets.json found under pairs/*.")
        print("  Re-run measure_composability_gap.py (updated version) to export grid assets.")
        return

    grid_cols = _resolve_grid_columns_from_filter(kw.get("pstar_filter"))
    grid_cols = _filter_available_grid_columns(assets, grid_cols, mode="trajectory")
    if not grid_cols:
        print("  Skipping plot 28: no trajectory columns selected after p* filtering.")
        return

    pair_payload = []
    for asset in assets:
        pair_dir = Path(asset["_pair_dir"])
        proj_method = asset.get("projection_method", "mds")
        prompt_key_map = asset.get("prompt_key_map", {})
        projected = {}
        labels = {}
        n_steps = 1

        # Preferred path: recompute projection jointly from saved flattened trajectories.
        flat_paths = asset.get("trajectory_flat_paths", {})
        flat_by_cond = {}
        for cond in grid_cols:
            rel = flat_paths.get(cond)
            if not rel:
                continue
            flat_path = pair_dir / rel
            if not flat_path.exists():
                continue
            try:
                arr = np.load(flat_path)
            except Exception:
                continue
            if arr.ndim != 2:
                continue
            flat_by_cond[cond] = arr
            labels[cond] = _GRID_COL_HEADER.get(cond, cond)

        if flat_by_cond:
            projected, n_steps = _project_flat_trajectories(flat_by_cond, method=proj_method)
            if projected is None:
                continue
            projected = {k: projected[k] for k in grid_cols if k in projected}
        else:
            # Backward-compatible fallback: precomputed 2D points in grid_assets.json
            traj = asset.get("trajectory_projection")
            if not traj or not traj.get("projected"):
                continue
            projected_raw = traj.get("projected", {})
            labels_raw = traj.get("labels", {})
            n_steps = int(traj.get("n_steps", 1))
            for cond in grid_cols:
                if cond not in projected_raw:
                    continue
                pts = np.asarray(projected_raw[cond], dtype=np.float32)
                if pts.ndim != 2 or pts.shape[1] != 2:
                    continue
                projected[cond] = pts
                labels[cond] = labels_raw.get(cond, _GRID_COL_HEADER.get(cond, cond))

        if projected:
            pair_payload.append({
                "prompt_key_map": prompt_key_map,
                "projected": projected,
                "labels": labels,
                "n_steps": n_steps,
            })

    if not pair_payload:
        print("  Skipping plot 28: no trajectory projection payload in grid assets.")
        print("  Re-run measure_composability_gap.py with trajectory export enabled.")
        return

    grid_proj_method = assets[0].get("projection_method", "mds")
    traj_grid_scale = max(float(kw.get("traj_grid_scale", 1.0)), 0.25)
    out_path = out_dir / "trajectory_manifold_grid.png"
    _plot_manifold_grid_from_projected(
        pair_payload,
        out_path,
        projection_method=grid_proj_method,
        max_plots=max_pairs,
        cell_size=5.8 * traj_grid_scale,
    )
