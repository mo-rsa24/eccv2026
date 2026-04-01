"""
Gap analysis — baseline plots 00–10 and 26.

  Terminal distributions  (data: per_seed_distances.json)
    00  Strip by pair     — raw dots (one per seed) + bold mean tick, per pair
    01  Grouped bar       — mean ± std per pair per condition
    02  Hist grouped      — x=distance, grouped bars per bin, N condition panels
    03  Pooled strip      — all seeds pooled across pairs, bold tick=mean
    04  KDE pooled        — smooth of plot 03; all pairs combined
    05  KDE by pair       — panels per concept pair, colour=condition

  Temporal dynamics  (data: trajectory_distances.json)
    06  Stacked bar pooled  — incremental divergence per time bin, pooled
    07  Stacked bar by pair — same, one group per concept pair
    08  Per-condition       — N subplots (one per condition), pair-coloured bands
    09  Individual seeds    — same as 08, thin per-seed traces + bold mean
    10  Generalised         — N subplots (one per condition), all pooled,
                              mean with ±1 std outer band and 95 % CI inner band
    26  Combined 04+06      — side-by-side KDE pooled (left) + temporal stacked bar (right)

p* columns (d_T_pstar_inv, d_T_pstar_pez, d_T_pstar_z2t, d_T_pstar_vlm) are
included in all plots when present in the data.  Baseline conditions (Mono,
Solo c₁, Solo c₂) are shown with solid lines; p* sources with dashed lines.
Run measure_composability_gap.py with --pstar-source X to add p* columns.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .utils import (
    TERM_COLOR, TERM_LABEL, TRAJ_COLOR, TRAJ_LABEL,
    TERM_CONDITIONS, TRAJ_CONDITIONS,
    PAIR_PALETTE, STEP_BINS, BIN_LABELS, BIN_ALPHAS,
    LABEL_D_T_MSE, LABEL_D_t_MSE, LABEL_CUM_DELTA_D_t,
    SCIPY_OK, stats,
    cap_pairs, get_present_poe, get_present_traj_poe,
    get_present_pstar, get_present_traj_pstar, apply_pstar_filter,
    pair_color_map, short_pair, traj_stats, bin_increments,
    hide_top_right, save_fig,
)


def _terminal_baselines(df_term):
    return TERM_CONDITIONS + get_present_poe(df_term)


def _terminal_conditions(df_term, pstar_filter):
    return _terminal_baselines(df_term) + apply_pstar_filter(
        get_present_pstar(df_term), pstar_filter
    )


def _traj_baselines(df_traj):
    return TRAJ_CONDITIONS + get_present_traj_poe(df_traj)


def _traj_conditions(df_traj, pstar_filter):
    return _traj_baselines(df_traj) + apply_pstar_filter(
        get_present_traj_pstar(df_traj), pstar_filter
    )


def _plot04_kde_params(all_vals, **kw):
    x_min = float(all_vals.min() * 0.85)
    x_max = float(all_vals.max() * 1.08)
    xq = float(kw.get("plot04_xmax_quantile", 1.0))
    if 0.0 < xq < 1.0:
        x_qmax = float(np.quantile(all_vals, xq))
        x_max = max(x_qmax, x_min + 1e-6)
    x_grid = np.linspace(x_min, x_max, 600)

    bw_scale = max(float(kw.get("plot04_bw_scale", 1.0)), 1e-6)
    bw_method = lambda kde: max(kde.scotts_factor() * bw_scale, 1e-6)
    return x_grid, xq, bw_method


# ===========================================================================
# Plot 00 — Strip chart: raw dots (one per seed) + bold mean tick, per pair
#           Foundation — shows raw data before any aggregation.
#           p* columns are included as additional x positions when present.
# ===========================================================================

def plot_00(df_term, df_traj, out_dir, **kw):
    pairs     = cap_pairs(sorted(df_term["pair"].unique()), kw.get("max_display_pairs"))
    all_conds = _terminal_conditions(df_term, kw.get("pstar_filter"))
    n_cond    = len(all_conds)
    rng       = np.random.default_rng(42)
    x_pos     = {c: float(i) for i, c in enumerate(all_conds)}

    y_max = df_term[all_conds].values.max()
    y_top = max(y_max * 1.08, 2.15)

    panel_w = max(4.5, 1.6 * n_cond)
    fig, axes = plt.subplots(1, len(pairs), figsize=(panel_w * len(pairs), 5.5), sharey=True)
    if len(pairs) == 1:
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

        ax.axhline(2.0, color="#CCCCCC", lw=1.2, ls=":", zorder=1)
        ax.set_title(short_pair(pair), fontsize=11)
        ax.set_xticks(list(range(n_cond)))
        ax.set_xticklabels(
            [TERM_LABEL[c] for c in all_conds],
            fontsize=9, rotation=30 if n_cond > 3 else 0,
            ha="right" if n_cond > 3 else "center",
        )
        ax.set_xlim(-0.5, n_cond - 0.5)
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
               loc="upper right", fontsize=10, title_fontsize=11,
               frameon=True, bbox_to_anchor=(1.0, 0.93))

    fig.suptitle(
        "Plot 00 — Terminal Latent Distance from SuperDiff-AND  (per Concept Pair)\n"
        "Bold tick = mean across seeds.  Each dot = one seed.",
        fontsize=13,
    )
    fig.text(0.5, -0.02,
             "Scale:  0 = identical to AND  │  ≈0.25 subtle  │  ≈0.60 noticeable  "
             "│  ≈2.0 unrelated latents (ceiling)",
             ha="center", fontsize=9, color="#888888")

    fig.tight_layout()
    save_fig(fig, out_dir / "plot_00_strip_by_pair.png")


# ===========================================================================
# Plot 01 — Grouped bar: mean ± std, x = pair, grouped by condition
#           p* sources add extra bars within each pair group.
# ===========================================================================

def plot_01(df_term, df_traj, out_dir, **kw):
    pairs     = sorted(df_term["pair"].unique())
    all_conds = _terminal_conditions(df_term, kw.get("pstar_filter"))
    n_cond    = len(all_conds)
    x         = np.arange(len(pairs))
    width     = 0.8 / n_cond
    offsets   = (np.arange(n_cond) - (n_cond - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(pairs)), 5))

    for i, cond in enumerate(all_conds):
        means = [df_term[df_term["pair"] == p][cond].mean() for p in pairs]
        stds  = [df_term[df_term["pair"] == p][cond].std()  for p in pairs]
        ax.bar(x + offsets[i], means, width * 0.9,
               yerr=stds, capsize=4, error_kw={"lw": 1.5, "ecolor": "#444"},
               color=TERM_COLOR[cond], label=TERM_LABEL[cond], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([short_pair(p) for p in pairs],
                       fontsize=10, rotation=45, ha="right")
    ax.set_ylabel(LABEL_D_T_MSE, fontsize=11)
    ax.set_title(
        "Plot 01 — Terminal Latent Distance from SuperDiff-AND\n"
        "Grouped bar: mean ± 1 std",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    hide_top_right(ax)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    save_fig(fig, out_dir / "plot_01_terminal_bars.png")


# ===========================================================================
# Plot 02 — Grouped bar histogram: x=distance bins, bars grouped by pair,
#           one panel per condition.  p* add extra panels when present.
# ===========================================================================

def plot_02(df_term, df_traj, out_dir, **kw):
    pairs     = cap_pairs(sorted(df_term["pair"].unique()), kw.get("max_display_pairs"))
    all_conds = _terminal_conditions(df_term, kw.get("pstar_filter"))
    n_cond    = len(all_conds)
    pcmap     = pair_color_map(pairs)
    n_pairs   = len(pairs)

    all_vals  = df_term[all_conds].values.flatten()
    bin_edges = np.linspace(all_vals.min() * 0.85, all_vals.max() * 1.08, 10)  # 9 bins
    bin_ctrs  = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_w     = bin_edges[1] - bin_edges[0]

    bar_w    = bin_w * 0.92 / n_pairs
    p_offset = (np.arange(n_pairs) - (n_pairs - 1) / 2) * bar_w

    fig, axes = plt.subplots(1, n_cond, figsize=(4.8 * n_cond, 4.5), sharey=True, sharex=True)
    if n_cond == 1:
        axes = [axes]

    for ax, cond in zip(axes, all_conds):
        for pi, pair in enumerate(pairs):
            vals   = df_term[df_term["pair"] == pair][cond].values
            counts, _ = np.histogram(vals, bins=bin_edges)
            ax.bar(bin_ctrs + p_offset[pi], counts, bar_w,
                   color=pcmap[pair], label=short_pair(pair),
                   alpha=0.87, edgecolor="white", linewidth=0.6)

        ax.set_title(TERM_LABEL[cond], fontsize=12, color=TERM_COLOR[cond])
        ax.set_xlabel(r"$d_T$  (per-element MSE)", fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        hide_top_right(ax)

    axes[0].set_ylabel("Count  (seeds)", fontsize=11)
    axes[-1].legend(fontsize=9, loc="upper right", title="Concept pair")

    fig.suptitle(
        "Plot 02 — Terminal Distance: Grouped Bar Histogram  (colour = concept pair)\n"
        "One panel per condition,  9 bins",
        fontsize=13,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_02_histogram_grouped.png")


# ===========================================================================
# Plot 03 — Pooled strip: all seeds from all pairs, bold tick = pooled mean,
#           colour = condition.  Direct bridge from per-pair view to KDE.
#           p* sources appear as additional x positions when present.
# ===========================================================================

def plot_03(df_term, df_traj, out_dir, **kw):
    all_conds = _terminal_conditions(df_term, kw.get("pstar_filter"))
    n_cond    = len(all_conds)
    rng       = np.random.default_rng(42)
    x_pos     = {c: float(i) for i, c in enumerate(all_conds)}

    y_max = df_term[all_conds].values.max()
    y_top = max(y_max * 1.08, 2.15)

    n_pairs = df_term["pair"].nunique()
    n_seeds = df_term["seed"].nunique()
    n_total = len(df_term)

    fig, ax = plt.subplots(figsize=(max(7, 2.0 * n_cond), 5.5))

    for cond in all_conds:
        vals   = df_term[cond].values
        x      = x_pos[cond]
        jitter = rng.uniform(-0.22, 0.22, len(vals))
        c      = TERM_COLOR[cond]

        ax.scatter(x + jitter, vals, color=c, s=45, alpha=0.50,
                   linewidths=1.0, edgecolors="white", zorder=3)
        ax.plot([x - 0.32, x + 0.32], [vals.mean(), vals.mean()],
                color=c, lw=5.5, solid_capstyle="butt", zorder=4)

    ax.axhline(2.0, color="#CCCCCC", lw=1.2, ls=":", zorder=1)
    ax.set_xticks(list(range(n_cond)))
    ax.set_xticklabels(
        [TERM_LABEL[c] for c in all_conds],
        fontsize=11, rotation=30 if n_cond > 3 else 0,
        ha="right" if n_cond > 3 else "center",
    )
    ax.set_xlim(-0.6, n_cond - 0.4)
    ax.set_ylim(bottom=0, top=y_top)
    ax.set_ylabel(LABEL_D_T_MSE, fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    hide_top_right(ax)

    ax.set_title(
        "Plot 03 — Pooled Terminal Distance  (all concept pairs combined)\n"
        f"Bold tick = mean across all seeds & pairs.  "
        f"N = {n_total} dots per condition  ({n_pairs} pairs × {n_seeds} seeds)",
        fontsize=12,
    )
    ax.text(0.5, -0.10 if n_cond > 3 else -0.08,
            "Scale:  0 = identical to AND  │  ≈0.25 subtle  │  ≈0.60 noticeable  "
            "│  ≈2.0 ceiling",
            ha="center", transform=ax.transAxes, fontsize=9, color="#888888")

    fig.tight_layout()
    save_fig(fig, out_dir / "plot_03_pooled_strip.png")


# ===========================================================================
# Plot 04 — KDE pooled: smooth of plot 03; all pairs combined, colour=condition
#           Baselines = solid lines; p* sources = dashed lines.
# ===========================================================================

def plot_04(df_term, df_traj, out_dir, **kw):
    if not SCIPY_OK:
        print("  Skipping plot 04 (scipy required).")
        return

    anchor_mode = kw.get("and_anchor", "seed")
    anchor_desc = "per-seed paired AND anchor" if anchor_mode == "seed" else "mean AND anchor"

    baseline_conds = _terminal_baselines(df_term)
    all_conds = baseline_conds + apply_pstar_filter(get_present_pstar(df_term), kw.get("pstar_filter"))
    all_vals  = df_term[all_conds].values.flatten()
    x_grid, xq, bw_method = _plot04_kde_params(all_vals, **kw)
    n         = len(df_term)

    plot04_scale = max(float(kw.get("plot04_scale", 1.0)), 0.25)
    fig, ax = plt.subplots(figsize=(8 * plot04_scale, 5 * plot04_scale))
    for cond in all_conds:
        vals = df_term[cond].values
        dens = stats.gaussian_kde(vals, bw_method=bw_method)(x_grid)
        ls   = "--" if cond not in baseline_conds else "-"
        ax.plot(x_grid, dens, color=TERM_COLOR[cond], label=TERM_LABEL[cond],
                lw=2.5, ls=ls)
        ax.fill_between(x_grid, dens, alpha=0.13, color=TERM_COLOR[cond])

    ax.set_xlabel("Terminal distance to SuperDiff AND (per-element MSE)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Plot 04 — Terminal Distance Distribution\n"
        f"Anchor: {anchor_desc}",
        fontsize=13,
    )
    if 0.0 < xq < 1.0:
        ax.set_xlim(left=max(0.0, x_grid.min()), right=x_grid.max())
        ax.text(
            0.99, 0.97, f"x <= q{xq * 100:.1f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#666666"
        )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.25)
    hide_top_right(ax)
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_04_kde_pooled.png")


# ===========================================================================
# Plot 05 — KDE faceted: one panel per concept pair, colour = condition
#           Baselines = solid lines; p* sources = dashed lines.
# ===========================================================================

def plot_05(df_term, df_traj, out_dir, **kw):
    if not SCIPY_OK:
        print("  Skipping plot 05 (scipy required).")
        return

    pairs     = cap_pairs(sorted(df_term["pair"].unique()), kw.get("max_display_pairs"))
    baseline_conds = _terminal_baselines(df_term)
    all_conds = baseline_conds + apply_pstar_filter(get_present_pstar(df_term), kw.get("pstar_filter"))

    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 4.5), sharey=False)
    if len(pairs) == 1:
        axes = [axes]

    for ax, pair in zip(axes, pairs):
        sub      = df_term[df_term["pair"] == pair]
        all_vals = sub[all_conds].values.flatten()
        x_grid   = np.linspace(all_vals.min() * 0.85, all_vals.max() * 1.08, 400)

        for cond in all_conds:
            vals = sub[cond].values
            dens = stats.gaussian_kde(vals, bw_method="scott")(x_grid)
            ls   = "--" if cond not in baseline_conds else "-"
            ax.plot(x_grid, dens, color=TERM_COLOR[cond], label=TERM_LABEL[cond],
                    lw=2, ls=ls)
            ax.fill_between(x_grid, dens, alpha=0.12, color=TERM_COLOR[cond])

        ax.set_title(short_pair(pair), fontsize=11)
        ax.set_xlabel(r"$d_T$  (per-element MSE)", fontsize=10)
        ax.grid(alpha=0.2)
        hide_top_right(ax)

    axes[0].set_ylabel("Density", fontsize=11)
    axes[-1].legend(fontsize=9, loc="upper right")
    fig.suptitle(
        "Plot 05 — Per-Pair Terminal Distance Distribution (KDE)\n"
        "One panel per concept pair,  colour = condition  "
        "(baselines solid · p* dashed)",
        fontsize=13,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_05_kde_by_pair.png")


# ===========================================================================
# Plot 06 — Stacked temporal bar (pooled across all pairs + seeds).
#
#   x = condition (all traj conditions + p* when present)
#   Stack = 5 time bins (0-10, 10-20, 20-30, 30-40, 40-50)
#   Each segment height = mean incremental divergence Δd over that 10-step window
#   Bar total height = terminal distance d_T  (because d_0 = 0)
#   Alpha gradient: opaque bottom (early) → transparent top (late)
# ===========================================================================

def plot_06(df_term, df_traj, out_dir, **kw):
    conditions = _traj_conditions(df_traj, kw.get("pstar_filter"))
    n_cond     = len(conditions)
    x_pos      = {c: float(i) for i, c in enumerate(conditions)}
    bar_width  = 0.52

    fig, ax = plt.subplots(figsize=(max(8, 2.2 * n_cond), 5.5))

    for cond in conditions:
        x   = x_pos[cond]
        c   = TRAJ_COLOR[cond]
        inc = bin_increments(df_traj, cond)

        bottom = 0.0
        for bi, height in enumerate(inc):
            ax.bar(x, height, bar_width, bottom=bottom,
                   color=c, alpha=BIN_ALPHAS[bi],
                   edgecolor="white", linewidth=0.8)
            bottom += height

        # Placeholder annotation — will be redrawn after autoscale
        ax.text(x, bottom, "", ha="center", va="bottom")

    # Recompute y-limit now that bars are placed
    ax.relim(); ax.autoscale_view()
    ymax = ax.get_ylim()[1]
    for text in ax.texts:
        text.remove()
    for cond in conditions:
        x     = x_pos[cond]
        c     = TRAJ_COLOR[cond]
        total = sum(bin_increments(df_traj, cond))
        ax.text(x, total + ymax * 0.015, f"{total:.3f}",
                ha="center", va="bottom", fontsize=9.5, color=c, fontweight="bold")

    ax.set_xticks([x_pos[c] for c in conditions])
    ax.set_xticklabels(
        [TRAJ_LABEL[c] for c in conditions],
        fontsize=11, rotation=30 if n_cond > 3 else 0,
        ha="right" if n_cond > 3 else "center",
    )
    ax.set_xlim(-0.6, n_cond - 0.4)
    ax.set_ylabel(LABEL_CUM_DELTA_D_t, fontsize=11)
    ax.set_title(
        "Plot 06 — Temporal Divergence from AND\n"
        "Opaque = early denoising steps;  transparent = late.",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.25)
    hide_top_right(ax)

    bin_handles = [
        Patch(facecolor="#777777", alpha=BIN_ALPHAS[bi],
              edgecolor="white", label=f"Steps {BIN_LABELS[bi]}")
        for bi in range(len(STEP_BINS))
    ]
    ax.legend(handles=bin_handles, title="Time bin", fontsize=9,
              loc="upper right", title_fontsize=10)

    fig.tight_layout()
    save_fig(fig, out_dir / "plot_06_stacked_bar_pooled.png")


# ===========================================================================
# Plot 26 — Combined view of plot 04 (left) and plot 06 (right).
#            Both subplots are rendered with equal panel sizes.
# ===========================================================================

def plot_26(df_term, df_traj, out_dir, **kw):
    if not SCIPY_OK:
        print("  Skipping plot 26 (scipy required for KDE panel).")
        return

    plot26_scale = max(float(kw.get("plot26_scale", 1.0)), 0.25)
    anchor_mode = kw.get("and_anchor", "seed")
    anchor_desc = "per-seed paired AND anchor" if anchor_mode == "seed" else "mean AND anchor"
    term_baselines = _terminal_baselines(df_term)
    term_conds = term_baselines + apply_pstar_filter(get_present_pstar(df_term), kw.get("pstar_filter"))
    traj_conds = _traj_conditions(df_traj, kw.get("pstar_filter"))

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(18 * plot26_scale, 6 * plot26_scale), gridspec_kw={"width_ratios": [1, 1]}
    )

    # ---- Left panel: plot 04 (KDE pooled) ----
    all_vals = df_term[term_conds].values.flatten()
    x_grid, xq, bw_method = _plot04_kde_params(all_vals, **kw)
    n = len(df_term)

    for cond in term_conds:
        vals = df_term[cond].values
        dens = stats.gaussian_kde(vals, bw_method=bw_method)(x_grid)
        ls = "--" if cond not in term_baselines else "-"
        ax_left.plot(x_grid, dens, color=TERM_COLOR[cond], label=TERM_LABEL[cond], lw=2.3, ls=ls)
        ax_left.fill_between(x_grid, dens, alpha=0.13, color=TERM_COLOR[cond])

    ax_left.set_xlabel("Terminal distance to SuperDiff AND (per-element MSE)", fontsize=11)
    ax_left.set_ylabel("Density", fontsize=11)
    ax_left.set_title(
        f"Plot 04 — Terminal Distance Distribution\n"
        f"Anchor: {anchor_desc}",
        fontsize=12,
    )
    if 0.0 < xq < 1.0:
        ax_left.set_xlim(left=max(0.0, x_grid.min()), right=x_grid.max())
        ax_left.text(
            0.99, 0.97, f"x <= q{xq * 100:.1f}",
            transform=ax_left.transAxes, ha="right", va="top",
            fontsize=8.5, color="#666666"
        )
    ax_left.legend(fontsize=9, loc="upper right")
    ax_left.grid(alpha=0.25)
    hide_top_right(ax_left)

    # ---- Right panel: plot 06 (stacked temporal bar) ----
    x_pos = {c: float(i) for i, c in enumerate(traj_conds)}
    bar_width = 0.52

    for cond in traj_conds:
        x = x_pos[cond]
        c = TRAJ_COLOR[cond]
        inc = bin_increments(df_traj, cond)

        bottom = 0.0
        for bi, height in enumerate(inc):
            ax_right.bar(x, height, bar_width, bottom=bottom,
                         color=c, alpha=BIN_ALPHAS[bi],
                         edgecolor="white", linewidth=0.8)
            bottom += height

        # Placeholder annotation — redrawn after autoscale
        ax_right.text(x, bottom, "", ha="center", va="bottom")

    ax_right.relim()
    ax_right.autoscale_view()
    ymax = ax_right.get_ylim()[1]
    for text in ax_right.texts:
        text.remove()
    for cond in traj_conds:
        x = x_pos[cond]
        c = TRAJ_COLOR[cond]
        total = sum(bin_increments(df_traj, cond))
        ax_right.text(x, total + ymax * 0.015, f"{total:.3f}",
                      ha="center", va="bottom", fontsize=9.0,
                      color=c, fontweight="bold")

    ax_right.set_xticks([x_pos[c] for c in traj_conds])
    ax_right.set_xticklabels(
        [TRAJ_LABEL[c] for c in traj_conds],
        fontsize=10, rotation=30 if len(traj_conds) > 3 else 0,
        ha="right" if len(traj_conds) > 3 else "center",
    )
    ax_right.set_xlim(-0.6, len(traj_conds) - 0.4)
    ax_right.set_ylabel(LABEL_CUM_DELTA_D_t, fontsize=11)
    ax_right.set_title(
        "Plot 06 — Temporal Divergence from AND\n"
        "Opaque = early denoising; transparent = late.",
        fontsize=12,
    )
    ax_right.grid(axis="y", alpha=0.25)
    hide_top_right(ax_right)

    bin_handles = [
        Patch(facecolor="#777777", alpha=BIN_ALPHAS[bi],
              edgecolor="white", label=f"Steps {BIN_LABELS[bi]}")
        for bi in range(len(STEP_BINS))
    ]
    ax_right.legend(handles=bin_handles, title="Time bin", fontsize=8.5,
                    loc="upper right", title_fontsize=9.5)

    fig.suptitle(
        "Observed pattern: terminal distance to AND remains elevated, and most divergence accumulates late in denoising",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, out_dir / "plot_26_plot_04_and_06.png")


# ===========================================================================
# Plot 07 — Stacked temporal bar by concept pair.
#   N_pairs groups, N_cond bars each, 5 stacked time-bin segments.
#   p* sources add extra bars within each group when present.
# ===========================================================================

def plot_07(df_term, df_traj, out_dir, **kw):
    pairs      = cap_pairs(sorted(df_traj["pair"].unique()), kw.get("max_display_pairs"))
    conditions = _traj_conditions(df_traj, kw.get("pstar_filter"))
    n_cond     = len(conditions)
    n_pairs    = len(pairs)

    bar_width    = 0.72 / n_cond
    cond_offsets = (np.arange(n_cond) - (n_cond - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * n_cond) * n_pairs, 5.5))

    group_x = np.arange(n_pairs, dtype=float)

    for pi, pair in enumerate(pairs):
        sub = df_traj[df_traj["pair"] == pair]
        for ci, cond in enumerate(conditions):
            x   = group_x[pi] + cond_offsets[ci]
            c   = TRAJ_COLOR[cond]
            inc = bin_increments(sub, cond)

            bottom = 0.0
            for bi, height in enumerate(inc):
                ax.bar(x, height, bar_width * 0.92, bottom=bottom,
                       color=c, alpha=BIN_ALPHAS[bi],
                       edgecolor="white", linewidth=0.5)
                bottom += height

    ax.set_xticks(group_x)
    ax.set_xticklabels([short_pair(p) for p in pairs],
                       fontsize=10, rotation=45, ha="right")
    ax.set_xlim(-0.5, n_pairs - 0.5)
    ax.set_ylabel(LABEL_CUM_DELTA_D_t, fontsize=11)
    ax.set_title(
        "Plot 07 — Temporal Divergence from AND  (stacked bar, by concept pair)\n"
        "Opaque = early denoising steps;  transparent = late.  "
        "Bar height = mean terminal MSE d_T.",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.25)
    hide_top_right(ax)

    cond_handles = [
        Patch(facecolor=TRAJ_COLOR[c], alpha=0.88,
              edgecolor="white", label=TRAJ_LABEL[c])
        for c in conditions
    ]
    bin_handles = [
        Patch(facecolor="#777777", alpha=BIN_ALPHAS[bi],
              edgecolor="white", label=f"Steps {BIN_LABELS[bi]}")
        for bi in range(len(STEP_BINS))
    ]

    leg1 = ax.legend(handles=cond_handles, title="Condition",
                     fontsize=9, loc="upper right", title_fontsize=10)
    ax.add_artist(leg1)
    ax.legend(handles=bin_handles, title="Time bin",
              fontsize=9, loc="upper left", title_fontsize=10)

    fig.tight_layout()
    save_fig(fig, out_dir / "plot_07_stacked_bar_by_pair.png")


# ===========================================================================
# Plot 08 — Per-condition temporal: one subplot per condition, colour = pair
#           Expands from 3 subplots to N when p* trajectory columns are present.
# ===========================================================================

def plot_08(df_term, df_traj, out_dir, **kw):
    pairs          = cap_pairs(sorted(df_traj["pair"].unique()), kw.get("max_display_pairs"))
    all_traj_conds = _traj_conditions(df_traj, kw.get("pstar_filter"))
    n_cond         = len(all_traj_conds)
    pcmap          = pair_color_map(pairs)
    steps          = sorted(df_traj["step"].unique())
    df_cap         = df_traj[df_traj["pair"].isin(pairs)]

    fig, axes = plt.subplots(1, n_cond, figsize=(5 * n_cond, 5), sharey=True)
    if n_cond == 1:
        axes = [axes]

    for ax, cond in zip(axes, all_traj_conds):
        for pair in pairs:
            c          = pcmap[pair]
            sub        = df_cap[df_cap["pair"] == pair]
            means, stds = traj_stats(sub, cond, steps)
            ax.plot(steps, means, color=c, label=short_pair(pair), lw=2)
            ax.fill_between(steps, means - stds, means + stds, alpha=0.15, color=c)
        ax.set_title(TRAJ_LABEL[cond], fontsize=12, color=TRAJ_COLOR[cond])
        ax.set_xlabel("Denoising step", fontsize=11)
        ax.grid(alpha=0.2)
        hide_top_right(ax)

    axes[0].set_ylabel(LABEL_D_t_MSE, fontsize=11)
    axes[-1].legend(fontsize=9, loc="upper left", title="Concept pair")
    fig.suptitle(
        "Plot 08 — Per-Condition Temporal Detail  (Mean ± Std,  colour = pair)",
        fontsize=13,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_08_per_condition_temporal.png")


# ===========================================================================
# Plot 09 — Individual seed trajectories (thin α) + bold mean, colour = pair
#           Expands to N subplots when p* columns are present.
#           Only iterates capped pairs in the groupby (fixes KeyError with cap).
# ===========================================================================

def plot_09(df_term, df_traj, out_dir, **kw):
    pairs          = cap_pairs(sorted(df_traj["pair"].unique()), kw.get("max_display_pairs"))
    all_traj_conds = _traj_conditions(df_traj, kw.get("pstar_filter"))
    n_cond         = len(all_traj_conds)
    pcmap          = pair_color_map(pairs)
    steps          = sorted(df_traj["step"].unique())
    df_cap         = df_traj[df_traj["pair"].isin(pairs)]  # restrict to capped pairs

    fig, axes = plt.subplots(1, n_cond, figsize=(5 * n_cond, 5), sharey=True)
    if n_cond == 1:
        axes = [axes]

    for ax, cond in zip(axes, all_traj_conds):
        # Thin per-seed traces (only capped pairs)
        for (pair, _seed), grp in df_cap.groupby(["pair", "seed"]):
            grp = grp.sort_values("step")
            ax.plot(grp["step"], grp[cond], color=pcmap[pair], lw=0.6, alpha=0.22)

        # Bold per-pair means
        for pair in pairs:
            c     = pcmap[pair]
            sub   = df_cap[df_cap["pair"] == pair]
            means = sub.groupby("step")[cond].mean().reindex(steps).values
            ax.plot(steps, means, color=c, lw=2.5, label=short_pair(pair))

        ax.set_title(TRAJ_LABEL[cond], fontsize=12, color=TRAJ_COLOR[cond])
        ax.set_xlabel("Denoising step", fontsize=11)
        ax.grid(alpha=0.2)
        hide_top_right(ax)

    axes[0].set_ylabel(LABEL_D_t_MSE, fontsize=11)
    axes[-1].legend(fontsize=9, loc="upper left", title="Concept pair")
    fig.suptitle(
        "Plot 09 — Individual Seed Trajectories (thin, α=0.22) + Mean (bold)\n"
        "Colour = concept pair",
        fontsize=13,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_09_individual_seeds.png")


# ===========================================================================
# Plot 10 — Generalised: one subplot per condition, all pairs+seeds pooled
#           Expands to N subplots when p* trajectory columns are present.
# ===========================================================================

def plot_10(df_term, df_traj, out_dir, **kw):
    all_traj_conds = _traj_conditions(df_traj, kw.get("pstar_filter"))
    n_cond         = len(all_traj_conds)
    steps          = sorted(df_traj["step"].unique())
    n_per_step     = int(df_traj[df_traj["step"] == steps[0]].shape[0])
    n_pairs        = df_traj["pair"].nunique()
    n_seeds        = df_traj["seed"].nunique()

    fig, axes = plt.subplots(1, n_cond, figsize=(5 * n_cond, 5), sharey=True)
    if n_cond == 1:
        axes = [axes]

    for ax, cond in zip(axes, all_traj_conds):
        means, stds = traj_stats(df_traj, cond, steps)
        ci95 = 1.96 * stds / np.sqrt(n_per_step)
        c    = TRAJ_COLOR[cond]

        ax.plot(steps, means, color=c, lw=2.5)
        ax.fill_between(steps, means - stds,  means + stds,
                        alpha=0.12, color=c, label="±1 std")
        ax.fill_between(steps, means - ci95,  means + ci95,
                        alpha=0.30, color=c, label="95 % CI")

        ax.set_title(TRAJ_LABEL[cond], fontsize=13, color=c, fontweight="bold")
        ax.set_xlabel("Denoising step", fontsize=11)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.2)
        hide_top_right(ax)

    axes[0].set_ylabel(LABEL_D_t_MSE, fontsize=11)
    fig.suptitle(
        "Plot 10 — Generalised Temporal Distance (All Pairs + Seeds Pooled)\n"
        f"N = {n_pairs} pairs × {n_seeds} seeds = {n_pairs * n_seeds} trajectories per condition",
        fontsize=13,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_10_generalised.png")
