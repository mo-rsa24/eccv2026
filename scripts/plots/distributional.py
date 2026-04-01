"""
Gap analysis — distributional plots 20–24 and 29–34 (Jeffrey's divergence suite).

  Distributional analysis — Jeffrey's divergence  (data: per_seed + within_and)
    20  ECDF comparison     — F(x) for all conditions + within-AND; non-parametric
                              Left shift = closer to AND; overlap with grey = gap closed
    21  Jeffrey's heatmap   — J(P_i, P_j) for all condition pairs; cluster structure
                              p* clustering with within-AND → gap closed
    22  Expressiveness ladder — J(p* source, within-AND) vs method expressiveness rank
                              THE paper figure: downward trend = language can express AND
    23  p* vs AND KL         — p* sources only, KL to SuperDiff-AND proxy
    24  Pointwise contributions — shaded KL/Jeffrey integrands (area = divergence)

  Plot-24 scaffolding (simple → full)
    29  Raw pooled samples   — where d_T observations come from
    30  Shared-bin histogram — convert samples to comparable discrete mass
    34  Shared-bin KL bridge — p_i/q_i → log-ratio → local KL → cumulative KL
    31  KDE overlays         — smooth shared-grid densities
    32  Pointwise KL terms   — local signed KL contributions by d_T
    33  Cumulative KL curves — running integral ending at total KL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .utils import (
    TERM_COLOR, TERM_LABEL,
    TERM_CONDITIONS,
    LABEL_D_T_MSE, LABEL_ECDF, LABEL_JEFFREYS, LABEL_J_SOURCE_WITHIN,
    SCIPY_OK, stats,
    get_present_poe, get_present_pstar, pair_color_map, short_pair,
    kde_pmf, jeffreys_div, ecdf_xy,
    hide_top_right, save_fig,
    load_within_and,
)


def _prepare_pointwise_inputs(df_term, data_dir, n_grid=1200):
    """Shared prep for plots 24 and 29–34."""
    if not SCIPY_OK:
        print("  Skipping (scipy required).")
        return None
    if data_dir is None:
        print("  Skipping: data_dir required for within_and_distances.json.")
        return None
    raw = load_within_and(data_dir)
    if raw is None:
        print("  Skipping: within_and_distances.json not found.")
        return None

    present_pstar = get_present_pstar(df_term)
    if not present_pstar:
        print("  Skipping: no d_T_pstar_* columns found.")
        return None

    within_vals = pd.DataFrame(raw)["d_within_and"].values
    all_vals = np.concatenate([within_vals] + [df_term[c].values for c in present_pstar])
    x_grid = np.linspace(max(all_vals.min() * 0.70, 0), all_vals.max() * 1.10, n_grid)
    dx = float(x_grid[1] - x_grid[0])

    p_and = kde_pmf(within_vals, x_grid)
    src_pmfs = {cond: kde_pmf(df_term[cond].values, x_grid) for cond in present_pstar}
    src_vals = {cond: df_term[cond].values for cond in present_pstar}
    return within_vals, present_pstar, x_grid, dx, p_and, src_pmfs, src_vals


# ===========================================================================
# Plot 20 — ECDF comparison: all conditions + within-AND noise floor.
#
#   Non-parametric: no bandwidth assumption needed — each curve is the
#   empirical CDF F(x) = P(distance ≤ x) computed directly from sorted data.
#
#   Key reading:
#     Curve left-shifted  → closer to AND (gap more closed)
#     Curve crossing within-AND → those seeds are within AND's own variance
#     Horizontal gap between two curves at p=0.5 = median difference
#     Area between two ECDF curves = Wasserstein-1 distance (in MSE units)
#
#   Solid lines = p* sources.  Dashed = baselines.  Grey dashed = within-AND.
#   Dotted verticals = per-condition medians.
# ===========================================================================

def plot_20(df_term, df_traj, out_dir, data_dir=None, **kw):
    if not SCIPY_OK:
        print("  Skipping plot 20 (scipy required for ECDF helpers).")
        return

    # Load within-AND reference distribution
    within_vals = None
    if data_dir is not None:
        raw = load_within_and(data_dir)
        if raw is not None:
            within_vals = pd.DataFrame(raw)["d_within_and"].values

    present_pstar  = get_present_pstar(df_term)
    pstar_conds    = present_pstar
    baseline_conds = TERM_CONDITIONS + get_present_poe(df_term)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # --- Within-AND reference (grey dashed + 5–95 percentile band) ---
    if within_vals is not None:
        wx, wy = ecdf_xy(within_vals)
        ax.step(wx, wy, color="#888888", lw=2.2, ls="--", where="post",
                label="Within-AND (noise floor)", zorder=5)
        ax.axvspan(np.percentile(within_vals, 5), np.percentile(within_vals, 95),
                   color="#888888", alpha=0.07, zorder=1,
                   label="Within-AND 5–95 pct")
        ax.axvline(np.median(within_vals), color="#888888", lw=0.8,
                   ls=":", alpha=0.50)

    # --- p* sources: solid lines ---
    for cond in pstar_conds:
        vals   = df_term[cond].values
        x, y   = ecdf_xy(vals)
        ax.step(x, y, color=TERM_COLOR[cond], lw=2.5, where="post",
                label=TERM_LABEL[cond], zorder=4)
        ax.axvline(np.median(vals), color=TERM_COLOR[cond],
                   lw=0.8, ls=":", alpha=0.45)

    # --- Baselines: dashed lines ---
    for cond in baseline_conds:
        vals   = df_term[cond].values
        x, y   = ecdf_xy(vals)
        ax.step(x, y, color=TERM_COLOR[cond], lw=2.0, ls="--", where="post",
                label=TERM_LABEL[cond], alpha=0.80, zorder=3)
        ax.axvline(np.median(vals), color=TERM_COLOR[cond],
                   lw=0.8, ls=":", alpha=0.40)

    ax.set_xlabel(LABEL_D_T_MSE, fontsize=12)
    ax.set_ylabel(LABEL_ECDF, fontsize=12)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.04)

    pstar_names = " / ".join(TERM_LABEL[c] for c in pstar_conds) if pstar_conds else "—"
    n = len(df_term)
    ax.set_title(
        "Plot 20 — ECDF: Distance from AND  (all conditions + within-AND)\n"
        f"Solid = p* [{pstar_names}]  ·  Dashed = baselines  ·  Grey = within-AND\n"
        f"All pairs pooled  (N = {n} per condition).  "
        "Left shift ≈ closer to AND.  Overlap with grey ≈ gap closed.",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.grid(alpha=0.20)
    hide_top_right(ax)
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_20_ecdf_comparison.png")


# ===========================================================================
# Plot 21 — Jeffrey's divergence heatmap: all pairwise conditions (pooled).
#
#   J(P,Q) = KL(P||Q) + KL(Q||P)  — symmetric, range [0,∞).
#   More sensitive to tail differences than JS².
#   Distributions estimated via KDE (Scott bandwidth) on a shared grid.
#
#   Conditions: [within-AND, p* sources…, mono, solo c₁, solo c₂]
#   Lower-left cluster around within-AND = those methods approach AND's
#   own variability.  A p* source clustering away with mono confirms the
#   gap is outside the text-conditioned manifold.
# ===========================================================================

def plot_21(df_term, df_traj, out_dir, data_dir=None, **kw):
    if not SCIPY_OK:
        print("  Skipping plot 21 (scipy required).")
        return

    within_vals = None
    if data_dir is not None:
        raw = load_within_and(data_dir)
        if raw is not None:
            within_vals = pd.DataFrame(raw)["d_within_and"].values

    present_pstar = get_present_pstar(df_term)

    # Build ordered condition list: within-AND first, then p* (most→least expressive),
    # then baselines.  Labels align with matrix axes.
    cond_registry = []
    if within_vals is not None:
        cond_registry.append(("Within-AND", "#888888", within_vals))
    for c in present_pstar:
        cond_registry.append((TERM_LABEL[c], TERM_COLOR[c], df_term[c].values))
    for c in TERM_CONDITIONS + get_present_poe(df_term):
        cond_registry.append((TERM_LABEL[c], TERM_COLOR[c], df_term[c].values))

    n_conds = len(cond_registry)
    if n_conds < 2:
        print("  Skipping plot 21: need at least 2 conditions (run measure_composability_gap.py first).")
        return

    # Shared x-grid spanning the full range of all distributions
    all_vals = np.concatenate([v for _, _, v in cond_registry])
    x_grid   = np.linspace(max(all_vals.min() * 0.70, 0),
                           all_vals.max() * 1.10, 1000)

    # Compute PMFs and Jeffrey's matrix
    pmfs  = [kde_pmf(v, x_grid) for _, _, v in cond_registry]
    labels = [lbl                for lbl, _, _ in cond_registry]
    colors = [col                for _, col, _ in cond_registry]

    J_mat = np.zeros((n_conds, n_conds))
    for i in range(n_conds):
        for j in range(i + 1, n_conds):
            j_val = jeffreys_div(pmfs[i], pmfs[j])
            J_mat[i, j] = j_val
            J_mat[j, i] = j_val

    fig_sz = max(7, n_conds * 1.15)
    fig, ax = plt.subplots(figsize=(fig_sz, fig_sz * 0.88))

    im = ax.imshow(J_mat, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=J_mat.max() * 1.05)
    plt.colorbar(im, ax=ax, shrink=0.80,
                 label=LABEL_JEFFREYS)

    thresh = J_mat.max() * 0.55
    for i in range(n_conds):
        for j in range(n_conds):
            txt_color = "white" if J_mat[i, j] > thresh else "#333333"
            ax.text(j, i, f"{J_mat[i, j]:.3f}",
                    ha="center", va="center", fontsize=9, color=txt_color)

    ax.set_xticks(range(n_conds))
    ax.set_yticks(range(n_conds))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Colour the tick labels to match condition colours
    for tick, col in zip(ax.get_xticklabels(), colors):
        tick.set_color(col)
    for tick, col in zip(ax.get_yticklabels(), colors):
        tick.set_color(col)

    ax.set_title(
        "Plot 21 — Jeffrey's Divergence Heatmap  (all pairs pooled)\n"
        f"{LABEL_JEFFREYS}  ·  symmetric  ·  diagonal = 0\n"
        "p* clustering with within-AND (small J) → gap closed;  "
        "clustering with mono → AND outside text manifold.",
        fontsize=11,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_21_jeffreys_heatmap.png")


# ===========================================================================
# Plot 22 — Expressiveness ladder: J(p* source, within-AND) vs method rank.
#
#   x-axis: p* sources in increasing language-expressiveness order
#           (CLIP inverter ≺ token opt. ≺ Zero2Text ≺ VLM caption)
#   y-axis: Jeffrey's divergence from within-AND distribution (↓ = gap closed)
#
#   Each line = one concept pair; bold black = pooled mean.
#   Horizontal dashed references: J(mono, within-AND), J(c₁, within-AND), etc.
#
#   Interpretation:
#     Downward trend → more expressive p* closes the gap — language can express AND.
#     Flat / plateau above reference lines → AND is outside the text manifold.
#     Crossing reference lines → that p* method fully accounts for the composition.
# ===========================================================================

def plot_22(df_term, df_traj, out_dir, data_dir=None, **kw):
    if not SCIPY_OK:
        print("  Skipping plot 22 (scipy required).")
        return
    if data_dir is None:
        print("  Skipping plot 22: data_dir required for within_and_distances.json.")
        return

    raw = load_within_and(data_dir)
    if raw is None:
        print("  Skipping plot 22: within_and_distances.json not found.")
        return

    present_pstar = get_present_pstar(df_term)
    if not present_pstar:
        print("  Skipping plot 22: no d_T_pstar_* columns found.")
        return

    df_within = pd.DataFrame(raw)
    pairs     = sorted(df_term["pair"].unique())
    pcmap     = pair_color_map(pairs)

    # Per-pair Jeffrey's divergence for each p* source and each baseline
    j_pstar = {cond: [] for cond in present_pstar}
    ref_conds = ["d_T_mono", "d_T_c1", "d_T_c2"] + get_present_poe(df_term)
    j_refs  = {c: [] for c in ref_conds}

    for pair in pairs:
        sub_term = df_term[df_term["pair"] == pair]

        # Within-AND values: prefer per-pair, fall back to pooled
        sub_w = df_within[df_within["pair"] == pair]
        if len(sub_w) < 3:
            sub_w = df_within
        within_pair_vals = sub_w["d_within_and"].values

        # Shared x-grid for this pair spanning all distributions
        all_v  = np.concatenate(
            [within_pair_vals]
            + [sub_term[c].values for c in present_pstar]
            + [sub_term[c].values for c in j_refs]
        )
        x_grid = np.linspace(max(all_v.min() * 0.70, 0),
                             all_v.max() * 1.10, 1000)

        pmf_w = kde_pmf(within_pair_vals, x_grid)

        for cond in present_pstar:
            j_pstar[cond].append(
                jeffreys_div(kde_pmf(sub_term[cond].values, x_grid), pmf_w))
        for cond in j_refs:
            j_refs[cond].append(
                jeffreys_div(kde_pmf(sub_term[cond].values, x_grid), pmf_w))

    # --- Build plot ---
    n_pstar = len(present_pstar)
    x_arr   = np.arange(n_pstar)

    fig, ax = plt.subplots(figsize=(max(8, n_pstar * 2.2), 5.5))

    # Per-pair lines (thin, pair-coloured)
    for pi, pair in enumerate(pairs):
        ys = [j_pstar[cond][pi] for cond in present_pstar]
        ax.plot(x_arr, ys, color=pcmap[pair], lw=1.6, alpha=0.65,
                marker="o", markersize=7, label=short_pair(pair))

    # Pooled mean (bold black)
    pooled_ys = [float(np.mean(j_pstar[cond])) for cond in present_pstar]
    ax.plot(x_arr, pooled_ys, color="black", lw=3.2, marker="D",
            markersize=10, label="Pooled mean", zorder=5)

    # Reference lines for baselines (dashed horizontals)
    ref_info = [
        ("d_T_mono", "Monolithic (ref)"),
        ("d_T_c1",   "Solo c₁  (ref)"),
        ("d_T_c2",   "Solo c₂  (ref)"),
    ]
    if "d_T_poe" in j_refs:
        ref_info.append(("d_T_poe", "PoE (ref)"))
    for cond, ref_lbl in ref_info:
        ref_mean = float(np.mean(j_refs[cond]))
        ax.axhline(ref_mean, color=TERM_COLOR[cond], lw=1.8, ls="--", alpha=0.75,
                   label=f"{ref_lbl}  J̄={ref_mean:.3f}", zorder=2)

    ax.set_xticks(x_arr)
    ax.set_xticklabels([TERM_LABEL[c] for c in present_pstar], fontsize=11)
    ax.set_ylabel(LABEL_J_SOURCE_WITHIN, fontsize=11)
    ax.set_xlim(-0.4, n_pstar - 0.6)

    ax.set_title(
        "Plot 22 — Expressiveness Ladder: Jeffrey's Divergence from Within-AND\n"
        "Each line = concept pair  ·  Bold = pooled mean  ·  "
        "Dashed horizontals = baseline divergences\n"
        "Downward slope → more expressive p* closes the gap.  "
        "Plateau above baselines → AND outside the text-conditioned latent manifold.",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.25)
    hide_top_right(ax)
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_22_expressiveness_ladder.png")


# ===========================================================================
# Plot 23 — p* vs SuperDiff-AND KL (p* sources only, no c1/c2 baselines shown)
#
#   We approximate the SuperDiff-AND distribution with within-AND pairwise
#   distances (same proxy used in Plot 11 / 20 / 22), then compute:
#       KL(P_AND || P_p*)   and   KL(P_p* || P_AND)
#   for each available p* source.
# ===========================================================================

def plot_23(df_term, df_traj, out_dir, data_dir=None, **kw):
    if not SCIPY_OK:
        print("  Skipping plot 23 (scipy required).")
        return
    if data_dir is None:
        print("  Skipping plot 23: data_dir required for within_and_distances.json.")
        return

    raw = load_within_and(data_dir)
    if raw is None:
        print("  Skipping plot 23: within_and_distances.json not found.")
        return

    present_pstar = get_present_pstar(df_term)
    if not present_pstar:
        print("  Skipping plot 23: no d_T_pstar_* columns found.")
        return

    within_vals = pd.DataFrame(raw)["d_within_and"].values
    all_vals = np.concatenate([within_vals] + [df_term[c].values for c in present_pstar])
    x_grid = np.linspace(max(all_vals.min() * 0.70, 0), all_vals.max() * 1.10, 1000)

    p_and = kde_pmf(within_vals, x_grid)

    kl_and_to_src = []
    kl_src_to_and = []
    labels = []
    colors = []
    for cond in present_pstar:
        p_src = kde_pmf(df_term[cond].values, x_grid)
        kl_and_to_src.append(float(stats.entropy(p_and, p_src)))
        kl_src_to_and.append(float(stats.entropy(p_src, p_and)))
        labels.append(TERM_LABEL[cond])
        colors.append(TERM_COLOR[cond])

    n = len(labels)
    x = np.arange(n)
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * n), 5.2))
    bars_a = ax.bar(x - width / 2, kl_and_to_src, width,
                    color=colors, alpha=0.86, edgecolor="white",
                    label=r"$D_{\mathrm{KL}}(P_{\mathrm{AND}}\|P_{p^*})$")
    bars_b = ax.bar(x + width / 2, kl_src_to_and, width,
                    color=colors, alpha=0.45, edgecolor="white", hatch="//",
                    label=r"$D_{\mathrm{KL}}(P_{p^*}\|P_{\mathrm{AND}})$")

    for b in list(bars_a) + list(bars_b):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.01 * max(1e-6, max(kl_and_to_src + kl_src_to_and)),
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(r"$D_{\mathrm{KL}}(P\|Q)$", fontsize=11)
    ax.set_title(
        "Plot 23 — p* vs SuperDiff-AND KL Divergence  (p* sources only)\n"
        "AND proxy = within-AND pairwise distance distribution.  Lower = closer.",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")
    hide_top_right(ax)
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_23_kl_pstar_vs_and.png")


# ===========================================================================
# Plot 24 — pointwise divergence contributions.
#
#   For each available p* source and shared x-grid:
#       k_AND->p*(x_i) = [p_AND(i) log(p_AND(i)/p_p*(i))] / Δx
#       k_p*->AND(x_i) = [p_p*(i) log(p_p*(i)/p_AND(i))] / Δx
#       j(x_i)         = [(p_AND(i)-p_p*(i)) log(p_AND(i)/p_p*(i))] / Δx
#
#   Since p's are PMFs on a uniform grid (sum=1), the shaded areas satisfy:
#       ∫ k_AND->p* dx = KL(P_AND || P_p*)
#       ∫ k_p*->AND dx = KL(P_p* || P_AND)
#       ∫ j dx         = J(P_AND, P_p*)
# ===========================================================================

def _kl_contrib_density(p, q, dx):
    contrib = p * np.log(p / q)
    return contrib / dx, float(np.sum(contrib))


def _jeffreys_contrib_density(p, q, dx):
    contrib = (p - q) * np.log(p / q)
    return contrib / dx, float(np.sum(contrib))


def _plot_signed_fill(ax, x_grid, y_vals, color):
    ax.fill_between(x_grid, 0.0, y_vals, where=(y_vals >= 0),
                    color=color, alpha=0.30, linewidth=0)
    ax.fill_between(x_grid, 0.0, y_vals, where=(y_vals < 0),
                    color="#666666", alpha=0.18, linewidth=0)
    ax.plot(x_grid, y_vals, color=color, lw=2.0)
    ax.axhline(0.0, color="#666666", lw=0.9, alpha=0.8)


def _overlay_faint_kdes(ax, x_grid, p_and, p_src, dx, and_color, src_color):
    # Convert PMFs back to grid-density curves so the overlay is shape-faithful.
    d_and = p_and / dx
    d_src = p_src / dx
    ax.plot(x_grid, d_and, ls="--", lw=1.3, color=and_color, alpha=0.42, zorder=1)
    ax.plot(x_grid, d_src, ls="--", lw=1.3, color=src_color, alpha=0.42, zorder=1)


def plot_29(df_term, df_traj, out_dir, data_dir=None, **kw):
    """Scaffold 1/5: pooled raw sample strip (within-AND + p* sources)."""
    prepared = _prepare_pointwise_inputs(df_term, data_dir, n_grid=600)
    if prepared is None:
        print("  Skipping plot 29.")
        return
    within_vals, present_pstar, _, _, _, _, src_vals = prepared

    labels = ["Within-AND"] + [TERM_LABEL[c] for c in present_pstar]
    colors = ["#1F77B4"] + [TERM_COLOR[c] for c in present_pstar]
    all_series = [within_vals] + [src_vals[c] for c in present_pstar]

    rng = np.random.default_rng(42)
    y_pos = np.arange(len(labels))
    fig_h = max(4.6, 1.2 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(11.0, fig_h))

    for yi, vals, color in zip(y_pos, all_series, colors):
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(vals, yi + jitter, s=24, color=color, alpha=0.28,
                   edgecolors="white", linewidths=0.5, zorder=2)
        ax.plot([vals.mean(), vals.mean()], [yi - 0.27, yi + 0.27],
                color=color, lw=3.2, zorder=3)
        ax.text(vals.mean(), yi + 0.34, f"{vals.mean():.3f}",
                color=color, fontsize=9, ha="center", va="bottom")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(r"$d_T$  (terminal distance to SuperDiff-AND)", fontsize=11)
    ax.set_title(
        "Plot 29 — Pointwise Scaffold (1/5): Raw pooled $d_T$ samples\n"
        "Dots = per-seed observations.  Vertical ticks = sample means.\n"
        "This is the data that later becomes histograms/KDEs for KL.",
        fontsize=11,
    )
    ax.grid(alpha=0.22)
    hide_top_right(ax)
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_29_pointwise_scaffold_raw_samples.png")


def plot_30(df_term, df_traj, out_dir, data_dir=None, **kw):
    """Scaffold 2/5: shared-bin normalized histograms (PMF intuition)."""
    prepared = _prepare_pointwise_inputs(df_term, data_dir, n_grid=600)
    if prepared is None:
        print("  Skipping plot 30.")
        return
    within_vals, present_pstar, x_grid, _, _, _, src_vals = prepared

    n_rows = len(present_pstar)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(12.0, max(3.2, 2.8 * n_rows)),
        sharex=True,
        squeeze=False,
    )
    bins = np.linspace(x_grid.min(), x_grid.max(), 46)

    for ridx, cond in enumerate(present_pstar):
        ax = axes[ridx][0]
        src_label = TERM_LABEL[cond]
        src_color = TERM_COLOR[cond]

        ax.hist(
            within_vals, bins=bins, density=True,
            color="#1F77B4", alpha=0.30, edgecolor="white", linewidth=0.4,
            label=r"Within-AND proxy  $P_{\mathrm{AND}}$",
        )
        ax.hist(
            src_vals[cond], bins=bins, density=True,
            color=src_color, alpha=0.30, edgecolor="white", linewidth=0.4,
            label=rf"{src_label}  $P_{{p^*}}$",
        )
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(
            f"{src_label} — shared-bin histogram (before KDE smoothing)",
            fontsize=10,
        )
        ax.grid(alpha=0.20)
        hide_top_right(ax)
        ax.legend(frameon=False, fontsize=9, loc="upper right")

    axes[-1][0].set_xlabel(r"$d_T$", fontsize=11)
    fig.suptitle(
        "Plot 30 — Pointwise Scaffold (2/5): Shared-bin histograms\n"
        "Use the same $d_T$ bins for both distributions so local mass is directly comparable.",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_fig(fig, out_dir / "plot_30_pointwise_scaffold_hist_shared_bins.png")


def _shared_bin_kl_terms(p_vals, q_vals, bins, eps=1e-12):
    """Discrete shared-bin KL ingredients used for the bridge scaffold."""
    c_p, _ = np.histogram(p_vals, bins=bins)
    c_q, _ = np.histogram(q_vals, bins=bins)

    p = c_p.astype(float)
    q = c_q.astype(float)
    if p.sum() <= 0 or q.sum() <= 0:
        return None

    p /= p.sum()
    q /= q.sum()

    log_pq = np.log((p + eps) / (q + eps))
    log_qp = -log_pq
    k_and_src = p * log_pq
    k_src_and = q * log_qp
    return p, q, log_pq, log_qp, k_and_src, k_src_and


def plot_34(df_term, df_traj, out_dir, data_dir=None, **kw):
    """Scaffold bridge: shared-bin mass -> log-ratio -> local KL -> cumulative KL."""
    prepared = _prepare_pointwise_inputs(df_term, data_dir, n_grid=600)
    if prepared is None:
        print("  Skipping plot 34.")
        return
    within_vals, present_pstar, x_grid, _, _, _, src_vals = prepared

    n_rows = len(present_pstar)
    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=(18.5, max(4.2, 3.5 * n_rows)),
        sharex=True,
        squeeze=False,
    )

    n_bins = 24
    bins = np.linspace(x_grid.min(), x_grid.max(), n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bar_w = (bins[1] - bins[0]) * 0.42

    for ridx, cond in enumerate(present_pstar):
        src_label = TERM_LABEL[cond]
        src_color = TERM_COLOR[cond]
        terms = _shared_bin_kl_terms(within_vals, src_vals[cond], bins)
        if terms is None:
            continue
        p_bin, q_bin, log_pq, log_qp, k_and_src, k_src_and = terms

        mass_and_label = r"$p_i$: AND bin mass"
        mass_src_label = rf"$q_i$: {src_label} bin mass"
        log_fwd_label = r"$\log(p_i/q_i)$"
        log_rev_label = r"$\log(q_i/p_i)$"
        k_fwd_label = rf"$k_{{\mathrm{{AND}}\to p^*}}(i)$ (sum={k_and_src.sum():.3f})"
        k_rev_label = rf"$k_{{p^*\to \mathrm{{AND}}}}(i)$ (sum={k_src_and.sum():.3f})"

        ax0, ax1, ax2, ax3 = axes[ridx]

        # Stage 1: same-bin masses.
        ax0.bar(centers - bar_w / 2, p_bin, width=bar_w, color="#1F77B4",
                alpha=0.82, edgecolor="white", linewidth=0.3, label=mass_and_label)
        ax0.bar(centers + bar_w / 2, q_bin, width=bar_w, color=src_color,
                alpha=0.70, edgecolor="white", linewidth=0.3, label=mass_src_label)
        ax0.set_ylabel("Mass per bin", fontsize=9.6)
        ax0.set_title(f"{src_label}: shared-bin masses", fontsize=9.8)

        # Stage 2: ratio-only term (why spikes happen when a denominator is tiny).
        log_fwd_colors = [src_color if v >= 0 else "#666666" for v in log_pq]
        log_rev_colors = [src_color if v >= 0 else "#A0A0A0" for v in log_qp]
        ax1.bar(centers - bar_w / 2, log_pq, width=bar_w, color=log_fwd_colors,
                alpha=0.84, edgecolor="white", linewidth=0.3, label=log_fwd_label)
        ax1.bar(centers + bar_w / 2, log_qp, width=bar_w, color=log_rev_colors,
                alpha=0.58, edgecolor="white", linewidth=0.3, label=log_rev_label)
        ax1.axhline(0.0, color="#666666", lw=0.9, alpha=0.9)
        ax1.set_ylabel("Log-ratio term", fontsize=9.6)
        ax1.set_title(f"{src_label}: ratio-only term", fontsize=9.8)

        # Stage 3: weighted local KL terms.
        k_fwd_colors = [src_color if v >= 0 else "#666666" for v in k_and_src]
        k_rev_colors = [src_color if v >= 0 else "#A0A0A0" for v in k_src_and]
        ax2.bar(centers - bar_w / 2, k_and_src, width=bar_w, color=k_fwd_colors,
                alpha=0.86, edgecolor="white", linewidth=0.3, label=k_fwd_label)
        ax2.bar(centers + bar_w / 2, k_src_and, width=bar_w, color=k_rev_colors,
                alpha=0.62, edgecolor="white", linewidth=0.3, label=k_rev_label)
        ax2.axhline(0.0, color="#666666", lw=0.9, alpha=0.9)
        ax2.set_ylabel("Local KL term", fontsize=9.6)
        ax2.set_title(f"{src_label}: weighted local KL", fontsize=9.8)

        # Stage 4: running discrete integrals over bins.
        csum_fwd = np.cumsum(k_and_src)
        csum_rev = np.cumsum(k_src_and)
        ax3.plot(centers, csum_fwd, color=src_color, lw=2.0,
                 label=rf"$\sum_{{j \leq i}} k_{{\mathrm{{AND}}\to p^*}}(j)$")
        ax3.plot(centers, csum_rev, color=src_color, lw=1.9, ls="--", alpha=0.92,
                 label=rf"$\sum_{{j \leq i}} k_{{p^*\to \mathrm{{AND}}}}(j)$")
        ax3.axhline(csum_fwd[-1], color=src_color, lw=0.85, alpha=0.35)
        ax3.axhline(csum_rev[-1], color=src_color, lw=0.85, alpha=0.35, ls="--")
        ax3.text(centers[-1], csum_fwd[-1], f"  {csum_fwd[-1]:.3f}",
                 color=src_color, fontsize=8.6, va="center")
        ax3.text(centers[-1], csum_rev[-1], f"  {csum_rev[-1]:.3f}",
                 color=src_color, fontsize=8.6, va="center")
        ax3.set_ylabel("Running KL", fontsize=9.6)
        ax3.set_title(f"{src_label}: cumulative sum over bins", fontsize=9.8)

        for ax in (ax0, ax1, ax2, ax3):
            ax.grid(alpha=0.18)
            hide_top_right(ax)
            ax.legend(frameon=False, fontsize=7.6, loc="upper right")

    for cidx in range(4):
        axes[-1][cidx].set_xlabel(r"$d_T$ (shared bins)", fontsize=10)

    fig.suptitle(
        "Plot 34 — Pointwise Scaffold (2.5/5): Shared-bin KL factorization bridge\n"
        r"$p_i,q_i$  ->  $\log(\cdot)$ ratio  ->  weighted local KL terms  ->  cumulative KL totals.",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_fig(fig, out_dir / "plot_34_pointwise_scaffold_shared_bin_factorization.png")


def plot_31(df_term, df_traj, out_dir, data_dir=None, **kw):
    """Scaffold 3/5: KDE overlays + signed density difference."""
    prepared = _prepare_pointwise_inputs(df_term, data_dir, n_grid=1200)
    if prepared is None:
        print("  Skipping plot 31.")
        return
    _, present_pstar, x_grid, dx, p_and, src_pmfs, _ = prepared

    n_rows = len(present_pstar)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(12.0, max(3.4, 3.0 * n_rows)),
        sharex=True,
        squeeze=False,
    )
    d_and = p_and / dx

    for ridx, cond in enumerate(present_pstar):
        ax = axes[ridx][0]
        src_color = TERM_COLOR[cond]
        src_label = TERM_LABEL[cond]
        d_src = src_pmfs[cond] / dx
        diff = d_and - d_src

        ax.plot(x_grid, d_and, color="#1F77B4", lw=2.0, ls="--",
                label=r"KDE $P_{\mathrm{AND}}$")
        ax.plot(x_grid, d_src, color=src_color, lw=2.0, ls="--",
                label=rf"KDE {src_label}")
        ax.fill_between(x_grid, 0.0, diff, where=(diff >= 0), color="#1F77B4",
                        alpha=0.14, linewidth=0)
        ax.fill_between(x_grid, 0.0, diff, where=(diff < 0), color=src_color,
                        alpha=0.14, linewidth=0)
        ax.axhline(0.0, color="#777777", lw=0.9, alpha=0.75)

        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(
            f"{src_label} — dashed KDEs + signed density difference "
            r"($P_{\mathrm{AND}} - P_{p^*}$)",
            fontsize=10,
        )
        ax.grid(alpha=0.20)
        hide_top_right(ax)
        ax.legend(frameon=False, fontsize=9, loc="upper right")

    axes[-1][0].set_xlabel(r"$d_T$", fontsize=11)
    fig.suptitle(
        "Plot 31 — Pointwise Scaffold (3/5): KDE overlays and where one dominates\n"
        "Blue-shaded regions: AND has more local mass.  Source-shaded regions: p* has more local mass.",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_fig(fig, out_dir / "plot_31_pointwise_scaffold_kde_overlay_diff.png")


def plot_32(df_term, df_traj, out_dir, data_dir=None, **kw):
    """Scaffold 4/5: local (pointwise) KL terms before final styled fill plot."""
    prepared = _prepare_pointwise_inputs(df_term, data_dir, n_grid=1200)
    if prepared is None:
        print("  Skipping plot 32.")
        return
    _, present_pstar, x_grid, dx, p_and, src_pmfs, _ = prepared

    n_rows = len(present_pstar)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(14.0, max(3.8, 3.3 * n_rows)),
        sharex=True,
        squeeze=False,
    )

    for ridx, cond in enumerate(present_pstar):
        p_src = src_pmfs[cond]
        src_color = TERM_COLOR[cond]
        src_label = TERM_LABEL[cond]

        k_and_src_y, kl_and_src = _kl_contrib_density(p_and, p_src, dx)
        k_src_and_y, kl_src_and = _kl_contrib_density(p_src, p_and, dx)

        ax0, ax1 = axes[ridx]
        _plot_signed_fill(ax0, x_grid, k_and_src_y, src_color)
        ax0.set_title(
            src_label + "\n" + rf"$k_{{\mathrm{{AND}}\rightarrow p^*}}(x)$;  integral = {kl_and_src:.3f}",
            fontsize=10,
        )
        ax0.set_ylabel("Local KL density", fontsize=10)

        _plot_signed_fill(ax1, x_grid, k_src_and_y, src_color)
        ax1.set_title(
            src_label + "\n" + rf"$k_{{p^*\rightarrow \mathrm{{AND}}}}(x)$;  integral = {kl_src_and:.3f}",
            fontsize=10,
        )

        for ax in (ax0, ax1):
            ax.grid(alpha=0.20)
            hide_top_right(ax)

    axes[-1][0].set_xlabel(r"$d_T$", fontsize=11)
    axes[-1][1].set_xlabel(r"$d_T$", fontsize=11)
    fig.suptitle(
        "Plot 32 — Pointwise Scaffold (4/5): Local signed KL contribution densities\n"
        "Same formulas as Plot 24, shown without Jeffrey's column.",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_fig(fig, out_dir / "plot_32_pointwise_scaffold_local_kl_terms.png")


def plot_33(df_term, df_traj, out_dir, data_dir=None, **kw):
    """Scaffold 5/5: running KL integrals over d_T (how area accumulates)."""
    prepared = _prepare_pointwise_inputs(df_term, data_dir, n_grid=1200)
    if prepared is None:
        print("  Skipping plot 33.")
        return
    _, present_pstar, x_grid, _, p_and, src_pmfs, _ = prepared

    n_rows = len(present_pstar)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(12.0, max(3.2, 2.9 * n_rows)),
        sharex=True,
        squeeze=False,
    )

    for ridx, cond in enumerate(present_pstar):
        ax = axes[ridx][0]
        p_src = src_pmfs[cond]
        src_color = TERM_COLOR[cond]
        src_label = TERM_LABEL[cond]

        # PMFs on a shared grid: cumulative sums are running discrete integrals.
        local_fwd = p_and * np.log(p_and / p_src)
        local_rev = p_src * np.log(p_src / p_and)
        csum_fwd = np.cumsum(local_fwd)
        csum_rev = np.cumsum(local_rev)

        ax.plot(
            x_grid,
            csum_fwd,
            color=src_color,
            lw=2.1,
            label=(
                r"$\sum_{x_i \leq x} p_{\mathrm{AND}}"
                r"\log\!\left(\frac{p_{\mathrm{AND}}}{p_{p^\ast}}\right)$"
            ),
        )
        ax.plot(
            x_grid,
            csum_rev,
            color=src_color,
            lw=2.0,
            ls="--",
            alpha=0.95,
            label=(
                r"$\sum_{x_i \leq x} p_{p^\ast}"
                r"\log\!\left(\frac{p_{p^\ast}}{p_{\mathrm{AND}}}\right)$"
            ),
        )
        ax.axhline(csum_fwd[-1], color=src_color, lw=0.9, alpha=0.35)
        ax.axhline(csum_rev[-1], color=src_color, lw=0.9, alpha=0.35, ls="--")
        ax.text(x_grid[-1], csum_fwd[-1], f"  {csum_fwd[-1]:.3f}", color=src_color,
                va="center", fontsize=9)
        ax.text(x_grid[-1], csum_rev[-1], f"  {csum_rev[-1]:.3f}", color=src_color,
                va="center", fontsize=9)

        ax.set_ylabel("Running KL", fontsize=10)
        ax.set_title(
            f"{src_label} — cumulative area up to each $d_T$",
            fontsize=10,
        )
        ax.grid(alpha=0.20)
        hide_top_right(ax)
        ax.legend(frameon=False, fontsize=8.8, loc="upper left")

    axes[-1][0].set_xlabel(r"$d_T$", fontsize=11)
    fig.suptitle(
        "Plot 33 — Pointwise Scaffold (5/5): Running KL integral\n"
        "End-point values match Plot 23 bars and Plot 24 panel integrals.",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_fig(fig, out_dir / "plot_33_pointwise_scaffold_cumulative_kl.png")


def plot_24(df_term, df_traj, out_dir, data_dir=None, **kw):
    prepared = _prepare_pointwise_inputs(df_term, data_dir, n_grid=1200)
    if prepared is None:
        print("  Skipping plot 24.")
        return

    _, present_pstar, x_grid, dx, p_and, src_pmfs, _ = prepared
    and_overlay_color = "#1F77B4"

    n_rows = len(present_pstar)
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(15.5, max(4.0, 3.5 * n_rows)),
        sharex=True,
        squeeze=False,
    )

    for ridx, cond in enumerate(present_pstar):
        p_src = src_pmfs[cond]
        color = TERM_COLOR[cond]
        src_label = TERM_LABEL[cond]

        k_and_src_y, kl_and_src = _kl_contrib_density(p_and, p_src, dx)
        k_src_and_y, kl_src_and = _kl_contrib_density(p_src, p_and, dx)
        j_y, j_val = _jeffreys_contrib_density(p_and, p_src, dx)

        ax0, ax1, ax2 = axes[ridx]

        _plot_signed_fill(ax0, x_grid, k_and_src_y, color)
        _overlay_faint_kdes(ax0, x_grid, p_and, p_src, dx, and_overlay_color, color)
        ax0.set_title(
            src_label + "\n" + rf"$\int k_{{\mathrm{{AND}}\rightarrow p^*}}(x)\,dx={kl_and_src:.3f}$",
            fontsize=10,
        )
        ax0.set_ylabel("Contribution density", fontsize=10)

        _plot_signed_fill(ax1, x_grid, k_src_and_y, color)
        _overlay_faint_kdes(ax1, x_grid, p_and, p_src, dx, and_overlay_color, color)
        ax1.set_title(
            src_label + "\n" + rf"$\int k_{{p^*\rightarrow \mathrm{{AND}}}}(x)\,dx={kl_src_and:.3f}$",
            fontsize=10,
        )

        ax2.fill_between(x_grid, 0.0, j_y, color=color, alpha=0.32, linewidth=0)
        ax2.plot(x_grid, j_y, color=color, lw=2.0)
        ax2.axhline(0.0, color="#666666", lw=0.9, alpha=0.8)
        _overlay_faint_kdes(ax2, x_grid, p_and, p_src, dx, and_overlay_color, color)
        ax2.set_title(
            src_label + "\n"
            + rf"$\int j(x)\,dx=J(P_{{\mathrm{{AND}}}},P_{{p^*}})={j_val:.3f}$",
            fontsize=10,
        )

        for ax in (ax0, ax1, ax2):
            ax.grid(alpha=0.20)
            hide_top_right(ax)

    for cidx in range(3):
        axes[-1][cidx].set_xlabel(r"$d_T$", fontsize=11)

    fig.suptitle(
        "Plot 24 — Pointwise Divergence Contributions vs SuperDiff-AND",
        fontsize=12,
        y=0.992,
    )
    legend_handles = [
        Line2D([0], [0], color=and_overlay_color, lw=1.3, ls="--", alpha=0.55,
               label=r"Overlay KDE $P_{\mathrm{AND}}$"),
    ]
    for cond in present_pstar:
        legend_handles.append(
            Line2D([0], [0], color=TERM_COLOR[cond], lw=1.3, ls="--", alpha=0.55,
                   label=rf"Overlay KDE {TERM_LABEL[cond]}")
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        frameon=False,
        fontsize=9,
        ncol=min(3, len(legend_handles)),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))

    # Column headers: keep method-level context (KL vs Jeffrey) without
    # crowding each subplot title.
    col_titles = [
        r"KL divergence  $D_{\mathrm{KL}}(P_{\mathrm{AND}}\|P_{p^*})$",
        r"KL divergence  $D_{\mathrm{KL}}(P_{p^*}\|P_{\mathrm{AND}})$",
        r"Jeffrey's divergence  $J(P_{\mathrm{AND}},P_{p^*})$",
    ]
    for cidx, title in enumerate(col_titles):
        bbox = axes[0][cidx].get_position()
        x_mid = 0.5 * (bbox.x0 + bbox.x1)
        y_pos = bbox.y1 + 0.085
        fig.text(x_mid, y_pos, title, ha="center", va="bottom",
                 fontsize=10.5, fontweight="bold")

    save_fig(fig, out_dir / "plot_24_pointwise_kl_jeffreys.png")
