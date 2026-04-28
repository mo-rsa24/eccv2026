"""Group-wise taxonomy figures for the paper-facing reporting bundle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from taxonomy_manifest import GROUP_LABEL_BY_KEY, GROUP_ORDER
except ImportError:
    from scripts.taxonomy_manifest import GROUP_LABEL_BY_KEY, GROUP_ORDER

from .baseline import _traj_conditions
from .utils import (
    active_logical_anchor_label,
    BIN_ALPHAS,
    BIN_LABELS,
    LABEL_CUM_DELTA_D_t,
    LABEL_D_T_MSE,
    LABEL_ECDF,
    LABEL_JEFFREYS,
    SCIPY_OK,
    TERM_COLOR,
    TERM_CONDITIONS,
    TERM_LABEL,
    TRAJ_COLOR,
    TRAJ_LABEL,
    bin_increments,
    ecdf_xy,
    get_present_poe,
    get_present_pstar,
    hide_top_right,
    kde_pmf,
    filter_semantic_baseline_scope,
    load_within_and,
    save_fig,
    stats,
    within_anchor_column,
    jeffreys_div,
)


def _ordered_groups(df: pd.DataFrame) -> list[tuple[str, str]]:
    if "taxonomy_group_key" not in df.columns:
        return []
    present = set(df["taxonomy_group_key"].dropna().tolist())
    return [(key, GROUP_LABEL_BY_KEY[key]) for key in GROUP_ORDER if key in present]


def _panel_grid(n_panels: int, figsize: tuple[float, float] | None = None):
    n_panels = max(1, int(n_panels))
    if n_panels <= 2:
        n_rows, n_cols = 1, n_panels
    elif n_panels <= 4:
        n_rows, n_cols = 2, 2
    else:
        n_rows, n_cols = int(np.ceil(n_panels / 2)), 2

    if figsize is None:
        figsize = (16, max(5.5, 5.2 * n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    return fig, axes, n_rows, n_cols


def _set_row_ylabel(axes, n_rows: int, n_cols: int, label: str, fontsize: int = 10) -> None:
    for row_idx in range(n_rows):
        ax_idx = row_idx * n_cols
        if ax_idx < len(axes):
            axes[ax_idx].set_ylabel(label, fontsize=fontsize)


def _tight_layout_rect(include_legend: bool = False) -> tuple[float, float, float, float]:
    return (0.0, 0.0, 0.97, 0.95) if include_legend else (0.0, 0.0, 1.0, 0.95)


def _blank_panel(ax, title: str, message: str = "No data") -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, transform=ax.transAxes)


def _resolve_within_df(data_dir=None, within_and_records=None) -> pd.DataFrame | None:
    records = within_and_records
    if records is None and data_dir is not None:
        records = load_within_and(data_dir)
    if not records:
        return None
    return pd.DataFrame(records)


def _load_joint_image_scores(data_dir) -> pd.DataFrame | None:
    if data_dir is None:
        return None
    try:
        from .joint_probe_bar import load_joint_probes
    except ImportError:
        from joint_probe_bar import load_joint_probes

    try:
        _, image_df = load_joint_probes(data_dir)
    except FileNotFoundError:
        return None
    if image_df.empty:
        return None
    return image_df


def plot_06_groupwise(df_term, df_traj, out_dir, **kw):
    groups = _ordered_groups(df_traj)
    if not groups:
        print("  Skipping groupwise plot 06: taxonomy metadata is missing.")
        return

    conditions = _traj_conditions(df_traj, kw.get("pstar_filter"))
    anchor_label = active_logical_anchor_label(df_traj, fallback="logical anchor")
    fig, axes, n_rows, n_cols = _panel_grid(len(groups), figsize=(16, max(8.5, 4.8 * int(np.ceil(len(groups) / 2)))))

    totals = []
    for group_key, _ in groups:
        sub = df_traj[df_traj["taxonomy_group_key"] == group_key]
        for cond in conditions:
            totals.append(sum(bin_increments(sub, cond)))
    y_top = max(max(totals) * 1.15, 0.6) if totals else 1.0

    for ax, (group_key, group_label) in zip(axes, groups):
        sub = df_traj[df_traj["taxonomy_group_key"] == group_key]
        if sub.empty:
            _blank_panel(ax, group_label)
            continue

        x_pos = {c: float(i) for i, c in enumerate(conditions)}
        for cond in conditions:
            x = x_pos[cond]
            color = TRAJ_COLOR[cond]
            increments = bin_increments(sub, cond)
            bottom = 0.0
            for bi, height in enumerate(increments):
                ax.bar(
                    x,
                    height,
                    0.52,
                    bottom=bottom,
                    color=color,
                    alpha=BIN_ALPHAS[bi],
                    edgecolor="white",
                    linewidth=0.8,
                )
                bottom += height
            ax.text(
                x,
                bottom + y_top * 0.015,
                f"{bottom:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=color,
                fontweight="bold",
            )

        ax.set_title(group_label, fontsize=11, fontweight="bold")
        ax.set_xticks([x_pos[c] for c in conditions])
        ax.set_xticklabels(
            [TRAJ_LABEL[c] for c in conditions],
            fontsize=9,
            rotation=25 if len(conditions) > 3 else 0,
            ha="right" if len(conditions) > 3 else "center",
        )
        ax.set_xlim(-0.6, len(conditions) - 0.4)
        ax.set_ylim(bottom=0, top=y_top)
        ax.grid(axis="y", alpha=0.25)
        hide_top_right(ax)

    for ax in axes[len(groups):]:
        ax.axis("off")

    _set_row_ylabel(axes, n_rows, n_cols, LABEL_CUM_DELTA_D_t, fontsize=10)
    bin_handles = [
        Patch(facecolor="#777777", alpha=BIN_ALPHAS[bi], edgecolor="white", label=f"Steps {BIN_LABELS[bi]}")
        for bi in range(len(BIN_LABELS))
    ]
    fig.legend(handles=bin_handles, title="Time bin", fontsize=9, title_fontsize=10, loc="upper right")
    fig.suptitle(
        f"Plot 06 — Temporal Divergence from the {anchor_label} Logical Anchor by Taxonomy Group\n"
        "Opaque = early denoising steps; transparent = late.",
        fontsize=13,
    )
    fig.tight_layout(rect=_tight_layout_rect(include_legend=True))
    save_fig(fig, out_dir / "plot_06_stacked_bar_groupwise.png")


def plot_11_groupwise(df_term, df_traj, out_dir, data_dir=None, **kw):
    within_df = _resolve_within_df(data_dir=data_dir, within_and_records=kw.get("within_and_records"))
    if within_df is None or within_df.empty:
        print("  Skipping groupwise plot 11: within_and_distances.json not found.")
        return

    groups = _ordered_groups(df_term)
    if not groups:
        print("  Skipping groupwise plot 11: taxonomy metadata is missing.")
        return

    gap_conds = list(TERM_CONDITIONS) + get_present_poe(df_term)
    all_vals = []
    for group_key, _group_label in groups:
        sub = df_term[df_term["taxonomy_group_key"] == group_key]
        if not sub.empty:
            all_vals.extend(sub[gap_conds].to_numpy().reshape(-1).tolist())
    all_vals.extend(within_df["d_within_and"].tolist())
    y_top = max(max(all_vals) * 1.2, 0.35) if all_vals else 1.0

    fig, axes, n_rows, n_cols = _panel_grid(len(groups), figsize=(17, max(8.8, 4.9 * int(np.ceil(len(groups) / 2)))))
    rng = np.random.default_rng(42)
    anchor_label = active_logical_anchor_label(df_term, fallback="logical anchor")
    within_col = within_anchor_column(within_df)
    within_color = "#999999"

    for ax, (group_key, group_label) in zip(axes, groups):
        sub = df_term[df_term["taxonomy_group_key"] == group_key]
        within_vals = within_df[within_df["taxonomy_group_key"] == group_key][within_col].values
        if sub.empty or len(within_vals) == 0:
            _blank_panel(ax, group_label)
            continue

        x = 0.0
        jitter = rng.uniform(-0.18, 0.18, len(within_vals))
        ax.scatter(x + jitter, within_vals, color=within_color, s=34, alpha=0.48, linewidths=0.7, edgecolors="white")
        ax.plot([x - 0.26, x + 0.26], [within_vals.mean(), within_vals.mean()], color=within_color, lw=4.5)
        w_mean = within_vals.mean()
        w_std = within_vals.std()
        ax.axhspan(w_mean - w_std, w_mean + w_std, alpha=0.07, color=within_color, zorder=1)
        ax.axhline(w_mean, color=within_color, lw=1.2, ls="--", alpha=0.55, zorder=2)

        for xi, cond in enumerate(gap_conds, start=1):
            vals = sub[cond].values
            jitter = rng.uniform(-0.18, 0.18, len(vals))
            color = TERM_COLOR[cond]
            ax.scatter(xi + jitter, vals, color=color, s=40, alpha=0.52, linewidths=0.8, edgecolors="white", zorder=3)
            ax.plot([xi - 0.28, xi + 0.28], [vals.mean(), vals.mean()], color=color, lw=4.5, zorder=4)
            ratio = vals.mean() / w_mean if w_mean > 0 else float("nan")
            ax.text(
                xi,
                vals.mean() + y_top * 0.02,
                f"{vals.mean():.3f}\n×{ratio:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                fontweight="bold",
            )

        ax.set_title(group_label, fontsize=11, fontweight="bold")
        labels = ["within-anchor"] + [TERM_LABEL[c] for c in gap_conds]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_ylim(bottom=0, top=y_top)
        ax.grid(axis="y", alpha=0.25)
        hide_top_right(ax)

    for ax in axes[len(groups):]:
        ax.axis("off")

    _set_row_ylabel(axes, n_rows, n_cols, r"Per-element MSE  $d_T$ or $d_{\mathrm{within}}$", fontsize=10)
    fig.suptitle(
        "Plot 11 — Gap Validation by Taxonomy Group\n"
        f"Within-{anchor_label} noise floor versus per-condition distances.",
        fontsize=13,
    )
    fig.tight_layout(rect=_tight_layout_rect())
    save_fig(fig, out_dir / "plot_11_within_and_noise_floor_groupwise.png")


def plot_15_groupwise(df_term, df_traj, out_dir, **kw):
    if not SCIPY_OK:
        print("  Skipping groupwise plot 15 (scipy required).")
        return

    pstar_conds = get_present_pstar(df_term)
    if not pstar_conds:
        print("  Skipping groupwise plot 15: no d_T_pstar_* column found.")
        return

    groups = _ordered_groups(df_term)
    baseline_conds = TERM_CONDITIONS + get_present_poe(df_term)
    all_conds = pstar_conds + baseline_conds
    anchor_label = active_logical_anchor_label(df_term, fallback="logical anchor")
    all_vals = df_term[all_conds].to_numpy().reshape(-1)
    x_grid = np.linspace(max(all_vals.min() * 0.8, 0), all_vals.max() * 1.1, 700)

    fig, axes, _n_rows, _n_cols = _panel_grid(len(groups), figsize=(16, max(8.8, 4.9 * int(np.ceil(len(groups) / 2)))))
    for ax, (group_key, group_label) in zip(axes, groups):
        sub = df_term[df_term["taxonomy_group_key"] == group_key]
        if sub.empty:
            _blank_panel(ax, group_label)
            continue

        for cond in pstar_conds:
            vals = sub[cond].dropna().values
            if len(vals) < 2:
                continue
            dens = stats.gaussian_kde(vals, bw_method="scott")(x_grid)
            color = TERM_COLOR[cond]
            ax.plot(x_grid, dens, color=color, lw=2.3, ls="-", label=TERM_LABEL[cond])
            ax.fill_between(x_grid, dens, alpha=0.12, color=color)

        for cond in baseline_conds:
            vals = sub[cond].dropna().values
            if len(vals) < 2:
                continue
            dens = stats.gaussian_kde(vals, bw_method="scott")(x_grid)
            color = TERM_COLOR[cond]
            ax.plot(x_grid, dens, color=color, lw=1.9, ls="--", alpha=0.82, label=TERM_LABEL[cond])
            ax.fill_between(x_grid, dens, alpha=0.07, color=color)

        ax.set_title(group_label, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Terminal distance to {anchor_label} (MSE)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.grid(alpha=0.22)
        hide_top_right(ax)

    for ax in axes[len(groups):]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], color=TERM_COLOR[c], lw=2.3, ls="-" if c in pstar_conds else "--", label=TERM_LABEL[c])
        for c in all_conds
    ]
    fig.legend(handles=handles, title="Condition", fontsize=9, title_fontsize=10, loc="upper right")
    fig.suptitle(
        "Plot 15 — Terminal Distance Distribution Across Conditions by Taxonomy Group",
        fontsize=13,
    )
    fig.tight_layout(rect=_tight_layout_rect(include_legend=True))
    save_fig(fig, out_dir / "plot_15_pstar_kde_groupwise.png")


def plot_17_groupwise(df_term, df_traj, out_dir, **kw):
    pstar_conds = get_present_pstar(df_term)
    if not pstar_conds:
        print("  Skipping groupwise plot 17: no d_T_pstar_* column found.")
        return

    groups = _ordered_groups(df_term)
    baseline_conds = TERM_CONDITIONS + get_present_poe(df_term)
    all_conds = pstar_conds + baseline_conds
    anchor_label = active_logical_anchor_label(df_term, fallback="logical anchor")
    y_top = max(float(df_term[all_conds].max().max()) * 1.12, 0.60)
    x_pos = {c: float(i) for i, c in enumerate(all_conds)}
    rng = np.random.default_rng(42)

    fig, axes, n_rows, n_cols = _panel_grid(len(groups), figsize=(16, max(8.8, 4.9 * int(np.ceil(len(groups) / 2)))))
    for ax, (group_key, group_label) in zip(axes, groups):
        sub = df_term[df_term["taxonomy_group_key"] == group_key]
        if sub.empty:
            _blank_panel(ax, group_label)
            continue

        for cond in all_conds:
            vals = sub[cond].values
            jitter = rng.uniform(-0.18, 0.18, len(vals))
            color = TERM_COLOR[cond]
            x = x_pos[cond]
            ax.scatter(x + jitter, vals, color=color, s=42, alpha=0.55, linewidths=1.0, edgecolors="white", zorder=3)
            ax.plot([x - 0.28, x + 0.28], [vals.mean(), vals.mean()], color=color, lw=4.6, zorder=4)
            ax.text(x, vals.mean() + y_top * 0.025, f"{vals.mean():.3f}", ha="center", va="bottom", fontsize=8, color=color)

        ax.axvline(len(pstar_conds) - 0.5, color="#CCCCCC", lw=1.0, ls="--", zorder=1)
        ax.set_title(group_label, fontsize=11, fontweight="bold")
        ax.set_xticks([x_pos[c] for c in all_conds])
        ax.set_xticklabels([TERM_LABEL[c] for c in all_conds], fontsize=8.5, rotation=18, ha="right")
        ax.set_xlim(-0.6, len(all_conds) - 0.4)
        ax.set_ylim(bottom=0, top=y_top)
        ax.grid(axis="y", alpha=0.25)
        hide_top_right(ax)

    for ax in axes[len(groups):]:
        ax.axis("off")

    _set_row_ylabel(axes, n_rows, n_cols, f"Terminal distance to {anchor_label} (per-element MSE)", fontsize=10)
    fig.suptitle(f"Plot 17 — Terminal Distance to {anchor_label} by Condition and Taxonomy Group", fontsize=13)
    fig.tight_layout(rect=_tight_layout_rect())
    save_fig(fig, out_dir / "plot_17_pstar_strip_groupwise.png")


def plot_reachability_semantic_groupwise(df_term, df_traj, out_dir, data_dir=None, **kw):
    required_cols = ["d_T_pstar_sdipc"]
    if not all(col in df_term.columns for col in required_cols):
        print("  Skipping semantics-qualified reachability plot: d_T_pstar_sdipc is missing.")
        return

    image_df = _load_joint_image_scores(data_dir)
    if image_df is None:
        print("  Skipping semantics-qualified reachability plot: joint_probe_scores.json not found.")
        return

    groups = _ordered_groups(df_term)
    if not groups:
        print("  Skipping semantics-qualified reachability plot: taxonomy metadata is missing.")
        return

    anchor_label = active_logical_anchor_label(df_term, fallback="logical anchor")
    within_df = _resolve_within_df(data_dir=data_dir, within_and_records=kw.get("within_and_records"))
    within_col = within_anchor_column(within_df)

    mono_rows = image_df[image_df["condition"] == "mono"].copy()
    if "semantic_pass" in mono_rows.columns:
        mono_scores = mono_rows[["pair_slug", "seed", "semantic_pass", "high_confidence_pass"]].copy()
    elif "joint_correctness_score" in mono_rows.columns:
        mono_rows = mono_rows[["pair_slug", "seed", "joint_correctness_score"]].copy()
        mono_rows["semantic_pass"] = mono_rows["joint_correctness_score"] >= 0.60
        mono_rows["high_confidence_pass"] = mono_rows["joint_correctness_score"] >= 0.75
        mono_scores = mono_rows[["pair_slug", "seed", "semantic_pass", "high_confidence_pass"]]
    else:
        print("  Skipping semantics-qualified reachability plot: monolithic semantic pass flags are unavailable.")
        return
    mono_scores = mono_scores.rename(
        columns={
            "semantic_pass": "mono_semantic_pass",
            "high_confidence_pass": "mono_high_confidence_pass",
        }
    )

    filtered_term = filter_semantic_baseline_scope(
        df_term,
        data_dir=data_dir,
        scope=kw.get("semantic_baseline_scope", "pair_qualified"),
        mono_pass_threshold=kw.get("mono_pass_threshold", 0.75),
        mono_seed_gate=kw.get("mono_seed_gate", "semantic"),
    )
    merged = filtered_term.merge(mono_scores, on=["pair_slug", "seed"], how="left")
    has_mono_ref = "d_T_pstar_sdipc_to_mono" in merged.columns

    y_top = max(
        float(
            pd.concat(
                [
                    merged["d_T_pstar_sdipc"],
                    merged["d_T_pstar_sdipc_to_mono"] if has_mono_ref else merged["d_T_pstar_sdipc"],
                ]
            ).max()
        )
        * 1.15,
        0.60,
    )

    fig, axes, n_rows, n_cols = _panel_grid(len(groups), figsize=(17, max(8.8, 4.9 * int(np.ceil(len(groups) / 2)))))
    rng = np.random.default_rng(42)
    x_labels = [
        f"PoE p* → {anchor_label}\n(all seeds)",
        "PoE p* → logical anchor\n(Mono semantic-pass)",
    ]
    if has_mono_ref:
        x_labels.append("PoE p* → Mono\n(all seeds)")
    x_pos = {label: float(i) for i, label in enumerate(x_labels)}

    for ax, (group_key, group_label) in zip(axes, groups):
        sub = merged[merged["taxonomy_group_key"] == group_key]
        if sub.empty:
            _blank_panel(ax, group_label)
            continue

        vals_all = sub["d_T_pstar_sdipc"].dropna().values
        vals_pass = sub[sub["mono_semantic_pass"] == True]["d_T_pstar_sdipc"].dropna().values
        vals_mono = sub["d_T_pstar_sdipc_to_mono"].dropna().values if has_mono_ref else np.array([])

        series = [
            (x_labels[0], vals_all, TERM_COLOR["d_T_pstar_sdipc"]),
            (x_labels[1], vals_pass, TERM_COLOR["d_T_pstar_sdipc"]),
        ]
        if has_mono_ref:
            series.append((x_labels[2], vals_mono, "#6F4E7C"))
        for label, vals, color in series:
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.16, 0.16, len(vals))
            x = x_pos[label]
            ax.scatter(x + jitter, vals, color=color, s=40, alpha=0.52, linewidths=0.9, edgecolors="white", zorder=3)
            ax.plot([x - 0.24, x + 0.24], [vals.mean(), vals.mean()], color=color, lw=4.2, zorder=4)
            ax.text(x, vals.mean() + y_top * 0.022, f"{vals.mean():.3f}", ha="center", va="bottom", fontsize=8, color=color)

        if within_df is not None and not within_df.empty:
            within_vals = within_df[within_df["taxonomy_group_key"] == group_key][within_col].dropna().values
            if len(within_vals):
                w_mean = float(within_vals.mean())
                ax.axhline(w_mean, color="#777777", lw=1.2, ls="--", alpha=0.70, zorder=1)
                ax.axhspan(
                    np.percentile(within_vals, 5),
                    np.percentile(within_vals, 95),
                    color="#777777",
                    alpha=0.07,
                    zorder=0,
                )
                ax.text(2.35, w_mean + y_top * 0.015, f"within-{anchor_label} {w_mean:.3f}", fontsize=8, color="#666666")

        pass_rate = sub["mono_semantic_pass"].fillna(False).mean()
        ax.set_title(f"{group_label}\nMono semantic-pass rate: {pass_rate:.2f}", fontsize=11, fontweight="bold")
        ax.set_xticks([x_pos[label] for label in x_labels])
        ax.set_xticklabels(x_labels, fontsize=8.5)
        ax.set_xlim(-0.5, len(x_labels) - 0.2)
        ax.set_ylim(bottom=0, top=y_top)
        ax.grid(axis="y", alpha=0.25)
        hide_top_right(ax)

    for ax in axes[len(groups):]:
        ax.axis("off")

    _set_row_ylabel(axes, n_rows, n_cols, "Terminal distance (per-element MSE)", fontsize=10)
    subtitle = (
        f"PoE p* distance to {anchor_label}, mono semantic-pass subset, and PoE p* distance to Mono."
        if has_mono_ref
        else f"PoE p* distance to {anchor_label} and the mono semantic-pass subset."
    )
    fig.suptitle(
        f"Semantics-Qualified Reachability by Taxonomy Group\n{subtitle}",
        fontsize=13,
    )
    fig.tight_layout(rect=_tight_layout_rect())
    save_fig(fig, out_dir / "plot_35_reachability_semantic_groupwise.png")


def plot_20_groupwise(df_term, df_traj, out_dir, data_dir=None, **kw):
    if not SCIPY_OK:
        print("  Skipping groupwise plot 20 (scipy required).")
        return

    within_df = _resolve_within_df(data_dir=data_dir, within_and_records=kw.get("within_and_records"))
    groups = _ordered_groups(df_term)
    pstar_conds = get_present_pstar(df_term)
    baseline_conds = TERM_CONDITIONS + get_present_poe(df_term)
    anchor_label = active_logical_anchor_label(df_term, fallback="logical anchor")
    within_col = within_anchor_column(within_df)

    fig, axes, n_rows, n_cols = _panel_grid(len(groups), figsize=(16, max(8.8, 4.9 * int(np.ceil(len(groups) / 2)))))
    for ax, (group_key, group_label) in zip(axes, groups):
        sub = df_term[df_term["taxonomy_group_key"] == group_key]
        if sub.empty:
            _blank_panel(ax, group_label)
            continue

        if within_df is not None:
            within_vals = within_df[within_df["taxonomy_group_key"] == group_key][within_col].values
            if len(within_vals):
                wx, wy = ecdf_xy(within_vals)
                ax.step(wx, wy, color="#888888", lw=2.2, ls="--", where="post", label=f"Within-{anchor_label}")
                ax.axvspan(np.percentile(within_vals, 5), np.percentile(within_vals, 95), color="#888888", alpha=0.07)
                ax.axvline(np.median(within_vals), color="#888888", lw=0.8, ls=":", alpha=0.50)

        for cond in pstar_conds:
            vals = sub[cond].values
            x, y = ecdf_xy(vals)
            ax.step(x, y, color=TERM_COLOR[cond], lw=2.4, where="post", label=TERM_LABEL[cond])
            ax.axvline(np.median(vals), color=TERM_COLOR[cond], lw=0.8, ls=":", alpha=0.45)

        for cond in baseline_conds:
            vals = sub[cond].values
            x, y = ecdf_xy(vals)
            ax.step(x, y, color=TERM_COLOR[cond], lw=2.0, ls="--", where="post", label=TERM_LABEL[cond], alpha=0.82)
            ax.axvline(np.median(vals), color=TERM_COLOR[cond], lw=0.8, ls=":", alpha=0.40)

        ax.set_title(group_label, fontsize=11, fontweight="bold")
        ax.set_xlabel(LABEL_D_T_MSE, fontsize=9)
        ax.set_ylabel(LABEL_ECDF, fontsize=10)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.04)
        ax.grid(alpha=0.20)
        hide_top_right(ax)

    for ax in axes[len(groups):]:
        ax.axis("off")

    legend_handles = []
    if within_df is not None and not within_df.empty:
        legend_handles.append(Line2D([0], [0], color="#888888", lw=2.2, ls="--", label=f"Within-{anchor_label}"))
    for cond in pstar_conds:
        legend_handles.append(Line2D([0], [0], color=TERM_COLOR[cond], lw=2.4, ls="-", label=TERM_LABEL[cond]))
    for cond in baseline_conds:
        legend_handles.append(Line2D([0], [0], color=TERM_COLOR[cond], lw=2.0, ls="--", label=TERM_LABEL[cond]))
    if legend_handles:
        fig.legend(handles=legend_handles, title="Condition", fontsize=9, title_fontsize=10, loc="upper right")

    fig.suptitle(f"Plot 20 — ECDF: Distance from the {anchor_label} Logical Anchor by Taxonomy Group", fontsize=13)
    _set_row_ylabel(axes, n_rows, n_cols, LABEL_ECDF, fontsize=10)
    fig.tight_layout(rect=_tight_layout_rect(include_legend=bool(legend_handles)))
    save_fig(fig, out_dir / "plot_20_ecdf_groupwise.png")


def plot_21_groupwise(df_term, df_traj, out_dir, data_dir=None, **kw):
    if not SCIPY_OK:
        print("  Skipping groupwise plot 21 (scipy required).")
        return

    within_df = _resolve_within_df(data_dir=data_dir, within_and_records=kw.get("within_and_records"))
    groups = _ordered_groups(df_term)
    if not groups:
        print("  Skipping groupwise plot 21: taxonomy metadata is missing.")
        return

    prepared = []
    global_max = 0.0
    for group_key, group_label in groups:
        sub = df_term[df_term["taxonomy_group_key"] == group_key]
        if sub.empty:
            prepared.append((group_key, group_label, None))
            continue

        cond_registry = []
        if within_df is not None:
            within_vals = within_df[within_df["taxonomy_group_key"] == group_key]["d_within_and"].values
            if len(within_vals):
                cond_registry.append(("Within-anchor", "#888888", within_vals))
        for cond in get_present_pstar(sub):
            cond_registry.append((TERM_LABEL[cond], TERM_COLOR[cond], sub[cond].values))
        for cond in TERM_CONDITIONS + get_present_poe(sub):
            cond_registry.append((TERM_LABEL[cond], TERM_COLOR[cond], sub[cond].values))

        if len(cond_registry) < 2:
            prepared.append((group_key, group_label, None))
            continue

        all_vals = np.concatenate([vals for _, _, vals in cond_registry])
        x_grid = np.linspace(max(all_vals.min() * 0.70, 0), all_vals.max() * 1.10, 1000)
        pmfs = [kde_pmf(vals, x_grid) for _, _, vals in cond_registry]
        labels = [label for label, _, _ in cond_registry]
        colors = [color for _, color, _ in cond_registry]
        mat = np.zeros((len(cond_registry), len(cond_registry)))
        for i in range(len(cond_registry)):
            for j in range(i + 1, len(cond_registry)):
                j_val = jeffreys_div(pmfs[i], pmfs[j])
                mat[i, j] = j_val
                mat[j, i] = j_val
        global_max = max(global_max, float(mat.max()))
        prepared.append((group_key, group_label, (mat, labels, colors)))

    fig, axes, _n_rows, _n_cols = _panel_grid(len(prepared), figsize=(16, max(9.2, 5.0 * int(np.ceil(len(prepared) / 2)))))
    image_artist = None
    for ax, (_group_key, group_label, payload) in zip(axes, prepared):
        if payload is None:
            _blank_panel(ax, group_label)
            continue
        mat, labels, colors = payload
        image_artist = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(global_max * 1.05, 1e-6))
        thresh = mat.max() * 0.55 if mat.max() > 0 else 0.0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                txt_color = "white" if mat[i, j] > thresh else "#333333"
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7.5, color=txt_color)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
        ax.set_yticklabels(labels, fontsize=8.5)
        for tick, color in zip(ax.get_xticklabels(), colors):
            tick.set_color(color)
        for tick, color in zip(ax.get_yticklabels(), colors):
            tick.set_color(color)
        ax.set_title(group_label, fontsize=11, fontweight="bold")

    for ax in axes[len(prepared):]:
        ax.axis("off")

    if image_artist is not None:
        fig.colorbar(image_artist, ax=axes.tolist(), shrink=0.8, label=LABEL_JEFFREYS)

    fig.suptitle("Plot 21 — Jeffrey's Divergence Heatmap by Taxonomy Group", fontsize=13)
    fig.tight_layout(rect=_tight_layout_rect())
    save_fig(fig, out_dir / "plot_21_jeffreys_heatmap_groupwise.png")
