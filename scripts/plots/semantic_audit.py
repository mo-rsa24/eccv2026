"""Semantic-baseline audit plots for SDXL-first analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from plots.utils import (
        GROUP_ORDER,
        GROUP_LABEL_BY_KEY,
        TERM_COLOR,
        hide_top_right,
        load_semantic_baseline_audit,
        save_fig,
    )
except ImportError:
    from utils import (
        GROUP_ORDER,
        GROUP_LABEL_BY_KEY,
        TERM_COLOR,
        hide_top_right,
        load_semantic_baseline_audit,
        save_fig,
    )


def _load_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit = load_semantic_baseline_audit(data_dir)
    if audit is None:
        raise FileNotFoundError(
            f"semantic_baseline_audit.json or joint_probe_scores.json not found under {data_dir}"
        )
    seed_df = pd.DataFrame(audit.get("seed_rows", []))
    pair_df = pd.DataFrame(audit.get("pair_rows", []))
    group_df = pd.DataFrame(audit.get("group_rows", []))
    return seed_df, pair_df, group_df


def plot_monolithic_semantic_audit(data_dir: Path, out_dir: Path) -> Path:
    _, pair_df, _ = _load_frames(data_dir)
    pair_df = pair_df.dropna(subset=["mono_mean_joint_correctness_score"])
    groups = [g for g in GROUP_ORDER if g in set(pair_df["taxonomy_group_key"].dropna())]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=True)
    axes = axes.flatten()
    rng = np.random.default_rng(42)

    y_top = 1.0
    for ax, group_key in zip(axes, groups):
        sub = pair_df[pair_df["taxonomy_group_key"] == group_key].sort_values(
            ["mono_semantic_pass_rate", "mono_mean_joint_correctness_score"],
            ascending=[False, False],
        )
        if sub.empty:
            ax.axis("off")
            continue
        xs = np.arange(len(sub))
        vals = sub["mono_mean_joint_correctness_score"].astype(float).to_numpy()
        colors = np.where(sub["pair_eligible_for_semantic_baseline"], "#3B8D5B", "#E15759")
        ax.bar(xs, vals, color=colors, alpha=0.86, width=0.72)
        jitter = rng.uniform(-0.12, 0.12, len(sub))
        ax.scatter(xs + jitter, sub["mono_semantic_pass_rate"], color="#4E79A7", s=34, zorder=3, label="Pass rate")
        ax.axhline(0.60, color="#6C757D", linestyle="--", linewidth=1.0)
        ax.axhline(0.75, color="#6C757D", linestyle=":", linewidth=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels([slug.replace("_", "\n") for slug in sub["pair_slug"]], fontsize=8)
        ax.set_ylim(0.0, y_top)
        ax.set_title(GROUP_LABEL_BY_KEY.get(group_key, group_key), fontsize=10, fontweight="bold")
        hide_top_right(ax)
    for ax in axes[len(groups):]:
        ax.axis("off")
    axes[0].set_ylabel("Monolithic score / pass rate", fontsize=10)
    axes[2].set_ylabel("Monolithic score / pass rate", fontsize=10)
    fig.suptitle("Monolithic semantic-baseline audit by taxonomy group", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path = out_dir / "semantic_baseline_audit_groupwise.png"
    save_fig(fig, out_path)
    return out_path


def plot_pair_retention_summary(data_dir: Path, out_dir: Path) -> Path:
    _, _, group_df = _load_frames(data_dir)
    if group_df.empty:
        raise ValueError("semantic baseline audit group rows are empty")
    group_df = group_df.set_index("taxonomy_group_key").reindex(GROUP_ORDER).dropna(how="all")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(group_df))
    accepted = group_df["accepted_count"].fillna(0).to_numpy()
    provisional = group_df["provisional_count"].fillna(0).to_numpy()
    excluded = group_df["excluded_count"].fillna(0).to_numpy()
    ax.bar(x, accepted, color="#3B8D5B", width=0.7, label="Accepted")
    ax.bar(x, provisional, bottom=accepted, color="#E8A838", width=0.7, label="Provisional")
    ax.bar(x, excluded, bottom=accepted + provisional, color="#D97B66", width=0.7, label="Excluded")
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABEL_BY_KEY.get(key, key) for key in group_df.index], fontsize=9)
    ax.set_ylabel("Pair count", fontsize=10)
    ax.set_title("Semantic-baseline roster retention summary", fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    hide_top_right(ax)
    fig.tight_layout()
    out_path = out_dir / "semantic_baseline_retention_summary.png"
    save_fig(fig, out_path)
    return out_path


def plot_semantic_pass_rate_comparison(data_dir: Path, out_dir: Path) -> Path:
    seed_df, _, _ = _load_frames(data_dir)
    seed_df = seed_df[seed_df["condition"].isin({"mono", "poe", "pstar_sdipc"})].copy()
    groups = [g for g in GROUP_ORDER if g in set(seed_df["taxonomy_group_key"].dropna())]
    agg = (
        seed_df.groupby(["taxonomy_group_key", "condition"])["semantic_pass"]
        .mean()
        .reset_index()
    )
    cond_order = ["mono", "poe", "pstar_sdipc"]
    colors = {"mono": TERM_COLOR["d_T_mono"], "poe": TERM_COLOR["d_T_poe"], "pstar_sdipc": TERM_COLOR["d_T_pstar_sdipc"]}
    labels = {"mono": r"A$\wedge$B", "poe": "PoE", "pstar_sdipc": r"PoE $p^\star$"}
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    x = np.arange(len(groups))
    width = 0.22
    for idx, cond in enumerate(cond_order):
        vals = []
        for group in groups:
            row = agg[(agg["taxonomy_group_key"] == group) & (agg["condition"] == cond)]
            vals.append(float(row["semantic_pass"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (idx - 1) * width, vals, width=width, color=colors[cond], alpha=0.88, label=labels[cond])
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABEL_BY_KEY.get(g, g) for g in groups], fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Semantic pass rate", fontsize=10)
    ax.set_title("Semantic pass-rate comparison by taxonomy group", fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    hide_top_right(ax)
    fig.tight_layout()
    out_path = out_dir / "semantic_pass_rate_comparison.png"
    save_fig(fig, out_path)
    return out_path


def plot_group34_boundary_audit(data_dir: Path, out_dir: Path) -> Path:
    _, pair_df, _ = _load_frames(data_dir)
    sub = pair_df[pair_df["taxonomy_group_key"].isin(["group3_feature_overlap", "group4_coherent_collision"])].copy()
    if sub.empty:
        raise ValueError("No Group 3/4 pair rows found in semantic baseline audit.")
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    palette = {
        "group3_feature_overlap": "#4E79A7",
        "group4_coherent_collision": "#E15759",
    }
    for group_key, group_sub in sub.groupby("taxonomy_group_key"):
        ax.scatter(
            group_sub["mono_hybridization_failure_rate"],
            group_sub["mono_omission_failure_rate"],
            s=90,
            alpha=0.86,
            color=palette[group_key],
            label=GROUP_LABEL_BY_KEY.get(group_key, group_key),
        )
        for _, row in group_sub.iterrows():
            ax.annotate(
                row["pair_slug"],
                (row["mono_hybridization_failure_rate"], row["mono_omission_failure_rate"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=7.5,
            )
    ax.set_xlabel("Monolithic hybridization failure rate", fontsize=10)
    ax.set_ylabel("Monolithic omission failure rate", fontsize=10)
    ax.set_title("Group 3 / Group 4 boundary audit", fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    hide_top_right(ax)
    fig.tight_layout()
    out_path = out_dir / "group34_boundary_audit.png"
    save_fig(fig, out_path)
    return out_path


def plot_semantic_filter_robustness(df_term_full: pd.DataFrame, df_traj_full: pd.DataFrame, df_term_qualified: pd.DataFrame, df_traj_qualified: pd.DataFrame, out_dir: Path) -> Path:
    groups = [g for g in GROUP_ORDER if g in set(df_term_full["taxonomy_group_key"].dropna())]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    terminal_full = []
    terminal_qualified = []
    temporal_full = []
    temporal_qualified = []
    for group in groups:
        sub_term_full = df_term_full[df_term_full["taxonomy_group_key"] == group]
        sub_term_qual = df_term_qualified[df_term_qualified["taxonomy_group_key"] == group]
        terminal_full.append(float(sub_term_full["d_T_mono"].mean()) if not sub_term_full.empty else np.nan)
        terminal_qualified.append(float(sub_term_qual["d_T_mono"].mean()) if not sub_term_qual.empty else np.nan)

        traj_full = df_traj_full[df_traj_full["taxonomy_group_key"] == group]
        traj_qual = df_traj_qualified[df_traj_qualified["taxonomy_group_key"] == group]
        temporal_full.append(float(traj_full.groupby("step")["d_t_mono"].mean().iloc[-1]) if not traj_full.empty else np.nan)
        temporal_qualified.append(float(traj_qual.groupby("step")["d_t_mono"].mean().iloc[-1]) if not traj_qual.empty else np.nan)

    x = np.arange(len(groups))
    width = 0.34
    for ax, title, vals_full, vals_qual in (
        (axes[0], "Terminal gap robustness", terminal_full, terminal_qualified),
        (axes[1], "Temporal gap robustness", temporal_full, temporal_qualified),
    ):
        ax.bar(x - width / 2, vals_full, width=width, color="#A0AEC0", alpha=0.9, label="Full descriptive")
        ax.bar(x + width / 2, vals_qual, width=width, color="#3B8D5B", alpha=0.9, label="Semantics-qualified")
        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_LABEL_BY_KEY.get(g, g) for g in groups], fontsize=8.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        hide_top_right(ax)
    axes[0].set_ylabel("Mean monolithic distance", fontsize=10)
    axes[0].legend(frameon=False)
    fig.suptitle("Full vs semantics-qualified robustness summary", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path = out_dir / "semantic_filter_robustness.png"
    save_fig(fig, out_path)
    return out_path
