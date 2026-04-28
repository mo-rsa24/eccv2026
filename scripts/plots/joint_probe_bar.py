"""
Pair-type-aware joint-probe grouped bar charts.

Produces the corrected output-level grouped bars used to complement BLIP-VQA
cue-presence diagnostics.
"""

from __future__ import annotations

import json
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
        hide_top_right,
        save_fig,
        enrich_taxonomy_dataframe,
    )
except ImportError:
    from utils import (
        GROUP_ORDER,
        GROUP_LABEL_BY_KEY,
        hide_top_right,
        save_fig,
        enrich_taxonomy_dataframe,
    )

_COND_COLOR = {
    "c1": "#4E79A7",
    "c2": "#59A14F",
    "mono": "#E15759",
    "poe": "#F28E2B",
    "pstar_sdipc": "#E8A838",
}

_COND_LABEL = {
    "c1": "A",
    "c2": "B",
    "mono": r"A$\wedge$B",
    "poe": "PoE",
    "pstar_sdipc": r"PoE $p^\star$",
}

_BASE_CONDITIONS = ["c1", "c2", "mono", "poe"]


def condition_color_map() -> dict[str, str]:
    """Expose the canonical joint-probe/group-hybrid condition colors."""
    return dict(_COND_COLOR)


def condition_label_map() -> dict[str, str]:
    """Expose the canonical grouped-bar labels."""
    return dict(_COND_LABEL)


def load_joint_probes(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = data_dir / "metrics" / "joint_probe_scores.json"
    if candidate.exists():
        path = candidate
    else:
        path = data_dir / "joint_probe_scores.json"
    if not path.exists():
        raise FileNotFoundError(
            f"joint_probe_scores.json not found under {data_dir}.\n"
            "Run: python scripts/eval_joint_probes.py --data-dir <run>"
        )
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        probe_df = pd.DataFrame(payload.get("probe_records", []))
        image_df = pd.DataFrame(payload.get("image_scores", []))
    else:
        probe_df = pd.DataFrame(payload)
        image_df = probe_df.copy()
    probe_df = enrich_taxonomy_dataframe(probe_df)
    image_df = enrich_taxonomy_dataframe(image_df)
    return probe_df, image_df


def plot_joint_probe_grouped_bar(
    df_scores: pd.DataFrame,
    out_dir: Path,
    pstar: bool = False,
    faceted: bool = True,
) -> None:
    conditions = list(_BASE_CONDITIONS)
    if pstar:
        conditions = conditions + ["pstar_sdipc"]

    groups = [g for g in GROUP_ORDER if g in df_scores["taxonomy_group_key"].values]
    if not groups:
        print("Warning: no recognised taxonomy groups in joint-probe data — skipping bar chart.")
        return

    n_groups = len(groups)
    nrows = 2 if faceted and n_groups > 2 else 1
    ncols = 2 if faceted and n_groups > 1 else max(1, n_groups)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows), sharey=True)
    axes_flat = np.array(axes).flatten() if n_groups > 1 else [axes]

    bar_width = 0.72
    gap = 0.28

    for ax_idx, (group_key, ax) in enumerate(zip(groups, axes_flat)):
        group_df = df_scores[df_scores["taxonomy_group_key"] == group_key]
        x_centers = np.arange(len(conditions)) * (bar_width + gap)
        for cond_idx, cond in enumerate(conditions):
            cond_df = group_df[group_df["condition"] == cond]
            if cond_df.empty:
                continue
            per_pair = cond_df.groupby("pair_slug")["joint_correctness_score"].mean()
            mean_score = float(per_pair.mean())
            sem_score = float(per_pair.sem()) if len(per_pair) > 1 else 0.0
            ax.bar(
                x_centers[cond_idx],
                mean_score,
                width=bar_width,
                color=_COND_COLOR.get(cond, "#888888"),
                alpha=0.88,
                yerr=sem_score,
                capsize=3,
                error_kw={"linewidth": 1.0, "ecolor": "#333333"},
            )

        ax.set_xticks(x_centers)
        ax.set_xticklabels([_COND_LABEL.get(c, c) for c in conditions], fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Joint correctness score" if ax_idx % ncols == 0 else "", fontsize=9)
        ax.set_title(GROUP_LABEL_BY_KEY.get(group_key, group_key), fontsize=10, fontweight="bold")
        hide_top_right(ax)

    for ax in axes_flat[n_groups:]:
        ax.set_visible(False)

    fig.suptitle(
        "Pair-type-aware joint-correctness probes",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    suffix = "_pstar" if pstar else ""
    out_path = out_dir / f"joint_probe_grouped_bar{suffix}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_path)


def plot_joint_probe_group_on_axis(
    ax,
    df_scores: pd.DataFrame,
    group_key: str,
    *,
    pstar: bool = False,
    show_ylabel: bool = True,
    show_title: bool = True,
    title: str | None = None,
) -> None:
    """Render a single taxonomy group's joint-probe grouped bars on a supplied axis."""
    conditions = list(_BASE_CONDITIONS)
    if pstar:
        conditions = conditions + ["pstar_sdipc"]

    group_df = df_scores[df_scores["taxonomy_group_key"] == group_key]
    if group_df.empty:
        raise ValueError(f"No joint-probe rows found for taxonomy group '{group_key}'.")

    bar_width = 0.72
    gap = 0.28
    x_centers = np.arange(len(conditions)) * (bar_width + gap)

    for cond_idx, cond in enumerate(conditions):
        cond_df = group_df[group_df["condition"] == cond]
        if cond_df.empty:
            continue
        per_pair = cond_df.groupby("pair_slug")["joint_correctness_score"].mean()
        mean_score = float(per_pair.mean())
        sem_score = float(per_pair.sem()) if len(per_pair) > 1 else 0.0
        ax.bar(
            x_centers[cond_idx],
            mean_score,
            width=bar_width,
            color=_COND_COLOR.get(cond, "#888888"),
            alpha=0.88,
            yerr=sem_score,
            capsize=3,
            error_kw={"linewidth": 1.0, "ecolor": "#333333"},
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels([_COND_LABEL.get(c, c) for c in conditions], fontsize=8.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Joint correctness score" if show_ylabel else "", fontsize=8.5)
    if show_title:
        ax.set_title(title or GROUP_LABEL_BY_KEY.get(group_key, group_key), fontsize=9.5, fontweight="bold")
    hide_top_right(ax)
