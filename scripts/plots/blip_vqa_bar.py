"""
BLIP-VQA concept-presence grouped bar charts.

Produces F0 (pstar=False) and F0b (pstar=True) for the NeurIPS paper.

Public API
----------
load_blip_vqa(data_dir)           -> pd.DataFrame
plot_blip_vqa_grouped_bar(df, ...) -> None
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
        _resolve_json,
    )
except ImportError:
    from utils import (
        GROUP_ORDER,
        GROUP_LABEL_BY_KEY,
        hide_top_right,
        save_fig,
        enrich_taxonomy_dataframe,
        _resolve_json,
    )

# ---------------------------------------------------------------------------
# Color constants for BLIP-VQA bars (reuse TERM_COLOR hues by condition key)
# ---------------------------------------------------------------------------

_COND_COLOR = {
    "c1":           "#4E79A7",   # blue  (same as TERM_COLOR d_T_c1)
    "c2":           "#59A14F",   # green (same as TERM_COLOR d_T_c2)
    "mono":         "#E15759",   # red   (same as TERM_COLOR d_T_mono)
    "poe":          "#F28E2B",   # orange (same as TERM_COLOR d_T_poe)
    "pstar_sdipc":  "#E8A838",   # amber  (same as TERM_COLOR d_T_pstar_sdipc)
}

_COND_LABEL = {
    "c1":          "A",
    "c2":          "B",
    "mono":        r"A$\wedge$B",
    "poe":         "PoE",
    "pstar_sdipc": r"PoE $p^\star$",
}

_BASE_CONDITIONS = ["c1", "c2", "mono", "poe"]


def condition_color_map() -> dict[str, str]:
    """Expose the canonical BLIP-VQA/joint-probe condition colors."""
    return dict(_COND_COLOR)


def condition_label_map() -> dict[str, str]:
    """Expose the canonical grouped-bar labels."""
    return dict(_COND_LABEL)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_blip_vqa(data_dir: Path) -> pd.DataFrame:
    """Load blip_vqa_scores.json.  Checks metrics/ subfolder first, then root."""
    candidate = data_dir / "metrics" / "blip_vqa_scores.json"
    if candidate.exists():
        path = candidate
    else:
        path = data_dir / "blip_vqa_scores.json"
    if not path.exists():
        raise FileNotFoundError(
            f"blip_vqa_scores.json not found under {data_dir}.\n"
            "Run: python scripts/eval_blip_vqa.py --data-dir <run>"
        )
    df = pd.DataFrame(json.loads(path.read_text()))
    df = enrich_taxonomy_dataframe(df)
    return df


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def plot_blip_vqa_grouped_bar(
    df_blip: pd.DataFrame,
    out_dir: Path,
    pstar: bool = False,
    faceted: bool = True,
) -> None:
    """
    Grouped bar chart of mean P(concept present) by condition, stratified by
    taxonomy group.

    Parameters
    ----------
    df_blip : DataFrame with columns pair_slug, taxonomy_group_key, condition,
              p_c1, p_c2 (from eval_blip_vqa.py).
    out_dir : Directory to write output PNG.
    pstar   : If True, include pstar_sdipc condition (F0b); else omit it (F0).
    faceted : If True, render 2×2 panel (one per taxonomy group).
    """
    conditions = list(_BASE_CONDITIONS)
    if pstar:
        conditions = conditions + ["pstar_sdipc"]

    groups = [g for g in GROUP_ORDER if g in df_blip["taxonomy_group_key"].values]
    if not groups:
        print("Warning: no recognised taxonomy groups in BLIP-VQA data — skipping bar chart.")
        return

    n_groups = len(groups)
    nrows = 2 if faceted and n_groups > 2 else 1
    ncols = 2 if faceted and n_groups > 1 else max(1, n_groups)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows), sharey=True)
    axes_flat = np.array(axes).flatten() if n_groups > 1 else [axes]

    bar_width = 0.35
    gap = 0.12          # gap between condition groups

    for ax_idx, (group_key, ax) in enumerate(zip(groups, axes_flat)):
        group_df = df_blip[df_blip["taxonomy_group_key"] == group_key]
        group_label = GROUP_LABEL_BY_KEY.get(group_key, group_key)

        n_conds = len(conditions)
        x_centers = np.arange(n_conds) * (2 * bar_width + gap)

        for cond_idx, cond in enumerate(conditions):
            cond_df = group_df[group_df["condition"] == cond]
            if cond_df.empty:
                continue

            # SEM: mean per pair_slug, then SEM over pairs
            per_pair_c1 = cond_df.groupby("pair_slug")["p_c1"].mean()
            per_pair_c2 = cond_df.groupby("pair_slug")["p_c2"].mean()

            mean_c1 = float(per_pair_c1.mean())
            mean_c2 = float(per_pair_c2.mean())
            sem_c1  = float(per_pair_c1.sem()) if len(per_pair_c1) > 1 else 0.0
            sem_c2  = float(per_pair_c2.sem()) if len(per_pair_c2) > 1 else 0.0

            color = _COND_COLOR.get(cond, "#888888")
            x_c = x_centers[cond_idx]

            # P(c1) — solid bar
            ax.bar(
                x_c,
                mean_c1,
                width=bar_width,
                color=color,
                alpha=0.88,
                yerr=sem_c1,
                capsize=3,
                error_kw={"linewidth": 1.0, "ecolor": "#333333"},
                label=f"P(A) {_COND_LABEL.get(cond, cond)}" if ax_idx == 0 else None,
            )
            # P(c2) — hatched bar
            ax.bar(
                x_c + bar_width,
                mean_c2,
                width=bar_width,
                color=color,
                alpha=0.55,
                hatch="///",
                yerr=sem_c2,
                capsize=3,
                error_kw={"linewidth": 1.0, "ecolor": "#333333"},
                label=f"P(B) {_COND_LABEL.get(cond, cond)}" if ax_idx == 0 else None,
            )

        tick_x = x_centers + bar_width / 2
        ax.set_xticks(tick_x)
        ax.set_xticklabels([_COND_LABEL.get(c, c) for c in conditions], fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("P(concept present)" if ax_idx % ncols == 0 else "", fontsize=9)
        ax.set_title(group_label, fontsize=10, fontweight="bold")
        hide_top_right(ax)

    # Hide unused axes
    for ax in axes_flat[n_groups:]:
        ax.set_visible(False)

    # Shared legend (P(c1) solid vs P(c2) hatched)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#888888", alpha=0.88, label="P(A present) — solid"),
        Patch(facecolor="#888888", alpha=0.55, hatch="///", label="P(B present) — hatched"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        fontsize=9,
        frameon=False,
    )

    suffix = "_pstar" if pstar else ""
    out_path = out_dir / f"blip_vqa_grouped_bar{suffix}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_path)


def plot_blip_vqa_group_on_axis(
    ax,
    df_blip: pd.DataFrame,
    group_key: str,
    *,
    pstar: bool = False,
    show_ylabel: bool = True,
    show_title: bool = True,
    title: str | None = None,
    legend_mode: str = "none",
) -> None:
    """Render a single taxonomy group's BLIP-VQA grouped bars on a supplied axis."""
    conditions = list(_BASE_CONDITIONS)
    if pstar:
        conditions = conditions + ["pstar_sdipc"]

    group_df = df_blip[df_blip["taxonomy_group_key"] == group_key]
    if group_df.empty:
        raise ValueError(f"No BLIP-VQA rows found for taxonomy group '{group_key}'.")

    bar_width = 0.35
    gap = 0.12
    x_centers = np.arange(len(conditions)) * (2 * bar_width + gap)

    for cond_idx, cond in enumerate(conditions):
        cond_df = group_df[group_df["condition"] == cond]
        if cond_df.empty:
            continue

        per_pair_c1 = cond_df.groupby("pair_slug")["p_c1"].mean()
        per_pair_c2 = cond_df.groupby("pair_slug")["p_c2"].mean()

        mean_c1 = float(per_pair_c1.mean())
        mean_c2 = float(per_pair_c2.mean())
        sem_c1 = float(per_pair_c1.sem()) if len(per_pair_c1) > 1 else 0.0
        sem_c2 = float(per_pair_c2.sem()) if len(per_pair_c2) > 1 else 0.0

        color = _COND_COLOR.get(cond, "#888888")
        x_c = x_centers[cond_idx]

        ax.bar(
            x_c,
            mean_c1,
            width=bar_width,
            color=color,
            alpha=0.88,
            yerr=sem_c1,
            capsize=3,
            error_kw={"linewidth": 1.0, "ecolor": "#333333"},
        )
        ax.bar(
            x_c + bar_width,
            mean_c2,
            width=bar_width,
            color=color,
            alpha=0.55,
            hatch="///",
            yerr=sem_c2,
            capsize=3,
            error_kw={"linewidth": 1.0, "ecolor": "#333333"},
        )

    tick_x = x_centers + bar_width / 2
    ax.set_xticks(tick_x)
    ax.set_xticklabels([_COND_LABEL.get(c, c) for c in conditions], fontsize=8.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("P(concept present)" if show_ylabel else "", fontsize=8.5)
    if show_title:
        ax.set_title(title or GROUP_LABEL_BY_KEY.get(group_key, group_key), fontsize=9.5, fontweight="bold")
    hide_top_right(ax)

    if legend_mode != "none":
        from matplotlib.patches import Patch

        legend_handles = [
            Patch(facecolor="#888888", alpha=0.88, label="P(A) solid"),
            Patch(facecolor="#888888", alpha=0.55, hatch="///", label="P(B) hatched"),
        ]
        loc = "upper right" if legend_mode == "inside" else "upper center"
        bbox = None if legend_mode == "inside" else (0.5, 1.16)
        ax.legend(
            handles=legend_handles,
            loc=loc,
            bbox_to_anchor=bbox,
            ncol=2 if legend_mode != "inside" else 1,
            fontsize=7.5,
            frameon=False,
        )
