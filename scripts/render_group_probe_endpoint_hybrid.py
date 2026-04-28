#!/usr/bin/env python3
"""Render a 3x2 hybrid figure: groupwise probes above representative endpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plots.blip_vqa_bar import (  # noqa: E402
    condition_color_map as blip_condition_color_map,
    condition_label_map as blip_condition_label_map,
    load_blip_vqa,
    plot_blip_vqa_group_on_axis,
)
from plots.joint_probe_bar import (  # noqa: E402
    load_joint_probes,
    plot_joint_probe_group_on_axis,
)
from render_taxonomy_paper_figure import load_panels, _read_image  # noqa: E402
from taxonomy_manifest import GROUP_ORDER, GROUP_LABEL_BY_KEY  # noqa: E402


DEFAULT_DATA_DIR = PROJECT_ROOT / "experiments" / "eccv2026" / "sdxl_final" / "sdxl_six_group_seed42_steps50_cfg7p5_frozen"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "paper"
    / "neurips"
    / "Comparing Semantic and Logical Composition Using Latent Diffusion Models"
    / "figures"
)

DEFAULT_ENDPOINT_CONDITIONS = ["prompt_a", "prompt_b", "monolithic", "poe"]
BAR_CONDITION_BY_ENDPOINT = {
    "prompt_a": "c1",
    "prompt_b": "c2",
    "monolithic": "mono",
    "poe": "poe",
    "pstar_sdipc": "pstar_sdipc",
}


def _load_panel_notes(value: str) -> dict[str, str]:
    if not value:
        return {}
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(value)


def _metric_title(metric: str) -> str:
    if metric == "blip_vqa":
        return "BLIP-VQA cue-presence diagnostic with representative decoded endpoints"
    if metric == "joint_probe":
        return "Pair-type-aware joint probes with representative decoded endpoints"
    raise ValueError(f"Unknown metric {metric!r}")


def _metric_output_name(metric: str) -> str:
    return {
        "blip_vqa": "blip_vqa_endpoint_hybrid_3x2.png",
        "joint_probe": "joint_probe_endpoint_hybrid_3x2.png",
    }[metric]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render nested 3x2 groupwise probe + endpoint hybrids.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Frozen SDXL final root for decoded endpoints.")
    parser.add_argument("--metrics-dir", default="", help="Run root containing blip_vqa_scores.json and/or joint_probe_scores.json.")
    parser.add_argument("--metric", choices=["blip_vqa", "joint_probe"], required=True)
    parser.add_argument(
        "--metric-scope",
        choices=["pair", "group"],
        default="pair",
        help=(
            "Whether metric bars should summarize only the displayed pair or the full taxonomy group. "
            "Default is 'pair' so the bars match the decoded endpoints shown underneath."
        ),
    )
    parser.add_argument("--pairs", nargs=6, required=True, metavar="PAIR", help="Six explicit pair slugs/paths in canonical G1..G6 order.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for decoded endpoint lookup.")
    parser.add_argument(
        "--endpoint-conditions",
        nargs="+",
        default=list(DEFAULT_ENDPOINT_CONDITIONS),
        choices=["prompt_a", "prompt_b", "monolithic", "poe", "pstar_sdipc"],
        help="Decoded endpoint conditions to render in the lower strip. Defaults to prompt_a prompt_b monolithic poe.",
    )
    parser.add_argument("--out", default="", help="Output PNG path.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--panel-notes", default="", help="Optional JSON file or inline JSON mapping group key -> short note.")
    return parser.parse_args()


def _draw_endpoint_strip(fig, spec, panel, decoded_grid, border_colors, condition_labels, endpoint_conditions):
    decoded_axes = []
    for cond_idx, cond in enumerate(endpoint_conditions):
        rel_path = panel.decoded_paths.get(cond)
        if rel_path is None:
            raise KeyError(f"Missing decoded image path for '{cond}' in {panel.pair_dir / 'grid_assets.json'}")
        ax = fig.add_subplot(decoded_grid[0, cond_idx])
        ax.imshow(_read_image(panel.pair_dir / rel_path))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        title = ax.set_title(condition_labels[cond], fontsize=7.5, pad=4.5, fontweight="bold", color=border_colors[cond])
        title.set_bbox(
            {
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": border_colors[cond],
                "linewidth": 0.85,
                "alpha": 0.95,
            }
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.6)
            spine.set_color(border_colors[cond])
        decoded_axes.append(ax)

    decoded_axes[0].text(
        0.0,
        1.23,
        f"{spec['group_label']}\n({panel.pair_title})",
        transform=decoded_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color="#20262E",
    )
    return decoded_axes


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else data_dir
    output_path = Path(args.out) if args.out else DEFAULT_OUTPUT_DIR / _metric_output_name(args.metric)
    panel_notes = _load_panel_notes(args.panel_notes)
    endpoint_conditions = list(args.endpoint_conditions)

    panels = load_panels(data_dir, list(args.pairs), monolithic_baseline="auto", seed=args.seed)

    if args.metric == "blip_vqa":
        df_blip = load_blip_vqa(metrics_dir)
        draw_group_metric = plot_blip_vqa_group_on_axis
        border_colors = {cond: blip_condition_color_map()[BAR_CONDITION_BY_ENDPOINT[cond]] for cond in endpoint_conditions}
        condition_labels = {cond: blip_condition_label_map()[BAR_CONDITION_BY_ENDPOINT[cond]] for cond in endpoint_conditions}
    else:
        _, df_scores = load_joint_probes(metrics_dir)
        draw_group_metric = plot_joint_probe_group_on_axis
        from plots.joint_probe_bar import condition_color_map as joint_condition_color_map, condition_label_map as joint_condition_label_map

        border_colors = {cond: joint_condition_color_map()[BAR_CONDITION_BY_ENDPOINT[cond]] for cond in endpoint_conditions}
        condition_labels = {cond: joint_condition_label_map()[BAR_CONDITION_BY_ENDPOINT[cond]] for cond in endpoint_conditions}

    fig = plt.figure(figsize=(18.5, 15.8), facecolor="white")
    outer = fig.add_gridspec(
        3,
        2,
        left=0.04,
        right=0.985,
        top=0.92,
        bottom=0.04,
        wspace=0.12,
        hspace=0.18,
    )

    for idx, (group_key, pair_value, panel) in enumerate(zip(GROUP_ORDER, args.pairs, panels)):
        row, col = divmod(idx, 2)
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[1.7, 1.0], hspace=0.16)
        metric_ax = fig.add_subplot(inner[0, 0])
        decoded_grid = inner[1, 0].subgridspec(1, len(endpoint_conditions), wspace=0.06)

        title = GROUP_LABEL_BY_KEY.get(group_key, group_key)
        show_ylabel = col == 0
        pair_slug = Path(pair_value).name
        if args.metric == "blip_vqa":
            df_metric = df_blip if args.metric_scope == "group" else df_blip[df_blip["pair_slug"] == pair_slug]
            if df_metric.empty:
                raise ValueError(
                    f"No BLIP-VQA rows found for pair '{pair_slug}' in {metrics_dir / 'blip_vqa_scores.json'}."
                )
            draw_group_metric(metric_ax, df_metric, group_key, show_ylabel=show_ylabel, show_title=True, title=title, legend_mode="none")
        else:
            df_metric = df_scores if args.metric_scope == "group" else df_scores[df_scores["pair_slug"] == pair_slug]
            if df_metric.empty:
                raise ValueError(
                    f"No joint-probe rows found for pair '{pair_slug}' in {metrics_dir / 'joint_probe_scores.json'}."
                )
            draw_group_metric(metric_ax, df_metric, group_key, show_ylabel=show_ylabel, show_title=True, title=title)

        _draw_endpoint_strip(
            fig,
            {"group_label": title, "pair_value": pair_value},
            panel,
            decoded_grid,
            border_colors,
            condition_labels,
            endpoint_conditions,
        )

        note = panel_notes.get(group_key, "")
        if note:
            metric_ax.text(
                0.99,
                0.98,
                note,
                transform=metric_ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.6,
                color="#4C566A",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "#F8F9FB",
                    "edgecolor": "#D7DEE8",
                    "linewidth": 0.8,
                },
            )

    if args.metric == "blip_vqa":
        from matplotlib.patches import Patch

        legend_handles = [
            Patch(facecolor="#888888", alpha=0.88, label="P(A present) solid"),
            Patch(facecolor="#888888", alpha=0.55, hatch="///", label="P(B present) hatched"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.975),
            ncol=2,
            fontsize=9,
            frameon=False,
        )

    fig.suptitle(_metric_title(args.metric), fontsize=14, fontweight="bold", y=0.985)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
