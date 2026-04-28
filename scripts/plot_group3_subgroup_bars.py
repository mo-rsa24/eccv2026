#!/usr/bin/env python3
r"""
Plot Group 3 subgroup joint-correctness bars from eval_joint_probes.py output.

Typical usage
-------------
python scripts/eval_joint_probes.py \
    --data-dir /datasets/mmolefe/eccv2026/experiments/eccv2026/taxonomy_qualitative_parallel \
    --conditions c1 c2 mono poe

python scripts/plot_group3_subgroup_bars.py \
    --data-dir /datasets/mmolefe/eccv2026/experiments/eccv2026/taxonomy_qualitative_parallel \
    --output-dir paper/neurips/Comparing\ Semantic\ and\ Logical\ Composition\ Using\ Latent\ Diffusion\ Models/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plots.joint_probe_bar import load_joint_probes
from plots.utils import hide_top_right, save_fig
from taxonomy_manifest import GROUP3_SUBGROUP_ORDER, GROUP3_SUBGROUP_LABEL_BY_KEY


COND_ORDER = ["c1", "c2", "mono", "poe"]
COND_COLOR = {
    "c1": "#4E79A7",
    "c2": "#59A14F",
    "mono": "#E15759",
    "poe": "#F28E2B",
}
COND_LABEL = {
    "c1": "A",
    "c2": "B",
    "mono": r"A$\wedge$B",
    "poe": "PoE",
}


def plot_group3_subgroup_bars(data_dir: Path, output_dir: Path) -> Path:
    _, image_df = load_joint_probes(data_dir)
    subgroup_keys = [k for k in GROUP3_SUBGROUP_ORDER if k in set(image_df["taxonomy_group_key"].dropna())]
    if not subgroup_keys:
        raise SystemExit(
            "No Group 3 subgroup records found in joint_probe_scores.json. "
            "Run eval_joint_probes.py on the Stage 0b qualitative directory first."
        )

    fig, axes = plt.subplots(1, len(subgroup_keys), figsize=(5.4 * len(subgroup_keys), 4.5), sharey=True)
    if len(subgroup_keys) == 1:
        axes = [axes]

    bar_width = 0.72
    gap = 0.28

    for idx, (ax, subgroup_key) in enumerate(zip(axes, subgroup_keys)):
        subgroup_df = image_df[image_df["taxonomy_group_key"] == subgroup_key]
        x_centers = np.arange(len(COND_ORDER)) * (bar_width + gap)
        for cond_idx, cond in enumerate(COND_ORDER):
            cond_df = subgroup_df[subgroup_df["condition"] == cond]
            if cond_df.empty:
                continue
            per_pair = cond_df.groupby("pair_slug")["joint_correctness_score"].mean()
            mean_score = float(per_pair.mean())
            sem_score = float(per_pair.sem()) if len(per_pair) > 1 else 0.0
            ax.bar(
                x_centers[cond_idx],
                mean_score,
                width=bar_width,
                color=COND_COLOR[cond],
                alpha=0.88,
                yerr=sem_score,
                capsize=3,
                error_kw={"linewidth": 1.0, "ecolor": "#333333"},
            )

        ax.set_xticks(x_centers)
        ax.set_xticklabels([COND_LABEL[c] for c in COND_ORDER], fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Joint correctness score" if idx == 0 else "", fontsize=9)
        ax.set_title(GROUP3_SUBGROUP_LABEL_BY_KEY.get(subgroup_key, subgroup_key), fontsize=10, fontweight="bold")
        hide_top_right(ax)

    fig.suptitle("Group 3 subgroup joint-correctness probes", fontsize=11, fontweight="bold", y=0.98)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "group3_subgroup_joint_probe_bar.png"
    save_fig(fig, out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Group 3 subgroup joint-correctness bars.")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Stage 0b qualitative directory containing joint_probe_scores.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory to write the PNG. Defaults to {data-dir}/figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "figures"
    out_path = plot_group3_subgroup_bars(data_dir, output_dir)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
