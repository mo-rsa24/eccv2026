"""
Figure: taxonomy qualitative overview as a nested 2x2 x 2 image grid.

This script renders the four taxonomy groups in a 2x2 outer grid. Each
group cell contains two saved result images from one representative pair:

1. trajectory_manifold.png
2. decoded_images.png

Expected directory layout:
    experiments/eccv2026/taxonomy_qualitative/<group>/<pair>/

Usage
-----
    python scripts/plot_trajectory_2x2.py
    python scripts/plot_trajectory_2x2.py --out figures/trajectory_2x2.pdf
    python scripts/plot_trajectory_2x2.py \
        --pairs \
            group1_cooccurrence/a_butterfly__x__a_flower_meadow \
            group2_disentangled/a_dog__x__oil_painting_style \
            group3_ood/a_desk_lamp__x__a_glacier \
            group4_collision/a_cat__x__a_dog

Output
------
    experiments/eccv2026/grid_figure/trajectory_2x2.png  (default)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "experiments" / "eccv2026" / "taxonomy_qualitative"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "eccv2026" / "grid_figure" / "trajectory_2x2.png"

GROUP_ORDER = [
    "group1_cooccurrence",
    "group2_disentangled",
    "group3_ood",
    "group4_collision",
]

GROUP_LABELS = {
    "group1_cooccurrence": "Group 1 - Manifold-Supported Co-occurrence",
    "group2_disentangled": "Group 2 - Feature-Space Disentangled",
    "group3_ood": "Group 3 - Low Co-occurrence, OOD",
    "group4_collision": "Group 4 - Adversarial Collision",
}

DEFAULT_PAIRS = [
    "group1_cooccurrence/a_butterfly__x__a_flower_meadow",
    "group2_disentangled/a_dog__x__oil_painting_style",
    "group3_ood/a_desk_lamp__x__a_glacier",
    "group4_collision/a_cat__x__a_dog",
]

VIEW_SPECS = [
    ("trajectory_manifold.png", "Trajectory manifold"),
    ("decoded_images.png", "Decoded images"),
]

INNER_HEIGHT_RATIOS = [2.7, 1.0]


@dataclass
class PanelSpec:
    pair_dir: Path
    group_key: str
    group_title: str
    pair_title: str


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  WARN: could not read {path}: {exc}", file=sys.stderr)
        return None


def _load_image(path: Path) -> np.ndarray | None:
    if not path.exists():
        print(f"  WARN: missing image: {path}", file=sys.stderr)
        return None
    try:
        return plt.imread(path)
    except OSError as exc:
        print(f"  WARN: could not read image {path}: {exc}", file=sys.stderr)
        return None


def _humanize_pair_slug(pair_slug: str) -> str:
    if "__x__" in pair_slug:
        left, right = pair_slug.split("__x__", maxsplit=1)
        return f"{left.replace('_', ' ')} x {right.replace('_', ' ')}"
    return pair_slug.replace("_", " ")


def _pair_title_from_summary(pair_dir: Path) -> str:
    summary = _load_json(pair_dir / "summary.json")
    if summary is None:
        return _humanize_pair_slug(pair_dir.name)

    config = summary.get("config", {})
    prompt_a = str(config.get("prompt_a", "")).strip()
    prompt_b = str(config.get("prompt_b", "")).strip()
    if prompt_a and prompt_b:
        return f"{prompt_a} x {prompt_b}"
    return _humanize_pair_slug(pair_dir.name)


def _first_available_pair(group_dir: Path) -> Path | None:
    if not group_dir.exists():
        return None
    pair_dirs = sorted(path for path in group_dir.iterdir() if path.is_dir())
    return pair_dirs[0] if pair_dirs else None


def _resolve_pair_dir(input_dir: Path, relative_or_absolute: str) -> Path:
    candidate = Path(relative_or_absolute)
    if candidate.is_absolute():
        return candidate
    return input_dir / candidate


def resolve_panel_specs(input_dir: Path, explicit_pairs: list[str] | None) -> list[PanelSpec]:
    pair_dirs: list[Path] = []

    if explicit_pairs is not None:
        pair_dirs = [_resolve_pair_dir(input_dir, pair) for pair in explicit_pairs]
    else:
        for group_key, default_pair in zip(GROUP_ORDER, DEFAULT_PAIRS):
            default_dir = input_dir / default_pair
            if default_dir.exists():
                pair_dirs.append(default_dir)
                continue

            fallback = _first_available_pair(input_dir / group_key)
            if fallback is None:
                print(f"  WARN: no pair directories found under {input_dir / group_key}", file=sys.stderr)
                pair_dirs.append(input_dir / group_key / "missing_pair")
            else:
                print(
                    f"  WARN: default pair missing for {group_key}; using {fallback.name}",
                    file=sys.stderr,
                )
                pair_dirs.append(fallback)

    panel_specs: list[PanelSpec] = []
    for pair_dir in pair_dirs:
        group_key = pair_dir.parent.name
        group_title = GROUP_LABELS.get(group_key, group_key.replace("_", " "))
        pair_title = _pair_title_from_summary(pair_dir)
        panel_specs.append(
            PanelSpec(
                pair_dir=pair_dir,
                group_key=group_key,
                group_title=group_title,
                pair_title=pair_title,
            )
        )

    return panel_specs


def _style_image_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("#d0d0d0")


def _draw_missing(ax: plt.Axes, missing_name: str) -> None:
    _style_image_axis(ax)
    ax.set_facecolor("#f4f4f4")
    ax.text(
        0.5,
        0.5,
        f"Missing\n{missing_name}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#666666",
    )


def _draw_image(ax: plt.Axes, image_path: Path, view_title: str) -> None:
    image = _load_image(image_path)
    if image is None:
        _draw_missing(ax, image_path.name)
    else:
        ax.imshow(image)
        _style_image_axis(ax)

    ax.text(
        0.02,
        0.98,
        view_title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color="#111111",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "#d0d0d0",
            "alpha": 0.9,
        },
    )


def make_figure(panel_specs: list[PanelSpec], output_path: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(14.0, 15.5))
    outer = fig.add_gridspec(
        2,
        2,
        left=0.035,
        right=0.985,
        top=0.94,
        bottom=0.035,
        wspace=0.08,
        hspace=0.14,
    )

    for idx, panel in enumerate(panel_specs):
        row, col = divmod(idx, 2)
        inner = outer[row, col].subgridspec(
            2,
            1,
            height_ratios=INNER_HEIGHT_RATIOS,
            hspace=0.04,
        )

        trajectory_ax = fig.add_subplot(inner[0, 0])
        decoded_ax = fig.add_subplot(inner[1, 0])

        trajectory_ax.set_title(
            f"{panel.group_title}\n({panel.pair_title})",
            fontsize=10,
            fontweight="bold",
            pad=10,
        )

        _draw_image(trajectory_ax, panel.pair_dir / VIEW_SPECS[0][0], VIEW_SPECS[0][1])
        _draw_image(decoded_ax, panel.pair_dir / VIEW_SPECS[1][0], VIEW_SPECS[1][1])

    fig.suptitle(
        "Taxonomy qualitative overview: trajectory manifolds and decoded images",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render a nested 2x2 taxonomy figure where each group cell contains "
            "trajectory_manifold.png and decoded_images.png."
        )
    )
    parser.add_argument(
        "--pairs",
        nargs=4,
        metavar="GROUP/PAIR",
        default=None,
        help=(
            "Four pair directories in reading order (G1 G2 G3 G4), relative to "
            "--input-dir unless absolute paths are supplied. If omitted, the "
            "script uses built-in representative pairs."
        ),
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Root taxonomy directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT),
        help=f"Output image path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output DPI (default: 180)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    panel_specs = resolve_panel_specs(input_dir, args.pairs)

    if len(panel_specs) != 4:
        parser.error(f"Expected 4 panel specs, found {len(panel_specs)}")

    for panel in panel_specs:
        print(f"Using {panel.group_key}: {panel.pair_dir}")

    make_figure(panel_specs, Path(args.out), dpi=args.dpi)


if __name__ == "__main__":
    main()
