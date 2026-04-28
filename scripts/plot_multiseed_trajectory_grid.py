"""
Figure: multi-seed trajectory overview as a nested 2x2 image grid.

This script renders four seeds in a 2x2 outer grid. Each seed cell contains
two saved result images from one representative pair run with that seed:

1. trajectory_manifold.png
2. decoded_images.png

Expected directory layout:
    experiments/eccv2026/multiseed/<pair>/<seed>/

Usage
-----
    python scripts/plot_multiseed_trajectory_grid.py \\
        --pair-dirs \\
            experiments/eccv2026/multiseed/butterfly_x_flower/seed_0/group1_cooccurrence/a_butterfly__x__a_flower_meadow \\
            experiments/eccv2026/multiseed/butterfly_x_flower/seed_7/group1_cooccurrence/a_butterfly__x__a_flower_meadow \\
            experiments/eccv2026/multiseed/butterfly_x_flower/seed_42/group1_cooccurrence/a_butterfly__x__a_flower_meadow \\
            experiments/eccv2026/multiseed/butterfly_x_flower/seed_99/group1_cooccurrence/a_butterfly__x__a_flower_meadow

Output
------
    multiseed_grid.png  (2x2 grid matching trajectory_2x2.py style)
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

TRAJECTORY_VIEW = ("trajectory_manifold.png", "Trajectory manifold")
INNER_HEIGHT_RATIOS = [2.7, 1.0]

CONDITION_ORDER = ["prompt_a", "prompt_b", "monolithic", "poe"]
CONDITION_LABELS = {
    "prompt_a": "A",
    "prompt_b": "B",
    "monolithic": "A∧B",
    "poe": "PoE",
}
CONDITION_COLORS = {
    "prompt_a": "#C84C5B",
    "prompt_b": "#2B6F97",
    "monolithic": "#3B8D5B",
    "poe": "#D9872B",
}
CONDITION_MARKERS = {
    "prompt_a": "o",
    "prompt_b": "s",
    "monolithic": "D",
    "poe": "^",
}
LABEL_OFFSETS = {
    "prompt_a": (10, 10),
    "prompt_b": (10, -12),
    "monolithic": (-16, 12),
    "poe": (-18, -12),
}
SECTION_HEADER_BBOX = {
    "boxstyle": "round,pad=0.22",
    "facecolor": "#F4F6F8",
    "edgecolor": "#D7DEE8",
    "linewidth": 0.8,
    "alpha": 0.98,
}


@dataclass
class PanelSpec:
    pair_dir: Path
    seed_number: int
    seed_title: str
    pair_title: str
    projection_method: str
    condition_labels: dict[str, str]
    projected_paths: dict[str, list[list[float]]]
    decoded_paths: dict[str, str]


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


def _extract_seed_number(pair_dir: Path) -> int:
    """Extract seed number from any ancestor directory named `seed_<n>`."""
    for ancestor in pair_dir.parents:
        name = ancestor.name
        if not name.startswith("seed_"):
            continue
        try:
            return int(name.split("_", maxsplit=1)[1])
        except (ValueError, IndexError):
            continue
    return 0


def _first_matching_image(pair_dir: Path, patterns: list[str]) -> str | None:
    for pattern in patterns:
        matches = sorted(pair_dir.glob(pattern))
        if matches:
            return str(matches[0].relative_to(pair_dir))
    return None


def _infer_decoded_paths(pair_dir: Path) -> dict[str, str]:
    decoded_paths: dict[str, str] = {}
    candidates = {
        "prompt_a": ["solo_a.png", "images/sdxl_solo_a_*.png", "images/sd14_solo_a_*.png", "images/sd35_solo_a_*.png"],
        "prompt_b": ["solo_b.png", "images/sdxl_solo_b_*.png", "images/sd14_solo_b_*.png", "images/sd35_solo_b_*.png"],
        "monolithic": ["monolithic.png", "images/sdxl_monolithic_*.png", "images/sd14_monolithic_*.png", "images/sd35_monolithic_*.png"],
        "poe": ["poe.png", "images/sdxl_poe_*.png", "images/sd14_poe_*.png", "images/sd35_poe_*.png"],
    }
    for cond, patterns in candidates.items():
        rel_path = _first_matching_image(pair_dir, patterns)
        if rel_path is not None:
            decoded_paths[cond] = rel_path
    return decoded_paths


def _decoded_paths_from_assets(pair_dir: Path) -> dict[str, str]:
    asset = _load_json(pair_dir / "grid_assets.json")
    if asset is not None:
        decoded_paths = dict(asset.get("decoded_image_paths", {}))
        if decoded_paths:
            return decoded_paths
    return _infer_decoded_paths(pair_dir)


def _trajectory_data_from_assets(pair_dir: Path) -> tuple[str, dict[str, str], dict[str, list[list[float]]]]:
    asset = _load_json(pair_dir / "grid_assets.json") or {}
    traj = asset.get("trajectory_projection") or {}
    projected_paths = dict(traj.get("projected", {}))
    condition_labels = dict(asset.get("condition_labels", {}))
    projection_method = str(traj.get("projection_method") or asset.get("projection_method") or "mds")
    return projection_method, condition_labels, projected_paths


def resolve_panel_specs(pair_dirs: list[Path]) -> list[PanelSpec]:
    """Create panel specs from a list of pair directories (one per seed)."""
    panel_specs: list[PanelSpec] = []
    for pair_dir in pair_dirs:
        seed_number = _extract_seed_number(pair_dir)
        pair_title = _pair_title_from_summary(pair_dir)
        projection_method, condition_labels, projected_paths = _trajectory_data_from_assets(pair_dir)
        panel_specs.append(
            PanelSpec(
                pair_dir=pair_dir,
                seed_number=seed_number,
                seed_title=f"Seed {seed_number}",
                pair_title=pair_title,
                projection_method=projection_method,
                condition_labels=condition_labels,
                projected_paths=projected_paths,
                decoded_paths=_decoded_paths_from_assets(pair_dir),
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


def _style_decoded_axis(ax: plt.Axes, color: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)
        spine.set_color(color)


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


def _section_header(ax: plt.Axes, text: str, y: float = 1.02) -> None:
    ax.text(
        0.0,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color="#2A2F36",
        bbox=SECTION_HEADER_BBOX,
        clip_on=False,
        zorder=10,
    )


def _style_manifold_axis(ax: plt.Axes, method: str) -> None:
    prefix = "MDS" if method == "mds" else "PC"
    ax.set_facecolor("#FBFCFD")
    ax.grid(True, color="#E1E7EF", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.margins(x=0.12, y=0.14)
    ax.set_xlabel(f"{prefix} 1", fontsize=9.0)
    ax.set_ylabel(f"{prefix} 2", fontsize=9.0)
    ax.tick_params(axis="both", labelsize=8.0, colors="#4C566A")
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#C7D0DB")


def _endpoint_is_crowded(cond: str, endpoints: dict[str, np.ndarray]) -> bool:
    current = endpoints[cond]
    others = [endpoints[name] for name in endpoints if name != cond]
    if not others:
        return False
    all_points = np.vstack(list(endpoints.values()))
    span = max(float(np.ptp(all_points[:, 0])), float(np.ptp(all_points[:, 1])), 1.0)
    min_dist = min(float(np.linalg.norm(current - other)) for other in others)
    return min_dist < 0.16 * span


def _annotate_endpoints(
    ax: plt.Axes,
    conditions: list[str],
    projected: dict[str, np.ndarray],
    condition_labels: dict[str, str],
) -> None:
    endpoints = {cond: projected[cond][-1] for cond in conditions}
    for cond in conditions:
        endpoint = endpoints[cond]
        color = CONDITION_COLORS.get(cond, "#4C566A")
        crowded = _endpoint_is_crowded(cond, endpoints)
        arrowprops = None
        if crowded:
            arrowprops = {
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.9,
                "shrinkA": 2,
                "shrinkB": 3,
                "alpha": 0.9,
            }
        ax.annotate(
            condition_labels.get(cond, CONDITION_LABELS.get(cond, cond)),
            xy=(endpoint[0], endpoint[1]),
            xytext=LABEL_OFFSETS.get(cond, (8, 8)),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=7.8,
            fontweight="bold",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.9,
                "alpha": 0.96,
            },
            arrowprops=arrowprops,
            zorder=7,
        )


def _plot_manifold(
    ax: plt.Axes,
    projected: dict[str, np.ndarray],
    conditions: list[str],
    method: str,
    condition_labels: dict[str, str],
) -> None:
    for cond in conditions:
        pts = projected[cond]
        color = CONDITION_COLORS.get(cond, "#4C566A")
        marker = CONDITION_MARKERS.get(cond, "o")

        if len(pts) >= 2:
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=color,
                linewidth=2.15,
                alpha=0.95,
                solid_capstyle="round",
                zorder=2,
            )
            tail = pts[-3:] if len(pts) >= 3 else pts[-2:]
            ax.plot(
                tail[:, 0],
                tail[:, 1],
                color=color,
                linewidth=3.0,
                alpha=1.0,
                solid_capstyle="round",
                zorder=3,
            )
            if np.linalg.norm(pts[-1] - pts[-2]) > 0:
                ax.annotate(
                    "",
                    xy=(pts[-1, 0], pts[-1, 1]),
                    xytext=(pts[-2, 0], pts[-2, 1]),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": color,
                        "linewidth": 1.5,
                        "shrinkA": 0,
                        "shrinkB": 0,
                        "mutation_scale": 10,
                    },
                    zorder=4,
                )

        mid_idx = max(1, len(pts) // 2)
        ax.plot(
            pts[mid_idx, 0],
            pts[mid_idx, 1],
            marker=marker,
            linestyle="none",
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.1,
            zorder=5,
        )
        ax.plot(
            pts[-1, 0],
            pts[-1, 1],
            marker=marker,
            color=color,
            linestyle="none",
            markersize=7.6,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=6,
        )

    origin = projected[conditions[0]][0]
    ax.plot(
        origin[0],
        origin[1],
        marker="o",
        color="black",
        linestyle="none",
        markersize=5.4,
        markeredgecolor="white",
        markeredgewidth=0.9,
        zorder=6,
    )
    ax.annotate(
        r"$x_T$",
        xy=(origin[0], origin[1]),
        fontsize=8.0,
        fontweight="bold",
        textcoords="offset points",
        xytext=(-11, -10),
        bbox={
            "boxstyle": "round,pad=0.14",
            "facecolor": "white",
            "edgecolor": "#B8C2CC",
            "linewidth": 0.8,
            "alpha": 0.94,
        },
        zorder=7,
    )

    _style_manifold_axis(ax, method)
    _annotate_endpoints(ax, conditions, projected, condition_labels)


def _draw_decoded_axis(ax: plt.Axes, image_path: Path | None, cond_key: str) -> None:
    color = CONDITION_COLORS[cond_key]
    label = CONDITION_LABELS[cond_key]

    if image_path is None:
        _draw_missing(ax, f"{label}\nmissing")
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)
            spine.set_color(color)
    else:
        image = _load_image(image_path)
        if image is None:
            _draw_missing(ax, image_path.name)
            for spine in ax.spines.values():
                spine.set_linewidth(1.4)
                spine.set_color(color)
        else:
            ax.imshow(image)
            _style_decoded_axis(ax, color)

    title = ax.set_title(label, fontsize=8.0, pad=5, fontweight="bold", color=color)
    title.set_bbox(
        {
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 0.9,
            "alpha": 0.96,
        }
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
        decoded_grid = inner[1, 0].subgridspec(1, len(CONDITION_ORDER), wspace=0.08)

        trajectory_ax.set_title(
            f"{panel.seed_title}\n({panel.pair_title})",
            fontsize=10,
            fontweight="bold",
            pad=10,
        )
        projected = {
            cond: np.asarray(panel.projected_paths[cond], dtype=np.float32)
            for cond in CONDITION_ORDER
            if cond in panel.projected_paths
        }
        if len(projected) == len(CONDITION_ORDER):
            _section_header(trajectory_ax, "Shared-noise manifold", y=1.02)
            _plot_manifold(
                trajectory_ax,
                projected,
                CONDITION_ORDER,
                panel.projection_method,
                panel.condition_labels or CONDITION_LABELS,
            )
        else:
            _draw_missing(trajectory_ax, "trajectory_projection")

        decoded_axes: list[plt.Axes] = []
        for cond_idx, cond_key in enumerate(CONDITION_ORDER):
            decoded_ax = fig.add_subplot(decoded_grid[0, cond_idx])
            rel_path = panel.decoded_paths.get(cond_key)
            image_path = panel.pair_dir / rel_path if rel_path is not None else None
            _draw_decoded_axis(decoded_ax, image_path, cond_key)
            decoded_axes.append(decoded_ax)

        decoded_axes[0].text(
            -0.02,
            1.14,
            "Decoded endpoints",
            transform=decoded_axes[0].transAxes,
            ha="left",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
            color="#3B4252",
        )

    fig.suptitle(
        "Multi-seed trajectory overview: trajectory manifolds and decoded endpoints",
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
            "Render a nested 2x2 multi-seed figure where each seed cell contains "
            "trajectory_manifold.png and decoded_images.png."
        )
    )
    parser.add_argument(
        "--pair-dirs",
        nargs=4,
        metavar="DIR",
        required=True,
        help=(
            "Four pair directories in reading order (S0 S1 S2 S3), one per seed. "
            "Each must contain trajectory_manifold.png and decoded_images.png."
        ),
    )
    parser.add_argument(
        "--output",
        default="multiseed_grid.png",
        help="Output image path (default: multiseed_grid.png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output DPI (default: 180)",
    )
    args = parser.parse_args()

    pair_dirs = [Path(d) for d in args.pair_dirs]

    if len(pair_dirs) != 4:
        parser.error(f"Expected exactly 4 seed directories, got {len(pair_dirs)}")

    # Validate all directories exist
    for pair_dir in pair_dirs:
        if not pair_dir.exists():
            parser.error(f"Pair directory does not exist: {pair_dir}")

    panel_specs = resolve_panel_specs(pair_dirs)

    for panel in panel_specs:
        print(f"Using {panel.seed_title}: {panel.pair_dir}")

    make_figure(panel_specs, Path(args.output), dpi=args.dpi)


if __name__ == "__main__":
    main()
