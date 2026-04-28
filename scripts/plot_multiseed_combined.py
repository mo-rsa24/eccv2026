"""
Multi-seed combined trajectory and decoded endpoints visualization.

Creates a figure with:
- Top: one large shared-noise manifold plot with all seeds overlaid
- Bottom: tightly stacked 4x4 grid of decoded endpoint images
  (rows=seeds, columns=A/B/A∧B/PoE)

Expected directory layout:
    experiments/eccv2026/multiseed/<pair>/seed_<N>/<group>/<pair>/

Usage
-----
    python scripts/plot_multiseed_combined.py \\
        --pair-dirs \\
            experiments/eccv2026/multiseed/butterfly_x_flower/seed_0/group1_cooccurrence/a_butterfly__x__a_flower_meadow \\
            experiments/eccv2026/multiseed/butterfly_x_flower/seed_7/group1_cooccurrence/a_butterfly__x__a_flower_meadow \\
            experiments/eccv2026/multiseed/butterfly_x_flower/seed_42/group1_cooccurrence/a_butterfly__x__a_flower_meadow \\
            experiments/eccv2026/multiseed/butterfly_x_flower/seed_99/group1_cooccurrence/a_butterfly__x__a_flower_meadow
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from plot_multiseed_trajectory_grid import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_MARKERS,
    CONDITION_ORDER,
    PanelSpec,
    _load_image,
    _draw_decoded_axis,
    _section_header,
    _style_manifold_axis,
    resolve_panel_specs,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEED_ALPHAS = [0.42, 0.58, 0.74, 0.9]
REQUIRED_CONDITIONS = tuple(CONDITION_ORDER)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _pair_slug_from_assets(pair_dir: Path) -> str:
    assets = _load_json(pair_dir / "grid_assets.json") or {}
    slug = str(assets.get("pair_slug") or "").strip()
    return slug if slug else pair_dir.name


def _materialize_multiseed_tree(
    pair_dirs: list[Path],
    root_dir: Path,
) -> tuple[list[Path], Path]:
    """Create a butterfly-style multiseed/<pair_slug>/seed_*/... tree with symlinks.

    Returns the rewritten pair_dirs pointing at the materialized tree and
    the pair root directory.
    """
    if not pair_dirs:
        raise ValueError("No pair dirs provided")

    pair_slug = _pair_slug_from_assets(pair_dirs[0])
    pair_root = root_dir / pair_slug
    pair_root.mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, str]] = []
    rewritten: list[Path] = []

    for pair_dir in pair_dirs:
        seed_number = 0
        for ancestor in pair_dir.parents:
            name = ancestor.name
            if not name.startswith("seed_"):
                continue
            try:
                seed_number = int(name.split("_", maxsplit=1)[1])
                break
            except (ValueError, IndexError):
                continue

        group_key = pair_dir.parent.name
        dest_parent = pair_root / f"seed_{seed_number}" / group_key
        dest_parent.mkdir(parents=True, exist_ok=True)
        dest_leaf = dest_parent / pair_slug

        resolved_src = pair_dir.resolve()
        if dest_leaf.is_symlink():
            if dest_leaf.resolve() != resolved_src:
                dest_leaf.unlink()
                dest_leaf.symlink_to(resolved_src, target_is_directory=True)
            rewritten.append(dest_leaf)
        elif dest_leaf.exists():
            if dest_leaf.resolve() != resolved_src:
                raise ValueError(
                    f"Materialized destination already exists and is not the requested source: {dest_leaf}"
                )
            rewritten.append(dest_leaf)
        else:
            dest_leaf.symlink_to(resolved_src, target_is_directory=True)
            rewritten.append(dest_leaf)

        sources.append(
            {
                "seed": str(seed_number),
                "group": group_key,
                "pair_dir": str(resolved_src),
                "materialized": str(dest_leaf),
            }
        )

    (pair_root / "sources.json").write_text(json.dumps({"pair_slug": pair_slug, "sources": sources}, indent=2) + "\n")
    return rewritten, pair_root


def _resolve_manifold_conditions(panels: list[PanelSpec]) -> list[str]:
    del panels
    return list(REQUIRED_CONDITIONS)


def _maybe_recompute_joint_projection(panels: list[PanelSpec]) -> None:
    """Recompute a single shared projection from flattened trajectories when present.

    Some directories (e.g., SD-IPC enrichment outputs) store per-seed projections
    that are not comparable across seeds. When flattened trajectories exist,
    we can project all (seed, condition) trajectories jointly to get a consistent
    2D coordinate system for the shared manifold.
    """
    series: list[tuple[int, str, np.ndarray]] = []
    projection_method = (panels[0].projection_method if panels else "mds") or "mds"

    for panel in panels:
        asset = _load_json(panel.pair_dir / "grid_assets.json") or {}
        projection_method = str(asset.get("projection_method") or projection_method or "mds")
        flat_map = asset.get("trajectory_flat_paths") or {}
        if not isinstance(flat_map, dict):
            continue
        for cond, rel in flat_map.items():
            if cond not in REQUIRED_CONDITIONS:
                continue
            if not rel:
                continue
            try:
                arr = np.load(panel.pair_dir / str(rel))
            except Exception:
                continue
            if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[0] < 2:
                continue
            series.append((panel.seed_number, str(cond), arr.astype(np.float32, copy=False)))

    if not series:
        return

    series.sort(key=lambda item: (item[0], item[1]))
    min_steps = min(arr.shape[0] for _, _, arr in series)
    max_dim = max(arr.shape[1] for _, _, arr in series)

    stacked_parts: list[np.ndarray] = []
    for _, _, arr in series:
        clipped = arr[:min_steps]
        if clipped.shape[1] < max_dim:
            pad = np.zeros((clipped.shape[0], max_dim - clipped.shape[1]), dtype=np.float32)
            clipped = np.concatenate([clipped, pad], axis=1)
        stacked_parts.append(clipped)
    stacked = np.vstack(stacked_parts)

    if projection_method == "pca":
        try:
            from sklearn.decomposition import PCA  # type: ignore
        except ImportError:
            return
        proj = PCA(n_components=2).fit_transform(stacked)
    else:
        try:
            from sklearn.manifold import MDS  # type: ignore
            from sklearn.metrics import pairwise_distances  # type: ignore
        except ImportError:
            return
        dist = pairwise_distances(stacked, metric="euclidean")
        proj = MDS(
            n_components=2,
            random_state=42,
            dissimilarity="precomputed",
            normalized_stress="auto",
        ).fit_transform(dist)

    start = 0
    for seed_number, cond, _ in series:
        end = start + min_steps
        pts = proj[start:end].astype(np.float32)
        for panel in panels:
            if panel.seed_number == seed_number:
                panel.projected_paths[cond] = pts.tolist()
                panel.projection_method = projection_method
                break
        start = end


def _validate_required_conditions(panels: list[PanelSpec]) -> None:
    missing_messages: list[str] = []
    for panel in panels:
        missing_projected = [cond for cond in REQUIRED_CONDITIONS if cond not in panel.projected_paths]
        missing_decoded = [cond for cond in REQUIRED_CONDITIONS if cond not in panel.decoded_paths]
        if missing_projected or missing_decoded:
            details: list[str] = []
            if missing_projected:
                details.append(f"missing trajectories={missing_projected}")
            if missing_decoded:
                details.append(f"missing decoded={missing_decoded}")
            missing_messages.append(f"{panel.pair_dir}: {', '.join(details)}")

    if missing_messages:
        joined = "\n  ".join(missing_messages)
        raise ValueError(
            "plot_multiseed_combined.py requires canonical A/B/A∧B/PoE assets for every seed.\n"
            f"  {joined}\n"
            "Use scripts/run_taxonomy_qualitative_sdxl.py to generate per-seed multiseed directories from scratch."
        )


def _plot_seed_trajectory(
    ax: plt.Axes,
    pts: np.ndarray,
    cond: str,
    alpha: float,
) -> None:
    color = CONDITION_COLORS.get(cond, "#4C566A")
    marker = CONDITION_MARKERS.get(cond, "o")

    if len(pts) >= 2:
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=color,
            linewidth=2.0,
            alpha=alpha,
            solid_capstyle="round",
            zorder=2,
        )
        tail = pts[-3:] if len(pts) >= 3 else pts[-2:]
        ax.plot(
            tail[:, 0],
            tail[:, 1],
            color=color,
            linewidth=2.8,
            alpha=min(1.0, alpha + 0.08),
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
                    "linewidth": 1.3,
                    "alpha": alpha,
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "mutation_scale": 9,
                },
                zorder=4,
            )

    mid_idx = max(1, len(pts) // 2)
    ax.plot(
        pts[mid_idx, 0],
        pts[mid_idx, 1],
        marker=marker,
        linestyle="none",
        markersize=4.3,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.0,
        alpha=alpha,
        zorder=5,
    )
    ax.plot(
        pts[-1, 0],
        pts[-1, 1],
        marker=marker,
        color=color,
        linestyle="none",
        markersize=6.4,
        markeredgecolor="white",
        markeredgewidth=0.8,
        alpha=alpha,
        zorder=6,
    )


def _plot_combined_manifold(ax: plt.Axes, panels: list[PanelSpec]) -> None:
    _section_header(ax, "Shared-noise manifold", y=1.02)

    manifold_conditions = _resolve_manifold_conditions(panels)

    all_points: list[np.ndarray] = []
    for seed_idx, panel in enumerate(panels):
        alpha = SEED_ALPHAS[min(seed_idx, len(SEED_ALPHAS) - 1)]
        for cond in manifold_conditions:
            pts_raw = panel.projected_paths.get(cond)
            if pts_raw is None:
                continue
            pts = np.asarray(pts_raw, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2:
                continue
            all_points.append(pts)
            _plot_seed_trajectory(ax, pts, cond, alpha=alpha)

            if cond == "prompt_a":
                origin = pts[0]
                ax.plot(
                    origin[0],
                    origin[1],
                    marker="o",
                    color="black",
                    linestyle="none",
                    markersize=4.6,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    alpha=max(0.5, alpha),
                    zorder=6,
                )

    if panels and panels[0].projected_paths.get("prompt_a") is not None:
        origin = np.asarray(panels[0].projected_paths["prompt_a"], dtype=np.float32)[0]
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

    method = panels[0].projection_method if panels else "mds"
    _style_manifold_axis(ax, method)

    legend_lines = []
    legend_labels = []
    label_overrides = panels[0].condition_labels if panels else {}
    for cond in manifold_conditions:
        legend_lines.append(
            plt.Line2D(
                [0],
                [0],
                color=CONDITION_COLORS.get(cond, "#4C566A"),
                marker=CONDITION_MARKERS.get(cond, "o"),
                markersize=6,
                linewidth=2.2,
                markeredgecolor="white",
                markeredgewidth=0.8,
            )
        )
        legend_labels.append(label_overrides.get(cond, CONDITION_LABELS.get(cond, cond)))
    if legend_lines:
        ax.legend(
            legend_lines,
            legend_labels,
            loc="upper right",
            fontsize=8.5,
            framealpha=0.96,
            edgecolor="#D7DEE8",
        )


def _overlay_box_size_pixels(image: np.ndarray, zoom: float, pad_px: float = 16.0) -> tuple[float, float]:
    height, width = image.shape[:2]
    return width * zoom + pad_px, height * zoom + pad_px


def _boxes_overlap(
    center_a: np.ndarray,
    size_a: tuple[float, float],
    center_b: np.ndarray,
    size_b: tuple[float, float],
    pad: float = 10.0,
) -> bool:
    return (
        abs(center_a[0] - center_b[0]) < 0.5 * (size_a[0] + size_b[0]) + pad
        and abs(center_a[1] - center_b[1]) < 0.5 * (size_a[1] + size_b[1]) + pad
    )


def _resolve_overlay_offsets(
    offsets: list[np.ndarray],
    anchor_points: list[np.ndarray],
    box_sizes: list[tuple[float, float]],
) -> list[np.ndarray]:
    resolved = [offset.copy() for offset in offsets]
    if not resolved:
        return resolved

    min_separation = 36.0
    radial_push = 18.0

    for _ in range(80):
        moved = False
        centers = [anchor + offset for anchor, offset in zip(anchor_points, resolved)]

        for idx in range(len(resolved)):
            for jdx in range(idx + 1, len(resolved)):
                if not _boxes_overlap(centers[idx], box_sizes[idx], centers[jdx], box_sizes[jdx]):
                    continue

                delta = centers[jdx] - centers[idx]
                distance = float(np.linalg.norm(delta))
                if distance < 1e-6:
                    delta = np.array([1.0, 0.0], dtype=np.float32)
                    distance = 1.0
                direction = delta / distance
                shift = direction * max(min_separation, 0.5 * (min_separation - distance) + 18.0)

                resolved[idx] -= shift
                resolved[jdx] += shift
                moved = True

            offset_norm = float(np.linalg.norm(resolved[idx]))
            if offset_norm < 1e-6:
                resolved[idx] = np.array([96.0, 0.0], dtype=np.float32)
            else:
                resolved[idx] += (resolved[idx] / offset_norm) * radial_push

        if not moved:
            break

    return resolved


def _add_terminal_image_overlays(ax: plt.Axes, panels: list[PanelSpec]) -> None:
    entries: list[dict[str, object]] = []
    terminal_points: list[np.ndarray] = []

    manifold_conditions = _resolve_manifold_conditions(panels)

    for panel in panels:
        for cond in manifold_conditions:
            pts_raw = panel.projected_paths.get(cond)
            rel_path = panel.decoded_paths.get(cond)
            if pts_raw is None or rel_path is None:
                continue
            pts = np.asarray(pts_raw, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
                continue

            image = _load_image(panel.pair_dir / rel_path)
            if image is None:
                continue

            terminal = pts[-1]
            entries.append(
                {
                    "terminal": terminal,
                    "image": image,
                    "color": CONDITION_COLORS.get(cond, "#4C566A"),
                    "seed_number": panel.seed_number,
                    "cond": cond,
                }
            )
            terminal_points.append(np.asarray(terminal, dtype=np.float32))

    if not entries:
        return

    terminals_arr = np.vstack(terminal_points)
    centroid = terminals_arr.mean(axis=0)
    data_span = max(float(np.ptp(terminals_arr[:, 0])), float(np.ptp(terminals_arr[:, 1])), 1e-3)
    image_zoom = 0.055 if len(entries) <= 8 else 0.045
    base_offset_px = max(110.0, min(170.0, 120.0 + 20.0 / data_span))

    fig = ax.figure
    fig.canvas.draw()
    anchor_points: list[np.ndarray] = []
    initial_offsets: list[np.ndarray] = []
    box_sizes: list[tuple[float, float]] = []

    for entry in entries:
        terminal = np.asarray(entry["terminal"], dtype=np.float32)
        terminal_disp = ax.transData.transform(terminal)
        anchor_points.append(np.asarray(terminal_disp, dtype=np.float32))

        direction = terminal - centroid
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            angle = (len(initial_offsets) / max(1, len(entries))) * 2.0 * np.pi
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        else:
            direction = direction / norm

        display_direction = ax.transData.transform(terminal + direction) - terminal_disp
        display_norm = float(np.linalg.norm(display_direction))
        if display_norm < 1e-6:
            display_direction = np.array([1.0, 0.0], dtype=np.float32)
            display_norm = 1.0
        display_direction = display_direction / display_norm

        initial_offsets.append(display_direction * base_offset_px)
        box_sizes.append(_overlay_box_size_pixels(np.asarray(entry["image"]), image_zoom))

    resolved_offsets = _resolve_overlay_offsets(initial_offsets, anchor_points, box_sizes)

    for entry, offset in zip(entries, resolved_offsets):
        image = entry["image"]
        color = str(entry["color"])
        terminal = np.asarray(entry["terminal"], dtype=np.float32)
        seed_number = int(entry["seed_number"])
        cond = str(entry["cond"])

        artist = AnnotationBbox(
            OffsetImage(image, zoom=image_zoom),
            (terminal[0], terminal[1]),
            xybox=(float(offset[0]), float(offset[1])),
            xycoords="data",
            boxcoords="offset pixels",
            frameon=True,
            pad=0.18,
            bboxprops={
                "edgecolor": color,
                "linewidth": 1.8,
                "facecolor": "white",
                "alpha": 0.98,
            },
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 1.1,
                "color": color,
                "alpha": 0.85,
                "shrinkA": 3,
                "shrinkB": 3,
            },
            zorder=8,
        )
        ax.add_artist(artist)
        ax.annotate(
            f"S{seed_number} {CONDITION_LABELS.get(cond, cond)}",
            xy=(terminal[0], terminal[1]),
            xytext=(float(offset[0]), float(offset[1] - 10.0)),
            xycoords="data",
            textcoords="offset pixels",
            ha="center",
            va="top",
            fontsize=6.6,
            fontweight="bold",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.8,
                "alpha": 0.92,
            },
            zorder=9,
        )


def make_combined_figure(
    panels: list[PanelSpec],
    output_path: Path,
    dpi: int,
    pair_name: str = "",
) -> None:
    if len(panels) != 4:
        raise ValueError(f"Expected exactly 4 panels, got {len(panels)}")

    fig = plt.figure(figsize=(12.8, 13.8))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.3, 0.95],
        left=0.05,
        right=0.985,
        top=0.94,
        bottom=0.045,
        hspace=0.08,
    )

    manifold_ax = fig.add_subplot(outer[0, 0])
    pair_title = pair_name.strip() or (panels[0].pair_title if panels else "")
    manifold_ax.set_title(
        f"({pair_title})",
        fontsize=10.8,
        fontweight="bold",
        pad=6,
    )
    _plot_combined_manifold(manifold_ax, panels)

    decoded_band = outer[1, 0].subgridspec(
        1,
        3,
        width_ratios=[0.14, 0.72, 0.14],
        wspace=0.0,
    )
    decoded_grid = decoded_band[0, 1].subgridspec(
        len(panels),
        len(CONDITION_ORDER),
        hspace=0.01,
        wspace=0.01,
    )

    for row_idx, panel in enumerate(panels):
        for col_idx, cond in enumerate(CONDITION_ORDER):
            ax = fig.add_subplot(decoded_grid[row_idx, col_idx])
            rel_path = panel.decoded_paths.get(cond)
            image_path = panel.pair_dir / rel_path if rel_path is not None else None
            _draw_decoded_axis(ax, image_path, cond)
            ax.set_box_aspect(1.0)

            if col_idx == 0:
                ax.text(
                    -0.06,
                    0.5,
                    f"S{panel.seed_number}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8.6,
                    fontweight="bold",
                    color="#2A2F36",
                )

    fig.text(
        0.105,
        0.355,
        "Decoded endpoints",
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color="#2A2F36",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "#F4F6F8",
            "edgecolor": "#D7DEE8",
            "linewidth": 0.8,
            "alpha": 0.98,
        },
    )

    fig.suptitle(
        "Multi-seed trajectory overview",
        fontsize=13.6,
        fontweight="bold",
        y=0.972,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {output_path}")


def make_terminal_overlay_figure(
    panels: list[PanelSpec],
    output_path: Path,
    dpi: int,
    pair_name: str = "",
) -> None:
    if len(panels) != 4:
        raise ValueError(f"Expected exactly 4 panels, got {len(panels)}")

    fig, manifold_ax = plt.subplots(figsize=(13.8, 10.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.08)

    pair_title = pair_name.strip() or (panels[0].pair_title if panels else "")
    manifold_ax.set_title(
        f"({pair_title})",
        fontsize=10.8,
        fontweight="bold",
        pad=6,
    )
    _plot_combined_manifold(manifold_ax, panels)
    _add_terminal_image_overlays(manifold_ax, panels)

    fig.suptitle(
        "Multi-seed trajectory overview with decoded endpoints",
        fontsize=13.6,
        fontweight="bold",
        y=0.972,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a combined multi-seed figure with one shared manifold and a tight 4x4 decoded grid."
    )
    parser.add_argument(
        "--pair-dirs",
        nargs=4,
        metavar="DIR",
        required=True,
        help="Exactly 4 pair directories (one per seed).",
    )
    parser.add_argument(
        "--output",
        default="multiseed_combined.png",
        help="Output image path (default: multiseed_combined.png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output DPI (default: 180)",
    )
    parser.add_argument(
        "--pair-name",
        default="",
        help="Optional pair title override shown above the manifold.",
    )
    parser.add_argument(
        "--overlay-output",
        default="",
        help="Optional second output path for the terminal-overlay figure (default: derived from --output).",
    )
    parser.add_argument(
        "--materialize-root",
        default="",
        help="Optional root directory to create a multiseed/<pair_slug>/seed_*/... symlink tree and save outputs there.",
    )
    args = parser.parse_args()

    pair_dirs = [Path(d) for d in args.pair_dirs]
    for pair_dir in pair_dirs:
        if not pair_dir.exists():
            parser.error(f"Pair directory does not exist: {pair_dir}")

    if args.materialize_root:
        pair_dirs, pair_root = _materialize_multiseed_tree(pair_dirs, Path(args.materialize_root))
        if args.output == "multiseed_combined.png":
            args.output = str(pair_root / "multiseed_combined.png")
        if not args.overlay_output:
            args.overlay_output = str(pair_root / "multiseed_combined_terminal_overlays.png")

    panels = resolve_panel_specs(pair_dirs)
    _maybe_recompute_joint_projection(panels)
    _validate_required_conditions(panels)
    for panel in panels:
        print(f"Using {panel.seed_title}: {panel.pair_dir}")

    make_combined_figure(
        panels,
        Path(args.output),
        dpi=args.dpi,
        pair_name=args.pair_name,
    )

    overlay_output = Path(args.overlay_output) if args.overlay_output else Path(args.output).with_name(
        f"{Path(args.output).stem}_terminal_overlays{Path(args.output).suffix or '.png'}"
    )
    make_terminal_overlay_figure(
        panels,
        overlay_output,
        dpi=args.dpi,
        pair_name=args.pair_name,
    )


if __name__ == "__main__":
    main()
