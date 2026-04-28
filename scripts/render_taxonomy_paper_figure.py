#!/usr/bin/env python3
"""Render paper-facing taxonomy composite figures from SDXL grid assets."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from taxonomy_manifest import (
        GROUP_DIR_ALIASES,
        GROUP_SPECS,
        REPRESENTATIVE_PAIR_SLUGS,
        GROUP3_SUBGROUP_SPECS,
        GROUP3_REPRESENTATIVE_PAIR_SLUGS,
        get_pair_taxonomy_from_slug,
    )
except ImportError:
    from scripts.taxonomy_manifest import (
        GROUP_DIR_ALIASES,
        GROUP_SPECS,
        REPRESENTATIVE_PAIR_SLUGS,
        GROUP3_SUBGROUP_SPECS,
        GROUP3_REPRESENTATIVE_PAIR_SLUGS,
        get_pair_taxonomy_from_slug,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "experiments" / "inversion" / "gap_analysis"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "paper"
    / "neurips"
    / "Comparing Semantic and Logical Composition Using Latent Diffusion Models"
    / "figures"
)

GROUP_LABELS = [spec["label"] for spec in GROUP_SPECS]
DEFAULT_PAIR_SLUGS = list(REPRESENTATIVE_PAIR_SLUGS)
DEFAULT_GROUP3_PAIR_SLUGS = list(GROUP3_REPRESENTATIVE_PAIR_SLUGS)
GROUP3_SUBGROUP_LABELS = [spec["label"] for spec in GROUP3_SUBGROUP_SPECS]
GROUP3_FAILURE_PATTERNS = [spec["failure_pattern"] for spec in GROUP3_SUBGROUP_SPECS]

# Legacy extreme-case view: G1 versus coherent collision (now G6)
DEFAULT_1A_PAIR_SLUGS = [DEFAULT_PAIR_SLUGS[0], DEFAULT_PAIR_SLUGS[-1]]

MODE_CONDITIONS = {
    "figure1": ["prompt_a", "prompt_b", "monolithic", "poe"],
    "figure1_endpoints": ["prompt_a", "prompt_b", "monolithic", "poe"],
    "figure2": ["prompt_a", "prompt_b", "monolithic", "poe", "pstar_sdipc"],
    "figure3": ["prompt_a", "prompt_b", "monolithic", "poe", "co3", "pstar_sdipc", "pstar_co3_sdipc"],
    "figure_group3": ["prompt_a", "prompt_b", "monolithic", "poe"],
    "figure1a": ["prompt_a", "prompt_b", "monolithic", "poe"],
    "figure2a": ["prompt_a", "prompt_b", "monolithic", "poe", "pstar_sdipc"],
}

MODE_OUTPUTS = {
    "figure1": DEFAULT_OUTPUT_DIR / "trajectory_3x2.png",
    "figure1_endpoints": DEFAULT_OUTPUT_DIR / "representative_endpoints_3x2.png",
    "figure2": DEFAULT_OUTPUT_DIR / "trajectory_3x2_sdipc.png",
    "figure3": DEFAULT_OUTPUT_DIR / "trajectory_3x2_co3.png",
    "figure_group3": DEFAULT_OUTPUT_DIR / "group3_subgroup_qualitative.png",
    "figure1a": DEFAULT_OUTPUT_DIR / "trajectory_g1g4.png",
    "figure2a": DEFAULT_OUTPUT_DIR / "trajectory_g1g4_sdipc.png",
    "figure_seed_sheet": DEFAULT_OUTPUT_DIR / "representative_seed_sheet_poe.png",
}

MODE_TITLES = {
    "figure1": "Six-group taxonomy overview: semantic composition and PoE",
    "figure1_endpoints": "Six-group representative decoded endpoints",
    "figure2": "Six-group taxonomy overview: semantic composition, PoE, and PoE p*",
    "figure3": "Six-group taxonomy overview: semantic composition, PoE, CO3, and p* reruns",
    "figure_group3": "Legacy Group 3 subgroup panel",
    "figure1a": "Legacy extreme-case trajectories: co-occurrence (G1) vs coherent collision (G6)",
    "figure2a": "Legacy reachability view: co-occurrence (G1) vs coherent collision (G6)",
    "figure_seed_sheet": "Representative seed sheet",
}

COND_LABELS = {
    "prompt_a": "A",
    "prompt_b": "B",
    "monolithic": "A∧B",
    "poe": "PoE",
    "co3": "CO3",
    "pstar_sdipc": "PoE p*",
    "pstar_co3_sdipc": "CO3 p*",
    "pstar_inv": "p* inv",
}

COND_COLORS = {
    "prompt_a": "#C84C5B",
    "prompt_b": "#2B6F97",
    "monolithic": "#3B8D5B",
    "poe": "#D9872B",
    "co3": "#5B8E7D",
    "pstar_sdipc": "#7A5CBA",
    "pstar_co3_sdipc": "#4D9E8E",
    "pstar_inv": "#6D597A",
}

COND_MARKERS = {
    "prompt_a": "o",
    "prompt_b": "s",
    "monolithic": "D",
    "poe": "^",
    "co3": "P",
    "pstar_sdipc": "v",
    "pstar_co3_sdipc": "h",
    "pstar_inv": "X",
}

LABEL_OFFSETS = {
    "prompt_a": (10, 10),
    "prompt_b": (10, -12),
    "monolithic": (-16, 12),
    "poe": (-18, -12),
    "co3": (12, 14),
    "pstar_sdipc": (12, -16),
    "pstar_co3_sdipc": (-18, 14),
    "pstar_inv": (-18, 0),
}

SECTION_HEADER_BBOX = {
    "boxstyle": "round,pad=0.22",
    "facecolor": "#F4F6F8",
    "edgecolor": "#D7DEE8",
    "linewidth": 0.8,
    "alpha": 0.98,
}

QUALITATIVE_GROUP_DIR_ALIASES = GROUP_DIR_ALIASES


@dataclass
class PanelAsset:
    pair_dir: Path
    group_title: str
    pair_title: str
    projection_method: str
    prompt_key_map: dict
    condition_labels: dict
    decoded_paths: dict
    flat_paths: dict
    projected_paths: dict
    projected_labels: dict
    manifold_image_path: str | None = None


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
        "pstar_sdipc": ["pstar_sdipc.png", "images/sdxl_pstar_sdipc_*.png", "images/sd14_pstar_sdipc_*.png", "images/sd35_pstar_sdipc_*.png"],
    }
    for cond, patterns in candidates.items():
        rel_path = _first_matching_image(pair_dir, patterns)
        if rel_path is not None:
            decoded_paths[cond] = rel_path
    return decoded_paths


def _infer_manifold_image_path(pair_dir: Path) -> str | None:
    return _first_matching_image(
        pair_dir,
        [
            "trajectory_manifold.png",
            "trajectories_mds.png",
            "trajectories_pca.png",
        ],
    )


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc


def _pair_slug_from_values(pair_values: tuple[str, str] | list[str]) -> str:
    prompt_a, prompt_b = pair_values
    clean_a = str(prompt_a).lower().replace(" ", "_").replace("'", "")
    clean_b = str(prompt_b).lower().replace(" ", "_").replace("'", "")
    return f"{clean_a}__x__{clean_b}"


def _resolve_default_pair_values(data_dir: Path, mode: str) -> list[str]:
    manifest_path = data_dir / "sdxl_qualitative_run_manifest.json"
    if not manifest_path.exists():
        if mode == "figure_group3":
            return list(DEFAULT_GROUP3_PAIR_SLUGS)
        if mode in {"figure1a", "figure2a"}:
            return list(DEFAULT_1A_PAIR_SLUGS)
        return list(DEFAULT_PAIR_SLUGS)

    manifest = _load_json(manifest_path)
    selected_pairs = list(manifest.get("selected_pairs") or [])
    if not selected_pairs:
        if mode == "figure_group3":
            return list(DEFAULT_GROUP3_PAIR_SLUGS)
        if mode in {"figure1a", "figure2a"}:
            return list(DEFAULT_1A_PAIR_SLUGS)
        return list(DEFAULT_PAIR_SLUGS)

    by_group: dict[str, list[str]] = {}
    for row in selected_pairs:
        group_key = str(row.get("taxonomy_group_key") or "")
        slug = str(row.get("qualitative_pair_slug") or row.get("pair_slug") or "")
        if not group_key or not slug:
            continue
        by_group.setdefault(group_key, []).append(slug)

    if mode == "figure_group3":
        group3_key = "group3_role_separable_object_scene"
        available = by_group.get(group3_key, [])
        if len(available) >= 3:
            return available[:3]
        return list(DEFAULT_GROUP3_PAIR_SLUGS)

    resolved: list[str] = []
    for spec in GROUP_SPECS:
        group_key = str(spec["key"])
        available = by_group.get(group_key, [])
        if not available:
            continue
        representative_slug = _pair_slug_from_values(tuple(spec["representative_pair"]))
        resolved.append(representative_slug if representative_slug in set(available) else available[0])

    if mode in {"figure1a", "figure2a"}:
        if len(resolved) >= 2:
            return [resolved[0], resolved[-1]]
        return list(DEFAULT_1A_PAIR_SLUGS)

    return resolved if len(resolved) == 6 else list(DEFAULT_PAIR_SLUGS)


def _humanize_pair(pair_values: list[str] | None, fallback_slug: str) -> str:
    if pair_values and len(pair_values) == 2:
        return f"{pair_values[0]} x {pair_values[1]}"
    return fallback_slug.replace("_", " ")


def _resolve_pair_dir(data_dir: Path, pair_value: str, seed: int | None = None) -> Path:
    candidate = Path(pair_value)
    if candidate.is_absolute():
        return candidate

    pairs_root = data_dir / "pairs"
    if (pairs_root / pair_value).exists():
        return pairs_root / pair_value
    if (data_dir / pair_value).exists():
        return data_dir / pair_value
    meta = get_pair_taxonomy_from_slug(pair_value)
    if meta is not None:
        group_key = meta.get("taxonomy_group_key")
        qualitative_slug = meta.get("qualitative_pair_slug")
        if group_key and qualitative_slug:
            search_roots = [data_dir]
            if seed is not None:
                search_roots = [data_dir / f"seed_{seed}", data_dir]
            for search_root in search_roots:
                for group_dir in QUALITATIVE_GROUP_DIR_ALIASES.get(group_key, [group_key]):
                    candidate_dir = search_root / group_dir / qualitative_slug
                    if candidate_dir.exists():
                        return candidate_dir
    return pairs_root / pair_value


def _apply_monolithic_baseline(asset: dict, mode: str) -> dict:
    if mode not in {"naive", "natural"}:
        return asset

    src_key = f"monolithic_{mode}"
    for map_key in ("decoded_image_paths", "trajectory_flat_paths"):
        mapping = asset.get(map_key)
        if isinstance(mapping, dict) and src_key in mapping:
            mapping["monolithic"] = mapping[src_key]

    traj = asset.get("trajectory_projection")
    if isinstance(traj, dict):
        projected = traj.get("projected")
        if isinstance(projected, dict) and src_key in projected:
            projected["monolithic"] = projected[src_key]
        labels = traj.get("labels")
        if isinstance(labels, dict) and src_key in labels:
            labels["monolithic"] = labels[src_key]

    return asset


def _load_panel_asset(
    data_dir: Path,
    pair_value: str,
    group_idx: int,
    monolithic_baseline: str,
    seed: int | None,
) -> PanelAsset:
    pair_dir = _resolve_pair_dir(data_dir, pair_value, seed=seed)
    asset_path = pair_dir / "grid_assets.json"
    if not asset_path.exists():
        decoded_paths = _infer_decoded_paths(pair_dir)
        if decoded_paths:
            meta = get_pair_taxonomy_from_slug(pair_dir.name)
            pair_values = None
            group_title = GROUP_LABELS[group_idx]
            if meta is not None:
                pair_values = [meta["prompt_a"], meta["prompt_b"]]
                group_title = str(meta.get("taxonomy_group_label") or group_title)
            return PanelAsset(
                pair_dir=pair_dir,
                group_title=group_title,
                pair_title=_humanize_pair(pair_values, pair_dir.name),
                projection_method="none",
                prompt_key_map={},
                condition_labels={},
                decoded_paths=decoded_paths,
                flat_paths={},
                projected_paths={},
                projected_labels={},
                manifold_image_path=_infer_manifold_image_path(pair_dir),
            )
        raise FileNotFoundError(f"Missing grid asset: {asset_path}")

    asset = _apply_monolithic_baseline(_load_json(asset_path), monolithic_baseline)
    traj = asset.get("trajectory_projection") or {}
    group_title = str(asset.get("taxonomy_group_label") or GROUP_LABELS[group_idx])
    decoded_paths = dict(asset.get("decoded_image_paths", {}))
    if not decoded_paths:
        decoded_paths = _infer_decoded_paths(pair_dir)

    return PanelAsset(
        pair_dir=pair_dir,
        group_title=group_title,
        pair_title=_humanize_pair(asset.get("pair"), pair_dir.name),
        projection_method=str(asset.get("projection_method", "mds")),
        prompt_key_map=asset.get("prompt_key_map", {}),
        condition_labels=asset.get("condition_labels", {}),
        decoded_paths=decoded_paths,
        flat_paths=dict(asset.get("trajectory_flat_paths", {})),
        projected_paths=dict(traj.get("projected", {})),
        projected_labels=dict(traj.get("labels", {})),
        manifold_image_path=_infer_manifold_image_path(pair_dir),
    )


def _project_flat_trajectories(flat_by_cond: dict[str, np.ndarray], method: str = "mds"):
    if not flat_by_cond:
        return None, None

    cond_names = list(flat_by_cond.keys())
    n_steps = int(next(iter(flat_by_cond.values())).shape[0])

    max_dim = max(arr.shape[1] for arr in flat_by_cond.values())
    stacked_parts = []
    for cond in cond_names:
        arr = flat_by_cond[cond].astype(np.float32, copy=False)
        if arr.shape[1] < max_dim:
            pad = np.zeros((arr.shape[0], max_dim - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        stacked_parts.append(arr)
    stacked = np.vstack(stacked_parts)

    if method == "pca":
        from sklearn.decomposition import PCA

        proj = PCA(n_components=2).fit_transform(stacked)
    else:
        from sklearn.manifold import MDS
        from sklearn.metrics import pairwise_distances

        dist = pairwise_distances(stacked, metric="euclidean")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._mds")
            proj = MDS(
                n_components=2,
                random_state=42,
                dissimilarity="precomputed",
                normalized_stress="auto",
                n_init=4,
            ).fit_transform(dist)

    projected = {}
    start = 0
    for cond in cond_names:
        end = start + n_steps
        projected[cond] = proj[start:end]
        start = end
    return projected, n_steps


def _load_projected(panel: PanelAsset, conditions: list[str]) -> tuple[dict[str, np.ndarray], int]:
    flat_by_cond: dict[str, np.ndarray] = {}
    for cond in conditions:
        rel_path = panel.flat_paths.get(cond)
        if not rel_path:
            flat_by_cond = {}
            break
        flat_path = panel.pair_dir / rel_path
        if not flat_path.exists():
            raise FileNotFoundError(f"Missing trajectory flat file: {flat_path}")
        arr = np.load(flat_path)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D flat trajectory array in {flat_path}, got {arr.shape}")
        flat_by_cond[cond] = arr

    if flat_by_cond:
        projected, n_steps = _project_flat_trajectories(flat_by_cond, method=panel.projection_method)
        if projected is None:
            raise RuntimeError(f"Failed to project trajectories for {panel.pair_dir}")
        return projected, n_steps

    projected = {}
    for cond in conditions:
        pts = panel.projected_paths.get(cond)
        if pts is None:
            raise KeyError(
                f"Missing trajectory projection for '{cond}' in {panel.pair_dir / 'grid_assets.json'}"
            )
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected projected trajectory with shape (T, 2) for '{cond}'")
        projected[cond] = arr

    n_steps = len(next(iter(projected.values())))
    return projected, n_steps


def _read_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing decoded image: {path}")
    return plt.imread(path)


def _condition_label(cond: str) -> str:
    return COND_LABELS.get(cond, cond)


def _condition_color(cond: str) -> str:
    return COND_COLORS.get(cond, "#4C566A")


def _condition_marker(cond: str) -> str:
    return COND_MARKERS.get(cond, "o")


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
        color = _condition_color(cond)
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
            condition_labels.get(cond, _condition_label(cond)),
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
        color = _condition_color(cond)
        marker = _condition_marker(cond)

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


def _draw_decoded_axis(ax: plt.Axes, image_path: Path, label: str, color: str) -> None:
    ax.imshow(_read_image(image_path))
    ax.set_xticks([])
    ax.set_yticks([])
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
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)
        spine.set_color(color)


def _draw_manifold_fallback_axis(ax: plt.Axes, image_path: Path) -> None:
    ax.imshow(_read_image(image_path))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#C7D0DB")


def _figure_size(n_conditions: int, n_panels: int, has_manifold: bool) -> tuple[float, float]:
    n_cols = 2
    n_rows = max(1, int(np.ceil(n_panels / n_cols)))
    width = 14.4 if n_conditions <= 4 else 16.8
    height_per_row = 5.2 if has_manifold else 2.9
    return width, height_per_row * n_rows + 1.2


def _decode_label_for_seed_sheet(condition: str) -> str:
    base = _condition_label(condition)
    return f"{base} across seeds"


def _panel_has_manifold(panel: PanelAsset, conditions: list[str]) -> bool:
    if panel.flat_paths and all(panel.flat_paths.get(cond) for cond in conditions):
        return True
    return bool(panel.projected_paths) and all(cond in panel.projected_paths for cond in conditions)


def _panel_has_manifold_view(panel: PanelAsset, conditions: list[str]) -> bool:
    return _panel_has_manifold(panel, conditions) or panel.manifold_image_path is not None


def _validate_joint_manifold_inputs(
    panels: list[PanelAsset],
    conditions: list[str],
    mode: str,
) -> None:
    """Reject mixed provenance when a figure is meant to compare shared-noise runs.

    A true joint MDS requires high-dimensional trajectories for every condition in
    the panel. Combining one fresh flat trajectory with several cached 2D curves
    does not preserve the underlying distance geometry.
    """
    strict_modes = {"figure2", "figure3", "figure2a"}
    if mode not in strict_modes:
        return

    missing: list[str] = []
    for panel in panels:
        absent = [cond for cond in conditions if not panel.flat_paths.get(cond)]
        if absent:
            missing.append(f"{panel.pair_dir}: missing flat trajectories for {', '.join(absent)}")
    if missing:
        joined = "\n".join(missing)
        raise ValueError(
            "Cannot rebuild a joint manifold from mixed cached projections and partial flat trajectories.\n"
            "Re-export all requested conditions as trajectory_flat_*.npy for each panel, then rerun.\n"
            f"{joined}"
        )


def render_figure(panels: list[PanelAsset], conditions: list[str], output_path: Path, figure_title: str, dpi: int) -> None:
    has_manifold = all(_panel_has_manifold_view(panel, conditions) for panel in panels)
    fig_width, fig_height = _figure_size(len(conditions), len(panels), has_manifold)
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
    n_cols = 2
    n_rows = max(1, int(np.ceil(len(panels) / n_cols)))
    outer = fig.add_gridspec(
        n_rows,
        n_cols,
        left=0.04,
        right=0.985,
        top=0.925,
        bottom=0.04,
        wspace=0.1,
        hspace=0.16,
    )

    for idx, panel in enumerate(panels):
        row, col = divmod(idx, n_cols)
        if has_manifold:
            use_raw_manifold = _panel_has_manifold(panel, conditions)
            inner = outer[row, col].subgridspec(2, 1, height_ratios=[3.1, 1.0], hspace=0.13)
            manifold_ax = fig.add_subplot(inner[0, 0])
            decoded_grid = inner[1, 0].subgridspec(1, len(conditions), wspace=0.08)
            manifold_ax.set_title(
                f"{panel.group_title}\n({panel.pair_title})" if use_raw_manifold else panel.group_title,
                fontsize=10.4,
                fontweight="bold",
                pad=14,
            )
            _section_header(manifold_ax, "Shared-noise manifold", y=1.02)
        else:
            use_raw_manifold = False
            decoded_grid = outer[row, col].subgridspec(1, len(conditions), wspace=0.08)

        panel_labels = {cond: _condition_label(cond) for cond in conditions}
        if has_manifold:
            if use_raw_manifold:
                projected, _ = _load_projected(panel, conditions)
                _plot_manifold(
                    manifold_ax,
                    projected,
                    conditions,
                    panel.projection_method,
                    panel_labels,
                )
            elif panel.manifold_image_path is not None:
                _draw_manifold_fallback_axis(
                    manifold_ax,
                    panel.pair_dir / panel.manifold_image_path,
                )
            else:
                raise KeyError(f"Missing manifold view for {panel.pair_dir}")

        decoded_axes: list[plt.Axes] = []
        for cond_idx, cond in enumerate(conditions):
            rel_path = panel.decoded_paths.get(cond)
            if rel_path is None:
                raise KeyError(
                    f"Missing decoded image path for '{cond}' in {panel.pair_dir / 'grid_assets.json'}"
                )
            decoded_ax = fig.add_subplot(decoded_grid[0, cond_idx])
            _draw_decoded_axis(
                decoded_ax,
                panel.pair_dir / rel_path,
                panel_labels[cond],
                _condition_color(cond),
            )
            decoded_axes.append(decoded_ax)

        if decoded_axes:
            if not has_manifold:
                decoded_axes[0].set_title(
                    f"{panel.group_title}\n({panel.pair_title})",
                    fontsize=10.4,
                    fontweight="bold",
                    pad=20,
                )
            _section_header(decoded_axes[0], "Decoded endpoints", y=1.14)

    fig.suptitle(figure_title, fontsize=13.2, fontweight="bold", y=0.993)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {output_path}")


def render_endpoint_figure(
    panels: list[PanelAsset],
    conditions: list[str],
    output_path: Path,
    figure_title: str,
    dpi: int,
) -> None:
    fig_width, fig_height = _figure_size(len(conditions), len(panels), has_manifold=False)
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
    n_cols = 2
    n_rows = max(1, int(np.ceil(len(panels) / n_cols)))
    outer = fig.add_gridspec(
        n_rows,
        n_cols,
        left=0.04,
        right=0.985,
        top=0.925,
        bottom=0.04,
        wspace=0.10,
        hspace=0.22,
    )

    for idx, panel in enumerate(panels):
        row, col = divmod(idx, n_cols)
        decoded_grid = outer[row, col].subgridspec(1, len(conditions), wspace=0.08)
        decoded_axes: list[plt.Axes] = []
        for cond_idx, cond in enumerate(conditions):
            rel_path = panel.decoded_paths.get(cond)
            if rel_path is None:
                raise KeyError(
                    f"Missing decoded image path for '{cond}' in {panel.pair_dir / 'grid_assets.json'}"
                )
            decoded_ax = fig.add_subplot(decoded_grid[0, cond_idx])
            _draw_decoded_axis(
                decoded_ax,
                panel.pair_dir / rel_path,
                _condition_label(cond),
                _condition_color(cond),
            )
            decoded_axes.append(decoded_ax)

        decoded_axes[0].set_title(
            f"{panel.group_title}\n({panel.pair_title})",
            fontsize=10.4,
            fontweight="bold",
            pad=20,
        )
        _section_header(decoded_axes[0], "Representative decoded endpoints", y=1.18)

    fig.suptitle(figure_title, fontsize=13.2, fontweight="bold", y=0.993)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {output_path}")


def render_seed_sheet(
    data_dir: Path,
    pair_values: list[str],
    seeds: list[int],
    condition: str,
    output_path: Path,
    figure_title: str,
    monolithic_baseline: str,
    dpi: int,
) -> None:
    if not seeds:
        raise ValueError("figure_seed_sheet requires at least one seed.")

    panels_by_seed = {
        seed: load_panels(data_dir, pair_values, monolithic_baseline, seed)
        for seed in seeds
    }

    n_rows = len(pair_values)
    n_cols = len(seeds)
    fig = plt.figure(figsize=(3.0 * n_cols + 2.4, 2.7 * n_rows + 1.4), facecolor="white")
    outer = fig.add_gridspec(
        n_rows,
        n_cols,
        left=0.11,
        right=0.985,
        top=0.90,
        bottom=0.05,
        wspace=0.05,
        hspace=0.22,
    )

    for row_idx in range(n_rows):
        for col_idx, seed in enumerate(seeds):
            panel = panels_by_seed[seed][row_idx]
            rel_path = panel.decoded_paths.get(condition)
            if rel_path is None:
                raise KeyError(
                    f"Missing decoded image path for '{condition}' in {panel.pair_dir / 'grid_assets.json'}"
                )
            ax = fig.add_subplot(outer[row_idx, col_idx])
            ax.imshow(_read_image(panel.pair_dir / rel_path))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("white")
            color = _condition_color(condition)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.2)
                spine.set_color(color)

            if row_idx == 0:
                ax.set_title(f"seed {seed}", fontsize=8.8, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(
                    panel.group_title.replace("Group ", "G"),
                    fontsize=8.6,
                    fontweight="bold",
                    rotation=0,
                    labelpad=52,
                    va="center",
                )
                ax.text(
                    -0.02,
                    1.08,
                    panel.pair_title,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7.4,
                    color="#4C566A",
                )

    fig.suptitle(
        f"{figure_title}: {_decode_label_for_seed_sheet(condition)}",
        fontsize=13.0,
        fontweight="bold",
        y=0.975,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {output_path}")


def render_figure_group3(
    panels: list[PanelAsset],
    conditions: list[str],
    output_path: Path,
    figure_title: str,
    subgroup_specs: list[dict],
    dpi: int,
) -> None:
    has_manifold = all(_panel_has_manifold_view(panel, conditions) for panel in panels)
    fig = plt.figure(figsize=(21.0, 9.0 if has_manifold else 4.8), facecolor="white")
    outer = fig.add_gridspec(
        1, 3,
        left=0.03, right=0.985,
        top=0.88, bottom=0.10,
        wspace=0.12,
    )
    for idx, (panel, spec) in enumerate(zip(panels, subgroup_specs)):
        panel_labels = {cond: _condition_label(cond) for cond in conditions}
        if has_manifold:
            use_raw_manifold = _panel_has_manifold(panel, conditions)
            inner = outer[0, idx].subgridspec(2, 1, height_ratios=[3.1, 1.0], hspace=0.13)
            manifold_ax = fig.add_subplot(inner[0, 0])
            decoded_grid = inner[1, 0].subgridspec(1, len(conditions), wspace=0.08)

            manifold_ax.set_title(
                f"{spec['label']}\n({panel.pair_title})" if use_raw_manifold else spec["label"],
                fontsize=10.4, fontweight="bold", pad=14,
            )
            _section_header(manifold_ax, "Shared-noise manifold", y=1.02)
            if use_raw_manifold:
                projected, _ = _load_projected(panel, conditions)
                _plot_manifold(manifold_ax, projected, conditions, panel.projection_method, panel_labels)
            elif panel.manifold_image_path is not None:
                _draw_manifold_fallback_axis(
                    manifold_ax,
                    panel.pair_dir / panel.manifold_image_path,
                )
            else:
                raise KeyError(f"Missing manifold view for {panel.pair_dir}")
        else:
            decoded_grid = outer[0, idx].subgridspec(1, len(conditions), wspace=0.08)

        decoded_axes: list[plt.Axes] = []
        for cond_idx, cond in enumerate(conditions):
            rel_path = panel.decoded_paths.get(cond)
            if rel_path is None:
                raise KeyError(
                    f"Missing decoded image path for '{cond}' in {panel.pair_dir / 'grid_assets.json'}"
                )
            decoded_ax = fig.add_subplot(decoded_grid[0, cond_idx])
            _draw_decoded_axis(
                decoded_ax,
                panel.pair_dir / rel_path,
                panel_labels[cond],
                _condition_color(cond),
            )
            decoded_axes.append(decoded_ax)

        if decoded_axes:
            decoded_axes[0].set_title(
                f"{spec['label']}\n({panel.pair_title})",
                fontsize=10.4,
                fontweight="bold",
                pad=20,
            )
            _section_header(decoded_axes[0], "Decoded endpoints", y=1.14)

        fig.text(
            (idx + 0.5) / 3.0,
            0.04,
            f"Failure: {spec['failure_pattern']}",
            ha="center", va="top",
            fontsize=8.5, fontstyle="italic", color="#4C566A",
        )

    fig.suptitle(figure_title, fontsize=13.2, fontweight="bold", y=0.975)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {output_path}")


def render_figure_1x2(
    panels: list[PanelAsset],
    conditions: list[str],
    output_path: Path,
    figure_title: str,
    column_titles: list[str],
    dpi: int,
) -> None:
    """Render a legacy 1×2 figure (exactly two panels, e.g. G1 and G6).

    Each column has the same inner subgridspec as render_figure():
      - manifold on top (height ratio 3.1)
      - decoded image strip below (height ratio 1.0)
    """
    n_conditions = len(conditions)
    width = 14.4 if n_conditions <= 4 else 16.8
    has_manifold = all(_panel_has_manifold_view(panel, conditions) for panel in panels)
    fig = plt.figure(figsize=(width, 9.0 if has_manifold else 4.8), facecolor="white")
    outer = fig.add_gridspec(
        1, 2,
        left=0.04, right=0.985,
        top=0.925, bottom=0.04,
        wspace=0.1,
    )

    for idx, (panel, col_title) in enumerate(zip(panels, column_titles)):
        if has_manifold:
            use_raw_manifold = _panel_has_manifold(panel, conditions)
            inner = outer[0, idx].subgridspec(2, 1, height_ratios=[3.1, 1.0], hspace=0.13)
            manifold_ax = fig.add_subplot(inner[0, 0])
            decoded_grid = inner[1, 0].subgridspec(1, len(conditions), wspace=0.08)
            manifold_ax.set_title(
                panel.pair_title if use_raw_manifold else panel.group_title,
                fontsize=10.4,
                fontweight="bold",
                pad=14,
            )
            _section_header(manifold_ax, col_title, y=1.02)
        else:
            use_raw_manifold = False
            decoded_grid = outer[0, idx].subgridspec(1, len(conditions), wspace=0.08)

        panel_labels = {cond: _condition_label(cond) for cond in conditions}
        if has_manifold:
            if use_raw_manifold:
                projected, _ = _load_projected(panel, conditions)
                _plot_manifold(
                    manifold_ax,
                    projected,
                    conditions,
                    panel.projection_method,
                    panel_labels,
                )
            elif panel.manifold_image_path is not None:
                _draw_manifold_fallback_axis(
                    manifold_ax,
                    panel.pair_dir / panel.manifold_image_path,
                )
            else:
                raise KeyError(f"Missing manifold view for {panel.pair_dir}")

        decoded_axes: list[plt.Axes] = []
        for cond_idx, cond in enumerate(conditions):
            rel_path = panel.decoded_paths.get(cond)
            if rel_path is None:
                raise KeyError(
                    f"Missing decoded image path for '{cond}' in {panel.pair_dir / 'grid_assets.json'}"
                )
            decoded_ax = fig.add_subplot(decoded_grid[0, cond_idx])
            _draw_decoded_axis(
                decoded_ax,
                panel.pair_dir / rel_path,
                panel_labels[cond],
                _condition_color(cond),
            )
            decoded_axes.append(decoded_ax)

        if decoded_axes:
            if not has_manifold:
                decoded_axes[0].set_title(
                    f"{panel.pair_title}",
                    fontsize=10.4,
                    fontweight="bold",
                    pad=20,
                )
                _section_header(decoded_axes[0], col_title, y=1.22)
            else:
                _section_header(decoded_axes[0], "Decoded endpoints", y=1.14)

    fig.suptitle(figure_title, fontsize=13.2, fontweight="bold", y=0.993)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {output_path}")


def load_panels(
    data_dir: Path,
    pair_values: list[str],
    monolithic_baseline: str,
    seed: int | None,
) -> list[PanelAsset]:
    return [
        _load_panel_asset(data_dir, pair_value, idx, monolithic_baseline, seed)
        for idx, pair_value in enumerate(pair_values)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render paper-facing taxonomy qualitative figures.")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Screening or final SDXL root containing group/<pair> or seed_<n>/<group>/<pair> assets.",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        metavar="PAIR",
        help=(
            "Pair directories or pair slugs. For figure1/figure2/figure3: six entries in "
            "reading order (G1..G6). For figure_group3: three legacy subgroup entries. "
            "Relative values are resolved under {data-dir}/pairs/."
        ),
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output image path. Defaults to a paper figure path based on --mode.",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "figure1",
            "figure1_endpoints",
            "figure2",
            "figure3",
            "figure_group3",
            "figure1a",
            "figure2a",
            "figure_seed_sheet",
            "custom",
        ],
        default="figure1",
        help="Predefined paper figure layout to render.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        metavar="COND",
        help=(
            "Explicit condition order for a custom render. "
            "Valid conditions: prompt_a prompt_b monolithic poe co3 pstar_sdipc pstar_inv."
        ),
    )
    parser.add_argument(
        "--monolithic-baseline",
        choices=["auto", "naive", "natural"],
        default="auto",
        help="Which monolithic baseline to treat as canonical in the grid assets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to read from a multi-seed SDXL final root laid out as seed_<n>/<group>/<pair>/...",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        metavar="SEED",
        help="Multiple seeds to use for figure_seed_sheet.",
    )
    parser.add_argument(
        "--seed-condition",
        default="poe",
        choices=sorted(COND_LABELS),
        help="Decoded condition to display in figure_seed_sheet.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI (default: 300).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    is_group3 = args.mode == "figure_group3"
    is_1x2    = args.mode in {"figure1a", "figure2a"}
    is_seed_sheet = args.mode == "figure_seed_sheet"

    pair_values = args.pairs or _resolve_default_pair_values(data_dir, args.mode)

    if is_1x2 and len(pair_values) != 2:
        raise SystemExit("figure1a/figure2a expect exactly 2 pairs (G1 and G6).")
    if is_group3 and len(pair_values) != 3:
        raise SystemExit("figure_group3 expects exactly 3 pairs.")
    if not (is_1x2 or is_group3) and args.mode != "custom" and len(pair_values) != 6:
        raise SystemExit("Expected exactly 6 pair entries in reading order (G1..G6).")

    if args.conditions:
        conditions = args.conditions
    elif args.mode != "custom" and not is_seed_sheet:
        conditions = MODE_CONDITIONS[args.mode]
    elif args.mode == "custom":
        raise SystemExit("--conditions is required when --mode custom is used.")
    else:
        conditions = []

    valid_conditions = set(COND_LABELS)
    unknown = [cond for cond in conditions if cond not in valid_conditions]
    if unknown:
        raise SystemExit(f"Unknown condition(s): {unknown}. Valid values: {sorted(valid_conditions)}")

    if args.mode in {"figure2", "figure2a"}:
        required = {"poe", "pstar_sdipc"}
        if not required.issubset(set(conditions)):
            raise SystemExit(f"{args.mode} requires both 'poe' and 'pstar_sdipc' in the condition list.")
    if args.mode == "figure3":
        required = {"poe", "co3", "pstar_sdipc", "pstar_co3_sdipc"}
        if not required.issubset(set(conditions)):
            raise SystemExit("Figure 3 requires 'poe', 'co3', 'pstar_sdipc', and 'pstar_co3_sdipc' in the condition list.")

    output_path = Path(args.out) if args.out else MODE_OUTPUTS.get(
        args.mode, DEFAULT_OUTPUT_DIR / "trajectory_3x2_custom.png"
    )
    figure_title = MODE_TITLES.get(args.mode, "Taxonomy qualitative overview")

    if is_seed_sheet:
        seeds = args.seeds or [42, 1, 7, 13]
        render_seed_sheet(
            data_dir,
            pair_values,
            seeds,
            args.seed_condition,
            output_path,
            figure_title,
            args.monolithic_baseline,
            dpi=args.dpi,
        )
        return

    panels = load_panels(data_dir, pair_values, args.monolithic_baseline, args.seed)
    _validate_joint_manifold_inputs(panels, conditions, args.mode)
    if args.mode == "figure1_endpoints":
        render_endpoint_figure(panels, conditions, output_path, figure_title, dpi=args.dpi)
    elif is_group3:
        render_figure_group3(panels, conditions, output_path, figure_title, GROUP3_SUBGROUP_SPECS, dpi=args.dpi)
    elif is_1x2:
        column_titles = [GROUP_LABELS[0], GROUP_LABELS[-1]]
        render_figure_1x2(panels, conditions, output_path, figure_title, column_titles, dpi=args.dpi)
    else:
        render_figure(panels, conditions, output_path, figure_title, dpi=args.dpi)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
