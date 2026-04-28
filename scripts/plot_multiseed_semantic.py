"""
Semantic-first multi-seed visualization for canonical SDXL multiseed pair dirs.

Outputs three artifacts:
1. CLIP endpoint manifold from decoded endpoint images
2. Probe-aware score-space scatter using joint probes + BLIP-VQA
3. Score-summary table treating semantic scores as the primary claim
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS
except ImportError as exc:
    raise SystemExit(
        "plot_multiseed_semantic.py requires scikit-learn. Install it before running."
    ) from exc

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError as exc:
    raise SystemExit(
        "plot_multiseed_semantic.py requires transformers. Install it before running."
    ) from exc

from plot_multiseed_trajectory_grid import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_MARKERS,
    CONDITION_ORDER,
    LABEL_OFFSETS,
    PanelSpec,
    _draw_decoded_axis,
    _load_image,
    _section_header,
    _style_manifold_axis,
    resolve_panel_specs,
)


CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
METRIC_CONDITION_MAP = {
    "prompt_a": "c1",
    "prompt_b": "c2",
    "monolithic": "mono",
    "poe": "poe",
}
SEED_ALPHAS = [0.42, 0.58, 0.74, 0.9]
SUMMARY_COLOR_MAP = {
    True: "#DFF3E3",
    False: "#F7D9D9",
    None: "#F4F6F8",
}


@dataclass
class SemanticPoint:
    panel: PanelSpec
    condition: str
    image_path: Path
    embedding: np.ndarray | None = None
    clip_xy: np.ndarray | None = None
    probe_xy: np.ndarray | None = None
    cue_presence_score: float | None = None
    anti_collapse_score: float | None = None
    hybrid_score: float | None = None
    omission_score: float | None = None
    joint_correctness_score: float | None = None
    semantic_pass: bool | None = None
    high_confidence_pass: bool | None = None
    hybridization_failure: bool | None = None
    omission_failure: bool | None = None
    p_c1: float | None = None
    p_c2: float | None = None


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _derive_pair_root(pair_dirs: list[Path]) -> Path:
    roots: list[Path] = []
    for pair_dir in pair_dirs:
        seed_ancestor = None
        for ancestor in pair_dir.parents:
            if ancestor.name.startswith("seed_"):
                seed_ancestor = ancestor
                break
        if seed_ancestor is None:
            raise ValueError(f"Could not locate seed_<n> ancestor for {pair_dir}")
        roots.append(seed_ancestor.parent)
    unique_roots = {root.resolve() for root in roots}
    if len(unique_roots) != 1:
        raise ValueError(f"Pair dirs do not share one multiseed root: {sorted(str(p) for p in unique_roots)}")
    return roots[0]


def _validate_panels(panels: list[PanelSpec]) -> None:
    if len(panels) != 4:
        raise ValueError(f"Expected exactly 4 panels, got {len(panels)}")
    missing_messages: list[str] = []
    for panel in panels:
        missing_proj = [cond for cond in CONDITION_ORDER if cond not in panel.projected_paths]
        missing_decoded = [cond for cond in CONDITION_ORDER if cond not in panel.decoded_paths]
        if missing_proj or missing_decoded:
            details: list[str] = []
            if missing_proj:
                details.append(f"missing trajectories={missing_proj}")
            if missing_decoded:
                details.append(f"missing decoded={missing_decoded}")
            missing_messages.append(f"{panel.pair_dir}: {', '.join(details)}")
    if missing_messages:
        joined = "\n  ".join(missing_messages)
        raise ValueError(f"Canonical A/B/A∧B/PoE assets required.\n  {joined}")


def _load_metric_payload(path: Path, *, label: str, upstream: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}\n"
            f"Generate it first with `python scripts/{upstream} --data-dir {path.parent}`"
        )
    return _load_json(path)


def _build_points(panels: list[PanelSpec]) -> list[SemanticPoint]:
    points: list[SemanticPoint] = []
    for panel in panels:
        for condition in CONDITION_ORDER:
            rel_path = panel.decoded_paths.get(condition)
            if rel_path is None:
                continue
            image_path = panel.pair_dir / rel_path
            if not image_path.exists():
                raise FileNotFoundError(f"Missing decoded endpoint image: {image_path}")
            points.append(
                SemanticPoint(
                    panel=panel,
                    condition=condition,
                    image_path=image_path,
                )
            )
    return points


def _load_clip_model(device: str) -> tuple[CLIPModel, CLIPProcessor, str]:
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(resolved_device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, processor, resolved_device


def _clip_image_features(model: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    vision_outputs = model.vision_model(pixel_values=pixel_values)
    pooled = vision_outputs.pooler_output
    projected = model.visual_projection(pooled).float()
    return F.normalize(projected, dim=-1)


@torch.no_grad()
def _compute_clip_embeddings(points: list[SemanticPoint], device: str) -> None:
    model, processor, resolved_device = _load_clip_model(device)
    try:
        batch_size = 8
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            pils = [Image.open(point.image_path).convert("RGB") for point in batch]
            inputs = processor(images=pils, return_tensors="pt")
            inputs = {key: value.to(resolved_device) for key, value in inputs.items()}
            embeds = _clip_image_features(model, inputs["pixel_values"]).cpu().numpy()
            for point, embedding in zip(batch, embeds):
                point.embedding = embedding.astype(np.float32, copy=False)
            for image in pils:
                image.close()
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _project_embeddings(points: list[SemanticPoint], projection: str) -> None:
    embed_arr = np.stack([point.embedding for point in points if point.embedding is not None], axis=0)
    if projection == "pca":
        projector = PCA(n_components=2)
        coords = projector.fit_transform(embed_arr)
    else:
        cosine_dist = 1.0 - np.clip(embed_arr @ embed_arr.T, -1.0, 1.0)
        projector = MDS(
            n_components=2,
            random_state=42,
            dissimilarity="precomputed",
            normalized_stress="auto",
        )
        coords = projector.fit_transform(cosine_dist)
    for point, coord in zip(points, coords):
        point.clip_xy = coord.astype(np.float32, copy=False)


def _index_joint_probe_records(payload: dict[str, Any], pair_slug: str) -> dict[tuple[int, str], dict[str, Any]]:
    image_scores = payload.get("image_scores", [])
    index: dict[tuple[int, str], dict[str, Any]] = {}
    for row in image_scores:
        if row.get("pair_slug") != pair_slug:
            continue
        try:
            seed = int(row["seed"])
        except (ValueError, TypeError, KeyError):
            continue
        condition = str(row.get("condition", "")).strip()
        index[(seed, condition)] = row
    return index


def _index_blip_records(payload: list[dict[str, Any]], pair_slug: str) -> dict[tuple[int, str], dict[str, Any]]:
    index: dict[tuple[int, str], dict[str, Any]] = {}
    for row in payload:
        if row.get("pair_slug") != pair_slug:
            continue
        try:
            seed = int(row["seed"])
        except (ValueError, TypeError, KeyError):
            continue
        condition = str(row.get("condition", "")).strip()
        index[(seed, condition)] = row
    return index


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _attach_metric_scores(
    points: list[SemanticPoint],
    joint_probe_payload: dict[str, Any],
    blip_payload: list[dict[str, Any]],
    pair_slug: str,
) -> None:
    joint_index = _index_joint_probe_records(joint_probe_payload, pair_slug)
    blip_index = _index_blip_records(blip_payload, pair_slug)
    missing: list[str] = []

    for point in points:
        metric_condition = METRIC_CONDITION_MAP[point.condition]
        key = (point.panel.seed_number, metric_condition)
        joint_row = joint_index.get(key)
        blip_row = blip_index.get(key)
        if joint_row is None:
            missing.append(f"joint_probe_scores.json missing seed={key[0]} condition={metric_condition}")
            continue
        if blip_row is None:
            missing.append(f"blip_vqa_scores.json missing seed={key[0]} condition={metric_condition}")
            continue

        point.cue_presence_score = _coerce_float(joint_row.get("cue_presence_score"))
        point.anti_collapse_score = _coerce_float(joint_row.get("anti_collapse_score"))
        point.hybrid_score = _coerce_float(joint_row.get("hybrid_score"))
        point.omission_score = _coerce_float(joint_row.get("omission_score"))
        point.joint_correctness_score = _coerce_float(joint_row.get("joint_correctness_score"))
        point.semantic_pass = _coerce_bool(joint_row.get("semantic_pass"))
        point.high_confidence_pass = _coerce_bool(joint_row.get("high_confidence_pass"))
        point.hybridization_failure = _coerce_bool(joint_row.get("hybridization_failure"))
        point.omission_failure = _coerce_bool(joint_row.get("omission_failure"))
        point.p_c1 = _coerce_float(blip_row.get("p_c1"))
        point.p_c2 = _coerce_float(blip_row.get("p_c2"))

        point.probe_xy = np.array(
            [
                _probe_x(point),
                _probe_y(point),
            ],
            dtype=np.float32,
        )

    if missing:
        preview = "\n  ".join(sorted(set(missing))[:12])
        raise ValueError(f"Metric files do not cover all requested seed/condition rows.\n  {preview}")


def _probe_x(point: SemanticPoint) -> float:
    if point.cue_presence_score is not None:
        return float(point.cue_presence_score)
    if point.condition == "prompt_a" and point.p_c1 is not None:
        return float(point.p_c1)
    if point.condition == "prompt_b" and point.p_c2 is not None:
        return float(point.p_c2)
    if point.joint_correctness_score is not None:
        return float(point.joint_correctness_score)
    raise ValueError(f"No cue-presence-compatible score for seed={point.panel.seed_number} {point.condition}")


def _probe_y(point: SemanticPoint) -> float:
    if point.anti_collapse_score is not None:
        return float(point.anti_collapse_score)
    if point.condition in {"prompt_a", "prompt_b"}:
        return 1.0
    if point.joint_correctness_score is not None:
        return float(point.joint_correctness_score)
    raise ValueError(f"No anti-collapse-compatible score for seed={point.panel.seed_number} {point.condition}")


def _section_grid(fig: plt.Figure, panels: list[PanelSpec], outer_spec: Any) -> None:
    decoded_band = outer_spec.subgridspec(1, 3, width_ratios=[0.14, 0.72, 0.14], wspace=0.0)
    decoded_grid = decoded_band[0, 1].subgridspec(len(panels), len(CONDITION_ORDER), hspace=0.01, wspace=0.01)
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


def _scatter_semantic_points(
    ax: plt.Axes,
    points: list[SemanticPoint],
    *,
    xy_attr: str,
    label_overrides: dict[str, str],
) -> None:
    for point in points:
        xy = getattr(point, xy_attr)
        color = CONDITION_COLORS[point.condition]
        marker = CONDITION_MARKERS[point.condition]
        alpha = SEED_ALPHAS[min(max(point.panel.seed_number - 1, 0), len(SEED_ALPHAS) - 1)]
        ax.scatter(
            float(xy[0]),
            float(xy[1]),
            s=74,
            color=color,
            marker=marker,
            alpha=alpha,
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )
        label = f"S{point.panel.seed_number} {label_overrides.get(point.condition, point.condition)}"
        ax.annotate(
            label,
            xy=(float(xy[0]), float(xy[1])),
            xytext=LABEL_OFFSETS.get(point.condition, (8, 8)),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.9,
                "alpha": 0.94,
            },
            zorder=6,
        )


def _add_probe_image_overlays(ax: plt.Axes, points: list[SemanticPoint]) -> None:
    offset_by_condition = {
        "prompt_a": (-54.0, 36.0),
        "prompt_b": (54.0, 36.0),
        "monolithic": (-54.0, -36.0),
        "poe": (54.0, -36.0),
    }
    jitter_by_seed = {
        1: (-6.0, 0.0),
        2: (6.0, 0.0),
        3: (-6.0, 10.0),
        4: (6.0, 10.0),
    }
    for point in points:
        image = _load_image(point.image_path)
        if image is None or point.probe_xy is None:
            continue
        base = np.array(offset_by_condition[point.condition], dtype=np.float32)
        jitter = np.array(jitter_by_seed.get(point.panel.seed_number, (0.0, 0.0)), dtype=np.float32)
        offset = base + jitter
        artist = AnnotationBbox(
            OffsetImage(image, zoom=0.11),
            (float(point.probe_xy[0]), float(point.probe_xy[1])),
            xybox=(float(offset[0]), float(offset[1])),
            xycoords="data",
            boxcoords="offset pixels",
            frameon=True,
            pad=0.18,
            bboxprops={
                "edgecolor": CONDITION_COLORS[point.condition],
                "linewidth": 1.6,
                "facecolor": "white",
                "alpha": 0.98,
            },
            zorder=7,
        )
        ax.add_artist(artist)
        ax.annotate(
            f"S{point.panel.seed_number} {CONDITION_LABELS[point.condition]}",
            xy=(float(point.probe_xy[0]), float(point.probe_xy[1])),
            xytext=(float(offset[0]), float(offset[1] - 10.0)),
            xycoords="data",
            textcoords="offset pixels",
            ha="center",
            va="top",
            fontsize=6.6,
            fontweight="bold",
            color=CONDITION_COLORS[point.condition],
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": CONDITION_COLORS[point.condition],
                "linewidth": 0.8,
                "alpha": 0.92,
            },
            zorder=8,
        )


def _add_condition_legend(ax: plt.Axes) -> None:
    handles = []
    for condition in CONDITION_ORDER:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=CONDITION_COLORS[condition],
                marker=CONDITION_MARKERS[condition],
                markersize=6,
                linewidth=0,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=CONDITION_LABELS[condition],
            )
        )
    ax.legend(handles=handles, loc="upper right", fontsize=8.5, framealpha=0.96, edgecolor="#D7DEE8")


def make_clip_figure(
    panels: list[PanelSpec],
    points: list[SemanticPoint],
    output_path: Path,
    projection: str,
    pair_name: str,
) -> None:
    fig = plt.figure(figsize=(12.8, 13.8))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.3, 0.95], left=0.05, right=0.985, top=0.94, bottom=0.045, hspace=0.08)
    ax = fig.add_subplot(outer[0, 0])
    _section_header(ax, "CLIP endpoint manifold", y=1.02)
    ax.set_title(f"({pair_name})", fontsize=10.8, fontweight="bold", pad=6)
    _style_manifold_axis(ax, projection)
    _scatter_semantic_points(ax, points, xy_attr="clip_xy", label_overrides=CONDITION_LABELS)
    _add_condition_legend(ax)
    fig.suptitle("Multi-seed semantic overview: CLIP endpoints", fontsize=13.6, fontweight="bold", y=0.972)
    _section_grid(fig, panels, outer[1, 0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_probe_figure(
    panels: list[PanelSpec],
    points: list[SemanticPoint],
    output_path: Path,
    pair_name: str,
) -> None:
    fig = plt.figure(figsize=(13.8, 13.8))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.35, 0.95], left=0.05, right=0.985, top=0.94, bottom=0.045, hspace=0.08)
    ax = fig.add_subplot(outer[0, 0])
    _section_header(ax, "Probe-aware score space", y=1.02)
    ax.set_title(f"({pair_name})", fontsize=10.8, fontweight="bold", pad=6)
    ax.set_facecolor("#FBFCFD")
    ax.grid(True, color="#E1E7EF", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_xlabel("Cue presence score", fontsize=9.0)
    ax.set_ylabel("Anti-collapse score", fontsize=9.0)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.tick_params(axis="both", labelsize=8.0, colors="#4C566A")
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#C7D0DB")
    _scatter_semantic_points(ax, points, xy_attr="probe_xy", label_overrides=CONDITION_LABELS)
    _add_probe_image_overlays(ax, points)
    _add_condition_legend(ax)
    fig.suptitle("Multi-seed semantic overview: probe-aware score space", fontsize=13.6, fontweight="bold", y=0.972)
    _section_grid(fig, panels, outer[1, 0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _summary_cell_lines(point: SemanticPoint) -> list[str]:
    lines: list[str] = []
    if point.joint_correctness_score is not None:
        lines.append(f"JP {point.joint_correctness_score:.2f}")
    if point.condition == "prompt_a" and point.p_c1 is not None:
        lines.append(f"BLIP A {point.p_c1:.2f}")
    elif point.condition == "prompt_b" and point.p_c2 is not None:
        lines.append(f"BLIP B {point.p_c2:.2f}")
    else:
        if point.p_c1 is not None:
            lines.append(f"BLIP A {point.p_c1:.2f}")
        if point.p_c2 is not None:
            lines.append(f"BLIP B {point.p_c2:.2f}")
        if point.hybridization_failure is not None:
            lines.append(f"Hybrid {'FAIL' if point.hybridization_failure else 'OK'}")
        if point.omission_failure is not None:
            lines.append(f"Omit {'FAIL' if point.omission_failure else 'OK'}")
    if point.high_confidence_pass is not None:
        lines.append(f"High {'PASS' if point.high_confidence_pass else 'NO'}")
    elif point.semantic_pass is not None:
        lines.append(f"Sem {'PASS' if point.semantic_pass else 'NO'}")
    return lines


def _summary_cell_color(point: SemanticPoint) -> str:
    return SUMMARY_COLOR_MAP[point.semantic_pass]


def make_score_summary_figure(
    panels: list[PanelSpec],
    points: list[SemanticPoint],
    output_path: Path,
    pair_name: str,
) -> None:
    point_index = {(point.panel.seed_number, point.condition): point for point in points}
    fig, axes = plt.subplots(len(panels), len(CONDITION_ORDER), figsize=(12.8, 8.8))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.08, wspace=0.04, hspace=0.08)

    for row_idx, panel in enumerate(sorted(panels, key=lambda item: item.seed_number)):
        for col_idx, condition in enumerate(CONDITION_ORDER):
            ax = axes[row_idx, col_idx]
            point = point_index[(panel.seed_number, condition)]
            ax.set_facecolor(_summary_cell_color(point))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.4)
                spine.set_color(CONDITION_COLORS[condition])
            if row_idx == 0:
                title = ax.set_title(CONDITION_LABELS[condition], fontsize=9.2, pad=8, fontweight="bold", color=CONDITION_COLORS[condition])
                title.set_bbox(
                    {
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": CONDITION_COLORS[condition],
                        "linewidth": 0.9,
                        "alpha": 0.96,
                    }
                )
            if col_idx == 0:
                ax.text(
                    -0.08,
                    0.5,
                    f"S{panel.seed_number}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8.8,
                    fontweight="bold",
                    color="#2A2F36",
                )
            lines = _summary_cell_lines(point)
            ax.text(
                0.03,
                0.96,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.0,
                fontweight="bold",
                color="#2A2F36",
            )
    fig.suptitle("Multi-seed semantic score summary", fontsize=13.6, fontweight="bold", y=0.965)
    fig.text(0.10, 0.92, f"({pair_name})", ha="left", va="center", fontsize=10.4, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CLIP, probe-aware, and score-summary multiseed semantic figures.")
    parser.add_argument("--pair-dirs", nargs=4, metavar="DIR", required=True, help="Exactly 4 canonical pair directories (one per seed).")
    parser.add_argument("--clip-output", default="", help="Path for the CLIP endpoint manifold figure.")
    parser.add_argument("--probe-output", default="", help="Path for the probe-aware score-space figure.")
    parser.add_argument("--score-output", default="", help="Path for the semantic score-summary figure.")
    parser.add_argument("--joint-probe-json", default="", help="Optional override path for joint_probe_scores.json.")
    parser.add_argument("--blip-vqa-json", default="", help="Optional override path for blip_vqa_scores.json.")
    parser.add_argument("--projection", default="mds", choices=["mds", "pca"], help="Projection for the CLIP endpoint manifold.")
    parser.add_argument("--pair-name", default="", help="Optional pair title override.")
    parser.add_argument("--device", default="auto", help="Torch device for CLIP embedding extraction (default: auto).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_dirs = [Path(path) for path in args.pair_dirs]
    for pair_dir in pair_dirs:
        if not pair_dir.exists():
            raise FileNotFoundError(f"Pair directory does not exist: {pair_dir}")

    panels = resolve_panel_specs(pair_dirs)
    _validate_panels(panels)
    pair_root = _derive_pair_root(pair_dirs)
    pair_slug = pair_dirs[0].name
    pair_name = args.pair_name.strip() or (panels[0].pair_title if panels else pair_slug.replace("_", " "))

    clip_output = Path(args.clip_output) if args.clip_output else pair_root / "multiseed_semantic_clip.png"
    probe_output = Path(args.probe_output) if args.probe_output else pair_root / "multiseed_semantic_probe.png"
    score_output = Path(args.score_output) if args.score_output else pair_root / "multiseed_semantic_scores.png"
    joint_probe_json = Path(args.joint_probe_json) if args.joint_probe_json else pair_root / "joint_probe_scores.json"
    blip_vqa_json = Path(args.blip_vqa_json) if args.blip_vqa_json else pair_root / "blip_vqa_scores.json"

    points = _build_points(panels)
    joint_probe_payload = _load_metric_payload(
        joint_probe_json,
        label="joint probe metrics",
        upstream="eval_joint_probes.py",
    )
    blip_payload = _load_metric_payload(
        blip_vqa_json,
        label="BLIP-VQA metrics",
        upstream="eval_blip_vqa.py",
    )
    if not isinstance(joint_probe_payload, dict) or "image_scores" not in joint_probe_payload:
        raise ValueError(f"Unexpected joint probe payload shape in {joint_probe_json}")
    if not isinstance(blip_payload, list):
        raise ValueError(f"Unexpected BLIP-VQA payload shape in {blip_vqa_json}")
    _attach_metric_scores(points, joint_probe_payload, blip_payload, pair_slug)
    _compute_clip_embeddings(points, args.device)
    _project_embeddings(points, args.projection)

    make_clip_figure(panels, points, clip_output, args.projection, pair_name)
    print(f"Saved -> {clip_output}")
    make_probe_figure(panels, points, probe_output, pair_name)
    print(f"Saved -> {probe_output}")
    make_score_summary_figure(panels, points, score_output, pair_name)
    print(f"Saved -> {score_output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
