"""
Build a qualitative grid for semantic composition vs PoE vs SD-IPC.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GROUP_ORDER = [
    "group1_cooccurrence",
    "group2_disentangled",
    "group3_ood",
    "group4_collision",
]

GROUP_LABELS = {
    "group1_cooccurrence": "G1 Co-occurrence",
    "group2_disentangled": "G2 Disentangled",
    "group3_ood": "G3 OOD",
    "group4_collision": "G4 Collision",
}


def load_records(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [row for row in payload["records"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError(f"Unsupported reachability JSON shape in {path}")


def _image_or_none(path_str: str) -> Image.Image | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        rooted = PROJECT_ROOT / path
        if rooted.exists():
            path = rooted
    if not path.exists():
        return None
    return Image.open(path).convert("RGB")


def _image_for_row(row: dict[str, Any], kind: str) -> Image.Image | None:
    run_dir = row.get("run_dir", "")
    candidates: list[str] = []
    if kind == "monolithic":
        candidates.append(str(row.get("monolithic_image_path", "")))
        if run_dir:
            candidates.append(str(Path(run_dir) / "monolithic.png"))
    elif kind == "poe":
        candidates.append(str(row.get("poe_image_path", "")))
        if run_dir:
            candidates.append(str(Path(run_dir) / "poe.png"))
    elif kind == "sdipc":
        candidates.append(str(row.get("sdipc_image_path", "")))

    for candidate in candidates:
        image = _image_or_none(candidate)
        if image is not None:
            return image
    return None


def _score(row: dict[str, Any]) -> float:
    value = row.get("d_T_sdipc_poe")
    if isinstance(value, (int, float)):
        return float(value)
    return float("inf")


def select_rows(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    max_rows: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            GROUP_ORDER.index(row.get("taxonomy_group", "")) if row.get("taxonomy_group", "") in GROUP_ORDER else 99,
            row.get("pair_slug", ""),
            int(row.get("seed", 0)),
        ),
    )
    if mode == "all":
        return ordered[:max_rows] if max_rows > 0 else ordered

    selected: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        group_rows = [row for row in ordered if row.get("taxonomy_group") == group]
        if not group_rows:
            continue
        best = min(group_rows, key=_score)
        selected.append(best)

    if max_rows > 0:
        selected = selected[:max_rows]
    return selected


def plot_grid(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No reachability rows selected for plotting.")

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9.6, 3.0 * n_rows))
    if n_rows == 1:
        axes = np.asarray([axes])

    col_titles = ["Semantic Composition", "PoE", "SD-IPC"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for row_idx, row in enumerate(rows):
        images = [
            _image_for_row(row, "monolithic"),
            _image_for_row(row, "poe"),
            _image_for_row(row, "sdipc"),
        ]
        for col_idx, image in enumerate(images):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if image is None:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=10, color="#777777")
            else:
                ax.imshow(image)

        left_ax = axes[row_idx, 0]
        label = (
            f"{GROUP_LABELS.get(row.get('taxonomy_group', ''), row.get('taxonomy_group', ''))}\n"
            f"A = {row.get('prompt_a', '')}\n"
            f"B = {row.get('prompt_b', '')}\n"
            f"seed = {row.get('seed', '')} | d_T* = {float(row.get('d_T_sdipc_poe', float('nan'))):.3f}"
        )
        left_ax.text(
            -0.14,
            0.02,
            label,
            transform=left_ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.suptitle("Phase 1 qualitative comparison: semantic composition vs PoE vs SD-IPC", fontsize=13, y=0.995)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a qualitative grid of semantic composition vs PoE vs SD-IPC."
    )
    parser.add_argument(
        "--reachability-json",
        type=str,
        default="",
        help="Path to phase1_sdipc_reachability.json (default: experiments/eccv2026/reachability/phase1_sdipc_reachability.json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output PNG path (default: experiments/eccv2026/reachability/phase1_sdipc_qualitative_grid.png).",
    )
    parser.add_argument(
        "--selection",
        choices=["best-per-group", "all"],
        default="best-per-group",
        help="Choose one representative row per group or include all available rows.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on rows in the final figure.",
    )
    args = parser.parse_args()

    reachability_json = Path(args.reachability_json) if args.reachability_json else (
        PROJECT_ROOT / "experiments" / "eccv2026" / "reachability" / "phase1_sdipc_reachability.json"
    )
    output = Path(args.output) if args.output else (
        PROJECT_ROOT / "experiments" / "eccv2026" / "reachability" / "phase1_sdipc_qualitative_grid.png"
    )

    rows = load_records(reachability_json)
    selected = select_rows(rows, mode=args.selection, max_rows=args.max_rows)
    plot_grid(selected, output)
    print(f"Saved → {output}")


if __name__ == "__main__":
    main()
