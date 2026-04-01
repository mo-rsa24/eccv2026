"""
W3 — Multi-group Trajectory Manifold Grid
==========================================
Loads per-pair trajectory_data.json from all taxonomy pairs and renders a
publication-quality grid figure where:

    Columns = taxonomy groups (G1–G4)
    Rows    = pairs within each group

Each cell shows the 4-condition latent trajectory fan-out plot in the
pre-computed PCA/MDS projection space (projected_x, projected_y from
trajectory_data.json).  No re-running of the model is required.

Usage
-----
    python scripts/plot_trajectory_grid.py
    python scripts/plot_trajectory_grid.py --input-dir experiments/eccv2026/taxonomy_qualitative
    python scripts/plot_trajectory_grid.py --out experiments/eccv2026/taxonomy_qualitative/trajectory_manifold_grid.png

Output
------
    experiments/eccv2026/taxonomy_qualitative/trajectory_manifold_grid.png (default)
    proposal/.../media/trajectory_manifold_grid.png                         (copy)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

GROUP_ORDER = [
    "group1_cooccurrence",
    "group2_disentangled",
    "group3_ood",
    "group4_collision",
]

GROUP_TITLES = {
    "group1_cooccurrence": "G1 — Co-occurrence",
    "group2_disentangled": "G2 — Disentangled",
    "group3_ood":          "G3 — OOD",
    "group4_collision":    "G4 — Collision",
}

# Condition colour scheme: matches _TAXONOMY_COLUMN_ORDER
COND_COLORS = {
    "prompt_a":   "#4878CF",   # blue    — solo A
    "prompt_b":   "#6ACC65",   # green   — solo B
    "monolithic": "#F5A623",   # orange  — monolithic
    "poe":        "#D7191C",   # red     — PoE/AND
}
COND_LABELS_SHORT = {
    "prompt_a":   "Solo A",
    "prompt_b":   "Solo B",
    "monolithic": "Monolithic",
    "poe":        "PoE/AND",
}
COND_ORDER = ["prompt_a", "prompt_b", "monolithic", "poe"]


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def discover_pairs(input_dir: Path) -> dict[str, list[Path]]:
    """Return {group_key: [pair_dir, ...]} using trajectory_data.json presence."""
    group_pairs: dict[str, list[Path]] = {g: [] for g in GROUP_ORDER}

    for gkey in GROUP_ORDER:
        gdir = input_dir / gkey
        if not gdir.exists():
            continue
        for pair_dir in sorted(gdir.iterdir()):
            if not pair_dir.is_dir():
                continue
            # Accept direct trajectory_data.json or first seed subdir
            if (pair_dir / "trajectory_data.json").exists():
                group_pairs[gkey].append(pair_dir)
            else:
                # Multi-seed: pick first seed subdir that has trajectory_data.json
                seed_subdirs = sorted(
                    [d for d in pair_dir.iterdir()
                     if d.is_dir() and d.name.startswith("seed_")]
                )
                if seed_subdirs:
                    first = seed_subdirs[0]
                    if (first / "trajectory_data.json").exists():
                        group_pairs[gkey].append(first)

    return group_pairs


def load_traj(pair_dir: Path) -> dict | None:
    traj_path = pair_dir / "trajectory_data.json"
    if not traj_path.exists():
        return None
    try:
        with open(traj_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  WARN: could not read {traj_path}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Single-cell trajectory plot
# ---------------------------------------------------------------------------

def plot_trajectory_cell(ax: plt.Axes, traj: dict, show_legend: bool = False) -> None:
    """Plot 4-condition trajectory fan in the pre-computed 2D projection."""
    per_cond = traj.get("per_condition", {})
    conditions = [c for c in COND_ORDER if c in per_cond]
    n_steps = traj.get("n_steps", 0)

    for cond in conditions:
        pc = per_cond[cond]
        xs = np.array(pc.get("projected_x", []))
        ys = np.array(pc.get("projected_y", []))
        if len(xs) == 0:
            continue
        color = COND_COLORS.get(cond, "grey")
        label = COND_LABELS_SHORT.get(cond, cond)

        # Full trajectory line
        ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.7, zorder=2)
        # Start marker (circle at t=T — high noise)
        ax.scatter(xs[0], ys[0], s=30, color=color, marker="o", zorder=4, alpha=0.9)
        # End marker (star at t=0 — clean image)
        ax.scatter(xs[-1], ys[-1], s=60, color=color, marker="*", zorder=5)

        if show_legend:
            ax.plot([], [], color=color, linewidth=1.5, label=label)

    # Shared start marker at t=T (all trajectories start from same x_T)
    first_cond = conditions[0] if conditions else None
    if first_cond:
        xs0 = per_cond[first_cond]["projected_x"]
        ys0 = per_cond[first_cond]["projected_y"]
        ax.scatter(xs0[0], ys0[0], s=80, color="black", marker="X", zorder=6,
                   label="Shared $x_T$" if show_legend else None)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#cccccc")


def _pair_label(pair_dir: Path) -> str:
    """Human-readable pair label from directory name."""
    slug = pair_dir.name
    # Strip seed_ prefix if present
    if slug.startswith("seed_"):
        slug = pair_dir.parent.name
    # Un-slugify: remove leading "a_", "an_", restore spaces
    parts = slug.split("__x__")
    if len(parts) == 2:
        a = parts[0].replace("_", " ").strip()
        b = parts[1].replace("_", " ").strip()
        return f"{a}\n× {b}"
    return slug.replace("_", " ")[:40]


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------

def make_grid(group_pairs: dict[str, list[Path]], output_path: Path) -> None:
    groups_present = [g for g in GROUP_ORDER if group_pairs.get(g)]
    if not groups_present:
        print("No trajectory data found.", file=sys.stderr)
        return

    n_cols = len(groups_present)
    max_rows = max(len(group_pairs[g]) for g in groups_present)

    cell_size = 2.6
    fig_w = n_cols * cell_size + 1.2      # extra for row labels
    fig_h = max_rows * cell_size + 1.2    # extra for column headers

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Reserve space: top strip for column titles, left strip for row labels
    left_margin   = 0.10
    right_margin  = 0.02
    top_margin    = 0.08
    bottom_margin = 0.08
    hgap = 0.01
    vgap = 0.01

    total_w = 1.0 - left_margin - right_margin
    total_h = 1.0 - top_margin - bottom_margin
    cell_w = (total_w - hgap * (n_cols - 1)) / n_cols
    cell_h = (total_h - vgap * (max_rows - 1)) / max_rows

    # Column title text
    for c_idx, gkey in enumerate(groups_present):
        x_center = left_margin + c_idx * (cell_w + hgap) + cell_w / 2
        fig.text(x_center, 1.0 - top_margin / 2,
                 GROUP_TITLES[gkey], ha="center", va="center",
                 fontsize=9, fontweight="bold")

    for c_idx, gkey in enumerate(groups_present):
        pairs = group_pairs[gkey]
        for r_idx, pair_dir in enumerate(pairs):
            traj = load_traj(pair_dir)
            if traj is None:
                continue

            left = left_margin + c_idx * (cell_w + hgap)
            bottom = (1.0 - top_margin) - (r_idx + 1) * cell_h - r_idx * vgap
            ax = fig.add_axes([left, bottom, cell_w, cell_h])

            show_legend = (c_idx == 0 and r_idx == max_rows - 1)
            plot_trajectory_cell(ax, traj, show_legend=show_legend)

            # Row label on leftmost column
            if c_idx == 0:
                label = _pair_label(pair_dir)
                ax.set_ylabel(label, fontsize=6, rotation=0,
                              ha="right", va="center", labelpad=6)

    # Shared legend at bottom-left
    legend_patches = [
        mpatches.Patch(color=COND_COLORS[c], label=COND_LABELS_SHORT[c])
        for c in COND_ORDER
    ]
    legend_patches.append(
        mpatches.Patch(color="black", label="Shared $x_T$")
    )
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(legend_patches), fontsize=7,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble multi-group trajectory manifold grid from trajectory_data.json files."
    )
    parser.add_argument("--input-dir", type=str, default="",
                        help="Root directory (default: experiments/eccv2026/taxonomy_qualitative/)")
    parser.add_argument("--out", type=str, default="",
                        help="Output PNG (default: input-dir/trajectory_manifold_grid.png)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else (
        PROJECT_ROOT / "experiments" / "eccv2026" / "taxonomy_qualitative"
    )
    if not input_dir.exists():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.out) if args.out else (input_dir / "trajectory_manifold_grid.png")

    group_pairs = discover_pairs(input_dir)
    total = sum(len(v) for v in group_pairs.values())
    print(f"Discovered {total} pair trajectory files:")
    for g in GROUP_ORDER:
        pairs = group_pairs.get(g, [])
        print(f"  {g}: {len(pairs)} pairs")

    if total == 0:
        print("No trajectory_data.json files found. Run --taxonomy-grid first.", file=sys.stderr)
        sys.exit(1)

    make_grid(group_pairs, output_path)

    # Copy to proposal media dir
    media_dir = (
        PROJECT_ROOT / "proposal" / "proposal_stage_3"
        / "chapters" / "research_method" / "media"
    )
    if media_dir.exists():
        dest = media_dir / "trajectory_manifold_grid.png"
        shutil.copy(output_path, dest)
        print(f"Copied → {dest}")


if __name__ == "__main__":
    main()
