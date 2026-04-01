"""
W2 — Taxonomy d_T Bar Chart + KDE
===================================
Loads taxonomy_d_T_summary.csv (produced by W1 / aggregate_taxonomy_results.py) and
produces a 2-panel figure:

    Left  — bar chart of mean d_T ± std per group (4 bars, monotonically increasing G1→G4)
    Right — KDE of per-seed d_T stratified by group (4 overlapping densities)

Usage
-----
    python scripts/plot_taxonomy_bars.py
    python scripts/plot_taxonomy_bars.py --csv experiments/eccv2026/taxonomy_qualitative/taxonomy_d_T_summary.csv
    python scripts/plot_taxonomy_bars.py --out media/taxonomy_d_T_bars.png

Output
------
    experiments/eccv2026/taxonomy_qualitative/taxonomy_d_T_bars.png  (default)
    proposal/.../media/taxonomy_d_T_bars.png                          (copy)
"""

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

GROUP_ORDER = [
    "group1_cooccurrence",
    "group2_disentangled",
    "group3_ood",
    "group4_collision",
]

GROUP_LABELS = {
    "group1_cooccurrence": "G1\nCo-occurrence",
    "group2_disentangled": "G2\nDisentangled",
    "group3_ood":          "G3\nOOD",
    "group4_collision":    "G4\nCollision",
}

# Colour palette: blue → teal → orange → red (intuitive severity ordering)
GROUP_COLORS = {
    "group1_cooccurrence": "#4878CF",
    "group2_disentangled": "#6ACC65",
    "group3_ood":          "#F5A623",
    "group4_collision":    "#D7191C",
}


# ---------------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------------

def load_csv(csv_path: Path) -> dict[str, list[float]]:
    """Returns {group: [d_T, ...]} from the CSV."""
    group_vals: dict[str, list[float]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = row["taxonomy_group"]
            try:
                group_vals[g].append(float(row["d_T"]))
            except (ValueError, KeyError):
                pass
    return dict(group_vals)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bars_and_kde(
    group_vals: dict[str, list[float]],
    output_path: Path,
) -> None:
    groups_present = [g for g in GROUP_ORDER if g in group_vals]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(
        r"Composability gap $d_T$ by taxonomy group  (PoE vs Monolithic terminal distance)",
        fontsize=12,
        y=1.02,
    )

    # ------------------------------------------------------------------
    # Left panel: bar chart mean ± std
    # ------------------------------------------------------------------
    ax = axes[0]
    xs = np.arange(len(groups_present))
    means = [np.mean(group_vals[g]) for g in groups_present]
    stds  = [np.std(group_vals[g], ddof=1) if len(group_vals[g]) > 1 else 0.0
             for g in groups_present]
    colors = [GROUP_COLORS[g] for g in groups_present]
    n_per_group = [len(group_vals[g]) for g in groups_present]

    bars = ax.bar(xs, means, yerr=stds, capsize=6,
                  color=colors, alpha=0.85, edgecolor="white", linewidth=0.8,
                  error_kw={"elinewidth": 1.5, "capthick": 1.5})

    # Annotate each bar with n
    for x, mean, n in zip(xs, means, n_per_group):
        ax.text(x, mean * 0.05, f"n={n}", ha="center", va="bottom",
                fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([GROUP_LABELS[g] for g in groups_present], fontsize=9)
    ax.set_ylabel(r"$d_T$ (PoE vs monolithic, L2)", fontsize=10)
    ax.set_xlabel("Taxonomy group", fontsize=10)
    ax.set_title("Mean $d_T$ ± std per group", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)

    # ------------------------------------------------------------------
    # Right panel: KDE by group
    # ------------------------------------------------------------------
    ax2 = axes[1]
    all_vals = [v for vals in group_vals.values() for v in vals]
    x_min = max(0.0, min(all_vals) - 5)
    x_max = max(all_vals) + 10
    x_grid = np.linspace(x_min, x_max, 400)

    for g in groups_present:
        vals = np.array(group_vals[g])
        color = GROUP_COLORS[g]
        label = GROUP_LABELS[g].replace("\n", " ")

        if len(vals) >= 2:
            kde = gaussian_kde(vals, bw_method="scott")
            density = kde(x_grid)
            ax2.plot(x_grid, density, color=color, linewidth=2, label=label)
            ax2.fill_between(x_grid, density, alpha=0.15, color=color)
        else:
            # Single observation: draw a vertical line
            ax2.axvline(vals[0], color=color, linewidth=2, linestyle="--", label=f"{label} (n=1)")

        # Mark individual observations as rug
        jitter = np.random.default_rng(0).uniform(-0.0003, 0.0003, len(vals))
        ax2.plot(vals, np.zeros_like(vals) + jitter, "|", color=color,
                 markersize=8, alpha=0.7, markeredgewidth=1.5)

    ax2.set_xlabel(r"$d_T$ (PoE vs monolithic, L2)", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.set_title(r"$d_T$ distribution per group (KDE)", fontsize=10)
    ax2.legend(fontsize=8, framealpha=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax2.set_axisbelow(True)
    ax2.set_xlim(x_min, x_max)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Statistical tests (optional, printed to stdout)
# ---------------------------------------------------------------------------

def run_stats(group_vals: dict[str, list[float]]) -> None:
    groups_present = [g for g in GROUP_ORDER if g in group_vals]
    all_group_data = [np.array(group_vals[g]) for g in groups_present]

    # Kruskal-Wallis across all 4 groups
    try:
        from scipy.stats import kruskal, mannwhitneyu
        stat, p = kruskal(*all_group_data)
        print(f"\nKruskal-Wallis H={stat:.3f}  p={p:.4f}  (all groups)")

        # Wilcoxon rank-sum: G1+G2 vs G3+G4
        low = np.concatenate([group_vals.get(g, []) for g in groups_present[:2]])
        high = np.concatenate([group_vals.get(g, []) for g in groups_present[2:]])
        if len(low) >= 2 and len(high) >= 2:
            u, p2 = mannwhitneyu(low, high, alternative="less")
            print(f"Mann-Whitney (G1+G2 < G3+G4): U={u:.1f}  p={p2:.4f}")
    except ImportError:
        print("scipy not available — skipping statistical tests")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="2-panel taxonomy d_T figure: bar chart + KDE by group."
    )
    parser.add_argument("--csv", type=str, default="",
                        help="Path to taxonomy_d_T_summary.csv (default: auto-discover)")
    parser.add_argument("--out", type=str, default="",
                        help="Output PNG path (default: taxonomy_qualitative dir + copy to media/)")
    parser.add_argument("--no-stats", action="store_true",
                        help="Skip statistical tests")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else (
        PROJECT_ROOT / "experiments" / "eccv2026" / "taxonomy_qualitative" / "taxonomy_d_T_summary.csv"
    )
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}\n"
              f"Run scripts/aggregate_taxonomy_results.py first.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.out) if args.out else (
        PROJECT_ROOT / "experiments" / "eccv2026" / "taxonomy_qualitative" / "taxonomy_d_T_bars.png"
    )

    group_vals = load_csv(csv_path)
    print(f"Loaded {sum(len(v) for v in group_vals.values())} records from {csv_path}")
    for g in GROUP_ORDER:
        if g in group_vals:
            vals = group_vals[g]
            print(f"  {g:<30}  n={len(vals):>3}  mean={np.mean(vals):.2f}  std={np.std(vals, ddof=max(1,len(vals))-1):.2f}")

    plot_bars_and_kde(group_vals, output_path)

    if not args.no_stats:
        run_stats(group_vals)

    # Copy to proposal media dir
    media_dir = (
        PROJECT_ROOT / "proposal" / "proposal_stage_3"
        / "chapters" / "research_method" / "media"
    )
    if media_dir.exists():
        dest = media_dir / "taxonomy_d_T_bars.png"
        shutil.copy(output_path, dest)
        print(f"Copied → {dest}")


if __name__ == "__main__":
    main()
