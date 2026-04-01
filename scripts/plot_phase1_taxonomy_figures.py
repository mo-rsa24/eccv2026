"""
Plot the main Phase 1 large-study figures from PoE-anchored tidy tables.
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phase1_taxonomy_common import (
    GROUP_COLORS,
    GROUP_LABELS,
    GROUP_ORDER,
    PHASE_BINS,
    PHASE_COLORS,
    PHASE_LABELS,
    PHASE_SPAN_COLORS,
    PROJECT_ROOT,
    ci95,
    ecdf,
    maybe_float,
    maybe_int,
    present_groups,
    read_csv,
)


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv(path)


def _jitter(xs: np.ndarray, width: float, size: int, seed: int) -> np.ndarray:
    if size == 0:
        return np.asarray([])
    rng = np.random.default_rng(seed)
    return xs + rng.uniform(-width, width, size=size)


def _group_pair_values(rows: list[dict[str, str]], value_field: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = maybe_float(row.get(value_field))
        if value is None:
            continue
        grouped[row["taxonomy_group"]].append(value)
    return grouped


def _style_group_axis(ax: plt.Axes, groups: list[str]) -> None:
    xs = np.arange(len(groups))
    ax.set_xticks(xs)
    ax.set_xticklabels([GROUP_LABELS[group] for group in groups], fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.4)


def _add_phase_spans(ax: plt.Axes) -> None:
    for phase_name, start, end in PHASE_BINS:
        ax.axvspan(start, end, color=PHASE_SPAN_COLORS[phase_name], alpha=0.32, zorder=0)
    ax.axvline(17, color=PHASE_COLORS["early"], linestyle="--", linewidth=1.2, alpha=0.9)
    ax.axvline(34, color=PHASE_COLORS["mid"], linestyle="--", linewidth=1.2, alpha=0.9)


def _annotate_phase_labels(ax: plt.Axes) -> None:
    ymin, ymax = ax.get_ylim()
    y = ymax - 0.04 * (ymax - ymin)
    for phase_name, start, end in PHASE_BINS:
        ax.text(
            (start + end) / 2.0,
            y,
            PHASE_LABELS[phase_name],
            ha="center",
            va="top",
            fontsize=9,
            color=PHASE_COLORS[phase_name],
            fontweight="bold",
        )


def _copy_outputs(output_paths: list[Path], proposal_media_dir: Path) -> None:
    if not proposal_media_dir.exists():
        return
    proposal_media_dir.mkdir(parents=True, exist_ok=True)
    for output_path in output_paths:
        shutil.copy(output_path, proposal_media_dir / output_path.name)


def plot_terminal_gap(
    terminal_pair_rows: list[dict[str, str]],
    terminal_seed_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    groups = [group for group in GROUP_ORDER if any(row["taxonomy_group"] == group for row in terminal_pair_rows)]
    pair_values = _group_pair_values(terminal_pair_rows, "d_T_mono_poe_mean")
    seed_values = _group_pair_values(terminal_seed_rows, "d_T_mono_poe")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.0, 4.8))
    fig.suptitle(
        r"Phase 1 terminal gap: pair means and pooled seed distributions for $d_T^{\mathrm{mono}\rightarrow\mathrm{poe}}$",
        fontsize=12,
        y=1.02,
    )

    xs = np.arange(len(groups))
    for idx, group in enumerate(groups):
        vals = pair_values.get(group, [])
        if vals:
            jittered = _jitter(np.full(len(vals), xs[idx]), width=0.09, size=len(vals), seed=100 + idx)
            ax_left.scatter(
                jittered,
                vals,
                s=42,
                color=GROUP_COLORS[group],
                alpha=0.82,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            mean, lo, hi = ci95(vals)
            if mean is not None and lo is not None and hi is not None:
                ax_left.errorbar(
                    xs[idx],
                    mean,
                    yerr=[[mean - lo], [hi - mean]],
                    fmt="o",
                    color="black",
                    ecolor="black",
                    elinewidth=1.6,
                    capsize=5,
                    markersize=5,
                    zorder=4,
                )
            ax_left.text(
                xs[idx],
                max(vals) + 0.03 * max(vals),
                f"n={len(vals)}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#555555",
            )

    _style_group_axis(ax_left, groups)
    ax_left.set_ylabel(r"Pair-mean $d_T^{\mathrm{mono}\rightarrow\mathrm{poe}}$", fontsize=10)
    ax_left.set_title("Inferential view: pair means + 95% CI", fontsize=10)

    for group in groups:
        values = seed_values.get(group, [])
        if not values:
            continue
        x_ecdf, y_ecdf = ecdf(values)
        ax_right.step(
            x_ecdf,
            y_ecdf,
            where="post",
            linewidth=2.2,
            color=GROUP_COLORS[group],
            label=GROUP_LABELS[group].replace("\n", " "),
        )

    ax_right.set_xlabel(r"Pooled seed $d_T^{\mathrm{mono}\rightarrow\mathrm{poe}}$", fontsize=10)
    ax_right.set_ylabel("ECDF", fontsize=10)
    ax_right.set_title("Descriptive view: pooled seed distributions", fontsize=10)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.grid(True, linewidth=0.4, alpha=0.35)
    ax_right.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_gap(trajectory_pair_rows: list[dict[str, str]], output_path: Path) -> None:
    groups = present_groups(trajectory_pair_rows)
    grouped_steps: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in trajectory_pair_rows:
        value = maybe_float(row.get("d_t_mono_poe_mean"))
        step = maybe_int(row.get("step"))
        if value is None or step is None:
            continue
        grouped_steps[(row["taxonomy_group"], step)].append(value)

    fig, ax = plt.subplots(1, 1, figsize=(10.6, 5.1))
    _add_phase_spans(ax)

    for group in groups:
        steps = sorted({step for g, step in grouped_steps if g == group})
        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []
        for step in steps:
            mean, lo, hi = ci95(grouped_steps[(group, step)])
            if mean is None or lo is None or hi is None:
                continue
            means.append(mean)
            lowers.append(lo)
            uppers.append(hi)
        if not steps or not means:
            continue
        ax.plot(
            steps[: len(means)],
            means,
            linewidth=2.3,
            color=GROUP_COLORS[group],
            label=GROUP_LABELS[group].replace("\n", " "),
        )
        ax.fill_between(
            steps[: len(means)],
            lowers,
            uppers,
            color=GROUP_COLORS[group],
            alpha=0.14,
        )

    ax.set_xlim(0, 50)
    ax.set_xlabel("Denoising step", fontsize=10)
    ax.set_ylabel(r"Group mean of pair-mean $d_t^{\mathrm{mono}\rightarrow\mathrm{poe}}$", fontsize=10)
    ax.set_title(
        "Phase 1 latent trajectory gap with early / mid / late decomposition",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(fontsize=8, framealpha=0.9, ncol=2)
    _annotate_phase_labels(ax)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_phase_contributions(phase_pair_rows: list[dict[str, str]], output_path: Path) -> None:
    groups = present_groups(phase_pair_rows)
    phase_fields = [
        ("early", "delta_early_mono_poe_mean"),
        ("mid", "delta_mid_mono_poe_mean"),
        ("late", "delta_late_mono_poe_mean"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(10.8, 5.0))
    xs = np.arange(len(groups))
    offsets = np.linspace(-0.24, 0.24, num=len(phase_fields))
    width = 0.22

    for offset, (phase_name, field_name) in zip(offsets, phase_fields):
        for idx, group in enumerate(groups):
            vals = [
                value
                for row in phase_pair_rows
                if row["taxonomy_group"] == group
                for value in [maybe_float(row.get(field_name))]
                if value is not None
            ]
            if not vals:
                continue
            mean, lo, hi = ci95(vals)
            x = xs[idx] + offset
            ax.bar(
                x,
                mean,
                width=width,
                color=PHASE_COLORS[phase_name],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.8,
                label=PHASE_LABELS[phase_name] if idx == 0 else None,
                zorder=2,
            )
            if mean is not None and lo is not None and hi is not None:
                ax.errorbar(
                    x,
                    mean,
                    yerr=[[mean - lo], [hi - mean]],
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.4,
                    capsize=4,
                    zorder=3,
                )
            jittered = _jitter(np.full(len(vals), x), width=0.035, size=len(vals), seed=200 + idx)
            ax.scatter(
                jittered,
                vals,
                s=28,
                color="#222222",
                alpha=0.55,
                zorder=4,
            )

    _style_group_axis(ax, groups)
    ax.set_ylabel(r"Pair-mean growth in $d_t^{\mathrm{mono}\rightarrow\mathrm{poe}}$", fontsize=10)
    ax.set_title("Phase-bin contributions to the monolithic-to-PoE gap", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_reachability(
    reachability_pair_rows: list[dict[str, str]],
    terminal_pair_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    groups = [group for group in GROUP_ORDER if any(row["taxonomy_group"] == group for row in (reachability_pair_rows or terminal_pair_rows))]
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.0, 4.8))
    fig.suptitle(
        r"Phase 1 reachability: SD-IPC residuals and image-level similarity to the original PoE target",
        fontsize=12,
        y=1.02,
    )

    xs = np.arange(len(groups))
    has_left = False
    for idx, group in enumerate(groups):
        vals = [
            value
            for row in reachability_pair_rows
            if row["taxonomy_group"] == group
            for value in [maybe_float(row.get("d_T_sdipc_poe_mean"))]
            if value is not None
        ]
        if vals:
            has_left = True
            jittered = _jitter(np.full(len(vals), xs[idx]), width=0.09, size=len(vals), seed=300 + idx)
            ax_left.scatter(
                jittered,
                vals,
                s=42,
                color=GROUP_COLORS[group],
                alpha=0.82,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            mean, lo, hi = ci95(vals)
            if mean is not None and lo is not None and hi is not None:
                ax_left.errorbar(
                    xs[idx],
                    mean,
                    yerr=[[mean - lo], [hi - mean]],
                    fmt="o",
                    color="black",
                    ecolor="black",
                    elinewidth=1.6,
                    capsize=5,
                    markersize=5,
                    zorder=4,
                )

    mono_reference = []
    for group in groups:
        vals = [
            value
            for row in terminal_pair_rows
            if row["taxonomy_group"] == group
            for value in [maybe_float(row.get("d_T_mono_poe_mean"))]
            if value is not None
        ]
        mono_reference.append(float(np.mean(vals)) if vals else np.nan)

    if has_left:
        ax_left.plot(
            xs,
            mono_reference,
            linestyle="--",
            linewidth=1.4,
            color="#7A7A7A",
            marker="D",
            markersize=4.5,
            alpha=0.85,
            label=r"Monolithic $\rightarrow$ PoE baseline",
        )
        _style_group_axis(ax_left, groups)
        ax_left.set_ylabel(r"Pair-mean $d_T^{\mathrm{sdipc}\rightarrow\mathrm{poe}}$", fontsize=10)
        ax_left.set_title("Terminal SD-IPC residuals vs PoE", fontsize=10)
        ax_left.legend(fontsize=8, framealpha=0.9)
    else:
        ax_left.axis("off")
        ax_left.text(
            0.5,
            0.55,
            "No SD-IPC latent residuals\nfound in the reachability tables.",
            ha="center",
            va="center",
            fontsize=11,
            color="#555555",
        )

    clip_has_data = False
    cycle_has_data = False
    cycle_means: list[float] = []
    for idx, group in enumerate(groups):
        clip_vals = [
            value
            for row in reachability_pair_rows
            if row["taxonomy_group"] == group
            for value in [maybe_float(row.get("clip_cosine_sdipc_poe_mean"))]
            if value is not None
        ]
        cycle_vals = [
            value
            for row in reachability_pair_rows
            if row["taxonomy_group"] == group
            for value in [maybe_float(row.get("cycle_close_rate_sdipc_poe_mean"))]
            if value is not None
        ]
        cycle_means.append(float(np.mean(cycle_vals)) if cycle_vals else np.nan)
        if clip_vals:
            clip_has_data = True
            jittered = _jitter(np.full(len(clip_vals), xs[idx]), width=0.09, size=len(clip_vals), seed=400 + idx)
            ax_right.scatter(
                jittered,
                clip_vals,
                s=42,
                color=GROUP_COLORS[group],
                alpha=0.82,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            mean, lo, hi = ci95(clip_vals)
            if mean is not None and lo is not None and hi is not None:
                ax_right.errorbar(
                    xs[idx],
                    mean,
                    yerr=[[mean - lo], [hi - mean]],
                    fmt="o",
                    color="black",
                    ecolor="black",
                    elinewidth=1.6,
                    capsize=5,
                    markersize=5,
                    zorder=4,
                )
        if cycle_vals:
            cycle_has_data = True

    if clip_has_data:
        _style_group_axis(ax_right, groups)
        ax_right.set_ylabel("Pair-mean CLIP cosine to original PoE image", fontsize=10)
        ax_right.set_title("Image-level reachability summary", fontsize=10)
        if cycle_has_data:
            twin = ax_right.twinx()
            twin.plot(
                xs,
                cycle_means,
                color="#2F2F2F",
                marker="s",
                linewidth=1.2,
                markersize=4.5,
                alpha=0.8,
            )
            twin.set_ylabel("Cycle-close rate", fontsize=9, color="#2F2F2F")
            twin.tick_params(axis="y", labelsize=8, colors="#2F2F2F")
            twin.set_ylim(0.0, 1.05)
    else:
        ax_right.axis("off")
        ax_right.text(
            0.5,
            0.55,
            "No SD-IPC image-similarity\nrecords found.",
            ha="center",
            va="center",
            fontsize=11,
            color="#555555",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the main Phase 1 group-level figures from PoE-anchored taxonomy tables."
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default="",
        help="Directory containing the phase1_* CSV tables (default: taxonomy_qualitative/phase1_tables).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for the Phase 1 figures (default: sibling phase1_figures/).",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        default=["all"],
        choices=["all", "terminal", "trajectory", "phase", "reachability"],
        help="Which figures to emit (default: all).",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Skip copying figures to proposal/proposal_stage_3_v1/.../media.",
    )
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir) if args.tables_dir else (
        PROJECT_ROOT / "experiments" / "eccv2026" / "taxonomy_qualitative" / "phase1_tables"
    )
    if not tables_dir.exists():
        raise SystemExit(f"Tables directory not found: {tables_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else (tables_dir.parent / "phase1_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.plots)
    if "all" in selected:
        selected = {"terminal", "trajectory", "phase", "reachability"}

    terminal_seed_rows = _load_rows(tables_dir / "phase1_terminal_seed.csv")
    terminal_pair_rows = _load_rows(tables_dir / "phase1_terminal_pair.csv")
    trajectory_pair_rows = _load_rows(tables_dir / "phase1_trajectory_pair.csv")
    phase_pair_rows = _load_rows(tables_dir / "phase1_phase_bins_pair.csv")
    reachability_pair_rows = _load_rows(tables_dir / "phase1_reachability_pair.csv")

    written: list[Path] = []

    if "terminal" in selected:
        out = output_dir / "phase1_group_terminal_gap.png"
        plot_terminal_gap(terminal_pair_rows, terminal_seed_rows, out)
        written.append(out)
        print(f"Saved → {out}")

    if "trajectory" in selected:
        out = output_dir / "phase1_group_trajectory_gap.png"
        plot_trajectory_gap(trajectory_pair_rows, out)
        written.append(out)
        print(f"Saved → {out}")

    if "phase" in selected:
        out = output_dir / "phase1_group_phase_contributions.png"
        plot_phase_contributions(phase_pair_rows, out)
        written.append(out)
        print(f"Saved → {out}")

    if "reachability" in selected:
        out = output_dir / "phase1_group_reachability.png"
        plot_reachability(reachability_pair_rows, terminal_pair_rows, out)
        written.append(out)
        print(f"Saved → {out}")

    if not args.no_copy:
        media_dir = (
            PROJECT_ROOT
            / "proposal"
            / "proposal_stage_3_v1"
            / "chapters"
            / "research_method"
            / "media"
        )
        _copy_outputs(written, media_dir)
        if media_dir.exists():
            print(f"Copied {len(written)} figures → {media_dir}")


if __name__ == "__main__":
    main()
