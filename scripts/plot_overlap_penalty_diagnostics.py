#!/usr/bin/env python3
"""
Plot per-step repair diagnostics saved by scripts/run_repair_methods.py.

Example
-------
python scripts/plot_overlap_penalty_diagnostics.py \
  --run-dir results/repair_comparison/a_cat_a_dog/seed_42 \
  --methods 11_tweedie_poe_corrector 12_overlap_penalty_corrector \
  --metrics contention_fraction overlap_energy correction_grad_norm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_infos(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _metric_series(infos: list[dict], key: str) -> list[float]:
    values = []
    for info in infos:
        value = info.get(key)
        if isinstance(value, list):
            values.append(sum(float(v) for v in value) / max(len(value), 1))
        else:
            values.append(float(value))
    return values


def main():
    parser = argparse.ArgumentParser(description="Plot saved per-step repair diagnostics")
    parser.add_argument("--run-dir", type=str, required=True, help="Pair/seed directory containing *_infos.json")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["11_tweedie_poe_corrector", "12_overlap_penalty_corrector"],
        help="Method names to plot",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["contention_fraction", "overlap_energy", "correction_grad_norm"],
        help="Metrics to plot",
    )
    parser.add_argument("--out", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out else run_dir / "overlap_penalty_diagnostics.png"

    fig, axes = plt.subplots(len(args.metrics), 1, figsize=(8, 3.0 * len(args.metrics)), sharex=True)
    if len(args.metrics) == 1:
        axes = [axes]

    for method in args.methods:
        infos_path = run_dir / f"{method}_infos.json"
        if not infos_path.exists():
            continue
        infos = _load_infos(infos_path)
        steps = list(range(len(infos)))
        for ax, metric in zip(axes, args.metrics):
            ax.plot(steps, _metric_series(infos, metric), label=method)
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("denoising step")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
