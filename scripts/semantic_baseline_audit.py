#!/usr/bin/env python3
"""Build semantic-baseline audit artifacts from pair-type-aware joint probes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from semantic_baseline_common import (
    DEFAULT_PAIR_PASS_THRESHOLD,
    build_audit_from_joint_probe_file,
    resolve_joint_probe_path,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize joint-probe outputs into a semantic-baseline eligibility audit.",
    )
    parser.add_argument(
        "--data-dir",
        default="experiments/inversion/gap_analysis",
        help="Run root containing joint_probe_scores.json.",
    )
    parser.add_argument(
        "--pair-pass-threshold",
        type=float,
        default=DEFAULT_PAIR_PASS_THRESHOLD,
        help="Pair eligibility threshold on mono semantic-pass rate (default: 0.75).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path (default: {data_dir}/semantic_baseline_audit.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    joint_probe_path = resolve_joint_probe_path(data_dir)
    if not joint_probe_path.exists():
        raise FileNotFoundError(
            f"joint_probe_scores.json not found under {data_dir}. "
            "Run scripts/eval_joint_probes.py first."
        )

    out_path = Path(args.output) if args.output else data_dir / "semantic_baseline_audit.json"
    audit = build_audit_from_joint_probe_file(
        joint_probe_path,
        pair_pass_threshold=float(args.pair_pass_threshold),
    )
    out_path.write_text(json.dumps(audit, indent=2))

    stem = out_path.with_suffix("")
    _write_csv(stem.with_name(f"{stem.name}_seed.csv"), audit["seed_rows"])
    _write_csv(stem.with_name(f"{stem.name}_pair.csv"), audit["pair_rows"])
    _write_csv(stem.with_name(f"{stem.name}_group.csv"), audit["group_rows"])
    _write_csv(stem.with_name(f"{stem.name}_roster.csv"), audit["roster_rows"])
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
