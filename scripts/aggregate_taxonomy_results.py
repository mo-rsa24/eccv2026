"""
W1 — Aggregate Taxonomy Results
================================
Scrapes all summary.json files under experiments/eccv2026/taxonomy_qualitative/,
extracts the PoE-vs-monolithic terminal distance d_T from endpoint_distances_l2,
and writes a tidy CSV for downstream plotting (W2).

Usage
-----
    python scripts/aggregate_taxonomy_results.py
    python scripts/aggregate_taxonomy_results.py --input-dir experiments/eccv2026/taxonomy_qualitative
    python scripts/aggregate_taxonomy_results.py --output taxonomy_d_T_summary.csv

Output columns
--------------
    taxonomy_group, pair_slug, prompt_a, prompt_b, seed, d_T, model_id
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Group order for consistent ordering in the output
# ---------------------------------------------------------------------------
GROUP_ORDER = [
    "group1_cooccurrence",
    "group2_disentangled",
    "group3_ood",
    "group4_collision",
]


def _find_poe_vs_monolithic_key(endpoint_dict: dict) -> str | None:
    """Return the key whose value is the PoE-vs-monolithic distance.

    The key format produced by trajectory_dynamics_experiment.py is:
        'CLIP AND: "<a> and <b>" vs PoE: "<a>" × "<b>"'
    We match on this pattern robustly.
    """
    poe_pattern = re.compile(r"CLIP AND:.*vs PoE:", re.IGNORECASE)
    for k in endpoint_dict:
        if poe_pattern.search(k):
            return k
    return None


def scrape_summary(summary_path: Path) -> dict | None:
    """Parse one summary.json and return a flat record dict or None on failure."""
    try:
        with open(summary_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  WARN: could not read {summary_path}: {exc}", file=sys.stderr)
        return None

    cfg = data.get("config", {})
    endpoint_l2 = data.get("endpoint_distances_l2", {})

    if not endpoint_l2:
        return None

    key = _find_poe_vs_monolithic_key(endpoint_l2)
    if key is None:
        # No PoE-vs-monolithic entry — skip (e.g., --no-poe runs)
        return None

    d_T = endpoint_l2[key]

    # taxonomy_group: prefer config field; fall back to path heuristic
    taxonomy_group = cfg.get("taxonomy_group", "")
    if not taxonomy_group:
        # Infer from the path: .../taxonomy_qualitative/<group>/...
        parts = summary_path.parts
        for i, part in enumerate(parts):
            if part.startswith("group") and "_" in part:
                taxonomy_group = part
                break

    return {
        "taxonomy_group": taxonomy_group,
        "prompt_a": cfg.get("prompt_a", ""),
        "prompt_b": cfg.get("prompt_b", ""),
        "seed": cfg.get("seed", -1),
        "d_T": d_T,
        "model_id": cfg.get("model_id", ""),
        "pair_slug": summary_path.parent.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate taxonomy summary.json files → taxonomy_d_T_summary.csv"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="Root directory to search (default: experiments/eccv2026/taxonomy_qualitative/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output CSV path (default: experiments/eccv2026/taxonomy_qualitative/taxonomy_d_T_summary.csv)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else (
        PROJECT_ROOT / "experiments" / "eccv2026" / "taxonomy_qualitative"
    )
    if not input_dir.exists():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else (
        input_dir / "taxonomy_d_T_summary.csv"
    )

    # ------------------------------------------------------------------
    # Recursively find all summary.json files
    # ------------------------------------------------------------------
    summary_files = sorted(input_dir.rglob("summary.json"))
    print(f"Found {len(summary_files)} summary.json files under {input_dir}")

    records: list[dict] = []
    for sp in summary_files:
        rec = scrape_summary(sp)
        if rec is not None:
            records.append(rec)

    if not records:
        print("No valid records found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Sort: group order → prompt_a → prompt_b → seed
    group_rank = {g: i for i, g in enumerate(GROUP_ORDER)}
    records.sort(
        key=lambda r: (
            group_rank.get(r["taxonomy_group"], 99),
            r["prompt_a"],
            r["prompt_b"],
            r["seed"],
        )
    )

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    fieldnames = ["taxonomy_group", "prompt_a", "prompt_b", "seed", "d_T", "model_id", "pair_slug"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} records → {output_path}")

    # Quick summary per group
    from collections import defaultdict
    import statistics

    group_d_T: dict[str, list[float]] = defaultdict(list)
    for r in records:
        group_d_T[r["taxonomy_group"]].append(r["d_T"])

    print("\nPer-group d_T summary:")
    print(f"  {'Group':<30}  {'n':>4}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    print(f"  {'-'*30}  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for gkey in GROUP_ORDER:
        vals = group_d_T.get(gkey, [])
        if not vals:
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {gkey:<30}  {len(vals):>4}  {mean:>8.2f}  {std:>8.2f}  {min(vals):>8.2f}  {max(vals):>8.2f}")


if __name__ == "__main__":
    main()
