#!/usr/bin/env python3
"""Freeze a six-group SDXL roster from manual audit decisions only."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from sdxl_paper_audit_common import DEFAULT_AUDIT_MANIFEST_PATH, index_manifest_entries, load_manifest
    from taxonomy_manifest import (
        DEFAULT_FINAL_SEEDS,
        GROUP_ORDER,
        GROUP_SPECS,
        TARGET_PAIRS_PER_GROUP,
        TOTAL_PAIRS,
        taxonomy_manifest_rows,
    )
except ImportError:
    from scripts.sdxl_paper_audit_common import DEFAULT_AUDIT_MANIFEST_PATH, index_manifest_entries, load_manifest
    from scripts.taxonomy_manifest import (
        DEFAULT_FINAL_SEEDS,
        GROUP_ORDER,
        GROUP_SPECS,
        TARGET_PAIRS_PER_GROUP,
        TOTAL_PAIRS,
        taxonomy_manifest_rows,
    )


ROSTER_VERSION = 3


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_roster_manifest(audit_manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    manifest = load_manifest(audit_manifest_path)
    audit_index = index_manifest_entries(manifest)
    canonical_rows = taxonomy_manifest_rows()

    rows_by_group: dict[str, list[dict[str, Any]]] = {group_key: [] for group_key in GROUP_ORDER}
    errors: list[str] = []

    for row in canonical_rows:
        pair_slug = str(row["pair_slug"])
        audit_entry = audit_index.get(pair_slug)
        if audit_entry is None:
            errors.append(f"missing audit entry for {pair_slug}")
            continue
        row_copy = dict(row)
        row_copy["audit_entry"] = audit_entry
        rows_by_group[row_copy["taxonomy_group_key"]].append(row_copy)

    manifest_groups: list[dict[str, Any]] = []
    pair_csv_rows: list[dict[str, Any]] = []
    group_csv_rows: list[dict[str, Any]] = []

    for spec in GROUP_SPECS:
        group_key = str(spec["key"])
        group_label = str(spec["label"])
        target_count = len(spec["pairs"])
        selected_pairs: list[dict[str, Any]] = []
        dropped_pairs: list[dict[str, Any]] = []
        group_rows = list(rows_by_group.get(group_key, []))

        for rank, row in enumerate(group_rows, start=1):
            audit_entry = dict(row["audit_entry"])
            decision = str(audit_entry.get("decision", "unreviewed"))
            source_pair_dir = audit_entry.get("source_pair_dir")

            if decision == "unreviewed":
                errors.append(f"{group_key}: {row['pair_slug']} is still unreviewed")
                continue
            if decision == "keep" and not source_pair_dir:
                errors.append(f"{group_key}: {row['pair_slug']} is marked keep but has no source assets")
                continue

            base_entry = {
                "rank": rank,
                "pair": [row["prompt_a"], row["prompt_b"]],
                "pair_slug": row["pair_slug"],
                "qualitative_pair_slug": row["qualitative_pair_slug"],
                "source_run": audit_entry.get("source_run", ""),
                "source_pair_dir": source_pair_dir,
                "decision": decision,
                "rationale_tags": list(audit_entry.get("rationale_tags", [])),
                "notes": audit_entry.get("notes", ""),
            }
            if decision == "keep":
                selected_pairs.append(base_entry)
                pair_csv_rows.append(
                    {
                        "taxonomy_group_key": group_key,
                        "taxonomy_group_label": group_label,
                        **base_entry,
                    }
                )
            else:
                dropped_pairs.append(base_entry)

        if len(selected_pairs) != target_count:
            errors.append(
                f"{group_key}: selected {len(selected_pairs)}/{target_count} keep decisions; "
                f"replace or rerender the dropped pairs before freeze"
            )

        manifest_groups.append(
            {
                "taxonomy_group_key": group_key,
                "taxonomy_group_label": group_label,
                "target_pair_count": target_count,
                "selected_count": len(selected_pairs),
                "selection_complete": len(selected_pairs) == target_count,
                "representative_pair_slug": row_slug(selected_pairs[0]) if selected_pairs else None,
                "selected_pairs": selected_pairs,
                "dropped_pairs": dropped_pairs,
            }
        )
        group_csv_rows.append(
            {
                "taxonomy_group_key": group_key,
                "taxonomy_group_label": group_label,
                "target_pair_count": target_count,
                "selected_count": len(selected_pairs),
                "selection_complete": len(selected_pairs) == target_count,
            }
        )

    roster_manifest = {
        "roster_manifest_version": ROSTER_VERSION,
        "model_family": "sdxl",
        "created_at": datetime.now().isoformat(),
        "source_audit_manifest": str(audit_manifest_path),
        "selection_policy": {
            "authority": "manual audit decisions only",
            "pair_order": "canonical taxonomy order",
            "target_pairs_per_group": TARGET_PAIRS_PER_GROUP,
            "target_total_pairs": TOTAL_PAIRS,
            "final_seeds": list(DEFAULT_FINAL_SEEDS),
            "grid_seed": int(DEFAULT_FINAL_SEEDS[0]),
        },
        "groups": manifest_groups,
        "selected_pair_slugs": [
            pair["pair_slug"]
            for group in manifest_groups
            for pair in group["selected_pairs"]
        ],
        "selected_qualitative_pair_slugs": [
            pair["qualitative_pair_slug"]
            for group in manifest_groups
            for pair in group["selected_pairs"]
        ],
        "summary": {
            "n_groups": len(manifest_groups),
            "n_selected_pairs": sum(len(group["selected_pairs"]) for group in manifest_groups),
            "selection_complete": all(bool(group["selection_complete"]) for group in manifest_groups),
        },
    }
    return roster_manifest, pair_csv_rows, group_csv_rows, errors


def row_slug(entry: dict[str, Any]) -> str | None:
    if not entry:
        return None
    return str(entry.get("pair_slug")) if entry.get("pair_slug") else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a six-group SDXL roster from manual screening decisions."
    )
    parser.add_argument(
        "--audit-manifest",
        type=Path,
        default=DEFAULT_AUDIT_MANIFEST_PATH,
        help="Manual audit manifest JSON path.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output roster path (default: <audit-manifest parent>/sdxl_paper_roster.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = (
        Path(args.output)
        if args.output
        else args.audit_manifest.parent / "sdxl_paper_roster.json"
    )

    manifest, pair_csv_rows, group_csv_rows, errors = build_roster_manifest(args.audit_manifest)
    if errors:
        formatted = "\n".join(f"  - {msg}" for msg in errors)
        raise SystemExit(
            "Refusing to freeze the SDXL paper roster because the manual audit is incomplete.\n"
            f"{formatted}"
        )

    output_path.write_text(json.dumps(manifest, indent=2))
    stem = output_path.with_suffix("")
    _write_csv(stem.with_name(f"{stem.name}_pairs.csv"), pair_csv_rows)
    _write_csv(stem.with_name(f"{stem.name}_groups.csv"), group_csv_rows)
    print(f"Saved frozen roster -> {output_path}")


if __name__ == "__main__":
    main()
