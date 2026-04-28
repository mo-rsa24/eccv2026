#!/usr/bin/env python3
"""Freeze the final 10 replacement Group 6 pairs from manual audit decisions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from group6_replacement_common import DEFAULT_GROUP6_CANDIDATE_MANIFEST, pair_row
except ImportError:
    from scripts.group6_replacement_common import DEFAULT_GROUP6_CANDIDATE_MANIFEST, pair_row


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT_MANIFEST = (
    PROJECT_ROOT / "experiments" / "eccv2026" / "sdxl_final" / "group6_replacement_audit_manifest.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiments" / "eccv2026" / "sdxl_final" / "group6_replacement_roster.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze Group 6 replacement roster from manual audit decisions.")
    parser.add_argument("--audit-manifest", type=Path, default=DEFAULT_AUDIT_MANIFEST)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_GROUP6_CANDIDATE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_manifest = json.loads(args.audit_manifest.read_text())
    candidate_manifest = json.loads(args.candidate_manifest.read_text())
    replacement_group_key = str(candidate_manifest["replacement_group_key"])
    replacement_group_label = str(candidate_manifest["replacement_group_label"])
    target_pair_count = int(candidate_manifest.get("target_pair_count", 10))

    selected_pairs: list[dict[str, Any]] = []
    for entry in audit_manifest.get("entries", []):
        if str(entry.get("decision", "unreviewed")) != "keep":
            continue
        prompt_a = str(entry["prompt_a"])
        prompt_b = str(entry["prompt_b"])
        row = pair_row(replacement_group_key, prompt_a, prompt_b)
        row["source_pair_dir"] = entry.get("source_pair_dir")
        row["decision"] = "keep"
        row["rationale_tags"] = list(entry.get("rationale_tags", []))
        row["notes"] = entry.get("notes", "")
        selected_pairs.append(row)

    unreviewed = [entry["pair_slug"] for entry in audit_manifest.get("entries", []) if entry.get("decision", "unreviewed") == "unreviewed"]
    if unreviewed:
        raise SystemExit(
            "Refusing to freeze replacement Group 6 because some candidates are still unreviewed:\n"
            + "\n".join(f"  - {slug}" for slug in unreviewed)
        )
    if len(selected_pairs) != target_pair_count:
        raise SystemExit(
            f"Refusing to freeze replacement Group 6: selected {len(selected_pairs)}/{target_pair_count} keep decisions."
        )

    payload = {
        "roster_manifest_version": 1,
        "created_at": datetime.now().isoformat(),
        "candidate_manifest": str(args.candidate_manifest),
        "source_audit_manifest": str(args.audit_manifest),
        "replacement_group_key": replacement_group_key,
        "replacement_group_label": replacement_group_label,
        "target_pair_count": target_pair_count,
        "selected_pairs": selected_pairs,
        "selected_qualitative_pair_slugs": [row["qualitative_pair_slug"] for row in selected_pairs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Saved Group 6 replacement roster -> {args.output}")


if __name__ == "__main__":
    main()
