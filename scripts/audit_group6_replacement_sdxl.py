#!/usr/bin/env python3
"""Build and export a manual review manifest for replacement Group 6 candidates."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from group6_replacement_common import DEFAULT_GROUP6_CANDIDATE_MANIFEST, candidate_pairs_from_manifest, format_pair_slug
except ImportError:
    from scripts.group6_replacement_common import DEFAULT_GROUP6_CANDIDATE_MANIFEST, candidate_pairs_from_manifest, format_pair_slug


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT_MANIFEST = (
    PROJECT_ROOT / "experiments" / "eccv2026" / "sdxl_final" / "group6_replacement_audit_manifest.json"
)

ALLOWED_DECISIONS = ["keep", "drop", "unreviewed"]


def _candidate_entry(screening_root: Path, group_key: str, prompt_a: str, prompt_b: str) -> dict[str, Any]:
    slug = format_pair_slug(prompt_a, prompt_b)
    pair_dir = screening_root / f"seed_42" / group_key / slug
    return {
        "pair_slug": slug,
        "qualitative_pair_slug": slug,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "taxonomy_group_key": group_key,
        "source_pair_dir": str(pair_dir) if pair_dir.exists() else None,
        "summary_json": str(pair_dir / "summary.json") if (pair_dir / "summary.json").exists() else None,
        "grid_assets_json": str(pair_dir / "grid_assets.json") if (pair_dir / "grid_assets.json").exists() else None,
        "trajectory_manifold_png": str(pair_dir / "trajectory_manifold.png") if (pair_dir / "trajectory_manifold.png").exists() else None,
        "decoded_images_png": str(pair_dir / "decoded_images.png") if (pair_dir / "decoded_images.png").exists() else None,
        "solo_a_png": str(pair_dir / "solo_a.png") if (pair_dir / "solo_a.png").exists() else None,
        "solo_b_png": str(pair_dir / "solo_b.png") if (pair_dir / "solo_b.png").exists() else None,
        "monolithic_png": str(pair_dir / "monolithic.png") if (pair_dir / "monolithic.png").exists() else None,
        "poe_png": str(pair_dir / "poe.png") if (pair_dir / "poe.png").exists() else None,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "manifest_version": 1,
            "allowed_decisions": ALLOWED_DECISIONS,
            "entries": [],
        }
    return json.loads(path.read_text())


def save_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def index_entries(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(entry.get("pair_slug", "")): entry for entry in manifest.get("entries", [])}


def build_manifest(screening_root: Path, candidate_manifest: Path, audit_manifest: Path) -> dict[str, Any]:
    existing = load_manifest(audit_manifest)
    existing_index = index_entries(existing)
    entries: list[dict[str, Any]] = []
    for group_key, prompt_a, prompt_b in candidate_pairs_from_manifest(candidate_manifest):
        candidate = _candidate_entry(screening_root, group_key, prompt_a, prompt_b)
        preserved = existing_index.get(candidate["pair_slug"], {})
        candidate["decision"] = preserved.get("decision", "unreviewed")
        candidate["rationale_tags"] = list(preserved.get("rationale_tags", []))
        candidate["notes"] = preserved.get("notes", "")
        candidate["reviewer"] = preserved.get("reviewer", "")
        candidate["reviewed_at"] = preserved.get("reviewed_at")
        entries.append(candidate)
    return {
        "manifest_version": 1,
        "candidate_manifest": str(candidate_manifest),
        "screening_root": str(screening_root),
        "allowed_decisions": ALLOWED_DECISIONS,
        "entries": entries,
    }


def export_review_pack(manifest: dict[str, Any], csv_path: Path, json_path: Path) -> None:
    rows = list(manifest.get("entries", []))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("")
    json_path.write_text(json.dumps(rows, indent=2))


def _parse_tags(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    return [part.strip() for part in str(raw_value).split(",") if part.strip()]


def _load_review_rows(review_csv: Path | None, review_json: Path | None) -> list[dict[str, Any]]:
    if review_json is not None:
        rows = json.loads(review_json.read_text())
        if not isinstance(rows, list):
            raise ValueError(f"Expected a list in review JSON: {review_json}")
        return [dict(row) for row in rows]
    if review_csv is not None:
        with review_csv.open(newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError("Provide --review-csv or --review-json for import-review.")


def import_review_decisions(
    *,
    base_manifest: dict[str, Any],
    review_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    review_index = {str(row.get("pair_slug", "")).strip(): row for row in review_rows}
    entries: list[dict[str, Any]] = []
    touched = 0
    for entry in base_manifest.get("entries", []):
        pair_slug = str(entry.get("pair_slug", "")).strip()
        merged = dict(entry)
        review_row = review_index.get(pair_slug)
        if review_row is not None:
            decision = str(review_row.get("decision", merged.get("decision", "unreviewed"))).strip() or "unreviewed"
            if decision not in ALLOWED_DECISIONS:
                raise ValueError(f"Invalid decision for {pair_slug}: {decision}")
            merged["decision"] = decision
            merged["rationale_tags"] = _parse_tags(review_row.get("rationale_tags", merged.get("rationale_tags", [])))
            merged["notes"] = str(review_row.get("notes", merged.get("notes", "")) or "")
            merged["reviewer"] = str(review_row.get("reviewer", merged.get("reviewer", "")) or "")
            reviewed_at = review_row.get("reviewed_at", merged.get("reviewed_at"))
            if decision != "unreviewed" and not reviewed_at:
                reviewed_at = datetime.now().isoformat(timespec="seconds")
            merged["reviewed_at"] = reviewed_at
            touched += 1
        entries.append(merged)
    payload = dict(base_manifest)
    payload["entries"] = entries
    payload["review_imported_at"] = datetime.now().isoformat(timespec="seconds")
    payload["review_import_count"] = touched
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit replacement Group 6 candidate renders.")
    parser.add_argument("--screening-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_GROUP6_CANDIDATE_MANIFEST)
    parser.add_argument("--audit-manifest", type=Path, default=DEFAULT_AUDIT_MANIFEST)
    parser.add_argument("--action", choices=["init-manifest", "refresh-manifest", "export-review", "import-review", "report"], required=True)
    parser.add_argument("--review-csv", type=Path, default=None)
    parser.add_argument("--review-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.screening_root, args.candidate_manifest, args.audit_manifest)
    if args.action in {"init-manifest", "refresh-manifest"}:
        save_manifest(args.audit_manifest, manifest)
        print(f"Saved candidate audit manifest -> {args.audit_manifest}")
        print(f"Entries: {len(manifest.get('entries', []))}")
        return
    if args.action == "export-review":
        review_csv = args.review_csv or args.audit_manifest.with_suffix(".review.csv")
        review_json = args.review_json or args.audit_manifest.with_suffix(".review.json")
        save_manifest(args.audit_manifest, manifest)
        export_review_pack(manifest, review_csv, review_json)
        print(f"Saved review CSV -> {review_csv}")
        print(f"Saved review JSON -> {review_json}")
        return
    if args.action == "import-review":
        review_rows = _load_review_rows(args.review_csv, args.review_json)
        merged_manifest = import_review_decisions(base_manifest=manifest, review_rows=review_rows)
        save_manifest(args.audit_manifest, merged_manifest)
        print(f"Imported review decisions -> {args.audit_manifest}")
        print(f"Updated entries: {merged_manifest.get('review_import_count', 0)}")
        return

    entries = manifest.get("entries", [])
    keep = sum(1 for row in entries if row.get("decision") == "keep")
    drop = sum(1 for row in entries if row.get("decision") == "drop")
    unreviewed = sum(1 for row in entries if row.get("decision") == "unreviewed")
    missing_assets = sum(1 for row in entries if not row.get("source_pair_dir"))
    print("GROUP 6 REPLACEMENT AUDIT STATUS")
    print("=" * 72)
    print(f"keep={keep} drop={drop} unreviewed={unreviewed} missing_assets={missing_assets}")
    print("=" * 72)
    for row in entries:
        print(f"{row['pair_slug']:<40} decision={row.get('decision','unreviewed')}")


if __name__ == "__main__":
    main()
