#!/usr/bin/env python3
"""Build and export a manual review manifest for SDXL screening pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from sdxl_paper_audit_common import (
        DEFAULT_AUDIT_MANIFEST_PATH,
        index_manifest_entries,
        load_manifest,
        merge_manifest_entry,
        save_manifest,
    )
    from taxonomy_manifest import GROUP_ORDER, GROUP_LABEL_BY_KEY, get_pair_taxonomy_from_slug, taxonomy_manifest_rows
except ImportError:
    from scripts.sdxl_paper_audit_common import (
        DEFAULT_AUDIT_MANIFEST_PATH,
        index_manifest_entries,
        load_manifest,
        merge_manifest_entry,
        save_manifest,
    )
    from scripts.taxonomy_manifest import (
        GROUP_ORDER,
        GROUP_LABEL_BY_KEY,
        get_pair_taxonomy_from_slug,
        taxonomy_manifest_rows,
    )


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(path)
    return ordered


def resolve_screening_roots(args: argparse.Namespace) -> list[Path]:
    roots: list[Path] = []
    if args.data_dir is not None:
        roots.append(args.data_dir)
    roots.extend(args.screening_roots or [])
    if not roots:
        raise SystemExit("Provide at least one screening root via --screening-roots or --data-dir.")
    roots = _dedupe_paths(roots)
    existing = [root for root in roots if root.exists()]
    missing = [str(root) for root in roots if not root.exists()]
    if missing and args.strict_screening_roots:
        raise SystemExit(f"Missing screening roots: {missing}")
    if missing:
        print(f"Skipping missing screening roots: {missing}")
    if not existing:
        raise SystemExit("None of the provided screening roots exist.")
    return existing


def _candidate_from_pair_dir(screening_root: Path, pair_dir: Path) -> dict[str, Any] | None:
    meta = get_pair_taxonomy_from_slug(pair_dir.name)
    if meta is None or not bool(meta.get("is_canonical_pair")):
        return None
    return {
        "pair_slug": str(meta["pair_slug"]),
        "qualitative_pair_slug": str(meta["qualitative_pair_slug"]),
        "prompt_a": str(meta["prompt_a"]),
        "prompt_b": str(meta["prompt_b"]),
        "taxonomy_group_key": str(meta["taxonomy_group_key"]),
        "taxonomy_group_label": str(meta["taxonomy_group_label"]),
        "source_run": screening_root.name,
        "source_pair_dir": str(pair_dir),
        "summary_json": str(pair_dir / "summary.json") if (pair_dir / "summary.json").exists() else None,
        "grid_assets_json": str(pair_dir / "grid_assets.json") if (pair_dir / "grid_assets.json").exists() else None,
        "trajectory_manifold_png": str(pair_dir / "trajectory_manifold.png") if (pair_dir / "trajectory_manifold.png").exists() else None,
        "decoded_images_png": str(pair_dir / "decoded_images.png") if (pair_dir / "decoded_images.png").exists() else None,
        "solo_a_png": str(pair_dir / "solo_a.png") if (pair_dir / "solo_a.png").exists() else None,
        "solo_b_png": str(pair_dir / "solo_b.png") if (pair_dir / "solo_b.png").exists() else None,
        "monolithic_png": str(pair_dir / "monolithic.png") if (pair_dir / "monolithic.png").exists() else None,
        "poe_png": str(pair_dir / "poe.png") if (pair_dir / "poe.png").exists() else None,
    }


def collect_candidate_assets(screening_roots: list[Path]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for screening_root in screening_roots:
        for group_dir in screening_root.glob("group*"):
            if not group_dir.is_dir():
                continue
            for pair_dir in sorted(child for child in group_dir.iterdir() if child.is_dir()):
                candidate = _candidate_from_pair_dir(screening_root, pair_dir)
                if candidate is None:
                    continue
                pair_slug = str(candidate["pair_slug"])
                candidates.setdefault(pair_slug, candidate)
    return candidates


def build_manifest_entries(
    screening_roots: list[Path],
    audit_manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_manifest(audit_manifest_path)
    existing_index = index_manifest_entries(manifest)
    candidate_assets = collect_candidate_assets(screening_roots)

    entries: list[dict[str, Any]] = []
    for row in taxonomy_manifest_rows():
        pair_slug = str(row["pair_slug"])
        candidate = candidate_assets.get(
            pair_slug,
            {
                "pair_slug": pair_slug,
                "qualitative_pair_slug": row["qualitative_pair_slug"],
                "prompt_a": row["prompt_a"],
                "prompt_b": row["prompt_b"],
                "taxonomy_group_key": row["taxonomy_group_key"],
                "taxonomy_group_label": row["taxonomy_group_label"],
                "source_run": "",
                "source_pair_dir": None,
                "summary_json": None,
                "grid_assets_json": None,
                "trajectory_manifold_png": None,
                "decoded_images_png": None,
                "solo_a_png": None,
                "solo_b_png": None,
                "monolithic_png": None,
                "poe_png": None,
            },
        )
        entries.append(merge_manifest_entry(existing_index.get(pair_slug), candidate))

    entries.sort(
        key=lambda entry: (
            GROUP_ORDER.index(entry["taxonomy_group_key"]) if entry["taxonomy_group_key"] in GROUP_ORDER else 99,
            str(entry["pair_slug"]),
        )
    )
    manifest["entries"] = entries
    return manifest, entries


def _review_row(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "taxonomy_group_key": entry.get("taxonomy_group_key"),
        "taxonomy_group_label": entry.get("taxonomy_group_label"),
        "pair_slug": entry.get("pair_slug"),
        "qualitative_pair_slug": entry.get("qualitative_pair_slug"),
        "prompt_a": entry.get("prompt_a"),
        "prompt_b": entry.get("prompt_b"),
        "decision": entry.get("decision", "unreviewed"),
        "rationale_tags": ",".join(entry.get("rationale_tags", [])),
        "notes": entry.get("notes", ""),
        "reviewer": entry.get("reviewer", ""),
        "reviewed_at": entry.get("reviewed_at"),
        "source_run": entry.get("source_run", ""),
        "source_pair_dir": entry.get("source_pair_dir"),
        "has_assets": bool(entry.get("source_pair_dir")),
        "summary_json": entry.get("summary_json"),
        "grid_assets_json": entry.get("grid_assets_json"),
        "trajectory_manifold_png": entry.get("trajectory_manifold_png"),
        "decoded_images_png": entry.get("decoded_images_png"),
        "solo_a_png": entry.get("solo_a_png"),
        "solo_b_png": entry.get("solo_b_png"),
        "monolithic_png": entry.get("monolithic_png"),
        "poe_png": entry.get("poe_png"),
    }


def export_review_pack(manifest: dict[str, Any], csv_path: Path, json_path: Path) -> None:
    rows = [_review_row(entry) for entry in manifest.get("entries", [])]
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


def print_report(entries: list[dict[str, Any]]) -> None:
    counts_by_group: dict[str, dict[str, int]] = {
        group_key: {"keep": 0, "drop": 0, "unreviewed": 0, "missing_assets": 0}
        for group_key in GROUP_ORDER
    }
    for entry in entries:
        group_key = str(entry["taxonomy_group_key"])
        status = str(entry.get("decision", "unreviewed"))
        counts_by_group.setdefault(group_key, {"keep": 0, "drop": 0, "unreviewed": 0, "missing_assets": 0})
        counts_by_group[group_key][status] = counts_by_group[group_key].get(status, 0) + 1
        if not entry.get("source_pair_dir"):
            counts_by_group[group_key]["missing_assets"] += 1

    print("")
    print("SDXL MANUAL AUDIT STATUS")
    print("=" * 96)
    print(f"{'GROUP':<42} {'KEEP':>5} {'DROP':>5} {'UNREVIEWED':>10} {'MISSING_ASSETS':>15}")
    print("=" * 96)
    for group_key in GROUP_ORDER:
        counts = counts_by_group.get(group_key, {})
        print(
            f"{GROUP_LABEL_BY_KEY.get(group_key, group_key):<42} "
            f"{counts.get('keep', 0):>5} "
            f"{counts.get('drop', 0):>5} "
            f"{counts.get('unreviewed', 0):>10} "
            f"{counts.get('missing_assets', 0):>15}"
        )
    print("=" * 96)

    missing = [entry["pair_slug"] for entry in entries if not entry.get("source_pair_dir")]
    if missing:
        print("Missing source assets:")
        for pair_slug in missing:
            print(f"  - {pair_slug}")
        print("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize and export the manual SDXL audit manifest.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Single screening root (legacy alias).")
    parser.add_argument("--screening-roots", nargs="+", type=Path, default=None, help="One or more screening roots.")
    parser.add_argument(
        "--strict-screening-roots",
        action="store_true",
        help="Fail if any provided screening root does not exist. Default behavior skips missing roots.",
    )
    parser.add_argument(
        "--action",
        required=True,
        choices=["init-manifest", "refresh-manifest", "export-review", "report"],
        help="Action to perform.",
    )
    parser.add_argument("--audit-manifest", type=Path, default=DEFAULT_AUDIT_MANIFEST_PATH, help="Audit manifest JSON path.")
    parser.add_argument("--review-csv", type=Path, default=None, help="Review CSV export path.")
    parser.add_argument("--review-json", type=Path, default=None, help="Review JSON export path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    screening_roots = resolve_screening_roots(args)
    manifest, entries = build_manifest_entries(screening_roots, args.audit_manifest)

    if args.action in {"init-manifest", "refresh-manifest"}:
        save_manifest(args.audit_manifest, manifest)
        print(f"Saved audit manifest -> {args.audit_manifest}")
        print(f"Entries: {len(entries)}")
        return

    if args.action == "export-review":
        save_manifest(args.audit_manifest, manifest)
        stem = args.audit_manifest.with_suffix("")
        review_csv = args.review_csv or stem.with_name(f"{stem.name}_review.csv")
        review_json = args.review_json or stem.with_name(f"{stem.name}_review.json")
        export_review_pack(manifest, review_csv, review_json)
        print(f"Saved audit manifest -> {args.audit_manifest}")
        print(f"Saved review CSV -> {review_csv}")
        print(f"Saved review JSON -> {review_json}")
        return

    if args.action == "report":
        print_report(entries)


if __name__ == "__main__":
    main()
