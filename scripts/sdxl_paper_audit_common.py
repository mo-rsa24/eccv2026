#!/usr/bin/env python3
"""Shared helpers for the manual SDXL paper audit manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "eccv2026"
    / "sdxl_screening"
    / "sdxl_paper_audit_manifest.json"
)

MANIFEST_VERSION = 2
DECISIONS = ("keep", "drop", "unreviewed")
RATIONALE_TAGS = (
    "paper_clean",
    "missing_concept",
    "dominance",
    "hybrid_artifact",
    "bad_scene_staging",
    "prior_hijack",
    "same_slot_collapse",
    "unclear_relation",
    "needs_replacement",
)


def empty_manifest() -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "allowed_decisions": list(DECISIONS),
        "allowed_rationale_tags": list(RATIONALE_TAGS),
        "entries": [],
    }


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_manifest()
    data = json.loads(path.read_text())
    if "entries" not in data or not isinstance(data["entries"], list):
        raise ValueError(f"Audit manifest at {path} is missing an 'entries' list")
    return data


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def manifest_key(pair_slug: str) -> str:
    return str(pair_slug)


def index_manifest_entries(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("entries", []):
        indexed[manifest_key(entry.get("pair_slug", ""))] = entry
    return indexed


def infer_source_run_from_pair_dir(pair_dir: Path) -> str:
    pair_dir = pair_dir.resolve()
    for ancestor in [pair_dir, *pair_dir.parents]:
        if (ancestor / "sdxl_qualitative_run_manifest.json").exists():
            return ancestor.name
    return pair_dir.parents[1].name if len(pair_dir.parents) > 1 else pair_dir.name


def merge_manifest_entry(existing: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    preserved = existing or {}
    return {
        "pair_slug": candidate["pair_slug"],
        "qualitative_pair_slug": candidate.get("qualitative_pair_slug"),
        "prompt_a": candidate.get("prompt_a"),
        "prompt_b": candidate.get("prompt_b"),
        "taxonomy_group_key": candidate.get("taxonomy_group_key"),
        "taxonomy_group_label": candidate.get("taxonomy_group_label"),
        "source_run": candidate.get("source_run", ""),
        "source_pair_dir": candidate.get("source_pair_dir"),
        "summary_json": candidate.get("summary_json"),
        "grid_assets_json": candidate.get("grid_assets_json"),
        "trajectory_manifold_png": candidate.get("trajectory_manifold_png"),
        "decoded_images_png": candidate.get("decoded_images_png"),
        "solo_a_png": candidate.get("solo_a_png"),
        "solo_b_png": candidate.get("solo_b_png"),
        "monolithic_png": candidate.get("monolithic_png"),
        "poe_png": candidate.get("poe_png"),
        "decision": preserved.get("decision", "unreviewed"),
        "rationale_tags": list(preserved.get("rationale_tags", [])),
        "notes": preserved.get("notes", ""),
        "reviewer": preserved.get("reviewer", ""),
        "reviewed_at": preserved.get("reviewed_at"),
    }
