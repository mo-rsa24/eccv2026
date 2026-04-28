#!/usr/bin/env python3
"""Shared helpers for Group 6 replacement workflows."""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from run_taxonomy_qualitative_sdxl import (
        DEFAULT_MODEL_ID,
        TaxonomyExperimentConfig,
        _load_models,
        _pair_output_dir,
        format_pair_slug,
        run_single_experiment,
    )
except ImportError:
    from scripts.run_taxonomy_qualitative_sdxl import (
        DEFAULT_MODEL_ID,
        TaxonomyExperimentConfig,
        _load_models,
        _pair_output_dir,
        format_pair_slug,
        run_single_experiment,
    )
try:
    from taxonomy_manifest import get_pair_taxonomy_from_slug
except ImportError:
    from scripts.taxonomy_manifest import get_pair_taxonomy_from_slug


DEFAULT_GROUP6_CANDIDATE_MANIFEST = PROJECT_ROOT / "scripts" / "group6_collision_candidates.json"
DEFAULT_EDITED_FINAL_DIR = (
    PROJECT_ROOT / "experiments" / "eccv2026" / "sdxl_final" / "sdxl_six_group_seed42_steps50_cfg7p5_group6_edit"
)
DEFAULT_EDITED_SDIPC_DIR = (
    PROJECT_ROOT / "experiments" / "eccv2026" / "sdxl_final" / "sdxl_six_group_seed42_steps50_cfg7p5_group6_edit_sdipc"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_candidate_manifest(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    candidates = payload.get("candidate_pairs")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Candidate manifest at {path} is missing candidate_pairs.")
    return payload


def candidate_pairs_from_manifest(path: Path) -> list[tuple[str, str, str]]:
    payload = load_candidate_manifest(path)
    group_key = str(payload.get("replacement_group_key", "group6_coherent_collision"))
    pairs: list[tuple[str, str, str]] = []
    for row in payload["candidate_pairs"]:
        pair_value = row.get("pair")
        if not isinstance(pair_value, list) or len(pair_value) != 2:
            raise ValueError(f"Invalid candidate pair entry in {path}: {row}")
        pairs.append((group_key, str(pair_value[0]), str(pair_value[1])))
    return pairs


def screening_assets_complete(pair_dir: Path) -> bool:
    required = [
        pair_dir / "decoded_images.png",
        pair_dir / "grid_assets.json",
        pair_dir / "monolithic.png",
        pair_dir / "poe.png",
        pair_dir / "solo_a.png",
        pair_dir / "solo_b.png",
        pair_dir / "summary.json",
        pair_dir / "trajectory_manifold.png",
    ]
    return all(path.exists() for path in required)


def pair_row(group_key: str, prompt_a: str, prompt_b: str) -> dict[str, Any]:
    slug = format_pair_slug(prompt_a, prompt_b)
    meta = get_pair_taxonomy_from_slug(slug)
    row = {
        "taxonomy_group_key": group_key,
        "pair": [prompt_a, prompt_b],
        "qualitative_pair_slug": slug,
    }
    if meta is not None:
        row["pair_slug"] = str(meta.get("pair_slug") or slug)
    else:
        row["pair_slug"] = slug
    return row


def discover_pairs_in_root(root: Path, *, seed: int = 42) -> list[dict[str, Any]]:
    probe_root = root / f"seed_{seed}"
    if not probe_root.exists():
        raise FileNotFoundError(f"Missing probe seed directory: {probe_root}")

    rows: list[dict[str, Any]] = []
    for group_dir in sorted(path for path in probe_root.iterdir() if path.is_dir()):
        group_key = group_dir.name
        for pair_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            slug = pair_dir.name
            meta = get_pair_taxonomy_from_slug(slug)
            if meta is not None:
                prompt_a = str(meta["prompt_a"])
                prompt_b = str(meta["prompt_b"])
            else:
                if "__x__" not in slug:
                    raise ValueError(f"Cannot infer pair prompts from {pair_dir}")
                prompt_a_slug, prompt_b_slug = slug.split("__x__", 1)
                prompt_a = prompt_a_slug.replace("_", " ")
                prompt_b = prompt_b_slug.replace("_", " ")
            rows.append(pair_row(group_key, prompt_a, prompt_b))
    return rows


def _safe_link_or_copy(src: Path, dst: Path, *, copy_mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            return
        if copy_mode == "copy":
            shutil.copy2(src, dst)
        else:
            os.symlink(src.resolve(), dst)
    except FileExistsError:
        return


def mirror_root(source_dir: Path, output_dir: Path, *, copy_mode: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in source_dir.rglob("*"):
        rel = src.relative_to(source_dir)
        dst = output_dir / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        _safe_link_or_copy(src, dst, copy_mode=copy_mode)


def archive_group_dirs(
    root: Path,
    *,
    source_group_key: str,
    archive_group_key: str,
) -> None:
    for seed_dir in sorted(path for path in root.glob("seed_*") if path.is_dir()):
        src_group_dir = seed_dir / source_group_key
        archived_group_dir = seed_dir / archive_group_key
        archived_exists = archived_group_dir.exists() or archived_group_dir.is_symlink()
        source_exists = src_group_dir.exists() or src_group_dir.is_symlink()
        if source_exists:
            archived_group_dir.parent.mkdir(parents=True, exist_ok=True)
            if archived_exists:
                # Idempotent re-entry: prior preparation already archived the old group.
                pass
            else:
                src_group_dir.rename(archived_group_dir)
        (seed_dir / source_group_key).mkdir(parents=True, exist_ok=True)


def build_edited_manifest(
    *,
    source_manifest: dict[str, Any],
    output_dir: Path,
    selected_pairs: list[dict[str, Any]],
    archived_group_key: str,
    archived_group_label: str,
    replacement_group_key: str,
    replacement_group_label: str,
    candidate_manifest_path: Path,
    source_dir: Path,
) -> dict[str, Any]:
    payload = dict(source_manifest)
    payload["output_dir"] = str(output_dir)
    payload["selected_pairs"] = list(selected_pairs)
    payload["all_requested_pairs"] = list(selected_pairs)
    payload["requested_pair_count"] = len(selected_pairs)
    payload["worker_pair_count"] = len(selected_pairs)
    payload["worker_record_count"] = len(selected_pairs) * len(payload.get("seeds", []))
    payload["group6_replacement"] = {
        "created_at": datetime.now().isoformat(),
        "source_dir": str(source_dir),
        "candidate_manifest": str(candidate_manifest_path),
        "replacement_group_key": replacement_group_key,
        "replacement_group_label": replacement_group_label,
        "archived_group_key": archived_group_key,
        "archived_group_label": archived_group_label,
        "status": "screening_pending",
    }
    return payload


def merge_group6_into_manifest(
    *,
    edited_manifest: dict[str, Any],
    frozen_group_pairs: list[dict[str, Any]],
    replacement_group_key: str,
    status: str,
) -> dict[str, Any]:
    selected_pairs = [
        row
        for row in edited_manifest.get("selected_pairs", [])
        if str(row.get("taxonomy_group_key")) != replacement_group_key
    ]
    selected_pairs.extend(frozen_group_pairs)
    selected_pairs.sort(key=lambda row: (str(row["taxonomy_group_key"]), str(row["qualitative_pair_slug"])))

    payload = dict(edited_manifest)
    payload["selected_pairs"] = selected_pairs
    payload["all_requested_pairs"] = selected_pairs
    payload["requested_pair_count"] = len(selected_pairs)
    payload["worker_pair_count"] = len(selected_pairs)
    payload["worker_record_count"] = len(selected_pairs) * len(payload.get("seeds", []))
    replacement_meta = dict(payload.get("group6_replacement", {}))
    replacement_meta["status"] = status
    replacement_meta["updated_at"] = datetime.now().isoformat()
    payload["group6_replacement"] = replacement_meta
    return payload


def render_pairs_into_root(
    *,
    pairs_to_run: list[tuple[str, str, str]],
    output_dir: Path,
    seeds: list[int],
    grid_seed: int,
    model_id: str = DEFAULT_MODEL_ID,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    projection: str = "mds",
    height: int = 1024,
    width: int = 1024,
    device: str = "auto",
    gpu_id: int | None = None,
) -> None:
    if device != "auto" and gpu_id is not None:
        raise ValueError("Use either device or gpu_id, not both.")
    if gpu_id is not None:
        device_str = f"cuda:{gpu_id}"
    elif device != "auto":
        device_str = device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device_str)
    dtype = torch.float16 if torch_device.type == "cuda" else torch.float32

    output_dir.mkdir(parents=True, exist_ok=True)
    models_tuple = _load_models(model_id, torch_device, dtype)
    for group_key, prompt_a, prompt_b in pairs_to_run:
        pair_slug = format_pair_slug(prompt_a, prompt_b)
        for seed in seeds:
            pair_dir = _pair_output_dir(
                output_dir,
                group_key,
                pair_slug,
                seed=seed,
                multi_seed_layout=len(seeds) > 1,
            )
            config = TaxonomyExperimentConfig(
                group=group_key,
                prompt_a=prompt_a,
                prompt_b=prompt_b,
                model_id=model_id,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=int(seed),
                projection=projection,
                height=height,
                width=width,
            )
            run_single_experiment(config, pair_dir, models_tuple, torch_device, dtype)
