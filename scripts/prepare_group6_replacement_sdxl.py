#!/usr/bin/env python3
"""Prepare edited SDXL final roots and stage Group 6 replacement screening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from group6_replacement_common import (
        DEFAULT_EDITED_FINAL_DIR,
        DEFAULT_EDITED_SDIPC_DIR,
        DEFAULT_GROUP6_CANDIDATE_MANIFEST,
        archive_group_dirs,
    build_edited_manifest,
    candidate_pairs_from_manifest,
    discover_pairs_in_root,
    format_pair_slug,
    load_candidate_manifest,
    load_json,
    mirror_root,
    render_pairs_into_root,
    save_json,
    screening_assets_complete,
)
except ImportError:
    from scripts.group6_replacement_common import (
        DEFAULT_EDITED_FINAL_DIR,
        DEFAULT_EDITED_SDIPC_DIR,
        DEFAULT_GROUP6_CANDIDATE_MANIFEST,
        archive_group_dirs,
        build_edited_manifest,
        candidate_pairs_from_manifest,
        discover_pairs_in_root,
        format_pair_slug,
        load_candidate_manifest,
        load_json,
        mirror_root,
        render_pairs_into_root,
        save_json,
        screening_assets_complete,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_FINAL_DIR = (
    PROJECT_ROOT / "experiments" / "eccv2026" / "sdxl_final" / "sdxl_six_group_seed42_steps50_cfg7p5_frozen"
)
DEFAULT_SOURCE_SDIPC_DIR = (
    PROJECT_ROOT / "experiments" / "eccv2026" / "sdxl_final" / "sdxl_six_group_seed42_steps50_cfg7p5_frozen_sdipc"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Group 6 replacement SDXL roots and screening assets.")
    parser.add_argument("--source-final-dir", type=Path, default=DEFAULT_SOURCE_FINAL_DIR)
    parser.add_argument("--source-sdipc-dir", type=Path, default=DEFAULT_SOURCE_SDIPC_DIR)
    parser.add_argument("--output-final-dir", type=Path, default=DEFAULT_EDITED_FINAL_DIR)
    parser.add_argument("--output-sdipc-dir", type=Path, default=DEFAULT_EDITED_SDIPC_DIR)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_GROUP6_CANDIDATE_MANIFEST)
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="symlink")
    parser.add_argument("--stage-screening", action="store_true", help="Render candidate pairs at the screening seed into the edited final root.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--projection", choices=["mds", "pca"], default="mds")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    return parser.parse_args()


def _filtered_pairs_for_pending_manifest(source_root: Path, archived_group_key: str, replacement_group_key: str) -> list[dict]:
    selected_pairs = discover_pairs_in_root(source_root, seed=42)
    return [
        row
        for row in selected_pairs
        if str(row["taxonomy_group_key"]) not in {replacement_group_key, archived_group_key}
    ]


def main() -> None:
    args = parse_args()
    candidate_manifest = load_candidate_manifest(args.candidate_manifest)
    archived_group_key = str(candidate_manifest["archived_group_key"])
    archived_group_label = str(candidate_manifest["archived_group_label"])
    replacement_group_key = str(candidate_manifest["replacement_group_key"])
    replacement_group_label = str(candidate_manifest["replacement_group_label"])

    if not args.source_final_dir.exists():
        raise SystemExit(f"Missing source final dir: {args.source_final_dir}")
    if not args.source_sdipc_dir.exists():
        raise SystemExit(f"Missing source SD-IPC dir: {args.source_sdipc_dir}")
    if args.output_final_dir.resolve() == args.source_final_dir.resolve():
        raise SystemExit("--output-final-dir must differ from --source-final-dir")
    if args.output_sdipc_dir.resolve() == args.source_sdipc_dir.resolve():
        raise SystemExit("--output-sdipc-dir must differ from --source-sdipc-dir")

    mirror_root(args.source_final_dir, args.output_final_dir, copy_mode=args.copy_mode)
    mirror_root(args.source_sdipc_dir, args.output_sdipc_dir, copy_mode=args.copy_mode)

    archive_group_dirs(
        args.output_final_dir,
        source_group_key=replacement_group_key,
        archive_group_key=archived_group_key,
    )
    archive_group_dirs(
        args.output_sdipc_dir,
        source_group_key=replacement_group_key,
        archive_group_key=archived_group_key,
    )

    source_final_manifest = load_json(args.source_final_dir / "sdxl_qualitative_run_manifest.json")
    pending_pairs = _filtered_pairs_for_pending_manifest(
        args.source_final_dir,
        archived_group_key=archived_group_key,
        replacement_group_key=replacement_group_key,
    )
    edited_final_manifest = build_edited_manifest(
        source_manifest=source_final_manifest,
        output_dir=args.output_final_dir,
        selected_pairs=pending_pairs,
        archived_group_key=archived_group_key,
        archived_group_label=archived_group_label,
        replacement_group_key=replacement_group_key,
        replacement_group_label=replacement_group_label,
        candidate_manifest_path=args.candidate_manifest,
        source_dir=args.source_final_dir,
    )
    save_json(args.output_final_dir / "sdxl_qualitative_run_manifest.json", edited_final_manifest)

    source_sdipc_manifest = load_json(args.source_sdipc_dir / "sdxl_qualitative_run_manifest.json")
    edited_sdipc_manifest = build_edited_manifest(
        source_manifest=source_sdipc_manifest,
        output_dir=args.output_sdipc_dir,
        selected_pairs=pending_pairs,
        archived_group_key=archived_group_key,
        archived_group_label=archived_group_label,
        replacement_group_key=replacement_group_key,
        replacement_group_label=replacement_group_label,
        candidate_manifest_path=args.candidate_manifest,
        source_dir=args.source_sdipc_dir,
    )
    save_json(args.output_sdipc_dir / "sdxl_qualitative_run_manifest.json", edited_sdipc_manifest)

    if args.stage_screening:
        screening_seed = int(candidate_manifest.get("screening_seed", 42))
        pairs_to_run = []
        for group_key, prompt_a, prompt_b in candidate_pairs_from_manifest(args.candidate_manifest):
            pair_dir = args.output_final_dir / f"seed_{screening_seed}" / group_key / format_pair_slug(prompt_a, prompt_b)
            if screening_assets_complete(pair_dir):
                continue
            pairs_to_run.append((group_key, prompt_a, prompt_b))
        if pairs_to_run:
            render_pairs_into_root(
                pairs_to_run=pairs_to_run,
                output_dir=args.output_final_dir,
                seeds=[screening_seed],
                grid_seed=screening_seed,
                model_id=args.model_id,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                projection=args.projection,
                height=args.height,
                width=args.width,
                device=args.device,
                gpu_id=args.gpu_id,
            )

    print(f"Prepared edited final root -> {args.output_final_dir}")
    print(f"Prepared edited SD-IPC root -> {args.output_sdipc_dir}")
    print(f"Archived {replacement_group_key} as {archived_group_key}")
    if args.stage_screening:
        print(f"Staged candidate screening from {args.candidate_manifest}")


if __name__ == "__main__":
    main()
