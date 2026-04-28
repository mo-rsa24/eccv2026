#!/usr/bin/env python3
"""Run the final 24-seed replacement Group 6 and optionally enrich SD-IPC."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from group6_replacement_common import (
        DEFAULT_EDITED_FINAL_DIR,
        DEFAULT_EDITED_SDIPC_DIR,
        merge_group6_into_manifest,
        render_pairs_into_root,
        save_json,
    )
except ImportError:
    from scripts.group6_replacement_common import (
        DEFAULT_EDITED_FINAL_DIR,
        DEFAULT_EDITED_SDIPC_DIR,
        merge_group6_into_manifest,
        render_pairs_into_root,
        save_json,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPLACEMENT_ROSTER = (
    PROJECT_ROOT / "experiments" / "eccv2026" / "sdxl_final" / "group6_replacement_roster.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final 24-seed replacement Group 6 renders.")
    parser.add_argument("--edited-final-dir", type=Path, default=DEFAULT_EDITED_FINAL_DIR)
    parser.add_argument("--edited-sdipc-dir", type=Path, default=DEFAULT_EDITED_SDIPC_DIR)
    parser.add_argument("--replacement-roster", type=Path, default=DEFAULT_REPLACEMENT_ROSTER)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--grid-seed", type=int, default=42)
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--projection", choices=["mds", "pca"], default="mds")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--enrich-sdipc", action="store_true")
    parser.add_argument("--skip-render", action="store_true", help="Do not render SDXL assets in this invocation.")
    parser.add_argument("--skip-manifest-update", action="store_true", help="Do not rewrite manifests in this invocation.")
    parser.add_argument(
        "--sync-manifests-only",
        action="store_true",
        help="Only merge the frozen replacement roster into the edited final and SD-IPC manifests.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    if args.sync_manifests_only:
        args.skip_render = True
        args.enrich_sdipc = False
    if args.sync_manifests_only and args.skip_manifest_update:
        raise SystemExit("--sync-manifests-only cannot be combined with --skip-manifest-update")

    edited_final_manifest_path = args.edited_final_dir / "sdxl_qualitative_run_manifest.json"
    if not edited_final_manifest_path.exists():
        raise SystemExit(f"Missing edited final manifest: {edited_final_manifest_path}")
    edited_manifest = _load_json(edited_final_manifest_path)
    roster = _load_json(args.replacement_roster)
    replacement_group_key = str(roster["replacement_group_key"])

    pairs_to_run = [
        (
            replacement_group_key,
            str(row["pair"][0]),
            str(row["pair"][1]),
        )
        for row in roster.get("selected_pairs", [])
    ]
    if not pairs_to_run:
        raise SystemExit(f"No selected_pairs found in {args.replacement_roster}")

    if not args.skip_render:
        render_pairs_into_root(
            pairs_to_run=pairs_to_run,
            output_dir=args.edited_final_dir,
            seeds=list(args.seeds),
            grid_seed=int(args.grid_seed),
            model_id=args.model_id,
            num_inference_steps=int(args.num_inference_steps),
            guidance_scale=float(args.guidance_scale),
            projection=args.projection,
            height=int(args.height),
            width=int(args.width),
            device=args.device,
            gpu_id=args.gpu_id,
        )

    merged_final_manifest = merge_group6_into_manifest(
        edited_manifest=edited_manifest,
        frozen_group_pairs=list(roster["selected_pairs"]),
        replacement_group_key=replacement_group_key,
        status="final_render_complete",
    )
    merged_final_manifest["grid_seed"] = int(args.grid_seed)
    merged_final_manifest["seeds"] = [int(seed) for seed in args.seeds]
    merged_final_manifest["num_inference_steps"] = int(args.num_inference_steps)
    merged_final_manifest["guidance_scale"] = float(args.guidance_scale)
    if not args.skip_manifest_update:
        save_json(edited_final_manifest_path, merged_final_manifest)

    if args.enrich_sdipc:
        enrich_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "enrich_sdxl_final_with_sdipc.py"),
            "--source-dir",
            str(args.edited_final_dir),
            "--output-dir",
            str(args.edited_sdipc_dir),
            "--groups",
            replacement_group_key,
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--grid-seed",
            str(args.grid_seed),
        ]
        subprocess.run(enrich_cmd, check=True)

        edited_sdipc_manifest_path = args.edited_sdipc_dir / "sdxl_qualitative_run_manifest.json"
        if edited_sdipc_manifest_path.exists():
            edited_sdipc_manifest = _load_json(edited_sdipc_manifest_path)
        else:
            edited_sdipc_manifest = dict(merged_final_manifest)
        merged_sdipc_manifest = merge_group6_into_manifest(
            edited_manifest=edited_sdipc_manifest,
            frozen_group_pairs=list(roster["selected_pairs"]),
            replacement_group_key=replacement_group_key,
            status="sdipc_enriched",
        )
        merged_sdipc_manifest["grid_seed"] = int(args.grid_seed)
        merged_sdipc_manifest["seeds"] = [int(seed) for seed in args.seeds]
        merged_sdipc_manifest["num_inference_steps"] = int(args.num_inference_steps)
        merged_sdipc_manifest["guidance_scale"] = float(args.guidance_scale)
        if not args.skip_manifest_update:
            save_json(edited_sdipc_manifest_path, merged_sdipc_manifest)
    elif args.sync_manifests_only:
        edited_sdipc_manifest_path = args.edited_sdipc_dir / "sdxl_qualitative_run_manifest.json"
        if edited_sdipc_manifest_path.exists():
            edited_sdipc_manifest = _load_json(edited_sdipc_manifest_path)
            merged_sdipc_manifest = merge_group6_into_manifest(
                edited_manifest=edited_sdipc_manifest,
                frozen_group_pairs=list(roster["selected_pairs"]),
                replacement_group_key=replacement_group_key,
                status="sdipc_enriched",
            )
            merged_sdipc_manifest["grid_seed"] = int(args.grid_seed)
            merged_sdipc_manifest["seeds"] = [int(seed) for seed in args.seeds]
            merged_sdipc_manifest["num_inference_steps"] = int(args.num_inference_steps)
            merged_sdipc_manifest["guidance_scale"] = float(args.guidance_scale)
            save_json(edited_sdipc_manifest_path, merged_sdipc_manifest)

    if args.sync_manifests_only:
        print(f"Synchronized replacement Group 6 manifests -> {args.edited_final_dir}")
        print(f"Synchronized replacement Group 6 manifests -> {args.edited_sdipc_dir}")
        return

    if not args.skip_render:
        print(f"Rendered replacement Group 6 final run -> {args.edited_final_dir}")
    if args.enrich_sdipc:
        print(f"Enriched replacement Group 6 SD-IPC -> {args.edited_sdipc_dir}")


if __name__ == "__main__":
    main()
