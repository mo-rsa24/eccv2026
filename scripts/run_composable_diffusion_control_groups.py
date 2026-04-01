#!/usr/bin/env python3
"""
Run the PoE/composable-diffusion SD1.4 grid over the curated control groups.

This wraps scripts/run_composable_diffusion_grid.sh and injects the pair list for
each group so we can compare how PoE behaves on the same orthogonal pairs used in
the control-landscape survey, plus any PoE-specific supplementary groups defined
locally in this script.

Examples
--------
python scripts/run_composable_diffusion_control_groups.py

python scripts/run_composable_diffusion_control_groups.py \
    --groups spatially_disjoint sky_ground \
    --steps 30 \
    --seed 43

python scripts/run_composable_diffusion_control_groups.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from run_control_landscape import GROUPS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID_SCRIPT = PROJECT_ROOT / "scripts" / "run_composable_diffusion_grid.sh"

# These are not orthogonal controls in the Phase-1 taxonomy. They are best read
# as a positive-control/easy-context group: object-scene pairs that are visually
# compatible and should be comparatively easy for semantic composition.
EXTRA_GROUPS = {
    "positive_object_scene": [
        ("a butterfly",         "a flower meadow"),
        ("a camel",             "a desert landscape"),
        ("a camping tent",      "a forest clearing"),
        ("a deer",              "a pine forest"),
        ("a dolphin",           "an ocean wave"),
        ("a hot air balloon",   "a mountain landscape"),
        ("a lighthouse",        "a stormy sea"),
        ("a lion",              "a savanna at sunset"),
        ("a palm tree",         "a tropical beach"),
        ("a red apple",         "a blue sky"),
        ("a sailboat",          "cloudy blue sky"),
        ("a snowman",           "a snowy field"),
        ("a toucan",            "a tropical rainforest"),
    ],
    "orthogonal_layered_layout": [
        ("an airplane in the sky",       "a green tractor on the ground"),
        ("a hot air balloon in the sky", "a wooden cabin on the ground"),
        ("white clouds in the sky",      "a sunflower field on the ground"),
        ("a flock of birds in the sky",  "a stone bridge on the ground"),
        ("a crescent moon in the sky",   "a wooden fence on the ground"),
        ("a rainbow in the sky",         "a red barn on the ground"),
        ("a kite in the sky",            "a park bench on the ground"),
        ("a blimp in the sky",           "a traffic cone on the ground"),
        ("a helicopter in the sky",      "a flower pot on the ground"),
        ("a sailboat on water",          "a lighthouse on shore"),
        ("a sailboat on water",          "a beach umbrella on sand"),
        ("a windmill in distance",       "a mailbox in foreground"),
        ("a ferris wheel in distance",   "a picnic basket in foreground"),
        ("a ceiling fan above",          "a wooden chair below"),
        ("a streetlamp above",           "a bathtub below"),
    ],
    "orthogonal_object_style": [
        ("a dog",        "oil painting style"),
        ("a lighthouse", "watercolor style"),
        ("a bicycle",    "sketch style"),
        ("a teapot",     "claymation style"),
        ("a barn",       "pencil drawing style"),
        ("a chair",      "black-and-white photo style"),
        ("a cactus",     "mosaic style"),
        ("a sailboat",   "ukiyo-e style"),
    ],
}

POE_GROUPS = {**GROUPS, **EXTRA_GROUPS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PoE/composable diffusion on the curated control groups."
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(POE_GROUPS.keys()),
        default=list(POE_GROUPS.keys()),
        help="Subset of orthogonal controls / supplementary PoE groups to run. Defaults to all groups.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help=(
            "Base output directory. Defaults to "
            "experiments/eccv2026/grid_figure/composable_diffusion_control_groups/"
        ),
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument(
        "--scheduler",
        choices=["lms", "ddim", "ddpm", "pndm"],
        default="ddim",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Optional pipe-delimited weights, e.g. '7.5 | 7.5'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-group commands without generating images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_out = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT
        / "experiments"
        / "eccv2026"
        / "grid_figure"
        / "composable_diffusion_control_groups"
    )
    base_out.mkdir(parents=True, exist_ok=True)

    total_pairs = sum(len(POE_GROUPS[group]) for group in args.groups)
    print("Composable diffusion control-group sweep", flush=True)
    print(f"  Base output : {base_out}", flush=True)
    print(f"  Groups      : {', '.join(args.groups)}", flush=True)
    print(f"  Total pairs : {total_pairs}", flush=True)
    print(f"  Seed        : {args.seed}", flush=True)
    print(f"  Steps       : {args.steps}", flush=True)
    print(f"  Scale       : {args.scale}", flush=True)
    if args.weights:
        print(f"  Weights     : {args.weights}", flush=True)
    if args.dry_run:
        print("  Dry run     : enabled", flush=True)

    for group_name in args.groups:
        pairs = POE_GROUPS[group_name]
        group_out = base_out / group_name
        env = os.environ.copy()
        env.update(
            {
                "PAIRS_JSON": json.dumps(pairs),
                "STEPS": str(args.steps),
                "SCALE": str(args.scale),
                "SEED": str(args.seed),
                "NUM_IMAGES": str(args.num_images),
                "SCHEDULER": args.scheduler,
                "MODEL_PATH": args.model_path,
            }
        )
        if args.weights:
            env["WEIGHTS"] = args.weights
        if args.dry_run:
            env["DRY_RUN"] = "1"

        print(f"\n[{group_name}] {len(pairs)} pairs -> {group_out}", flush=True)
        for prompt_a, prompt_b in pairs:
            print(f"  - {prompt_a} | {prompt_b}", flush=True)

        subprocess.run(
            ["bash", str(GRID_SCRIPT), str(group_out)],
            check=True,
            env=env,
        )

    print(f"\nAll requested groups completed. Outputs in: {base_out}", flush=True)


if __name__ == "__main__":
    main()
