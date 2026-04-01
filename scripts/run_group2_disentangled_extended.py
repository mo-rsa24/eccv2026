#!/usr/bin/env python3
"""
Run an expanded Group 2 (feature-space disentangled) comparison sweep.

This is a thin wrapper around ``scripts/trajectory_dynamics_experiment.py``
that keeps the current Phase 1 Group 2 examples and adds several extra
object-style pairs so we can compare whether the dog/oil-painting pattern
persists across similar style-content compositions.

By default, results are written into the existing
``experiments/eccv2026/taxonomy_qualitative/group2_disentangled`` folder so
the new pair folders sit alongside the current ones and the group-level grids
are regenerated in place.

Example
-------
    python scripts/run_group2_disentangled_extended.py

    python scripts/run_group2_disentangled_extended.py --include-optional

    python scripts/run_group2_disentangled_extended.py \
        --seed 43 \
        --steps 50 \
        --guidance 7.5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_SCRIPT = PROJECT_ROOT / "scripts" / "trajectory_dynamics_experiment.py"


# Canonical Phase 1 Group 2 pairs plus three additional style-content controls
# already used elsewhere in the repo's control-landscape scripts.
DEFAULT_GROUP2_PAIRS = [
    ("a dog", "oil painting style"),
    ("a lighthouse", "watercolour style"),
    ("a bicycle", "sketch style"),
    ("a teapot", "claymation style"),
    ("a barn", "pencil drawing style"),
    ("a cactus", "mosaic style"),
]


# Optional extra comparisons that are still style-content, but a bit more
# stylistically aggressive than the default six.
OPTIONAL_GROUP2_PAIRS = [
    ("a chair", "black-and-white photo style"),
    ("a sailboat", "ukiyo-e style"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an expanded Group 2 style-content comparison sweep."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument(
        "--model-id",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Model to use. Defaults to SD 1.4 for comparability with Phase 1 outputs.",
    )
    parser.add_argument(
        "--projection",
        type=str,
        choices=["pca", "mds"],
        default="mds",
        help="Trajectory projection used in the per-pair manifold plots.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help=(
            "Output directory. Defaults to "
            "experiments/eccv2026/taxonomy_qualitative/group2_disentangled"
        ),
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also run the two optional style-content comparisons.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without running it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pairs = list(DEFAULT_GROUP2_PAIRS)
    if args.include_optional:
        pairs.extend(OPTIONAL_GROUP2_PAIRS)

    output_dir = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT
        / "experiments"
        / "eccv2026"
        / "taxonomy_qualitative"
        / "group2_disentangled"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(EXPERIMENT_SCRIPT),
        "--grid-pairs",
        json.dumps(pairs),
        "--grid-output-dir",
        str(output_dir),
        "--model-id",
        args.model_id,
        "--seed",
        str(args.seed),
        "--steps",
        str(args.steps),
        "--guidance",
        str(args.guidance),
        "--projection",
        args.projection,
        "--taxonomy-group",
        "group2_disentangled",
        "--no-superdiff",
        "--uniform-color",
    ]

    print("Expanded Group 2 comparison sweep")
    print(f"  Model   : {args.model_id}")
    print(f"  Seed    : {args.seed}")
    print(f"  Steps   : {args.steps}")
    print(f"  CFG     : {args.guidance}")
    print(f"  Proj    : {args.projection}")
    print(f"  Output  : {output_dir}")
    print(f"  Pairs   : {len(pairs)}")
    for prompt_a, prompt_b in pairs:
        print(f"    - {prompt_a} x {prompt_b}")

    if args.dry_run:
        print("\nDry run only. Command:")
        print(" ".join(cmd))
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
