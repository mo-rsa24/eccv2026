#!/usr/bin/env python3
"""
Run the author SuperDiff SD1.4 implementation over curated control groups.

This wrapper targets compositions/super-diffusion/scripts/compose_and.py via
scripts/run_compose_and_grid.sh, so it stays on the vendored author codepath
rather than our local math reimplementation.

Examples
--------
python scripts/run_compose_and_control_groups.py

python scripts/run_compose_and_control_groups.py \
    --groups positive_object_scene \
    --steps 30 \
    --seed 43

python scripts/run_compose_and_control_groups.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from run_control_landscape import GROUPS
from run_composable_diffusion_control_groups import EXTRA_GROUPS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID_SCRIPT = PROJECT_ROOT / "scripts" / "run_compose_and_grid.sh"

PRESETS = {
    # Matches the author image-eval path more closely:
    # clip_eval.py --method and --num_inference_steps 1000 --seed 1 --batch_size 1
    "author": {
        "mode": "stochastic",
        "steps": 1000,
        "scale": 7.5,
        "seed": 1,
        "batch_size": 1,
    },
    # Kept as a cheaper exploratory preset.
    "fast": {
        "mode": "deterministic",
        "steps": 50,
        "scale": 7.5,
        "seed": 21,
        "batch_size": 1,
    },
}

# SuperDiff-positive pairs should encourage a single compromise object or a
# strongly fused hybrid, rather than simple co-presence in a compatible scene.
SUPERDIFF_EXTRA_GROUPS = {
    "author_clip_eval_superdiff": [
        ("a mountain landscape", "silhouette of a dog"),
        ("a flamingo", "a candy cane"),
        ("a dragonfly", "a helicopter"),
        ("dandelion", "fireworks"),
        ("a sunflower", "a lemon"),
        ("a rocket", "a cactus"),
        ("moon", "cookie"),
        ("a snail", "a cinnamon roll"),
        ("an eagle", "an airplane"),
        ("zebra", "barcode"),
        ("chess pawn", "bottle cap"),
        ("a pineapple", "a beehive"),
        ("a spider web", "a bicycle wheel"),
        ("a waffle cone", "a volcano"),
        ("a cat", "a dog"),
        ("a chair", "an avocado"),
        ("a donut", "a map"),
        ("otter", "duck"),
        ("pebbles on a beach", "a turtle"),
        ("teddy bear", "panda"),
    ],
    "positive_blend_superdiff": [
        ("a donut",         "rainbow sprinkles"),
        ("a white mug",     "blue polka dots"),
        ("a black umbrella","neon pink trim"),
        ("a flamingo",      "a candy cane"),
        ("a dragonfly",     "a helicopter"),
        ("a sunflower",     "a lemon"),
        ("a snail",         "a cinnamon roll"),
        ("an eagle",        "an airplane"),
        ("zebra",           "barcode"),
        ("a pineapple",     "a beehive"),
        ("a spider web",    "a bicycle wheel"),
        ("teddy bear",      "panda"),
    ],
    "positive_superimposition_superdiff": [
        ("a mountain landscape", "silhouette of a dog"),
        ("a flamingo",           "a candy cane"),
        ("a dragonfly",          "a helicopter"),
        ("dandelion",            "fireworks"),
        ("a sunflower",          "a lemon"),
        ("moon",                 "cookie"),
        ("a snail",              "a cinnamon roll"),
        ("an eagle",             "an airplane"),
        ("zebra",                "barcode"),
        ("chess pawn",           "bottle cap"),
        ("a pineapple",          "a beehive"),
        ("a spider web",         "a bicycle wheel"),
        ("a waffle cone",        "a volcano"),
    ],
    "positive_real_blend_superdiff": [
        ("a white mug",          "blue polka dots"),
        ("a black umbrella",     "neon pink trim"),
        ("a donut",              "rainbow sprinkles"),
        ("a cupcake",            "pink frosting"),
        ("a cookie",             "chocolate chips"),
        ("a ceramic vase",       "blue floral pattern"),
        ("a backpack",           "camouflage pattern"),
        ("a sweater",            "knit pattern"),
        ("a guitar",             "sunburst finish"),
        ("a lamp",               "stained glass"),
        ("a bowl",               "marble texture"),
        ("a skateboard",         "graffiti art"),
    ],
}

SD14_GROUPS = {**GROUPS, **EXTRA_GROUPS, **SUPERDIFF_EXTRA_GROUPS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run author SuperDiff SD1.4 on curated control groups."
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(SD14_GROUPS.keys()),
        default=["positive_superimposition_superdiff"],
        help="Subset of groups to run. Defaults to the SD1.4 SuperDiff-positive superimposition group.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="author",
        help=(
            "Sampling preset. 'author' matches the vendored clip_eval.py "
            "regime much more closely; 'fast' keeps the earlier cheap defaults."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help=(
            "Base output directory. Defaults to "
            "experiments/eccv2026/grid_figure/compose_and_control_groups_sd14/"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["deterministic", "stochastic", "average", "single"],
        default=None,
        help="Author compose_and.py mode. Overrides the selected preset.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of denoising steps. Overrides the selected preset.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Guidance scale. Overrides the selected preset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed to run. Ignored when --seeds is provided.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional seed sweep. Useful because the paper shows the first 6 "
            "seeds and also cherry-picks favourites from 20 seeds."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of images to generate in parallel. Overrides the selected preset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-group plans without generating images.",
    )
    return parser.parse_args()


def resolve_run_config(args: argparse.Namespace) -> dict[str, object]:
    config = PRESETS[args.preset].copy()
    if args.mode is not None:
        config["mode"] = args.mode
    if args.steps is not None:
        config["steps"] = args.steps
    if args.scale is not None:
        config["scale"] = args.scale
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.seed is not None:
        config["seed"] = args.seed
    return config


def main() -> None:
    args = parse_args()
    config = resolve_run_config(args)
    seeds = args.seeds if args.seeds is not None else [int(config["seed"])]
    base_out = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT
        / "experiments"
        / "eccv2026"
        / "grid_figure"
        / "compose_and_control_groups_sd14"
    )
    base_out.mkdir(parents=True, exist_ok=True)

    total_pairs = sum(len(SD14_GROUPS[group]) for group in args.groups)
    total_runs = total_pairs * len(seeds)
    print("Author SuperDiff SD1.4 control-group sweep", flush=True)
    print(f"  Base output : {base_out}", flush=True)
    print(f"  Preset      : {args.preset}", flush=True)
    print(f"  Groups      : {', '.join(args.groups)}", flush=True)
    print(f"  Total pairs : {total_pairs}", flush=True)
    print(f"  Total runs  : {total_runs}", flush=True)
    print(f"  Mode        : {config['mode']}", flush=True)
    print(f"  Seeds       : {', '.join(str(seed) for seed in seeds)}", flush=True)
    print(f"  Steps       : {config['steps']}", flush=True)
    print(f"  Scale       : {config['scale']}", flush=True)
    print(f"  Batch size  : {config['batch_size']}", flush=True)
    if args.dry_run:
        print("  Dry run     : enabled", flush=True)

    for group_name in args.groups:
        pairs = SD14_GROUPS[group_name]
        group_out_root = base_out / group_name
        print(f"\n[{group_name}] {len(pairs)} pairs", flush=True)
        for prompt_a, prompt_b in pairs:
            print(f"  - {prompt_a} | {prompt_b}", flush=True)

        for seed in seeds:
            group_out = (
                group_out_root / f"seed_{seed:04d}"
                if len(seeds) > 1
                else group_out_root
            )
            env = os.environ.copy()
            env.update(
                {
                    "PAIRS_JSON": json.dumps(pairs),
                    "MODE": str(config["mode"]),
                    "STEPS": str(config["steps"]),
                    "BATCH_SIZE": str(config["batch_size"]),
                    "SEED": str(seed),
                    "SCALE": str(config["scale"]),
                }
            )
            if args.dry_run:
                env["DRY_RUN"] = "1"

            print(f"    seed {seed} -> {group_out}", flush=True)
            subprocess.run(
                ["bash", str(GRID_SCRIPT), str(group_out)],
                check=True,
                env=env,
            )

    print(f"\nAll requested groups completed. Outputs in: {base_out}", flush=True)


if __name__ == "__main__":
    main()
