"""
Control Group Landscape Survey
===============================
Runs trajectory_dynamics_experiment.py over all candidate control pairs
(independent / orthogonal concepts) to identify which pairs are consistent
and well-suited for the control group in Phase 1.

Produces per-pair decoded images and trajectory manifolds only.
No classifier probes. Models are loaded once and reused across all pairs.

Output structure
----------------
experiments/eccv2026/taxonomy/
    group1_cooccurrence/
        camel__x__desert_landscape/
            decoded_images.png
            trajectory_manifold.png
            trajectory_subplots.png
            pairwise_distances.png
            summary.json              (includes "taxonomy_group": "group1_cooccurrence")
        butterfly__x__flower_meadow/
            ...
        (6 pairs)
    group2_disentangled/
        dog__x__oil_painting_style/
            ...
        (6 pairs)
    group3_ood/
        desk_lamp__x__glacier/
            ...
        (6 pairs)
    group4_collision/
        cat__x__dog/
            ...
        (6 pairs)

Usage
-----
    conda activate attend_excite
    python scripts/run_control_landscape.py

    # Override seed or steps:
    python scripts/run_control_landscape.py --seed 43 --steps 30
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_SCRIPT = PROJECT_ROOT / "scripts" / "trajectory_dynamics_experiment.py"

# ---------------------------------------------------------------------------
# Candidate control pairs, grouped by the reason for orthogonality
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Four-group theory-grounded taxonomy (Bradley et al. 2502.04549)
# Group 1: Manifold-Supported Co-occurrence   (Setting C / Bayes regime)
# Group 2: Feature-Space Disentangled         (Theorem 6.1 / style-content)
# Group 3: Low Co-occurrence, OOD             (Lemma 8.1 high orth_dot)
# Group 4: Adversarial Collision              (single semantic slot competition)
# ---------------------------------------------------------------------------

GROUPS = {
    "group1_cooccurrence": [
        ("a camel",             "a desert landscape"),
        ("a butterfly",         "a flower meadow"),
        ("a dolphin",           "an ocean wave"),
        ("a lion",              "a savanna at sunset"),
        ("a lighthouse",        "a stormy sea"),
        ("a sailboat",          "a lighthouse on shore"),
    ],
    "group2_disentangled": [
        ("a dog",               "oil painting style"),
        ("a lighthouse",        "watercolour style"),
        ("a bicycle",           "sketch style"),
        ("a teapot",            "claymation style"),
        ("a barn",              "pencil drawing style"),
        ("a cactus",            "mosaic style"),
    ],
    "group3_ood": [
        ("a desk lamp",         "a glacier"),           # orth_dot 0.251
        ("a bathtub",           "a streetlamp"),        # orth_dot 0.253
        ("a lab microscope",    "a hay bale"),          # orth_dot 0.267
        ("a black grand piano", "a white vase"),        # orth_dot 0.311
        ("a typewriter",        "a cactus"),            # orth_dot 0.315
    ],
    "group4_collision": [
        ("a cat",               "a dog"),
        ("a cat",               "an owl"),
        ("a cat",               "a bear"),
        ("a teddy bear",        "a panda"),
        ("an otter",            "a duck"),
        ("a tiger",             "a lion"),
    ],
}


def _slugify(text: str) -> str:
    return text.lower().replace(" ", "_").replace(",", "").replace("'", "")


def run_group(group_name: str, pairs: list, base_out: Path, seed: int, steps: int) -> None:
    group_out = base_out / group_name
    group_out.mkdir(parents=True, exist_ok=True)

    grid_pairs_json = json.dumps([[a, b] for a, b in pairs])

    cmd = [
        sys.executable, str(EXPERIMENT_SCRIPT),
        "--grid-pairs", grid_pairs_json,
        "--grid-output-dir", str(group_out),
        "--seed", str(seed),
        "--steps", str(steps),
        "--projection", "mds",
        "--no-clip-probe",
        "--uniform-color",
        "--superdiff-variant", "fm_ode",
        "--taxonomy-group", group_name,
    ]

    print(f"\n{'='*60}")
    print(f"  Group: {group_name}  ({len(pairs)} pairs, seed={seed})")
    print(f"  Output: {group_out}")
    print(f"{'='*60}")
    for a, b in pairs:
        print(f"    {a}  x  {b}")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[WARNING] Group '{group_name}' exited with code {result.returncode}. Continuing.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Control group landscape survey.")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(GROUPS.keys()),
        default=list(GROUPS.keys()),
        help="Subset of orthogonal-control groups to run. Defaults to all groups.",
    )
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--steps", type=int, default=50,
                        help="Denoising steps per run. Use 30 for a faster sweep.")
    parser.add_argument("--out",   type=str, default="",
                        help="Base output directory. Defaults to experiments/control_landscape/")
    args = parser.parse_args()

    base_out = Path(args.out) if args.out else PROJECT_ROOT / "experiments" / "eccv2026" / "taxonomy"
    base_out.mkdir(parents=True, exist_ok=True)

    print(f"\nControl landscape survey")
    print(f"  Base output : {base_out}")
    print(f"  Groups      : {', '.join(args.groups)}")
    print(f"  Seed        : {args.seed}")
    print(f"  Steps       : {args.steps}")
    total = sum(len(GROUPS[group]) for group in args.groups)
    print(f"  Total pairs : {total} across {len(args.groups)} groups")

    for group_name in args.groups:
        run_group(group_name, GROUPS[group_name], base_out, seed=args.seed, steps=args.steps)

    print(f"\nAll groups complete. Results in: {base_out}")
    print("Each group folder contains:")
    print("  <pair_slug>/decoded_images.png")
    print("  <pair_slug>/trajectory_manifold.png")
    print("  decoded_images_grid.png   (group-level overview)")
    print("  trajectory_manifold_grid.png")


if __name__ == "__main__":
    main()
