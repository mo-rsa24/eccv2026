"""
Phase 1 Taxonomy Qualitative Decoded Images
============================================
Generates decoded images for the Phase 1 qualitative figure using the original
ComposableStableDiffusionPipeline (SD 1.4, diffusers 0.10.2).

Four conditions per pair, all starting from the same x_T (shared-noise protocol):
    (i)   solo A
    (ii)  solo B
    (iii) monolithic  ("A and B")
    (iv)  PoE / AND   ("A | B", pipe-separated, equal weights)

Taxonomy pairs are exactly those cited in
    proposal/proposal_stage_3/chapters/research_method/phase_1.tex:

    Group 1 — 4 pairs: manifold-supported co-occurrence (Bayes Setting C)
    Group 2 — 3 pairs: feature-space disentangled (style-content, Theorem 6.1)
    Group 3 — 5 pairs: low co-occurrence, OOD (Lemma 8.1 high orth_dot)
    Group 4 — 3 pairs: adversarial collision (single semantic slot)

Usage
-----
    conda activate compose_diff
    python scripts/run_taxonomy_qualitative.py

    # Override seed, steps, or guidance scale:
    python scripts/run_taxonomy_qualitative.py --seed 43 --steps 50 --scale 7.5

    # Run a single group:
    python scripts/run_taxonomy_qualitative.py --groups group4_collision

Output structure
----------------
experiments/eccv2026/taxonomy_qualitative/
    group1_cooccurrence/
        camel__x__desert_landscape/
            solo_a.png
            solo_b.png
            monolithic.png
            poe.png
            panel.png          <- 1×4 combined panel for this pair
        ...
        decoded_images_grid.png  <- group-level overview
    group2_disentangled/
    group3_ood/
    group4_collision/
    decoded_images_grid.png      <- full proposal figure (all groups)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch as th
import torchvision.utils as tvu
from PIL import Image

# ---------------------------------------------------------------------------
# Project / repo paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPOSABLE_REPO = (
    PROJECT_ROOT
    / "compositions"
    / "Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch"
)
sys.path.insert(0, str(COMPOSABLE_REPO))

from diffusers import DDIMScheduler  # noqa: E402
from composable_diffusion.composable_stable_diffusion.pipeline_composable_stable_diffusion import (  # noqa: E402
    ComposableStableDiffusionPipeline,
)

# ---------------------------------------------------------------------------
# Taxonomy pairs — exactly as cited in phase_1.tex
# ---------------------------------------------------------------------------
TAXONOMY_GROUPS = {
    "group1_cooccurrence": [
        ("a camel",             "a desert landscape"),   # §3 p.54
        ("a butterfly",         "a flower meadow"),
        ("a dolphin",           "an ocean wave"),
        ("a lion",              "a savanna at sunset"),
    ],
    "group2_disentangled": [
        ("a dog",               "oil painting style"),   # §3 p.60
        ("a lighthouse",        "watercolour style"),
        ("a bicycle",           "sketch style"),
    ],
    "group3_ood": [
        ("a desk lamp",         "a glacier"),            # orth_dot 0.251  §3 p.67
        ("a bathtub",           "a streetlamp"),         # orth_dot 0.253
        ("a lab microscope",    "a hay bale"),           # orth_dot 0.267
        ("a black grand piano", "a white vase"),         # orth_dot 0.311
        ("a typewriter",        "a cactus"),             # orth_dot 0.315
    ],
    "group4_collision": [
        ("a cat",               "a dog"),                # §3 p.74
        ("a cat",               "an owl"),
        ("a cat",               "a bear"),
    ],
}

# Condition order used for all panels / grids
CONDITIONS = ["solo_a", "solo_b", "monolithic", "poe"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace(",", "")
        .replace("'", "")
        .replace("/", "")
    )


def _pil_to_tensor(img: Image.Image) -> th.Tensor:
    """PIL Image → CHW float tensor in [0, 1]."""
    return th.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def _generate(
    pipe: ComposableStableDiffusionPipeline,
    prompt: str,
    weights: str,
    scale: float,
    steps: int,
    seed: int,
    device_str: str,
) -> Image.Image:
    """Run the pipeline for one condition.

    The generator is re-seeded before each call so that every condition within
    a pair starts from the same initial Gaussian noise x_T — the shared-noise
    protocol that isolates the composition operator as the sole variable.
    """
    generator = th.Generator(device_str).manual_seed(seed)
    result = pipe(
        prompt,
        guidance_scale=scale,
        num_inference_steps=steps,
        weights=weights,
        generator=generator,
    )
    return result.images[0]


# ---------------------------------------------------------------------------
# Per-pair runner
# ---------------------------------------------------------------------------

def run_pair(
    pipe: ComposableStableDiffusionPipeline,
    concept_a: str,
    concept_b: str,
    out_dir: Path,
    scale: float,
    steps: int,
    seed: int,
    device_str: str,
) -> th.Tensor:
    """Generate all 4 conditions for one pair.

    Returns a (4, C, H, W) stacked tensor (one image per condition, in
    CONDITIONS order).  Also saves individual PNGs and a 1×4 panel.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_single = str(scale)
    weight_poe = f"{scale} | {scale}"

    condition_configs: dict[str, tuple[str, str]] = {
        "solo_a":     (concept_a,                          weight_single),
        "solo_b":     (concept_b,                          weight_single),
        "monolithic": (f"{concept_a} and {concept_b}",     weight_single),
        "poe":        (f"{concept_a} | {concept_b}",       weight_poe),
    }

    tensors: list[th.Tensor] = []
    for cond in CONDITIONS:
        prompt, weights = condition_configs[cond]
        img = _generate(pipe, prompt, weights, scale, steps, seed, device_str)
        img.save(out_dir / f"{cond}.png")
        tensors.append(_pil_to_tensor(img))

    row = th.stack(tensors)           # (4, C, H, W)
    panel = tvu.make_grid(row, nrow=4, padding=2)
    tvu.save_image(panel, out_dir / "panel.png")
    print(f"    [{concept_a}]  ×  [{concept_b}]  →  {out_dir.name}")
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 taxonomy qualitative decoded images (SD 1.4, composable diffusion)."
    )
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--steps",  type=int,   default=50)
    parser.add_argument("--scale",  type=float, default=7.5,
                        help="CFG guidance scale (applied to all conditions).")
    parser.add_argument("--model",  type=str,   default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--out",    type=str,   default="",
                        help="Base output directory. Defaults to "
                             "experiments/eccv2026/taxonomy_qualitative/")
    parser.add_argument(
        "--groups", nargs="+",
        choices=sorted(TAXONOMY_GROUPS.keys()),
        default=list(TAXONOMY_GROUPS.keys()),
        help="Subset of taxonomy groups to run. Defaults to all four groups.",
    )
    args = parser.parse_args()

    base_out = (
        Path(args.out) if args.out
        else PROJECT_ROOT / "experiments" / "eccv2026" / "taxonomy_qualitative"
    )
    base_out.mkdir(parents=True, exist_ok=True)

    has_cuda = th.cuda.is_available()
    device_str = "cuda" if has_cuda else "cpu"
    device = th.device(device_str)

    total_pairs = sum(len(TAXONOMY_GROUPS[g]) for g in args.groups)
    print(f"\nPhase 1 taxonomy qualitative run")
    print(f"  Model   : {args.model}")
    print(f"  Seed    : {args.seed}   Steps: {args.steps}   Scale: {args.scale}")
    print(f"  Groups  : {', '.join(args.groups)}")
    print(f"  Pairs   : {total_pairs}  ×  4 conditions = {total_pairs * 4} images")
    print(f"  Output  : {base_out}\n")

    print(f"Loading pipeline ...")
    pipe = ComposableStableDiffusionPipeline.from_pretrained(args.model).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    print("Pipeline ready.\n")

    all_rows: list[th.Tensor] = []

    for group_name in args.groups:
        pairs = TAXONOMY_GROUPS[group_name]
        group_out = base_out / group_name
        group_out.mkdir(parents=True, exist_ok=True)

        print(f"{'='*60}")
        print(f"  {group_name}  ({len(pairs)} pairs)")
        print(f"{'='*60}")

        group_rows: list[th.Tensor] = []
        for concept_a, concept_b in pairs:
            pair_slug = f"{_slugify(concept_a)}__x__{_slugify(concept_b)}"
            pair_out = group_out / pair_slug
            row = run_pair(
                pipe, concept_a, concept_b, pair_out,
                scale=args.scale, steps=args.steps,
                seed=args.seed, device_str=device_str,
            )
            group_rows.append(row)      # each: (4, C, H, W)

        # Group-level grid — each pair occupies one row of 4 images
        group_tensor = th.cat(group_rows, dim=0)        # (4*n_pairs, C, H, W)
        group_grid = tvu.make_grid(group_tensor, nrow=4, padding=2)
        tvu.save_image(group_grid, group_out / "decoded_images_grid.png")
        print(f"  Group grid → {group_out / 'decoded_images_grid.png'}\n")
        all_rows.append(group_tensor)

    # Full proposal figure — all groups concatenated
    full_tensor = th.cat(all_rows, dim=0)
    full_grid = tvu.make_grid(full_tensor, nrow=4, padding=2)
    full_grid_path = base_out / "decoded_images_grid.png"
    tvu.save_image(full_grid, full_grid_path)

    print(f"\nAll groups complete.")
    print(f"Full proposal figure → {full_grid_path}")
    print(f"Copy to paper assets :")
    print(f"  cp {full_grid_path} proposal/proposal_stage_3/chapters/research_method/media/decoded_images_grid.png")


if __name__ == "__main__":
    main()
