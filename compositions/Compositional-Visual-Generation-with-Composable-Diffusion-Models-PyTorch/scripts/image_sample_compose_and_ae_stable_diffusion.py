"""
image_sample_compose_and_ae_stable_diffusion.py
================================================
Runner for ComposableAttendAndExcitePipeline: AND compositional diffusion +
Attend-and-Excite Generative Semantic Nursing.

Usage example
-------------
    python scripts/image_sample_compose_and_ae_stable_diffusion.py \
        --joint_prompt "a cat and a dog" \
        --subject_prompts "a cat | a dog" \
        --indices_to_alter "2,5" \
        --comp_weights "7.5 | 7.5" \
        --steps 50 \
        --scale 7.5 \
        --seed 42

The ``--indices_to_alter`` values are 1-based token positions in the
tokenised ``--joint_prompt`` (BOS token is index 0, which is excluded).
Run with ``--show_token_map`` to print the token→index mapping before
generating so you can pick the right indices for your joint prompt.

Environment
-----------
Activate the `attend_excite` conda environment before running:
    conda activate attend_excite
    # diffusers==0.12.1, torch==2.7.0, transformers==4.26.0

The composable_diffusion package (this repo) must also be installed:
    pip install -e .   (from the repo root)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as tvu

# ── Add the composable-diffusion scripts dir to sys.path so that the
#    local pipeline module is importable without package installation.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from pipeline_composable_attend_excite import ComposableAttendAndExcitePipeline

# ── Scheduler choices (same as the original composable SD script) ─────────────
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AND compositional diffusion + Attend-and-Excite GSN"
    )

    # ── Prompts ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--joint_prompt", type=str, required=True,
        help=(
            "Single combined prompt used for A&E attention monitoring. "
            'Example: "a cat and a dog"'
        ),
    )
    p.add_argument(
        "--subject_prompts", type=str, required=True,
        help=(
            "Pipe-separated individual subject prompts for AND composition. "
            'Example: "a cat | a dog"'
        ),
    )
    p.add_argument(
        "--indices_to_alter", type=str, required=True,
        help=(
            "Comma-separated 1-based token indices in --joint_prompt for "
            "the subject tokens A&E should enforce. "
            'Example: "2,5"  (token 2 = "cat", token 5 = "dog")'
        ),
    )
    p.add_argument(
        "--comp_weights", type=str, default=None,
        help=(
            "Pipe-separated guidance weights for AND composition. "
            "Defaults to --scale for every subject. "
            'Example: "7.5 | 7.5"'
        ),
    )

    # ── Generation ────────────────────────────────────────────────────────────
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--num_images", type=int, default=1)
    p.add_argument("--model_path", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument(
        "--scheduler", type=str, default="ddim",
        choices=["lms", "ddim", "ddpm", "pndm"],
    )
    p.add_argument("--sd_2_1", action="store_true", help="Use SD 2.1 (enables EOT normalisation).")

    # ── A&E layout-phase params ───────────────────────────────────────────────
    p.add_argument(
        "--max_iter_to_alter", type=int, default=25,
        help="Number of timesteps (from t=0) during which A&E gradient corrections are applied.",
    )
    p.add_argument(
        "--attention_res", type=int, default=16,
        help="Cross-attention map resolution used for A&E (16 = 16×16).",
    )
    p.add_argument(
        "--scale_factor", type=int, default=20,
        help="Base gradient step size for the A&E latent update.",
    )
    p.add_argument(
        "--smooth_attentions", action="store_true", default=True,
        help="Apply Gaussian smoothing (kernel=3, σ=0.5) to attention maps.",
    )
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--kernel_size", type=int, default=3)

    # ── Internal noise-blend (advanced, leave at defaults for pure AND+A&E) ────
    # These control an optional in-loop blend of AND noise with monolithic noise.
    # Default is alpha=1.0 constant → pure AND+A&E, no blending.
    # For a visual sweep between the two modes, use alpha_sweep.py instead.
    p.add_argument(
        "--alpha", type=float, default=1.0,
        help=(
            "Noise-prediction blend weight: "
            "1.0 = pure AND+A&E (default), 0.0 = pure monolithic CFG. "
            "Use alpha_sweep.py for pixel-level interpolation between two generated images."
        ),
    )
    p.add_argument(
        "--alpha_end", type=float, default=1.0,
        help="Final α at the last denoising step.  Default 1.0 (no decay → pure AND+A&E).",
    )
    p.add_argument(
        "--alpha_schedule", type=str, default="constant",
        choices=["constant", "linear", "cosine"],
        help="How α changes over timesteps.  Default 'constant' (no decay).",
    )

    # ── A&E loss aggregation mode ─────────────────────────────────────────────
    p.add_argument(
        "--loss_mode", type=str, default="sum",
        choices=["max", "sum"],
        help=(
            "How per-token A&E losses are combined. "
            "'sum' (default): all neglected tokens push grad simultaneously. "
            "'max' (original A&E): only the single weakest token."
        ),
    )

    # ── Diagnostics ───────────────────────────────────────────────────────────
    p.add_argument(
        "--show_token_map", action="store_true",
        help="Print the token→index map for --joint_prompt and exit.",
    )

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Token-map helper
# ─────────────────────────────────────────────────────────────────────────────

def print_token_map(pipe: ComposableAttendAndExcitePipeline, prompt: str) -> None:
    """
    Prints the 1-based token index → decoded sub-word mapping for `prompt`.
    Use this to identify which indices to pass to --indices_to_alter.
    """
    token_ids = pipe.tokenizer(prompt)["input_ids"]
    print(f"\nToken map for: '{prompt}'")
    print("  idx  token")
    print("  ---  -----")
    for idx, tid in enumerate(token_ids):
        word = pipe.tokenizer.decode([tid])
        print(f"  {idx:>3}  {word!r}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────────
    model_id = args.model_path
    if args.sd_2_1:
        model_id = "stabilityai/stable-diffusion-2-1-base"

    print(f"Loading model: {model_id}")
    pipe = ComposableAttendAndExcitePipeline.from_pretrained(model_id).to(device)
    pipe.safety_checker = None  # disable for research use

    # Optional: override scheduler
    if args.scheduler == "lms":
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == "ddpm":
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == "pndm":
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)

    # ── Optionally show token map and exit ────────────────────────────────────
    if args.show_token_map:
        print_token_map(pipe, args.joint_prompt)
        return

    # ── Parse CLI args ────────────────────────────────────────────────────────
    subject_prompts = [p.strip() for p in args.subject_prompts.split("|")]
    indices_to_alter = [int(i.strip()) for i in args.indices_to_alter.split(",")]

    print(f"\nJoint prompt    : {args.joint_prompt}")
    print(f"Subject prompts : {subject_prompts}")
    print(f"Indices to alter: {indices_to_alter}")
    print(f"Comp weights    : {args.comp_weights or 'equal (= --scale)'}")
    print(f"Max A&E steps   : {args.max_iter_to_alter} / {args.steps}")
    print(f"Loss mode       : {args.loss_mode}")
    blend_info = "pure AND+A&E" if args.alpha == 1.0 and args.alpha_end == 1.0 else f"α {args.alpha:.2f} → {args.alpha_end:.2f} ({args.alpha_schedule})"
    print(f"Noise blend     : {blend_info}")
    print()

    # ── Generate ──────────────────────────────────────────────────────────────
    images = []
    generator = torch.Generator(device).manual_seed(args.seed)

    for _ in range(args.num_images):
        output = pipe(
            # Joint prompt for A&E
            prompt=args.joint_prompt,
            # Subject prompts for AND composition
            subject_prompts=subject_prompts,
            # Token indices to enforce
            indices_to_alter=indices_to_alter,
            # Composition weights
            comp_weights=args.comp_weights,
            # Standard generation params
            guidance_scale=args.scale,
            num_inference_steps=args.steps,
            generator=generator,
            # A&E params
            max_iter_to_alter=args.max_iter_to_alter,
            attention_res=args.attention_res,
            scale_factor=args.scale_factor,
            smooth_attentions=args.smooth_attentions,
            sigma=args.sigma,
            kernel_size=args.kernel_size,
            thresholds={0: 0.05, 10: 0.5, 20: 0.8},
            scale_range=(1.0, 0.5),
            sd_2_1=args.sd_2_1,
            # Guided blend
            alpha=args.alpha,
            alpha_end=args.alpha_end,
            alpha_schedule=args.alpha_schedule,
            # Loss mode
            loss_mode=args.loss_mode,
        )
        images.append(
            torch.from_numpy(np.array(output.images[0])).permute(2, 0, 1) / 255.0
        )

    # ── Save output ───────────────────────────────────────────────────────────
    grid = tvu.make_grid(torch.stack(images, dim=0), nrow=4, padding=0)

    slug_joint = args.joint_prompt.strip().replace(" ", "_")[:60]
    slug_subj  = "_AND_".join(p.strip().replace(" ", "_") for p in subject_prompts)
    out_path   = f"{slug_subj}__AE_{slug_joint}__seed{args.seed}.png"

    tvu.save_image(grid, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
