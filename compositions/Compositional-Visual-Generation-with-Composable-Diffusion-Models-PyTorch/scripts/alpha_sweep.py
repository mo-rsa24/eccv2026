"""
alpha_sweep.py
==============
Sweeps α ∈ [0, 1] to find the balance between:

    α = 0  →  monolithic SD  ("a cat and a dog", single prompt, no composition)
    α = 1  →  AND+Attend-and-Excite  (full compositional pipeline)

At every denoising timestep the noise prediction is:

    ε_out = (1 − α) · ε_mono + α · ε_AND

Attend-and-Excite GSN runs identically for all α values, ensuring the
spatial layout is enforced regardless of composition strength.  Only the
noise estimate fed to the scheduler changes.

Usage
-----
    python scripts/alpha_sweep.py \\
        --joint_prompt "a cat and a dog" \\
        --subject_prompts "a cat | a dog" \\
        --indices_to_alter "2,5" \\
        --alphas "0.0,0.25,0.5,0.75,1.0" \\
        --seed 42

All other generation and A&E arguments are the same as
image_sample_compose_and_ae_stable_diffusion.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as tvu
from PIL import Image, ImageDraw, ImageFont

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from pipeline_composable_attend_excite import ComposableAttendAndExcitePipeline

from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep α between monolithic SD and AND+A&E composition."
    )

    # ── Prompts ───────────────────────────────────────────────────────────────
    p.add_argument("--joint_prompt",    type=str, required=True)
    p.add_argument("--subject_prompts", type=str, required=True,
                   help='Pipe-separated, e.g. "a cat | a dog"')
    p.add_argument("--indices_to_alter", type=str, required=True,
                   help='Comma-separated 1-based token indices, e.g. "2,5"')
    p.add_argument("--comp_weights", type=str, default=None)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--alphas", type=str, default="0.0,0.25,0.5,0.75,1.0",
        help=(
            "Comma-separated α values to sweep.  "
            "0 = pure monolithic, 1 = pure AND+A&E.  "
            'Default: "0.0,0.25,0.5,0.75,1.0"'
        ),
    )

    # ── Generation ────────────────────────────────────────────────────────────
    p.add_argument("--steps",      type=int,   default=50)
    p.add_argument("--scale",      type=float, default=7.5)
    p.add_argument("--seed",       type=int,   default=8)
    p.add_argument("--model_path", type=str,   default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--scheduler",  type=str,   default="ddim",
                   choices=["lms", "ddim", "ddpm", "pndm"])
    p.add_argument("--sd_2_1",     action="store_true")

    # ── A&E params ────────────────────────────────────────────────────────────
    p.add_argument("--max_iter_to_alter", type=int,   default=25)
    p.add_argument("--attention_res",     type=int,   default=16)
    p.add_argument("--scale_factor",      type=int,   default=20)
    p.add_argument("--smooth_attentions", action="store_true", default=True)
    p.add_argument("--sigma",             type=float, default=0.5)
    p.add_argument("--kernel_size",       type=int,   default=3)
    p.add_argument("--loss_mode",         type=str,   default="sum",
                   choices=["max", "sum"])

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--out", type=str, default="alpha_sweep.png",
                   help="Output grid image path.")
    p.add_argument("--label", action="store_true", default=True,
                   help="Annotate each image with its α value (default: on).")
    p.add_argument("--no_label", dest="label", action="store_false")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Annotation helper
# ─────────────────────────────────────────────────────────────────────────────

def annotate(tensor: torch.Tensor, text: str, font_size: int = 22) -> torch.Tensor:
    """Burn a label into the bottom of a [3, H, W] float tensor."""
    arr = (tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except Exception:
        font = ImageFont.load_default()

    W, H = img.size
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = (W - tw) // 2, H - th - 6
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)   # shadow
    draw.text((x, y),         text, fill=(255, 255, 255), font=font)

    out = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(out).permute(2, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    alphas = [float(a.strip()) for a in args.alphas.split(",")]
    print(f"Alpha sweep: {alphas}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id = "stabilityai/stable-diffusion-2-1-base" if args.sd_2_1 else args.model_path
    print(f"Loading model: {model_id}")
    pipe = ComposableAttendAndExcitePipeline.from_pretrained(model_id).to(device)
    pipe.safety_checker = None

    sched_map = {
        "lms":  LMSDiscreteScheduler,
        "ddim": DDIMScheduler,
        "ddpm": DDPMScheduler,
        "pndm": PNDMScheduler,
    }
    pipe.scheduler = sched_map[args.scheduler].from_config(pipe.scheduler.config)

    subject_prompts  = [p.strip() for p in args.subject_prompts.split("|")]
    indices_to_alter = [int(i.strip()) for i in args.indices_to_alter.split(",")]

    print(f"Joint prompt    : {args.joint_prompt}")
    print(f"Subject prompts : {subject_prompts}")
    print(f"Indices to alter: {indices_to_alter}")
    print()

    frames: list[torch.Tensor] = []

    for alpha in alphas:
        label = f"α={alpha:.2f}"
        if alpha == 0.0:
            label += " (mono)"
        elif alpha == 1.0:
            label += " (AND+A&E)"
        print(f"── Generating {label} ──")

        generator = torch.Generator(device).manual_seed(args.seed)

        output = pipe(
            prompt=args.joint_prompt,
            subject_prompts=subject_prompts,
            indices_to_alter=indices_to_alter,
            comp_weights=args.comp_weights,
            guidance_scale=args.scale,
            num_inference_steps=args.steps,
            generator=generator,
            max_iter_to_alter=args.max_iter_to_alter,
            attention_res=args.attention_res,
            scale_factor=args.scale_factor,
            smooth_attentions=args.smooth_attentions,
            sigma=args.sigma,
            kernel_size=args.kernel_size,
            thresholds={0: 0.05, 10: 0.5, 20: 0.8},
            scale_range=(1.0, 0.5),
            sd_2_1=args.sd_2_1,
            # Fixed α throughout — pure constant blend at this value
            alpha=alpha,
            alpha_end=alpha,
            alpha_schedule="constant",
            loss_mode=args.loss_mode,
        )

        img_t = torch.from_numpy(
            np.array(output.images[0])
        ).permute(2, 0, 1).float() / 255.0

        if args.label:
            img_t = annotate(img_t, label)

        frames.append(img_t)
        print()

    grid = tvu.make_grid(torch.stack(frames), nrow=len(frames), padding=4)
    tvu.save_image(grid, args.out)
    print(f"Saved: {args.out}  ({len(frames)} images)")


if __name__ == "__main__":
    main()
