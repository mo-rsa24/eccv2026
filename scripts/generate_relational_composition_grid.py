#!/usr/bin/env python3
"""
Generate a 4-column image-only grid for relational prompt pairs using the
composable Stable Diffusion method from:

  compositions/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch

Columns:
  1. Concept A
  2. Concept B
  3. Semantic composition (single monolithic prompt)
  4. PoE composition (prompt_a | prompt_b)

By default the script builds 10 cat/dog left-right prompt variants, including
the exact examples:
  - "A cat on the left" + "A dog on the right"
  - "A cat on the left of the photo" + "A dog on the right of the photo"

Example:
  python scripts/generate_relational_composition_grid.py \
      --concept-a "a cat" \
      --concept-b "a dog" \
      --steps 50 \
      --scale 7.5
"""

import argparse
import gc
import inspect
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPOSABLE_ROOT = (
    PROJECT_ROOT
    / "compositions"
    / "Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch"
)

DEFAULT_RELATIONAL_VARIANTS = [
    {
        "name": "plain_left_right",
        "prompt_a": "{concept_a_cap} on the left",
        "prompt_b": "{concept_b_cap} on the right",
        "semantic_prompt": "{concept_a_cap} on the left and {concept_b} on the right",
    },
    {
        "name": "plain_left_right_photo",
        "prompt_a": "{concept_a_cap} on the left of the photo",
        "prompt_b": "{concept_b_cap} on the right of the photo",
        "semantic_prompt": (
            "{concept_a_cap} on the left of the photo and {concept_b} on the right of the photo"
        ),
    },
    {
        "name": "photo_prefix_left_right",
        "prompt_a": "A photo of {concept_a} on the left",
        "prompt_b": "A photo of {concept_b} on the right",
        "semantic_prompt": "A photo of {concept_a} on the left and {concept_b} on the right",
    },
    {
        "name": "photo_prefix_photo",
        "prompt_a": "A photo of {concept_a} on the left of the photo",
        "prompt_b": "A photo of {concept_b} on the right of the photo",
        "semantic_prompt": (
            "A photo of {concept_a} on the left of the photo and {concept_b} on the right of the photo"
        ),
    },
    {
        "name": "side_of_image",
        "prompt_a": "{concept_a_cap} on the left side of the image",
        "prompt_b": "{concept_b_cap} on the right side of the image",
        "semantic_prompt": (
            "{concept_a_cap} on the left side of the image and {concept_b} on the right side of the image"
        ),
    },
    {
        "name": "photo_side_of_image",
        "prompt_a": "A photo of {concept_a} on the left side of the image",
        "prompt_b": "A photo of {concept_b} on the right side of the image",
        "semantic_prompt": (
            "A photo of {concept_a} on the left side of the image and {concept_b} on the right side of the image"
        ),
    },
    {
        "name": "photo_positioned",
        "prompt_a": "A photo of {concept_a} positioned on the left",
        "prompt_b": "A photo of {concept_b} positioned on the right",
        "semantic_prompt": (
            "A photo of {concept_a} positioned on the left and {concept_b} positioned on the right"
        ),
    },
    {
        "name": "half_photo",
        "prompt_a": "{concept_a_cap} in the left half of the photo",
        "prompt_b": "{concept_b_cap} in the right half of the photo",
        "semantic_prompt": (
            "{concept_a_cap} in the left half of the photo and {concept_b} in the right half of the photo"
        ),
    },
    {
        "name": "photo_frame",
        "prompt_a": "A photo of {concept_a} on the left side of the frame",
        "prompt_b": "A photo of {concept_b} on the right side of the frame",
        "semantic_prompt": (
            "A photo of {concept_a} on the left side of the frame and {concept_b} on the right side of the frame"
        ),
    },
    {
        "name": "far_left_far_right",
        "prompt_a": "{concept_a_cap} on the far left",
        "prompt_b": "{concept_b_cap} on the far right",
        "semantic_prompt": "{concept_a_cap} on the far left and {concept_b} on the far right",
    },
]

COLUMN_SPECS = [
    ("prompt_a", "Concept A"),
    ("prompt_b", "Concept B"),
    ("semantic_prompt", "Semantic"),
    ("poe", "PoE"),
]


def default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a 4-column grid for relational prompt pairs using the "
            "composable diffusion Stable Diffusion pipeline."
        )
    )
    parser.add_argument(
        "--concept-a",
        type=str,
        default="a cat",
        help='Base concept phrase for the left-side object, e.g. "a cat".',
    )
    parser.add_argument(
        "--concept-b",
        type=str,
        default="a dog",
        help='Base concept phrase for the right-side object, e.g. "a dog".',
    )
    parser.add_argument(
        "--pairs-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file with a list of prompt specs. Each item must have "
            '"prompt_a" and "prompt_b"; "semantic_prompt" is optional.'
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of prompt rows to render.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed. Row i uses seed + i to keep rows reproducible.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of diffusion steps.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--poe-weights",
        type=str,
        default="",
        help=(
            'Optional compositional weights string for PoE, e.g. "7.5 | 7.5". '
            "If omitted, the upstream pipeline uses equal positive weights."
        ),
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Optional negative prompt passed to all four columns.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Model id or local path for the composable Stable Diffusion pipeline.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["lms", "ddim", "ddpm", "pndm"],
        default="ddim",
        help="Diffusion scheduler.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Generation height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Generation width.",
    )
    parser.add_argument(
        "--grid-cell-size",
        type=int,
        default=384,
        help="Displayed cell size inside the saved grid.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Torch device, e.g. "auto", "cuda", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="Optional Hugging Face token.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "relational_composition_grids",
        help="Directory for the generated grid and manifest.",
    )
    parser.add_argument(
        "--filename-prefix",
        type=str,
        default="relational_composition",
        help="Prefix for the timestamped output directory.",
    )
    return parser.parse_args()


def raise_missing_runtime_deps(exc: Exception) -> None:
    raise RuntimeError(
        "Missing runtime dependencies for composable diffusion generation. "
        "Install the upstream package and its dependencies first, for example:\n"
        "  cd compositions/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch\n"
        "  pip install -e .\n"
        "  pip install diffusers transformers accelerate torchvision"
    ) from exc


def ensure_composable_import_path() -> None:
    if not COMPOSABLE_ROOT.exists():
        raise FileNotFoundError(f"Composable diffusion repo not found: {COMPOSABLE_ROOT}")
    root_str = str(COMPOSABLE_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def call_with_supported_kwargs(callable_obj: Callable[..., Any], **kwargs: Any) -> Any:
    signature = inspect.signature(callable_obj)
    accepted = {}
    for key, value in kwargs.items():
        if key in signature.parameters and value is not None:
            accepted[key] = value
    return callable_obj(**accepted)


def add_token_if_supported(from_pretrained: Callable[..., Any], kwargs: Dict[str, Any], token: Optional[str]) -> Dict[str, Any]:
    if not token:
        return kwargs
    signature = inspect.signature(from_pretrained)
    if "token" in signature.parameters:
        kwargs["token"] = token
    elif "use_auth_token" in signature.parameters:
        kwargs["use_auth_token"] = token
    return kwargs


def capitalize_first(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def lower_first(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]


def render_template(template: str, concept_a: str, concept_b: str) -> str:
    return template.format(
        concept_a=concept_a,
        concept_b=concept_b,
        concept_a_cap=capitalize_first(concept_a),
        concept_b_cap=capitalize_first(concept_b),
    )


def infer_semantic_prompt(prompt_a: str, prompt_b: str) -> str:
    return f"{prompt_a} and {lower_first(prompt_b)}"


def build_default_pairs(concept_a: str, concept_b: str) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    for idx, spec in enumerate(DEFAULT_RELATIONAL_VARIANTS, start=1):
        prompt_a = render_template(spec["prompt_a"], concept_a, concept_b)
        prompt_b = render_template(spec["prompt_b"], concept_a, concept_b)
        semantic_prompt = render_template(spec["semantic_prompt"], concept_a, concept_b)
        pairs.append(
            {
                "row": idx,
                "name": spec["name"],
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "semantic_prompt": semantic_prompt,
            }
        )
    return pairs


def load_pairs(args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.pairs_json is None:
        pairs = build_default_pairs(args.concept_a, args.concept_b)
    else:
        with args.pairs_json.open("r", encoding="utf-8") as handle:
            raw_pairs = json.load(handle)

        if not isinstance(raw_pairs, list):
            raise ValueError("--pairs-json must contain a JSON list.")

        pairs = []
        for idx, raw_pair in enumerate(raw_pairs, start=1):
            if not isinstance(raw_pair, dict):
                raise ValueError(f"Pair {idx} is not a JSON object.")
            prompt_a = str(raw_pair["prompt_a"]).strip()
            prompt_b = str(raw_pair["prompt_b"]).strip()
            semantic_prompt = str(
                raw_pair.get("semantic_prompt") or infer_semantic_prompt(prompt_a, prompt_b)
            ).strip()
            pairs.append(
                {
                    "row": idx,
                    "name": str(raw_pair.get("name") or f"pair_{idx:02d}"),
                    "prompt_a": prompt_a,
                    "prompt_b": prompt_b,
                    "semantic_prompt": semantic_prompt,
                }
            )

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        pairs = pairs[: args.limit]

    for pair in pairs:
        pair["poe"] = f'{pair["prompt_a"]} | {pair["prompt_b"]}'

    if not pairs:
        raise ValueError("No prompt pairs were loaded.")

    return pairs


def make_generator(device: str, seed: int):
    try:
        import torch
    except Exception as exc:
        raise_missing_runtime_deps(exc)

    try:
        return torch.Generator(device=device).manual_seed(seed)
    except Exception:
        return torch.Generator().manual_seed(seed)


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return default_device()


def load_pipeline(args: argparse.Namespace):
    ensure_composable_import_path()
    try:
        import torch
        from diffusers import DDIMScheduler, DDPMScheduler, LMSDiscreteScheduler, PNDMScheduler
        from composable_diffusion.composable_stable_diffusion.pipeline_composable_stable_diffusion import (
            ComposableStableDiffusionPipeline,
        )
    except Exception as exc:
        raise_missing_runtime_deps(exc)

    device = torch.device(resolve_device(args.device))
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    load_kwargs: Dict[str, Any] = {}
    load_kwargs = add_token_if_supported(
        ComposableStableDiffusionPipeline.from_pretrained,
        load_kwargs,
        args.hf_token,
    )
    if "torch_dtype" in inspect.signature(ComposableStableDiffusionPipeline.from_pretrained).parameters:
        load_kwargs["torch_dtype"] = dtype

    pipe = ComposableStableDiffusionPipeline.from_pretrained(
        args.model_path,
        **load_kwargs,
    ).to(device)

    if args.scheduler == "lms":
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == "ddpm":
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == "pndm":
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)

    pipe.safety_checker = None

    for maybe_enable in ("enable_attention_slicing", "enable_vae_slicing"):
        fn = getattr(pipe, maybe_enable, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    return pipe, device, dtype


def prepare_shared_latents(pipe: Any, height: int, width: int, dtype: Any, device: Any, seed: int):
    generator = make_generator(str(device), seed)
    return pipe.prepare_latents(
        1,
        pipe.unet.in_channels,
        height,
        width,
        dtype,
        device,
        generator,
        None,
    )


def run_condition(
    pipe: Any,
    prompt: str,
    steps: int,
    scale: float,
    latents: Any,
    negative_prompt: Optional[str],
    weights: Optional[str] = None,
):
    call_kwargs = dict(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=scale,
        negative_prompt=negative_prompt,
        latents=latents,
        weights=weights if weights else None,
    )
    result = call_with_supported_kwargs(pipe.__call__, **call_kwargs)
    return result.images[0]


def safe_empty_cache(device: Any) -> None:
    try:
        import torch

        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception:
        pass


def make_error_image(width: int, height: int, text: str):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), color=(70, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.multiline_text((12, 12), text, fill=(255, 220, 220), spacing=4)
    return image


def text_size(draw: Any, text: str) -> List[int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text)
        return [right - left, bottom - top]
    return list(draw.textsize(text))


def lanczos_resample():
    from PIL import Image

    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def fit_with_padding(image: Any, size: int):
    from PIL import Image, ImageOps

    contained = ImageOps.contain(image.convert("RGB"), (size, size), method=lanczos_resample())
    canvas = Image.new("RGB", (size, size), color=(255, 255, 255))
    x = (size - contained.width) // 2
    y = (size - contained.height) // 2
    canvas.paste(contained, (x, y))
    return canvas


def save_grid(rows: List[Dict[str, Any]], output_path: Path, cell_size: int) -> None:
    from PIL import Image, ImageDraw

    n_rows = len(rows)
    n_cols = len(COLUMN_SPECS)
    left_margin = 44
    top_margin = 40
    grid_width = left_margin + n_cols * cell_size
    grid_height = top_margin + n_rows * cell_size

    canvas = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for col_idx, (_, title) in enumerate(COLUMN_SPECS):
        title_w, title_h = text_size(draw, title)
        x0 = left_margin + col_idx * cell_size
        title_x = x0 + (cell_size - title_w) // 2
        title_y = max(8, (top_margin - title_h) // 2)
        draw.text((title_x, title_y), title, fill=(0, 0, 0))

    for row_idx, row in enumerate(rows):
        y0 = top_margin + row_idx * cell_size
        label = f"{row_idx + 1:02d}"
        label_w, label_h = text_size(draw, label)
        draw.text((max(4, (left_margin - label_w) // 2), y0 + 8), label, fill=(0, 0, 0))

        for col_idx, (key, _) in enumerate(COLUMN_SPECS):
            image = fit_with_padding(row["images"][key], cell_size)
            x0 = left_margin + col_idx * cell_size
            canvas.paste(image, (x0, y0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_manifest(rows: List[Dict[str, Any]], output_path: Path, args: argparse.Namespace) -> None:
    manifest = {
        "model_path": args.model_path,
        "scheduler": args.scheduler,
        "steps": args.steps,
        "scale": args.scale,
        "poe_weights": args.poe_weights or "equal_positive_weights_from_scale",
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "grid_cell_size": args.grid_cell_size,
        "rows": [],
    }

    for idx, row in enumerate(rows, start=1):
        spec = row["spec"]
        manifest["rows"].append(
            {
                "row": idx,
                "name": spec["name"],
                "seed": row["seed"],
                "prompt_a": spec["prompt_a"],
                "prompt_b": spec["prompt_b"],
                "semantic_prompt": spec["semantic_prompt"],
                "poe_prompt": spec["poe"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> None:
    args = parse_args()
    pairs = load_pairs(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{args.filename_prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Relational compositional grid")
    print(f"Rows        : {len(pairs)}")
    print(f"Model       : {args.model_path}")
    print(f"Scheduler   : {args.scheduler}")
    print(f"Steps       : {args.steps}")
    print(f"Scale       : {args.scale}")
    print(f"Device      : {resolve_device(args.device)}")
    print(f"Output dir  : {run_dir}")
    print("=" * 88)

    pipe = None
    rows: List[Dict[str, Any]] = []

    try:
        pipe, device, dtype = load_pipeline(args)

        for row_idx, pair in enumerate(pairs):
            seed = args.seed + row_idx
            print(f"\n[ROW {row_idx + 1:02d}] {pair['name']} seed={seed}")
            print(f"  A        : {pair['prompt_a']}")
            print(f"  B        : {pair['prompt_b']}")
            print(f"  Semantic : {pair['semantic_prompt']}")

            try:
                shared_latents = prepare_shared_latents(
                    pipe=pipe,
                    height=args.height,
                    width=args.width,
                    dtype=dtype,
                    device=device,
                    seed=seed,
                )
                images = {
                    "prompt_a": run_condition(
                        pipe=pipe,
                        prompt=pair["prompt_a"],
                        steps=args.steps,
                        scale=args.scale,
                        latents=shared_latents.clone(),
                        negative_prompt=args.negative_prompt,
                    ),
                    "prompt_b": run_condition(
                        pipe=pipe,
                        prompt=pair["prompt_b"],
                        steps=args.steps,
                        scale=args.scale,
                        latents=shared_latents.clone(),
                        negative_prompt=args.negative_prompt,
                    ),
                    "semantic_prompt": run_condition(
                        pipe=pipe,
                        prompt=pair["semantic_prompt"],
                        steps=args.steps,
                        scale=args.scale,
                        latents=shared_latents.clone(),
                        negative_prompt=args.negative_prompt,
                    ),
                    "poe": run_condition(
                        pipe=pipe,
                        prompt=pair["poe"],
                        steps=args.steps,
                        scale=args.scale,
                        latents=shared_latents.clone(),
                        negative_prompt=args.negative_prompt,
                        weights=args.poe_weights or None,
                    ),
                }
            except Exception as exc:
                print(f"  [ERROR] {exc}")
                error_image = make_error_image(
                    args.width,
                    args.height,
                    f"Row {row_idx + 1:02d}\n{pair['name']}\n{str(exc)[:220]}",
                )
                images = {key: error_image.copy() for key, _ in COLUMN_SPECS}

            rows.append(
                {
                    "seed": seed,
                    "spec": pair,
                    "images": images,
                }
            )
            gc.collect()
            safe_empty_cache(device)

    finally:
        if pipe is not None:
            del pipe
        gc.collect()
        safe_empty_cache(resolve_device(args.device))

    grid_path = run_dir / "grid.png"
    manifest_path = run_dir / "manifest.json"
    save_grid(rows, grid_path, args.grid_cell_size)
    save_manifest(rows, manifest_path, args)

    print(f"\nSaved grid    : {grid_path}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
