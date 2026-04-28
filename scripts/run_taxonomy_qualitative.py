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
    conda activate co3
    python scripts/run_taxonomy_qualitative.py

    # Override seed, steps, or guidance scale:
    python scripts/run_taxonomy_qualitative.py --seed 43 --steps 50 --scale 7.5

    # Run a single group:
    python scripts/run_taxonomy_qualitative.py --groups group4_collision

    # Generate CO3 in-process (5th column, no separate step needed):
    python scripts/run_taxonomy_qualitative.py --run-co3

    # Assemble 5-column grids from pre-generated CO3 files on disk:
    python scripts/run_taxonomy_qualitative.py --with-co3 --skip-generation
    python scripts/run_taxonomy_qualitative.py --with-co3 --skip-generation \
        --co3-base /datasets/mmolefe/eccv2026/experiments/eccv2026/taxonomy_qualitative

    # Load SD 1.4 CO3 outputs without overwriting the existing SDXL co3.png:
    python scripts/run_taxonomy_qualitative.py --with-co3 --skip-generation \
        --co3-filename co3_sd14.png

Output structure
----------------
experiments/eccv2026/taxonomy_qualitative/
    group1_cooccurrence/
        camel__x__desert_landscape/
            solo_a.png
            solo_b.png
            monolithic.png
            poe.png
            co3.png            <- optional SDXL CO3 output
            co3_sd14.png       <- optional SD 1.4 CO3 output
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
DEFAULT_OUT_DIR = PROJECT_ROOT / "experiments" / "eccv2026" / "taxonomy_qualitative"
LEGACY_CO3_OUT_DIR = Path("/datasets/mmolefe/eccv2026/experiments/eccv2026/taxonomy_qualitative")
COMPOSABLE_REPO = (
    PROJECT_ROOT
    / "compositions"
    / "Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch"
)
CO3_REPO = PROJECT_ROOT / "compositions" / "co3"
sys.path.insert(0, str(COMPOSABLE_REPO))
sys.path.insert(0, str(CO3_REPO))

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
    # Group 3 sub-regimes (for the figure_group3 qualitative figure)
    "group3a_missing_support": [
        ("a penguin",           "a desert landscape"),
        ("a snowman",           "a tropical beach"),
        ("a cactus",            "the Arctic tundra"),
    ],
    "group3b_entanglement": [
        ("a red",               "a cube"),
        ("small",               "an elephant"),
        ("striped",             "a sphere"),
    ],
    "group3c_interference": [
        ("a wooden chair",      "metallic texture"),
        ("a transparent glass", "a dog"),
        ("fluffy",              "a stone"),
    ],
}

# Condition order used for all panels / grids
CONDITIONS = ["solo_a", "solo_b", "monolithic", "poe"]
# Extended order when CO3 images are available
CONDITIONS_WITH_CO3 = CONDITIONS + ["co3"]


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


def _pil_to_tensor(img: Image.Image, size: int | None = None) -> th.Tensor:
    """PIL Image → CHW float tensor in [0, 1].

    If *size* is given the image is resized to (size × size) first, which is
    used to bring high-resolution CO3 / SDXL outputs down to the same cell
    size as the SD 1.4 images before stacking.
    """
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    return th.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def _co3_placeholder(cell_size: int = 512) -> th.Tensor:
    """Return a gray CHW tensor with 'CO3 N/A' text when the CO3 image is absent."""
    img = Image.new("RGB", (cell_size, cell_size), color=(180, 180, 180))
    return _pil_to_tensor(img)


# ---------------------------------------------------------------------------
# CO3 in-process generation
# ---------------------------------------------------------------------------

CO3_DEFAULTS = dict(
    guidance_scale=0.8,
    n_timesteps=50,
    num_ts_to_correct=6,
    num_latent_corrector_steps=5,
    num_resampling_steps=3,
    corrector_algo="co3-hybrid",
    modulate_comp_weights=True,
    beta=0.9,
    lmda=0.8,
    negative_prompt="",
)
CO3_SD_VERSION = "xl"
CO3_RESOLUTION = 1024


def _load_co3_model(concept_a: str, concept_b: str, out_dir: Path, seed: int):
    """Lazily import Co3 and return a loaded model for the given pair."""
    from composers.Co3 import Co3
    from composers.config import Co3Config

    prompt_orig = f"{concept_a} and {concept_b}"
    prompt = f"{concept_a}+{concept_b}+{prompt_orig}"
    config = Co3Config(
        prompt=prompt,
        prompt_orig=prompt_orig,
        seeds=[seed],
        output_path=str(out_dir),
        output_path_all=str(out_dir),
        sd_version=CO3_SD_VERSION,
        resolution_h=CO3_RESOLUTION,
        resolution_w=CO3_RESOLUTION,
        **CO3_DEFAULTS,
    )
    return Co3(config), config


def _run_co3_for_pair(
    co3_model,
    concept_a: str,
    concept_b: str,
    out_dir: Path,
    seed: int,
) -> Image.Image:
    """Run CO3 for one pair, reusing an already-loaded model."""
    from composers.config import Co3Config
    from composers.utils_custom import seed_everything

    prompt_orig = f"{concept_a} and {concept_b}"
    prompt = f"{concept_a}+{concept_b}+{prompt_orig}"
    config = Co3Config(
        prompt=prompt,
        prompt_orig=prompt_orig,
        seeds=[seed],
        output_path=str(out_dir),
        output_path_all=str(out_dir),
        sd_version=CO3_SD_VERSION,
        resolution_h=CO3_RESOLUTION,
        resolution_w=CO3_RESOLUTION,
        **CO3_DEFAULTS,
    )
    co3_model.config = config
    co3_model.config.latent_corrector_ts = (
        co3_model.scheduler.timesteps[: config.num_ts_to_correct]
        if config.num_ts_to_correct >= 0
        else []
    )
    co3_model.prepare_prompts(config)
    co3_model.prepare_embeds()
    seed_everything(seed)
    co3_model.config.seed = seed
    co3_model.config.output_dir = str(out_dir)
    imgs = co3_model.run_sampling()
    return imgs[0]


def _resolve_co3_png(
    out_dir: Path,
    base_out: Path,
    co3_base: Path | None,
    co3_filename: str,
) -> Path | None:
    """Resolve the CO3 image for one pair, with backward-compatible fallbacks."""
    candidates = [out_dir / co3_filename]
    relative_pair_dir = out_dir.relative_to(base_out)

    if co3_base is not None:
        candidates.append(co3_base / relative_pair_dir / co3_filename)
    elif base_out != LEGACY_CO3_OUT_DIR:
        # Older runs wrote CO3 outputs into /datasets while SD 1.4 outputs lived
        # in the repo-local experiments tree. Keep that layout working.
        candidates.append(LEGACY_CO3_OUT_DIR / relative_pair_dir / co3_filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


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
    with_co3: bool = False,
    skip_generation: bool = False,
    base_out: Path | None = None,
    co3_base: Path | None = None,
    co3_filename: str = "co3.png",
    co3_model=None,
) -> th.Tensor:
    """Generate all conditions for one pair.

    Returns a (4, C, H, W) or (5, C, H, W) stacked tensor depending on
    *with_co3*.  Also saves individual PNGs and a panel image.

    Args:
        with_co3: When True, appends the pre-generated CO3 image as a 5th
            column. If it is absent a gray placeholder is used.
        skip_generation: When True, skip SD 1.4 generation and only load
            existing PNGs.  Useful when regenerating grids without re-running
            the diffusion model (e.g. after adding a CO3 image).
        base_out: Base directory for the current qualitative run.
        co3_base: Optional alternate base directory where CO3 image files
            were generated.
        co3_filename: Per-pair CO3 image filename, e.g. ``co3.png`` or
            ``co3_sd14.png``.
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

    # Infer cell size from an existing image so CO3 outputs are resized to match.
    cell_size: int | None = None
    existing = next(
        (out_dir / f"{c}.png" for c in CONDITIONS if (out_dir / f"{c}.png").exists()),
        None,
    )
    if existing is not None:
        w, h = Image.open(existing).size
        cell_size = w  # square images; SD 1.4 → 512

    tensors: list[th.Tensor] = []
    for cond in CONDITIONS:
        png = out_dir / f"{cond}.png"
        if png.exists() and (skip_generation or pipe is None):
            img = Image.open(png).convert("RGB")
        elif pipe is None:
            raise FileNotFoundError(
                f"{png} not found and pipe=None (--skip-generation mode). "
                "Run without --skip-generation to generate missing images first."
            )
        else:
            prompt, weights = condition_configs[cond]
            img = _generate(pipe, prompt, weights, scale, steps, seed, device_str)
            img.save(png)
        tensors.append(_pil_to_tensor(img))

    # Determine cell size from the first generated/loaded image.
    if cell_size is None and tensors:
        cell_size = tensors[0].shape[-1]

    if with_co3:
        co3_png_out = out_dir / "co3.png"
        if co3_model is not None:
            # Generate CO3 in-process and save to co3.png
            co3_img = _run_co3_for_pair(co3_model, concept_a, concept_b, out_dir, seed)
            co3_img.save(co3_png_out)
            tensors.append(_pil_to_tensor(co3_img, size=cell_size))
        else:
            # Fall back to loading from disk
            if base_out is None:
                raise ValueError("base_out must be provided when with_co3=True and co3_model is None")
            co3_png = _resolve_co3_png(out_dir, base_out, co3_base, co3_filename)
            if co3_png is not None:
                co3_img = Image.open(co3_png).convert("RGB")
                tensors.append(_pil_to_tensor(co3_img, size=cell_size))
            else:
                print(f"    [warn] {co3_filename} missing for {out_dir.name} — using placeholder")
                tensors.append(_co3_placeholder(cell_size or 512))

    n_cols = len(tensors)
    row = th.stack(tensors)                           # (n_cols, C, H, W)
    panel = tvu.make_grid(row, nrow=n_cols, padding=2)
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
    parser.add_argument(
        "--seeds", type=int, default=None,
        help="Alias for --seed. For compatibility with older launcher habits.",
    )
    parser.add_argument("--steps",  type=int,   default=50)
    parser.add_argument("--scale",  type=float, default=7.5,
                        help="CFG guidance scale (applied to all conditions).")
    parser.add_argument("--model",  type=str,   default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--out",    type=str,   default="",
                        help="Base output directory. Defaults to "
                             "experiments/eccv2026/taxonomy_qualitative/")
    parser.add_argument(
        "--output-dir", type=str, default="",
        help="Alias for --out.",
    )
    parser.add_argument(
        "--co3-base", type=str, default="",
        help="Optional alternate base directory containing per-pair CO3 image "
             "files. If unset, the script first checks --out and then falls "
             "back to the legacy /datasets CO3 location.",
    )
    parser.add_argument(
        "--co3-filename", type=str, default="co3.png",
        help="Per-pair CO3 image filename to load when --with-co3 is used, "
             "for example co3.png or co3_sd14.png.",
    )
    parser.add_argument(
        "--groups", nargs="+",
        choices=sorted(TAXONOMY_GROUPS.keys()),
        default=list(TAXONOMY_GROUPS.keys()),
        help="Subset of taxonomy groups to run. Defaults to all four groups.",
    )
    parser.add_argument(
        "--with-co3", action="store_true", default=False,
        help="Include a 5th CO3 column by loading pre-generated CO3 image files. "
             "Run run_co3_taxonomy.py first to generate them.",
    )
    parser.add_argument(
        "--run-co3", action="store_true", default=False,
        help="Generate CO3 in-process (SDXL) during this run instead of loading from disk. "
             "Requires compositions/co3 to be on the path. Implies --with-co3.",
    )
    parser.add_argument(
        "--skip-generation", action="store_true", default=False,
        help="Skip SD 1.4 image generation and only load existing PNGs. "
             "Useful for quickly regenerating grids after adding a CO3 image.",
    )
    args = parser.parse_args()
    if args.seeds is not None:
        args.seed = int(args.seeds)
    if args.output_dir and args.out and args.output_dir != args.out:
        raise ValueError(f"--out ({args.out}) and --output-dir ({args.output_dir}) disagree")
    if args.output_dir:
        args.out = args.output_dir
    if args.run_co3:
        args.with_co3 = True

    base_out = (
        Path(args.out) if args.out
        else DEFAULT_OUT_DIR
    )
    co3_base = Path(args.co3_base) if args.co3_base else None
    base_out.mkdir(parents=True, exist_ok=True)

    has_cuda = th.cuda.is_available()
    device_str = "cuda" if has_cuda else "cpu"
    device = th.device(device_str)

    n_cols = 5 if args.with_co3 else 4
    total_pairs = sum(len(TAXONOMY_GROUPS[g]) for g in args.groups)
    print(f"\nPhase 1 taxonomy qualitative run")
    print(f"  Model   : {args.model}")
    print(f"  Seed    : {args.seed}   Steps: {args.steps}   Scale: {args.scale}")
    print(f"  Groups  : {', '.join(args.groups)}")
    print(f"  Columns : {n_cols}  ({'A | B | Semantic | PoE | CO3' if args.with_co3 else 'A | B | Semantic | PoE'})")
    print(f"  Pairs   : {total_pairs}  ×  {n_cols} conditions = {total_pairs * n_cols} images")
    print(f"  Output  : {base_out}\n")
    if args.with_co3:
        print(f"  CO3 file: {args.co3_filename}")
        if co3_base is not None:
            print(f"  CO3 base: {co3_base}")
        elif base_out != LEGACY_CO3_OUT_DIR:
            print(f"  CO3 base: {base_out} (fallback: {LEGACY_CO3_OUT_DIR})\n")

    if args.skip_generation:
        print("Skip-generation mode: loading existing PNGs only (no SD 1.4 inference).\n")
        pipe = None
    else:
        print(f"Loading pipeline ...")
        pipe = ComposableStableDiffusionPipeline.from_pretrained(args.model).to(device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.safety_checker = None
        print("Pipeline ready.\n")

    co3_model = None
    if args.run_co3:
        print(f"Loading CO3 model (SD {CO3_SD_VERSION}) ...")
        first_group = args.groups[0]
        first_pair = TAXONOMY_GROUPS[first_group][0]
        first_out = base_out / first_group / f"{_slugify(first_pair[0])}__x__{_slugify(first_pair[1])}"
        co3_model, _ = _load_co3_model(first_pair[0], first_pair[1], first_out, args.seed)
        print(f"CO3 model ready (SD {CO3_SD_VERSION}, {CO3_RESOLUTION}px).\n")

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
                with_co3=args.with_co3,
                skip_generation=args.skip_generation,
                base_out=base_out,
                co3_base=co3_base,
                co3_filename=args.co3_filename,
                co3_model=co3_model,
            )
            group_rows.append(row)      # each: (n_cols, C, H, W)

        # Group-level grid — each pair occupies one row of n_cols images
        group_tensor = th.cat(group_rows, dim=0)        # (n_cols*n_pairs, C, H, W)
        group_grid = tvu.make_grid(group_tensor, nrow=n_cols, padding=2)
        tvu.save_image(group_grid, group_out / "decoded_images_grid.png")
        print(f"  Group grid → {group_out / 'decoded_images_grid.png'}\n")
        all_rows.append(group_tensor)

    # Full proposal figure — all groups concatenated
    full_tensor = th.cat(all_rows, dim=0)
    full_grid = tvu.make_grid(full_tensor, nrow=n_cols, padding=2)
    full_grid_path = base_out / "decoded_images_grid.png"
    tvu.save_image(full_grid, full_grid_path)

    print(f"\nAll groups complete.")
    print(f"Full proposal figure → {full_grid_path}")
    print(f"Copy to paper assets :")
    print(f"  cp {full_grid_path} proposal/proposal_stage_3/chapters/research_method/media/decoded_images_grid.png")


if __name__ == "__main__":
    main()
