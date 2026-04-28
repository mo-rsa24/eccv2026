"""
CO3 Taxonomy Qualitative Images
================================
Generates CO3 images for every taxonomy pair and writes them directly into the
existing pair directories created by ``run_taxonomy_qualitative.py``.

Run this **from the co3 conda environment**, from the CO3 repo directory:

    conda activate co3
    cd /home-mscluster/mmolefe/Playground/PhD/eccv2026/compositions/co3
    export PYTHONPATH=$(pwd)
    python ../../scripts/run_co3_taxonomy.py

    # Override output base, seed, or groups:
    python ../../scripts/run_co3_taxonomy.py --seed 42 --groups group4_collision
    python ../../scripts/run_co3_taxonomy.py \\
        --out /datasets/mmolefe/eccv2026/experiments/eccv2026/taxonomy_qualitative

Output
------
For each pair directory that already exists, writes:
    co3.png       (default SDXL output, 1024×1024)
    co3_sd14.png  (recommended SD 1.4 output for fair comparison to PoE)

The image is later resized to match the SD 1.4 cell size when
``run_taxonomy_qualitative.py --with-co3`` assembles the 5-column grid.
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CO3 imports (must run in the co3 conda env from the co3 repo root)
# ---------------------------------------------------------------------------
from composers.Co3 import Co3
from composers.config import Co3Config
from composers.utils_custom import seed_everything

DEFAULT_TAXONOMY_OUT = Path("/datasets/mmolefe/eccv2026/experiments/eccv2026/taxonomy_qualitative")
SUPPORTED_SD_VERSIONS = ("1.4", "1.5", "2.0", "2.1", "xl")

# ---------------------------------------------------------------------------
# Taxonomy — must match run_taxonomy_qualitative.py exactly
# ---------------------------------------------------------------------------
TAXONOMY_GROUPS = {
    "group1_cooccurrence": [
        ("a camel",             "a desert landscape"),
        ("a butterfly",         "a flower meadow"),
        ("a dolphin",           "an ocean wave"),
        ("a lion",              "a savanna at sunset"),
    ],
    "group2_disentangled": [
        ("a dog",               "oil painting style"),
        ("a lighthouse",        "watercolour style"),
        ("a bicycle",           "sketch style"),
    ],
    "group3_ood": [
        ("a desk lamp",         "a glacier"),
        ("a bathtub",           "a streetlamp"),
        ("a lab microscope",    "a hay bale"),
        ("a black grand piano", "a white vase"),
        ("a typewriter",        "a cactus"),
    ],
    "group4_collision": [
        ("a cat",               "a dog"),
        ("a cat",               "an owl"),
        ("a cat",               "a bear"),
    ],
}

# CO3 generation hyper-parameters — match the published run_sample_co3.sh
CO3_COMMON_DEFAULTS = dict(
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


def _default_resolution(sd_version: str) -> int:
    return 1024 if sd_version == "xl" else 512


def _default_filename(sd_version: str) -> str:
    if sd_version == "xl":
        return "co3.png"
    return f"co3_sd{sd_version.replace('.', '')}.png"


def _make_config(
    concept_a: str,
    concept_b: str,
    out_dir: Path,
    seed: int,
    sd_version: str,
    resolution: int,
) -> Co3Config:
    """Build a Co3Config for one concept pair."""
    prompt_orig = f"{concept_a} and {concept_b}"
    prompt = f"{concept_a}+{concept_b}+{prompt_orig}"
    return Co3Config(
        prompt=prompt,
        prompt_orig=prompt_orig,
        seeds=[seed],
        output_path=str(out_dir),
        output_path_all=str(out_dir),
        sd_version=sd_version,
        resolution_h=resolution,
        resolution_w=resolution,
        **CO3_COMMON_DEFAULTS,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CO3 images for all taxonomy pairs."
    )
    parser.add_argument("--seed",  type=int, default=42,
                        help="Random seed (should match run_taxonomy_qualitative.py).")
    parser.add_argument(
        "--sd-version", type=str, default="xl",
        choices=SUPPORTED_SD_VERSIONS,
        help="Stable Diffusion backbone for CO3. Use 1.4 for a fair comparison to PoE.",
    )
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="Optional square output resolution. Defaults to 1024 for SDXL and 512 otherwise.",
    )
    parser.add_argument(
        "--filename", type=str, default="",
        help="Output filename inside each pair directory. Defaults to co3.png for SDXL "
             "and co3_sdXX.png for non-XL runs.",
    )
    parser.add_argument(
        "--out", type=str, default="",
        help="Base taxonomy_qualitative directory. "
             "Defaults to /datasets/mmolefe/eccv2026/experiments/eccv2026/taxonomy_qualitative",
    )
    parser.add_argument(
        "--groups", nargs="+",
        choices=sorted(TAXONOMY_GROUPS.keys()),
        default=list(TAXONOMY_GROUPS.keys()),
        help="Subset of taxonomy groups to run.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip pairs whose target CO3 output file already exists (default: True).",
    )
    args = parser.parse_args()
    resolution = args.resolution or _default_resolution(args.sd_version)
    output_name = args.filename or _default_filename(args.sd_version)

    base_out = (
        Path(args.out) if args.out
        else DEFAULT_TAXONOMY_OUT
    )

    total_pairs = sum(len(TAXONOMY_GROUPS[g]) for g in args.groups)
    print(f"\nCO3 taxonomy qualitative run")
    print(f"  Model   : CO3 / SD {args.sd_version}")
    print(f"  Seed    : {args.seed}")
    print(f"  Size    : {resolution}x{resolution}")
    print(f"  File    : {output_name}")
    print(f"  Groups  : {', '.join(args.groups)}")
    print(f"  Pairs   : {total_pairs}")
    print(f"  Output  : {base_out}\n")

    # -----------------------------------------------------------------------
    # Load Co3 once with the first pair's config, then reuse for all pairs.
    # -----------------------------------------------------------------------
    model: Co3 | None = None
    skipped = 0
    generated = 0

    for group_name in args.groups:
        pairs = TAXONOMY_GROUPS[group_name]
        group_out = base_out / group_name

        print(f"{'='*60}")
        print(f"  {group_name}  ({len(pairs)} pairs)")
        print(f"{'='*60}")

        for concept_a, concept_b in pairs:
            pair_slug = f"{_slugify(concept_a)}__x__{_slugify(concept_b)}"
            pair_dir  = group_out / pair_slug
            co3_path  = pair_dir / output_name

            if args.skip_existing and co3_path.exists():
                print(f"  [skip]  {pair_slug}  ({output_name} already exists)")
                skipped += 1
                continue

            pair_dir.mkdir(parents=True, exist_ok=True)
            config = _make_config(
                concept_a,
                concept_b,
                pair_dir,
                args.seed,
                args.sd_version,
                resolution,
            )

            if model is None:
                print(f"  Loading CO3 / SD {args.sd_version} model (first pair) ...")
                model = Co3(config)
                print("  Model ready.\n")
            else:
                # Reuse the loaded model; swap config and recompute the
                # attributes that Co3.__init__ sets dynamically on the config
                # object (lost when we replace it with a fresh Co3Config).
                model.config = config
                model.config.latent_corrector_ts = (
                    model.scheduler.timesteps[:config.num_ts_to_correct]
                    if config.num_ts_to_correct >= 0
                    else []
                )
                model.prepare_prompts(config)
                model.prepare_embeds()

            seed_everything(args.seed)
            model.config.seed = args.seed
            model.config.output_dir = str(pair_dir)

            print(f"  Generating: [{concept_a}]  ×  [{concept_b}]")
            imgs = model.run_sampling()   # returns list[PIL.Image]
            imgs[0].save(co3_path)
            print(f"    → {co3_path}")
            generated += 1

    print(f"\nDone.  Generated: {generated}   Skipped: {skipped}")
    qualitative_cmd = "python scripts/run_taxonomy_qualitative.py --with-co3 --skip-generation"
    if output_name != "co3.png":
        qualitative_cmd += f" --co3-filename {output_name}"
    if base_out != DEFAULT_TAXONOMY_OUT:
        qualitative_cmd += f" --co3-base {base_out}"
    print("Now run:\n  conda activate compose_diff\n  " + qualitative_cmd)


if __name__ == "__main__":
    main()
