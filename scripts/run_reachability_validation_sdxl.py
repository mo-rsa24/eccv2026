#!/usr/bin/env python3
"""Generate and audit an SDXL validation subset for semantic reachability checks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from run_taxonomy_qualitative_sdxl import _load_models, run_pair
from taxonomy_manifest import GROUP_ORDER, GROUP_LABEL_BY_KEY, PAIR_LOOKUP_BY_SLUG


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "eccv2026" / "reachability_validation" / "sdxl"
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

DEFAULT_PAIR_SLUGS = [
    "a_butterfly_a_flower_meadow",
    "a_fox_a_snow_covered_pine_forest",
    "a_dog_oil_painting_style",
    "a_lighthouse_watercolour_style",
    "a_bookcase_a_glacier",
    "a_lighthouse_a_desert_dune",
    "a_bathtub_a_streetlamp",
    "a_typewriter_a_cactus",
    "fluffy_a_stone",
    "a_fur_coat_a_goldfish",
    "a_fox_a_wolf",
    "a_raven_a_crow",
]


def _resolve_pairs(pair_slugs: list[str]) -> list[dict]:
    pairs = []
    for slug in pair_slugs:
        meta = PAIR_LOOKUP_BY_SLUG.get(slug)
        if meta is None:
            raise ValueError(f"Unknown taxonomy pair slug: {slug}")
        pairs.append(meta)
    return pairs


def _write_pair_metadata(pair_dir: Path, meta: dict, seeds: list[int]) -> None:
    payload = {
        "pair": [meta["prompt_a"], meta["prompt_b"]],
        "pair_slug": meta["pair_slug"],
        "taxonomy_group_key": meta["taxonomy_group_key"],
        "taxonomy_group_label": meta["taxonomy_group_label"],
        "is_representative_pair": meta["is_representative_pair"],
        "model_family": "sdxl",
        "model_id": DEFAULT_MODEL_ID,
        "seeds": seeds,
    }
    (pair_dir / "grid_assets.json").write_text(json.dumps(payload, indent=2))


def _rename_seed_outputs(seed_tmp_dir: Path, images_dir: Path, seed: int) -> None:
    rename_map = {
        "solo_a.png": f"sdxl_solo_a_{seed}.png",
        "solo_b.png": f"sdxl_solo_b_{seed}.png",
        "monolithic.png": f"sdxl_monolithic_{seed}.png",
        "poe.png": f"sdxl_poe_{seed}.png",
    }
    for src_name, dst_name in rename_map.items():
        src = seed_tmp_dir / src_name
        if src.exists():
            shutil.move(str(src), str(images_dir / dst_name))


def generate_subset(
    out_dir: Path,
    pairs: list[dict],
    seeds: list[int],
    steps: int,
    scale: float,
    height: int,
    width: int,
    model_id: str,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    tokenizer, tokenizer_2, text_encoder, text_encoder_2, unet, vae, scheduler = _load_models(
        model_id, device, dtype
    )

    for meta in pairs:
        pair_dir = out_dir / "pairs" / meta["pair_slug"]
        images_dir = pair_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        _write_pair_metadata(pair_dir, meta, seeds)

        for seed in seeds:
            seed_tmp_dir = pair_dir / f"_seed_{seed:03d}"
            if seed_tmp_dir.exists():
                shutil.rmtree(seed_tmp_dir)
            seed_tmp_dir.mkdir(parents=True, exist_ok=True)
            run_pair(
                meta["prompt_a"],
                meta["prompt_b"],
                seed_tmp_dir,
                tokenizer,
                tokenizer_2,
                text_encoder,
                text_encoder_2,
                unet,
                vae,
                scheduler,
                scale=scale,
                steps=steps,
                seed=seed,
                height=height,
                width=width,
                device=device,
                dtype=dtype,
            )
            _rename_seed_outputs(seed_tmp_dir, images_dir, seed)
            shutil.rmtree(seed_tmp_dir)


def run_joint_probe_eval(out_dir: Path, pairs: list[dict], batch_size: int) -> None:
    from eval_joint_probes import enrich_records, evaluate as evaluate_joint_probes, summarize_pairs

    pair_slugs = [meta["pair_slug"] for meta in pairs]
    probe_records, image_scores = evaluate_joint_probes(
        data_dir=out_dir,
        conditions=["c1", "c2", "mono", "poe"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=batch_size,
        pairs_filter=pair_slugs,
    )
    probe_records = enrich_records(probe_records)
    image_scores = enrich_records(image_scores)
    payload = {
        "model_id": "Salesforce/blip-vqa-base",
        "version": 2,
        "semantic_pass_threshold": 0.60,
        "high_confidence_pass_threshold": 0.75,
        "low_confidence_threshold": 0.55,
        "probe_records": probe_records,
        "image_scores": image_scores,
        "pair_summaries": enrich_records(summarize_pairs(image_scores)),
    }
    (out_dir / "joint_probe_scores.json").write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an SDXL subset for semantic reachability validation and optional joint-probe audit.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIR_SLUGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = _resolve_pairs(args.pairs)

    manifest = {
        "model_family": "sdxl",
        "model_id": args.model_id,
        "seeds": list(args.seeds),
        "pair_slugs": [meta["pair_slug"] for meta in pairs],
        "group_order": GROUP_ORDER,
        "group_labels": GROUP_LABEL_BY_KEY,
    }
    (out_dir / "validation_manifest.json").write_text(json.dumps(manifest, indent=2))

    if not args.skip_generation:
        generate_subset(
            out_dir=out_dir,
            pairs=pairs,
            seeds=list(args.seeds),
            steps=args.steps,
            scale=args.scale,
            height=args.height,
            width=args.width,
            model_id=args.model_id,
        )

    if not args.skip_eval:
        run_joint_probe_eval(out_dir, pairs, batch_size=args.batch_size)

    print(f"SDXL reachability validation subset ready at {out_dir}")


if __name__ == "__main__":
    main()
