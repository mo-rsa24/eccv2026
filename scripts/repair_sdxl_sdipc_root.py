#!/usr/bin/env python3
"""Repair an SDXL SD-IPC qualitative root so paper figure commands work unchanged."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from enrich_sdxl_final_with_sdipc import (
    _load_manifest,
    _mirror_pair_dir,
    _remove_existing_sdipc_artifacts,
    _selected_pairs,
    _write_manifest,
)
from run_taxonomy_qualitative_sdxl import (
    DEFAULT_MODEL_ID,
    TaxonomyExperimentConfig,
    _load_models,
    _load_runtime_modules,
    _models_dict_from_tuple,
    run_single_experiment,
)
from sdxl_sdipc_utils import (
    SDIPC_CONDITION,
    build_sdxl_sdipc_runtime,
    decode_latents_to_tensor,
    load_shared_init_latents,
    merge_grid_assets,
    run_sdxl_precomputed_cond,
    sdipc_project_sdxl_image,
    write_canonical_grid_assets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair an SDXL SD-IPC root by restoring base flats and rebuilding grid assets.",
    )
    parser.add_argument("--source-dir", type=Path, required=True, help="Base non-SD-IPC SDXL final root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Derived SD-IPC qualitative root to repair.")
    parser.add_argument("--pairs", nargs="+", default=None, help="Optional pair filter using taxonomy or qualitative slugs.")
    parser.add_argument("--groups", nargs="+", default=None, help="Optional taxonomy group filter.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Optional seed filter.")
    parser.add_argument("--grid-seed", type=int, default=None, help="Representative grid seed.")
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "symlink"],
        default="symlink",
        help="How base source assets should be mirrored into the repaired SD-IPC root.",
    )
    parser.add_argument(
        "--rerun-missing-base",
        action="store_true",
        help="If a source pair is missing base flats, rerun that pair in the source root.",
    )
    parser.add_argument(
        "--overwrite-sdipc",
        action="store_true",
        help="Regenerate pstar_sdipc even when it already exists in the output root.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Torch device to use.")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU id to pin this worker to.")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for SDXL inference.",
    )
    parser.add_argument(
        "--keep-clip-on-gpu",
        action="store_true",
        help="Keep CLIP projection models on GPU between records.",
    )
    parser.add_argument("--guidance-scale", type=float, default=None, help="Override guidance scale.")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Override number of inference steps.")
    parser.add_argument("--model-id", default=None, help="Override model ID.")
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device != "auto" and args.gpu_id is not None:
        raise ValueError("Use either --device or --gpu-id, not both.")
    if args.gpu_id is not None:
        return torch.device(f"cuda:{args.gpu_id}")
    if args.device != "auto":
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_base_pair_ready(
    *,
    source_pair_dir: Path,
    group_key: str,
    prompt_a: str,
    prompt_b: str,
    seed: int,
    model_id: str,
    guidance_scale: float,
    num_inference_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    rerun_missing_base: bool,
    models_tuple: tuple[Any, Any, Any, Any, Any, Any, Any] | None,
) -> tuple[tuple[Any, Any, Any, Any, Any, Any, Any] | None, bool]:
    try:
        write_canonical_grid_assets(pair_dir=source_pair_dir, require_base_flats=True)
        return models_tuple, False
    except FileNotFoundError:
        if not rerun_missing_base:
            raise

    if models_tuple is None:
        models_tuple = _load_models(model_id, device, dtype)

    cfg = TaxonomyExperimentConfig(
        group=group_key,
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        model_id=model_id,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=int(seed),
    )
    source_pair_dir.mkdir(parents=True, exist_ok=True)
    run_single_experiment(cfg, source_pair_dir, models_tuple, device, dtype)
    write_canonical_grid_assets(pair_dir=source_pair_dir, require_base_flats=True)
    return models_tuple, True


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(source_dir)
    seeds = list(args.seeds) if args.seeds is not None else list(manifest.get("seeds", []))
    if not seeds:
        raise ValueError("No seeds resolved for repair.")
    valid_seeds = set(manifest.get("seeds", []))
    bad_seeds = [seed for seed in seeds if seed not in valid_seeds]
    if bad_seeds:
        raise ValueError(f"Requested seeds are not present in the source manifest: {bad_seeds}")

    pair_rows = _selected_pairs(source_dir, manifest, seeds, args.pairs, args.groups)
    if not pair_rows:
        raise ValueError("No pairs selected for repair.")

    grid_seed = int(args.grid_seed) if args.grid_seed is not None else int(manifest.get("grid_seed", seeds[0]))
    if grid_seed not in valid_seeds:
        raise ValueError(f"FATAL: --grid-seed {grid_seed} does not appear in source manifest seeds.")
    if grid_seed not in seeds:
        seeds.append(grid_seed)

    device = _resolve_device(args)
    dtype = _torch_dtype(args.dtype) if device.type == "cuda" else torch.float32
    model_id = args.model_id or str(manifest.get("model_id") or DEFAULT_MODEL_ID)
    num_inference_steps = int(
        args.num_inference_steps
        if args.num_inference_steps is not None
        else manifest.get("num_inference_steps", 50)
    )
    guidance_scale = float(
        args.guidance_scale
        if args.guidance_scale is not None
        else manifest.get("guidance_scale", 7.5)
    )
    projection = str(manifest.get("projection") or "mds")
    height = int(manifest.get("height", 1024))
    width = int(manifest.get("width", 1024))

    runtime_modules = _load_runtime_modules()
    models_tuple: tuple[Any, Any, Any, Any, Any, Any, Any] | None = None
    models: dict[str, Any] | None = None
    sdipc_runtime: dict[str, Any] | None = None
    euler_scheduler = None
    euler_sigma = None

    repaired_source = 0
    repaired_output = 0
    generated_sdipc = 0

    all_records = [(row, int(seed)) for row in pair_rows for seed in seeds]
    for row, seed in all_records:
        group_key = str(row["taxonomy_group_key"])
        prompt_a, prompt_b = row["pair"]
        pair_slug = str(row["qualitative_pair_slug"])
        source_pair_dir = source_dir / f"seed_{seed}" / group_key / pair_slug
        output_pair_dir = output_dir / f"seed_{seed}" / group_key / pair_slug

        models_tuple, did_rerun = _ensure_base_pair_ready(
            source_pair_dir=source_pair_dir,
            group_key=group_key,
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            seed=seed,
            model_id=model_id,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            device=device,
            dtype=dtype,
            rerun_missing_base=args.rerun_missing_base,
            models_tuple=models_tuple,
        )
        repaired_source += int(did_rerun)

        _mirror_pair_dir(source_pair_dir, output_pair_dir, copy_mode=args.copy_mode)

        need_sdipc = args.overwrite_sdipc
        if not need_sdipc:
            try:
                payload = write_canonical_grid_assets(pair_dir=output_pair_dir, require_base_flats=True)
                need_sdipc = SDIPC_CONDITION not in (payload.get("trajectory_flat_paths") or {})
            except FileNotFoundError:
                need_sdipc = True

        if need_sdipc:
            if models_tuple is None:
                models_tuple = _load_models(model_id, device, dtype)
            if models is None:
                models = _models_dict_from_tuple(models_tuple)
            if sdipc_runtime is None:
                sdipc_runtime = build_sdxl_sdipc_runtime(
                    models=models,
                    model_id=model_id,
                    device=device,
                    dtype=dtype,
                    keep_clip_on_device=bool(args.keep_clip_on_gpu),
                )
                euler_scheduler = runtime_modules["EulerDiscreteScheduler"].from_pretrained(model_id, subfolder="scheduler")
                euler_scheduler.set_timesteps(num_inference_steps)
                euler_sigma = float(euler_scheduler.init_noise_sigma)

            _remove_existing_sdipc_artifacts(output_pair_dir)
            poe_path = source_pair_dir / "poe.png"
            if not poe_path.exists():
                raise FileNotFoundError(f"Missing PoE image required for SD-IPC repair: {poe_path}")

            init_latents = load_shared_init_latents(
                pair_dir=source_pair_dir,
                euler_init_noise_sigma=euler_sigma,
                device=device,
                dtype=dtype,
            )
            prompt_embeds, pooled = sdipc_project_sdxl_image(
                Image.open(poe_path).convert("RGB"),
                runtime=sdipc_runtime,
                device=device,
                dtype=dtype,
            )
            latents_sdipc, tracker_sdipc = run_sdxl_precomputed_cond(
                init_latents=init_latents,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled,
                models=models,
                runtime=sdipc_runtime,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=dtype,
                model_id=model_id,
                euler_init_noise_sigma=euler_sigma,
                height=height,
                width=width,
            )
            image_sdipc = decode_latents_to_tensor(models["vae"], latents_sdipc).cpu()
            merge_grid_assets(
                pair_dir=output_pair_dir,
                seed=int(seed),
                decoded_images={"pstar_sdipc": image_sdipc},
                trackers={"pstar_sdipc": tracker_sdipc},
                projection_method=projection,
                condition_labels={"pstar_sdipc": "PoE p*"},
                source_prompts={"pstar_sdipc": "SD-IPC closed-form rerun from PoE"},
            )
            generated_sdipc += 1

        write_canonical_grid_assets(pair_dir=output_pair_dir, require_base_flats=True)
        repaired_output += 1
        print(f"  repaired {group_key}/{pair_slug}/seed_{seed}")

    _write_manifest(
        source_dir=source_dir,
        output_dir=output_dir,
        source_manifest=manifest,
        pair_rows=pair_rows,
        seeds=seeds,
        grid_seed=grid_seed,
        projection=projection,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        copy_mode=args.copy_mode,
    )

    print(
        "\nCompleted SDXL SD-IPC root repair: "
        f"{repaired_output} pair-seed records rebuilt, "
        f"{generated_sdipc} SD-IPC reruns generated, "
        f"{repaired_source} source pair-seed records rerun."
    )
    print(f"Repaired root: {output_dir}")


if __name__ == "__main__":
    main()
