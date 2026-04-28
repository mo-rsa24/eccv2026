"""Augment a frozen multi-seed SDXL qualitative root with PoE SD-IPC reruns."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_taxonomy_qualitative_sdxl import (
    DEFAULT_MODEL_ID,
    _load_models,
    _load_runtime_modules,
    _models_dict_from_tuple,
    _resolve_pair_token,
    format_pair_slug,
    normalize_group_key,
)
from sdxl_sdipc_utils import (
    build_sdxl_sdipc_runtime,
    decode_latents_to_tensor,
    load_shared_init_latents,
    merge_grid_assets,
    run_sdxl_precomputed_cond,
    sdipc_project_sdxl_image,
    write_canonical_grid_assets,
)
try:
    from taxonomy_manifest import get_pair_taxonomy_from_slug
except ImportError:
    from scripts.taxonomy_manifest import get_pair_taxonomy_from_slug


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "eccv2026"
    / "sdxl_final"
    / "sdxl_six_group_seed42_steps50_cfg7p5_frozen_sdipc"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich a frozen SDXL taxonomy root with PoE SD-IPC reruns.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Frozen SDXL qualitative root containing sdxl_qualitative_run_manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Derived output root that will receive the SD-IPC-enriched assets.",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Optional pair filter using taxonomy or qualitative slugs.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Optional taxonomy group filter (e.g. group1, group2_factorization).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional seed filter. Defaults to all seeds in the source manifest.",
    )
    parser.add_argument(
        "--grid-seed",
        type=int,
        default=None,
        help="Representative grid seed. Defaults to the source manifest grid_seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated SD-IPC files and merged grid metadata if they already exist.",
    )
    parser.add_argument(
        "--projection",
        choices=["mds", "pca"],
        default=None,
        help="Projection method for the merged trajectory export. Defaults to the source manifest setting.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "symlink"],
        default="symlink",
        help="How unchanged source assets should be mirrored into the derived root.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device to run enrichment on (auto = cuda if available).",
    )
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU id to pin this worker to.")
    parser.add_argument("--num-workers", type=int, default=1, help="Shard records across workers.")
    parser.add_argument("--worker-index", type=int, default=0, help="Zero-based worker shard index.")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for SDXL inference.",
    )
    parser.add_argument(
        "--keep-clip-on-gpu",
        action="store_true",
        help="Keep CLIP projection models on GPU between records (faster, uses more VRAM).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Override guidance scale. Defaults to the source manifest value.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override number of inference steps. Defaults to the source manifest value.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override model ID. Defaults to the source manifest value.",
    )
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _load_manifest(source_dir: Path) -> dict[str, Any]:
    manifest_path = source_dir / "sdxl_qualitative_run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    if payload.get("model_family") != "sdxl":
        raise ValueError(f"Expected an SDXL manifest, got {payload.get('model_family')!r}")
    if payload.get("layout") != "multi_seed":
        raise ValueError(
            "SDXL SD-IPC enrichment currently expects the multi-seed final layout."
        )
    return payload


def _selected_pairs(
    source_dir: Path,
    manifest: dict[str, Any],
    seeds: list[int],
    pair_filters: list[str] | None,
    group_filters: list[str] | None,
) -> list[dict[str, Any]]:
    manifest_pairs = list(manifest.get("selected_pairs", []) or [])
    manifest_by_key = {
        (str(row.get("taxonomy_group_key")), str(row.get("qualitative_pair_slug"))): row
        for row in manifest_pairs
    }

    probe_roots = [source_dir / f"seed_{seed}" for seed in seeds] + [source_dir]
    discovered_pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for probe_root in probe_roots:
        if not probe_root.exists():
            continue
        for group_dir in sorted(path for path in probe_root.iterdir() if path.is_dir()):
            group_key = normalize_group_key(group_dir.name)
            for pair_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
                pair_slug = pair_dir.name
                key = (group_key, pair_slug)
                if key in seen:
                    continue
                seen.add(key)

                row = dict(manifest_by_key.get(key, {}))
                row["taxonomy_group_key"] = group_key
                row["qualitative_pair_slug"] = pair_slug

                meta = get_pair_taxonomy_from_slug(pair_slug)
                if meta is not None:
                    row.setdefault("pair", [str(meta["prompt_a"]), str(meta["prompt_b"])])
                elif "pair" not in row:
                    if "__x__" not in pair_slug:
                        raise ValueError(f"Cannot infer pair prompts from slug: {pair_slug}")
                    prompt_a_slug, prompt_b_slug = pair_slug.split("__x__", 1)
                    row["pair"] = [
                        prompt_a_slug.replace("_", " "),
                        prompt_b_slug.replace("_", " "),
                    ]
                discovered_pairs.append(row)
        if discovered_pairs:
            break

    selected_pairs = discovered_pairs or manifest_pairs
    if group_filters:
        wanted_groups = {normalize_group_key(group) for group in group_filters}
        present_groups = {str(row.get("taxonomy_group_key")) for row in selected_pairs}
        missing_groups = sorted(group for group in wanted_groups if group not in present_groups)
        if missing_groups:
            raise ValueError(f"Unknown or unselected group filter(s): {missing_groups}")
        selected_pairs = [row for row in selected_pairs if str(row.get("taxonomy_group_key")) in wanted_groups]

    if not pair_filters:
        return selected_pairs

    wanted: set[str] = set()
    for token in pair_filters:
        group, prompt_a, prompt_b = _resolve_pair_token(token)
        slug = format_pair_slug(prompt_a, prompt_b)
        wanted.add(f"{group}/{slug}")
        wanted.add(slug)

    filtered = []
    for row in selected_pairs:
        group_key = str(row["taxonomy_group_key"])
        slug = str(row["qualitative_pair_slug"])
        if slug in wanted or f"{group_key}/{slug}" in wanted:
            filtered.append(row)
    missing = sorted(
        token for token in pair_filters
        if not any(
            token == row["qualitative_pair_slug"]
            or token == f"{row['taxonomy_group_key']}/{row['qualitative_pair_slug']}"
            or token == format_pair_slug(*row["pair"])
            for row in filtered
        )
    )
    if missing:
        raise ValueError(f"Unknown or unselected pair filter(s): {missing}")
    return filtered


def _safe_mirror_file(src: Path, dst: Path, *, copy_mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if copy_mode == "copy":
            shutil.copy2(src, dst)
        else:
            os.symlink(src.resolve(), dst)
    except FileExistsError:
        return


def _mirror_pair_dir(
    src_pair_dir: Path,
    dst_pair_dir: Path,
    *,
    copy_mode: str,
) -> None:
    dst_pair_dir.mkdir(parents=True, exist_ok=True)

    for src in src_pair_dir.iterdir():
        if src.name == "grid_assets.json":
            continue
        if src.is_dir():
            if src.name == "grid_assets":
                dst_assets_dir = dst_pair_dir / "grid_assets"
                dst_assets_dir.mkdir(parents=True, exist_ok=True)
                for asset in src.rglob("*"):
                    if asset.is_dir():
                        continue
                    if asset.name.startswith("pstar_sdipc"):
                        continue
                    if asset.name.startswith("trajectory_flat_pstar_sdipc"):
                        continue
                    rel = asset.relative_to(src)
                    _safe_mirror_file(asset, dst_assets_dir / rel, copy_mode=copy_mode)
                continue

            for child in src.rglob("*"):
                if child.is_dir():
                    continue
                rel = child.relative_to(src_pair_dir)
                _safe_mirror_file(child, dst_pair_dir / rel, copy_mode=copy_mode)
            continue

        rel = src.relative_to(src_pair_dir)
        _safe_mirror_file(src, dst_pair_dir / rel, copy_mode=copy_mode)


def _acquire_manifest_lock(output_dir: Path, *, timeout_s: float = 600.0) -> Path | None:
    lock_path = output_dir / ".sdipc_enrich.lock"
    manifest_path = output_dir / "sdxl_qualitative_run_manifest.json"
    deadline = time.time() + timeout_s

    output_dir.mkdir(parents=True, exist_ok=True)

    while time.time() < deadline:
        if manifest_path.exists():
            return None
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return lock_path
        except FileExistsError:
            time.sleep(0.5)
            continue

    raise TimeoutError(
        f"Timed out waiting for {manifest_path} (lock held at {lock_path})."
    )


def _write_manifest(
    source_dir: Path,
    output_dir: Path,
    *,
    source_manifest: dict[str, Any],
    pair_rows: list[dict[str, Any]],
    seeds: list[int],
    grid_seed: int,
    projection: str,
    guidance_scale: float,
    num_inference_steps: int,
    copy_mode: str,
) -> None:
    payload = dict(source_manifest)
    payload["output_dir"] = str(output_dir)
    payload["projection"] = projection
    payload["guidance_scale"] = guidance_scale
    payload["num_inference_steps"] = num_inference_steps
    payload["sdipc_enrichment"] = {
        "created_at": datetime.now().isoformat(),
        "source_dir": str(source_dir),
        "copy_mode": copy_mode,
        "grid_seed": int(grid_seed),
        "seeds": [int(seed) for seed in seeds],
        "pair_count": len(pair_rows),
        "pairs": [
            {
                "taxonomy_group_key": row["taxonomy_group_key"],
                "qualitative_pair_slug": row["qualitative_pair_slug"],
            }
            for row in pair_rows
        ],
        "generated_condition": "pstar_sdipc",
    }
    out_path = output_dir / "sdxl_qualitative_run_manifest.json"
    tmp_path = output_dir / f".sdxl_qualitative_run_manifest.{os.getpid()}.tmp"
    tmp_path.write_text(json.dumps(payload, indent=2))
    os.replace(tmp_path, out_path)


def _remove_existing_sdipc_artifacts(pair_dir: Path) -> None:
    candidates = [
        pair_dir / "grid_assets" / "pstar_sdipc.png",
        pair_dir / "grid_assets" / "trajectory_flat_pstar_sdipc.npy",
        pair_dir / "pstar_sdipc.png",
        pair_dir / "pstar_sdipc.npy",
    ]
    for path in candidates:
        if path.exists() or path.is_symlink():
            path.unlink()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == source_dir:
        raise ValueError("--output-dir must be different from --source-dir.")

    manifest = _load_manifest(source_dir)
    seeds = list(args.seeds) if args.seeds is not None else list(manifest.get("seeds", []))
    if not seeds:
        raise ValueError("No seeds resolved for enrichment.")
    valid_seeds = set(manifest.get("seeds", []))
    bad_seeds = [seed for seed in seeds if seed not in valid_seeds]
    if bad_seeds:
        raise ValueError(f"Requested seeds are not present in the source manifest: {bad_seeds}")

    pair_rows = _selected_pairs(source_dir, manifest, seeds, args.pairs, args.groups)
    if not pair_rows:
        raise ValueError("No pairs selected for enrichment.")

    grid_seed = int(args.grid_seed) if args.grid_seed is not None else int(manifest.get("grid_seed", seeds[0]))
    if grid_seed not in valid_seeds:
        raise ValueError(f"FATAL: --grid-seed {grid_seed} does not appear in source manifest seeds.")
    if grid_seed not in seeds:
        seeds.append(grid_seed)

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
    projection = args.projection or str(manifest.get("projection", "mds"))
    height = int(manifest.get("height", 1024))
    width = int(manifest.get("width", 1024))

    out_manifest_path = output_dir / "sdxl_qualitative_run_manifest.json"
    if args.overwrite:
        try:
            out_manifest_path.unlink()
        except FileNotFoundError:
            pass

    lock_path = None
    if not out_manifest_path.exists():
        lock_path = _acquire_manifest_lock(output_dir)
        if lock_path is not None and not out_manifest_path.exists():
            _write_manifest(
                source_dir,
                output_dir,
                source_manifest=manifest,
                pair_rows=pair_rows,
                seeds=seeds,
                grid_seed=grid_seed,
                projection=projection,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                copy_mode=args.copy_mode,
            )
    if lock_path is not None:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    if args.device != "auto" and args.gpu_id is not None:
        raise ValueError("Use either --device or --gpu-id, not both.")
    if args.gpu_id is not None:
        device_str = f"cuda:{args.gpu_id}"
    elif args.device != "auto":
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    dtype = _torch_dtype(args.dtype)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    models_tuple = _load_models(model_id=model_id, device=device, dtype=dtype)
    models = _models_dict_from_tuple(models_tuple)
    euler_scheduler = models["scheduler"]
    euler_sigma = float(getattr(euler_scheduler, "init_noise_sigma", 1.0))
    sdipc_runtime = build_sdxl_sdipc_runtime(
        models=models,
        model_id=model_id,
        device=device,
        dtype=dtype,
        keep_clip_on_device=bool(args.keep_clip_on_gpu),
    )

    runtime_modules = _load_runtime_modules()
    get_latents = runtime_modules["get_latents"]

    processed = 0
    skipped = 0

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.worker_index < 0 or args.worker_index >= args.num_workers:
        raise ValueError("--worker-index must satisfy 0 <= worker-index < num-workers")

    # Deterministic record ordering for sharding.
    pair_rows_sorted = sorted(
        pair_rows,
        key=lambda r: (str(r["taxonomy_group_key"]), str(r["qualitative_pair_slug"])),
    )
    all_records: list[tuple[dict[str, Any], int]] = []
    for row in pair_rows_sorted:
        for seed in seeds:
            all_records.append((row, int(seed)))
    shard_records = [
        rec for idx, rec in enumerate(all_records) if idx % args.num_workers == args.worker_index
    ]

    for row, seed in shard_records:
        group_key = str(row["taxonomy_group_key"])
        prompt_a, prompt_b = row["pair"]
        pair_slug = str(row["qualitative_pair_slug"])
        src_pair_dir = source_dir / f"seed_{seed}" / group_key / pair_slug
        dst_pair_dir = output_dir / f"seed_{seed}" / group_key / pair_slug
        if not src_pair_dir.exists():
            raise FileNotFoundError(f"Missing source pair directory: {src_pair_dir}")
        try:
            write_canonical_grid_assets(pair_dir=src_pair_dir, require_base_flats=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Source pair is not paper-ready; rerun or repair the base root before SD-IPC enrichment: {exc}"
            ) from exc
        _mirror_pair_dir(src_pair_dir, dst_pair_dir, copy_mode=args.copy_mode)

        src_asset = src_pair_dir / "grid_assets.json"
        out_asset = dst_pair_dir / "grid_assets.json"
        if not src_asset.exists():
            raise FileNotFoundError(f"Missing source grid asset: {src_asset}")
        if args.overwrite or not out_asset.exists():
            shutil.copy2(src_asset, out_asset)

        if out_asset.exists() and not args.overwrite:
            try:
                existing = json.loads(out_asset.read_text())
            except Exception:
                existing = {}
            if (
                existing.get("decoded_image_paths", {}).get("pstar_sdipc")
                and existing.get("trajectory_flat_paths", {}).get("pstar_sdipc")
            ):
                write_canonical_grid_assets(pair_dir=dst_pair_dir, require_base_flats=True)
                skipped += 1
                continue

        if args.overwrite:
            _remove_existing_sdipc_artifacts(dst_pair_dir)

        poe_path = src_pair_dir / "poe.png"
        if not poe_path.exists():
            raise FileNotFoundError(f"Missing PoE image required for SD-IPC enrichment: {poe_path}")

        init_latents = load_shared_init_latents(
            pair_dir=src_pair_dir,
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
            pair_dir=dst_pair_dir,
            seed=int(seed),
            decoded_images={"pstar_sdipc": image_sdipc},
            trackers={"pstar_sdipc": tracker_sdipc},
            projection_method=projection,
            condition_labels={"pstar_sdipc": "PoE p*"},
            source_prompts={"pstar_sdipc": "SD-IPC closed-form rerun from PoE"},
        )
        write_canonical_grid_assets(pair_dir=dst_pair_dir, require_base_flats=True)
        processed += 1
        print(f"  enriched {group_key}/{pair_slug}/seed_{seed}")

    print(
        f"\nCompleted SDXL SD-IPC enrichment: {processed} pair-seed record(s) generated, "
        f"{skipped} skipped."
    )
    print(f"Derived root: {output_dir}")


if __name__ == "__main__":
    main()
