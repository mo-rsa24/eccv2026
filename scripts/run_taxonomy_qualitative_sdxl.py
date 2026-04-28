"""Run canonical SDXL taxonomy renders in flat or multi-seed layouts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from taxonomy_manifest import (
        GROUP_KEY_ALIASES,
        GROUP_DIR_ALIASES,
        GROUP_ORDER as MANIFEST_GROUP_ORDER,
        GROUP_SPECS,
        GROUP_LABEL_BY_KEY,
        PAIR_LOOKUP_BY_SLUG,
        DEFAULT_FINAL_SEEDS,
        normalize_group_key,
    )
except ImportError:
    from scripts.taxonomy_manifest import (
        GROUP_KEY_ALIASES,
        GROUP_DIR_ALIASES,
        GROUP_ORDER as MANIFEST_GROUP_ORDER,
        GROUP_SPECS,
        GROUP_LABEL_BY_KEY,
        PAIR_LOOKUP_BY_SLUG,
        DEFAULT_FINAL_SEEDS,
        normalize_group_key,
)


DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
GROUP_ORDER = list(MANIFEST_GROUP_ORDER)
TAXONOMY_GROUPS = {
    spec["key"]: {
        "label": spec["label"],
        "pairs": list(spec["pairs"]),
        "representative_pair": tuple(spec["representative_pair"]),
    }
    for spec in GROUP_SPECS
}


@dataclass
class TaxonomyExperimentConfig:
    group: str
    prompt_a: str
    prompt_b: str
    model_id: str = DEFAULT_MODEL_ID
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: int = 42
    projection: str = "mds"
    height: int = 1024
    width: int = 1024
    log_interaction_gap: bool = True
    interaction_gap_mode: str = "reference_and_poe"
    save_interaction_arrays: bool = False
    representative_only_interaction_arrays: bool = True


def format_pair_slug(prompt_a: str, prompt_b: str) -> str:
    clean_a = prompt_a.lower().replace(" ", "_").replace("'", "")
    clean_b = prompt_b.lower().replace(" ", "_").replace("'", "")
    return f"{clean_a}__x__{clean_b}"


def _canonical_to_legacy_group_dirs(group_key: str) -> list[str]:
    return list(GROUP_DIR_ALIASES.get(group_key, [group_key]))


def _load_runtime_modules() -> dict[str, Any]:
    from diffusers import EulerDiscreteScheduler

    from notebooks.dynamics import get_latents
    from notebooks.utils import get_image, get_sd_models
    from trajectory_dynamics_experiment_sdxl import (
        run_sdxl_reference_gap_with_tracking,
        poe_sdxl_monolithic_with_trajectory_tracking,
        poe_sdxl_with_trajectory_tracking,
    )

    return {
        "EulerDiscreteScheduler": EulerDiscreteScheduler,
        "get_latents": get_latents,
        "get_image": get_image,
        "get_sd_models": get_sd_models,
        "poe_sdxl_monolithic_with_trajectory_tracking": poe_sdxl_monolithic_with_trajectory_tracking,
        "poe_sdxl_with_trajectory_tracking": poe_sdxl_with_trajectory_tracking,
        "run_sdxl_reference_gap_with_tracking": run_sdxl_reference_gap_with_tracking,
    }


def _resolve_pair_token(pair_token: str) -> tuple[str, str, str]:
    group_hint: str | None = None
    slug_token = pair_token
    if "/" in pair_token:
        group_hint, slug_token = pair_token.split("/", 1)
        group_hint = normalize_group_key(group_hint.strip())
    slug_token = slug_token.strip()

    meta = PAIR_LOOKUP_BY_SLUG.get(slug_token)
    if meta is not None:
        group_key = normalize_group_key(str(meta["taxonomy_group_key"]))
        if group_hint is not None and group_hint != group_key:
            raise ValueError(
                f"Pair token '{pair_token}' requested group '{group_hint}' but resolves to '{group_key}'."
            )
        return group_key, str(meta["prompt_a"]), str(meta["prompt_b"])

    if "__x__" not in slug_token:
        raise ValueError(
            f"Unknown pair token '{pair_token}'. Use a taxonomy slug like "
            "'a_butterfly__x__a_flower_meadow' or prefix it with 'group1_cooccurrence/...'."
        )
    if group_hint is None:
        raise ValueError(
            f"Pair token '{pair_token}' is not in the canonical taxonomy manifest. "
            "Provide it as 'group_key/prompt_a__x__prompt_b'."
        )
    prompt_a_slug, prompt_b_slug = slug_token.split("__x__", 1)
    return group_hint, prompt_a_slug.replace("_", " "), prompt_b_slug.replace("_", " ")


def _load_roster_manifest(path: Path) -> list[tuple[str, str, str]]:
    payload = json.loads(path.read_text())
    groups = payload.get("groups", []) or []
    pairs_to_run: list[tuple[str, str, str]] = []
    for group in groups:
        group_key = normalize_group_key(str(group.get("taxonomy_group_key", "")).strip())
        for row in group.get("selected_pairs", []) or []:
            pair_value = row.get("pair")
            if isinstance(pair_value, list) and len(pair_value) == 2:
                prompt_a, prompt_b = str(pair_value[0]), str(pair_value[1])
            else:
                pair_slug = row.get("qualitative_pair_slug") or row.get("pair_slug")
                if not pair_slug:
                    raise ValueError(f"Roster entry in {path} is missing pair information: {row}")
                _, prompt_a, prompt_b = _resolve_pair_token(str(pair_slug))
            pairs_to_run.append((group_key, prompt_a, prompt_b))
    if not pairs_to_run:
        raise ValueError(f"No selected_pairs found in roster manifest: {path}")
    return pairs_to_run


def _shard_pairs(
    pairs_to_run: list[tuple[str, str, str]],
    *,
    num_workers: int,
    worker_index: int,
) -> list[tuple[str, str, str]]:
    if num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if worker_index < 0 or worker_index >= num_workers:
        raise ValueError("--worker-index must satisfy 0 <= worker-index < num-workers")
    if num_workers == 1:
        return pairs_to_run
    return [pair for idx, pair in enumerate(pairs_to_run) if idx % num_workers == worker_index]


def _pair_output_dir(
    output_root: Path,
    group: str,
    pair_slug: str,
    *,
    seed: int,
    multi_seed_layout: bool,
) -> Path:
    if multi_seed_layout:
        pair_dir = output_root / f"seed_{seed}" / group / pair_slug
    else:
        pair_dir = output_root / group / pair_slug
    pair_dir.mkdir(parents=True, exist_ok=True)
    return pair_dir


def _build_run_manifest(
    *,
    pairs_to_run: list[tuple[str, str, str]],
    all_pairs_requested: list[tuple[str, str, str]],
    args: argparse.Namespace,
    output_dir: Path,
    seeds: list[int],
    multi_seed_layout: bool,
) -> dict[str, Any]:
    return {
        "run_manifest_version": 2,
        "model_family": "sdxl",
        "created_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "layout": "multi_seed" if multi_seed_layout else "flat",
        "source_roster_manifest": str(args.roster_manifest) if args.roster_manifest else None,
        "model_id": args.model_id,
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "projection": args.projection,
        "worker_index": int(args.worker_index),
        "num_workers": int(args.num_workers),
        "gpu_id": args.gpu_id,
        "device": args.device,
        "height": int(args.height),
        "width": int(args.width),
        "seeds": list(seeds),
        "grid_seed": int(args.grid_seed),
        "requested_pair_count": len(all_pairs_requested),
        "worker_pair_count": len(pairs_to_run),
        "worker_record_count": len(pairs_to_run) * len(seeds),
        "all_requested_pairs": [
            {
                "taxonomy_group_key": group,
                "pair": [prompt_a, prompt_b],
                "qualitative_pair_slug": format_pair_slug(prompt_a, prompt_b),
            }
            for group, prompt_a, prompt_b in all_pairs_requested
        ],
        "selected_pairs": [
            {
                "taxonomy_group_key": group,
                "legacy_group_dirs": _canonical_to_legacy_group_dirs(group),
                "pair": [prompt_a, prompt_b],
                "qualitative_pair_slug": format_pair_slug(prompt_a, prompt_b),
            }
            for group, prompt_a, prompt_b in pairs_to_run
        ],
    }


def _load_models(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    runtime = _load_runtime_modules()
    models = runtime["get_sd_models"](model_id=model_id, dtype=dtype, device=device)
    scheduler = runtime["EulerDiscreteScheduler"].from_pretrained(model_id, subfolder="scheduler")
    return (
        models["tokenizer"],
        models["tokenizer_2"],
        models["text_encoder"],
        models["text_encoder_2"],
        models["unet"],
        models["vae"],
        scheduler,
    )


def _models_dict_from_tuple(models_tuple: tuple[Any, Any, Any, Any, Any, Any, Any]) -> dict[str, Any]:
    tokenizer, tokenizer_2, text_encoder, text_encoder_2, unet, vae, scheduler = models_tuple
    return {
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "unet": unet,
        "vae": vae,
        "scheduler": scheduler,
    }


def project_trajectories(
    trajectories_dict: dict[str, torch.Tensor],
    projection: str = "mds",
) -> dict[str, np.ndarray]:
    flat_traj = {}
    for name, traj in trajectories_dict.items():
        t_steps, batch, channels, height, width = traj.shape
        flat_traj[name] = traj.reshape(t_steps, batch * channels * height * width).cpu().numpy()

    all_points = np.vstack([flat_traj[name] for name in sorted(flat_traj.keys())])
    if projection == "pca":
        projector = PCA(n_components=2)
    else:
        projector = MDS(n_components=2, random_state=42, dissimilarity="euclidean")
    proj_2d = projector.fit_transform(all_points)

    result: dict[str, np.ndarray] = {}
    offset = 0
    for name in sorted(flat_traj.keys()):
        t_steps = flat_traj[name].shape[0]
        result[name] = proj_2d[offset : offset + t_steps]
        offset += t_steps
    return result


def plot_trajectories(
    projected_dict: dict[str, np.ndarray],
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(10, 8))
    colors = {
        "solo_a": "steelblue",
        "solo_b": "seagreen",
        "poe": "purple",
        "monolithic": "orange",
    }
    for name, traj_2d in projected_dict.items():
        color = colors.get(name, "gray")
        label = name.replace("_", " ").title()
        plt.plot(traj_2d[:, 0], traj_2d[:, 1], color=color, alpha=0.7, linewidth=2, label=label)
        plt.scatter(*traj_2d[0], color=color, s=100, marker="o", edgecolors="black", linewidth=1)
        plt.scatter(*traj_2d[-1], color=color, s=100, marker="*", edgecolors="black", linewidth=1)

    plt.xlabel("PC1 / MDS1")
    plt.ylabel("PC2 / MDS2")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_decoded_images(
    vae: Any,
    latents_dict: dict[str, torch.Tensor],
    output_path: Path,
    prompts: dict[str, str],
) -> None:
    runtime = _load_runtime_modules()
    order = [key for key in ["solo_a", "solo_b", "poe", "monolithic"] if key in latents_dict]
    fig, axes = plt.subplots(1, len(order), figsize=(12, 5))
    if len(order) == 1:
        axes = [axes]
    for col_idx, name in enumerate(order):
        image = runtime["get_image"](vae, latents_dict[name], nrow=1, ncol=1)
        axes[col_idx].imshow(image)
        axes[col_idx].set_title(prompts.get(name, name))
        axes[col_idx].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _trajectory_gap(poe_traj: torch.Tensor, mono_traj: torch.Tensor) -> tuple[list[float], float]:
    d_t_list = ((poe_traj - mono_traj) ** 2).mean(dim=(1, 2, 3, 4)).tolist()
    return d_t_list, float(d_t_list[-1])


def _is_representative_pair(group: str, prompt_a: str, prompt_b: str) -> bool:
    group_data = TAXONOMY_GROUPS.get(group, {})
    rep = tuple(group_data.get("representative_pair", ()))
    return rep == (prompt_a, prompt_b)


def _save_interaction_gap_artifacts(
    *,
    output_dir: Path,
    pair_slug: str,
    group: str,
    group_label: str,
    prompt_a: str,
    prompt_b: str,
    seed: int,
    interaction_result: dict[str, Any],
    interaction_gap_mode: str,
    save_interaction_arrays: bool,
    save_representative_arrays: bool,
) -> dict[str, Any]:
    interaction_dir = output_dir / "grid_assets" / "interaction_gap"
    interaction_dir.mkdir(parents=True, exist_ok=True)

    timeseries_rows = []
    for row in interaction_result["timeseries"]:
        payload = {
            "pair_slug": pair_slug,
            "pair": [prompt_a, prompt_b],
            "taxonomy_group_key": group,
            "taxonomy_group_label": group_label,
            "seed": int(seed),
            **row,
        }
        timeseries_rows.append(payload)

    timeseries_path = interaction_dir / "reference_gap_timeseries.json"
    summary_path = interaction_dir / "reference_gap_summary.json"
    delta_ref_path = interaction_dir / "delta_ref_l2_t.npy"
    delta_poe_path = interaction_dir / "delta_poe_l2_t.npy"
    traj_mse_path = interaction_dir / "traj_mse_mono_poe_t.npy"
    traj_cos_path = interaction_dir / "traj_cosdist_mono_poe_t.npy"

    timeseries_path.write_text(json.dumps(timeseries_rows, indent=2))
    np.save(delta_ref_path, interaction_result["delta_ref_l2_t"])
    np.save(delta_poe_path, interaction_result["delta_poe_l2_t"])
    np.save(traj_mse_path, interaction_result["traj_mse_mono_poe_t"])
    np.save(traj_cos_path, interaction_result["traj_cosdist_mono_poe_t"])

    artifact_paths = {
        "reference_gap_timeseries": str(timeseries_path.relative_to(output_dir)),
        "reference_gap_summary": str(summary_path.relative_to(output_dir)),
        "delta_ref_l2_t": str(delta_ref_path.relative_to(output_dir)),
        "delta_poe_l2_t": str(delta_poe_path.relative_to(output_dir)),
        "traj_mse_mono_poe_t": str(traj_mse_path.relative_to(output_dir)),
        "traj_cosdist_mono_poe_t": str(traj_cos_path.relative_to(output_dir)),
    }

    write_arrays = bool(save_interaction_arrays or save_representative_arrays)
    if write_arrays and interaction_result.get("peak_ref_absmap") is not None:
        ref_absmap_path = interaction_dir / "delta_ref_peak_absmap.npy"
        np.save(ref_absmap_path, interaction_result["peak_ref_absmap"])
        artifact_paths["delta_ref_peak_absmap"] = str(ref_absmap_path.relative_to(output_dir))
    if write_arrays and interaction_result.get("peak_poe_absmap") is not None:
        poe_absmap_path = interaction_dir / "delta_poe_peak_absmap.npy"
        np.save(poe_absmap_path, interaction_result["peak_poe_absmap"])
        artifact_paths["delta_poe_peak_absmap"] = str(poe_absmap_path.relative_to(output_dir))

    summary_payload = {
        "pair_slug": pair_slug,
        "pair": [prompt_a, prompt_b],
        "taxonomy_group_key": group,
        "taxonomy_group_label": group_label,
        "seed": int(seed),
        "interaction_gap_mode": interaction_gap_mode,
        "interaction_gap_definition": {
            "delta_ref_eps": "eps_mono(x_t^mono) - eps_poe(x_t^mono)",
            "delta_poe_eps": "eps_mono(x_t^poe) - eps_poe(x_t^poe)",
        },
        "summary": interaction_result["summary"],
        "artifact_paths": artifact_paths,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    return summary_payload


def run_pair(
    prompt_a: str,
    prompt_b: str,
    output_dir: Path,
    tokenizer: Any,
    tokenizer_2: Any,
    text_encoder: Any,
    text_encoder_2: Any,
    unet: Any,
    vae: Any,
    scheduler: Any,
    *,
    scale: float = 7.5,
    steps: int = 50,
    seed: int = 42,
    height: int = 1024,
    width: int = 1024,
    device: torch.device,
    dtype: torch.dtype,
    model_id: str = DEFAULT_MODEL_ID,
    projection: str = "mds",
    group: str = "",
    log_interaction_gap: bool = True,
    interaction_gap_mode: str = "reference_and_poe",
    save_interaction_arrays: bool = False,
    representative_only_interaction_arrays: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = _load_runtime_modules()

    x_t = runtime["get_latents"](
        scheduler,
        z_channels=4,
        device=device,
        dtype=dtype,
        num_inference_steps=steps,
        batch_size=1,
        latent_width=width // 8,
        latent_height=height // 8,
        seed=seed,
    )
    euler_sigma = float(getattr(scheduler, "init_noise_sigma", 1.0))
    monolithic_prompt = f"{prompt_a} and {prompt_b}"
    interaction_result = runtime["run_sdxl_reference_gap_with_tracking"](
        x_t.clone(),
        prompt_a,
        prompt_b,
        monolithic_prompt,
        scheduler,
        unet,
        tokenizer,
        text_encoder,
        tokenizer_2,
        text_encoder_2,
        guidance_scale=scale,
        num_inference_steps=steps,
        batch_size=1,
        device=device,
        dtype=dtype,
        model_id=model_id,
        euler_init_noise_sigma=euler_sigma,
        height=height,
        width=width,
        interaction_gap_mode=interaction_gap_mode,
    )
    latents_mono = interaction_result["latents_mono"]
    latents_poe = interaction_result["latents_poe"]
    tracker_mono = interaction_result["tracker_mono"]
    tracker_poe = interaction_result["tracker_poe"]

    latents_solo_a, tracker_solo_a = runtime["poe_sdxl_monolithic_with_trajectory_tracking"](
        x_t.clone(),
        prompt_a,
        scheduler,
        unet,
        tokenizer,
        text_encoder,
        tokenizer_2,
        text_encoder_2,
        guidance_scale=scale,
        num_inference_steps=steps,
        batch_size=1,
        device=device,
        dtype=dtype,
        model_id=model_id,
        euler_init_noise_sigma=euler_sigma,
        height=height,
        width=width,
    )

    latents_solo_b, tracker_solo_b = runtime["poe_sdxl_monolithic_with_trajectory_tracking"](
        x_t.clone(),
        prompt_b,
        scheduler,
        unet,
        tokenizer,
        text_encoder,
        tokenizer_2,
        text_encoder_2,
        guidance_scale=scale,
        num_inference_steps=steps,
        batch_size=1,
        device=device,
        dtype=dtype,
        model_id=model_id,
        euler_init_noise_sigma=euler_sigma,
        height=height,
        width=width,
    )

    solo_a_traj = tracker_solo_a.trajectories.float()
    solo_b_traj = tracker_solo_b.trajectories.float()
    poe_traj = tracker_poe.trajectories.float()
    mono_traj = tracker_mono.trajectories.float()
    d_t_list, d_t_terminal = _trajectory_gap(poe_traj, mono_traj)

    projected = project_trajectories(
        {
            "solo_a": solo_a_traj,
            "solo_b": solo_b_traj,
            "poe": poe_traj,
            "monolithic": mono_traj,
        },
        projection=projection,
    )
    plot_trajectories(
        projected,
        output_dir / "trajectory_manifold.png",
        title=f"{group or 'sdxl'} / {format_pair_slug(prompt_a, prompt_b)} / seed {seed}",
    )

    latents_for_decode = {
        "solo_a": latents_solo_a,
        "solo_b": latents_solo_b,
        "poe": latents_poe,
        "monolithic": latents_mono,
    }
    prompts_for_plot = {
        "solo_a": f"A: {prompt_a}",
        "solo_b": f"B: {prompt_b}",
        "poe": f"PoE: {prompt_a} ∧ {prompt_b}",
        "monolithic": f"Mono: {monolithic_prompt}",
    }
    plot_decoded_images(vae, latents_for_decode, output_dir / "decoded_images.png", prompts_for_plot)

    solo_a_image = runtime["get_image"](vae, latents_solo_a, nrow=1, ncol=1)
    solo_b_image = runtime["get_image"](vae, latents_solo_b, nrow=1, ncol=1)
    poe_image = runtime["get_image"](vae, latents_poe, nrow=1, ncol=1)
    mono_image = runtime["get_image"](vae, latents_mono, nrow=1, ncol=1)
    solo_a_image.save(output_dir / "solo_a.png")
    solo_b_image.save(output_dir / "solo_b.png")
    poe_image.save(output_dir / "poe.png")
    mono_image.save(output_dir / "monolithic.png")

    assets_dir = output_dir / "grid_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    trajectory_flat_paths = {}
    for cond, tracker in {
        "prompt_a": tracker_solo_a,
        "prompt_b": tracker_solo_b,
        "monolithic": tracker_mono,
        "poe": tracker_poe,
    }.items():
        flat = tracker.trajectories[:, 0].reshape(tracker.trajectories.shape[0], -1)
        out_flat = assets_dir / f"trajectory_flat_{cond}.npy"
        np.save(out_flat, flat.detach().cpu().numpy().astype(np.float16))
        trajectory_flat_paths[cond] = str(out_flat.relative_to(output_dir))

    pair_slug = format_pair_slug(prompt_a, prompt_b)
    group_label = GROUP_LABEL_BY_KEY.get(group, group)
    interaction_gap_summary = None
    if log_interaction_gap:
        is_representative = _is_representative_pair(group, prompt_a, prompt_b)
        interaction_gap_summary = _save_interaction_gap_artifacts(
            output_dir=output_dir,
            pair_slug=pair_slug,
            group=group,
            group_label=group_label,
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            seed=seed,
            interaction_result=interaction_result,
            interaction_gap_mode=interaction_gap_mode,
            save_interaction_arrays=save_interaction_arrays,
            save_representative_arrays=bool(representative_only_interaction_arrays and is_representative),
        )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "group": group,
        "group_label": group_label,
        "pair_slug": pair_slug,
        "pair": [prompt_a, prompt_b],
        "seed": int(seed),
        "model_family": "sdxl",
        "model_id": model_id,
        "metrics": {
            "d_T_poe_vs_mono": d_t_terminal,
            "d_t_poe_vs_mono": d_t_list,
        },
        "outputs": {
            "trajectory_manifold": str(output_dir / "trajectory_manifold.png"),
            "decoded_images": str(output_dir / "decoded_images.png"),
            "solo_a_image": str(output_dir / "solo_a.png"),
            "solo_b_image": str(output_dir / "solo_b.png"),
            "poe_image": str(output_dir / "poe.png"),
            "monolithic_image": str(output_dir / "monolithic.png"),
        },
    }
    if interaction_gap_summary is not None:
        summary["interaction_gap"] = interaction_gap_summary
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    grid_assets = {
        "model_family": "sdxl",
        "model_id": model_id,
        "seed": int(seed),
        "pair": [prompt_a, prompt_b],
        "pair_slug": pair_slug,
        "taxonomy_group_key": group,
        "taxonomy_group_label": group_label,
        "projection_method": projection,
        "prompt_key_map": {
            "A": prompt_a,
            "B": prompt_b,
        },
        "condition_labels": {
            "prompt_a": "A",
            "prompt_b": "B",
            "monolithic": "A∧B",
            "poe": "PoE",
        },
        "decoded_image_paths": {
            "prompt_a": "solo_a.png",
            "prompt_b": "solo_b.png",
            "monolithic": "monolithic.png",
            "poe": "poe.png",
        },
        "trajectory_flat_paths": trajectory_flat_paths,
        "trajectory_projection": {
            "projection_method": projection,
            "n_steps": int(poe_traj.shape[0]),
            "labels": {
                "prompt_a": "A",
                "prompt_b": "B",
                "monolithic": "A∧B",
                "poe": "PoE",
            },
            "projected": {
                "prompt_a": projected["solo_a"].tolist(),
                "prompt_b": projected["solo_b"].tolist(),
                "monolithic": projected["monolithic"].tolist(),
                "poe": projected["poe"].tolist(),
            },
        },
    }
    if interaction_gap_summary is not None:
        grid_assets["interaction_gap_paths"] = interaction_gap_summary["artifact_paths"]
        grid_assets["interaction_gap_summary"] = interaction_gap_summary["summary"]
        grid_assets["interaction_gap_definition"] = interaction_gap_summary["interaction_gap_definition"]
        grid_assets["reference_prompt_style"] = "operational_monolithic"
    (output_dir / "grid_assets.json").write_text(json.dumps(grid_assets, indent=2))
    return summary


def run_single_experiment(
    config: TaxonomyExperimentConfig,
    output_dir: Path,
    models_tuple: tuple[Any, Any, Any, Any, Any, Any, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    tokenizer, tokenizer_2, text_encoder, text_encoder_2, unet, vae, scheduler = models_tuple
    summary = run_pair(
        config.prompt_a,
        config.prompt_b,
        output_dir,
        tokenizer,
        tokenizer_2,
        text_encoder,
        text_encoder_2,
        unet,
        vae,
        scheduler,
        scale=config.guidance_scale,
        steps=config.num_inference_steps,
        seed=config.seed,
        height=config.height,
        width=config.width,
        device=device,
        dtype=dtype,
        model_id=config.model_id,
        projection=config.projection,
        group=config.group,
        log_interaction_gap=config.log_interaction_gap,
        interaction_gap_mode=config.interaction_gap_mode,
        save_interaction_arrays=config.save_interaction_arrays,
        representative_only_interaction_arrays=config.representative_only_interaction_arrays,
    )
    summary["config"] = asdict(config)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SDXL taxonomy qualitative experiments.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="SDXL model ID.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output root (default: experiments/eccv2026/taxonomy_qualitative_sdxl).",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=GROUP_ORDER,
        help="Taxonomy groups to run. Accepts canonical keys and legacy aliases.",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=[],
        help="Specific pairs as 'group/prompt_a__x__prompt_b'. If specified, overrides --groups.",
    )
    parser.add_argument(
        "--roster-manifest",
        type=str,
        default="",
        help="Frozen SDXL roster manifest. When set, run only its selected pairs.",
    )
    parser.add_argument("--max-per-group", type=int, default=0, help="Limit to N pairs per group (0 = all).")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of diffusion steps.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Legacy single-seed shorthand.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Explicit seed list.")
    parser.add_argument(
        "--grid-seed",
        type=int,
        default=None,
        help="Representative seed used by downstream figure scripts. Defaults to the first seed.",
    )
    parser.add_argument("--projection", type=str, default="mds", choices=["pca", "mds"])
    parser.add_argument(
        "--log-interaction-gap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log theorem-aligned mono-vs-PoE interaction-gap diagnostics.",
    )
    parser.add_argument(
        "--interaction-gap-mode",
        type=str,
        default="reference_and_poe",
        choices=["reference_only", "reference_and_poe"],
        help="Whether to save only reference-path residuals or both reference and PoE-path residuals.",
    )
    parser.add_argument(
        "--save-interaction-arrays",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist residual heatmap arrays for all pairs.",
    )
    parser.add_argument(
        "--representative-only-interaction-arrays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist residual heatmap arrays for canonical representative pairs only.",
    )
    parser.add_argument("--height", type=int, default=1024, help="Render height in pixels.")
    parser.add_argument("--width", type=int, default=1024, help="Render width in pixels.")
    parser.add_argument("--device", type=str, default="auto", help="Torch device to use.")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU id to pin this worker to.")
    parser.add_argument("--num-workers", type=int, default=1, help="Shard pair list across workers.")
    parser.add_argument("--worker-index", type=int, default=0, help="Zero-based worker shard index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "experiments" / "eccv2026" / "taxonomy_qualitative_sdxl"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_groups = [normalize_group_key(group) for group in args.groups]
    invalid_groups = [group for group in normalized_groups if group not in TAXONOMY_GROUPS]
    if invalid_groups:
        raise ValueError(
            f"Unknown --groups entries: {invalid_groups}. Valid groups: {sorted(TAXONOMY_GROUPS)}"
        )
    args.groups = normalized_groups

    if args.device != "auto" and args.gpu_id is not None:
        raise ValueError("Use either --device or --gpu-id, not both.")
    if args.gpu_id is not None:
        device_str = f"cuda:{args.gpu_id}"
    elif args.device != "auto":
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    seeds = list(args.seeds) if args.seeds is not None else [int(args.seed)]
    if not seeds:
        raise ValueError("At least one seed is required.")
    args.grid_seed = int(args.grid_seed) if args.grid_seed is not None else int(seeds[0])
    if args.grid_seed not in seeds:
        raise ValueError(f"--grid-seed {args.grid_seed} must appear in --seeds {seeds}")
    multi_seed_layout = len(seeds) > 1

    if args.roster_manifest and args.pairs:
        raise ValueError("Use either --pairs or --roster-manifest, not both.")

    pairs_to_run: list[tuple[str, str, str]] = []
    if args.roster_manifest:
        pairs_to_run = _load_roster_manifest(Path(args.roster_manifest))
    elif args.pairs:
        seen: set[tuple[str, str, str]] = set()
        for pair_spec in args.pairs:
            resolved = _resolve_pair_token(pair_spec)
            if resolved in seen:
                continue
            seen.add(resolved)
            pairs_to_run.append(resolved)
    else:
        for group in args.groups:
            group_data = TAXONOMY_GROUPS[group]
            group_added = 0
            for prompt_a, prompt_b in group_data["pairs"]:
                pairs_to_run.append((group, prompt_a, prompt_b))
                group_added += 1
                if args.max_per_group > 0 and group_added >= args.max_per_group:
                    break

    all_pairs_requested = list(pairs_to_run)
    pairs_to_run = _shard_pairs(
        pairs_to_run,
        num_workers=int(args.num_workers),
        worker_index=int(args.worker_index),
    )

    run_manifest = _build_run_manifest(
        pairs_to_run=pairs_to_run,
        all_pairs_requested=all_pairs_requested,
        args=args,
        output_dir=output_dir,
        seeds=seeds,
        multi_seed_layout=multi_seed_layout,
    )
    (output_dir / "sdxl_qualitative_run_manifest.json").write_text(json.dumps(run_manifest, indent=2))

    print(f"Loading SDXL from {args.model_id} on {device}...")
    models_tuple = _load_models(args.model_id, device, dtype)

    total_records = len(pairs_to_run) * len(seeds)
    print(
        f"Running {len(pairs_to_run)} pairs across {len(seeds)} seed(s) "
        f"({total_records} pair-seed records) on worker {args.worker_index}/{args.num_workers}."
    )
    for group, prompt_a, prompt_b in tqdm(pairs_to_run, desc="Pairs"):
        pair_slug = format_pair_slug(prompt_a, prompt_b)
        for seed in seeds:
            pair_dir = _pair_output_dir(
                output_dir,
                group,
                pair_slug,
                seed=seed,
                multi_seed_layout=multi_seed_layout,
            )
            config = TaxonomyExperimentConfig(
                group=group,
                prompt_a=prompt_a,
                prompt_b=prompt_b,
                model_id=args.model_id,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=seed,
                projection=args.projection,
                height=args.height,
                width=args.width,
                log_interaction_gap=bool(args.log_interaction_gap),
                interaction_gap_mode=str(args.interaction_gap_mode),
                save_interaction_arrays=bool(args.save_interaction_arrays),
                representative_only_interaction_arrays=bool(args.representative_only_interaction_arrays),
            )
            try:
                run_single_experiment(config, pair_dir, models_tuple, device, dtype)
                print(f"  ✓ {group}/{pair_slug}/seed_{seed}")
            except Exception as exc:
                print(f"  ✗ {group}/{pair_slug}/seed_{seed}: {exc}")

    print("\n✓ Taxonomy qualitative SDXL experiments complete")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
