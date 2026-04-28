"""Reusable SDXL + SD-IPC helpers for taxonomy enrichment workflows."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from notebooks.composition_experiments import LatentTrajectoryCollector
else:
    try:
        from notebooks.composition_experiments import LatentTrajectoryCollector
    except Exception:
        from generate_spatial_relations_poe_grid_sdxl import LatentTrajectoryCollector


CLIP_L_ID = "openai/clip-vit-large-patch14"
CLIP_G_ID = "ViT-bigG-14"
CLIP_G_PRETRAINED = "laion2b_s39b_b160k"
SDIPC_SCALE = 27.5
BASE_CONDITIONS = ("prompt_a", "prompt_b", "monolithic", "poe")
SDIPC_CONDITION = "pstar_sdipc"
BASE_CONDITION_LABELS = {
    "prompt_a": "A",
    "prompt_b": "B",
    "monolithic": "A∧B",
    "poe": "PoE",
}
SDIPC_CONDITION_LABELS = {
    "pstar_sdipc": "PoE p*",
}
CANONICAL_DECODED_PATHS = {
    "prompt_a": "solo_a.png",
    "prompt_b": "solo_b.png",
    "monolithic": "monolithic.png",
    "poe": "poe.png",
    "pstar_sdipc": "grid_assets/pstar_sdipc.png",
}
CANONICAL_FLAT_PATHS = {
    "prompt_a": "grid_assets/trajectory_flat_prompt_a.npy",
    "prompt_b": "grid_assets/trajectory_flat_prompt_b.npy",
    "monolithic": "grid_assets/trajectory_flat_monolithic.npy",
    "poe": "grid_assets/trajectory_flat_poe.npy",
    "pstar_sdipc": "grid_assets/trajectory_flat_pstar_sdipc.npy",
}
PSTAR_IMAGE_CANDIDATES = (
    Path("grid_assets/pstar_sdipc.png"),
    Path("pstar_sdipc.png"),
)
PSTAR_FLAT_CANDIDATES = (
    Path("grid_assets/trajectory_flat_pstar_sdipc.npy"),
    Path("pstar_sdipc.npy"),
)


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    data = torch.from_numpy(np.array(image.convert("RGB"), copy=True))
    return data.permute(2, 0, 1).float() / 255.0


def build_sdxl_sdipc_runtime(
    *,
    models: dict[str, Any],
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    keep_clip_on_device: bool = False,
) -> dict[str, Any]:
    """Build reusable SDXL SD-IPC projection state."""
    import open_clip
    from diffusers import DDIMScheduler
    from transformers import CLIPModel, CLIPProcessor

    clip_l = CLIPModel.from_pretrained(CLIP_L_ID)
    clip_l.eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_L_ID)

    clip_g, _, preproc_g = open_clip.create_model_and_transforms(
        CLIP_G_ID,
        pretrained=CLIP_G_PRETRAINED,
    )
    clip_g.eval()
    if keep_clip_on_device:
        clip_l.to(device)
        clip_g.to(device)

    tokenizer = models["tokenizer"]
    tokenizer_2 = models["tokenizer_2"]
    text_encoder = models["text_encoder"]
    text_encoder_2 = models["text_encoder_2"]

    with torch.no_grad():
        w_l_pinv = torch.linalg.pinv(
            clip_l.text_projection.weight.float(),
            atol=0.3,
        )
        w_g_pinv = torch.linalg.pinv(
            text_encoder_2.text_projection.weight.float(),
            atol=0.3,
        )

        toks_l_null = tokenizer(
            "",
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        null_clip_l_seq = text_encoder(**toks_l_null).last_hidden_state.to(
            device=device,
            dtype=dtype,
        )

        toks_g_null = tokenizer_2(
            "",
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        out_g_null = text_encoder_2(**toks_g_null)
        null_clip_g_seq = out_g_null.last_hidden_state.to(device=device, dtype=dtype)
        null_clip_g_pooled = out_g_null.text_embeds.to(device=device, dtype=dtype)

    null_prompt_embeds = torch.cat([null_clip_l_seq, null_clip_g_seq], dim=-1)
    ddim_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    return {
        "keep_clip_on_device": bool(keep_clip_on_device),
        "clip_l": clip_l,
        "clip_processor": clip_processor,
        "clip_g": clip_g,
        "preproc_g": preproc_g,
        "w_l_pinv": w_l_pinv,
        "w_g_pinv": w_g_pinv,
        "null_clip_l_seq": null_clip_l_seq,
        "null_clip_g_seq": null_clip_g_seq,
        "null_clip_g_pooled": null_clip_g_pooled,
        "null_prompt_embeds": null_prompt_embeds,
        "ddim_scheduler": ddim_scheduler,
    }


@torch.no_grad()
def sdipc_project_sdxl_image(
    image: Image.Image,
    *,
    runtime: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project an RGB image into SDXL prompt embeds + pooled embeds."""
    clip_l = runtime["clip_l"]
    clip_g = runtime["clip_g"]
    clip_processor = runtime["clip_processor"]
    preproc_g = runtime["preproc_g"]
    w_l_pinv = runtime["w_l_pinv"]
    w_g_pinv = runtime["w_g_pinv"]
    keep_clip = bool(runtime.get("keep_clip_on_device", False))

    if not keep_clip:
        clip_l.to(device)
    pixel_values = clip_processor(images=image, return_tensors="pt").pixel_values.to(device)
    feat_l = clip_l.visual_projection(
        clip_l.vision_model(pixel_values=pixel_values).pooler_output.float()
    )
    feat_l = feat_l / feat_l.norm(dim=-1, keepdim=True).clamp_min(1e-8) * SDIPC_SCALE
    proj_l = feat_l @ w_l_pinv.to(device).T
    if not keep_clip:
        clip_l.cpu()

    if not keep_clip:
        clip_g.to(device)
    img_t = preproc_g(image).unsqueeze(0).to(device)
    feat_g = clip_g.encode_image(img_t).float()
    feat_g = feat_g / feat_g.norm(dim=-1, keepdim=True).clamp_min(1e-8) * SDIPC_SCALE
    proj_g = feat_g @ w_g_pinv.to(device).T
    if not keep_clip:
        clip_g.cpu()

    seq_l = torch.zeros_like(runtime["null_clip_l_seq"])
    seq_l[:, 0] = runtime["null_clip_l_seq"][:, 0]
    seq_l[:, 1:] = proj_l.to(dtype=seq_l.dtype).unsqueeze(1)

    seq_g = torch.zeros_like(runtime["null_clip_g_seq"])
    seq_g[:, 0] = runtime["null_clip_g_seq"][:, 0]
    seq_g[:, 1:] = proj_g.to(dtype=seq_g.dtype).unsqueeze(1)

    prompt_embeds = torch.cat([seq_l, seq_g], dim=-1).to(device=device, dtype=dtype)
    pooled = (feat_g / SDIPC_SCALE).to(device=device, dtype=dtype)
    return prompt_embeds, pooled


def _add_time_ids(*, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [[height, width, 0, 0, height, width]],
        dtype=dtype,
        device=device,
    )


@torch.no_grad()
def run_sdxl_precomputed_cond(
    *,
    init_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    models: dict[str, Any],
    runtime: dict[str, Any],
    guidance_scale: float,
    num_inference_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    model_id: str,
    euler_init_noise_sigma: float,
    height: int,
    width: int,
) -> tuple[torch.Tensor, LatentTrajectoryCollector]:
    """Run an SDXL denoising pass with precomputed conditioning and trajectory tracking."""
    del model_id  # runtime carries the scheduler already
    ddim = runtime["ddim_scheduler"]
    ddim.set_timesteps(num_inference_steps)

    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    uncond_prompt_embeds = runtime["null_prompt_embeds"].to(device=device, dtype=dtype)
    uncond_pooled = runtime["null_clip_g_pooled"].to(device=device, dtype=dtype)

    tracker = LatentTrajectoryCollector(
        num_inference_steps,
        1,
        latents.shape[1],
        latents.shape[2],
        latents.shape[3],
    )

    extra_step_kwargs: dict[str, Any] = {}
    if "eta" in inspect.signature(ddim.step).parameters:
        extra_step_kwargs["eta"] = 0.0

    pe_in = torch.cat([uncond_prompt_embeds, prompt_embeds], dim=0)
    pool_in = torch.cat([uncond_pooled, pooled_prompt_embeds], dim=0)
    time_ids = _add_time_ids(height=height, width=width, device=device, dtype=dtype).repeat(2, 1)
    added_cond_kwargs = {
        "text_embeds": pool_in,
        "time_ids": time_ids,
    }

    unet = models["unet"]
    for i, t in enumerate(ddim.timesteps):
        latent_model_input = ddim.scale_model_input(latents.repeat(2, 1, 1, 1), t)
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=pe_in,
            added_cond_kwargs=added_cond_kwargs,
            timestep_cond=None,
        ).sample
        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        tracker.store_step(i, latents, noise_pred, float(i) / num_inference_steps, t.item())
        latents = ddim.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    tracker.store_final(latents)
    return latents, tracker


@torch.no_grad()
def decode_latents_to_tensor(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    latents = latents.to(dtype=vae.dtype)
    shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
    images = vae.decode(
        latents / vae.config.scaling_factor + shift_factor,
        return_dict=False,
    )[0]
    return ((images / 2 + 0.5).clamp(0, 1)).float()


def _save_tensor_png(image_tensor: torch.Tensor, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    array = (
        image_tensor.detach()
        .cpu()
        .clamp(0, 1)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    Image.fromarray(array).save(out_path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _first_existing_relative(pair_dir: Path, candidates: tuple[Path, ...] | list[Path]) -> str | None:
    for rel in candidates:
        if (pair_dir / rel).exists():
            return str(rel)
    return None


def _resolve_decoded_paths(pair_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    decoded_paths = dict(payload.get("decoded_image_paths", {}) or {})
    for cond in BASE_CONDITIONS:
        rel = Path(CANONICAL_DECODED_PATHS[cond])
        if (pair_dir / rel).exists():
            decoded_paths[cond] = str(rel)

    pstar_rel = _first_existing_relative(pair_dir, PSTAR_IMAGE_CANDIDATES)
    if pstar_rel is not None:
        decoded_paths[SDIPC_CONDITION] = pstar_rel
    return decoded_paths


def _resolve_flat_paths(
    pair_dir: Path,
    payload: dict[str, Any],
    *,
    require_base_flats: bool,
) -> dict[str, str]:
    flat_paths = dict(payload.get("trajectory_flat_paths", {}) or {})
    resolved: dict[str, str] = {}

    for cond in BASE_CONDITIONS:
        canonical_rel = Path(CANONICAL_FLAT_PATHS[cond])
        if (pair_dir / canonical_rel).exists():
            resolved[cond] = str(canonical_rel)
            continue

        rel = flat_paths.get(cond)
        if rel and (pair_dir / rel).exists():
            resolved[cond] = rel
            continue

        if require_base_flats:
            raise FileNotFoundError(
                f"Missing base flat trajectory for '{cond}' in {pair_dir}"
            )

    pstar_rel = _first_existing_relative(pair_dir, PSTAR_FLAT_CANDIDATES)
    if pstar_rel is not None:
        resolved[SDIPC_CONDITION] = pstar_rel
    else:
        rel = flat_paths.get(SDIPC_CONDITION)
        if rel and (pair_dir / rel).exists():
            resolved[SDIPC_CONDITION] = rel

    return resolved


def canonicalize_grid_asset_payload(
    *,
    pair_dir: Path,
    require_base_flats: bool = True,
) -> dict[str, Any]:
    asset_path = pair_dir / "grid_assets.json"
    if not asset_path.exists():
        raise FileNotFoundError(f"Missing grid asset: {asset_path}")

    payload = _load_json(asset_path)
    decoded_paths = _resolve_decoded_paths(pair_dir, payload)
    flat_paths = _resolve_flat_paths(pair_dir, payload, require_base_flats=require_base_flats)

    flat_by_cond: dict[str, np.ndarray] = {}
    for cond, rel in flat_paths.items():
        flat_by_cond[cond] = np.load(pair_dir / rel).astype(np.float32, copy=False)

    projection_method = str(
        payload.get("projection_method")
        or (payload.get("trajectory_projection") or {}).get("projection_method")
        or "mds"
    )
    projected, n_steps = project_flat_trajectories(flat_by_cond, method=projection_method)

    condition_labels = dict(payload.get("condition_labels", {}) or {})
    condition_labels.update(BASE_CONDITION_LABELS)
    if SDIPC_CONDITION in decoded_paths or SDIPC_CONDITION in flat_paths:
        condition_labels.update(SDIPC_CONDITION_LABELS)

    existing_labels = dict(((payload.get("trajectory_projection") or {}).get("labels")) or {})
    existing_labels.update({cond: condition_labels.get(cond, cond) for cond in projected})

    payload["projection_method"] = projection_method
    payload["decoded_image_paths"] = decoded_paths
    payload["trajectory_flat_paths"] = flat_paths
    payload["condition_labels"] = condition_labels
    payload["trajectory_projection"] = {
        "projection_method": projection_method,
        "n_steps": int(n_steps),
        "projected": {cond: coords.tolist() for cond, coords in projected.items()},
        "labels": existing_labels,
    }
    return payload


def write_canonical_grid_assets(
    *,
    pair_dir: Path,
    require_base_flats: bool = True,
) -> dict[str, Any]:
    payload = canonicalize_grid_asset_payload(
        pair_dir=pair_dir,
        require_base_flats=require_base_flats,
    )
    (pair_dir / "grid_assets.json").write_text(json.dumps(payload, indent=2))
    return payload


def load_shared_init_latents(
    *,
    pair_dir: Path,
    euler_init_noise_sigma: float,
    device: torch.device,
    dtype: torch.dtype,
    reference_condition: str = "poe",
) -> torch.Tensor:
    """Recover the exact shared x_T used by the base qualitative run.

    The stored trajectory flats begin from the DDIM-space latent after dividing by
    Euler's init_noise_sigma. Multiply by that sigma to reconstruct the original
    Euler-scaled x_T, which keeps SD-IPC reruns aligned with the source root
    regardless of device-specific RNG differences.
    """
    payload = canonicalize_grid_asset_payload(pair_dir=pair_dir, require_base_flats=True)
    flat_paths = payload.get("trajectory_flat_paths") or {}
    rel = flat_paths.get(reference_condition)
    if not rel:
        raise KeyError(
            f"Missing trajectory flat path for reference condition '{reference_condition}' in {pair_dir / 'grid_assets.json'}"
        )

    arr = np.load(pair_dir / rel).astype(np.float32, copy=False)
    if arr.ndim != 2 or arr.shape[1] % 4 != 0:
        raise ValueError(f"Expected flat SDXL trajectory array for {pair_dir / rel}, got {arr.shape}")
    first = arr[0]
    spatial = first.shape[0] // 4
    side = int(round(spatial ** 0.5))
    if side * side * 4 != first.shape[0]:
        raise ValueError(f"Cannot reshape initial latent from {pair_dir / rel} with shape {arr.shape}")
    latents = first.reshape(1, 4, side, side) * float(euler_init_noise_sigma)
    return torch.from_numpy(latents).to(device=device, dtype=dtype)


def project_flat_trajectories(
    flat_by_cond: dict[str, np.ndarray],
    *,
    method: str = "mds",
) -> tuple[dict[str, np.ndarray], int]:
    cond_names = list(flat_by_cond.keys())
    if not cond_names:
        return {}, 0

    n_steps = int(next(iter(flat_by_cond.values())).shape[0])
    max_dim = max(arr.shape[1] for arr in flat_by_cond.values())
    stacked_parts: list[np.ndarray] = []
    for cond in cond_names:
        arr = flat_by_cond[cond].astype(np.float32, copy=False)
        if arr.shape[1] < max_dim:
            pad = np.zeros((arr.shape[0], max_dim - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        stacked_parts.append(arr)
    stacked = np.vstack(stacked_parts)

    if method == "pca":
        from sklearn.decomposition import PCA

        proj = PCA(n_components=2).fit_transform(stacked)
    else:
        from sklearn.manifold import MDS
        from sklearn.metrics import pairwise_distances

        dist = pairwise_distances(stacked, metric="euclidean")
        proj = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=42,
            normalized_stress="auto",
        ).fit_transform(dist)

    result: dict[str, np.ndarray] = {}
    offset = 0
    for cond in cond_names:
        steps = flat_by_cond[cond].shape[0]
        result[cond] = proj[offset : offset + steps]
        offset += steps
    return result, n_steps


def merge_grid_assets(
    *,
    pair_dir: Path,
    seed: int,
    decoded_images: dict[str, torch.Tensor],
    trackers: dict[str, LatentTrajectoryCollector],
    projection_method: str,
    condition_labels: dict[str, str],
    source_prompts: dict[str, str],
) -> None:
    asset_path = pair_dir / "grid_assets.json"
    if not asset_path.exists():
        raise FileNotFoundError(f"Missing grid asset: {asset_path}")

    payload = json.loads(asset_path.read_text())
    assets_dir = pair_dir / "grid_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    decoded_paths = dict(payload.get("decoded_image_paths", {}))
    for cond, image_tensor in decoded_images.items():
        out_img = assets_dir / f"{cond}.png"
        _save_tensor_png(image_tensor, out_img)
        decoded_paths[cond] = str(out_img.relative_to(pair_dir))

    trajectory_flat_paths = dict(payload.get("trajectory_flat_paths", {}))
    for cond, tracker in trackers.items():
        flat = tracker.trajectories[:, 0].reshape(tracker.trajectories.shape[0], -1)
        flat_np = flat.detach().cpu().numpy().astype(np.float16)
        out_flat = assets_dir / f"trajectory_flat_{cond}.npy"
        np.save(out_flat, flat_np)
        trajectory_flat_paths[cond] = str(out_flat.relative_to(pair_dir))

    merged_condition_labels = dict(payload.get("condition_labels", {}))
    merged_condition_labels.update(condition_labels)

    merged_source_prompts = dict(payload.get("source_prompts", {}))
    merged_source_prompts.update(source_prompts)

    payload["seed"] = int(seed)
    payload["projection_method"] = projection_method
    payload["decoded_image_paths"] = decoded_paths
    payload["trajectory_flat_paths"] = trajectory_flat_paths
    payload["condition_labels"] = merged_condition_labels
    payload["source_prompts"] = merged_source_prompts
    # NOTE: trajectory_projection is intentionally NOT updated here.
    # write_canonical_grid_assets() must be called after merge_grid_assets() to
    # recompute the joint MDS projection over all conditions (including any newly
    # added pstar_sdipc flat). Mixing old cached projections with a fresh partial
    # projection produces incompatible coordinate systems and misaligns x_T.
    asset_path.write_text(json.dumps(payload, indent=2))
