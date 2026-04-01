"""
Export SD-IPC reachability records for the Phase 1 taxonomy study.

For each taxonomy pair-seed record, this script:

1. Loads the stored PoE image from `taxonomy_qualitative/`
2. Projects it into SD text-conditioning space with SD-IPC
3. Reruns SD 1.4 from the same initial noise while tracking the SD-IPC latent path
4. Reruns the original PoE condition from the same initial noise while tracking the PoE path
5. Measures d_t^{sdipc->poe}, d_T^{sdipc->poe}, and CLIP image similarity to the
   original PoE image

The resulting JSON can be passed into:
    scripts/aggregate_phase1_taxonomy_study.py --reachability-json <path>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from diffusers import DDIMScheduler, EulerDiscreteScheduler
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from notebooks.composition_experiments import LatentTrajectoryCollector
from notebooks.dynamics import get_latents
from notebooks.utils import get_sd_models
from phase1_taxonomy_common import (
    GROUP_ORDER,
    infer_pair_slug,
    infer_taxonomy_group,
    maybe_int,
    pair_sort_key,
    write_json,
)
from trajectory_dynamics_experiment import poe_sd_with_trajectory_tracking


CLIP_ID = "openai/clip-vit-large-patch14"
CYCLE_BREAK_THRESHOLD = 0.80


@torch.no_grad()
def _encode_prompt(texts: list[str], tokenizer, text_encoder, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(
        texts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(encoded.input_ids.to(device))[0]


@torch.no_grad()
def sample_sd1_with_precomputed_cond_tracking(
    latents: torch.Tensor,
    cond_emb: torch.Tensor,
    scheduler,
    unet,
    tokenizer,
    text_encoder,
    *,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    batch_size: int = 1,
    device: torch.device,
    dtype: torch.dtype,
    model_id: str,
    euler_init_noise_sigma: float = 1.0,
) -> tuple[torch.Tensor, LatentTrajectoryCollector]:
    """Deterministic DDIM rerun with precomputed conditioning sequence."""
    import inspect as _inspect

    ddim = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    ddim.set_timesteps(num_inference_steps)

    latents = (latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    cond_emb = cond_emb.to(device=device, dtype=dtype)
    uncond_emb = _encode_prompt([""] * batch_size, tokenizer, text_encoder, device).to(dtype=dtype)

    tracker = LatentTrajectoryCollector(
        num_inference_steps,
        batch_size,
        latents.shape[1],
        latents.shape[2],
        latents.shape[3],
    )

    extra_step_kwargs = {}
    if "eta" in _inspect.signature(ddim.step).parameters:
        extra_step_kwargs["eta"] = 0.0

    for i, t in enumerate(ddim.timesteps):
        latent_model_input = ddim.scale_model_input(latents, t)
        noise_pred_uncond = unet(latent_model_input, t, encoder_hidden_states=uncond_emb).sample
        noise_pred_cond = unet(latent_model_input, t, encoder_hidden_states=cond_emb).sample
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        tracker.store_step(i, latents, noise_pred, float(i) / num_inference_steps, t.item())
        latents = ddim.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    tracker.store_final(latents)
    return latents, tracker


def _clip_feature_tensor(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state[:, 0]
    if isinstance(outputs, (list, tuple)) and outputs:
        first = outputs[0]
        if isinstance(first, torch.Tensor):
            if first.ndim >= 3:
                return first[:, 0]
            return first
    raise TypeError(f"Unsupported CLIP output type: {type(outputs)!r}")


@torch.no_grad()
def clip_image_embedding(image: Image.Image, clip_model, clip_processor, device: torch.device) -> torch.Tensor:
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    outputs = clip_model.get_image_features(**inputs)
    feats = _clip_feature_tensor(outputs).float()
    return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))


@torch.no_grad()
def decode_to_pil(vae, latents: torch.Tensor) -> Image.Image:
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)
    latents = latents.to(device=next(vae.parameters()).device, dtype=vae.dtype)
    shift_factor = getattr(vae.config, "shift_factor", 0.0)
    if shift_factor is None:
        shift_factor = 0.0
    image = vae.decode(latents / vae.config.scaling_factor + shift_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image.permute(0, 2, 3, 1) * 255).to(torch.uint8)[0].cpu().numpy()
    return Image.fromarray(image)


def load_sdipc_projector(device: torch.device) -> tuple[CLIPModel, CLIPProcessor, torch.Tensor, torch.Tensor]:
    clip_model = CLIPModel.from_pretrained(CLIP_ID).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)
    clip_model.eval()
    with torch.no_grad():
        inv_text = torch.linalg.pinv(clip_model.text_projection.weight.float(), atol=0.3).to(device)
        visual_proj = clip_model.visual_projection.weight.float().to(device)
    return clip_model, clip_processor, inv_text, visual_proj


@torch.no_grad()
def sdipc_project(image: Image.Image, clip_model, clip_processor, inv_text, visual_proj, device: torch.device) -> torch.Tensor:
    pixel_values = clip_processor(images=image, return_tensors="pt").pixel_values.to(device)
    vision_outputs = clip_model.vision_model(pixel_values=pixel_values)
    pooler_out = _clip_feature_tensor(vision_outputs).float()
    joint = pooler_out @ visual_proj.T
    text_space = joint @ inv_text.T
    text_space = text_space / text_space.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return 27.5 * text_space


@torch.no_grad()
def sdipc_to_seq(proj_vec: torch.Tensor, tokenizer, text_encoder, device: torch.device) -> torch.Tensor:
    null_seq = _encode_prompt([""], tokenizer, text_encoder, device)
    seq = torch.zeros_like(null_seq)
    seq[:, 0] = null_seq[:, 0]
    proj_vec = proj_vec.to(device=device, dtype=seq.dtype)
    seq[:, 1:] = proj_vec.unsqueeze(1)
    return seq


def build_record(summary_path: Path) -> dict[str, Any] | None:
    try:
        with open(summary_path) as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    run_dir = summary_path.parent
    poe_image_path = run_dir / "poe.png"
    if not poe_image_path.exists():
        return None

    cfg = summary.get("config", {})
    taxonomy_group = infer_taxonomy_group(summary_path, cfg)
    if taxonomy_group not in GROUP_ORDER:
        return None

    prompt_a = cfg.get("prompt_a", "")
    prompt_b = cfg.get("prompt_b", "")
    if not prompt_a or not prompt_b:
        return None

    seed = maybe_int(cfg.get("seed"))
    if seed is None:
        return None

    return {
        "summary_path": summary_path,
        "run_dir": run_dir,
        "poe_image_path": poe_image_path,
        "taxonomy_group": taxonomy_group,
        "pair_slug": infer_pair_slug(run_dir),
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "seed": seed,
        "model_id": cfg.get("model_id", "CompVis/stable-diffusion-v1-4"),
        "num_inference_steps": maybe_int(cfg.get("num_inference_steps")) or 50,
        "guidance_scale": float(cfg.get("guidance_scale", 7.5)),
    }


def compute_reachability_record(
    record: dict[str, Any],
    *,
    models: dict[str, Any],
    clip_model,
    clip_processor,
    inv_text,
    visual_proj,
    device: torch.device,
    dtype: torch.dtype,
    image_root: Path | None = None,
) -> dict[str, Any]:
    model_id = record["model_id"]
    num_inference_steps = record["num_inference_steps"]
    guidance_scale = record["guidance_scale"]
    seed = record["seed"]

    euler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    x_T = get_latents(
        euler,
        z_channels=4,
        device=device,
        dtype=dtype,
        num_inference_steps=num_inference_steps,
        batch_size=1,
        latent_width=64,
        latent_height=64,
        seed=seed,
    )
    euler_sigma = float(getattr(euler, "init_noise_sigma", 1.0))

    poe_final, poe_tracker = poe_sd_with_trajectory_tracking(
        x_T.clone(),
        record["prompt_a"],
        record["prompt_b"],
        euler,
        models["unet"],
        models["tokenizer"],
        models["text_encoder"],
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        batch_size=1,
        device=device,
        dtype=dtype,
        model_id=model_id,
        euler_init_noise_sigma=euler_sigma,
    )

    poe_source = Image.open(record["poe_image_path"]).convert("RGB")
    proj_vec = sdipc_project(poe_source, clip_model, clip_processor, inv_text, visual_proj, device)
    cond_seq = sdipc_to_seq(proj_vec, models["tokenizer"], models["text_encoder"], device)
    sdipc_final, sdipc_tracker = sample_sd1_with_precomputed_cond_tracking(
        x_T.clone(),
        cond_seq,
        euler,
        models["unet"],
        models["tokenizer"],
        models["text_encoder"],
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        batch_size=1,
        device=device,
        dtype=dtype,
        model_id=model_id,
        euler_init_noise_sigma=euler_sigma,
    )

    poe_traj = poe_tracker.trajectories.float()
    sdipc_traj = sdipc_tracker.trajectories.float()
    d_t = ((sdipc_traj - poe_traj) ** 2).mean(dim=(1, 2, 3, 4)).tolist()
    d_T = float(d_t[-1])

    sdipc_image = decode_to_pil(models["vae"], sdipc_final)
    poe_anchor_image = decode_to_pil(models["vae"], poe_final)

    sdipc_image_path = ""
    poe_rerun_image_path = ""
    if image_root is not None:
        seed_dir = image_root / record["taxonomy_group"] / record["pair_slug"] / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        sdipc_path = seed_dir / "sdipc.png"
        poe_rerun_path = seed_dir / "poe_rerun.png"
        sdipc_image.save(sdipc_path)
        poe_anchor_image.save(poe_rerun_path)
        sdipc_image_path = str(sdipc_path)
        poe_rerun_image_path = str(poe_rerun_path)

    source_emb = clip_image_embedding(poe_source, clip_model, clip_processor, device)
    regen_emb = clip_image_embedding(sdipc_image, clip_model, clip_processor, device)
    poe_anchor_emb = clip_image_embedding(poe_anchor_image, clip_model, clip_processor, device)
    clip_cos = cosine_sim(source_emb, regen_emb)
    poe_anchor_cos = cosine_sim(source_emb, poe_anchor_emb)
    cycle_close = 1.0 if clip_cos >= CYCLE_BREAK_THRESHOLD else 0.0

    return {
        "taxonomy_group": record["taxonomy_group"],
        "pair_slug": record["pair_slug"],
        "prompt_a": record["prompt_a"],
        "prompt_b": record["prompt_b"],
        "seed": seed,
        "model_id": model_id,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "summary_path": str(record["summary_path"]),
        "run_dir": str(record["run_dir"]),
        "monolithic_image_path": str(record["run_dir"] / "monolithic.png"),
        "poe_image_path": str(record["poe_image_path"]),
        "sdipc_image_path": sdipc_image_path,
        "poe_rerun_image_path": poe_rerun_image_path,
        "d_t_sdipc_poe": [float(value) for value in d_t],
        "d_T_sdipc_poe": d_T,
        "clip_cosine_sdipc_poe": clip_cos,
        "cycle_close_rate_sdipc_poe": cycle_close,
        "cycle_broken": bool(cycle_close < 0.5),
        "clip_cosine_poe_rerun_vs_source": poe_anchor_cos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SD-IPC reachability records for the Phase 1 taxonomy study."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="Root taxonomy directory (default: experiments/eccv2026/taxonomy_qualitative).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Output JSON path (default: experiments/eccv2026/reachability/phase1_sdipc_reachability.json).",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="",
        help="Directory for exported SD-IPC / PoE-rerun images (default: sibling phase1_sdipc_images/).",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(GROUP_ORDER),
        choices=list(GROUP_ORDER),
        help="Subset of taxonomy groups to process.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional cap for quick smoke tests.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else (
        PROJECT_ROOT / "experiments" / "eccv2026" / "taxonomy_qualitative"
    )
    if not input_dir.exists():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_json = Path(args.output_json) if args.output_json else (
        PROJECT_ROOT / "experiments" / "eccv2026" / "reachability" / "phase1_sdipc_reachability.json"
    )
    image_dir = Path(args.image_dir) if args.image_dir else (
        output_json.parent / "phase1_sdipc_images"
    )
    image_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for summary_path in sorted(input_dir.rglob("summary.json")):
        record = build_record(summary_path)
        if record is None:
            continue
        if record["taxonomy_group"] not in args.groups:
            continue
        all_records.append(record)

    all_records.sort(
        key=lambda row: pair_sort_key(
            row["taxonomy_group"],
            row["pair_slug"],
            row["seed"],
        )
    )
    if args.max_records > 0:
        all_records = all_records[: args.max_records]

    if not all_records:
        print("ERROR: no taxonomy pair-seed records found for SD-IPC reachability export.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading SD 1.4 + CLIP on {device} ...")
    models = get_sd_models(
        model_id="CompVis/stable-diffusion-v1-4",
        dtype=dtype,
        device=device,
    )
    models["vae"].eval()
    models["unet"].eval()
    models["text_encoder"].eval()

    clip_model, clip_processor, inv_text, visual_proj = load_sdipc_projector(device)

    exported: list[dict[str, Any]] = []
    total = len(all_records)
    for idx, record in enumerate(all_records, start=1):
        print(
            f"[{idx}/{total}] {record['taxonomy_group']} / {record['pair_slug']} / seed {record['seed']}"
        )
        exported.append(
            compute_reachability_record(
                record,
                models=models,
                clip_model=clip_model,
                clip_processor=clip_processor,
                inv_text=inv_text,
                visual_proj=visual_proj,
                device=device,
                dtype=dtype,
                image_root=image_dir,
            )
        )

    manifest = {
        "records": exported,
        "config": {
            "input_dir": str(input_dir),
            "groups": list(args.groups),
            "max_records": args.max_records,
            "cycle_break_threshold": CYCLE_BREAK_THRESHOLD,
            "device": str(device),
            "model_id": "CompVis/stable-diffusion-v1-4",
            "image_dir": str(image_dir),
        },
    }
    write_json(output_json, manifest)
    print(f"Wrote SD-IPC reachability records → {output_json}")


if __name__ == "__main__":
    main()
