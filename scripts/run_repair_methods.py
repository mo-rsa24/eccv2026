"""
run_repair_methods.py
=====================
Unified runner script that applies all PoE repair methods to a pair of
concepts under the shared-noise protocol and records per-step diagnostics.

Supports SD1.x / SD2.x / SDXL and SD3.x models.

Usage
-----
# SD1.4 (fits in ~6 GB VRAM):
python scripts/run_repair_methods.py \
    --c1 "a cat" \
    --c2 "a dog" \
    --model_id CompVis/stable-diffusion-v1-4 \
    --method all --seed 42 --steps 50

# SD3.5 (requires ~24 GB VRAM):
python scripts/run_repair_methods.py \
    --c1 "a cat" \
    --c2 "a dog" \
    --model_id stabilityai/stable-diffusion-3.5-medium \
    --method all --seed 42 --steps 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# ---- Project imports -------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from notebooks.utils import get_sd_models, get_sd3_models, get_text_embedding, get_sd3_text_embedding, get_image
from notebooks.dynamics import get_latents

from compositions.repair_methods import (
    AdaptiveWeightingConfig,     run_adaptive_weighting,
    GradientSurgeryConfig,       run_gradient_surgery,
    TrustRegionConfig,           run_trust_region,
    MaskGatedPoEConfig,          run_mask_gated_poe,
    ResidualAdapterConfig,       run_residual_adapter,
    CurvaturePrecondConfig,      run_curvature_preconditioning,
    MCMCCorrectorConfig,         run_mcmc_corrector,
    BranchAndSelectConfig,       run_branch_and_select,
    ProbeEnergyConfig,           run_probe_energy,
    AdaptiveDiagnosticsConfig,   run_adaptive_diagnostics,
    TweediePoeConfig,            run_tweedie_poe_corrector,
    OverlapPenaltyConfig,        run_overlap_penalty_corrector,
    SpatialRoutingConfig,        run_spatial_routing,
    PoeAnchoredContrastiveConfig,  run_poe_anchored_contrastive,
    CorrectedSpatialMaskingConfig, run_corrected_spatial_masking,
    IRPoEConfig,                   run_ir_poe,
)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "01_adaptive_weighting":         (AdaptiveWeightingConfig,     run_adaptive_weighting),
    "02_gradient_surgery":           (GradientSurgeryConfig,       run_gradient_surgery),
    "03_trust_region":               (TrustRegionConfig,           run_trust_region),
    "04_mask_gated_poe":             (MaskGatedPoEConfig,          run_mask_gated_poe),
    "05_residual_adapter":           (ResidualAdapterConfig,       run_residual_adapter),
    "06_curvature_preconditioning":  (CurvaturePrecondConfig,      run_curvature_preconditioning),
    "07_mcmc_corrector":             (MCMCCorrectorConfig,         run_mcmc_corrector),
    "08_branch_and_select":          (BranchAndSelectConfig,       run_branch_and_select),
    "09_probe_energy":               (ProbeEnergyConfig,           run_probe_energy),
    "10_adaptive_diagnostics":       (AdaptiveDiagnosticsConfig,   run_adaptive_diagnostics),
    "11_tweedie_poe_corrector":      (TweediePoeConfig,            run_tweedie_poe_corrector),
    "12_overlap_penalty_corrector":  (OverlapPenaltyConfig,        run_overlap_penalty_corrector),
    "13_spatial_routing":            (SpatialRoutingConfig,         run_spatial_routing),
    "14_poe_anchored_contrastive":   (PoeAnchoredContrastiveConfig,  run_poe_anchored_contrastive),
    "15_corrected_spatial_masking":  (CorrectedSpatialMaskingConfig, run_corrected_spatial_masking),
    "16_ir_poe":                     (IRPoEConfig,                   run_ir_poe),
}


# ---------------------------------------------------------------------------
# Model-family detection
# ---------------------------------------------------------------------------

def _is_sd3(model_id: str) -> bool:
    return "stable-diffusion-3" in model_id.lower() or "sd3" in model_id.lower()


# ---------------------------------------------------------------------------
# Velocity function factory
# ---------------------------------------------------------------------------

def make_vel_fn(unet, device, dtype, scheduler):
    """
    Returns a vel_fn(latents, t, sigma, embeddings, cond_kwargs) -> Vel
    compatible with the repair method run_* interfaces.
    """
    def vel_fn(latents, t, sigma, embeddings, cond_kwargs=None):
        import inspect

        t_in = t.to(device, dtype=torch.float16)
        x_in = latents.to(device=device, dtype=dtype)
        e_in = embeddings.to(device=device, dtype=dtype)
        # scale_model_input handles σ-scaling for EulerDiscrete (= x / sqrt(σ²+1))
        # and is a no-op for schedulers that don't need it.  Calling it also
        # clears diffusers' internal "did you forget to scale?" warning flag.
        sample_in = scheduler.scale_model_input(x_in, t)

        forward_params = list(inspect.signature(unet.forward).parameters.keys())
        first_arg = "hidden_states" if "hidden_states" in forward_params else "sample"

        cond_kwargs_in = None
        if cond_kwargs is not None:
            cond_kwargs_in = {
                key: value.to(device=device, dtype=dtype)
                for key, value in cond_kwargs.items()
            }

        grad_enabled = latents.requires_grad
        autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"
        with torch.set_grad_enabled(grad_enabled):
            with torch.autocast(autocast_device, dtype=dtype, enabled=device.type == "cuda"):
                if cond_kwargs_in is None:
                    return unet(
                        **{first_arg: sample_in},
                        timestep=t_in,
                        encoder_hidden_states=e_in,
                    ).sample
                return unet(
                    **{first_arg: sample_in},
                    timestep=t_in,
                    encoder_hidden_states=e_in,
                    added_cond_kwargs=cond_kwargs_in,
                ).sample
    return vel_fn


# ---------------------------------------------------------------------------
# Diagnostics summariser
# ---------------------------------------------------------------------------

def summarise_infos(infos: list) -> dict:
    """Extract scalar summaries from per-step info dicts."""
    summary = {}
    if not infos:
        return summary
    keys = [k for k, v in infos[0].items() if isinstance(v, torch.Tensor) and v.ndim <= 1]
    for k in keys:
        vals = []
        for info in infos:
            v = info.get(k)
            if v is not None and isinstance(v, torch.Tensor):
                vals.append(v.float().mean().item())
        if vals:
            summary[k + "_mean"] = float(sum(vals) / len(vals))
            summary[k + "_final"] = vals[-1]
    return summary


def _to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return float(obj.detach().cpu().item())
        return obj.detach().cpu().float().tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_sd3 = _is_sd3(args.model_id)
    dtype = torch.bfloat16 if use_sd3 else torch.float16

    print(f"\n{'='*60}")
    print(f"  PoE Repair Method Runner")
    print(f"  model={args.model_id}")
    print(f"  c1='{args.c1}'  c2='{args.c2}'")
    print(f"  seed={args.seed}  steps={args.steps}  guidance={args.guidance}")
    print(f"{'='*60}\n")

    # --- Load models ---
    if use_sd3:
        print("Loading SD3 models...")
        models = get_sd3_models(model_id=args.model_id, dtype=dtype, device=device)
        unet = models["transformer"]
        vae  = models["vae"]
        from diffusers import FlowMatchEulerDiscreteScheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler")
        z_channels = 16

        def encode(prompt):
            emb, pooled = get_sd3_text_embedding(
                prompt,
                models["tokenizer"],   models["text_encoder"],
                models["tokenizer_2"], models["text_encoder_2"],
                models["tokenizer_3"], models["text_encoder_3"],
                device=device,
            )
            return emb, {"pooled_projections": pooled}

    else:
        print(f"Loading SD1/2 models...")
        models = get_sd_models(model_id=args.model_id, dtype=dtype, device=device)
        unet = models["unet"]
        vae  = models["vae"]
        from diffusers import EulerDiscreteScheduler
        scheduler = EulerDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler")
        z_channels = 4

        is_sdxl = models.get("is_sdxl", False)

        def encode(prompt):
            if is_sdxl:
                emb, pooled = get_text_embedding(
                    prompt,
                    models["tokenizer"],   models["text_encoder"],
                    device=device,
                    tokenizer_2=models["tokenizer_2"],
                    text_encoder_2=models["text_encoder_2"],
                    return_pooled=True,
                )
                add_time_ids = torch.tensor(
                    [[512, 512, 0, 0, 512, 512]],
                    device=device,
                    dtype=dtype,
                )
                return emb, {"text_embeds": pooled, "time_ids": add_time_ids}
            else:
                emb = get_text_embedding(prompt, models["tokenizer"], models["text_encoder"], device=device)
                return emb, None   # no pooled projections for SD1/2

    print("Encoding prompts...")
    emb_c1,  kwargs_c1  = encode(args.c1)
    emb_c2,  kwargs_c2  = encode(args.c2)
    emb_unc, kwargs_unc = encode("")

    # --- Initial latents (shared noise) ---
    scheduler.set_timesteps(args.steps)
    latents_init = get_latents(
        scheduler,
        z_channels=z_channels,
        device=device,
        dtype=dtype,
        num_inference_steps=args.steps,
        batch_size=1,
        latent_width=64,
        latent_height=64,
        seed=args.seed,
    )

    vel_fn = make_vel_fn(unet, device, dtype)

    # --- Determine which methods to run ---
    if args.method.lower() == "all":
        methods_to_run = list(METHOD_REGISTRY.keys())
    else:
        requested = [m.strip() for m in args.method.split(",")]
        methods_to_run = []
        for r in requested:
            matches = [k for k in METHOD_REGISTRY if k.startswith(r) or r in k]
            if not matches:
                print(f"  WARNING: method '{r}' not found. Available: {list(METHOD_REGISTRY.keys())}")
            else:
                methods_to_run.extend(matches)
        methods_to_run = list(dict.fromkeys(methods_to_run))

    # --- Output directory ---
    out_dir = Path(args.out_dir) / f"{args.c1}_{args.c2}".replace(" ", "_") / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for method_name in methods_to_run:
        cfg_cls, run_fn = METHOD_REGISTRY[method_name]
        cfg = cfg_cls(guidance_scale=args.guidance)

        print(f"\n[{method_name}]")
        start = time.time()

        # Reset to shared noise
        scheduler.set_timesteps(args.steps)
        latents = latents_init.clone()

        try:
            final_latents, infos = run_fn(
                latents=latents,
                vel_fn=vel_fn,
                embeddings_c1=emb_c1,
                embeddings_c2=emb_c2,
                embeddings_uncond=emb_unc,
                scheduler=scheduler,
                cfg=cfg,
                cond_kwargs_c1=kwargs_c1,
                cond_kwargs_c2=kwargs_c2,
                cond_kwargs_uncond=kwargs_unc,
            )

            elapsed = time.time() - start
            print(f"  Done in {elapsed:.1f}s")

            img = get_image(vae, final_latents, nrow=1, ncol=1)
            img_path = out_dir / f"{method_name}.png"
            img.save(img_path)
            print(f"  Image saved: {img_path}")

            summary = summarise_infos(infos)
            summary["elapsed_s"] = elapsed
            results[method_name] = {"status": "ok", **summary}

            infos_path = out_dir / f"{method_name}_infos.json"
            with open(infos_path, "w") as f:
                json.dump(_to_serializable(infos), f, indent=2)

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results[method_name] = {"status": "error", "error": str(e)}

    # --- Save summary JSON ---
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # --- Print table ---
    print(f"\n{'Method':<35} {'Status':<10} {'Time (s)':<12}")
    print("-" * 60)
    for k, v in results.items():
        status = v.get("status", "?")
        t = v.get("elapsed_s", float("nan"))
        print(f"{k:<35} {status:<10} {t:<12.1f}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run PoE repair methods on a concept pair"
    )
    parser.add_argument("--c1", type=str, default="a cat", help="Concept 1 prompt")
    parser.add_argument("--c2", type=str, default="a dog", help="Concept 2 prompt")
    parser.add_argument(
        "--method", type=str, default="all",
        help="Method(s) to run: 'all', comma-separated IDs (e.g. '01,04,07'), or method name"
    )
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--steps",    type=int,   default=50)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument(
        "--model_id", type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument(
        "--out_dir", type=str, default="results/repair_comparison",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
