import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from anneal_samplers import (
    AnnealedCHASampler,
    AnnealedMALASampler,
    AnnealedUHASampler,
    AnnealedULASampler,
)
from composable_diffusion.model_creation import (
    Sampler_create_gaussian_diffusion,
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


SHAPES_TO_IDX = {"cube": 0, "sphere": 1, "cylinder": 2}
IDX_TO_SHAPE = {v: k for k, v in SHAPES_TO_IDX.items()}


def convert_images(batch: th.Tensor):
    scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(th.uint8).permute(0, 2, 3, 1)
    return scaled


def get_caption_simple(label_idx: int):
    return f"A {IDX_TO_SHAPE[int(label_idx)]}"


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", required=True)
parser.add_argument(
    "--sampler",
    type=str,
    default="MALA",
    choices=["MALA", "HMC", "UHMC", "ULA", "Rev_Diff"],
)
parser.add_argument(
    "--concept-a",
    dest="concept_a",
    type=str,
    default="sphere",
    choices=list(SHAPES_TO_IDX.keys()),
)
parser.add_argument(
    "--concept-b",
    dest="concept_b",
    type=str,
    default="cylinder",
    choices=list(SHAPES_TO_IDX.keys()),
)
parser.add_argument("--guidance-scale", dest="guidance_scale", type=float, default=4.0)
parser.add_argument("--batch-size", dest="batch_size", type=int, default=1)
parser.add_argument("--num-runs", dest="num_runs", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default="")
args = parser.parse_args()

if args.batch_size < 1:
    raise ValueError("--batch-size must be >= 1")
if args.num_runs < 1:
    raise ValueError("--num-runs must be >= 1")

th.manual_seed(args.seed)
if th.cuda.is_available():
    th.cuda.manual_seed_all(args.seed)

has_cuda = th.cuda.is_available()
device = th.device("cpu" if not has_cuda else "cuda")

options = model_and_diffusion_defaults()

# 64x64
model_path1 = args.ckpt_path
options["noise_schedule"] = "linear"
options["learn_sigma"] = False
options["use_fp16"] = False
options["num_classes"] = "3,"
options["dataset"] = "clevr_norel"
options["image_size"] = 64
options["num_channels"] = 128
options["num_res_blocks"] = 3
options["energy_mode"] = True

base_timestep_respacing = "100"

if options["energy_mode"]:
    print("Using energy mode")
    diffusion = Sampler_create_gaussian_diffusion(
        steps=100,
        learn_sigma=options["learn_sigma"],
        noise_schedule=options["noise_schedule"],
        timestep_respacing=base_timestep_respacing,
    )
else:
    diffusion = create_gaussian_diffusion(
        steps=100,
        learn_sigma=options["learn_sigma"],
        noise_schedule=options["noise_schedule"],
        timestep_respacing=base_timestep_respacing,
    )

if len(model_path1) > 0:
    assert os.path.exists(model_path1), f"Failed to resume from {model_path1}, file does not exist."
    weights = th.load(model_path1, map_location="cpu")
    model1, _ = create_model_and_diffusion(**options)
    model1.load_state_dict(weights)

model1 = model1.to(device)
model1.eval()

guidance_scale = args.guidance_scale
batch_size = args.batch_size

labels = th.tensor(
    [[[SHAPES_TO_IDX[args.concept_a]], [SHAPES_TO_IDX[args.concept_b]]]],
    dtype=th.long,
)
print(f"Composing: {get_caption_simple(labels[0, 0, 0])} + {get_caption_simple(labels[0, 1, 0])}")

labels = [x.squeeze(dim=1) for x in th.chunk(labels, labels.shape[1], dim=1)]
full_batch_size = batch_size * (len(labels) + 1)

masks = [True] * len(labels) * batch_size + [False] * batch_size
labels = th.cat((labels + [th.zeros_like(labels[0])]), dim=0)

model_kwargs = dict(
    y=labels.clone().detach().to(device),
    masks=th.tensor(masks, dtype=th.bool, device=device),
)


def model_fn_t(x_t, ts, **kwargs):
    cond_eps = model1(x_t, ts, eval=True, **kwargs)
    kwargs["y"] = th.zeros(kwargs["y"].shape, dtype=th.long, device=device)
    kwargs["masks"] = th.tensor([False] * batch_size, dtype=th.bool, device=device)
    uncond_eps = model1(x_t, ts, eval=True, **kwargs)

    eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)

    return eps


def cfg_model_fn(x_t, ts, **kwargs):
    half = x_t[:1]
    combined = th.cat([half] * kwargs["y"].size(0), dim=0)
    eps = model1(combined, ts, eval=True, **kwargs)

    cond_eps, uncond_eps = eps[:-1], eps[-1:]
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps).sum(dim=0, keepdim=True)
    eps = th.cat([half_eps] * x_t.size(0), dim=0)
    return eps


def cfg_model_fn_noen(x_t, ts, **kwargs):
    half = x_t[:1]
    combined = th.cat([half] * kwargs["y"].size(0), dim=0)
    eps = model1(combined, ts, **kwargs)

    cond_eps, uncond_eps = eps[:-1], eps[-1:]
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps).sum(dim=0, keepdim=True)
    eps = th.cat([half_eps] * x_t.size(0), dim=0)
    return eps


alphas = 1 - diffusion.betas
alphas_cumprod = np.cumprod(alphas)
scalar = np.sqrt(1 / (1 - alphas_cumprod))


def gradient(x_t, ts, **kwargs):
    half = x_t[:1]
    combined = th.cat([half] * kwargs["y"].size(0), dim=0)
    eps = model1(combined, ts, eval=True, **kwargs)

    cond_eps, uncond_eps = eps[:-1], eps[-1:]
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps).sum(dim=0, keepdim=True)
    eps = th.cat([half_eps] * x_t.size(0), dim=0)
    scale = scalar[ts[0]]
    return -1 * scale * eps


num_steps = 100

# ULA
la_steps = 20
la_step_sizes = diffusion.betas * 2

# HMC / UHMC
ha_steps = 10
num_leapfrog_steps = 3
damping_coeff = 0.7
mass_diag_sqrt = diffusion.betas
ha_step_sizes = diffusion.betas * 0.1

# MALA
la_steps = 20
la_step_sizes = diffusion.betas * 0.035


def gradient_cha(x_t, ts, **kwargs):
    half = x_t[:1]
    combined = th.cat([half] * kwargs["y"].size(0), dim=0)
    energy_norm, eps = model1(combined, ts, mala_sampler=True, **kwargs)

    cond_energy, uncond_energy = energy_norm[:-1], energy_norm[-1:]
    total_energy = uncond_energy.sum() + guidance_scale * (cond_energy.sum() - uncond_energy.sum())

    cond_eps, uncond_eps = eps[:-1], eps[-1:]
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps).sum(dim=0, keepdim=True)
    eps = th.cat([half_eps] * x_t.size(0), dim=0)

    scale = scalar[ts[0]]
    return -scale * total_energy, -1 * scale * eps


if args.sampler == "MALA":
    sampler = AnnealedMALASampler(num_steps, la_steps, la_step_sizes, gradient_cha)
elif args.sampler == "ULA":
    sampler = AnnealedULASampler(num_steps, la_steps, la_step_sizes, gradient)
elif args.sampler == "UHMC":
    sampler = AnnealedUHASampler(
        num_steps,
        ha_steps,
        ha_step_sizes,
        damping_coeff,
        mass_diag_sqrt,
        num_leapfrog_steps,
        gradient,
    )
elif args.sampler == "HMC":
    sampler = AnnealedCHASampler(
        num_steps,
        ha_steps,
        ha_step_sizes,
        damping_coeff,
        mass_diag_sqrt,
        num_leapfrog_steps,
        gradient_cha,
    )
elif args.sampler == "Rev_Diff":
    print("Using Reverse Diffusion Sampling only")
    sampler = None

print("Using Sampler:", args.sampler)
all_samp = []

for _ in range(args.num_runs):
    if options["energy_mode"]:
        samples = diffusion.p_sample_loop(
            sampler,
            cfg_model_fn,
            (full_batch_size, 3, 64, 64),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
    else:
        samples = diffusion.p_sample_loop(
            cfg_model_fn_noen,
            (full_batch_size, 3, 128, 128),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
    sample = samples.contiguous()
    sample = convert_images(sample)

    show_img = sample.cpu().detach().numpy()
    all_samp.append(show_img)

arr = np.concatenate(all_samp, axis=0)
show_img = th.tensor(arr).permute(0, 3, 1, 2)

num_images = show_img.shape[0]
columns = int(np.ceil(np.sqrt(num_images)))
rows = int(np.ceil(num_images / columns))
fig = plt.figure(figsize=(8 * columns, 8 * rows))

cap = f"A {args.concept_a} And A {args.concept_b}"
for i in range(num_images):
    img = show_img[i].permute(1, 2, 0).numpy()
    fig.add_subplot(rows, columns, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(cap, fontsize=16)

guidance_slug = str(guidance_scale).replace(".", "p")
default_out = f"Energy_Object_{args.concept_a}_and_{args.concept_b}_{args.sampler}_{guidance_slug}.png"
out_path = args.out or default_out
out_dir = os.path.dirname(out_path)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

plt.tight_layout()
plt.savefig(out_path)
print(f"Saved -> {out_path}")
