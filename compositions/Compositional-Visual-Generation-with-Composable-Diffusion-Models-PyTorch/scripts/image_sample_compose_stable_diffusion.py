import torch as th
import numpy as np
import torchvision.utils as tvu

from diffusers import LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler, PNDMScheduler
from composable_diffusion.composable_stable_diffusion.pipeline_composable_stable_diffusion import \
    ComposableStableDiffusionPipeline

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prompts", type=str, default="mystical trees | A magical pond | dark")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--scale", type=float, default=7.5)
parser.add_argument('--weights', type=str, default="7.5 | 7.5 | 7.5")
parser.add_argument("--seed", type=int, default=8)
parser.add_argument("--model_path", type=str, default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--num_images", type=int, default=1)
parser.add_argument("--scheduler", type=str, choices=["lms", "ddim", "ddpm", "pndm"], default="ddim",
                    help="ddpm may generate pure noises when using fewer steps.")
parser.add_argument("--r3_sampler", type=str, choices=["none", "ula"], default="none",
                    help="Enable Reduce-Reuse-Recycle style MCMC refinement.")
parser.add_argument("--r3_ula_steps", type=int, default=0,
                    help="Number of ULA updates per denoising step when --r3_sampler ula.")
parser.add_argument("--r3_ula_step_scale", type=float, default=2.0,
                    help="Multiplier on beta_t for ULA step size (R3-style).")
parser.add_argument("--r3_ula_t_min", type=int, default=500,
                    help="Apply ULA only when diffusion timestep t > this threshold.")
parser.add_argument("--r3_ula_noise_scale", type=float, default=1.0,
                    help="Noise multiplier for ULA updates.")
args = parser.parse_args()

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

prompts = args.prompts
weights = args.weights
scale = args.scale
steps = args.steps

pipe = ComposableStableDiffusionPipeline.from_pretrained(
    args.model_path,
).to(device)

# you can find more schedulers from https://github.com/huggingface/diffusers/blob/main/src/diffusers/__init__.py#L54
if args.scheduler == "lms":
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
elif args.scheduler == "ddim":
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
elif args.scheduler == "ddpm":
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
elif args.scheduler == "pndm":
    pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)

pipe.safety_checker = None

images = []
generator_device = "cuda" if device.type == "cuda" else "cpu"
generator = th.Generator(generator_device).manual_seed(args.seed)
for i in range(args.num_images):
    image = pipe(prompts, guidance_scale=scale, num_inference_steps=steps,
                 weights=args.weights, generator=generator,
                 r3_sampler=args.r3_sampler,
                 r3_ula_steps=args.r3_ula_steps,
                 r3_ula_step_scale=args.r3_ula_step_scale,
                 r3_ula_t_min=args.r3_ula_t_min,
                 r3_ula_noise_scale=args.r3_ula_noise_scale).images[0]
    images.append(th.from_numpy(np.array(image)).permute(2, 0, 1) / 255.)
grid = tvu.make_grid(th.stack(images, dim=0), nrow=4, padding=0)
slug = "_AND_".join(p.strip().replace(" ", "_") for p in args.prompts.split("|"))
tvu.save_image(grid, f'{slug}_seed{args.seed}.png')
