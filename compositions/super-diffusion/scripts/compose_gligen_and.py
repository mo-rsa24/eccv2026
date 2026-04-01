"""
compose_gligen_and.py
=====================
Inference-time composition of three score functions on SD 1.4:

    ε_final = a · ε_SD(mono_prompt)
            + b · ε_AND(concept_1 ∧ concept_2 ∧ ...)   [κ-weighted]
            + c · ε_GLIGEN(instance boxes)

All three components share the same GLIGEN-modified SD 1.4 UNet and its
4-channel latent space.  No retraining required.

Component roles
---------------
a · ε_SD     Monolithic semantic prior — global realism, co-occurrence
             statistics from training data, prevents OOD artifacts.

b · ε_AND    Logical conjunction — enforces that every concept is present.
             Composable Diffusion weighted sum (Liu et al., 2022):
                 ε_AND = ε_uncond + Σ_i w_i · (ε_i − ε_uncond)
             Each w_i is a per-concept guidance scale (typically = scale).
             Fixes entity omission without adaptive κ-solving.

c · ε_GLIGEN Spatial grounding — pretrained GLIGEN bounding-box conditioning
             steers each entity instance into its assigned region during the
             UNet forward pass itself (not post-hoc masking).  Prevents
             spatial hybridisation and enforces counting via box count.

Score combination
-----------------
Each component contributes a conditional direction dir_X = ε_X_cond − ε_uncond.

Monolithic and GLIGEN directions are scaled by `scale` (standard CFG).
The AND direction uses per-concept guidance weights w_i (Composable Diffusion
convention) and is NOT additionally scaled by `scale` — the w_i already encode
guidance strength.  `b` is then a plain mixing coefficient:

    ε_final = ε_uncond
            + a · scale · dir_SD
            + b · Σ_i w_i · dir_concept_i       ← Composable Diffusion AND
            + c · scale · dir_GLIGEN

GLIGEN scheduled sampling
--------------------------
GLIGEN grounding is applied at full strength during the first
`--gligen_phase` fraction of steps (layout phase) and disabled
afterwards.  The monolithic + AND components carry the detail phase without
spatial constraint, restoring texture naturalness.

Environment  (new env — GLIGEN needs diffusers ≥ 0.28)
-------------------------------------------------------
    conda create -n compose_gligen python=3.10 -y
    conda activate compose_gligen
    # RTX 5090 (Blackwell sm_120) requires PyTorch ≥ 2.7 + cu128
    pip install "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cu128
    pip install "diffusers>=0.28.0" "transformers>=4.36.0" "huggingface_hub>=0.21.0" accelerate pillow

Run command
-----------
    conda activate compose_gligen
    python compositions/super-diffusion/scripts/compose_gligen_and.py \\
        --mono_prompt     "2 dogs and 3 cats in a field" \\
        --concept_prompts "2 dogs | 3 cats" \\
        --comp_weights    "7.5 | 7.5" \\
        --gligen_phrases  "dog | dog | cat | cat | cat" \\
        --gligen_boxes    "0.05,0.1,0.4,0.9 0.45,0.1,0.8,0.9 0.0,0.45,0.33,0.95 0.33,0.45,0.66,0.95 0.66,0.45,1.0,0.95" \\
        --a 0.3 --b 1.0 --c 0.3 \\
        --seed 42 --out compose_gligen_and.png

Box format
----------
Each box: "x1,y1,x2,y2"  (normalised [0,1], space-separated, one per phrase).
Dogs occupy the upper half; cats the lower.  Adjust to taste.

API note (diffusers ≥ 0.28)
----------------------------
cross_attention_kwargs["gligen"] takes raw tensors:
    {"boxes": [B, MAX_OBJS, 4], "positive_embeddings": [B, MAX_OBJS, dim], "masks": [B, MAX_OBJS]}

The UNet's forward method calls unet.position_net(**gligen_args) internally to
produce objs, then injects {"objs": ...} into each attention block's kwargs.
Text embeddings use CLIPTextModel.pooler_output (EOS-token projection), matching
the pipeline's own __call__.  The gated fuser is only invoked when the "gligen"
key is present in cross_attention_kwargs — omitting it from uncond/mono/concept
calls leaves those predictions unaffected by spatial grounding.
"""

import argparse
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionGLIGENPipeline, DDIMScheduler

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--mono_prompt",     type=str,
                    default="2 dogs and 3 cats in a field",
                    help="Monolithic SD prompt (semantic prior)")
parser.add_argument("--concept_prompts", type=str,
                    default="2 dogs | 3 cats",
                    help="Pipe-separated concept prompts for AND composition")
parser.add_argument("--gligen_phrases",  type=str,
                    default="dog | dog | cat | cat | cat",
                    help="Pipe-separated instance labels for GLIGEN boxes")
parser.add_argument("--gligen_boxes",    type=str,
                    default="0.05,0.1,0.4,0.9 0.45,0.1,0.8,0.9 0.0,0.45,0.33,0.95 0.33,0.45,0.66,0.95 0.66,0.45,1.0,0.95",
                    help="Space-separated 'x1,y1,x2,y2' boxes, one per phrase")
parser.add_argument("--a",            type=float, default=0.3,
                    help="Weight for monolithic SD direction")
parser.add_argument("--b",            type=float, default=0.4,
                    help="Weight for Composable Diffusion AND direction")
parser.add_argument("--c",            type=float, default=0.3,
                    help="Weight for GLIGEN grounding direction")
parser.add_argument("--comp_weights",  type=str,   default="7.5 | 7.5",
                    help="Pipe-separated per-concept guidance weights for "
                         "Composable Diffusion AND (one per concept prompt, "
                         "e.g. '7.5 | 7.5').  Mirrors the --weights arg of "
                         "image_sample_compose_stable_diffusion.py.")
parser.add_argument("--scale",        type=float, default=7.5,
                    help="CFG guidance scale")
parser.add_argument("--steps",        type=int,   default=50)
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--height",       type=int,   default=512)
parser.add_argument("--width",        type=int,   default=512)
parser.add_argument("--gligen_phase", type=float, default=0.6,
                    help="Fraction of steps with full GLIGEN grounding active "
                         "(remainder: grounding off)")
parser.add_argument("--model_id",     type=str,
                    default="masterful/gligen-1-4-generation-text-box")
parser.add_argument("--out",          type=str,   default="compose_gligen_and.png")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float32

# ── Parse structured inputs ───────────────────────────────────────────────────

concept_prompts = [p.strip() for p in args.concept_prompts.split("|")]
gligen_phrases  = [p.strip() for p in args.gligen_phrases.split("|")]
gligen_boxes    = [
    [float(v) for v in b.strip().split(",")]
    for b in args.gligen_boxes.split()
]

assert len(gligen_phrases) == len(gligen_boxes), (
    f"gligen_phrases ({len(gligen_phrases)}) and gligen_boxes ({len(gligen_boxes)}) "
    f"must have the same length"
)

N = len(concept_prompts)
comp_weights = [float(w.strip()) for w in args.comp_weights.split("|")]
assert len(comp_weights) == N, (
    f"--comp_weights needs {N} values (one per concept prompt), got {len(comp_weights)}"
)

print(f"Mono prompt  : '{args.mono_prompt}'")
print(f"Concepts     : {concept_prompts}  w={comp_weights}")
print(f"GLIGEN       : {list(zip(gligen_phrases, gligen_boxes))}")
print(f"Weights      : a={args.a} (SD)  b={args.b} (AND)  c={args.c} (GLIGEN)")
print(f"Scale / steps: {args.scale} / {args.steps}  seed={args.seed}")

# ── Load model ────────────────────────────────────────────────────────────────

print(f"\nLoading {args.model_id} ...")
pipe = StableDiffusionGLIGENPipeline.from_pretrained(
    args.model_id, torch_dtype=dtype
).to(device)

# Replace default PNDM scheduler with DDIM for deterministic sampling
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

pipe.unet.eval()
pipe.vae.eval()
pipe.text_encoder.eval()

unet = pipe.unet
vae  = pipe.vae

# ── Text encoding ─────────────────────────────────────────────────────────────

MAX_OBJS = 30


@torch.no_grad()
def encode_text(prompt: str) -> torch.Tensor:
    """CLIP text encoder → last hidden state [1, 77, 768]."""
    tok = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return pipe.text_encoder(tok.input_ids.to(device))[0]


uncond_emb   = encode_text("")
mono_emb     = encode_text(args.mono_prompt)
concept_embs = [encode_text(p) for p in concept_prompts]

# ── GLIGEN grounding preparation ──────────────────────────────────────────────

@torch.no_grad()
def prepare_gligen_grounding(phrases, boxes):
    """
    Build (boxes_t, text_embs, masks) for cross_attention_kwargs["gligen"].

    In diffusers ≥ 0.28 the UNet's forward method calls
        unet.position_net(boxes=…, positive_embeddings=…, masks=…)
    internally to produce objs, so we pass the raw tensors.

    Text embeddings use CLIPTextModel.pooler_output (EOS-token projection),
    matching the pipeline's own __call__ implementation.
    """
    n_objs    = len(phrases)
    cross_dim = unet.config.cross_attention_dim

    # Batch-encode all phrases; pooler_output → [n_objs, cross_dim]
    tok_inputs = pipe.tokenizer(
        phrases, padding=True, return_tensors="pt"
    ).to(device)
    phrase_embeddings = pipe.text_encoder(**tok_inputs).pooler_output  # [n_objs, cross_dim]

    boxes_t   = torch.zeros(1, MAX_OBJS, 4,        device=device, dtype=dtype)
    text_embs = torch.zeros(1, MAX_OBJS, cross_dim, device=device, dtype=dtype)
    masks     = torch.zeros(1, MAX_OBJS,            device=device, dtype=dtype)

    boxes_t[0, :n_objs]   = torch.tensor(boxes, device=device, dtype=dtype)
    text_embs[0, :n_objs] = phrase_embeddings.to(dtype)
    masks[0, :n_objs]     = 1.0

    return boxes_t, text_embs, masks


print("Preparing GLIGEN grounding tokens ...")
gligen_boxes_t, gligen_text_embs, gligen_masks = prepare_gligen_grounding(gligen_phrases, gligen_boxes)

# cross_attention_kwargs for the GLIGEN UNet forward pass.
# The UNet calls unet.position_net(**gligen_args) internally.
# Omitting this dict from uncond/mono/concept calls leaves them ungrounded.
GLIGEN_CA_KWARGS = {
    "gligen": {
        "boxes":               gligen_boxes_t,
        "positive_embeddings": gligen_text_embs,
        "masks":               gligen_masks,
    }
}

# ── Latent initialisation ─────────────────────────────────────────────────────

generator = torch.Generator(device=device).manual_seed(args.seed)
latents   = torch.randn(
    (1, unet.config.in_channels, args.height // 8, args.width // 8),
    generator=generator, device=device, dtype=dtype,
)
pipe.scheduler.set_timesteps(args.steps)
latents = latents * pipe.scheduler.init_noise_sigma

# ── Denoising loop ────────────────────────────────────────────────────────────

print(f"\nDenoising ...")
for i, t in enumerate(pipe.scheduler.timesteps):

    # Scheduled sampling: full grounding during layout phase, off during detail phase
    ot = 1.0 if (i / args.steps) < args.gligen_phase else 0.0

    lt = pipe.scheduler.scale_model_input(latents, t)

    with torch.no_grad():

        # Unconditional baseline (shared across all three components)
        eps_uncond = unet(lt, t, encoder_hidden_states=uncond_emb).sample

        # Component 1 — monolithic SD
        eps_mono = unet(lt, t, encoder_hidden_states=mono_emb).sample

        # Component 2 — per-concept conditionals for AND composition
        eps_concepts = [
            unet(lt, t, encoder_hidden_states=emb).sample
            for emb in concept_embs
        ]

        # Component 3 — GLIGEN grounded conditional
        # Uses the monolithic text embedding so the grounding layer adds
        # spatial structure on top of the global semantic prior.
        # When grounding is off (detail phase) reuse eps_mono — saves a UNet call
        # and is equivalent since the gated fuser is only triggered by the
        # presence of "gligen" in cross_attention_kwargs.
        if ot > 0:
            eps_gligen = unet(
                lt, t,
                encoder_hidden_states=mono_emb,
                cross_attention_kwargs=GLIGEN_CA_KWARGS,
            ).sample
        else:
            eps_gligen = eps_mono

    # Conditional directions relative to uncond
    dir_mono   = eps_mono - eps_uncond

    # Composable Diffusion AND (Liu et al. 2022, Eq. matching pipeline L539):
    #   ε_AND = ε_uncond + Σ_i w_i · (ε_i − ε_uncond)
    # comp_weights carry their own guidance scale so this is NOT rescaled by
    # args.scale below — b is a plain mixing coefficient on top.
    dir_and = sum(
        w * (eps_c - eps_uncond)
        for w, eps_c in zip(comp_weights, eps_concepts)
    )

    # GLIGEN: pure grounding direction (zero when ot=0, matches dir_mono)
    dir_gligen = eps_gligen - eps_uncond

    # Final noise prediction:
    #   ε_final = ε_uncond
    #           + a · scale · dir_SD
    #           + b · Σ w_i · dir_i       (Composable Diffusion, w_i ≈ scale)
    #           + c · scale · dir_GLIGEN
    eps_final = eps_uncond + (
        args.a * args.scale * dir_mono +
        args.b * dir_and               +
        args.c * args.scale * dir_gligen
    )

    latents = pipe.scheduler.step(eps_final, t, latents).prev_sample

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  step {i+1:3d}/{args.steps}  ot={ot:.0f}")

# ── Decode and save ───────────────────────────────────────────────────────────

print("\nDecoding ...")
with torch.no_grad():
    image = vae.decode(latents / vae.config.scaling_factor).sample
image = (image / 2 + 0.5).clamp(0, 1)
image = (image[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
Image.fromarray(image).save(args.out)
print(f"Saved → {args.out}")
