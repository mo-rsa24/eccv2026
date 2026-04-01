"""
Text → SD3.5 image → Text  closed-loop roundtrip.

Generates an image from a text prompt using SD3.5, then immediately runs that
image through a chosen prompt-recovery method (PEZ / Zero2Text / VLM) to
produce a text description.  The recovered prompt is saved to a .txt file and
printed to the terminal.

Usage
-----
# VLM captioning (BLIP-2, no extra args needed):
python scripts/prompt_roundtrip.py \\
    --prompt "a cat and a dog" \\
    --method vlm

# PEZ hard-prompt optimisation:
python scripts/prompt_roundtrip.py \\
    --prompt "a cat and a dog" \\
    --method pez \\
    --pez-tokens 16 --pez-steps 300 --pez-lr 0.4

# Zero2Text ridge-regression (c1/c2 auto-split from prompt on ' and '):
python scripts/prompt_roundtrip.py \\
    --prompt "a cat and a dog" \\
    --method z2t

# Explicit concept split for Zero2Text:
python scripts/prompt_roundtrip.py \\
    --prompt "a cat and a dog" \\
    --method z2t --c1 "a cat" --c2 "a dog"

Outputs (written to --output-dir, default outputs/roundtrip/):
    <slug>_s<seed>.png           — generated image
    <slug>_s<seed>_<method>.txt  — recovered prompt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from diffusers import StableDiffusion3Pipeline
from transformers import CLIPModel, CLIPTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID_DEFAULT = "stabilityai/stable-diffusion-3.5-medium"
CLIP_MODEL_ID    = "openai/clip-vit-large-patch14"

_CLIP_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275,  0.40821073],
                         [0.26862954, 0.26130258, 0.27577711]),
])

# ---------------------------------------------------------------------------
# CLIP helpers (shared by PEZ and Z2T)
# ---------------------------------------------------------------------------

def _pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    """PIL Image → (1, 3, H, W) float32 [0, 1]."""
    t = transforms.ToTensor()(pil_img.convert("RGB"))  # (3, H, W)
    return t.unsqueeze(0)


def _clip_image_feat(img_01: torch.Tensor, clip_model, device: torch.device) -> torch.Tensor:
    """CLIP-L image feature, L2-normalised.  Returns (1, D_feat)."""
    result = []
    for img in img_01.cpu():
        pil = transforms.ToPILImage()(img)
        result.append(_CLIP_EVAL_TRANSFORM(pil))
    clip_in = torch.stack(result).to(device)
    feat = clip_model.get_image_features(pixel_values=clip_in).float()
    return F.normalize(feat, dim=-1)


# ---------------------------------------------------------------------------
# PEZ  (Wen et al., NeurIPS 2023)
# ---------------------------------------------------------------------------

def _clip_encode_soft_embeds(
    clip_model,
    soft_embeds: torch.Tensor,   # (n_tokens, D_tok) float32
    device: torch.device,
) -> torch.Tensor:
    """
    Run CLIP text encoder with injected soft token embeddings (bypass lookup).
    Layout: [BOS] [soft × n_tokens] [EOS] [PAD × …]
    Returns L2-normalised (1, D_feat) pooled text feature.
    """
    text_model = clip_model.text_model
    emb_layer  = text_model.embeddings
    cfg        = text_model.config
    max_len    = cfg.max_position_embeddings   # 77

    n_tok = min(soft_embeds.shape[0], max_len - 2)
    n_pad = max_len - n_tok - 2

    bos_emb = emb_layer.token_embedding(
        torch.tensor([cfg.bos_token_id], device=device)).float()
    eos_emb = emb_layer.token_embedding(
        torch.tensor([cfg.eos_token_id], device=device)).float()

    parts = [bos_emb, soft_embeds[:n_tok].float(), eos_emb]
    if n_pad > 0:
        pad_emb = emb_layer.token_embedding(
            torch.zeros(n_pad, dtype=torch.long, device=device)).float()
        parts.append(pad_emb)

    seq     = torch.cat(parts, dim=0).unsqueeze(0)   # (1, max_len, D_tok)
    pos_ids = torch.arange(max_len, device=device).unsqueeze(0)
    hidden  = seq + emb_layer.position_embedding(pos_ids).float()

    seq_len     = 1 + n_tok + 1
    causal_mask = text_model._build_causal_attention_mask(
        1, max_len, hidden.dtype).to(device)

    ext_mask = torch.zeros(1, 1, 1, max_len, device=device)
    if n_pad > 0:
        ext_mask[0, 0, 0, seq_len:] = -1e4

    out    = text_model.encoder(inputs_embeds=hidden,
                                attention_mask=ext_mask,
                                causal_attention_mask=causal_mask)
    normed = text_model.final_layer_norm(out.last_hidden_state)
    pooled = normed[:, seq_len - 1, :]
    feat   = clip_model.text_projection(pooled)
    return F.normalize(feat, dim=-1)


def pez_invert_image(
    img_01:       torch.Tensor,
    clip_model,
    clip_tokenizer,
    n_tokens:     int   = 16,
    n_iters:      int   = 300,
    lr:           float = 0.4,
    device:       torch.device = None,
) -> str:
    """
    PEZ hard-prompt optimisation.
    Optimises n_tokens soft embeddings in CLIP-L token-embedding space to
    maximise cosine similarity with the target image embedding, then decodes
    the nearest discrete tokens to a text string.
    """
    device = device or torch.device("cuda")

    img_feat = _clip_image_feat(img_01, clip_model, device)

    tok_emb  = clip_model.text_model.embeddings.token_embedding.weight.detach().float()
    tok_emb  = tok_emb.to(device)

    init_ids = torch.randint(0, tok_emb.shape[0], (n_tokens,), device=device)
    soft     = tok_emb[init_ids].clone().requires_grad_(True)
    optim    = torch.optim.Adam([soft], lr=lr)

    for _ in range(n_iters):
        optim.zero_grad()
        with torch.no_grad():
            normed_soft  = F.normalize(soft, dim=-1)
            normed_vocab = F.normalize(tok_emb, dim=-1)
            nearest_ids  = (normed_soft @ normed_vocab.T).argmax(dim=-1)
            hard         = tok_emb[nearest_ids]
        ste       = soft + (hard - soft).detach()   # STE: forward=hard, backward=soft
        text_feat = _clip_encode_soft_embeds(clip_model, ste, device)
        loss      = -(text_feat * img_feat).sum()
        loss.backward()
        optim.step()

    with torch.no_grad():
        normed_soft  = F.normalize(soft.detach(), dim=-1)
        normed_vocab = F.normalize(tok_emb, dim=-1)
        final_ids    = (normed_soft @ normed_vocab.T).argmax(dim=-1)

    prompt = clip_tokenizer.decode(final_ids.cpu().tolist(), skip_special_tokens=True)
    return prompt.strip() or "a photo"


# ---------------------------------------------------------------------------
# Zero2Text-style  (Kim et al., arXiv 2602.01757)
# ---------------------------------------------------------------------------

_Z2T_TEMPLATES = [
    "{c1} and {c2}",
    "a photo of {c1} and {c2}",
    "{c1} next to {c2}",
    "{c1} beside {c2}",
    "{c1} with {c2}",
    "a {c1} and a {c2} together",
    "{c1} and {c2} in the same scene",
    "a scene containing {c1} and {c2}",
    "{c1} alongside {c2}",
    "an image of {c1} and {c2}",
    "realistic photo of {c1} and {c2}",
    "{c1} near {c2}",
    "{c1} and {c2}, photorealistic",
    "a high quality image of {c1} and {c2}",
    "{c1} together with {c2}",
    "a picture of {c1} and {c2}",
    "{c2} and {c1}",
    "photo of {c2} next to {c1}",
    "{c2} beside {c1}",
    "{c2} with {c1}",
]


@torch.no_grad()
def z2t_invert_image(
    img_01:      torch.Tensor,
    clip_model,
    clip_tokenizer,
    c1:          str,
    c2:          str,
    n_iters:     int   = 5,
    ridge_alpha: float = 0.01,
    device:      torch.device = None,
) -> str:
    """
    Zero2Text-style training-free prompt recovery via recursive ridge regression.
    Returns the best text from the candidate template pool.
    """
    device = device or torch.device("cuda")

    z_target = _clip_image_feat(img_01, clip_model, device)

    def embed_prompts(prompts):
        tokens = clip_tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=77,
        ).to(device)
        feats = clip_model.get_text_features(**tokens).float()
        return F.normalize(feats, dim=-1)

    best_prompt = f"{c1} and {c2}"

    for it in range(n_iters):
        pool = [t.format(c1=c1, c2=c2) for t in _Z2T_TEMPLATES]
        if it > 0:
            pool += [best_prompt]
        pool = list(dict.fromkeys(pool))   # deduplicate

        E       = embed_prompts(pool)
        M       = E.shape[0]
        EEt     = E @ E.T
        reg     = ridge_alpha * torch.eye(M, device=device, dtype=EEt.dtype)
        lam     = torch.linalg.solve(EEt + reg, E @ z_target.T)
        z_align = F.normalize((E.T @ lam).T, dim=-1)
        sims    = (E @ z_align.T).squeeze(-1)
        best_prompt = pool[sims.argmax().item()]

    return best_prompt


# ---------------------------------------------------------------------------
# VLM captioning  (BLIP-2)
# ---------------------------------------------------------------------------

def load_blip2(model_id: str, device: torch.device):
    """Load BLIP-2 processor and model."""
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    print(f"  Loading BLIP-2 ({model_id}) ...")
    processor = Blip2Processor.from_pretrained(model_id)
    model     = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16,
    ).to(device).eval()
    return processor, model


@torch.no_grad()
def vlm_caption_image(
    img_01:         torch.Tensor,
    blip2_proc,
    blip2_model,
    device:         torch.device,
    max_new_tokens: int = 60,
) -> str:
    """Caption img_01 with BLIP-2 and return the text string."""
    pil     = transforms.ToPILImage()(img_01.squeeze(0).clamp(0, 1))
    inputs  = blip2_proc(images=pil, return_tensors="pt").to(device, torch.float16)
    out_ids = blip2_model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = blip2_proc.decode(out_ids[0], skip_special_tokens=True).strip()
    return caption or "a photo"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Text → SD3.5 image → Text closed-loop roundtrip",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Input
    p.add_argument("--prompt",   required=True, help="Input text prompt for image generation.")
    p.add_argument("--method",   required=True, choices=["pez", "z2t", "vlm"],
                   help="Prompt-recovery method: pez | z2t | vlm")

    # Generation
    p.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    p.add_argument("--steps",    type=int,   default=28)
    p.add_argument("--guidance", type=float, default=4.5)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--width",    type=int,   default=1024)
    p.add_argument("--height",   type=int,   default=1024)
    p.add_argument("--dtype",    choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--output-dir", default="outputs/roundtrip")

    # PEZ options
    p.add_argument("--pez-tokens", type=int,   default=16,  help="Number of tokens to optimise (PEZ).")
    p.add_argument("--pez-steps",  type=int,   default=300, help="Gradient steps (PEZ).")
    p.add_argument("--pez-lr",     type=float, default=0.4, help="Adam learning rate (PEZ).")

    # Z2T options
    p.add_argument("--c1",          default=None, help="Concept 1 for Z2T template pool.")
    p.add_argument("--c2",          default=None, help="Concept 2 for Z2T template pool.")
    p.add_argument("--z2t-iters",   type=int,   default=5,    help="Ridge-regression rounds (Z2T).")
    p.add_argument("--z2t-alpha",   type=float, default=0.01, help="Ridge regularisation (Z2T).")

    # VLM options
    p.add_argument("--vlm-model-id",    default="Salesforce/blip2-opt-2.7b")
    p.add_argument("--vlm-max-tokens",  type=int, default=60)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if args.dtype == "float16" else torch.bfloat16

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # File-name slug from prompt + seed
    slug     = args.prompt[:60].replace(" ", "_").replace("/", "-")
    img_path = out_dir / f"{slug}_s{args.seed}.png"
    txt_path = out_dir / f"{slug}_s{args.seed}_{args.method}.txt"

    # ------------------------------------------------------------------
    # Step 1: Generate image with SD3.5
    # ------------------------------------------------------------------
    print(f"\n[1/2]  Generating image with SD3.5 ...")
    print(f"       prompt   : {args.prompt!r}")
    print(f"       model    : {args.model_id}")
    print(f"       steps    : {args.steps}  guidance: {args.guidance}  seed: {args.seed}")

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(desc="  Sampling")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    result    = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        generator=generator,
    )
    pil_image = result.images[0]
    pil_image.save(img_path)
    print(f"       saved    : {img_path}")

    # Convert to (1, 3, H, W) float32 [0, 1] tensor for inversion methods
    img_tensor = _pil_to_tensor01(pil_image)

    # Free SD3.5 VRAM before loading inversion models
    del pipe
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 2: Recover prompt from image
    # ------------------------------------------------------------------
    print(f"\n[2/2]  Running prompt recovery: {args.method.upper()} ...")

    recovered: str

    if args.method == "vlm":
        blip2_proc, blip2_model = load_blip2(args.vlm_model_id, device)
        recovered = vlm_caption_image(
            img_tensor, blip2_proc, blip2_model, device,
            max_new_tokens=args.vlm_max_tokens,
        )

    elif args.method == "pez":
        print(f"  Loading CLIP-L ({CLIP_MODEL_ID}) ...")
        clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
        clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_ID)
        print(f"  Optimising {args.pez_tokens} tokens × {args.pez_steps} steps ...")
        recovered = pez_invert_image(
            img_tensor, clip_model, clip_tokenizer,
            n_tokens=args.pez_tokens,
            n_iters=args.pez_steps,
            lr=args.pez_lr,
            device=device,
        )

    elif args.method == "z2t":
        # Auto-split on ' and ' if c1/c2 not given
        c1, c2 = args.c1, args.c2
        if c1 is None or c2 is None:
            parts = [p.strip() for p in args.prompt.split(" and ", 1)]
            if len(parts) == 2:
                c1, c2 = parts
                print(f"  Auto-split prompt → c1={c1!r}  c2={c2!r}")
            else:
                c1 = args.prompt
                c2 = args.prompt
                print(f"  WARNING: could not split prompt on ' and '; using full prompt for both c1 and c2.")
        print(f"  Loading CLIP-L ({CLIP_MODEL_ID}) ...")
        clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
        clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_ID)
        recovered = z2t_invert_image(
            img_tensor, clip_model, clip_tokenizer,
            c1=c1, c2=c2,
            n_iters=args.z2t_iters,
            ridge_alpha=args.z2t_alpha,
            device=device,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    txt_path.write_text(recovered + "\n")

    print(f"\n{'─' * 60}")
    print(f"  Input prompt   : {args.prompt}")
    print(f"  Method         : {args.method.upper()}")
    print(f"  Recovered text : {recovered}")
    print(f"{'─' * 60}")
    print(f"  Image saved    : {img_path}")
    print(f"  Text saved     : {txt_path}")
    print()


if __name__ == "__main__":
    main()
