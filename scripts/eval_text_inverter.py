"""
Closed-loop benchmark for SD35TextInverter.

Three evaluation modes:

  Phase 1 — Single-concept fidelity  (validates the inverter is trustworthy)
  ──────────────────────────────────
  For each prompt in a curated test set:
    1. Generate an image with SD3.5(prompt)               → img
    2. Invert: SD35TextInverter(img)                      → p*
    3. Re-generate: SD3.5(p*)                             → img*
    4. Metrics:
         • CLIP image–image similarity:  sim(img, img*)
         • CLIP text–text  similarity:   sim(prompt, p*)
         • Print side-by-side table

  Phase 2 — Composition (AND chimera inversion)
  ─────────────────────────────────────────────
  Pass --image-path to invert an arbitrary image (e.g. a SuperDiff-AND result).
  SD3.5 is NOT loaded in this mode — only the text inverter and CLIP.

Usage
-----
# Phase 1: single-concept benchmark (generates images on the fly)
conda run -n superdiff python scripts/eval_text_inverter.py \\
    --ckpt ckpt/text_inverter/best.pt \\
    --mode phase1 \\
    [--prompts "a cat" "a dog" "a red car"] \\
    [--seeds 0 1 2] \\
    [--output-dir outputs/text_inverter_eval]

# Phase 2: invert a specific image (AND chimera)
conda run -n superdiff python scripts/eval_text_inverter.py \\
    --ckpt ckpt/text_inverter/best.pt \\
    --mode image \\
    --image-path outputs/and_result.png
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import GPT2Tokenizer, CLIPModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.text_inverter import (
    load_text_inverter,
    make_clip_preprocessor,
    CLIP_SIZE, CLIP_MEAN, CLIP_STD,
)


# ---------------------------------------------------------------------------
# Default single-concept benchmark prompts
# Held out from generate_inversion_training_data.py training set
# ---------------------------------------------------------------------------

_PHASE1_DEFAULT_PROMPTS = [
    # Tier A — single concepts
    "a cat",
    "a dog",
    "a car",
    "a person",
    "a horse",
    "a bicycle",
    # Tier C — attribute-rich
    "a red car",
    "a black cat",
    "a fluffy white dog",
    "a yellow school bus",
    "a tiny red bird",
]


# ---------------------------------------------------------------------------
# CLIP similarity helpers
# ---------------------------------------------------------------------------

_CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize(CLIP_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(CLIP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])


@torch.no_grad()
def clip_image_sim(clip_model, pil_a: Image.Image, pil_b: Image.Image,
                   device: torch.device) -> float:
    """Cosine similarity between two PIL images in CLIP image-feature space."""
    ta = _CLIP_TRANSFORM(pil_a).unsqueeze(0).to(device)
    tb = _CLIP_TRANSFORM(pil_b).unsqueeze(0).to(device)
    fa = F.normalize(clip_model.get_image_features(pixel_values=ta).float(), dim=-1)
    fb = F.normalize(clip_model.get_image_features(pixel_values=tb).float(), dim=-1)
    return (fa * fb).sum().item()


@torch.no_grad()
def clip_text_sim(clip_model, clip_tok, text_a: str, text_b: str,
                  device: torch.device) -> float:
    """Cosine similarity between two text strings in CLIP text-feature space."""
    enc_a = clip_tok(text_a, return_tensors="pt", truncation=True,
                     max_length=77, padding=True).to(device)
    enc_b = clip_tok(text_b, return_tensors="pt", truncation=True,
                     max_length=77, padding=True).to(device)
    fa = F.normalize(clip_model.get_text_features(**enc_a).float(), dim=-1)
    fb = F.normalize(clip_model.get_text_features(**enc_b).float(), dim=-1)
    return (fa * fb).sum().item()


# ---------------------------------------------------------------------------
# Image preprocessing for the inverter
# ---------------------------------------------------------------------------

def pil_to_clip_tensor(pil_img: Image.Image, device: torch.device) -> torch.Tensor:
    """PIL → (1, 3, 224, 224) CLIP-normalised float32."""
    t = _CLIP_TRANSFORM(pil_img.convert("RGB")).unsqueeze(0)
    return t.to(device)


# ---------------------------------------------------------------------------
# SD3.5 generation helper (loaded lazily only in phase1 mode)
# ---------------------------------------------------------------------------

def load_sd35(model_id: str, dtype: torch.dtype, device: torch.device):
    from diffusers import StableDiffusion3Pipeline
    print(f"  Loading SD3.5 ({model_id}) ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id, torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_image(pipe, prompt: str, seed: int, steps: int = 28,
                   guidance: float = 4.5,
                   height: int = 512, width: int = 512) -> Image.Image:
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width,
        generator=gen,
    ).images[0]


# ---------------------------------------------------------------------------
# Phase 1: single-concept closed-loop evaluation
# ---------------------------------------------------------------------------

def run_phase1(args, model, gpt2_tok, device):
    """
    For each prompt × seed:
      SD3.5(prompt) → img → TextInverter → p* → SD3.5(p*) → img*
    Measures CLIP image-image and text-text similarity.
    """
    dtype    = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    prompts  = args.prompts or _PHASE1_DEFAULT_PROMPTS
    seeds    = args.seeds

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load SD3.5 (needed to generate images)
    pipe = load_sd35(args.model_id, dtype, device)

    # Load CLIP for metrics
    print("  Loading CLIP-L for evaluation metrics ...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clip_tok   = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    preprocess = make_clip_preprocessor(device)

    results = []
    print(f"\n{'Prompt':38s} {'Seed':>4}  {'p*':38s}  {'img-img':>7}  {'txt-txt':>7}")
    print("─" * 100)

    for prompt in prompts:
        for seed in seeds:
            # 1. Generate original image
            img_orig = generate_image(pipe, prompt, seed,
                                      steps=args.steps, guidance=args.guidance,
                                      height=args.image_size, width=args.image_size)

            # 2. Invert → p*
            pv  = pil_to_clip_tensor(img_orig, device)
            p_star = model.generate(pv, gpt2_tok, max_new_tokens=50, num_beams=4)[0]

            # 3. Re-generate from p*
            img_star = generate_image(pipe, p_star, seed,
                                      steps=args.steps, guidance=args.guidance,
                                      height=args.image_size, width=args.image_size)

            # 4. Metrics
            img_sim = clip_image_sim(clip_model, img_orig, img_star, device)
            txt_sim = clip_text_sim(clip_model, clip_tok, prompt, p_star, device)

            print(f"{prompt:38s} {seed:>4}  {p_star:38s}  {img_sim:7.4f}  {txt_sim:7.4f}")
            results.append({
                "prompt": prompt, "seed": seed,
                "p_star": p_star, "img_sim": img_sim, "txt_sim": txt_sim,
            })

            # Save images
            slug = prompt[:40].replace(" ", "_")
            img_orig.save(out_dir / f"{slug}_s{seed}_orig.png")
            img_star.save(out_dir / f"{slug}_s{seed}_star.png")
            (out_dir / f"{slug}_s{seed}_pstar.txt").write_text(p_star + "\n")

    # Summary
    avg_img = sum(r["img_sim"] for r in results) / len(results)
    avg_txt = sum(r["txt_sim"] for r in results) / len(results)
    print("─" * 100)
    print(f"{'Average':38s}        {'':38s}  {avg_img:7.4f}  {avg_txt:7.4f}\n")

    import json
    with open(out_dir / "phase1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_dir}/phase1_results.json")


# ---------------------------------------------------------------------------
# Phase 2 / image mode: invert an arbitrary image
# ---------------------------------------------------------------------------

def run_image_mode(args, model, gpt2_tok, device):
    """
    Invert a single image (e.g. a SuperDiff-AND chimera) to text.
    Does not require SD3.5 to be loaded.
    """
    img_path = Path(args.image_path)
    assert img_path.exists(), f"Image not found: {img_path}"

    pil_img = Image.open(img_path).convert("RGB")
    pv      = pil_to_clip_tensor(pil_img, device)

    print(f"\nInverting: {img_path}")
    p_star = model.generate(pv, gpt2_tok, max_new_tokens=60, num_beams=4)[0]

    print(f"\n  Recovered text: {p_star}")

    # Save
    out_path = img_path.with_suffix("") .parent / (img_path.stem + "_pstar_textinv.txt")
    out_path.write_text(p_star + "\n")
    print(f"  Saved to: {out_path}")

    return p_star


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate SD35TextInverter (closed-loop benchmark)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ckpt",       required=True, help="Path to text inverter checkpoint (.pt)")
    p.add_argument("--mode",       choices=["phase1", "image"], default="phase1")

    # Phase 1 options
    p.add_argument("--prompts",    nargs="+", default=None,
                   help="Prompts for Phase 1 (default: built-in single-concept set)")
    p.add_argument("--seeds",      type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--output-dir", default="outputs/text_inverter_eval")
    p.add_argument("--model-id",   default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--steps",      type=int,   default=28)
    p.add_argument("--guidance",   type=float, default=4.5)
    p.add_argument("--image-size", type=int,   default=512)
    p.add_argument("--dtype",      choices=["bfloat16", "float16"], default="bfloat16")

    # Image mode option
    p.add_argument("--image-path", default=None,
                   help="Path to an image to invert (used in --mode image)")

    # Model options
    p.add_argument("--clip-model-id", default="openai/clip-vit-large-patch14")
    p.add_argument("--gpt2-model-id", default="gpt2")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    print("Loading GPT-2 tokenizer ...")
    gpt2_tok = GPT2Tokenizer.from_pretrained(args.gpt2_model_id)
    gpt2_tok.pad_token = gpt2_tok.eos_token

    # Load text inverter
    print(f"Loading SD35TextInverter from {args.ckpt} ...")
    model = load_text_inverter(
        args.ckpt,
        clip_model_id=args.clip_model_id,
        gpt2_model_id=args.gpt2_model_id,
        device=device,
    ).eval()
    print(f"  Trainable: {model.num_trainable_params/1e6:.1f}M")

    if args.mode == "phase1":
        run_phase1(args, model, gpt2_tok, device)
    elif args.mode == "image":
        if args.image_path is None:
            print("ERROR: --image-path is required in --mode image")
            sys.exit(1)
        run_image_mode(args, model, gpt2_tok, device)


if __name__ == "__main__":
    main()
