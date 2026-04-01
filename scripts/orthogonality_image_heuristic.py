#!/usr/bin/env python3
"""
Image-Embedding Orthogonality Heuristic for Group 3 / Group 4 Pair Selection.

Implements Lemma 8.1 from Bradley et al. (2502.04549) faithfully:

    μ_i = E_{p_i}[x]  estimated as the average CLIP *image* embedding
    over N images generated from the diffusion model conditioned on concept i.

    orth_dot(i,j) = cos_sim(μ_i - μ_b,  μ_j - μ_b)

    orth_dot ≈ 0  →  factorized conditionals can hold  →  composition may work
    orth_dot high →  necessary condition violated       →  composition will fail

This fixes the limitation of composability_orthogonality_heuristic.py, which
used a single CLIP *text* embedding and produced an inflated positivity floor
(~0.25) that made Group 2 style+content pairs (theoretically orthogonal) look
*less* orthogonal than Group 3 candidates — the wrong ordering.

Two-phase pipeline
------------------
Phase 1  (--phase generate)
    Generate N images per concept with the SD model.
    Saves images to:
        OUT_DIR/images/{concept_slug}/img_{i:03d}.png
    Resumable: skips concepts whose images already exist.

Phase 2  (--phase embed)
    Load generated images, compute averaged CLIP image embeddings,
    compute pairwise orth_dot, assign group labels, save results.
    Saves:
        OUT_DIR/embeddings/{concept_slug}.pt   (averaged μ_i)
        OUT_DIR/results.json
        OUT_DIR/results.csv

Run both phases in sequence (default):
    python scripts/orthogonality_image_heuristic.py

Run only generation (useful on GPU node without CLIP env clash):
    python scripts/orthogonality_image_heuristic.py --phase generate

Run only embedding (if images already exist):
    python scripts/orthogonality_image_heuristic.py --phase embed

Usage
-----
    # default model = SD 3.5 medium, N=20 images per concept
    python scripts/orthogonality_image_heuristic.py

    # SD 1.4, fewer images, custom output dir
    python scripts/orthogonality_image_heuristic.py \\
        --model CompVis/stable-diffusion-v1-4 \\
        --n-images 10 \\
        --out experiments/eccv2026/orthogonality_image_heuristic_sd14

    # Only re-run the embedding phase (generation already done)
    python scripts/orthogonality_image_heuristic.py --phase embed

Group thresholds (over the *image-embedding* orth_dot distribution)
--------------------------------------------------------------------
    Group 3:  orth_dot < g3_hi              moderate non-orthogonality
    Group 4:  orth_dot in [g4_lo, g1_lo)   strong non-orthogonality (collision)
    Group 1:  orth_dot >= g1_lo             co-occurrence / in-distribution

Defaults (--g3-lo 0.15  --g3-hi 0.35  --g4-lo 0.50) are deliberately loose;
inspect the printed distribution and adjust with --g3-lo / --g3-hi / --g4-lo.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Candidate concept pool  (same pairs as composability_orthogonality_heuristic)
# ---------------------------------------------------------------------------

CANDIDATE_PAIRS: list[tuple[str, str]] = [
    # spatially disjoint
    ("a lighthouse",              "a rubber duck"),
    ("a typewriter",              "a cactus"),
    ("a bathtub",                 "a streetlamp"),
    ("a traffic cone",            "a hot air balloon"),
    ("a wooden chair",            "a ceiling fan"),
    ("a park bench",              "a kite"),
    ("a mailbox",                 "a windmill"),
    ("a flower pot",              "a helicopter"),
    ("a picnic basket",           "a ferris wheel"),
    ("a street sign",             "a sailboat"),
    # texture + foreground
    ("a brick wall",              "a potted plant"),
    ("a wooden surface",          "a red apple"),
    ("a tiled floor",             "a blue suitcase"),
    ("a marble countertop",       "a glass teapot"),
    ("a stone wall",              "a yellow raincoat"),
    ("a brick wall",              "a white bicycle"),
    ("a marble countertop",       "a green pear"),
    ("a wooden tabletop",         "a ceramic mug"),
    ("a woven rug",               "a glass vase"),
    # color distinctive
    ("a black grand piano",       "a white vase"),
    ("a red fire hydrant",        "a blue mailbox"),
    ("a purple suitcase",         "an orange skateboard"),
    ("a silver teapot",           "a gold trumpet"),
    # sky / ground
    ("a full moon",               "a lit candle"),
    ("a bright rainbow",          "a red barn"),
    ("a crescent moon",           "a wooden fence"),
    ("a flock of birds in the sky", "a stone pathway"),
    # style + content (Group 2 sanity-check — should score near 0)
    ("a dog",                     "oil painting style"),
    ("a lighthouse",              "watercolor style"),
    ("a bicycle",                 "sketch style"),
    ("a teapot",                  "claymation style"),
    # high co-occurrence (Group 1 sanity-check — should score near 0)
    ("a butterfly",               "a flower meadow"),
    ("a camel",                   "a desert landscape"),
    ("a dolphin",                 "an ocean wave"),
    # novel Group 3 / 4 candidates (existing)
    ("a desk lamp",               "a glacier"),
    ("a lab microscope",          "a hay bale"),
    ("a bathtub",                 "a hot air balloon"),
    ("a grand piano",             "a lighthouse"),
    ("a filing cabinet",          "a telescope"),
    ("a hospital bed",            "a combine harvester"),
    ("a rotary telephone",        "a hot air balloon"),
    ("a typewriter",              "a glacier"),
    ("a church pew",              "a cargo ship"),
    # --- 30 new pairs for threshold calibration ---
    # Expected Group 1 (co-occurrence, orth_dot > ~0.62):
    # animal + natural habitat / scene
    ("a swan",                    "a lake"),
    ("a polar bear",              "a snowy tundra"),
    ("a penguin",                 "an iceberg"),
    ("a horse",                   "a grassy meadow"),
    ("a surfer",                  "an ocean wave"),
    ("a skier",                   "a snowy mountain slope"),
    ("a campfire",                "a forest at night"),
    ("a sailboat",                "an open sea"),
    ("a hot air balloon",         "a clear blue sky"),
    ("a cactus",                  "a desert landscape"),
    # Expected Group 4 (visual-domain collision, orth_dot ~0.50–0.62):
    # objects from the same functional or spatial category
    ("a fork",                    "a spoon"),
    ("a hammer",                  "a screwdriver"),
    ("a laptop",                  "a keyboard"),
    ("a sofa",                    "a coffee table"),
    ("a shirt",                   "a tie"),
    ("a pen",                     "a pencil"),
    ("a mug",                     "a kettle"),
    ("a pillow",                  "a blanket"),
    ("a chair",                   "a desk"),
    ("a candle",                  "a candlestick holder"),
    # Expected Group 3 (OOD low-orth, orth_dot ~0.37–0.50):
    # objects from entirely different domains with no natural scene
    ("a grand piano",             "a surfboard"),
    ("a chandelier",              "a fishing rod"),
    ("a chess board",             "a fire hydrant"),
    ("a trophy",                  "a hay bale"),
    ("a bowling ball",            "a weather vane"),
    ("a fax machine",             "a canoe"),
    ("a microscope",              "a cowboy hat"),
    ("a chandelier",              "a submarine"),
    ("a grandfather clock",       "a surfboard"),
    ("a typewriter",              "a life jacket"),
    # --- additional pairs for denser coverage ---
    # Expected Group 1 (strong co-occurrence, orth_dot > ~0.62):
    # the scene is essentially "A in/on/near B" as a natural state
    ("a shark",                   "the ocean"),
    ("a monkey",                  "a jungle canopy"),
    ("a lion",                    "a savanna at sunset"),
    ("a flamingo",                "a shallow saltwater lake"),
    ("an eagle",                  "a mountain cliff"),
    ("a rowboat",                 "a calm lake"),
    ("a snowman",                 "a snowy field"),
    ("a telescope",               "a starry night sky"),
    ("a fishing rod",             "a riverbank"),
    ("a lawnmower",               "a green lawn"),
    ("a beach umbrella",          "a sandy beach"),
    ("a plow",                    "a plowed farm field"),
    ("a deck chair",              "a swimming pool"),
    ("a fire hydrant",            "a city sidewalk"),
    ("a windmill",                "a flat open field"),
    # Expected Group 4 (same functional/spatial domain, orth_dot ~0.50–0.62):
    # share a room, category, or use-context without being the same object
    ("a refrigerator",            "a microwave"),
    ("a toilet",                  "a bathroom sink"),
    ("a bed",                     "a wardrobe"),
    ("a bicycle",                 "a motorcycle"),
    ("a violin",                  "a cello"),
    ("a hammer",                  "a wrench"),
    ("a sofa",                    "an armchair"),
    ("a wine glass",              "a champagne flute"),
    ("a tennis racket",           "a badminton racket"),
    ("a backpack",                "a suitcase"),
    ("a clock",                   "a calendar"),
    ("a plate",                   "a bowl"),
    ("a broom",                   "a dustpan"),
    ("a stapler",                 "a hole punch"),
    ("a bucket",                  "a mop"),
    # Expected Group 3 (OOD, orth_dot < ~0.50):
    # drawn from entirely different semantic worlds, no natural scene
    ("a grand piano",             "an igloo"),
    ("a chandelier",              "a tractor"),
    ("a top hat",                 "an anchor"),
    ("a filing cabinet",          "a snowboard"),
    ("a gramophone",              "a scuba tank"),
    ("a bookcase",                "a jet ski"),
    ("a globe",                   "a fire extinguisher"),
    ("a xylophone",               "a combine harvester"),
    ("a monocle",                 "a drill press"),
    ("a grandfather clock",       "a life raft"),
    ("a chessboard",              "a hay bale"),
    ("a harpsichord",             "a bulldozer"),
    ("a abacus",                  "a surfboard"),
    ("a typewriter",              "a scuba mask"),
    ("a lectern",                 "a jet ski"),
    # --- True Group 4: semantic slot collision (same scale, same category) ---
    # two animals competing for the same semantic slot
    ("a cat",                     "a dog"),
    ("a cat",                     "an owl"),
    ("a cat",                     "a bear"),
    ("a dog",                     "a wolf"),
    ("a dog",                     "a fox"),
    ("a lion",                    "a tiger"),
    ("a rabbit",                  "a squirrel"),
    ("a duck",                    "a goose"),
    ("a horse",                   "a donkey"),
    ("a crow",                    "a raven"),
    # two vehicles of the same scale
    ("a car",                     "a truck"),
    ("a bicycle",                 "a scooter"),
    ("a bus",                     "a tram"),
    ("a canoe",                   "a kayak"),
    ("a fighter jet",             "a bomber plane"),
    # two people / human figures
    ("a knight",                  "a samurai"),
    ("a cowboy",                  "a sheriff"),
    # two household objects competing for the same scene slot
    ("a lamp",                    "a floor lamp"),
    ("a mug",                     "a cup"),
    ("a sofa",                    "a loveseat"),
]

BACKGROUND_PROMPT = "a photo"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slug(text: str) -> str:
    """Convert concept text to a filesystem-safe slug."""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def unique_concepts(pairs: list[tuple[str, str]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for a, b in pairs:
        for c in (a, b):
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


# ---------------------------------------------------------------------------
# Phase 1: generate images
# ---------------------------------------------------------------------------

def load_sd_pipeline(model_id: str, device: torch.device, dtype: torch.dtype):
    """Load the appropriate SD pipeline based on model_id."""
    if "stable-diffusion-3" in model_id:
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(device)
    else:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype, safety_checker=None
        ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_concept_images(
    concepts: list[str],
    out_dir: Path,
    model_id: str,
    n_images: int,
    guidance_scale: float,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    seed_base: int,
) -> None:
    """
    Generate n_images images per concept and save to out_dir/images/{slug}/.
    Skips concepts whose target directory already contains n_images images.
    """
    img_root = out_dir / "images"
    img_root.mkdir(parents=True, exist_ok=True)

    to_generate = []
    for concept in concepts:
        concept_dir = img_root / slug(concept)
        existing = list(concept_dir.glob("img_*.png")) if concept_dir.exists() else []
        if len(existing) >= n_images:
            print(f"  [skip] {concept!r}  ({len(existing)} images already exist)")
        else:
            to_generate.append(concept)

    if not to_generate:
        print("All concepts already have images. Skipping generation phase.")
        return

    print(f"\nLoading {model_id} for image generation...")
    pipe = load_sd_pipeline(model_id, device, dtype)

    for concept in to_generate:
        concept_dir = img_root / slug(concept)
        concept_dir.mkdir(parents=True, exist_ok=True)
        existing = list(concept_dir.glob("img_*.png"))
        start_i = len(existing)

        print(f"  Generating {n_images - start_i} images for {concept!r} ...")

        prompt = f"a photo of {concept}" if "style" not in concept else concept

        for i in range(start_i, n_images):
            generator = torch.Generator(device=device).manual_seed(seed_base + i)
            result = pipe(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                generator=generator,
                height=512, width=512,
            )
            img = result.images[0]
            img.save(concept_dir / f"img_{i:03d}.png")

    # Free VRAM before loading CLIP
    del pipe
    torch.cuda.empty_cache()
    print("Generation complete.\n")


# ---------------------------------------------------------------------------
# Phase 2: embed + score
# ---------------------------------------------------------------------------

def load_clip(clip_model_id: str, device: torch.device):
    from transformers import CLIPModel, CLIPProcessor
    print(f"Loading CLIP ({clip_model_id})...")
    model = CLIPModel.from_pretrained(clip_model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(clip_model_id)
    return model, processor


def embed_concept_images(
    concept: str,
    img_dir: Path,
    clip_model,
    clip_processor,
    device: torch.device,
) -> torch.Tensor:
    """
    Load all images for a concept, compute CLIP image embeddings,
    return the L2-normalised mean vector, shape (D,).
    """
    from PIL import Image

    image_paths = sorted(img_dir.glob("img_*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    embeddings = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs).float()
        embeddings.append(F.normalize(emb, dim=-1).squeeze(0).cpu())

    mean_emb = torch.stack(embeddings).mean(dim=0)
    return F.normalize(mean_emb.unsqueeze(0), dim=-1).squeeze(0)


def run_embedding_phase(
    pairs: list[tuple[str, str]],
    out_dir: Path,
    clip_model_id: str,
    device: torch.device,
    g3_hi: float,
    g4_lo: float,
    g1_lo: float,
    top: int,
) -> None:
    img_root = out_dir / "images"
    emb_root = out_dir / "embeddings"
    emb_root.mkdir(parents=True, exist_ok=True)

    clip_model, clip_processor = load_clip(clip_model_id, device)

    # Collect all unique concepts + background
    concepts = unique_concepts(pairs)
    all_concepts = [BACKGROUND_PROMPT] + concepts

    # Compute or load cached embeddings
    mu: dict[str, torch.Tensor] = {}
    for concept in all_concepts:
        cache_path = emb_root / f"{slug(concept)}.pt"
        if cache_path.exists():
            mu[concept] = torch.load(cache_path, map_location="cpu")
            print(f"  [cache] {concept!r}")
            continue

        concept_dir = img_root / slug(concept)
        if not concept_dir.exists() or not list(concept_dir.glob("img_*.png")):
            print(f"  [WARN] No images for {concept!r} — skipping.")
            continue

        print(f"  Embedding {concept!r} ...")
        mu[concept] = embed_concept_images(
            concept, concept_dir, clip_model, clip_processor, device
        )
        torch.save(mu[concept], cache_path)

    mu_b = mu.get(BACKGROUND_PROMPT)
    if mu_b is None:
        sys.exit("ERROR: background embedding missing. Run generation phase first.")

    # Pairwise scoring
    results = []
    for a, b in pairs:
        if a not in mu or b not in mu:
            print(f"  [skip pair] {a!r} + {b!r} (missing embeddings)")
            continue

        delta_a = F.normalize((mu[a] - mu_b).unsqueeze(0), dim=-1).squeeze()
        delta_b = F.normalize((mu[b] - mu_b).unsqueeze(0), dim=-1).squeeze()
        orth_dot = (delta_a @ delta_b).item()

        # Direct cosine similarity between concept embeddings
        concept_sim = (mu[a] @ mu[b]).item()

        is_style = any(
            x in c for c in [a, b]
            for x in ["style", "painting", "claymation", "sketch", "watercolor"]
        )

        if is_style:
            label = "Group2_style"
        elif orth_dot >= g1_lo:
            label = "Group1_cooccurrence"
        elif orth_dot >= g4_lo:
            label = "Group4_collision"
        elif orth_dot < g3_hi:
            label = "Group3_cand"
        else:
            label = "ambiguous"

        results.append({
            "concept_a":        a,
            "concept_b":        b,
            "orth_dot":         round(orth_dot, 4),
            "concept_sim":      round(concept_sim, 4),
            "label":            label,
        })

    results.sort(key=lambda r: r["orth_dot"])

    # Print distribution summary
    dots = [r["orth_dot"] for r in results if r["label"] != "Group2_style"]
    if dots:
        print(f"\north_dot distribution (excluding style pairs):")
        print(f"  min={min(dots):.4f}  max={max(dots):.4f}  "
              f"mean={sum(dots)/len(dots):.4f}")
    print(f"\nGroup thresholds:  G3<{g3_hi}   G4=[{g4_lo}, {g1_lo})   G1>={g1_lo}")

    # Print table
    col = 32
    print(f"\n{'#':<4} {'Concept A':<{col}} {'Concept B':<{col}} "
          f"{'orth_dot':>9} {'conc_sim':>9}  label")
    print("-" * 100)
    for i, r in enumerate(results[:top], 1):
        marker = " <--" if r["label"] in ("Group3_cand", "Group4_collision",
                                           "Group1_cooccurrence") else ""
        print(f"{i:<4} {r['concept_a']:<{col}} {r['concept_b']:<{col}}"
              f" {r['orth_dot']:>9.4f} {r['concept_sim']:>9.4f}  {r['label']}{marker}")

    # Count groups
    for grp in ("Group1_cooccurrence", "Group2_style", "Group3_cand",
                "Group4_collision", "ambiguous"):
        members = [r for r in results if r["label"] == grp]
        if members:
            dots = [r["orth_dot"] for r in members]
            print(f"\n{grp} ({len(members)} pairs, "
                  f"orth_dot {min(dots):.3f}–{max(dots):.3f}):")
            for r in members:
                print(f"  orth={r['orth_dot']:.3f}  {r['concept_a']}  +  {r['concept_b']}")

    # Save
    json_path = out_dir / "results.json"
    csv_path  = out_dir / "results.csv"

    with open(json_path, "w") as f:
        json.dump({
            "background": BACKGROUND_PROMPT,
            "g3_hi": g3_hi, "g4_lo": g4_lo, "g1_lo": g1_lo,
            "pairs": results,
        }, f, indent=2)

    with open(csv_path, "w") as f:
        headers = ["concept_a", "concept_b", "orth_dot", "concept_sim", "label"]
        f.write(",".join(headers) + "\n")
        for r in results:
            f.write(",".join(str(r[h]) for h in headers) + "\n")

    print(f"\nResults saved to:\n  {json_path}\n  {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Image-embedding orthogonality heuristic (Lemma 8.1)."
    )
    parser.add_argument("--phase", choices=["generate", "embed", "both"],
                        default="both",
                        help="Which phase to run (default: both).")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium",
                        help="HuggingFace SD model ID for image generation.")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14",
                        help="HuggingFace CLIP model ID.")
    parser.add_argument("--n-images", type=int, default=20,
                        help="Images to generate per concept (default: 20).")
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--num-steps", type=int, default=28)
    parser.add_argument("--seed-base", type=int, default=42,
                        help="RNG seed offset; concept i image j uses seed_base+j.")
    parser.add_argument("--g3-hi", type=float, default=0.50,
                        help="Upper bound (exclusive) of Group 3 orth_dot band.")
    parser.add_argument("--g4-lo", type=float, default=0.50,
                        help="Lower bound of Group 4 collision band.")
    parser.add_argument("--g1-lo", type=float, default=0.62,
                        help="Minimum orth_dot for Group 1 co-occurrence.")
    parser.add_argument("--top", type=int, default=40,
                        help="Rows to print in the summary table.")
    parser.add_argument("--out", default="",
                        help="Output directory. Defaults to "
                             "experiments/eccv2026/orthogonality_image_heuristic/")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out) if args.out else (
        project_root / "experiments" / "eccv2026" / "orthogonality_image_heuristic"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32

    concepts = unique_concepts(CANDIDATE_PAIRS)
    all_concepts = [BACKGROUND_PROMPT] + concepts
    print(f"Concepts: {len(all_concepts)} ({len(CANDIDATE_PAIRS)} pairs)")
    print(f"Output:   {out_dir}")
    print(f"Device:   {device}  dtype={dtype}")

    if args.phase in ("generate", "both"):
        print("\n=== Phase 1: Generate images ===")
        generate_concept_images(
            all_concepts,
            out_dir=out_dir,
            model_id=args.model,
            n_images=args.n_images,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            device=device,
            dtype=dtype,
            seed_base=args.seed_base,
        )

    if args.phase in ("embed", "both"):
        print("\n=== Phase 2: Embed + score ===")
        run_embedding_phase(
            CANDIDATE_PAIRS,
            out_dir=out_dir,
            clip_model_id=args.clip_model,
            device=device,
            g3_hi=args.g3_hi,
            g4_lo=args.g4_lo,
            g1_lo=args.g1_lo,
            top=args.top,
        )


if __name__ == "__main__":
    main()
