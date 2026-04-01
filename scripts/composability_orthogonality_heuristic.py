#!/usr/bin/env python3
"""
Orthogonality Heuristic for Group 3 Candidate Pair Selection.

Implements Lemma 8.1 from "Mechanisms of Projective Composition of Diffusion
Models" (Bradley et al., 2502.04549):

    If Factorized Conditionals hold, the score-delta vectors must be orthogonal:
        (μ_i − μ_b)^T (μ_j − μ_b) = 0   for all i ≠ j

CLIP text embeddings proxy for the score-field mean directions.

Scoring
-------
For each pair (A, B):

  delta_A = embed(A) - embed(background)   [normalised]
  delta_B = embed(B) - embed(background)   [normalised]

  orthogonality_dot  = delta_A · delta_B   (want ≈ 0)
  concept_similarity = embed(A) · embed(B) (want low — low semantic relatedness)

  group3_score = -|orthogonality_dot| - 0.5 * concept_similarity
                 (higher = better Group 3 candidate)

Also computes a crude co-occurrence proxy: cosine similarity between the
joint caption embedding and the background embedding.  High similarity → pair
exists in the training distribution → Group 1 (in-distribution), not Group 3.

Usage
-----
    conda activate attend_excite
    python scripts/composability_orthogonality_heuristic.py

    # Only print the top-20 Group 3 candidates:
    python scripts/composability_orthogonality_heuristic.py --top 20

    # Use a different background phrase:
    python scripts/composability_orthogonality_heuristic.py --background "a photo"

Output
------
    experiments/eccv2026/orthogonality_heuristic/results.json
    experiments/eccv2026/orthogonality_heuristic/results.csv
    (printed table to stdout)
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

# ---------------------------------------------------------------------------
# Candidate pool — all groups from run_control_landscape.py + EXTRA_GROUPS
# ---------------------------------------------------------------------------

CANDIDATE_PAIRS: list[tuple[str, str]] = [
    # spatially_disjoint
    ("a lighthouse",              "a rubber duck"),
    ("a typewriter",              "a cactus"),
    ("a bathtub",                 "a streetlamp"),
    ("a traffic cone",            "a hot air balloon"),
    ("a brick wall",              "a sailboat"),
    ("a wooden chair",            "a ceiling fan"),
    # spatially_disjoint_v2
    ("a park bench",              "a kite"),
    ("a mailbox",                 "a windmill"),
    ("a traffic cone",            "a blimp"),
    ("a flower pot",              "a helicopter"),
    ("a picnic basket",           "a ferris wheel"),
    ("a street sign",             "a sailboat"),
    # texture_foreground
    ("a brick wall",              "a potted plant"),
    ("a wooden surface",          "a red apple"),
    ("a cobblestone street",      "a flock of birds"),
    ("a tiled floor",             "a blue suitcase"),
    ("a marble countertop",       "a glass teapot"),
    ("a stone wall",              "a yellow raincoat"),
    # texture_foreground_v2
    ("a tiled floor",             "a red suitcase"),
    ("a brick wall",              "a white bicycle"),
    ("a marble countertop",       "a green pear"),
    ("a wooden tabletop",         "a ceramic mug"),
    ("a stone pathway",           "a yellow raincoat"),
    ("a woven rug",               "a glass vase"),
    # color_distinctive
    ("a bright yellow taxi",      "a neon green umbrella"),
    ("a black grand piano",       "a white vase"),
    ("a red fire hydrant",        "a blue mailbox"),
    ("a purple suitcase",         "an orange skateboard"),
    ("a silver teapot",           "a gold trumpet"),
    ("a blue bicycle",            "a pink suitcase"),
    # color_distinctive_v2
    ("a red mailbox",             "a blue suitcase"),
    ("a yellow raincoat",         "a purple skateboard"),
    ("a green bicycle",           "a pink handbag"),
    ("a silver teapot",           "a gold trumpet"),
    ("a black umbrella",          "a white lantern"),
    ("an orange traffic cone",    "a turquoise chair"),
    # sky_ground
    ("a blue sky with clouds",    "a green meadow"),
    ("a full moon",               "a lit candle"),
    ("a red sports car",          "a green bicycle"),
    ("a bright rainbow",          "a red barn"),
    ("a crescent moon",           "a wooden fence"),
    ("a flock of birds in the sky", "a stone pathway"),
    # sky_ground_v2
    ("a rainbow in the sky",      "a red barn"),
    ("a crescent moon",           "a wooden fence"),
    ("white clouds in the sky",   "a sunflower field"),
    ("a flock of birds in the sky", "a stone bridge"),
    ("a hot air balloon in the sky", "a small cabin"),
    ("an airplane in the sky",    "a green tractor"),
    # orthogonal_object_style (from EXTRA_GROUPS — should rank low, act as sanity check)
    ("a dog",                     "oil painting style"),
    ("a lighthouse",              "watercolor style"),
    ("a bicycle",                 "sketch style"),
    ("a teapot",                  "claymation style"),
    # positive_object_scene (should rank high co-occurrence → Group 1)
    ("a butterfly",               "a flower meadow"),
    ("a camel",                   "a desert landscape"),
    ("a lion",                    "a savanna at sunset"),
    ("a dolphin",                 "an ocean wave"),
    # Extended novel candidates for Group 3
    ("a typewriter",              "a rubber duck"),
    ("a bathtub",                 "a hot air balloon"),
    ("a grand piano",             "a lighthouse"),
    ("a filing cabinet",          "a telescope"),
    ("a fire extinguisher",       "a windmill"),
    ("a parking meter",           "a submarine"),
    ("a hospital bed",            "a combine harvester"),
    ("a filing cabinet",          "a kite"),
    ("a rotary telephone",        "a hot air balloon"),
    ("a desk lamp",               "a glacier"),
    ("a chess board",             "a fishing boat"),
    ("a typewriter",              "a glacier"),
    ("a fire hydrant",            "a hot air balloon"),
    ("a park bench",              "a submarine"),
    ("a rocking chair",           "a lighthouse"),
    ("a shopping cart",           "a windmill"),
    ("a manhole cover",           "a blimp"),
    ("a vending machine",         "a canoe"),
    ("a church pew",              "a cargo ship"),
    ("a lab microscope",          "a hay bale"),
]


def embed_texts(
    texts: list[str],
    model: CLIPModel,
    tokenizer: CLIPTokenizer,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Return L2-normalised CLIP text embeddings, shape (N, D)."""
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            embeds = model.get_text_features(**inputs).float()
        embeds = F.normalize(embeds, dim=-1)
        all_embeds.append(embeds.cpu())
    return torch.cat(all_embeds, dim=0)


def build_joint_caption(a: str, b: str) -> str:
    """Simple conjunction for co-occurrence proximity estimate."""
    a_clean = a.strip().rstrip(".")
    b_clean = b.strip().rstrip(".")
    return f"{a_clean} and {b_clean}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Group 3 orthogonality heuristic.")
    parser.add_argument("--background", type=str, default="a photo",
                        help="Background text for score-delta origin.")
    parser.add_argument("--model", type=str, default="openai/clip-vit-large-patch14",
                        help="HuggingFace CLIP checkpoint.")
    parser.add_argument("--top", type=int, default=25,
                        help="Print the top-N Group 3 candidates.")
    parser.add_argument("--out", type=str, default="",
                        help="Output directory. Defaults to "
                             "experiments/eccv2026/orthogonality_heuristic/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device}...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(device).eval()

    # Deduplicate pairs
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for a, b in CANDIDATE_PAIRS:
        key = (a, b)
        if key not in seen:
            seen.add(key)
            pairs.append((a, b))

    # Build flat text list for a single forward pass
    background = args.background
    all_texts = [background]
    for a, b in pairs:
        all_texts.extend([a, b, build_joint_caption(a, b)])

    print(f"Embedding {len(all_texts)} strings ({len(pairs)} pairs + background)...")
    all_embeds = embed_texts(all_texts, model, tokenizer, device)

    bg_embed = all_embeds[0]  # shape (D,)

    results = []
    for i, (a, b) in enumerate(pairs):
        base = 1 + i * 3
        e_a    = all_embeds[base]       # embed(A)
        e_b    = all_embeds[base + 1]   # embed(B)
        e_joint = all_embeds[base + 2]  # embed("A and B")

        # Score deltas (already L2-normalised embeddings; delta need not be unit)
        delta_a = e_a - bg_embed
        delta_b = e_b - bg_embed

        # Normalise deltas for stable dot product
        delta_a_n = F.normalize(delta_a.unsqueeze(0), dim=-1).squeeze()
        delta_b_n = F.normalize(delta_b.unsqueeze(0), dim=-1).squeeze()

        orthogonality_dot = (delta_a_n @ delta_b_n).item()  # ≈ 0 is good

        # Direct concept similarity (without background subtraction)
        concept_sim = (e_a @ e_b).item()  # cosine sim (both are L2-normalised)

        # Co-occurrence proxy: how close is the joint caption embedding to the
        # individual concept embeddings?  High = likely in training distribution.
        cooccurrence_proxy = (
            (e_joint @ e_a).item() + (e_joint @ e_b).item()
        ) / 2.0

        # Group 3 score: want small |orthogonality_dot| AND low concept_sim
        # Penalise high co-occurrence (Group 1 intruders)
        group3_score = (
            -abs(orthogonality_dot)
            - 0.3 * concept_sim
            - 0.3 * cooccurrence_proxy
        )

        results.append({
            "concept_a":           a,
            "concept_b":           b,
            "orthogonality_dot":   round(orthogonality_dot, 4),
            "concept_sim":         round(concept_sim, 4),
            "cooccurrence_proxy":  round(cooccurrence_proxy, 4),
            "group3_score":        round(group3_score, 4),
            # label assigned after all scores are known (see post-processing below)
            "group3_label":        "pending",
        })

    # ------------------------------------------------------------------
    # Adaptive labelling: Group 3 threshold = floor + 0.08
    # CLIP text embeddings have a positivity floor (~0.25) for visual
    # concepts, so a strict |dot| < 0.15 threshold finds nothing.
    # We use the empirical minimum as the baseline.
    # ------------------------------------------------------------------
    all_dots = [abs(r["orthogonality_dot"]) for r in results]
    dot_min = min(all_dots)
    dot_max = max(all_dots)
    dot_floor = dot_min          # empirical floor
    group3_thresh = dot_floor + 0.08   # within 0.08 of the minimum

    for r in results:
        a_, b_ = r["concept_a"], r["concept_b"]
        is_style = any("style" in x or "painting" in x or "claymation" in x
                       for x in [a_, b_])
        is_indistr = r["cooccurrence_proxy"] > 0.82
        is_g3 = (abs(r["orthogonality_dot"]) <= group3_thresh
                 and not is_style and not is_indistr)
        is_g4 = r["orthogonality_dot"] > 0.50

        r["group3_label"] = (
            "Group1_indistr"    if is_indistr
            else "Group2_style" if is_style
            else "Group3_cand"  if is_g3
            else "Group4_coll"  if is_g4
            else "ambiguous"
        )

    # Sort by group3_score descending
    results.sort(key=lambda r: r["group3_score"], reverse=True)

    # Output directory
    project_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out) if args.out else (
        project_root / "experiments" / "eccv2026" / "orthogonality_heuristic"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump({"background": background, "pairs": results}, f, indent=2)

    # Save CSV
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w") as f:
        headers = ["concept_a", "concept_b", "orthogonality_dot",
                   "concept_sim", "cooccurrence_proxy", "group3_score", "group3_label"]
        f.write(",".join(headers) + "\n")
        for r in results:
            f.write(",".join(str(r[h]) for h in headers) + "\n")

    # Print table
    col_w = 32
    print()
    print(f"Background: '{background}'")
    print(f"CLIP text-embedding floor (min |orth_dot|): {dot_floor:.4f}")
    print(f"Group 3 threshold (floor + 0.08):           {group3_thresh:.4f}")
    print()
    print(f"{'#':<4} {'Concept A':<{col_w}} {'Concept B':<{col_w}} "
          f"{'orth_dot':>9} {'conc_sim':>9} {'cooccur':>8} {'g3_score':>9}  label")
    print("-" * 125)
    for rank, r in enumerate(results[:args.top], 1):
        marker = " <<<" if r["group3_label"] == "Group3_cand" else ""
        print(
            f"{rank:<4} {r['concept_a']:<{col_w}} {r['concept_b']:<{col_w}}"
            f" {r['orthogonality_dot']:>9.4f} {r['concept_sim']:>9.4f}"
            f" {r['cooccurrence_proxy']:>8.4f} {r['group3_score']:>9.4f}"
            f"  {r['group3_label']}{marker}"
        )

    g3 = [r for r in results if r["group3_label"] == "Group3_cand"]
    print()
    print(f"Group 3 candidates ({len(g3)} found):")
    for r in g3:
        print(f"  orth={r['orthogonality_dot']:.3f}  "
              f"{r['concept_a']}  +  {r['concept_b']}")
    print()
    print(f"Full results saved to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
