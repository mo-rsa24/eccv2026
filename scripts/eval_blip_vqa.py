#!/usr/bin/env python3
"""
BLIP-VQA concept-presence evaluation.

For each (pair, seed, condition) image, records P(c1 present) and P(c2 present)
using Salesforce/blip-vqa-base.  Saves results to {data_dir}/blip_vqa_scores.json.

Question template: "Is there {concept} in the image? Answer yes or no."
P("yes") is computed as softmax over the first-token logits for tokens "yes" vs "no".

Usage
-----
python scripts/eval_blip_vqa.py \\
    --data-dir experiments/inversion/gap_analysis/<run> \\
    [--device cuda] [--batch-size 8] \\
    [--conditions mono poe c1 c2 pstar_sdipc]
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor

# Make scripts/plots/ importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plots.utils import enrich_taxonomy_dataframe
from taxonomy_manifest import get_pair_taxonomy_from_slug

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONDITIONS = ["c1", "c2", "mono", "poe", "pstar_sdipc"]

# Canonical mapping from evaluator condition names to image / grid-asset names.
CONDITION_ALIASES = {
    "c1": ["c1", "c1_only", "prompt_a", "solo_a"],
    "c2": ["c2", "c2_only", "prompt_b", "solo_b"],
    "mono": ["mono", "monolithic"],
    "poe": ["poe"],
    "pstar_sdipc": ["pstar_sdipc"],
}

MODEL_ID = "Salesforce/blip-vqa-base"
GRID_PADDING = 2
GRID_NROW = 4


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device: str):
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    return processor, model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _p_yes(logits_first_token: torch.Tensor, processor: BlipProcessor) -> float:
    """Return P("yes") via softmax over logits for "yes" vs "no" first-token ids."""
    vocab = processor.tokenizer
    yes_id = vocab.convert_tokens_to_ids("yes")
    no_id  = vocab.convert_tokens_to_ids("no")
    selected = logits_first_token[:, [yes_id, no_id]]   # (batch, 2)
    probs = torch.softmax(selected.float(), dim=-1)
    return float(probs[:, 0].mean().item())


@torch.no_grad()
def score_batch(
    images: list[Image.Image],
    questions: list[str],
    processor: BlipProcessor,
    model: BlipForQuestionAnswering,
    device: str,
) -> list[float]:
    """Return P("yes") for each (image, question) pair."""
    inputs = processor(images=images, text=questions, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # transformers>=5 returns BlipTextVisionModelOutput from forward(), which
    # no longer exposes decoder logits. For VQA scoring we need the first-step
    # decoder logits explicitly, so run the BLIP submodules directly.
    vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
    image_embeds = vision_outputs.last_hidden_state
    image_attention_mask = torch.ones(
        image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
    )

    question_outputs = model.text_encoder(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_attention_mask,
    )
    question_embeds = question_outputs[0]

    decoder_input_ids = inputs["input_ids"][:, :1]
    decoder_attention_mask = torch.ones_like(decoder_input_ids, device=decoder_input_ids.device)
    answer_output = model.text_decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=question_embeds,
        encoder_attention_mask=inputs.get("attention_mask"),
    )

    first_token_logits = answer_output.logits[:, 0, :]   # (batch, vocab)
    vocab = processor.tokenizer
    yes_id = vocab.convert_tokens_to_ids("yes")
    no_id  = vocab.convert_tokens_to_ids("no")
    selected = first_token_logits[:, [yes_id, no_id]]
    probs = torch.softmax(selected.float(), dim=-1)
    return probs[:, 0].tolist()   # P("yes") per image


# ---------------------------------------------------------------------------
# Image discovery / grid recovery
# ---------------------------------------------------------------------------

def _load_pair_asset(pair_dir: Path) -> dict:
    asset_path = pair_dir / "grid_assets.json"
    if not asset_path.exists():
        return {}
    return json.loads(asset_path.read_text())


def _seed_from_asset(asset: dict) -> int | None:
    if isinstance(asset.get("seed"), int):
        return int(asset["seed"])
    seeds = asset.get("seeds")
    if isinstance(seeds, list) and seeds:
        return int(seeds[0])
    return None


def _parse_grid_manifest(manifest_path: Path) -> tuple[int, int, list[int]]:
    text = manifest_path.read_text()
    layout_match = re.search(r"Layout\s*:\s*(\d+)\s+row\(s\)\s×\s(\d+)\s+columns", text)
    if not layout_match:
        raise RuntimeError(f"Could not parse grid layout from {manifest_path}")
    n_rows = int(layout_match.group(1))
    n_cols = int(layout_match.group(2))

    seeds_match = re.search(r"Seeds\s*:\s*(\[[^\n]+\])", text)
    if not seeds_match:
        raise RuntimeError(f"Could not parse seed list from {manifest_path}")
    try:
        seeds = [int(v) for v in ast.literal_eval(seeds_match.group(1))]
    except (SyntaxError, ValueError) as exc:
        raise RuntimeError(f"Could not parse seed list in {manifest_path}: {exc}") from exc

    rowcol_matches = re.findall(r"Row\s+(\d+),\s+Col\s+(\d+)\s+:\s+seed\s+(-?\d+)", text)
    if len(rowcol_matches) != len(seeds):
        raise RuntimeError(
            f"Manifest mismatch in {manifest_path}: listed {len(seeds)} seeds but found "
            f"{len(rowcol_matches)} row/col assignments"
        )

    ordered_by_index: list[int | None] = [None] * len(seeds)
    for row_s, col_s, seed_s in rowcol_matches:
        row = int(row_s)
        col = int(col_s)
        idx = row * GRID_NROW + col
        if idx >= len(ordered_by_index):
            raise RuntimeError(f"Manifest cell index out of range in {manifest_path}: row={row} col={col}")
        ordered_by_index[idx] = int(seed_s)
    if any(seed is None for seed in ordered_by_index):
        raise RuntimeError(f"Manifest index coverage incomplete in {manifest_path}")

    return n_rows, n_cols, [int(seed) for seed in ordered_by_index]


def _grid_cell_boxes(grid_path: Path, n_rows: int, n_cols: int) -> list[tuple[int, int, int, int]]:
    with Image.open(grid_path) as img:
        width, height = img.size

    cell_w_num = width - GRID_PADDING * (n_cols + 1)
    cell_h_num = height - GRID_PADDING * (n_rows + 1)
    if cell_w_num <= 0 or cell_h_num <= 0:
        raise RuntimeError(f"Invalid grid dimensions in {grid_path}: {width}x{height}")
    if cell_w_num % n_cols != 0 or cell_h_num % n_rows != 0:
        raise RuntimeError(
            f"Grid {grid_path} is not consistent with padding={GRID_PADDING}, "
            f"rows={n_rows}, cols={n_cols}"
        )

    cell_w = cell_w_num // n_cols
    cell_h = cell_h_num // n_rows
    boxes = []
    for idx in range(n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        left = GRID_PADDING + col * (cell_w + GRID_PADDING)
        top = GRID_PADDING + row * (cell_h + GRID_PADDING)
        boxes.append((left, top, left + cell_w, top + cell_h))
    return boxes


def _grid_sources_for_condition(grid_path: Path, manifest_path: Path) -> dict[int, dict]:
    n_rows, n_cols, ordered_seeds = _parse_grid_manifest(manifest_path)
    expected_rows = math.ceil(len(ordered_seeds) / GRID_NROW)
    if n_rows != expected_rows:
        raise RuntimeError(
            f"Manifest layout mismatch in {manifest_path}: expected {expected_rows} rows "
            f"for {len(ordered_seeds)} seeds with nrow={GRID_NROW}, got {n_rows}"
        )
    boxes = _grid_cell_boxes(grid_path, n_rows, n_cols)
    if len(boxes) < len(ordered_seeds):
        raise RuntimeError(
            f"Crop box mismatch in {grid_path}: {len(boxes)} boxes for {len(ordered_seeds)} seeds"
        )

    return {
        seed: {"kind": "grid_crop", "path": grid_path, "box": boxes[idx]}
        for idx, seed in enumerate(ordered_seeds)
    }


def _find_images_for_pair(pair_dir: Path, conditions: list[str]) -> dict[str, dict[int, dict]]:
    """
    Return {condition: {seed: source_spec}} for a single pair directory.

    Strategy:
    1. Direct per-seed files under images/
    2. Batched condition grid + manifest under images/
    3. Single-seed fallback via grid_assets.json
    """
    result: dict[str, dict[int, dict]] = {c: {} for c in conditions}
    asset = _load_pair_asset(pair_dir)
    asset_seed = _seed_from_asset(asset)

    images_dir = pair_dir / "images"
    if images_dir.is_dir():
        for img_path in sorted(images_dir.glob("*.png")):
            stem = img_path.stem
            for cond in conditions:
                aliases = CONDITION_ALIASES.get(cond, [cond])
                for alias in aliases:
                    for prefix in (alias, f"sd14_{alias}", f"sd35_{alias}", f"sdxl_{alias}"):
                        if stem.startswith(prefix + "_"):
                            seed_str = stem[len(prefix) + 1:]
                            try:
                                seed = int(seed_str)
                                result[cond][seed] = {"kind": "path", "path": img_path}
                            except ValueError:
                                pass
                    if result[cond]:
                        break

        for cond in conditions:
            if result[cond]:
                continue
            aliases = CONDITION_ALIASES.get(cond, [cond])
            for alias in aliases:
                for prefix in (alias, f"sd14_{alias}", f"sd35_{alias}"):
                    manifest_path = images_dir / f"{prefix}_manifest.txt"
                    grid_path = images_dir / f"{prefix}.png"
                    if manifest_path.exists() and grid_path.exists():
                        result[cond] = _grid_sources_for_condition(grid_path, manifest_path)
                        break
                if result[cond]:
                    break

    # Fallback: grid_assets.json
    if asset:
        decoded = asset.get("decoded_image_paths", {})
        seeds_list = asset.get("seeds", [])
        for cond in conditions:
            if result[cond]:
                continue
            aliases = CONDITION_ALIASES.get(cond, [cond])
            for alias in aliases:
                rel_path = decoded.get(alias)
                if rel_path is None:
                    continue
                full_path = pair_dir / rel_path
                if not full_path.exists():
                    continue
                stem = full_path.stem
                matched = False
                for prefix in (alias, f"sd14_{alias}", f"sd35_{alias}"):
                    if stem == prefix:
                        matched = True
                        break
                    if stem.startswith(prefix + "_"):
                        seed_str = stem[len(prefix) + 1:]
                        try:
                            seed = int(seed_str)
                            result[cond][seed] = {"kind": "path", "path": full_path}
                            matched = True
                            break
                        except ValueError:
                            pass
                if matched:
                    if not result[cond]:
                        if isinstance(asset_seed, int):
                            result[cond][int(asset_seed)] = {"kind": "path", "path": full_path}
                        elif seeds_list:
                            result[cond][int(seeds_list[0])] = {"kind": "path", "path": full_path}
                        else:
                            result[cond][0] = {"kind": "path", "path": full_path}
                    break

    # Stage 0b qualitative fallback: direct PNGs in the pair directory itself.
    direct_name_by_cond = {
        "c1": "solo_a.png",
        "c2": "solo_b.png",
        "mono": "monolithic.png",
        "poe": "poe.png",
        "pstar_sdipc": "pstar_sdipc.png",
    }
    for cond in conditions:
        if result[cond]:
            continue
        png_name = direct_name_by_cond.get(cond)
        if not png_name:
            continue
        full_path = pair_dir / png_name
        if full_path.exists():
            seed = int(asset_seed) if isinstance(asset_seed, int) else 0
            result[cond][seed] = {"kind": "path", "path": full_path}

    return result


def _iter_pair_dirs(data_dir: Path) -> list[Path]:
    pairs_root = data_dir / "pairs"
    if pairs_root.exists():
        return sorted(p for p in pairs_root.iterdir() if p.is_dir())

    multiseed_pair_dirs: list[Path] = []
    seed_roots = sorted(
        path for path in data_dir.iterdir()
        if path.is_dir() and path.name.startswith("seed_")
    )
    if seed_roots:
        for seed_root in seed_roots:
            for group_dir in sorted(path for path in seed_root.iterdir() if path.is_dir()):
                if not group_dir.name.startswith("group"):
                    continue
                for pair_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
                    if any((pair_dir / f"{name}.png").exists() for name in ("solo_a", "solo_b", "monolithic", "poe")):
                        multiseed_pair_dirs.append(pair_dir)
        if multiseed_pair_dirs:
            return multiseed_pair_dirs

    direct_pair_dirs: list[Path] = []
    for group_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if not group_dir.name.startswith("group"):
            continue
        for pair_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            if any((pair_dir / f"{name}.png").exists() for name in ("solo_a", "solo_b", "monolithic", "poe")):
                direct_pair_dirs.append(pair_dir)
    return direct_pair_dirs


def _infer_pair_metadata(pair_dir: Path) -> tuple[list[str] | None, str | None]:
    asset_path = pair_dir / "grid_assets.json"
    if asset_path.exists():
        asset = json.loads(asset_path.read_text())
        pair_concepts = asset.get("pair")
        taxonomy_group_key = asset.get("taxonomy_group_key")
        if pair_concepts is not None and len(pair_concepts) >= 2:
            return pair_concepts, taxonomy_group_key

    taxonomy_group_key = pair_dir.parent.name if pair_dir.parent.name.startswith("group") else None
    meta = get_pair_taxonomy_from_slug(pair_dir.name)
    if meta is not None:
        return [meta["prompt_a"], meta["prompt_b"]], taxonomy_group_key or meta.get("taxonomy_group_key")
    return None, taxonomy_group_key


def _load_image_from_source(source: dict, cache: dict[Path, Image.Image]) -> Image.Image:
    path = Path(source["path"])
    kind = source["kind"]
    if kind == "path":
        return Image.open(path).convert("RGB")
    if kind == "grid_crop":
        if path not in cache:
            cache[path] = Image.open(path).convert("RGB")
        return cache[path].crop(tuple(source["box"]))
    raise ValueError(f"Unknown image source kind: {kind}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    data_dir: Path,
    conditions: list[str],
    device: str,
    batch_size: int,
    pairs_filter: list[str] | None,
) -> list[dict]:
    pair_dirs = _iter_pair_dirs(data_dir)
    if pairs_filter:
        pair_dirs = [p for p in pair_dirs if p.name in pairs_filter]

    if not pair_dirs:
        sys.exit(f"No pair directories found under {data_dir}")

    print(f"Loading model {MODEL_ID} ...")
    processor, model = load_model(device)

    # Collect all (source_spec, question, meta) tuples for batching
    tasks: list[tuple[dict, str | None, dict]] = []

    for pair_dir in pair_dirs:
        # Read pair metadata from grid_assets.json
        pair_concepts, taxonomy_group_key = _infer_pair_metadata(pair_dir)
        if pair_concepts is None or len(pair_concepts) < 2:
            print(f"  Warning: cannot determine pair concepts for {pair_dir.name}, skipping")
            continue

        c1, c2 = pair_concepts[0], pair_concepts[1]
        pair_str = f"{c1} + {c2}"
        pair_slug = pair_dir.name

        cond_images = _find_images_for_pair(pair_dir, conditions)

        # Skip pstar_sdipc if no images found (optional condition)
        for cond in conditions:
            if cond == "pstar_sdipc" and not cond_images.get(cond):
                continue
            seed_map = cond_images.get(cond, {})
            if not seed_map:
                print(f"  Warning: no images for condition '{cond}' in {pair_dir.name}")
                continue
            for seed, source in sorted(seed_map.items()):
                tasks.append((
                    source,
                    None,   # placeholder — questions built below
                    {
                        "pair": pair_str,
                        "c1": c1,
                        "c2": c2,
                        "pair_slug": pair_slug,
                        "taxonomy_group_key": taxonomy_group_key,
                        "seed": seed,
                        "condition": cond,
                    }
                ))

    if not tasks:
        print("No tasks found. Check that images exist under pairs/*/images/")
        return []

    print(f"Found {len(tasks)} (pair, seed, condition) evaluation tasks")

    # Build questions for each task
    enriched_tasks = []
    for source, _, meta in tasks:
        q_c1 = f"Is there {meta['c1']} in the image? Answer yes or no."
        q_c2 = f"Is there {meta['c2']} in the image? Answer yes or no."
        enriched_tasks.append((source, q_c1, q_c2, meta))

    # Evaluate in batches
    records: list[dict] = []
    n = len(enriched_tasks)
    image_cache: dict[Path, Image.Image] = {}
    for i in range(0, n, batch_size):
        batch = enriched_tasks[i : i + batch_size]
        imgs   = [_load_image_from_source(t[0], image_cache) for t in batch]
        q1s    = [t[1] for t in batch]
        q2s    = [t[2] for t in batch]
        metas  = [t[3] for t in batch]

        p_c1_list = score_batch(imgs, q1s, processor, model, device)
        p_c2_list = score_batch(imgs, q2s, processor, model, device)

        for meta, p_c1, p_c2 in zip(metas, p_c1_list, p_c2_list):
            records.append({
                **meta,
                "p_c1": round(p_c1, 6),
                "p_c2": round(p_c2, 6),
            })

        if (i // batch_size) % 10 == 0:
            print(f"  [{i + len(batch)}/{n}] ...")
        for img in imgs:
            img.close()

    for img in image_cache.values():
        img.close()

    return records


# ---------------------------------------------------------------------------
# Taxonomy enrichment
# ---------------------------------------------------------------------------

def enrich_records(records: list[dict]) -> list[dict]:
    if not records:
        return records
    df = pd.DataFrame(records)
    df = enrich_taxonomy_dataframe(df)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BLIP-VQA concept-presence evaluation for gap-analysis images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir",
        default="experiments/inversion/gap_analysis",
        help="Gap-analysis run root containing pairs/*/images/ or pairs/*/grid_assets.json.",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (default: cuda if available, else cpu).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="Number of (image, question) pairs per forward pass (default: 8).",
    )
    p.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        metavar="COND",
        help=(
            f"Conditions to evaluate (default: {' '.join(DEFAULT_CONDITIONS)}). "
            "pstar_sdipc is skipped silently if no images are found."
        ),
    )
    p.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        metavar="SLUG",
        help="Restrict to specific pair slugs (directory names under pairs/).",
    )
    p.add_argument(
        "--output",
        default="",
        help="Output JSON path (default: {data_dir}/blip_vqa_scores.json).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_path = Path(args.output) if args.output else data_dir / "blip_vqa_scores.json"

    records = evaluate(
        data_dir=data_dir,
        conditions=args.conditions,
        device=args.device,
        batch_size=args.batch_size,
        pairs_filter=args.pairs,
    )

    records = enrich_records(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))
    print(f"\nSaved {len(records)} records to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
