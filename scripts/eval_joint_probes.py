#!/usr/bin/env python3
"""
Pair-type-aware BLIP-VQA evaluation for joint correctness.

This script keeps BLIP-VQA as the backbone model but changes the measurement
protocol. Singleton conditions (c1 / c2) are scored against their intended
marginal target, while joint conditions (mono / poe / pstar_*) are scored with
pair-type-aware probes that can penalize hybrid collapse.

Outputs {data_dir}/joint_probe_scores.json with:
  - probe_records: one record per (pair, seed, condition, probe)
  - image_scores: one aggregate record per (pair, seed, condition)

Usage
-----
python scripts/eval_joint_probes.py \
    --data-dir experiments/inversion/gap_analysis/<run> \
    [--device cuda] [--batch-size 8] \
    [--conditions c1 c2 mono poe pstar_sdipc]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

# Make scripts/ importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_blip_vqa import (  # reuse image discovery + BLIP scoring path
    DEFAULT_CONDITIONS,
    MODEL_ID,
    _find_images_for_pair,
    _infer_pair_metadata,
    _iter_pair_dirs,
    _load_image_from_source,
    load_model,
    score_batch,
)
from plots.utils import enrich_taxonomy_dataframe

PAIR_TYPE_BY_GROUP = {
    "group1_cooccurrence": "object_object",
    "group2_disentangled": "factorized",
    "group3_feature_overlap": "overlap",
    "group4_coherent_collision": "collision",
    # Legacy qualitative directory names.
    "group3_ood": "overlap",
    "group4_collision": "collision",
}

SEMANTIC_PASS_THRESHOLD = 0.58
HIGH_CONFIDENCE_PASS_THRESHOLD = 0.72
LOW_CONFIDENCE_THRESHOLD = 0.52
HYBRIDIZATION_FAILURE_THRESHOLD = 0.50
OMISSION_FAILURE_THRESHOLD = 0.45
SEMANTIC_DRIFT_GATE_THRESHOLD = 0.70

def pair_type_for_group(group_key: str | None) -> str:
    return PAIR_TYPE_BY_GROUP.get(group_key or "", "object_object")


def _probe_spec(meta: dict) -> list[dict]:
    condition = meta["condition"]
    c1 = meta["c1"]
    c2 = meta["c2"]
    pair_type = meta["pair_type"]

    if condition == "c1":
        return [
            {
                "probe": "a_present",
                "question": f"Is there {c1} in the image? Answer yes or no.",
                "positive": True,
                "component": "cue_presence",
            }
        ]
    if condition == "c2":
        return [
            {
                "probe": "b_present",
                "question": f"Is there {c2} in the image? Answer yes or no.",
                "positive": True,
                "component": "cue_presence",
            }
        ]

    if pair_type == "factorized":
        return [
            {
                "probe": "content_preserved",
                "question": f"Does the image depict {c1}? Answer yes or no.",
                "positive": True,
                "component": "cue_presence",
            },
            {
                "probe": "factor_applied",
                "question": f"Does the image also realize {c2}? Answer yes or no.",
                "positive": True,
                "component": "joint",
            },
        ]

    probes = [
        {
            "probe": "a_present",
            "question": f"Is there {c1} in the image? Answer yes or no.",
            "positive": True,
            "component": "cue_presence",
        },
        {
            "probe": "b_present",
            "question": f"Is there {c2} in the image? Answer yes or no.",
            "positive": True,
            "component": "cue_presence",
        },
        {
            "probe": "distinct_entities",
            "question": (
                f"Are there two distinct things in the image, one {c1} and one {c2}, "
                "rather than a single merged object? Answer yes or no."
            ),
            "positive": True,
            "component": "joint",
        },
        {
            "probe": "both_present",
            "question": (
                f"Does the image show both {c1} and {c2} at the same time? "
                "Answer yes or no."
            ),
            "positive": True,
            "component": "joint",
        },
        {
            "probe": "merged_or_absorbed",
            "question": (
                f"Is one of {c1} or {c2} merged into or absorbed by the other, "
                "rather than appearing separately? Answer yes or no."
            ),
            "positive": False,
            "component": "anti_collapse",
        },
        {
            "probe": "one_missing",
            "question": (
                f"Is one of {c1} or {c2} missing from the image? Answer yes or no."
            ),
            "positive": False,
            "component": "anti_omission",
        },
        {
            "probe": "intentional_composition",
            "question": (
                f"Does the image appear to be an intentional composition of {c1} and {c2}, "
                f"rather than incidental placement or accidental co-occurrence? Answer yes or no."
            ),
            "positive": False,
            "component": "anti_accident",
        },
    ]
    if pair_type in {"overlap", "collision"}:
        probes.append(
            {
                "probe": "dominance_failure",
                "question": (
                    f"Does one concept dominate the image so strongly that the other is not "
                    f"independently realized as {c1} and {c2}? Answer yes or no."
                ),
                "positive": False,
                "component": "anti_collapse",
            }
        )
        probes.append(
            {
                "probe": "hybrid_object",
                "question": (
                    f"Does the image depict a single fused or hybrid object combining "
                    f"{c1} and {c2}? Answer yes or no."
                ),
                "positive": False,
                "component": "anti_hybrid",
            }
        )
        probes.append(
            {
                "probe": "concept_confusion",
                "question": (
                    f"Could the objects or elements in the image be confused or misidentified "
                    f"as something other than {c1} and {c2}? Answer yes or no."
                ),
                "positive": False,
                "component": "anti_confusion",
            }
        )
        probes.append(
            {
                "probe": "c1_not_subsumed_by_c2",
                "question": (
                    f"Could {c1} in the image be described as just a feature or attribute "
                    f"of {c2}, rather than its own separate thing? Answer yes or no."
                ),
                "positive": False,
                "component": "anti_semantic_drift",
            }
        )
        probes.append(
            {
                "probe": "semantic_role_correct",
                "question": (
                    f"Is the main subject of the image something other than {c1} and {c2}? "
                    "Answer yes or no."
                ),
                "positive": False,
                "component": "anti_semantic_drift",
            }
        )
        probes.append(
            {
                "probe": "c1_independent_realization",
                "question": (
                    f"Is {c1} shown in the image as its own distinct object or element, "
                    f"separate from {c2}? Answer yes or no."
                ),
                "positive": True,
                "component": "cue_presence",
            }
        )
    return probes


def _aggregate_scores(meta: dict, scored_probes: list[dict]) -> dict:
    condition = meta["condition"]
    pair_type = meta["pair_type"]

    score_by_probe = {row["probe"]: float(row["score"]) for row in scored_probes}
    anti_collapse_scores = [
        1.0 - float(row["score"])
        for row in scored_probes
        if not row["probe_positive"]
    ]

    cue_presence_score = None
    if condition == "c1":
        cue_presence_score = score_by_probe.get("a_present")
    elif condition == "c2":
        cue_presence_score = score_by_probe.get("b_present")
    elif pair_type == "factorized":
        cue_presence_score = score_by_probe.get("content_preserved")
    else:
        cue_terms = [score_by_probe[p] for p in ("a_present", "b_present") if p in score_by_probe]
        cue_presence_score = sum(cue_terms) / len(cue_terms) if cue_terms else None

    anti_collapse_score = None
    if anti_collapse_scores:
        anti_collapse_score = sum(anti_collapse_scores) / len(anti_collapse_scores)

    omission_score = None
    if "one_missing" in score_by_probe:
        omission_score = 1.0 - score_by_probe["one_missing"]

    hybrid_score = None
    hybrid_terms = []
    if "hybrid_object" in score_by_probe:
        hybrid_terms.append(1.0 - score_by_probe["hybrid_object"])
    if "merged_or_absorbed" in score_by_probe:
        hybrid_terms.append(1.0 - score_by_probe["merged_or_absorbed"])
    if hybrid_terms:
        hybrid_score = sum(hybrid_terms) / len(hybrid_terms)

    if condition in {"c1", "c2"}:
        joint_correctness_score = cue_presence_score
    elif pair_type == "factorized":
        terms = [score_by_probe[p] for p in ("content_preserved", "factor_applied") if p in score_by_probe]
        joint_correctness_score = sum(terms) / len(terms) if terms else None
    else:
        # Positive presence probes (includes new c1_independent_realization)
        positive_probe_names = (
            "a_present", "b_present", "distinct_entities",
            "both_present", "c1_independent_realization",
        )
        presence_terms = [score_by_probe[p] for p in positive_probe_names if p in score_by_probe]
        presence_score = sum(presence_terms) / len(presence_terms) if presence_terms else None

        # Tier 1: Structural anti-collapse (averaged — uniformly high for good images)
        tier1_names = ("merged_or_absorbed", "one_missing", "dominance_failure", "hybrid_object")
        tier1_terms = [score_by_probe[p] for p in tier1_names if p in score_by_probe]
        tier1_quality = sum(tier1_terms) / len(tier1_terms) if tier1_terms else None

        # Tier 2: Semantic drift (min — the weakest is most diagnostic)
        tier2_names = (
            "concept_confusion", "intentional_composition",
            "c1_not_subsumed_by_c2", "semantic_role_correct",
        )
        tier2_terms = [score_by_probe[p] for p in tier2_names if p in score_by_probe]
        tier2_quality = min(tier2_terms) if tier2_terms else None

        # Gate: if any drift probe fires, disable tier1 blending
        semantic_drift_flagged = (
            tier2_quality is not None and tier2_quality < SEMANTIC_DRIFT_GATE_THRESHOLD
        )
        if tier2_quality is not None and tier1_quality is not None:
            if semantic_drift_flagged:
                quality_score = tier2_quality         # tier1 cannot rescue a drifted image
            else:
                quality_score = 0.5 * tier1_quality + 0.5 * tier2_quality
        elif tier2_quality is not None:
            quality_score = tier2_quality
        elif tier1_quality is not None:
            quality_score = tier1_quality
        else:
            quality_score = None

        # Joint correctness: 40% presence, 60% quality
        if presence_score is not None and quality_score is not None:
            joint_correctness_score = 0.40 * presence_score + 0.60 * quality_score
        elif presence_score is not None:
            joint_correctness_score = presence_score
        else:
            joint_correctness_score = quality_score

    semantic_pass = None
    high_confidence_pass = None
    low_confidence = None
    if joint_correctness_score is not None:
        semantic_pass = joint_correctness_score >= SEMANTIC_PASS_THRESHOLD
        high_confidence_pass = joint_correctness_score >= HIGH_CONFIDENCE_PASS_THRESHOLD
        low_confidence = joint_correctness_score < LOW_CONFIDENCE_THRESHOLD

    omission_failure = None
    if omission_score is not None:
        omission_failure = omission_score < OMISSION_FAILURE_THRESHOLD

    hybridization_failure = None
    if hybrid_score is not None:
        hybridization_failure = hybrid_score < HYBRIDIZATION_FAILURE_THRESHOLD

    result = {
        "cue_presence_score": round(cue_presence_score, 6) if cue_presence_score is not None else None,
        "anti_collapse_score": round(anti_collapse_score, 6) if anti_collapse_score is not None else None,
        "omission_score": round(omission_score, 6) if omission_score is not None else None,
        "hybrid_score": round(hybrid_score, 6) if hybrid_score is not None else None,
        "joint_correctness_score": round(joint_correctness_score, 6) if joint_correctness_score is not None else None,
        "paper_score": round(joint_correctness_score, 6) if joint_correctness_score is not None else None,
        "semantic_pass": semantic_pass,
        "high_confidence_pass": high_confidence_pass,
        "low_confidence": low_confidence,
        "hybridization_failure": hybridization_failure,
        "omission_failure": omission_failure,
        "n_probes": len(scored_probes),
    }
    # Add tiered quality diagnostic fields for overlap/collision pairs
    if pair_type in {"overlap", "collision"}:
        result["semantic_drift_flagged"] = semantic_drift_flagged if pair_type in {"overlap", "collision"} else None
        result["tier1_quality_score"] = round(tier1_quality, 6) if pair_type in {"overlap", "collision"} and tier1_quality is not None else None
        result["tier2_quality_score"] = round(tier2_quality, 6) if pair_type in {"overlap", "collision"} and tier2_quality is not None else None

    return result


def _pair_group_fit(row: dict) -> tuple[str, float]:
    joint = row.get("mean_joint_correctness_score")
    hybrid_rate = row.get("hybridization_failure_rate")
    omission_rate = row.get("omission_failure_rate")
    if joint is None:
        return "ambiguous", 0.0
    joint = float(joint)
    hybrid_rate = float(hybrid_rate or 0.0)
    omission_rate = float(omission_rate or 0.0)

    if joint >= 0.75 and hybrid_rate <= 0.20:
        return "group1_cooccurrence", joint - hybrid_rate
    if joint >= 0.60 and hybrid_rate <= 0.30:
        return "group2_disentangled", joint - 0.5 * hybrid_rate
    if omission_rate >= hybrid_rate:
        return "group3_feature_overlap", omission_rate + 0.5 * (1.0 - joint)
    if hybrid_rate > omission_rate:
        return "group4_coherent_collision", hybrid_rate + 0.5 * (1.0 - joint)
    return "ambiguous", 0.0


def summarize_pairs(image_scores: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in image_scores:
        grouped[(row["pair_slug"], row["condition"])].append(row)

    summaries: list[dict] = []
    for (pair_slug, condition), rows in grouped.items():
        joint_scores = [float(r["joint_correctness_score"]) for r in rows if r.get("joint_correctness_score") is not None]
        if not joint_scores:
            continue
        cue_scores = [float(r["cue_presence_score"]) for r in rows if r.get("cue_presence_score") is not None]
        anti_scores = [float(r["anti_collapse_score"]) for r in rows if r.get("anti_collapse_score") is not None]
        hybrid_scores = [float(r["hybrid_score"]) for r in rows if r.get("hybrid_score") is not None]
        omission_scores = [float(r["omission_score"]) for r in rows if r.get("omission_score") is not None]
        semantic_passes = [bool(r["semantic_pass"]) for r in rows if r.get("semantic_pass") is not None]
        high_conf_passes = [bool(r["high_confidence_pass"]) for r in rows if r.get("high_confidence_pass") is not None]
        low_conf = [bool(r["low_confidence"]) for r in rows if r.get("low_confidence") is not None]
        hybrid_fail = [bool(r["hybridization_failure"]) for r in rows if r.get("hybridization_failure") is not None]
        omission_fail = [bool(r["omission_failure"]) for r in rows if r.get("omission_failure") is not None]

        summary = {
            "pair": rows[0]["pair"],
            "c1": rows[0]["c1"],
            "c2": rows[0]["c2"],
            "pair_slug": pair_slug,
            "taxonomy_group_key": rows[0].get("taxonomy_group_key"),
            "taxonomy_group_label": rows[0].get("taxonomy_group_label"),
            "pair_type": rows[0]["pair_type"],
            "condition": condition,
            "n_seeds": len(rows),
            "mean_joint_correctness_score": round(sum(joint_scores) / len(joint_scores), 6),
            "median_joint_correctness_score": round(float(statistics.median(joint_scores)), 6),
            "semantic_pass_rate": round(sum(semantic_passes) / len(semantic_passes), 6) if semantic_passes else None,
            "high_confidence_pass_rate": round(sum(high_conf_passes) / len(high_conf_passes), 6) if high_conf_passes else None,
            "low_confidence_rate": round(sum(low_conf) / len(low_conf), 6) if low_conf else None,
            "hybridization_failure_rate": round(sum(hybrid_fail) / len(hybrid_fail), 6) if hybrid_fail else None,
            "omission_failure_rate": round(sum(omission_fail) / len(omission_fail), 6) if omission_fail else None,
            "mean_cue_presence_score": round(sum(cue_scores) / len(cue_scores), 6) if cue_scores else None,
            "mean_anti_collapse_score": round(sum(anti_scores) / len(anti_scores), 6) if anti_scores else None,
            "mean_hybrid_score": round(sum(hybrid_scores) / len(hybrid_scores), 6) if hybrid_scores else None,
            "mean_omission_score": round(sum(omission_scores) / len(omission_scores), 6) if omission_scores else None,
            "semantic_pass_threshold": SEMANTIC_PASS_THRESHOLD,
            "high_confidence_pass_threshold": HIGH_CONFIDENCE_PASS_THRESHOLD,
        }
        if condition in {"mono", "poe", "pstar_sdipc"}:
            recommended_group, fit_score = _pair_group_fit(summary)
            summary["recommended_taxonomy_group_key"] = recommended_group
            summary["recommended_group_fit_score"] = round(float(fit_score), 6)
            summary["group_matches_recommendation"] = (
                summary.get("taxonomy_group_key") == recommended_group
                if recommended_group != "ambiguous"
                else None
            )
        summaries.append(summary)
    return summaries


def evaluate(
    data_dir: Path,
    conditions: list[str],
    device: str,
    batch_size: int,
    pairs_filter: list[str] | None,
) -> tuple[list[dict], list[dict]]:
    pair_dirs = _iter_pair_dirs(data_dir)
    if pairs_filter:
        pair_dirs = [p for p in pair_dirs if p.name in pairs_filter]
    if not pair_dirs:
        sys.exit(f"No pair directories found under {data_dir}")

    print(f"Loading model {MODEL_ID} ...")
    processor, model = load_model(device)

    image_tasks: list[dict] = []
    for pair_dir in pair_dirs:
        pair_concepts, taxonomy_group_key = _infer_pair_metadata(pair_dir)
        if pair_concepts is None or len(pair_concepts) < 2:
            print(f"  Warning: cannot determine pair concepts for {pair_dir.name}, skipping")
            continue

        c1, c2 = pair_concepts[0], pair_concepts[1]
        pair_type = pair_type_for_group(taxonomy_group_key)
        cond_images = _find_images_for_pair(pair_dir, conditions)

        for cond in conditions:
            if cond == "pstar_sdipc" and not cond_images.get(cond):
                continue
            seed_map = cond_images.get(cond, {})
            if not seed_map:
                print(f"  Warning: no images for condition '{cond}' in {pair_dir.name}")
                continue
            for seed, source in sorted(seed_map.items()):
                meta = {
                    "pair": f"{c1} + {c2}",
                    "c1": c1,
                    "c2": c2,
                    "pair_slug": pair_dir.name,
                    "taxonomy_group_key": taxonomy_group_key,
                    "pair_type": pair_type,
                    "seed": seed,
                    "condition": cond,
                }
                image_tasks.append({
                    "source": source,
                    "meta": meta,
                    "probes": _probe_spec(meta),
                })

    if not image_tasks:
        print("No tasks found. Check that images exist under pairs/*/images/")
        return [], []

    total_probe_tasks = sum(len(task["probes"]) for task in image_tasks)
    print(
        f"Found {len(image_tasks)} image tasks and {total_probe_tasks} probe questions "
        "for pair-type-aware evaluation"
    )

    flat_tasks: list[tuple[dict, dict, dict]] = []
    for task in image_tasks:
        for probe in task["probes"]:
            flat_tasks.append((task["source"], task["meta"], probe))

    probe_records: list[dict] = []
    image_cache: dict[Path, object] = {}
    for start in range(0, len(flat_tasks), batch_size):
        batch = flat_tasks[start : start + batch_size]
        imgs = [_load_image_from_source(source, image_cache) for source, _, _ in batch]
        questions = [probe["question"] for _, _, probe in batch]
        scores = score_batch(imgs, questions, processor, model, device)
        for (_, meta, probe), score in zip(batch, scores):
            probe_records.append({
                **meta,
                "probe": probe["probe"],
                "question": probe["question"],
                "probe_positive": probe["positive"],
                "probe_component": probe["component"],
                "score": round(float(score), 6),
            })
        if (start // batch_size) % 10 == 0:
            print(f"  [{start + len(batch)}/{len(flat_tasks)}] ...")
        for img in imgs:
            img.close()

    for img in image_cache.values():
        img.close()

    image_scores: list[dict] = []
    grouped: dict[tuple, list[dict]] = {}
    for row in probe_records:
        key = (
            row["pair_slug"],
            row["seed"],
            row["condition"],
        )
        grouped.setdefault(key, []).append(row)

    for rows in grouped.values():
        base = {k: rows[0][k] for k in (
            "pair", "c1", "c2", "pair_slug", "taxonomy_group_key", "pair_type", "seed", "condition"
        )}
        image_scores.append({
            **base,
            **_aggregate_scores(base, rows),
        })

    return probe_records, image_scores


def enrich_records(records: list[dict]) -> list[dict]:
    if not records:
        return records
    df = pd.DataFrame(records)
    df = enrich_taxonomy_dataframe(df)
    return df.to_dict(orient="records")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pair-type-aware BLIP-VQA evaluation for joint correctness.",
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
        help="Output JSON path (default: {data_dir}/joint_probe_scores.json).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_path = Path(args.output) if args.output else data_dir / "joint_probe_scores.json"

    probe_records, image_scores = evaluate(
        data_dir=data_dir,
        conditions=args.conditions,
        device=args.device,
        batch_size=args.batch_size,
        pairs_filter=args.pairs,
    )

    probe_records = enrich_records(probe_records)
    image_scores = enrich_records(image_scores)

    payload = {
        "model_id": MODEL_ID,
        "version": 3,
        "semantic_pass_threshold": SEMANTIC_PASS_THRESHOLD,
        "high_confidence_pass_threshold": HIGH_CONFIDENCE_PASS_THRESHOLD,
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "semantic_drift_gate_threshold": SEMANTIC_DRIFT_GATE_THRESHOLD,
        "probe_records": probe_records,
        "image_scores": image_scores,
        "pair_summaries": enrich_records(summarize_pairs(image_scores)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(
        f"\nSaved {len(probe_records)} probe records and {len(image_scores)} image scores "
        f"to {output_path}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
