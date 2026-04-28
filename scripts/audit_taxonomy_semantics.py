#!/usr/bin/env python3
"""Summarize joint-probe outputs into pair-level taxonomy audit recommendations."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _resolve_joint_probe_path(data_dir: Path) -> Path:
    candidate = data_dir / "metrics" / "joint_probe_scores.json"
    if candidate.exists():
        return candidate
    return data_dir / "joint_probe_scores.json"


def _fit_status(row: dict) -> str:
    recommended = row.get("recommended_taxonomy_group_key")
    observed = row.get("taxonomy_group_key")
    if recommended in (None, "ambiguous"):
        return "ambiguous"
    if observed == recommended:
        return "matches"
    return "reclassify"


def _recommended_group(mean_joint: float, hybrid_rate: float, omission_rate: float) -> str:
    if mean_joint >= 0.75 and hybrid_rate <= 0.20:
        return "group1_cooccurrence"
    if mean_joint >= 0.60 and hybrid_rate <= 0.30:
        return "group2_disentangled"
    if omission_rate >= hybrid_rate:
        return "group3_feature_overlap"
    if hybrid_rate > omission_rate:
        return "group4_coherent_collision"
    return "ambiguous"


def _build_pair_summaries_from_image_scores(image_scores: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in image_scores:
        grouped[(row["pair_slug"], row["condition"])].append(row)

    summaries = []
    for (pair_slug, condition), rows in grouped.items():
        target_rows = [r for r in rows if r.get("joint_correctness_score") is not None]
        if not target_rows:
            continue
        joint = [float(r["joint_correctness_score"]) for r in target_rows]
        semantic = [
            bool(r.get("semantic_pass", float(r["joint_correctness_score"]) >= 0.60))
            for r in target_rows
        ]
        hybrid = [bool(r.get("hybridization_failure", False)) for r in target_rows]
        omission = [bool(r.get("omission_failure", False)) for r in target_rows]
        mean_joint = float(sum(joint) / len(joint))
        hybrid_rate = float(sum(hybrid) / len(hybrid)) if hybrid else 0.0
        omission_rate = float(sum(omission) / len(omission)) if omission else 0.0
        summaries.append(
            {
                "pair": rows[0].get("pair"),
                "pair_slug": pair_slug,
                "taxonomy_group_key": rows[0].get("taxonomy_group_key"),
                "taxonomy_group_label": rows[0].get("taxonomy_group_label"),
                "condition": condition,
                "n_seeds": len(rows),
                "mean_joint_correctness_score": round(mean_joint, 6),
                "median_joint_correctness_score": round(float(statistics.median(joint)), 6),
                "semantic_pass_rate": round(sum(semantic) / len(semantic), 6),
                "hybridization_failure_rate": round(hybrid_rate, 6),
                "omission_failure_rate": round(omission_rate, 6),
                "recommended_taxonomy_group_key": _recommended_group(mean_joint, hybrid_rate, omission_rate),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit pair-level taxonomy fit from joint_probe_scores.json.",
    )
    parser.add_argument(
        "--data-dir",
        default="experiments/inversion/gap_analysis",
        help="Run root containing joint_probe_scores.json.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path (default: {data_dir}/taxonomy_semantic_audit.json).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    path = _resolve_joint_probe_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"joint_probe_scores.json not found under {data_dir}. "
            "Run scripts/eval_joint_probes.py first."
        )

    payload = json.loads(path.read_text())
    pair_summaries = payload.get("pair_summaries", [])
    if not pair_summaries and payload.get("image_scores"):
        pair_summaries = _build_pair_summaries_from_image_scores(payload["image_scores"])
    target_conditions = {"mono", "poe", "pstar_sdipc"}
    filtered = [row for row in pair_summaries if row.get("condition") in target_conditions]

    pair_rows = {}
    for row in filtered:
        pair_slug = row["pair_slug"]
        pair_rows.setdefault(
            pair_slug,
            {
                "pair": row.get("pair"),
                "pair_slug": pair_slug,
                "taxonomy_group_key": row.get("taxonomy_group_key"),
                "taxonomy_group_label": row.get("taxonomy_group_label"),
                "conditions": {},
            },
        )["conditions"][row["condition"]] = row

    audit_rows = []
    for pair_slug, bundle in sorted(pair_rows.items()):
        mono = bundle["conditions"].get("mono", {})
        poe = bundle["conditions"].get("poe", {})
        pstar = bundle["conditions"].get("pstar_sdipc", {})
        # Prefer PoE as the reclassification anchor, then Mono, then p*.
        recommendation_source = poe or mono or pstar
        recommended_group = recommendation_source.get("recommended_taxonomy_group_key", "ambiguous")
        fit_status = _fit_status(
            {
                "recommended_taxonomy_group_key": recommended_group,
                "taxonomy_group_key": bundle.get("taxonomy_group_key"),
            }
        )
        audit_rows.append(
            {
                "pair": bundle.get("pair"),
                "pair_slug": pair_slug,
                "taxonomy_group_key": bundle.get("taxonomy_group_key"),
                "taxonomy_group_label": bundle.get("taxonomy_group_label"),
                "recommended_taxonomy_group_key": recommended_group,
                "fit_status": fit_status,
                "poe_semantic_pass_rate": poe.get("semantic_pass_rate"),
                "mono_semantic_pass_rate": mono.get("semantic_pass_rate"),
                "pstar_semantic_pass_rate": pstar.get("semantic_pass_rate"),
                "poe_hybridization_failure_rate": poe.get("hybridization_failure_rate"),
                "mono_hybridization_failure_rate": mono.get("hybridization_failure_rate"),
                "pstar_hybridization_failure_rate": pstar.get("hybridization_failure_rate"),
                "poe_omission_failure_rate": poe.get("omission_failure_rate"),
                "mono_omission_failure_rate": mono.get("omission_failure_rate"),
                "pstar_omission_failure_rate": pstar.get("omission_failure_rate"),
            }
        )

    out_path = Path(args.output) if args.output else data_dir / "taxonomy_semantic_audit.json"
    result = {
        "source": str(path),
        "n_pairs": len(audit_rows),
        "audit_rows": audit_rows,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved taxonomy semantic audit to {out_path}")


if __name__ == "__main__":
    main()
