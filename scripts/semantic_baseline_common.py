"""Shared helpers for SDXL semantic-baseline auditing and filtering."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from taxonomy_manifest import GROUP_SPECS
except ImportError:
    from scripts.taxonomy_manifest import GROUP_SPECS


DEFAULT_PAIR_PASS_THRESHOLD = 0.75


def resolve_joint_probe_path(data_dir: Path) -> Path:
    candidate = data_dir / "metrics" / "joint_probe_scores.json"
    if candidate.exists():
        return candidate
    return data_dir / "joint_probe_scores.json"


def resolve_semantic_audit_path(data_dir: Path) -> Path:
    candidate = data_dir / "metrics" / "semantic_baseline_audit.json"
    if candidate.exists():
        return candidate
    return data_dir / "semantic_baseline_audit.json"


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _condition_label(condition: str) -> str:
    return {
        "mono": "monolithic",
        "poe": "poe",
        "pstar_sdipc": "pstar_sdipc",
    }.get(condition, condition)


def _build_pair_summaries_from_image_scores(image_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in image_scores:
        cond = str(row.get("condition", "")).strip()
        if cond in {"mono", "poe", "pstar_sdipc"}:
            grouped[(row.get("pair_slug"), cond)].append(row)

    summaries: list[dict[str, Any]] = []
    for (pair_slug, cond), rows in grouped.items():
        joint = [_float_or_none(r.get("joint_correctness_score")) for r in rows]
        joint = [x for x in joint if x is not None]
        if not joint:
            continue
        semantic = [
            bool(r.get("semantic_pass", (_float_or_none(r.get("joint_correctness_score")) or 0.0) >= 0.60))
            for r in rows
        ]
        high_conf = [
            bool(r.get("high_confidence_pass", (_float_or_none(r.get("joint_correctness_score")) or 0.0) >= 0.75))
            for r in rows
        ]
        hybrid = [bool(r.get("hybridization_failure", False)) for r in rows]
        omission = [bool(r.get("omission_failure", False)) for r in rows]
        summaries.append(
            {
                "pair": rows[0].get("pair"),
                "pair_slug": pair_slug,
                "taxonomy_group_key": rows[0].get("taxonomy_group_key"),
                "taxonomy_group_label": rows[0].get("taxonomy_group_label"),
                "condition": cond,
                "mean_joint_correctness_score": sum(joint) / len(joint),
                "median_joint_correctness_score": sorted(joint)[len(joint) // 2],
                "semantic_pass_rate": sum(semantic) / len(semantic),
                "high_confidence_pass_rate": sum(high_conf) / len(high_conf),
                "hybridization_failure_rate": sum(hybrid) / len(hybrid) if hybrid else 0.0,
                "omission_failure_rate": sum(omission) / len(omission) if omission else 0.0,
                "recommended_taxonomy_group_key": rows[0].get("recommended_taxonomy_group_key"),
                "recommended_group_fit_score": _float_or_none(rows[0].get("recommended_group_fit_score")),
                "group_matches_recommendation": rows[0].get("group_matches_recommendation"),
            }
        )
    return summaries


def _score_sort_key(row: dict[str, Any]) -> tuple:
    return (
        int(bool(row.get("pair_eligible_for_semantic_baseline"))),
        _float_or_none(row.get("mono_semantic_pass_rate")) or -1.0,
        _float_or_none(row.get("mono_mean_joint_correctness_score")) or -1.0,
        -(_float_or_none(row.get("mono_hybridization_failure_rate")) or 0.0),
        -(_float_or_none(row.get("mono_omission_failure_rate")) or 0.0),
    )


def build_semantic_baseline_audit(
    payload: dict[str, Any],
    pair_pass_threshold: float = DEFAULT_PAIR_PASS_THRESHOLD,
) -> dict[str, Any]:
    image_scores = payload.get("image_scores", []) or []
    pair_summaries = payload.get("pair_summaries", []) or []
    if not image_scores:
        raise ValueError("joint_probe_scores payload does not contain image_scores")
    if not pair_summaries:
        pair_summaries = _build_pair_summaries_from_image_scores(image_scores)

    seed_rows: list[dict[str, Any]] = []
    grouped_seed_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in image_scores:
        cond = str(row.get("condition", "")).strip()
        if cond not in {"mono", "poe", "pstar_sdipc"}:
            continue
        seed_row = {
            "pair": row.get("pair"),
            "pair_slug": row.get("pair_slug"),
            "taxonomy_group_key": row.get("taxonomy_group_key"),
            "taxonomy_group_label": row.get("taxonomy_group_label"),
            "condition": cond,
            "condition_label": _condition_label(cond),
            "seed": row.get("seed"),
            "joint_correctness_score": _float_or_none(row.get("joint_correctness_score")),
            "semantic_pass": _bool_or_none(row.get("semantic_pass")),
            "high_confidence_pass": _bool_or_none(row.get("high_confidence_pass")),
            "hybridization_failure": _bool_or_none(row.get("hybridization_failure")),
            "omission_failure": _bool_or_none(row.get("omission_failure")),
        }
        seed_row["seed_eligible_for_semantic_baseline"] = (
            seed_row["condition"] == "mono" and bool(seed_row["semantic_pass"])
        )
        seed_rows.append(seed_row)
        grouped_seed_rows[(seed_row["pair_slug"], cond)].append(seed_row)

    summary_by_pair_cond: dict[tuple[str, str], dict[str, Any]] = {}
    for row in pair_summaries:
        cond = str(row.get("condition", "")).strip()
        if cond in {"mono", "poe", "pstar_sdipc"}:
            summary_by_pair_cond[(row.get("pair_slug"), cond)] = row

    pair_rows: list[dict[str, Any]] = []
    mono_pairs_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mono_pair_slug_to_row: dict[str, dict[str, Any]] = {}
    for (pair_slug, cond), summary in summary_by_pair_cond.items():
        if cond != "mono":
            continue
        pair = summary.get("pair")
        pair_row = {
            "pair": pair,
            "pair_slug": pair_slug,
            "taxonomy_group_key": summary.get("taxonomy_group_key"),
            "taxonomy_group_label": summary.get("taxonomy_group_label"),
            "mono_mean_joint_correctness_score": _float_or_none(summary.get("mean_joint_correctness_score")),
            "mono_median_joint_correctness_score": _float_or_none(summary.get("median_joint_correctness_score")),
            "mono_semantic_pass_rate": _float_or_none(summary.get("semantic_pass_rate")),
            "mono_high_confidence_pass_rate": _float_or_none(summary.get("high_confidence_pass_rate")),
            "mono_hybridization_failure_rate": _float_or_none(summary.get("hybridization_failure_rate")),
            "mono_omission_failure_rate": _float_or_none(summary.get("omission_failure_rate")),
            "recommended_taxonomy_group_key": summary.get("recommended_taxonomy_group_key"),
            "recommended_group_fit_score": _float_or_none(summary.get("recommended_group_fit_score")),
            "group_matches_recommendation": summary.get("group_matches_recommendation"),
            "pair_eligible_for_semantic_baseline": (
                (_float_or_none(summary.get("semantic_pass_rate")) or 0.0) >= pair_pass_threshold
            ),
            "pair_high_confidence_qualified": (
                (_float_or_none(summary.get("high_confidence_pass_rate")) or 0.0) >= pair_pass_threshold
            ),
        }
        mono_pairs_by_group[pair_row["taxonomy_group_key"]].append(pair_row)
        mono_pair_slug_to_row[pair_slug] = pair_row
        pair_rows.append(pair_row)

    group_rows: list[dict[str, Any]] = []
    roster_rows: list[dict[str, Any]] = []
    for spec in GROUP_SPECS:
        group_key = spec["key"]
        candidates = sorted(
            mono_pairs_by_group.get(group_key, []),
            key=_score_sort_key,
            reverse=True,
        )
        target_count = len(spec["pairs"])
        selected = candidates[:target_count]
        selected_slugs = {row["pair_slug"] for row in selected}
        accepted = 0
        provisional = 0
        excluded = 0
        for row in candidates:
            if row["pair_slug"] in selected_slugs:
                if row["pair_eligible_for_semantic_baseline"]:
                    status = "accepted"
                    accepted += 1
                else:
                    status = "provisional"
                    provisional += 1
            else:
                status = "excluded"
                excluded += 1
            roster_rows.append(
                {
                    "taxonomy_group_key": group_key,
                    "taxonomy_group_label": spec["label"],
                    "pair": row["pair"],
                    "pair_slug": row["pair_slug"],
                    "selection_status": status,
                    "pair_eligible_for_semantic_baseline": row["pair_eligible_for_semantic_baseline"],
                    "mono_semantic_pass_rate": row["mono_semantic_pass_rate"],
                    "mono_mean_joint_correctness_score": row["mono_mean_joint_correctness_score"],
                    "mono_hybridization_failure_rate": row["mono_hybridization_failure_rate"],
                    "mono_omission_failure_rate": row["mono_omission_failure_rate"],
                }
            )
        group_rows.append(
            {
                "taxonomy_group_key": group_key,
                "taxonomy_group_label": spec["label"],
                "candidate_count": len(candidates),
                "target_pair_count": target_count,
                "accepted_count": accepted,
                "provisional_count": provisional,
                "excluded_count": excluded,
                "replacement_required": provisional > 0 or len(selected) < target_count,
                "selected_pair_slugs": [row["pair_slug"] for row in selected],
            }
        )

    return {
        "semantic_baseline_version": 1,
        "pair_pass_threshold": pair_pass_threshold,
        "seed_gate_default": "semantic",
        "pair_gate_default": "semantic_pass_rate",
        "seed_rows": seed_rows,
        "pair_rows": pair_rows,
        "group_rows": group_rows,
        "roster_rows": roster_rows,
        "counts": {
            "n_seed_rows": len(seed_rows),
            "n_pair_rows": len(pair_rows),
            "n_groups": len(group_rows),
        },
    }


def build_audit_from_joint_probe_file(
    joint_probe_path: Path,
    pair_pass_threshold: float = DEFAULT_PAIR_PASS_THRESHOLD,
) -> dict[str, Any]:
    payload = json.loads(joint_probe_path.read_text())
    audit = build_semantic_baseline_audit(payload, pair_pass_threshold=pair_pass_threshold)
    audit["source"] = str(joint_probe_path)
    return audit


def select_qualified_pair_slugs(
    audit: dict[str, Any],
    scope: str,
    mono_pass_threshold: float,
    mono_seed_gate: str,
) -> tuple[set[str], set[tuple[str, Any]]]:
    if scope == "full":
        return set(), set()

    if mono_seed_gate not in {"semantic", "high_confidence"}:
        raise ValueError(f"Unknown mono_seed_gate: {mono_seed_gate}")

    pair_rows = audit.get("pair_rows", [])
    seed_rows = audit.get("seed_rows", [])

    if scope == "pair_qualified":
        qualified_pairs = {
            row["pair_slug"]
            for row in pair_rows
            if (
                (row.get("mono_high_confidence_pass_rate") if mono_seed_gate == "high_confidence" else row.get("mono_semantic_pass_rate"))
                or 0.0
            ) >= mono_pass_threshold
        }
        return qualified_pairs, set()

    if scope == "seed_qualified":
        qualified_pairs = {
            row["pair_slug"]
            for row in pair_rows
            if (
                (row.get("mono_high_confidence_pass_rate") if mono_seed_gate == "high_confidence" else row.get("mono_semantic_pass_rate"))
                or 0.0
            ) >= mono_pass_threshold
        }
        qualified_seed_pairs = {
            (row["pair_slug"], row["seed"])
            for row in seed_rows
            if row.get("condition") == "mono"
            and (
                row.get("high_confidence_pass") if mono_seed_gate == "high_confidence" else row.get("semantic_pass")
            )
        }
        return qualified_pairs, qualified_seed_pairs

    raise ValueError(f"Unknown semantic-baseline scope: {scope}")
