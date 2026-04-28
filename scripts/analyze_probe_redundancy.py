#!/usr/bin/env python3
"""Analyze probe redundancy and probe usefulness across inspection JSONs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from scipy.stats import pearsonr

try:
    from sdxl_paper_audit_common import (
        DEFAULT_AUDIT_MANIFEST_PATH,
        index_manifest_entries,
        infer_source_run_from_pair_dir,
        load_manifest,
        manifest_key,
    )
except ImportError:
    from scripts.sdxl_paper_audit_common import (
        DEFAULT_AUDIT_MANIFEST_PATH,
        index_manifest_entries,
        infer_source_run_from_pair_dir,
        load_manifest,
        manifest_key,
    )


DEFAULT_DRILLDOWN_PAIRS = [
    "a_transparent_glass__x__a_dog",
    "a_cactus__x__the_arctic_tundra",
    "a_typewriter__x__a_cactus",
    "a_snowman__x__a_tropical_beach",
]


def load_inspection_files(data_dirs: list[Path]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for data_dir in data_dirs:
        for json_path in data_dir.rglob("**/probe_inspection/monolithic_probe_inspection.json"):
            try:
                payload = json.loads(json_path.read_text())
                payload["_json_path"] = str(json_path)
                payload["_source_run"] = infer_source_run_from_pair_dir(Path(payload.get("pair_dir", json_path.parent.parent.parent)))
                payloads.append(payload)
            except Exception as exc:
                print(f"Warning: failed to load {json_path}: {exc}", file=sys.stderr)
    return payloads


def _probe_metadata(payloads: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for probe in payload.get("probes", []):
            probe_name = str(probe["probe"])
            metadata.setdefault(
                probe_name,
                {
                    "positive": bool(probe.get("probe_positive", True)),
                    "component": probe.get("probe_component"),
                    "question": probe.get("question"),
                },
            )
    return metadata


def build_probe_matrix(payloads: list[dict[str, Any]]) -> tuple[dict[str, list[float]], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    metadata = _probe_metadata(payloads)
    scores_by_probe: dict[str, list[float]] = defaultdict(list)
    records_by_probe: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for payload in payloads:
        aggregate = payload.get("aggregate", {})
        meta = payload.get("meta", {})
        for probe in payload.get("probes", []):
            record = {
                "pair_slug": meta.get("pair_slug"),
                "taxonomy_group_key": meta.get("taxonomy_group_key"),
                "condition": meta.get("condition"),
                "source_run": payload.get("_source_run"),
                "score": float(probe["score"]),
                "joint_correctness_score": aggregate.get("joint_correctness_score"),
                "semantic_pass": aggregate.get("semantic_pass"),
                "tier1_quality_score": aggregate.get("tier1_quality_score"),
                "tier2_quality_score": aggregate.get("tier2_quality_score"),
                "semantic_drift_flagged": aggregate.get("semantic_drift_flagged"),
                "probe_positive": bool(probe.get("probe_positive", True)),
                "probe_component": probe.get("probe_component"),
            }
            scores_by_probe[str(probe["probe"])].append(float(probe["score"]))
            records_by_probe[str(probe["probe"])].append(record)

    return dict(scores_by_probe), dict(records_by_probe), metadata


def compute_pairwise_correlation(records_by_probe: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, str], float]:
    correlations: dict[tuple[str, str], float] = {}
    aligned: dict[str, dict[tuple[str, str], float]] = {}
    for probe, records in records_by_probe.items():
        aligned[probe] = {
            (str(r["source_run"]), str(r["pair_slug"])): float(r["score"])
            for r in records
        }
    probe_names = sorted(records_by_probe.keys())
    for idx, p1 in enumerate(probe_names):
        for p2 in probe_names[idx + 1 :]:
            common = sorted(set(aligned[p1]) & set(aligned[p2]))
            if len(common) < 2:
                continue
            scores1 = np.asarray([aligned[p1][k] for k in common], dtype=np.float32)
            scores2 = np.asarray([aligned[p2][k] for k in common], dtype=np.float32)
            try:
                r, _ = pearsonr(scores1, scores2)
                correlations[(p1, p2)] = float(r)
            except Exception as exc:
                print(f"Warning: correlation ({p1}, {p2}): {exc}", file=sys.stderr)
    return correlations


def compute_disagreement_rate(
    records_by_probe: dict[str, list[dict[str, Any]]],
    probe_metadata: dict[str, dict[str, Any]],
    *,
    pass_threshold: float,
) -> dict[tuple[str, str], float]:
    disagreements: dict[tuple[str, str], float] = {}
    aligned: dict[str, dict[tuple[str, str], bool]] = {}
    for probe, records in records_by_probe.items():
        positive = bool(probe_metadata.get(probe, {}).get("positive", True))
        pass_map = {}
        for record in records:
            score = float(record["score"])
            passed = score >= pass_threshold if positive else score <= (1.0 - pass_threshold)
            pass_map[(str(record["source_run"]), str(record["pair_slug"]))] = passed
        aligned[probe] = pass_map
    probe_names = sorted(records_by_probe.keys())
    for idx, p1 in enumerate(probe_names):
        for p2 in probe_names[idx + 1 :]:
            common = sorted(set(aligned[p1]) & set(aligned[p2]))
            if len(common) < 2:
                continue
            disagree = sum(aligned[p1][k] != aligned[p2][k] for k in common) / len(common)
            disagreements[(p1, p2)] = disagree
    return disagreements


def compute_false_penalization_rates(
    records_by_probe: dict[str, list[dict[str, Any]]],
    probe_metadata: dict[str, dict[str, Any]],
    *,
    pass_threshold: float,
    audit_index: dict[tuple[str, str], dict[str, Any]] | None,
    good_image_threshold: float,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for probe, records in records_by_probe.items():
        positive = bool(probe_metadata.get(probe, {}).get("positive", True))
        penalized_good = 0
        n_good = 0
        penalized_human_keep = 0
        n_human_keep = 0
        keep_scores: list[float] = []
        drop_scores: list[float] = []
        for record in records:
            score = float(record["score"])
            passed = score >= pass_threshold if positive else score <= (1.0 - pass_threshold)
            joint = record.get("joint_correctness_score")
            if joint is not None and float(joint) >= good_image_threshold:
                n_good += 1
                if not passed:
                    penalized_good += 1
            if audit_index is not None:
                key = manifest_key(str(record["source_run"]), str(record["pair_slug"]))
                entry = audit_index.get(key)
                if entry is None:
                    continue
                decision = entry.get("decision")
                if decision == "keep":
                    n_human_keep += 1
                    keep_scores.append(score)
                    if not passed:
                        penalized_human_keep += 1
                elif decision == "drop":
                    drop_scores.append(score)
        results[probe] = {
            "false_pen_rate_metric_good": (penalized_good / n_good) if n_good else None,
            "false_pen_rate_human_keep": (penalized_human_keep / n_human_keep) if n_human_keep else None,
            "mean_keep_score": mean(keep_scores) if keep_scores else None,
            "mean_drop_score": mean(drop_scores) if drop_scores else None,
            "separation": abs(mean(keep_scores) - mean(drop_scores)) if keep_scores and drop_scores else None,
            "n_human_keep": n_human_keep,
            "n_human_drop": len(drop_scores),
        }
    return results


def classify_probe(
    probe: str,
    correlations: dict[tuple[str, str], float],
    disagreement_rates: dict[tuple[str, str], float],
    probe_stats: dict[str, Any],
    probe_metadata: dict[str, dict[str, Any]],
    high_correlation_threshold: float,
    n_payloads: int,
) -> str:
    false_pen_human = probe_stats.get("false_pen_rate_human_keep")
    false_pen_metric = probe_stats.get("false_pen_rate_metric_good")
    separation = probe_stats.get("separation")
    component = str(probe_metadata.get(probe, {}).get("component") or "")

    best_corr = 0.0
    best_disagreement = 1.0
    for (p1, p2), corr in correlations.items():
        if probe not in {p1, p2}:
            continue
        other = p2 if probe == p1 else p1
        best_corr = max(best_corr, corr)
        best_disagreement = min(best_disagreement, disagreement_rates.get((probe, other), disagreement_rates.get((other, probe), 1.0)))

    if best_corr >= max(0.85, high_correlation_threshold) and best_disagreement <= 0.10 and (false_pen_human or false_pen_metric or 0.0) <= 0.20:
        return "remove_candidate"
    if false_pen_human is not None and false_pen_human >= 0.35:
        return "review_wording"
    if separation is not None and separation < 0.08:
        return "keep_but_weak_signal"
    if component in {"anti_hybrid", "anti_confusion", "anti_semantic_drift"} and false_pen_human is not None and false_pen_human >= 0.25:
        return "review_wording"
    if n_payloads < 10:
        return "keep_but_weak_signal"
    return "keep"


def _format_rate(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "N/A"


def _format_float(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "N/A"


def print_report(
    *,
    payloads: list[dict[str, Any]],
    scores_by_probe: dict[str, list[float]],
    records_by_probe: dict[str, list[dict[str, Any]]],
    probe_metadata: dict[str, dict[str, Any]],
    correlations: dict[tuple[str, str], float],
    disagreement_rates: dict[tuple[str, str], float],
    false_penalization_rates: dict[str, dict[str, Any]],
    high_correlation_threshold: float,
    output_path: Path | None,
    audit_index: dict[tuple[str, str], dict[str, Any]] | None,
    drilldown_pairs: list[str],
) -> None:
    n_payloads = len(payloads)
    lines: list[str] = []
    lines.append("=" * 110)
    lines.append("PROBE AUDIT DIAGNOSTIC REPORT")
    lines.append("=" * 110)
    lines.append(f"Inspection files analyzed: {n_payloads}")
    if n_payloads < 10:
        lines.append("WARNING: N < 10. Correlations and weak-signal judgments are documentation-grade only.")
    if audit_index is not None:
        decisions = [entry.get("decision") for entry in audit_index.values()]
        lines.append(
            "Audit manifest decisions: "
            f"keep={sum(d == 'keep' for d in decisions)}, "
            f"drop={sum(d == 'drop' for d in decisions)}, "
            f"backup={sum(d == 'backup' for d in decisions)}, "
            f"unreviewed={sum(d == 'unreviewed' for d in decisions)}"
        )
    lines.append("")

    lines.append("SECTION 1: Per-probe validity and usefulness")
    lines.append("-" * 110)
    for probe in sorted(scores_by_probe):
        scores = scores_by_probe[probe]
        probe_stats = false_penalization_rates[probe]
        classification = classify_probe(
            probe,
            correlations,
            disagreement_rates,
            probe_stats,
            probe_metadata,
            high_correlation_threshold,
            n_payloads,
        )
        lines.append(
            f"{probe:<28} class={classification:<20} component={str(probe_metadata[probe].get('component')):<18} "
            f"positive={str(probe_metadata[probe].get('positive')):<5} mean={mean(scores):.3f} "
            f"stdev={(stdev(scores) if len(scores) > 1 else 0.0):.3f} "
            f"false_pen(metric_good)={_format_rate(probe_stats.get('false_pen_rate_metric_good')):<8} "
            f"false_pen(human_keep)={_format_rate(probe_stats.get('false_pen_rate_human_keep')):<8} "
            f"separation={_format_float(probe_stats.get('separation'))}"
        )
    lines.append("")

    lines.append("SECTION 2: Redundancy")
    lines.append("-" * 110)
    high_corr = sorted(
        [(p1, p2, corr) for (p1, p2), corr in correlations.items() if corr >= high_correlation_threshold],
        key=lambda item: item[2],
        reverse=True,
    )
    if not high_corr:
        lines.append("No probe pairs crossed the configured correlation threshold.")
    for p1, p2, corr in high_corr:
        disagreement = disagreement_rates.get((p1, p2), disagreement_rates.get((p2, p1)))
        lines.append(f"{p1} <-> {p2}: r={corr:.3f}, disagreement={_format_rate(disagreement)}")
    lines.append("")

    lines.append("SECTION 3: Capability mismatch / wording review candidates")
    lines.append("-" * 110)
    flagged = []
    for probe in sorted(scores_by_probe):
        classification = classify_probe(
            probe,
            correlations,
            disagreement_rates,
            false_penalization_rates[probe],
            probe_metadata,
            high_correlation_threshold,
            n_payloads,
        )
        if classification in {"review_wording", "keep_but_weak_signal"}:
            flagged.append((probe, classification, false_penalization_rates[probe]))
    if not flagged:
        lines.append("No probes were flagged for wording/capability review.")
    for probe, classification, stats in flagged:
        lines.append(
            f"{probe}: {classification}, false_pen(human_keep)={_format_rate(stats.get('false_pen_rate_human_keep'))}, "
            f"separation={_format_float(stats.get('separation'))}"
        )
    lines.append("")

    if audit_index is not None:
        lines.append("SECTION 4: Metric vs human audit disagreement")
        lines.append("-" * 110)
        disagreement_counts = defaultdict(int)
        for payload in payloads:
            meta = payload.get("meta", {})
            key = manifest_key(str(payload.get("_source_run")), str(meta.get("pair_slug")))
            entry = audit_index.get(key)
            if entry is None:
                continue
            decision = entry.get("decision")
            semantic_pass = payload.get("aggregate", {}).get("semantic_pass")
            if decision == "keep" and semantic_pass is False:
                disagreement_counts["metric_fail / human_keep"] += 1
            elif decision == "drop" and semantic_pass is True:
                disagreement_counts["metric_keep / human_drop"] += 1
        if not disagreement_counts:
            lines.append("No reviewed pair-level disagreements found.")
        for label, count in sorted(disagreement_counts.items()):
            lines.append(f"{label}: {count}")
        lines.append("")

    lines.append("SECTION 5: Pair-level drilldown")
    lines.append("-" * 110)
    found_drilldown = False
    for pair_slug in drilldown_pairs:
        matching = [payload for payload in payloads if payload.get("meta", {}).get("pair_slug") == pair_slug]
        if not matching:
            continue
        found_drilldown = True
        for payload in matching:
            aggregate = payload.get("aggregate", {})
            lines.append(
                f"{pair_slug} [{payload.get('_source_run')}] "
                f"joint={_format_float(aggregate.get('joint_correctness_score'))} "
                f"tier1={_format_float(aggregate.get('tier1_quality_score'))} "
                f"tier2={_format_float(aggregate.get('tier2_quality_score'))} "
                f"drift={aggregate.get('semantic_drift_flagged')}"
            )
            for probe in sorted(payload.get("probes", []), key=lambda item: float(item.get("score", 0.0))):
                p_name = str(probe["probe"])
                positive = bool(probe.get("probe_positive", True))
                score = float(probe["score"])
                passed = score >= 0.58 if positive else score <= 0.42
                lines.append(
                    f"  {p_name:<28} score={score:.3f} passed={passed!s:<5} "
                    f"component={str(probe.get('probe_component')):<18} question={probe.get('question')}"
                )
            lines.append("")
    if not found_drilldown:
        lines.append("None of the requested drilldown pairs were present in the inspection payloads.")

    report_text = "\n".join(lines)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text)
        print(f"Report saved to {output_path}")
    else:
        print(report_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze probe redundancy and usefulness from inspection JSONs.")
    parser.add_argument("--data-dirs", nargs="+", required=True, type=Path, help="Screening run directories containing probe_inspection/ subdirs.")
    parser.add_argument("--output-report", type=Path, default=None, help="Path to save report (default: print to stdout).")
    parser.add_argument("--audit-manifest", type=Path, default=DEFAULT_AUDIT_MANIFEST_PATH, help="Optional human audit manifest for false-negative analysis.")
    parser.add_argument("--high-correlation-threshold", type=float, default=0.80, help="Threshold for flagging correlated probe pairs.")
    parser.add_argument("--pass-threshold", type=float, default=0.58, help="BLIP-VQA pass threshold for defining pass/fail.")
    parser.add_argument("--good-image-threshold", type=float, default=0.60, help="Fallback threshold for metric-good images when no human keep decision exists.")
    parser.add_argument("--drilldown-pairs", nargs="*", default=DEFAULT_DRILLDOWN_PAIRS, help="Pair slugs to include in the drilldown section.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = load_inspection_files(args.data_dirs)
    if not payloads:
        print("No inspection files found.", file=sys.stderr)
        sys.exit(1)

    scores_by_probe, records_by_probe, probe_metadata = build_probe_matrix(payloads)
    correlations = compute_pairwise_correlation(records_by_probe)
    disagreement_rates = compute_disagreement_rate(records_by_probe, probe_metadata, pass_threshold=args.pass_threshold)

    audit_index = None
    if args.audit_manifest and args.audit_manifest.exists():
        audit_index = index_manifest_entries(load_manifest(args.audit_manifest))

    false_pen_rates = compute_false_penalization_rates(
        records_by_probe,
        probe_metadata,
        pass_threshold=args.pass_threshold,
        audit_index=audit_index,
        good_image_threshold=args.good_image_threshold,
    )

    print_report(
        payloads=payloads,
        scores_by_probe=scores_by_probe,
        records_by_probe=records_by_probe,
        probe_metadata=probe_metadata,
        correlations=correlations,
        disagreement_rates=disagreement_rates,
        false_penalization_rates=false_pen_rates,
        high_correlation_threshold=args.high_correlation_threshold,
        output_path=args.output_report,
        audit_index=audit_index,
        drilldown_pairs=args.drilldown_pairs,
    )


if __name__ == "__main__":
    main()
