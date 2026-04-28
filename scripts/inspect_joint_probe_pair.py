#!/usr/bin/env python3
"""Inspect a single SDXL pair with the monolithic joint-probe bundle.

This is a local diagnostic tool for iterating on Group 3 / Group 4 question
bundles. It loads one pair's `monolithic.png`, runs the current pair-type-aware
questions (or a caller-provided override bundle), saves a styled bar chart of
per-question probabilities, and writes a JSON summary with the aggregate score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(f"PyTorch is required: {exc}") from exc

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_blip_vqa import _find_images_for_pair, _infer_pair_metadata, _load_image_from_source, load_model, score_batch
from eval_joint_probes import (
    HIGH_CONFIDENCE_PASS_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    MODEL_ID,
    OMISSION_FAILURE_THRESHOLD,
    HYBRIDIZATION_FAILURE_THRESHOLD,
    SEMANTIC_PASS_THRESHOLD,
    SEMANTIC_DRIFT_GATE_THRESHOLD,
    _aggregate_scores,
    _probe_spec,
    pair_type_for_group,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "eccv2026"
    / "sdxl_screening"
    / "sdxl_g34_replacements_seed42_steps50_cfg7p5_20260413"
)


def _resolve_pair_dir(data_dir: Path, pair_value: str, group: str | None) -> Path:
    pair_token = Path(pair_value).name
    if group:
        candidate = data_dir / group / pair_token
        if candidate.exists():
            return candidate
        seed_candidates = sorted(data_dir.glob(f"seed_*/{group}/{pair_token}"))
        if seed_candidates:
            return seed_candidates[0]
        raise FileNotFoundError(f"Pair '{pair_token}' not found under group '{group}' in {data_dir}")

    matches = [p for p in data_dir.rglob(pair_token) if p.is_dir() and p.parent.name.startswith("group")]
    if not matches:
        raise FileNotFoundError(f"Could not find pair '{pair_token}' under {data_dir}")
    if len(matches) > 1:
        rels = ", ".join(str(p.relative_to(data_dir)) for p in matches)
        raise FileExistsError(f"Pair '{pair_token}' is ambiguous under {data_dir}: {rels}")
    return matches[0]


def _load_question_spec(path: Path, meta: dict) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        if "mono" in payload:
            payload = payload["mono"]
        elif "questions" in payload:
            payload = payload["questions"]
    if not isinstance(payload, list):
        raise ValueError(f"Question spec at {path} must be a list or a dict containing a list.")

    probes: list[dict] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Question spec row {idx} must be an object, got {type(row).__name__}")
        question = str(row.get("question", "")).strip()
        if not question:
            raise ValueError(f"Question spec row {idx} is missing a question.")
        probes.append(
            {
                "probe": str(row.get("probe") or f"custom_{idx+1}"),
                "question": question.format(**meta),
                "positive": bool(row.get("positive", True)),
                "component": str(row.get("component", "custom")),
            }
        )
    return probes


def _wrapped_labels(rows: list[dict]) -> list[str]:
    labels = []
    for row in rows:
        prefix = "+" if row["probe_positive"] else "-"
        # Use probe name directly, shortened if needed
        probe_name = row['probe']
        # Abbreviate long probe names for space
        abbrev_map = {
            'intentional_composition': 'intent',
            'concept_confusion': 'confusion',
            'dominance_failure': 'dominance',
            'merged_or_absorbed': 'merged',
            'distinct_entities': 'distinct',
            'hybrid_object': 'hybrid',
            'one_missing': 'missing',
            'both_present': 'both',
            'a_present': 'A',
            'b_present': 'B',
            'c1_not_subsumed_by_c2': 'not_subsumed',
            'semantic_role_correct': 'role_correct',
            'c1_independent_realization': 'c1_indep',
        }
        short_name = abbrev_map.get(probe_name, probe_name)
        labels.append(f"{prefix}{short_name}")
    return labels


def _build_plot(
    *,
    image_path: Path,
    pair_title: str,
    pair_slug: str,
    group_key: str,
    condition: str,
    probe_rows: list[dict],
    aggregate: dict,
    output_path: Path,
) -> None:
    # Scale figure height based on number of probes
    n_probes = len(probe_rows)
    fig_height = max(10.5, 8.0 + (n_probes - 8) * 0.3)
    fig = plt.figure(figsize=(18, fig_height))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.4], width_ratios=[1.0, 1.6], hspace=0.28, wspace=0.22)

    ax_img = fig.add_subplot(gs[:, 0])
    ax_raw = fig.add_subplot(gs[0, 1])
    ax_eff = fig.add_subplot(gs[1, 1])

    img = plt.imread(image_path)
    ax_img.imshow(img)
    ax_img.set_title("Monolithic image", fontsize=13, fontweight="bold")
    ax_img.axis("off")

    xs = np.arange(len(probe_rows))
    raw_scores = np.array([float(row["score"]) for row in probe_rows], dtype=float)
    effective_scores = np.array(
        [float(row["score"]) if row["probe_positive"] else 1.0 - float(row["score"]) for row in probe_rows],
        dtype=float,
    )
    raw_colors = ["#2F7D4A" if row["probe_positive"] else "#C44E52" for row in probe_rows]
    eff_colors = ["#2F7D4A" if row["probe_positive"] else "#7A5CBA" for row in probe_rows]

    ax_raw.bar(xs, raw_scores, color=raw_colors, alpha=0.9, width=0.72)
    ax_raw.axhline(LOW_CONFIDENCE_THRESHOLD, color="#6C757D", linestyle=":", linewidth=1.0, label=f"Low ({LOW_CONFIDENCE_THRESHOLD})")
    ax_raw.axhline(SEMANTIC_PASS_THRESHOLD, color="#8C6D1F", linestyle="--", linewidth=1.1, label=f"Semantic ({SEMANTIC_PASS_THRESHOLD})")
    ax_raw.axhline(HIGH_CONFIDENCE_PASS_THRESHOLD, color="#1F4E79", linestyle=":", linewidth=1.2, label=f"High ({HIGH_CONFIDENCE_PASS_THRESHOLD})")
    ax_raw.set_ylim(0.0, 1.02)
    ax_raw.set_ylabel('Raw P("yes")', fontsize=11)
    ax_raw.set_xticks(xs)
    ax_raw.set_xticklabels(_wrapped_labels(probe_rows), fontsize=8, rotation=45, ha="right")
    ax_raw.set_title("Per-question yes-probability", fontsize=12, fontweight="bold")
    ax_raw.grid(axis="y", alpha=0.22)
    ax_raw.legend(frameon=False, fontsize=8, loc="upper right")

    ax_eff.bar(xs, effective_scores, color=eff_colors, alpha=0.92, width=0.72)
    ax_eff.axhline(LOW_CONFIDENCE_THRESHOLD, color="#6C757D", linestyle=":", linewidth=1.0, alpha=0.7)
    ax_eff.axhline(SEMANTIC_PASS_THRESHOLD, color="#8C6D1F", linestyle="--", linewidth=1.1, alpha=0.7)
    ax_eff.axhline(HIGH_CONFIDENCE_PASS_THRESHOLD, color="#1F4E79", linestyle=":", linewidth=1.2, alpha=0.7)
    ax_eff.axhline(OMISSION_FAILURE_THRESHOLD, color="#D97B66", linestyle=":", linewidth=1.0, alpha=0.6)
    ax_eff.axhline(HYBRIDIZATION_FAILURE_THRESHOLD, color="#B07AA1", linestyle=":", linewidth=1.0, alpha=0.6)
    ax_eff.set_ylim(0.0, 1.02)
    ax_eff.set_ylabel("Effective contribution", fontsize=11)
    ax_eff.set_xticks(xs)
    ax_eff.set_xticklabels(_wrapped_labels(probe_rows), fontsize=8, rotation=45, ha="right")
    ax_eff.set_title("Effective score contribution (negative probes raw, not inverted)", fontsize=12, fontweight="bold")
    ax_eff.grid(axis="y", alpha=0.22)

    score_txt = (
        f"joint={aggregate.get('joint_correctness_score')}\n"
        f"cue={aggregate.get('cue_presence_score')}\n"
        f"anti={aggregate.get('anti_collapse_score')}\n"
        f"omission={aggregate.get('omission_score')}\n"
        f"hybrid={aggregate.get('hybrid_score')}\n"
        f"semantic_pass={aggregate.get('semantic_pass')}\n"
        f"high_confidence_pass={aggregate.get('high_confidence_pass')}\n"
        f"omission_failure={aggregate.get('omission_failure')}\n"
        f"hybridization_failure={aggregate.get('hybridization_failure')}"
    )
    ax_eff.text(
        1.01,
        0.98,
        score_txt,
        transform=ax_eff.transAxes,
        va="top",
        ha="left",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8F9FB", "edgecolor": "#D0D7DE"},
    )

    fig.suptitle(
        f"{pair_title}\n{group_key} | {pair_slug} | {condition}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


CONDITION_TO_IMAGE_ALIAS = {
    "monolithic": "mono",
    "poe": "poe",
    "prompt_a": "c1",
    "prompt_b": "c2",
}


def build_probe_inspection_payload(
    *,
    pair_dir: Path,
    condition: str = "monolithic",
    seed: int | None = None,
    question_spec_path: Path | None = None,
    device: str,
) -> dict:
    """Run pair-type-aware probe inspection for a single decoded endpoint."""
    condition_key = CONDITION_TO_IMAGE_ALIAS.get(condition)
    if condition_key is None:
        raise ValueError(f"Unsupported inspection condition '{condition}'.")

    if seed is not None and pair_dir.parent.parent.name.startswith("seed_"):
        group_dir = pair_dir.parent.name
        candidate = pair_dir.parent.parent.parent / f"seed_{seed}" / group_dir / pair_dir.name
        if candidate.exists():
            pair_dir = candidate

    pair_concepts, taxonomy_group_key = _infer_pair_metadata(pair_dir)
    if pair_concepts is None or len(pair_concepts) < 2:
        raise RuntimeError(f"Could not infer pair metadata for {pair_dir}")
    c1, c2 = pair_concepts[0], pair_concepts[1]

    cond_images = _find_images_for_pair(pair_dir, [condition_key])
    seed_map = cond_images.get(condition_key, {})
    if not seed_map:
        raise FileNotFoundError(f"No {condition} image found for {pair_dir}")

    if seed is not None:
        if seed not in seed_map:
            raise KeyError(f"Seed {seed} not available for {pair_dir.name}. Available: {sorted(seed_map)}")
        chosen_seed = int(seed)
    else:
        chosen_seed = sorted(seed_map)[0]
    source = seed_map[chosen_seed]
    image_path = Path(source["path"])

    meta = {
        "pair": f"{c1} + {c2}",
        "c1": c1,
        "c2": c2,
        "pair_slug": pair_dir.name,
        "taxonomy_group_key": taxonomy_group_key,
        "pair_type": pair_type_for_group(taxonomy_group_key),
        "seed": chosen_seed,
        "condition": condition_key,
    }

    if question_spec_path is not None:
        probes = _load_question_spec(question_spec_path, meta)
    else:
        probes = _probe_spec(meta)

    print(f"Loading model {MODEL_ID} on {device} ...")
    processor, model = load_model(device)
    questions = [probe["question"] for probe in probes]

    pil_image = _load_image_from_source(source, {})
    batch_images = [pil_image.copy() for _ in probes]
    try:
        scores = score_batch(batch_images, questions, processor, model, device)
    finally:
        pil_image.close()
        for img in batch_images:
            img.close()

    probe_rows = []
    for probe, score in zip(probes, scores):
        probe_rows.append(
            {
                **meta,
                "probe": probe["probe"],
                "question": probe["question"],
                "probe_positive": probe["positive"],
                "probe_component": probe["component"],
                "score": round(float(score), 6),
            }
        )

    aggregate = _aggregate_scores(meta, probe_rows)
    return {
        "model_id": MODEL_ID,
        "pair_dir": str(pair_dir),
        "image_path": str(image_path),
        "thresholds": {
            "low_confidence": LOW_CONFIDENCE_THRESHOLD,
            "semantic_pass": SEMANTIC_PASS_THRESHOLD,
            "high_confidence": HIGH_CONFIDENCE_PASS_THRESHOLD,
            "hybridization_failure": HYBRIDIZATION_FAILURE_THRESHOLD,
            "omission_failure": OMISSION_FAILURE_THRESHOLD,
        },
        "meta": meta,
        "probes": probe_rows,
        "aggregate": aggregate,
        "bundle": probes,
    }


def save_probe_inspection_artifacts(payload: dict, output_dir: Path) -> None:
    """Save the inspection JSON, question bundle, and styled bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = payload["meta"]
    pair_dir = Path(payload["pair_dir"])
    c1 = meta["c1"]
    c2 = meta["c2"]
    condition = str(meta["condition"])
    display_condition = {
        "mono": "mono",
        "poe": "poe",
        "c1": "prompt_a",
        "c2": "prompt_b",
    }.get(condition, condition)

    stem = "monolithic_probe_inspection" if condition == "mono" else f"{display_condition}_probe_inspection"
    (output_dir / f"{stem}.json").write_text(json.dumps({k: v for k, v in payload.items() if k != "bundle"}, indent=2))
    (output_dir / f"{stem}_bundle.json").write_text(json.dumps(payload["bundle"], indent=2))
    _build_plot(
        image_path=Path(payload["image_path"]),
        pair_title=f"{c1} x {c2}",
        pair_slug=pair_dir.name,
        group_key=str(meta["taxonomy_group_key"]),
        condition=str(meta["condition"]),
        probe_rows=payload["probes"],
        aggregate=payload["aggregate"],
        output_path=output_dir / f"{stem}.png",
    )


def render_probe_inspection_png(payload: dict, output_path: Path) -> None:
    """Render the styled inspection chart to a caller-provided PNG path."""
    meta = payload["meta"]
    pair_dir = Path(payload["pair_dir"])
    c1 = meta["c1"]
    c2 = meta["c2"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _build_plot(
        image_path=Path(payload["image_path"]),
        pair_title=f"{c1} x {c2}",
        pair_slug=pair_dir.name,
        group_key=str(meta["taxonomy_group_key"]),
        condition=str(meta["condition"]),
        probe_rows=payload["probes"],
        aggregate=payload["aggregate"],
        output_path=output_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one pair's monolithic joint-probe bundle and save a bar-chart diagnostic.",
    )
    parser.add_argument(
        "--pair",
        required=True,
        help="Pair slug, e.g. a_typewriter__x__a_cactus",
    )
    parser.add_argument(
        "--group",
        default="",
        help="Optional taxonomy group if you want to disambiguate search.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Screening root containing taxonomy group folders.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device for BLIP-VQA.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed to inspect when multiple mono images exist.",
    )
    parser.add_argument(
        "--question-spec",
        default="",
        help="Optional JSON file containing a replacement question bundle for mono evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Where to save the inspection PNG/JSON. Default: <pair_dir>/probe_inspection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    pair_dir = _resolve_pair_dir(data_dir, args.pair, args.group or None)
    payload = build_probe_inspection_payload(
        pair_dir=pair_dir,
        condition="monolithic",
        seed=args.seed,
        question_spec_path=Path(args.question_spec) if args.question_spec else None,
        device=args.device,
    )
    output_dir = Path(args.output_dir) if args.output_dir else pair_dir / "probe_inspection"
    save_probe_inspection_artifacts(payload, output_dir)
    print(f"Saved -> {output_dir}")


if __name__ == "__main__":
    main()
