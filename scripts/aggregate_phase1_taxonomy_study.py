"""
Aggregate Phase 1 taxonomy outputs into PoE-anchored tidy tables.

This script reads the Phase 1 taxonomy `summary.json` and `trajectory_data.json`
artifacts produced by `trajectory_dynamics_experiment.py`, then writes seed-level
and pair-level tables for:

1. Terminal gap summaries (`d_T^{mono->poe}` plus contextual `c1/c2 -> poe`)
2. Full latent trajectories (`d_t^{mono->poe}` plus contextual baselines)
3. Fixed-window temporal increments (early / mid / late)
4. Optional SD-IPC reachability metrics, if a compatible JSON is supplied

The inferential unit is the concept pair: seeds are first aggregated within each
pair, and the plotting layer can then compare groups using pair means.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from phase1_taxonomy_common import (
    EXPECTED_LAST_STEP,
    PROJECT_ROOT,
    aggregate_rows,
    compute_phase_deltas,
    find_pairwise_series,
    infer_pair_slug,
    infer_taxonomy_group,
    load_json,
    maybe_float,
    maybe_int,
    nested_get,
    pair_sort_key,
    write_csv,
    write_json,
)


TERMINAL_NUMERIC_FIELDS = [
    "d_T_mono_poe",
    "d_T_c1_poe",
    "d_T_c2_poe",
    "d_0_mono_poe",
    "d_17_mono_poe",
    "d_34_mono_poe",
    "d_50_mono_poe",
]

TRAJECTORY_NUMERIC_FIELDS = [
    "d_t_mono_poe",
    "d_t_c1_poe",
    "d_t_c2_poe",
]

PHASE_NUMERIC_FIELDS = [
    "delta_early_mono_poe",
    "delta_mid_mono_poe",
    "delta_late_mono_poe",
    "delta_total_mono_poe",
    "delta_early_c1_poe",
    "delta_mid_c1_poe",
    "delta_late_c1_poe",
    "delta_total_c1_poe",
    "delta_early_c2_poe",
    "delta_mid_c2_poe",
    "delta_late_c2_poe",
    "delta_total_c2_poe",
]

REACHABILITY_NUMERIC_FIELDS = [
    "d_T_sdipc_poe",
    "clip_cosine_sdipc_poe",
    "cycle_close_rate_sdipc_poe",
]

REACHABILITY_TRAJ_NUMERIC_FIELDS = ["d_t_sdipc_poe"]

REACHABILITY_PHASE_NUMERIC_FIELDS = [
    "delta_early_sdipc_poe",
    "delta_mid_sdipc_poe",
    "delta_late_sdipc_poe",
    "delta_total_sdipc_poe",
]


def _coerce_record_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("records"), list):
            return [item for item in payload["records"] if isinstance(item, dict)]
        if isinstance(payload.get("data"), list):
            return [item for item in payload["data"] if isinstance(item, dict)]
    return []


def _load_cycle_consistency(path: Path | None) -> dict[tuple[Any, ...], dict[str, Any]]:
    if path is None or not path.exists():
        return {}

    records = _coerce_record_list(load_json(path))
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        condition = str(record.get("condition", "")).strip().lower()
        key = (
            record.get("taxonomy_group", ""),
            record.get("prompt_a", ""),
            record.get("prompt_b", ""),
            maybe_int(record.get("seed")),
        )
        entry = merged.setdefault(key, {})
        sim = maybe_float(record.get("sim_cycle"))
        broken = record.get("cycle_broken")
        if condition == "poe":
            entry["clip_cosine_sdipc_poe"] = sim if sim is not None else ""
            if broken is not None:
                entry["cycle_close_rate_sdipc_poe"] = 0.0 if bool(broken) else 1.0
        elif condition == "monolithic":
            entry["clip_cosine_sdipc_monolithic"] = sim if sim is not None else ""
            if broken is not None:
                entry["cycle_close_rate_sdipc_monolithic"] = 0.0 if bool(broken) else 1.0
    return merged


def _extract_reachability_series(record: dict[str, Any]) -> list[float] | None:
    candidates = [
        ("d_t_sdipc_poe",),
        ("trajectory", "d_t_sdipc_poe"),
        ("trajectory", "distance_to_poe"),
        ("sdipc_to_poe", "trajectory"),
        ("sdipc_to_poe", "d_t"),
    ]
    for path in candidates:
        value = nested_get(record, path)
        if isinstance(value, list):
            return [float(v) for v in value]
    return None


def _extract_reachability_terminal(record: dict[str, Any], series: list[float] | None) -> float | None:
    candidates = [
        ("d_T_sdipc_poe",),
        ("terminal", "d_T_sdipc_poe"),
        ("terminal_distance_to_poe",),
        ("sdipc_to_poe", "d_T"),
    ]
    for path in candidates:
        value = maybe_float(nested_get(record, path))
        if value is not None:
            return value
    if series:
        return float(series[-1])
    return None


def _extract_reachability_clip(record: dict[str, Any]) -> float | None:
    candidates = [
        ("clip_cosine_sdipc_poe",),
        ("clip_similarity_to_poe",),
        ("clip", "cosine_to_poe"),
        ("sdipc_to_poe", "clip_cosine"),
        ("sim_cycle",),
    ]
    for path in candidates:
        value = maybe_float(nested_get(record, path))
        if value is not None:
            return value
    return None


def _extract_reachability_cycle(record: dict[str, Any]) -> float | None:
    candidates = [
        ("cycle_close_rate_sdipc_poe",),
        ("cycle_close_rate",),
        ("cycle_close",),
        ("cycle_closed",),
    ]
    for path in candidates:
        value = nested_get(record, path)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        numeric = maybe_float(value)
        if numeric is not None:
            return numeric
    broken = record.get("cycle_broken")
    if broken is not None:
        return 0.0 if bool(broken) else 1.0
    return None


def _load_reachability_records(path: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if path is None or not path.exists():
        return [], [], []

    seed_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []

    for record in _coerce_record_list(load_json(path)):
        taxonomy_group = str(record.get("taxonomy_group", "")).strip()
        prompt_a = str(record.get("prompt_a", "")).strip()
        prompt_b = str(record.get("prompt_b", "")).strip()
        seed = maybe_int(record.get("seed"))
        if not taxonomy_group or not prompt_a or not prompt_b or seed is None:
            continue

        pair_slug = str(record.get("pair_slug", "")).strip() or (
            f"{prompt_a}__x__{prompt_b}".replace(" ", "_")
        )
        model_id = str(record.get("model_id", "")).strip()
        series = _extract_reachability_series(record)
        terminal = _extract_reachability_terminal(record, series)
        clip_cos = _extract_reachability_clip(record)
        cycle_rate = _extract_reachability_cycle(record)

        base_row = {
            "taxonomy_group": taxonomy_group,
            "pair_slug": pair_slug,
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "seed": seed,
            "model_id": model_id,
            "d_T_sdipc_poe": terminal if terminal is not None else "",
            "clip_cosine_sdipc_poe": clip_cos if clip_cos is not None else "",
            "cycle_close_rate_sdipc_poe": cycle_rate if cycle_rate is not None else "",
        }
        seed_rows.append(base_row)

        if series is None:
            continue
        if len(series) <= EXPECTED_LAST_STEP:
            continue

        phase = compute_phase_deltas(series)
        phase_rows.append(
            {
                "taxonomy_group": taxonomy_group,
                "pair_slug": pair_slug,
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "seed": seed,
                "model_id": model_id,
                "delta_early_sdipc_poe": phase["delta_early"],
                "delta_mid_sdipc_poe": phase["delta_mid"],
                "delta_late_sdipc_poe": phase["delta_late"],
                "delta_total_sdipc_poe": phase["delta_total"],
            }
        )
        for step, value in enumerate(series):
            trajectory_rows.append(
                {
                    "taxonomy_group": taxonomy_group,
                    "pair_slug": pair_slug,
                    "prompt_a": prompt_a,
                    "prompt_b": prompt_b,
                    "seed": seed,
                    "model_id": model_id,
                    "step": step,
                    "d_t_sdipc_poe": float(value),
                }
            )

    return seed_rows, trajectory_rows, phase_rows


def _build_h1_rows(summary_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]] | None:
    summary = load_json(summary_path)
    cfg = summary.get("config", {})
    run_dir = summary_path.parent
    trajectory_path = run_dir / "trajectory_data.json"
    if not trajectory_path.exists():
        return None

    trajectory = load_json(trajectory_path)
    pairwise_l2 = trajectory.get("pairwise_l2", {})
    mono_series = find_pairwise_series(pairwise_l2, "monolithic", "poe")
    if mono_series is None:
        return None
    if len(mono_series) <= EXPECTED_LAST_STEP:
        raise ValueError(
            f"{trajectory_path} only has {len(mono_series)} steps; expected {EXPECTED_LAST_STEP + 1}."
        )

    c1_series = find_pairwise_series(pairwise_l2, "prompt_a", "poe")
    c2_series = find_pairwise_series(pairwise_l2, "prompt_b", "poe")

    taxonomy_group = infer_taxonomy_group(summary_path, cfg)
    prompt_a = cfg.get("prompt_a", "")
    prompt_b = cfg.get("prompt_b", "")
    seed = maybe_int(cfg.get("seed"))
    if seed is None:
        seed = maybe_int(trajectory.get("config", {}).get("seed"))
    if seed is None:
        seed = -1

    pair_slug = infer_pair_slug(run_dir)
    model_id = cfg.get("model_id", trajectory.get("config", {}).get("model_id", ""))
    num_steps = maybe_int(cfg.get("num_inference_steps"))
    if num_steps is None:
        num_steps = maybe_int(trajectory.get("config", {}).get("num_inference_steps"))

    mono_phase = compute_phase_deltas(mono_series)
    c1_phase = compute_phase_deltas(c1_series) if c1_series is not None else None
    c2_phase = compute_phase_deltas(c2_series) if c2_series is not None else None

    terminal_row = {
        "taxonomy_group": taxonomy_group,
        "pair_slug": pair_slug,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "seed": seed,
        "model_id": model_id,
        "num_inference_steps": num_steps if num_steps is not None else "",
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "trajectory_data_path": str(trajectory_path),
        "d_T_mono_poe": float(mono_series[-1]),
        "d_T_c1_poe": float(c1_series[-1]) if c1_series is not None else "",
        "d_T_c2_poe": float(c2_series[-1]) if c2_series is not None else "",
        "d_0_mono_poe": mono_phase["d_0"],
        "d_17_mono_poe": mono_phase["d_17"],
        "d_34_mono_poe": mono_phase["d_34"],
        "d_50_mono_poe": mono_phase["d_50"],
    }

    trajectory_rows: list[dict[str, Any]] = []
    for step, mono_value in enumerate(mono_series):
        trajectory_rows.append(
            {
                "taxonomy_group": taxonomy_group,
                "pair_slug": pair_slug,
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "seed": seed,
                "model_id": model_id,
                "step": step,
                "d_t_mono_poe": float(mono_value),
                "d_t_c1_poe": float(c1_series[step]) if c1_series is not None else "",
                "d_t_c2_poe": float(c2_series[step]) if c2_series is not None else "",
            }
        )

    phase_row = {
        "taxonomy_group": taxonomy_group,
        "pair_slug": pair_slug,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "seed": seed,
        "model_id": model_id,
        "delta_early_mono_poe": mono_phase["delta_early"],
        "delta_mid_mono_poe": mono_phase["delta_mid"],
        "delta_late_mono_poe": mono_phase["delta_late"],
        "delta_total_mono_poe": mono_phase["delta_total"],
        "delta_early_c1_poe": c1_phase["delta_early"] if c1_phase else "",
        "delta_mid_c1_poe": c1_phase["delta_mid"] if c1_phase else "",
        "delta_late_c1_poe": c1_phase["delta_late"] if c1_phase else "",
        "delta_total_c1_poe": c1_phase["delta_total"] if c1_phase else "",
        "delta_early_c2_poe": c2_phase["delta_early"] if c2_phase else "",
        "delta_mid_c2_poe": c2_phase["delta_mid"] if c2_phase else "",
        "delta_late_c2_poe": c2_phase["delta_late"] if c2_phase else "",
        "delta_total_c2_poe": c2_phase["delta_total"] if c2_phase else "",
    }

    return terminal_row, trajectory_rows, phase_row


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: pair_sort_key(
            str(row.get("taxonomy_group", "")),
            str(row.get("pair_slug", "")),
            maybe_int(row.get("seed")),
            maybe_int(row.get("step")),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Phase 1 taxonomy summary.json + trajectory_data.json into PoE-anchored tidy tables."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="Root taxonomy directory (default: experiments/eccv2026/taxonomy_qualitative).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for tidy tables (default: <input-dir>/phase1_tables).",
    )
    parser.add_argument(
        "--cycle-json",
        type=str,
        default="",
        help="Optional cycle-consistency JSON used as SD-IPC image similarity input.",
    )
    parser.add_argument(
        "--reachability-json",
        type=str,
        default="",
        help=(
            "Optional SD-IPC reachability JSON. Supported records should contain "
            "taxonomy_group, prompt_a, prompt_b, seed, and optionally d_t_sdipc_poe, "
            "d_T_sdipc_poe, clip_cosine_sdipc_poe, cycle_close_rate_sdipc_poe."
        ),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else (
        PROJECT_ROOT / "experiments" / "eccv2026" / "taxonomy_qualitative"
    )
    if not input_dir.exists():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else (input_dir / "phase1_tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    cycle_json = Path(args.cycle_json) if args.cycle_json else (
        PROJECT_ROOT / "experiments" / "eccv2026" / "cycle_consistency" / "cycle_consistency_taxonomy.json"
    )
    if args.cycle_json and not cycle_json.exists():
        print(f"ERROR: cycle JSON not found: {cycle_json}", file=sys.stderr)
        sys.exit(1)
    if not cycle_json.exists():
        cycle_json = None

    reachability_json = Path(args.reachability_json) if args.reachability_json else None
    if reachability_json is not None and not reachability_json.exists():
        print(f"ERROR: reachability JSON not found: {reachability_json}", file=sys.stderr)
        sys.exit(1)

    warnings: list[str] = []
    summary_files = sorted(input_dir.rglob("summary.json"))
    print(f"Found {len(summary_files)} summary.json files under {input_dir}")

    terminal_seed_rows: list[dict[str, Any]] = []
    trajectory_seed_rows: list[dict[str, Any]] = []
    phase_seed_rows: list[dict[str, Any]] = []

    for summary_path in summary_files:
        try:
            built = _build_h1_rows(summary_path)
        except Exception as exc:
            warnings.append(f"{summary_path}: {exc}")
            continue
        if built is None:
            continue
        terminal_row, trajectory_rows, phase_row = built
        terminal_seed_rows.append(terminal_row)
        trajectory_seed_rows.extend(trajectory_rows)
        phase_seed_rows.append(phase_row)

    if not terminal_seed_rows:
        print("ERROR: no valid PoE taxonomy runs found.", file=sys.stderr)
        sys.exit(1)

    cycle_map = _load_cycle_consistency(cycle_json)
    reach_seed_rows, reach_traj_rows, reach_phase_rows = _load_reachability_records(reachability_json)
    reach_map: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in reach_seed_rows:
        key = (
            row.get("taxonomy_group", ""),
            row.get("prompt_a", ""),
            row.get("prompt_b", ""),
            maybe_int(row.get("seed")),
        )
        reach_map[key] = row

    # Merge cycle-consistency similarity into the reachability seed table.
    metadata_by_key = {
        (
            row.get("taxonomy_group", ""),
            row.get("prompt_a", ""),
            row.get("prompt_b", ""),
            maybe_int(row.get("seed")),
        ): row
        for row in terminal_seed_rows
    }
    for key, cycle_fields in cycle_map.items():
        if key not in reach_map:
            base = metadata_by_key.get(key)
            if base is None:
                group, prompt_a, prompt_b, seed = key
                base = {
                    "taxonomy_group": group,
                    "pair_slug": f"{prompt_a}__x__{prompt_b}".replace(" ", "_"),
                    "prompt_a": prompt_a,
                    "prompt_b": prompt_b,
                    "seed": seed if seed is not None else "",
                    "model_id": "",
                    "d_T_sdipc_poe": "",
                    "clip_cosine_sdipc_poe": "",
                    "cycle_close_rate_sdipc_poe": "",
                }
            reach_map[key] = dict(base)
        for field, value in cycle_fields.items():
            if reach_map[key].get(field, "") in ("", None):
                reach_map[key][field] = value

    reachability_seed_rows = _sort_rows(list(reach_map.values()))
    reachability_traj_rows = _sort_rows(reach_traj_rows)
    reachability_phase_rows = _sort_rows(reach_phase_rows)

    terminal_seed_rows = _sort_rows(terminal_seed_rows)
    trajectory_seed_rows = _sort_rows(trajectory_seed_rows)
    phase_seed_rows = _sort_rows(phase_seed_rows)

    terminal_pair_rows = aggregate_rows(
        terminal_seed_rows,
        id_fields=["taxonomy_group", "pair_slug", "prompt_a", "prompt_b"],
        carry_fields=["model_id", "num_inference_steps"],
        numeric_fields=TERMINAL_NUMERIC_FIELDS,
    )
    trajectory_pair_rows = aggregate_rows(
        trajectory_seed_rows,
        id_fields=["taxonomy_group", "pair_slug", "prompt_a", "prompt_b", "step"],
        carry_fields=["model_id"],
        numeric_fields=TRAJECTORY_NUMERIC_FIELDS,
    )
    phase_pair_rows = aggregate_rows(
        phase_seed_rows,
        id_fields=["taxonomy_group", "pair_slug", "prompt_a", "prompt_b"],
        carry_fields=["model_id"],
        numeric_fields=PHASE_NUMERIC_FIELDS,
    )

    reachability_pair_rows = aggregate_rows(
        reachability_seed_rows,
        id_fields=["taxonomy_group", "pair_slug", "prompt_a", "prompt_b"],
        carry_fields=["model_id"],
        numeric_fields=REACHABILITY_NUMERIC_FIELDS,
    ) if reachability_seed_rows else []
    reachability_traj_pair_rows = aggregate_rows(
        reachability_traj_rows,
        id_fields=["taxonomy_group", "pair_slug", "prompt_a", "prompt_b", "step"],
        carry_fields=["model_id"],
        numeric_fields=REACHABILITY_TRAJ_NUMERIC_FIELDS,
    ) if reachability_traj_rows else []
    reachability_phase_pair_rows = aggregate_rows(
        reachability_phase_rows,
        id_fields=["taxonomy_group", "pair_slug", "prompt_a", "prompt_b"],
        carry_fields=["model_id"],
        numeric_fields=REACHABILITY_PHASE_NUMERIC_FIELDS,
    ) if reachability_phase_rows else []

    terminal_seed_fields = [
        "taxonomy_group",
        "pair_slug",
        "prompt_a",
        "prompt_b",
        "seed",
        "model_id",
        "num_inference_steps",
        "run_dir",
        "summary_path",
        "trajectory_data_path",
        *TERMINAL_NUMERIC_FIELDS,
    ]
    terminal_pair_fields = [
        "taxonomy_group",
        "pair_slug",
        "prompt_a",
        "prompt_b",
        "model_id",
        "num_inference_steps",
        "n_rows",
    ]
    for field in TERMINAL_NUMERIC_FIELDS:
        terminal_pair_fields.extend(
            [f"{field}_mean", f"{field}_std", f"{field}_sem", f"{field}_n"]
        )

    trajectory_seed_fields = [
        "taxonomy_group",
        "pair_slug",
        "prompt_a",
        "prompt_b",
        "seed",
        "model_id",
        "step",
        *TRAJECTORY_NUMERIC_FIELDS,
    ]
    trajectory_pair_fields = [
        "taxonomy_group",
        "pair_slug",
        "prompt_a",
        "prompt_b",
        "model_id",
        "step",
        "n_rows",
    ]
    for field in TRAJECTORY_NUMERIC_FIELDS:
        trajectory_pair_fields.extend(
            [f"{field}_mean", f"{field}_std", f"{field}_sem", f"{field}_n"]
        )

    phase_seed_fields = [
        "taxonomy_group",
        "pair_slug",
        "prompt_a",
        "prompt_b",
        "seed",
        "model_id",
        *PHASE_NUMERIC_FIELDS,
    ]
    phase_pair_fields = [
        "taxonomy_group",
        "pair_slug",
        "prompt_a",
        "prompt_b",
        "model_id",
        "n_rows",
    ]
    for field in PHASE_NUMERIC_FIELDS:
        phase_pair_fields.extend(
            [f"{field}_mean", f"{field}_std", f"{field}_sem", f"{field}_n"]
        )

    reach_seed_fields = [
        "taxonomy_group",
        "pair_slug",
        "prompt_a",
        "prompt_b",
        "seed",
        "model_id",
        "d_T_sdipc_poe",
        "clip_cosine_sdipc_poe",
        "cycle_close_rate_sdipc_poe",
    ]
    reach_pair_fields = [
        "taxonomy_group",
        "pair_slug",
        "prompt_a",
        "prompt_b",
        "model_id",
        "n_rows",
    ]
    for field in REACHABILITY_NUMERIC_FIELDS:
        reach_pair_fields.extend(
            [f"{field}_mean", f"{field}_std", f"{field}_sem", f"{field}_n"]
        )

    reach_traj_seed_fields = [
        "taxonomy_group",
        "pair_slug",
        "prompt_a",
        "prompt_b",
        "seed",
        "model_id",
        "step",
        "d_t_sdipc_poe",
    ]
    reach_traj_pair_fields = [
        "taxonomy_group",
        "pair_slug",
        "prompt_a",
        "prompt_b",
        "model_id",
        "step",
        "n_rows",
        "d_t_sdipc_poe_mean",
        "d_t_sdipc_poe_std",
        "d_t_sdipc_poe_sem",
        "d_t_sdipc_poe_n",
    ]

    reach_phase_seed_fields = [
        "taxonomy_group",
        "pair_slug",
        "prompt_a",
        "prompt_b",
        "seed",
        "model_id",
        *REACHABILITY_PHASE_NUMERIC_FIELDS,
    ]
    reach_phase_pair_fields = [
        "taxonomy_group",
        "pair_slug",
        "prompt_a",
        "prompt_b",
        "model_id",
        "n_rows",
    ]
    for field in REACHABILITY_PHASE_NUMERIC_FIELDS:
        reach_phase_pair_fields.extend(
            [f"{field}_mean", f"{field}_std", f"{field}_sem", f"{field}_n"]
        )

    write_csv(output_dir / "phase1_terminal_seed.csv", terminal_seed_fields, terminal_seed_rows)
    write_csv(output_dir / "phase1_terminal_pair.csv", terminal_pair_fields, _sort_rows(terminal_pair_rows))
    write_csv(output_dir / "phase1_trajectory_seed.csv", trajectory_seed_fields, trajectory_seed_rows)
    write_csv(output_dir / "phase1_trajectory_pair.csv", trajectory_pair_fields, _sort_rows(trajectory_pair_rows))
    write_csv(output_dir / "phase1_phase_bins_seed.csv", phase_seed_fields, phase_seed_rows)
    write_csv(output_dir / "phase1_phase_bins_pair.csv", phase_pair_fields, _sort_rows(phase_pair_rows))

    if reachability_seed_rows:
        write_csv(output_dir / "phase1_reachability_seed.csv", reach_seed_fields, reachability_seed_rows)
        write_csv(output_dir / "phase1_reachability_pair.csv", reach_pair_fields, _sort_rows(reachability_pair_rows))
    if reachability_traj_rows:
        write_csv(output_dir / "phase1_reachability_trajectory_seed.csv", reach_traj_seed_fields, reachability_traj_rows)
        write_csv(
            output_dir / "phase1_reachability_trajectory_pair.csv",
            reach_traj_pair_fields,
            _sort_rows(reachability_traj_pair_rows),
        )
    if reachability_phase_rows:
        write_csv(output_dir / "phase1_reachability_phase_bins_seed.csv", reach_phase_seed_fields, reachability_phase_rows)
        write_csv(
            output_dir / "phase1_reachability_phase_bins_pair.csv",
            reach_phase_pair_fields,
            _sort_rows(reachability_phase_pair_rows),
        )

    per_group_pairs: dict[str, set[str]] = {}
    for row in terminal_seed_rows:
        per_group_pairs.setdefault(row["taxonomy_group"], set()).add(row["pair_slug"])
    per_group_seeds: dict[str, set[tuple[str, int]]] = {}
    for row in terminal_seed_rows:
        per_group_seeds.setdefault(row["taxonomy_group"], set()).add((row["pair_slug"], int(row["seed"])))

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "cycle_json": str(cycle_json) if cycle_json is not None else "",
        "reachability_json": str(reachability_json) if reachability_json is not None else "",
        "n_summary_files": len(summary_files),
        "n_seed_records": len(terminal_seed_rows),
        "n_pair_records": len(terminal_pair_rows),
        "group_pair_counts": {group: len(per_group_pairs.get(group, set())) for group in per_group_pairs},
        "group_pair_seed_counts": {group: len(per_group_seeds.get(group, set())) for group in per_group_seeds},
        "phase_bins": [
            {"name": "early", "start": 0, "end": 17},
            {"name": "mid", "start": 17, "end": 34},
            {"name": "late", "start": 34, "end": 50},
        ],
        "warnings": warnings,
    }
    write_json(output_dir / "phase1_aggregation_manifest.json", manifest)

    print(f"Wrote Phase 1 tables → {output_dir}")
    print(f"  Terminal seed rows      : {len(terminal_seed_rows)}")
    print(f"  Terminal pair rows      : {len(terminal_pair_rows)}")
    print(f"  Trajectory seed rows    : {len(trajectory_seed_rows)}")
    print(f"  Phase-bin seed rows     : {len(phase_seed_rows)}")
    print(f"  Reachability seed rows  : {len(reachability_seed_rows)}")
    if warnings:
        print(f"  Warnings                : {len(warnings)}")


if __name__ == "__main__":
    main()
