"""
Shared helpers for the Phase 1 taxonomy aggregation and plotting pipeline.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

GROUP_ORDER = [
    "group1_cooccurrence",
    "group2_disentangled",
    "group3_ood",
    "group4_collision",
]

GROUP_LABELS = {
    "group1_cooccurrence": "G1\nCo-occurrence",
    "group2_disentangled": "G2\nDisentangled",
    "group3_ood": "G3\nOOD",
    "group4_collision": "G4\nCollision",
}

GROUP_TITLES = {
    "group1_cooccurrence": "Group 1: Co-occurrence",
    "group2_disentangled": "Group 2: Disentangled",
    "group3_ood": "Group 3: OOD",
    "group4_collision": "Group 4: Collision",
}

GROUP_COLORS = {
    "group1_cooccurrence": "#4878CF",
    "group2_disentangled": "#6ACC65",
    "group3_ood": "#F5A623",
    "group4_collision": "#D7191C",
}

PHASE_BINS = [
    ("early", 0, 17),
    ("mid", 17, 34),
    ("late", 34, 50),
]

PHASE_LABELS = {
    "early": "Early (0-17)",
    "mid": "Mid (17-34)",
    "late": "Late (34-50)",
}

PHASE_COLORS = {
    "early": "#4C78A8",
    "mid": "#F58518",
    "late": "#E45756",
}

PHASE_SPAN_COLORS = {
    "early": "#DCE9F6",
    "mid": "#FDE7CC",
    "late": "#F9D6D8",
}

EXPECTED_LAST_STEP = 50


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def present_groups(rows: Iterable[dict[str, Any]], field: str = "taxonomy_group") -> list[str]:
    present = {row.get(field, "") for row in rows}
    return [group for group in GROUP_ORDER if group in present]


def pair_sort_key(group: str, pair_slug: str, seed: int | None = None, step: int | None = None) -> tuple:
    group_rank = GROUP_ORDER.index(group) if group in GROUP_ORDER else 99
    key = [group_rank, pair_slug]
    if seed is not None:
        key.append(seed)
    if step is not None:
        key.append(step)
    return tuple(key)


def infer_pair_slug(run_dir: Path) -> str:
    if run_dir.name.startswith("seed_") and "__x__" in run_dir.parent.name:
        return run_dir.parent.name
    return run_dir.name


def infer_taxonomy_group(summary_path: Path, cfg: dict[str, Any]) -> str:
    taxonomy_group = cfg.get("taxonomy_group", "")
    if taxonomy_group:
        return taxonomy_group
    for part in summary_path.parts:
        if part.startswith("group") and "_" in part:
            return part
    return ""


def find_pairwise_series(pairwise_l2: dict[str, Any], left: str, right: str) -> list[float] | None:
    direct = f"{left}|{right}"
    reverse = f"{right}|{left}"
    values = pairwise_l2.get(direct)
    if values is None:
        values = pairwise_l2.get(reverse)
    if values is None:
        return None
    return [float(v) for v in values]


def compute_phase_deltas(series: list[float]) -> dict[str, float]:
    if len(series) <= EXPECTED_LAST_STEP:
        raise ValueError(
            f"Expected at least {EXPECTED_LAST_STEP + 1} trajectory points, got {len(series)}."
        )
    d0 = float(series[0])
    d17 = float(series[17])
    d34 = float(series[34])
    d50 = float(series[50])
    return {
        "d_0": d0,
        "d_17": d17,
        "d_34": d34,
        "d_50": d50,
        "delta_early": d17 - d0,
        "delta_mid": d34 - d17,
        "delta_late": d50 - d34,
        "delta_total": d50 - d0,
    }


def mean_std_sem(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, 0.0, 0.0
    std = float(arr.std(ddof=1))
    sem = float(std / math.sqrt(arr.size))
    return mean, std, sem


def ci95(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    mean, _, sem = mean_std_sem(values)
    if mean is None or sem is None:
        return None, None, None
    if len(values) <= 1:
        return mean, mean, mean
    critical = 1.96
    try:
        from scipy.stats import t as student_t

        critical = float(student_t.ppf(0.975, df=len(values) - 1))
    except Exception:
        critical = 1.96
    delta = critical * sem
    return mean, mean - delta, mean + delta


def nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    id_fields: list[str],
    carry_fields: list[str],
    numeric_fields: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in id_fields)
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for key, bucket in grouped.items():
        out = {field: value for field, value in zip(id_fields, key)}
        for field in carry_fields:
            out[field] = bucket[0].get(field, "")
        out["n_rows"] = len(bucket)
        for field in numeric_fields:
            values = []
            for row in bucket:
                value = maybe_float(row.get(field))
                if value is not None:
                    values.append(value)
            mean, std, sem = mean_std_sem(values)
            out[f"{field}_mean"] = mean if mean is not None else ""
            out[f"{field}_std"] = std if std is not None else ""
            out[f"{field}_sem"] = sem if sem is not None else ""
            out[f"{field}_n"] = len(values)
        aggregated.append(out)
    return aggregated


def ecdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0:
        return np.asarray([]), np.asarray([])
    ys = np.arange(1, arr.size + 1, dtype=float) / arr.size
    return arr, ys
