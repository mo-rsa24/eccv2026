#!/usr/bin/env python3
"""Build a paper-facing report bundle from a taxonomy-driven gap-analysis run."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from plots.groupwise import (
    plot_06_groupwise,
    plot_11_groupwise,
    plot_15_groupwise,
    plot_17_groupwise,
    plot_20_groupwise,
    plot_21_groupwise,
)
from plots.utils import (
    canonical_condition_name,
    get_present_poe,
    get_present_pstar,
    get_present_traj_poe,
    get_present_traj_pstar,
    load_terminal,
    load_trajectory,
    load_within_and,
)
from render_taxonomy_paper_figure import MODE_CONDITIONS, MODE_TITLES, load_panels, render_figure
try:
    from taxonomy_manifest import (
        TARGET_PAIRS_PER_GROUP,
        TOTAL_PAIRS,
        GROUP_ORDER,
        GROUP_LABEL_BY_KEY,
        REPRESENTATIVE_PAIR_SLUGS,
        taxonomy_manifest_payload,
        taxonomy_manifest_rows,
    )
except ImportError:
    from scripts.taxonomy_manifest import (
        TARGET_PAIRS_PER_GROUP,
        TOTAL_PAIRS,
        GROUP_ORDER,
        GROUP_LABEL_BY_KEY,
        REPRESENTATIVE_PAIR_SLUGS,
        taxonomy_manifest_payload,
        taxonomy_manifest_rows,
    )


FIGURE_PLAN = [
    {
        "figure_order": 1,
        "figure_key": "fig01",
        "section": "taxonomy",
        "source_plot": "figure1",
        "paper_export": "trajectory_3x2.png",
        "section_path": "figures/01_taxonomy/fig01_taxonomy_semantic_overview.png",
        "notes": "Representative taxonomy overview with solo prompts, monolithic, and PoE.",
    },
    {
        "figure_order": 2,
        "figure_key": "fig02",
        "section": "taxonomy",
        "source_plot": "figure2",
        "paper_export": "trajectory_3x2_sdipc.png",
        "section_path": "figures/01_taxonomy/fig02_taxonomy_sdipc_closure.png",
        "notes": "Representative taxonomy overview after adding the SD-IPC rerun from p*.",
    },
    {
        "figure_order": 3,
        "figure_key": "fig03",
        "section": "validity",
        "source_plot": "11",
        "paper_export": "plot_11_within_and_noise_floor_groupwise.png",
        "section_path": "figures/02_validity/fig03_within_and_noise_floor_groupwise.png",
        "notes": "Within-AND noise floor versus taxonomy-group distances.",
    },
    {
        "figure_order": 4,
        "figure_key": "fig04",
        "section": "dynamics",
        "source_plot": "06",
        "paper_export": "plot_06_stacked_bar_groupwise.png",
        "section_path": "figures/03_dynamics/fig04_temporal_divergence_groupwise.png",
        "notes": "Temporal divergence accumulation by taxonomy group.",
    },
    {
        "figure_order": 5,
        "figure_key": "fig05",
        "section": "reachability",
        "source_plot": "17",
        "paper_export": "plot_17_pstar_strip_groupwise.png",
        "section_path": "figures/04_reachability/fig05_pstar_terminal_strip_groupwise.png",
        "notes": "Terminal strip plot for SD-IPC versus baselines by taxonomy group.",
    },
    {
        "figure_order": 6,
        "figure_key": "fig06",
        "section": "reachability",
        "source_plot": "15",
        "paper_export": "plot_15_pstar_kde_groupwise.png",
        "section_path": "figures/04_reachability/fig06_pstar_kde_groupwise.png",
        "notes": "Group-wise pooled terminal-distance KDEs for SD-IPC versus baselines.",
    },
    {
        "figure_order": 7,
        "figure_key": "fig07",
        "section": "distribution",
        "source_plot": "20",
        "paper_export": "plot_20_ecdf_groupwise.png",
        "section_path": "figures/05_distribution/fig07_ecdf_groupwise.png",
        "notes": "Group-wise ECDF comparison with within-AND noise floor.",
    },
    {
        "figure_order": 8,
        "figure_key": "fig08",
        "section": "distribution",
        "source_plot": "21",
        "paper_export": "plot_21_jeffreys_heatmap_groupwise.png",
        "section_path": "figures/05_distribution/fig08_jeffreys_heatmap_groupwise.png",
        "notes": "Jeffrey's divergence heatmap by taxonomy group.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paper-facing gap-analysis report bundle.")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Gap-analysis run root containing metrics/, pairs/, and figures/.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Report-bundle output directory. Defaults to <data-dir>/report_bundle.",
    )
    parser.add_argument(
        "--monolithic-baseline",
        choices=["auto", "naive", "natural"],
        default="naive",
        help="Canonical monolithic baseline used when loading run data.",
    )
    parser.add_argument(
        "--taxonomy-view",
        choices=["groupwise"],
        default="groupwise",
        help="Report bundles currently support the paper's groupwise taxonomy view only.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output DPI for rendered report figures.",
    )
    return parser.parse_args()


def _terminal_condition_columns(df_term: pd.DataFrame) -> list[str]:
    return ["d_T_mono", "d_T_c1", "d_T_c2"] + get_present_poe(df_term) + get_present_pstar(df_term)


def _trajectory_condition_columns(df_traj: pd.DataFrame) -> list[str]:
    return ["d_t_mono", "d_t_c1", "d_t_c2"] + get_present_traj_poe(df_traj) + get_present_traj_pstar(df_traj)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _validate_run(df_term: pd.DataFrame, df_traj: pd.DataFrame) -> dict:
    expected_manifest_rows = taxonomy_manifest_rows()
    expected_pairs = {row["pair_slug"] for row in expected_manifest_rows}
    actual_pairs = set(df_term["pair_slug"].dropna().tolist())

    missing = sorted(expected_pairs - actual_pairs)
    unexpected = sorted(actual_pairs - expected_pairs)
    if missing or unexpected:
        raise SystemExit(
            f"Taxonomy run does not match the expected {TOTAL_PAIRS}-pair benchmark.\n"
            f"Missing pairs: {missing}\nUnexpected pairs: {unexpected}"
        )

    seeds_per_pair = df_term.groupby("pair_slug")["seed"].nunique().to_dict()
    unique_seed_counts = sorted(set(seeds_per_pair.values()))
    if len(unique_seed_counts) != 1:
        raise SystemExit(
            "Expected a uniform seed count per pair for the report bundle, got: "
            f"{seeds_per_pair}"
        )
    seeds_per_pair_count = int(unique_seed_counts[0])
    validation_profile = "paper" if seeds_per_pair_count == 24 else "smoke"

    group_pair_counts = df_term.groupby("taxonomy_group_key")["pair_slug"].nunique().to_dict()
    bad_group_pairs = {
        key: n for key, n in group_pair_counts.items() if n != TARGET_PAIRS_PER_GROUP
    }
    if bad_group_pairs:
        raise SystemExit(
            f"Expected {TARGET_PAIRS_PER_GROUP} pairs per taxonomy group, got: {bad_group_pairs}"
        )

    group_record_counts = df_term.groupby("taxonomy_group_key").size().to_dict()
    expected_group_records = TARGET_PAIRS_PER_GROUP * seeds_per_pair_count
    bad_group_records = {
        key: n for key, n in group_record_counts.items() if n != expected_group_records
    }
    if bad_group_records:
        raise SystemExit(
            "Expected one record per (pair, seed) within each taxonomy group, got: "
            f"{bad_group_records}"
        )

    traj_pair_counts = df_traj.groupby("taxonomy_group_key")["pair_slug"].nunique().to_dict()
    if any(traj_pair_counts.get(key, 0) != TARGET_PAIRS_PER_GROUP for key in GROUP_ORDER):
        raise SystemExit(f"Trajectory data does not cover all taxonomy groups: {traj_pair_counts}")

    return {
        "validation_profile": validation_profile,
        "n_pairs": len(actual_pairs),
        "n_records_terminal": int(len(df_term)),
        "n_records_trajectory": int(len(df_traj)),
        "seeds_per_pair": seeds_per_pair,
        "seed_count_per_pair": seeds_per_pair_count,
        "group_record_counts": group_record_counts,
    }


def _phase_steps(df_traj: pd.DataFrame) -> tuple[int, int, int, int]:
    steps = sorted(int(x) for x in df_traj["step"].unique())
    start = steps[0]
    last = steps[-1]
    early = 17 if 17 in steps else steps[min(len(steps) - 1, max(1, len(steps) // 3))]
    mid = 34 if 34 in steps else steps[min(len(steps) - 1, max(2, 2 * len(steps) // 3))]
    return start, int(early), int(mid), int(last)


def _build_group_terminal_summary(df_term: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_key in GROUP_ORDER:
        sub = df_term[df_term["taxonomy_group_key"] == group_key]
        for cond in _terminal_condition_columns(df_term):
            vals = sub[cond].dropna()
            if vals.empty:
                continue
            rows.append(
                {
                    "taxonomy_group_key": group_key,
                    "taxonomy_group_label": GROUP_LABEL_BY_KEY[group_key],
                    "condition": canonical_condition_name(cond),
                    "n_pairs": int(sub["pair_slug"].nunique()),
                    "n_records": int(len(vals)),
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "median": float(vals.median()),
                    "q25": float(vals.quantile(0.25)),
                    "q75": float(vals.quantile(0.75)),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }
            )
    return pd.DataFrame(rows)


def _build_pair_terminal_summary(df_term: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pair_order = {row["pair_slug"]: idx for idx, row in enumerate(taxonomy_manifest_rows())}
    for pair_slug, pair_sub in sorted(df_term.groupby("pair_slug"), key=lambda item: pair_order[item[0]]):
        first = pair_sub.iloc[0]
        for cond in _terminal_condition_columns(df_term):
            vals = pair_sub[cond].dropna()
            if vals.empty:
                continue
            rows.append(
                {
                    "taxonomy_group_key": first["taxonomy_group_key"],
                    "taxonomy_group_label": first["taxonomy_group_label"],
                    "pair_slug": pair_slug,
                    "prompt_a": first["c1"],
                    "prompt_b": first["c2"],
                    "condition": canonical_condition_name(cond),
                    "n_seeds": int(pair_sub["seed"].nunique()),
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "median": float(vals.median()),
                }
            )
    return pd.DataFrame(rows)


def _build_group_phase_summary(df_traj: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    start, early, mid, last = _phase_steps(df_traj)
    phase_rows = []
    traj_conditions = _trajectory_condition_columns(df_traj)

    for group_key in GROUP_ORDER:
        group_sub = df_traj[df_traj["taxonomy_group_key"] == group_key]
        for cond in traj_conditions:
            per_seed = []
            for (_pair_slug, _seed), pair_seed_sub in group_sub.groupby(["pair_slug", "seed"]):
                by_step = pair_seed_sub.set_index("step")[cond]
                if start not in by_step.index or early not in by_step.index or mid not in by_step.index or last not in by_step.index:
                    continue
                per_seed.extend(
                    [
                        {"phase": "early", "value": float(by_step.loc[early] - by_step.loc[start])},
                        {"phase": "mid", "value": float(by_step.loc[mid] - by_step.loc[early])},
                        {"phase": "late", "value": float(by_step.loc[last] - by_step.loc[mid])},
                    ]
                )

            if not per_seed:
                continue

            phase_df = pd.DataFrame(per_seed)
            for phase, phase_sub in phase_df.groupby("phase"):
                vals = phase_sub["value"]
                phase_rows.append(
                    {
                        "taxonomy_group_key": group_key,
                        "taxonomy_group_label": GROUP_LABEL_BY_KEY[group_key],
                        "condition": canonical_condition_name(cond),
                        "phase": phase,
                        "n_pairs": int(group_sub["pair_slug"].nunique()),
                        "n_records": int(len(vals)),
                        "mean": float(vals.mean()),
                        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                        "median": float(vals.median()),
                    }
                )

    return pd.DataFrame(phase_rows), {
        "start": start,
        "early": early,
        "mid": mid,
        "late": last,
    }


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "report_bundle"
    manifest_dir = output_dir / "manifest"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    paper_exports_dir = output_dir / "paper_exports"

    for directory in (manifest_dir, figures_dir, tables_dir, paper_exports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    df_term = load_terminal(data_dir, monolithic_baseline=args.monolithic_baseline, and_anchor="seed")
    df_traj = load_trajectory(data_dir, monolithic_baseline=args.monolithic_baseline)
    within_records = load_within_and(data_dir) or []
    validation = _validate_run(df_term, df_traj)

    # Representative taxonomy figures
    panels = load_panels(data_dir, list(REPRESENTATIVE_PAIR_SLUGS), args.monolithic_baseline, None)
    figure1_export = paper_exports_dir / "trajectory_3x2.png"
    figure2_export = paper_exports_dir / "trajectory_3x2_sdipc.png"
    render_figure(panels, MODE_CONDITIONS["figure1"], figure1_export, MODE_TITLES["figure1"], dpi=args.dpi)
    render_figure(panels, MODE_CONDITIONS["figure2"], figure2_export, MODE_TITLES["figure2"], dpi=args.dpi)

    # Group-wise quantitative figures
    shared_plot_kw = {
        "data_dir": data_dir,
        "within_and_records": within_records,
    }
    plot_11_groupwise(df_term, df_traj, paper_exports_dir, **shared_plot_kw)
    plot_06_groupwise(df_term, df_traj, paper_exports_dir, **shared_plot_kw, pstar_filter=frozenset())
    plot_17_groupwise(df_term, df_traj, paper_exports_dir, **shared_plot_kw)
    plot_15_groupwise(df_term, df_traj, paper_exports_dir, **shared_plot_kw)
    plot_20_groupwise(df_term, df_traj, paper_exports_dir, **shared_plot_kw)
    plot_21_groupwise(df_term, df_traj, paper_exports_dir, **shared_plot_kw)

    # Section-organized copies
    for spec in FIGURE_PLAN:
        src = paper_exports_dir / spec["paper_export"]
        if not src.exists():
            raise SystemExit(f"Expected figure not found: {src}")
        _copy(src, output_dir / spec["section_path"])

    taxonomy_df = pd.DataFrame(taxonomy_manifest_rows())
    group_terminal_df = _build_group_terminal_summary(df_term)
    group_phase_df, phase_steps = _build_group_phase_summary(df_traj)
    pair_terminal_df = _build_pair_terminal_summary(df_term)
    figure_inventory_df = pd.DataFrame(
        [
            {
                **spec,
                "relative_path": spec["section_path"],
                "taxonomy_view": args.taxonomy_view,
            }
            for spec in FIGURE_PLAN
        ]
    )[
        ["figure_order", "figure_key", "section", "source_plot", "relative_path", "taxonomy_view", "notes"]
    ]

    _write_csv(taxonomy_df, manifest_dir / "taxonomy_manifest.csv")
    _write_csv(figure_inventory_df, manifest_dir / "figure_inventory.csv")

    _write_csv(taxonomy_df, tables_dir / "table01_taxonomy_manifest.csv")
    _write_csv(group_terminal_df, tables_dir / "table02_group_terminal_summary.csv")
    _write_csv(group_phase_df, tables_dir / "table03_group_phase_summary.csv")
    _write_csv(pair_terminal_df, tables_dir / "table04_pair_terminal_summary.csv")
    _write_csv(figure_inventory_df, tables_dir / "table05_figure_inventory.csv")

    report_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "monolithic_baseline": args.monolithic_baseline,
        "taxonomy_view": args.taxonomy_view,
        "validation": validation,
        "phase_steps": phase_steps,
        "taxonomy_manifest": taxonomy_manifest_payload(),
        "figure_inventory": figure_inventory_df.to_dict(orient="records"),
        "paper_exports": sorted(path.name for path in paper_exports_dir.glob("*.png")),
    }
    (manifest_dir / "report_manifest.json").write_text(json.dumps(report_manifest, indent=2))

    print(f"Report bundle written to {output_dir}")


if __name__ == "__main__":
    main()
