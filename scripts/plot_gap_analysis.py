#!/usr/bin/env python3
"""
Gap analysis visualisation suite — Plots 00 through 34.

Each plot builds on the previous to form a progressive statistical narrative:

  Terminal distributions  (data: per_seed_distances.json)
    00  Strip by pair     — raw dots (one per seed) + bold mean tick, 4 panels  ← foundation
    01  Grouped bar       — mean ± std per pair per condition
    02  Hist grouped      — x=distance, grouped bars per bin (colour=pair), 3 condition panels
    03  Pooled strip      — all seeds pooled across pairs, bold tick=mean, colour=condition
    04  KDE pooled        — smooth of plot 03; all pairs combined, colour=condition
    05  KDE by pair       — 4 panels, one per concept pair, colour=condition

  Temporal dynamics  (data: trajectory_distances.json)
    06  Stacked bar pooled  — incremental divergence per time bin, pooled across all pairs
    07  Stacked bar by pair — same, one group per concept pair
    08  Per-condition       — 3 subplots, 4 pair-coloured mean ± std bands
    09  Individual seeds    — same as 08, thin per-seed traces + bold mean
    10  Generalised         — 3 subplots (one per condition), all pooled,
                              mean with ±1 std outer band and 95 % CI inner band
    26  Combined 04+06      — side-by-side pooled KDE (left) and temporal stack (right)

  Gap validity  (data: within_and_distances.json)
    11  Within-AND noise floor — pairwise AND distances vs gap conditions (×ratio)
                                 Confirms the gap is structural, not stochastic

  p* progressive abstraction  (data: per_seed_distances.json; mirrors 00–10)
    12  p* strip by pair    — raw seed dots per p* source + mono, 4 pair panels  ← foundation
    13  p* grouped bars     — mean ± std per pair, bars = p* sources + mono
    14  p* histogram        — distance histogram per pair, bars by p* source
    15  p* KDE pooled       — smooth of all conditions (p* solid, baselines dashed)
                              Key visual: overlap with mono ≈ gap closed
    16  p* temporal         — per-step MSE for all p* sources + mono, pooled, mean+CI

  Closing the gap — synthesis  (data: per_seed_distances.json + all_pairs_gap.json)
    17  p* terminal strip   — terminal distance from AND: all p* sources vs baselines
    18  CLIP comparison     — per-pair CLIP sim AND↔p* vs AND↔mono (all p* sources)
    19  JS² divergence      — JS²(P_p*, P_mono) per pair; lower = p* closer to mono
    25  Combined 17+15      — side-by-side strip (left) and KDE (right), equal panel size

  Distributional analysis — Jeffrey's divergence  (data: per_seed + within_and)
    20  ECDF comparison     — F(x) for all conditions + within-AND; non-parametric
                              Left shift = closer to AND; overlap with grey = gap closed
    21  Jeffrey's heatmap   — J(P_i, P_j) for all condition pairs; cluster structure
                              p* clustering with within-AND → gap closed
    22  Expressiveness ladder — J(p* source, within-AND) vs method expressiveness rank
                              THE paper figure: downward trend = language can express AND
    23  p* vs AND KL         — p* sources only, KL to SuperDiff-AND proxy
    24  Pointwise KL/Jeffrey — shaded integrands, area = divergence

  Plot-24 scaffolding (simple → full)  (data: per_seed + within_and)
    29  Raw pooled samples   — where d_T observations come from
    30  Shared-bin histogram — convert samples to comparable discrete mass
    34  Shared-bin KL bridge — p_i/q_i → log-ratio → local KL → cumulative KL
    31  KDE overlays         — smooth shared-grid densities
    32  Pointwise KL terms   — local signed KL contributions by d_T
    33  Cumulative KL curves — running integral ending at total KL

  Grid reconstruction  (data: pairs/*/grid_assets.json)
    27  Decoded image grid     — selected p* columns (SD-IPC, learned inverter, legacy Z2T/VLM)
    28  Trajectory manifold    — selected p* trajectories (SD-IPC, learned inverter, legacy Z2T/VLM)

Usage
-----
# All plots:
conda run -n superdiff python scripts/plot_gap_analysis.py

# Single plot (both forms accepted):
conda run -n superdiff python scripts/plot_gap_analysis.py --plot 06
conda run -n superdiff python scripts/plot_gap_analysis.py --plot 18

# Custom data / output directories:
conda run -n superdiff python scripts/plot_gap_analysis.py \\
    --data-dir   experiments/inversion/gap_analysis \\
    --output-dir experiments/inversion/gap_analysis/figures \\
    --plot all

Notes
-----
Plot 11 requires within_and_distances.json — generated automatically by measure_composability_gap.py.
Plots 12–19 and 25 require d_T_pstar_* in per_seed_distances.json.
Re-run measure_composability_gap.py with --pstar-source {sdipc,inverter,pez} (z2t/vlm are legacy optional).
Use --merge to accumulate multiple p* sources into the same JSON without overwriting.
Plot 18 uses all_pairs_gap.json (CLIP scores computed in the main run).
Plot 16 requires d_t_pstar_* in trajectory_distances.json.
Plots 27–28 require pairs/*/grid_assets.json (exported by measure_composability_gap.py).
"""

import argparse
import sys
from pathlib import Path

# Make scripts/plots/ importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plots.baseline import (
    plot_00, plot_01, plot_02, plot_03, plot_04, plot_05,
    plot_06, plot_07, plot_08, plot_09, plot_10, plot_26,
)
from plots.pstar import (
    plot_11, plot_12, plot_13, plot_14, plot_15, plot_16,
    plot_17, plot_18, plot_19, plot_25, plot_27, plot_28,
)
from plots.distributional import plot_20, plot_21, plot_22
from plots.distributional import (
    plot_23, plot_24, plot_29, plot_30, plot_31, plot_32, plot_33, plot_34,
)
from plots.groupwise import (
    plot_06_groupwise,
    plot_11_groupwise,
    plot_15_groupwise,
    plot_17_groupwise,
    plot_reachability_semantic_groupwise,
    plot_20_groupwise,
    plot_21_groupwise,
)
from plots.semantic_audit import (
    plot_group34_boundary_audit,
    plot_monolithic_semantic_audit,
    plot_pair_retention_summary,
    plot_semantic_filter_robustness,
    plot_semantic_pass_rate_comparison,
)
from plots.utils import (
    filter_semantic_baseline_scope,
    load_terminal,
    load_trajectory,
    PSTAR_PRIORITY,
    TERM_LABEL,
)
from plots.blip_vqa_bar import load_blip_vqa, plot_blip_vqa_grouped_bar
from plots.joint_probe_bar import load_joint_probes, plot_joint_probe_grouped_bar


# ===========================================================================
# Registry and CLI
# ===========================================================================

GROUPS = {
    "baseline":      [f"{i:02d}" for i in range(0, 6)],    # 00–05
    "temporal":      [f"{i:02d}" for i in range(6, 11)],   # 06–10
    "gap-validity":  ["11"],                                # 11
    "pstar":         [f"{i:02d}" for i in range(12, 17)],  # 12–16
    "synthesis":     [f"{i:02d}" for i in range(17, 20)] + ["25", "26"],  # 17–19, 25–26
    "distributional":[f"{i:02d}" for i in range(20, 25)],  # 20–24
    "pointwise-scaffold": [f"{i:02d}" for i in range(29, 35)],  # 29–34
    "grid":          ["27", "28"],                          # 27–28
}

PLOTS = {
    "00": ("Strip chart by pair (foundation)",              plot_00),
    "01": ("Terminal grouped bars",                         plot_01),
    "02": ("Grouped bar histogram by pair",                 plot_02),
    "03": ("Pooled strip (all pairs combined)",             plot_03),
    "04": ("KDE pooled",                                    plot_04),
    "05": ("KDE by pair",                                   plot_05),
    "06": ("Stacked temporal bar — pooled",                 plot_06),
    "07": ("Stacked temporal bar — by pair",                plot_07),
    "08": ("Per-condition temporal detail",                 plot_08),
    "09": ("Individual seed traces + mean",                 plot_09),
    "10": ("Generalised per-condition",                     plot_10),
    "26": ("Combined: plot 04 (left) + plot 06 (right)  [core]", plot_26),
    # --- gap validity (no p* dependency) ---
    "11": ("Within-AND noise-floor validation  [validity]", plot_11),
    # --- p* progressive abstraction sequence (mirrors 00–10) ---
    "12": ("p* raw seed strip by pair  [p*-foundation]",    plot_12),
    "13": ("p* grouped bars mean±std  [p*-aggregation]",    plot_13),
    "14": ("p* distance histogram by pair  [p*-dist]",      plot_14),
    "15": ("p* pooled KDE all conditions  [p*-smooth]",     plot_15),
    "16": ("p* temporal trajectory  [p*-temporal]",         plot_16),
    # --- p* synthesis / conclusions ---
    "17": ("p* terminal strip  [close-it]",                 plot_17),
    "18": ("CLIP comparison bar  [close-it]",               plot_18),
    "19": ("JS² divergence: p* vs mono  [close-it]",        plot_19),
    "25": ("Combined: plot 17 (left) + plot 15 (right)  [close-it]", plot_25),
    # --- distributional analysis (Jeffrey's divergence) ---
    "20": ("ECDF all conditions + within-AND  [distrib]",   plot_20),
    "21": ("Jeffrey's divergence heatmap  [distrib]",       plot_21),
    "22": ("Expressiveness ladder  [distrib-key]",          plot_22),
    "23": ("p* vs AND KL (p*-only)  [distrib-kl]",          plot_23),
    "24": ("Pointwise KL + Jeffrey contributions  [distrib-int]", plot_24),
    # --- plot-24 scaffolding (simple → full) ---
    "29": ("Pointwise scaffold 1/5: raw pooled d_T samples", plot_29),
    "30": ("Pointwise scaffold 2/5: shared-bin histograms",  plot_30),
    "34": ("Pointwise scaffold 2.5/5: shared-bin KL factorization bridge", plot_34),
    "31": ("Pointwise scaffold 3/5: KDE overlays + signed diff", plot_31),
    "32": ("Pointwise scaffold 4/5: local KL terms",         plot_32),
    "33": ("Pointwise scaffold 5/5: cumulative KL area",     plot_33),
    # --- qualitative grid reconstructions ---
    "27": ("Decoded image grid with selected p* columns  [grid]",    plot_27),
    "28": ("Trajectory manifold grid with selected p* trajectories  [grid]", plot_28),
}

GROUPWISE_PLOTS = {
    "06": plot_06_groupwise,
    "11": plot_11_groupwise,
    "15": plot_15_groupwise,
    "17": plot_17_groupwise,
    "20": plot_20_groupwise,
    "21": plot_21_groupwise,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Gap analysis figures 00–34",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir", default="experiments/inversion/gap_analysis",
        help="Run root from measure_composability_gap.py.  JSONs are read from "
             "{data-dir}/metrics/ if present, otherwise {data-dir}/ directly "
             "(backward-compatible with pre-timestamped runs).  "
             "Example: experiments/inversion/gap_analysis/small_20260302_143000",
    )
    p.add_argument(
        "--output-dir", default="",
        help="Directory to write PNG files.  Defaults to {data-dir}/figures/ if not set.",
    )
    p.add_argument(
        "--max-display-pairs", type=int, default=None, metavar="N",
        help="Cap per-pair panel plots at N pairs (randomly sampled, fixed seed). "
             "Pooled plots always use all pairs. Recommended for --regime large (use 4).",
    )
    p.add_argument(
        "--pstar-sources",
        nargs="+",
        default=["auto"],
        metavar="SRC",
        help=(
            "Which p* sources to include in baseline plots 00–10 and grid plots 27–28.\n"
            "  auto              — trusted defaults (sdipc/inverter/pez) when present\n"
            "  none              — hide all p* from baseline plots\n"
            "  sdipc inverter pez z2t vlm  — show specific sources (space-separated)\n"
            "Plots 11–25 and 29–34 use the script's p* auto-detection (legacy Z2T is not in default mixes).\n"
            "Plot 26 follows baseline filtering (same behavior as plots 00–10). "
            "Plots 27–28 apply this filter to supported qualitative p* columns "
            "(SD-IPC, learned inverter, legacy Z2T/VLM)."
        ),
    )
    p.add_argument(
        "--monolithic-baseline",
        choices=["auto", "naive", "natural"],
        default="auto",
        help=(
            "Which monolithic baseline to treat as canonical in plots. "
            "'auto' uses d_T_mono/d_t_mono/gap_and_mono as stored. "
            "'naive' remaps to *_mono_naive, 'natural' remaps to *_mono_natural "
            "when those columns/keys exist."
        ),
    )
    p.add_argument(
        "--and-anchor",
        choices=["seed", "mean"],
        default="seed",
        help=(
            "Terminal-distance anchor used by plots that read d_T_* columns. "
            "'seed' uses per-seed paired AND (default). "
            "'mean' remaps to *_meananchor columns when available "
            "(e.g., d_T_mono_meananchor, d_T_c1_meananchor, d_T_c2_meananchor)."
        ),
    )
    p.add_argument(
        "--paper-size-preset",
        choices=["default", "plot26_large_traj_small"],
        default="default",
        help=(
            "Paper-facing figure size preset. "
            "'plot26_large_traj_small' enlarges plot 26 and shrinks "
            "trajectory_manifold_grid in a single switch."
        ),
    )
    p.add_argument(
        "--plot04-separate",
        action="store_true",
        help=(
            "Clarity preset for plot 04/26 KDEs: zoom x-range to central mass, "
            "slightly sharpen KDE bandwidth, and enlarge plot 04."
        ),
    )
    p.add_argument(
        "--plot04-xmax-quantile",
        type=float,
        default=1.0,
        help=(
            "Upper x-limit quantile for plot 04/26 KDEs (0 < q <= 1). "
            "Example: 0.97 trims long right tails for better separation."
        ),
    )
    p.add_argument(
        "--plot04-bw-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplicative scale on Scott KDE bandwidth for plot 04/26. "
            "<1 sharpens curves; >1 smooths more."
        ),
    )
    p.add_argument(
        "--plot04-scale",
        type=float,
        default=1.0,
        help="Figure size scale for standalone plot 04.",
    )
    group_names = " | ".join(GROUPS)
    p.add_argument(
        "--plot", default="all",
        help=(
            "Which plot(s) to generate.  Options:\n"
            "  all            — every plot (default)\n"
            f"  {group_names}\n"
            "  06 / 13 …      — single plot number\n"
            "  blip_vqa       — BLIP-VQA concept-presence bar charts (F0 and F0b)\n"
            "  joint_probes   — pair-type-aware joint-correctness bar charts\n"
            "  reachability_semantic — semantics-qualified p* reachability figure\n"
            "  semantic_audit — monolithic semantic-baseline audit by group\n"
            "  semantic_retention — pair retention / provisional summary\n"
            "  semantic_pass_compare — mono vs PoE semantic pass-rate comparison\n"
            "  semantic_boundary — Group 3 vs Group 4 boundary audit\n"
            "  semantic_robustness — full vs semantics-qualified robustness summary"
        ),
    )
    p.add_argument(
        "--taxonomy-view",
        choices=["pooled", "groupwise", "both"],
        default="pooled",
        help=(
            "How to render taxonomy-aware plots 06, 11, 15, 17, 20, and 21. "
            "'pooled' keeps the legacy outputs, 'groupwise' renders taxonomy panels "
            "for all present groups in canonical order, and 'both' writes both forms."
        ),
    )
    p.add_argument(
        "--semantic-baseline-scope",
        choices=["full", "seed_qualified", "pair_qualified"],
        default="full",
        help=(
            "Filter scope driven by the monolithic semantic-baseline audit. "
            "'full' keeps all records, 'seed_qualified' keeps only mono-qualified "
            "pair-seed records inside mono-qualified pairs, and 'pair_qualified' keeps "
            "all records from mono-qualified pairs."
        ),
    )
    p.add_argument(
        "--mono-pass-threshold",
        type=float,
        default=0.75,
        help="Pair-level monolithic pass-rate threshold for semantic qualification.",
    )
    p.add_argument(
        "--mono-seed-gate",
        choices=["semantic", "high_confidence"],
        default="semantic",
        help="Seed-level monolithic gate used by semantic-baseline qualification.",
    )
    return p.parse_args()


def main():
    args     = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir) if args.output_dir else data_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = args.plot.strip().lower()

    # Routes that do not require per_seed_distances / trajectory_distances.
    if target == "semantic_audit":
        print("\n[semantic_audit] Monolithic semantic-baseline audit by group")
        plot_monolithic_semantic_audit(data_dir, out_dir)
        print(f"\nDone. Figures written to {out_dir}/")
        return

    if target == "semantic_retention":
        print("\n[semantic_retention] Pair retention / provisional summary")
        plot_pair_retention_summary(data_dir, out_dir)
        print(f"\nDone. Figures written to {out_dir}/")
        return

    if target == "semantic_pass_compare":
        print("\n[semantic_pass_compare] Monolithic vs PoE semantic pass-rate comparison")
        plot_semantic_pass_rate_comparison(data_dir, out_dir)
        print(f"\nDone. Figures written to {out_dir}/")
        return

    if target == "semantic_boundary":
        print("\n[semantic_boundary] Group 3 vs Group 4 boundary audit")
        plot_group34_boundary_audit(data_dir, out_dir)
        print(f"\nDone. Figures written to {out_dir}/")
        return

    print(f"Loading data from {data_dir} ...")
    df_term = load_terminal(
        data_dir,
        monolithic_baseline=args.monolithic_baseline,
        and_anchor=args.and_anchor,
    )
    df_traj = load_trajectory(data_dir, monolithic_baseline=args.monolithic_baseline)
    print(f"  Monolithic baseline mode: {args.monolithic_baseline}")
    print(f"  Terminal anchor mode (d_T_*): {args.and_anchor}")
    print(f"  Terminal:   {len(df_term)} records "
          f"({df_term['pair'].nunique()} pairs × {df_term['seed'].nunique()} seeds)")
    print(f"  Trajectory: {len(df_traj)} records")
    print(f"  Taxonomy view: {args.taxonomy_view}")
    if args.semantic_baseline_scope != "full":
        df_term = filter_semantic_baseline_scope(
            df_term,
            data_dir,
            scope=args.semantic_baseline_scope,
            mono_pass_threshold=args.mono_pass_threshold,
            mono_seed_gate=args.mono_seed_gate,
        )
        df_traj = filter_semantic_baseline_scope(
            df_traj,
            data_dir,
            scope=args.semantic_baseline_scope,
            mono_pass_threshold=args.mono_pass_threshold,
            mono_seed_gate=args.mono_seed_gate,
        )
        print(
            "  Semantic-baseline scope: "
            f"{args.semantic_baseline_scope} "
            f"(pair threshold={args.mono_pass_threshold:.2f}, seed gate={args.mono_seed_gate})"
        )
        print(f"  Filtered terminal records: {len(df_term)}")
        print(f"  Filtered trajectory records: {len(df_traj)}")
    pstar_present = [c for c in PSTAR_PRIORITY if c in df_term.columns]
    if not pstar_present and "d_T_pstar_z2t" in df_term.columns:
        pstar_present = ["d_T_pstar_z2t"]
    if pstar_present:
        src_names = ", ".join(TERM_LABEL[c] for c in pstar_present)
        print(f"  p* columns present: {src_names} — p*-dependent plots (12–19, 23–25, 29–34) can run.")
    else:
        print("  No d_T_pstar_* columns found — p*-dependent plots (12–19, 23–25, 29–34) will be skipped.")
        print("  Re-run measure_composability_gap.py with --pstar-source {sdipc,inverter,pez} (z2t/vlm are legacy optional).")

    # Resolve --pstar-sources into a frozenset of allowed terminal column names.
    # Default is a trusted allowlist (sdipc/inverter/pez), excluding legacy Z2T/VLM.
    _SRC_COLS = {
        "sdipc":    {"d_T_pstar_sdipc"},
        "inverter": {"d_T_pstar_inv", "d_T_pstar"},
        "pez":      {"d_T_pstar_pez"},
        "z2t":      {"d_T_pstar_z2t"},
        "vlm":      {"d_T_pstar_vlm"},
    }
    _TRUSTED_DEFAULT = frozenset({"d_T_pstar_sdipc", "d_T_pstar_inv", "d_T_pstar", "d_T_pstar_pez"})
    src_arg = [s.lower() for s in args.pstar_sources]
    if src_arg == ["auto"] or src_arg == ["all"]:
        # Exclude legacy Z2T/VLM from defaults; still available explicitly.
        pstar_filter = _TRUSTED_DEFAULT
    elif src_arg == ["none"]:
        pstar_filter = frozenset()             # suppress all p* from baseline plots
    else:
        cols: set = set()
        for s in src_arg:
            if s in _SRC_COLS:
                cols |= _SRC_COLS[s]
            else:
                print(f"  Warning: unknown --pstar-sources value '{s}'.  "
                      f"Valid: {list(_SRC_COLS)} | auto | none")
        pstar_filter = frozenset(cols)

    if pstar_filter is not None:
        if pstar_filter:
            names = ", ".join(TERM_LABEL[c] for c in sorted(pstar_filter) if c in TERM_LABEL)
            print(f"  p* filter (baseline 00–10 + grid 27–28): {names}")
        else:
            print("  p* filter (baseline 00–10 + grid 27–28): none — p* hidden there")

    # --- BLIP-VQA special route (separate from the numbered plots) ---
    if target == "blip_vqa":
        print("\n[blip_vqa] BLIP-VQA concept-presence grouped bar charts")
        try:
            df_blip = load_blip_vqa(data_dir)
        except FileNotFoundError as exc:
            print(f"  ERROR: {exc}")
            sys.exit(1)
        include_pstar = bool(pstar_filter)   # omit p* bar when pstar_filter is empty
        plot_blip_vqa_grouped_bar(df_blip, out_dir, pstar=False)
        if include_pstar:
            plot_blip_vqa_grouped_bar(df_blip, out_dir, pstar=True)
        else:
            print("  p* filter is empty — skipping F0b (blip_vqa_grouped_bar_pstar.png)")
        print(f"\nDone. Figures written to {out_dir}/")
        return

    if target == "joint_probes":
        print("\n[joint_probes] Pair-type-aware joint-correctness grouped bar charts")
        try:
            _, df_joint = load_joint_probes(data_dir)
        except FileNotFoundError as exc:
            print(f"  ERROR: {exc}")
            sys.exit(1)
        include_pstar = bool(pstar_filter)
        plot_joint_probe_grouped_bar(df_joint, out_dir, pstar=False)
        if include_pstar:
            plot_joint_probe_grouped_bar(df_joint, out_dir, pstar=True)
        else:
            print("  p* filter is empty — skipping joint_probe_grouped_bar_pstar.png")
        print(f"\nDone. Figures written to {out_dir}/")
        return

    if target == "reachability_semantic":
        print("\n[reachability_semantic] Semantics-qualified p* reachability figure")
        try:
            df_term = load_terminal(data_dir, monolithic_baseline=args.monolithic_baseline, and_anchor=args.and_anchor)
            df_traj = load_trajectory(data_dir, monolithic_baseline=args.monolithic_baseline)
        except SystemExit:
            raise
        plot_reachability_semantic_groupwise(
            df_term,
            df_traj,
            out_dir,
            data_dir=data_dir,
            semantic_baseline_scope=args.semantic_baseline_scope,
            mono_pass_threshold=args.mono_pass_threshold,
            mono_seed_gate=args.mono_seed_gate,
        )
        print(f"\nDone. Figures written to {out_dir}/")
        return

    if target == "semantic_robustness":
        print("\n[semantic_robustness] Full vs semantics-qualified robustness summary")
        df_term_full = load_terminal(
            data_dir,
            monolithic_baseline=args.monolithic_baseline,
            and_anchor=args.and_anchor,
        )
        df_traj_full = load_trajectory(data_dir, monolithic_baseline=args.monolithic_baseline)
        df_term_qualified = filter_semantic_baseline_scope(
            df_term_full,
            data_dir,
            scope="pair_qualified",
            mono_pass_threshold=args.mono_pass_threshold,
            mono_seed_gate=args.mono_seed_gate,
        )
        df_traj_qualified = filter_semantic_baseline_scope(
            df_traj_full,
            data_dir,
            scope="pair_qualified",
            mono_pass_threshold=args.mono_pass_threshold,
            mono_seed_gate=args.mono_seed_gate,
        )
        plot_semantic_filter_robustness(df_term_full, df_traj_full, df_term_qualified, df_traj_qualified, out_dir)
        print(f"\nDone. Figures written to {out_dir}/")
        return

    if target == "all":
        keys = sorted(PLOTS.keys())
    elif target in GROUPS:
        keys = GROUPS[target]
    else:
        key = target.zfill(2)
        if key not in PLOTS:
            valid_groups = list(GROUPS.keys())
            print(f"Unknown plot '{target}'.\n"
                  f"Valid groups: {valid_groups}\n"
                  f"Valid numbers: {list(PLOTS.keys())} or 'all'")
            sys.exit(1)
        keys = [key]

    size_preset = args.paper_size_preset
    if size_preset == "plot26_large_traj_small":
        plot26_scale = 1.16
        traj_grid_scale = 0.84
    else:
        plot26_scale = 1.0
        traj_grid_scale = 1.0
    if size_preset != "default":
        print(
            "  paper size preset: "
            f"{size_preset} (plot_26 x{plot26_scale:.2f}, "
            f"trajectory grid x{traj_grid_scale:.2f})"
        )

    plot04_xmax_quantile = float(args.plot04_xmax_quantile)
    plot04_bw_scale = float(args.plot04_bw_scale)
    plot04_scale = float(args.plot04_scale)

    if args.plot04_separate:
        if plot04_xmax_quantile == 1.0:
            plot04_xmax_quantile = 0.97
        if plot04_bw_scale == 1.0:
            plot04_bw_scale = 0.75
        if plot04_scale == 1.0:
            plot04_scale = 1.25
        print(
            "  plot04 separate preset: "
            f"xmax_quantile={plot04_xmax_quantile:.3f}, "
            f"bw_scale={plot04_bw_scale:.3f}, "
            f"plot04_scale={plot04_scale:.2f}"
        )

    if not (0.0 < plot04_xmax_quantile <= 1.0):
        print(
            "  Warning: --plot04-xmax-quantile must satisfy 0 < q <= 1. "
            "Falling back to 1.0."
        )
        plot04_xmax_quantile = 1.0
    if plot04_bw_scale <= 0:
        print(
            "  Warning: --plot04-bw-scale must be > 0. Falling back to 1.0."
        )
        plot04_bw_scale = 1.0
    if plot04_scale <= 0:
        print(
            "  Warning: --plot04-scale must be > 0. Falling back to 1.0."
        )
        plot04_scale = 1.0

    plot_kw = {
        "data_dir":          data_dir,
        "max_display_pairs": args.max_display_pairs,
        "pstar_filter":      pstar_filter,
        "monolithic_baseline": args.monolithic_baseline,
        "and_anchor":        args.and_anchor,
        "plot26_scale":      plot26_scale,
        "traj_grid_scale":   traj_grid_scale,
        "plot04_xmax_quantile": plot04_xmax_quantile,
        "plot04_bw_scale":   plot04_bw_scale,
        "plot04_scale":      plot04_scale,
    }

    for key in keys:
        label, fn = PLOTS[key]
        print(f"\n[{key}] {label}")
        render_pooled = args.taxonomy_view in {"pooled", "both"} or key not in GROUPWISE_PLOTS
        render_groupwise = key in GROUPWISE_PLOTS and args.taxonomy_view in {"groupwise", "both"}

        if render_pooled:
            fn(df_term, df_traj, out_dir, **plot_kw)
        if render_groupwise:
            GROUPWISE_PLOTS[key](df_term, df_traj, out_dir, **plot_kw)

    print(f"\nDone. Figures written to {out_dir}/")


if __name__ == "__main__":
    main()
