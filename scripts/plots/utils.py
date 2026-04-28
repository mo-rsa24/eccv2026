"""
Shared constants, helpers, and data-loading functions for the gap-analysis plot suite.
All plot modules (baseline, pstar, distributional) import from here.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from taxonomy_manifest import (
        GROUP_LABEL_BY_KEY,
        GROUP_ORDER,
        get_pair_taxonomy_from_row,
    )
except ImportError:
    from scripts.taxonomy_manifest import (
        GROUP_LABEL_BY_KEY,
        GROUP_ORDER,
        get_pair_taxonomy_from_row,
    )

try:
    from semantic_baseline_common import (
        build_audit_from_joint_probe_file,
        resolve_joint_probe_path,
        resolve_semantic_audit_path,
        select_qualified_pair_slugs,
    )
except ImportError:
    from scripts.semantic_baseline_common import (
        build_audit_from_joint_probe_file,
        resolve_joint_probe_path,
        resolve_semantic_audit_path,
        select_qualified_pair_slugs,
    )

try:
    from scipy import stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    stats = None
    print("Warning: scipy not found — plots 04, 05, 15, 19–22 require it (pip install scipy).")


# ---------------------------------------------------------------------------
# Colour / label constants
# ---------------------------------------------------------------------------

TERM_COLOR = {
    # Baselines
    "d_T_mono":      "#E15759",   # red
    "d_T_c1":        "#4E79A7",   # blue
    "d_T_c2":        "#59A14F",   # green
    "d_T_poe":       "#F28E2B",   # orange
    "d_T_co3":       "#B07AA1",   # mauve   — CO3 logical composition
    # p* variants — ordered by increasing language expressiveness
    "d_T_pstar":     "#9467BD",   # purple  (backward-compat alias for inverter)
    "d_T_pstar_sdipc":     "#E8A838", # amber   — SD-IPC closed-form projection from PoE
    "d_T_pstar_co3_sdipc": "#76B7B2", # teal    — SD-IPC closed-form projection from CO3
    "d_T_pstar_inv": "#9467BD",   # purple  — trained CLIP inverter
    "d_T_pstar_pez": "#17BECF",   # cyan    — discrete token optimisation (PEZ / VGD)
    "d_T_pstar_z2t": "#E377C2",   # magenta — Zero2Text ridge regression
    "d_T_pstar_vlm": "#C9A227",   # gold    — legacy VLM caption rerun
}
TERM_LABEL = {
    "d_T_mono":      "Monolithic",
    "d_T_c1":        "Solo c₁",
    "d_T_c2":        "Solo c₂",
    "d_T_poe":       "PoE",
    "d_T_co3":       "CO3",
    "d_T_pstar":     "PoE p* (inverter)",
    "d_T_pstar_sdipc":     "PoE p*",
    "d_T_pstar_co3_sdipc": "CO3 p*",
    "d_T_pstar_inv": "PoE p* (CLIP inverter)",
    "d_T_pstar_pez": "PoE p* (token opt.)",
    "d_T_pstar_z2t": "PoE p* (Zero2Text)",
    "d_T_pstar_vlm": "p* (VLM caption, legacy)",
}
TRAJ_COLOR = {
    "d_t_mono":      "#E15759",
    "d_t_c1":        "#4E79A7",
    "d_t_c2":        "#59A14F",
    "d_t_poe":       "#F28E2B",
    "d_t_co3":       "#B07AA1",
    "d_t_pstar":     "#9467BD",
    "d_t_pstar_sdipc":     "#E8A838",
    "d_t_pstar_co3_sdipc": "#76B7B2",
    "d_t_pstar_inv": "#9467BD",
    "d_t_pstar_pez": "#17BECF",
    "d_t_pstar_z2t": "#E377C2",
    "d_t_pstar_vlm": "#C9A227",
}
TRAJ_LABEL = {
    "d_t_mono":      "Monolithic",
    "d_t_c1":        "Solo c₁",
    "d_t_c2":        "Solo c₂",
    "d_t_poe":       "PoE",
    "d_t_co3":       "CO3",
    "d_t_pstar":     "PoE p* (inverter)",
    "d_t_pstar_sdipc":     "PoE p*",
    "d_t_pstar_co3_sdipc": "CO3 p*",
    "d_t_pstar_inv": "PoE p* (CLIP inverter)",
    "d_t_pstar_pez": "PoE p* (token opt.)",
    "d_t_pstar_z2t": "PoE p* (Zero2Text)",
    "d_t_pstar_vlm": "p* (VLM caption, legacy)",
}

TERM_CONDITIONS = ["d_T_mono", "d_T_c1", "d_T_c2"]
TRAJ_CONDITIONS = ["d_t_mono", "d_t_c1", "d_t_c2"]

# Canonical priority for auto-detection: most expressive → least expressive.
# Primary path excludes legacy Z2T from defaults.
PSTAR_PRIORITY = [
    "d_T_pstar_sdipc",
    "d_T_pstar_co3_sdipc",  # SD-IPC projection from CO3 image
    "d_T_pstar_pez",
    "d_T_pstar_inv",
    "d_T_pstar",     # backward-compat alias — shown only if _inv is absent
]
# Legacy-only sources kept for backward compatibility with older JSONs.
PSTAR_PRIORITY_LEGACY = ["d_T_pstar_vlm", "d_T_pstar_z2t"]
# Parallel list for trajectory records
TRAJ_PSTAR_PRIORITY = [p.replace("d_T_", "d_t_") for p in PSTAR_PRIORITY]
TRAJ_PSTAR_PRIORITY_LEGACY = [p.replace("d_T_", "d_t_") for p in PSTAR_PRIORITY_LEGACY]

# Corresponding gap key in all_pairs_gap.json for each terminal pstar column
PSTAR_GAP_KEY = {
    "d_T_pstar_sdipc": "gap_and_pstar_sdipc",
    "d_T_pstar_vlm": "gap_and_pstar_vlm",
    "d_T_pstar_pez": "gap_and_pstar_pez",
    "d_T_pstar_z2t": "gap_and_pstar_z2t",
    "d_T_pstar_inv": "gap_and_pstar_inv",
    "d_T_pstar":     "gap_and_pstar",
}

# Four visually distinct hues — one per concept pair
PAIR_PALETTE = ["#F28E2B", "#76B7B2", "#B07AA1", "#9C755F"]

# Time bins for stacked temporal bar charts
# 3 phase windows aligned to early/mid/late denoising decomposition:
#   Early (0–17): high-noise structural divergence
#   Mid   (17–34): refinement phase
#   Late  (34–50): low-noise fine-detail resolution
# Sum of incremental heights = terminal MSE (since d_0 = 0)
STEP_BINS  = [(0, 17), (17, 34), (34, 50)]
BIN_LABELS = ["Early (0–17)", "Mid (17–34)", "Late (34–50)"]
BIN_ALPHAS = [0.92, 0.55, 0.18]   # opaque = early, transparent = late

# Axis-label math strings (shared across baseline/p* /distributional plots)
LABEL_D_T_MSE = (
    r"Per-element MSE  "
    r"$d_T=\frac{1}{N}\left\|z_T^{\mathrm{cond}}-z_T^{\mathrm{anchor}}\right\|_2^2$"
)
LABEL_D_t_MSE = (
    r"Per-element MSE  "
    r"$d_t=\frac{1}{N}\left\|z_t^{\mathrm{cond}}-z_t^{\mathrm{anchor}}\right\|_2^2$"
)
LABEL_CUM_DELTA_D_t = r"Cumulative $\Delta d_t$ from logical anchor  (incremental per time bin)"
LABEL_ECDF = r"Cumulative probability  $P(d_T \leq x)$"
LABEL_JEFFREYS = r"$J(P,Q)=D_{\mathrm{KL}}(P\|Q)+D_{\mathrm{KL}}(Q\|P)$"
LABEL_JS2 = r"$\mathrm{JS}^2(P_{p^*}, P_{\mathrm{mono}})$  ($\downarrow$ better)"
LABEL_J_SOURCE_WITHIN = (
    r"$J(P_{\mathrm{source}}, P_{\mathrm{within\!-\!anchor}})$  "
    r"($\downarrow$ closer to the logical anchor)"
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def cap_pairs(pairs: list, max_n, rng_seed: int = 0) -> list:
    """Return at most max_n pairs, randomly sampled (fixed seed) when needed.
    Used by per-pair panel plots so large-regime figures stay readable.
    Pooled plots always pass the full pair list and ignore this helper."""
    if max_n is None or len(pairs) <= max_n:
        return pairs
    chosen = sorted(np.random.default_rng(rng_seed).choice(
        pairs, max_n, replace=False).tolist())
    return chosen


def get_present_pstar(df: pd.DataFrame) -> list:
    """Return pstar columns present in df, in canonical priority order.
    Suppresses the legacy 'd_T_pstar' alias when 'd_T_pstar_inv' is also present."""
    present = [c for c in PSTAR_PRIORITY if c in df.columns]
    present += [c for c in PSTAR_PRIORITY_LEGACY if c in df.columns]
    if "d_T_pstar_inv" in present and "d_T_pstar" in present:
        present.remove("d_T_pstar")
    return present


def get_present_poe(df: pd.DataFrame) -> list:
    """PoE is the anchor (distance = 0 by construction); exclude from conditions."""
    return []


def get_present_traj_pstar(df: pd.DataFrame) -> list:
    """Return trajectory pstar columns present in df, in canonical priority order.
    Suppresses the legacy 'd_t_pstar' alias when 'd_t_pstar_inv' is also present."""
    present = [c for c in TRAJ_PSTAR_PRIORITY if c in df.columns]
    present += [c for c in TRAJ_PSTAR_PRIORITY_LEGACY if c in df.columns]
    if "d_t_pstar_inv" in present and "d_t_pstar" in present:
        present.remove("d_t_pstar")
    return present


def get_present_traj_poe(df: pd.DataFrame) -> list:
    """PoE is the anchor (distance = 0 by construction); exclude from conditions."""
    return []


def get_present_co3(df: pd.DataFrame) -> list:
    """Return terminal CO3 column if present."""
    return ["d_T_co3"] if "d_T_co3" in df.columns else []


def get_present_traj_co3(df: pd.DataFrame) -> list:
    """Return trajectory CO3 column if present.
    CO3 is endpoint-only, so this is expected to be absent from trajectory data;
    the function is provided for symmetry and graceful no-op behaviour."""
    return ["d_t_co3"] if "d_t_co3" in df.columns else []


def apply_pstar_filter(pstar_cols: list, pstar_filter) -> list:
    """Apply an explicit p* source allowlist to a list of auto-detected column names.

    pstar_filter=None        → return pstar_cols unchanged  (auto-detect, default).
    pstar_filter=frozenset() → return []                    (suppress all p* sources).
    pstar_filter={...}       → keep only columns in the set.

    The filter stores terminal column names (d_T_*).  Trajectory column names
    (d_t_*) are matched automatically by prefix substitution so a single filter
    works for both terminal and trajectory calls.
    """
    if pstar_filter is None:
        return pstar_cols
    allowed = frozenset(pstar_filter) | frozenset(
        c.replace("d_T_", "d_t_") for c in pstar_filter
    )
    return [c for c in pstar_cols if c in allowed]


def pair_color_map(pairs):
    return {p: PAIR_PALETTE[i % len(PAIR_PALETTE)] for i, p in enumerate(sorted(pairs))}


def short_pair(pair: str) -> str:
    """'a cat + a dog'  →  'cat + dog'  (strips leading articles)."""
    def strip_article(s):
        for art in ("a ", "an ", "the "):
            if s.lower().startswith(art):
                return s[len(art):]
        return s
    return " + ".join(strip_article(p) for p in pair.split(" + "))


def enrich_taxonomy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    def _lookup(row: pd.Series):
        meta = get_pair_taxonomy_from_row(row.to_dict())
        row_is_repr = row.get("is_representative_pair", False)
        if pd.isna(row_is_repr):
            row_is_repr = False
        if meta is None:
            return pd.Series(
                {
                    "pair_slug": row.get("pair_slug") or row.get("slug"),
                    "taxonomy_group_key": row.get("taxonomy_group_key") or row.get("pair_group"),
                    "taxonomy_group_label": row.get("taxonomy_group_label"),
                    "is_representative_pair": bool(row_is_repr),
                }
            )
        return pd.Series(
            {
                "pair_slug": row.get("pair_slug") or row.get("slug") or meta["pair_slug"],
                "taxonomy_group_key": row.get("taxonomy_group_key") or meta["taxonomy_group_key"],
                "taxonomy_group_label": (
                    row.get("taxonomy_group_label") or meta["taxonomy_group_label"]
                ),
                "is_representative_pair": bool(row_is_repr or meta["is_representative_pair"]),
            }
        )

    extra = df.apply(_lookup, axis=1)
    for col in extra.columns:
        df[col] = extra[col]
    if "taxonomy_group_label" in df.columns:
        df["taxonomy_group_label"] = df["taxonomy_group_label"].fillna(
            df["taxonomy_group_key"].map(GROUP_LABEL_BY_KEY)
        )
    return df


def enrich_taxonomy_records(records: list[dict]) -> list[dict]:
    enriched = []
    for rec in records:
        rec = dict(rec)
        meta = get_pair_taxonomy_from_row(rec)
        if meta is not None:
            rec.setdefault("pair_slug", meta["pair_slug"])
            rec.setdefault("taxonomy_group_key", meta["taxonomy_group_key"])
            rec.setdefault("taxonomy_group_label", meta["taxonomy_group_label"])
            rec.setdefault("is_representative_pair", meta["is_representative_pair"])
        if rec.get("taxonomy_group_label") is None and rec.get("taxonomy_group_key") in GROUP_LABEL_BY_KEY:
            rec["taxonomy_group_label"] = GROUP_LABEL_BY_KEY[rec["taxonomy_group_key"]]
        enriched.append(rec)
    return enriched


def taxonomy_group_values(df: pd.DataFrame) -> list[str]:
    if "taxonomy_group_key" not in df.columns:
        return []
    present = set(df["taxonomy_group_key"].dropna().tolist())
    return [key for key in GROUP_ORDER if key in present]


def active_logical_anchor_label(df: pd.DataFrame | None, fallback: str = "logical anchor") -> str:
    if df is None or getattr(df, "empty", True):
        return fallback
    if "logical_anchor_label" in df.columns:
        vals = [str(v).strip() for v in df["logical_anchor_label"].dropna().unique().tolist() if str(v).strip()]
        if vals:
            return vals[0]
    if "logical_anchor" in df.columns:
        vals = [str(v).strip() for v in df["logical_anchor"].dropna().unique().tolist() if str(v).strip()]
        if vals:
            return vals[0]
    return fallback


def within_anchor_column(df: pd.DataFrame | None) -> str:
    if df is None or getattr(df, "empty", True):
        return "d_within_and"
    if "d_within_poe" in df.columns:
        return "d_within_poe"
    return "d_within_and"


def canonical_condition_name(metric_key: str) -> str:
    return (
        metric_key.replace("d_T_", "")
        .replace("d_t_", "")
        .replace("gap_and_", "")
    )


def kde_pmf(vals: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE (Scott bandwidth) normalised to a sum-to-one PMF.
    Returns a uniform PMF if fewer than 2 data points (KDE undefined)."""
    if len(vals) < 2:
        uniform = np.ones(len(x_grid), dtype=float)
        return uniform / uniform.sum()
    d = np.maximum(stats.gaussian_kde(vals, bw_method="scott")(x_grid), 1e-12)
    return d / d.sum()


def jeffreys_div(p: np.ndarray, q: np.ndarray) -> float:
    """Jeffrey's divergence J(P,Q) = KL(P||Q) + KL(Q||P).
    Both p and q must be PMFs on the same grid (SCIPY_OK required).
    scipy.stats.entropy handles 0*log(0/q)=0 internally; kde_pmf also
    floors at 1e-12 so true zeros do not arise in practice."""
    return float(stats.entropy(p, q) + stats.entropy(q, p))


def ecdf_xy(vals: np.ndarray):
    """Return (x, y) step-function coordinates for an empirical CDF.
    Prepends a zero entry so the curve starts cleanly at y=0."""
    x = np.sort(vals)
    y = np.arange(1, len(x) + 1) / len(x)
    x = np.concatenate([[x[0] * 0.95], x])
    y = np.concatenate([[0.0], y])
    return x, y


def traj_stats(df: pd.DataFrame, cond: str, steps):
    """Return (means, stds) arrays aligned to `steps` for a trajectory column."""
    agg = df.groupby("step")[cond].agg(["mean", "std"]).reindex(steps)
    return agg["mean"].values, agg["std"].values


def bin_increments(df_traj: pd.DataFrame, cond: str) -> list:
    """
    For each bin in STEP_BINS compute mean(d_at_t2) - mean(d_at_t1) across all
    rows in df_traj.  Since d_0 = 0 (shared noise seed), the sum of all
    increments equals the mean terminal distance d_T.
    """
    increments = []
    for (t1, t2) in STEP_BINS:
        rows1 = df_traj[df_traj["step"] == t1]
        rows2 = df_traj[df_traj["step"] == t2]
        d1 = rows1[cond].mean() if len(rows1) > 0 else 0.0
        d2 = rows2[cond].mean() if len(rows2) > 0 else 0.0
        increments.append(max(d2 - d1, 0.0))
    return increments


def hide_top_right(ax):
    ax.spines[["top", "right"]].set_visible(False)


_PLOT_PREFIX_RE = re.compile(r"(?m)^\s*Plot\s+\d+\s*[—-]\s*")


def _strip_plot_prefix(text: str) -> str:
    """Remove leading 'Plot XX — ' prefixes while preserving the rest."""
    return _PLOT_PREFIX_RE.sub("", text).lstrip()


def _strip_plot_prefixes_in_figure(fig):
    # Keep all titles/suptitles, but drop the numeric plot prefix for paper-ready figures.
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        suptitle.set_text(_strip_plot_prefix(suptitle.get_text()))

    for ax in fig.axes:
        ax.title.set_text(_strip_plot_prefix(ax.title.get_text()))


def save_fig(fig, path: Path):
    _strip_plot_prefixes_in_figure(fig)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resolve_json(data_dir: Path, filename: str) -> Path:
    """Return the path to `filename`, checking metrics/ subfolder first then root.
    New runs write to {data_dir}/metrics/; old runs write directly to {data_dir}/."""
    candidate = data_dir / "metrics" / filename
    if candidate.exists():
        return candidate
    return data_dir / filename


def _apply_monolithic_baseline_terminal(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Optionally remap d_T_mono to d_T_mono_{mode}."""
    if mode not in {"naive", "natural"}:
        return df
    src = f"d_T_mono_{mode}"
    if src in df.columns:
        df["d_T_mono"] = df[src]
    else:
        print(
            f"Warning: requested monolithic baseline '{mode}' but column '{src}' "
            "is missing; keeping existing d_T_mono."
        )
    return df


def _apply_monolithic_baseline_trajectory(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Optionally remap d_t_mono to d_t_mono_{mode}."""
    if mode not in {"naive", "natural"}:
        return df
    src = f"d_t_mono_{mode}"
    if src in df.columns:
        df["d_t_mono"] = df[src]
    else:
        print(
            f"Warning: requested monolithic baseline '{mode}' but column '{src}' "
            "is missing; keeping existing d_t_mono."
        )
    return df


def _apply_monolithic_baseline_gap(rows: list, mode: str) -> list:
    """Optionally remap gap_and_mono to gap_and_mono_{mode}."""
    if mode not in {"naive", "natural"}:
        return rows
    src = f"gap_and_mono_{mode}"
    if rows and src not in rows[0]:
        print(
            f"Warning: requested monolithic baseline '{mode}' but key '{src}' "
            "is missing; keeping existing gap_and_mono."
        )
        return rows
    for rec in rows:
        if src in rec:
            rec["gap_and_mono"] = rec[src]
    return rows


def _apply_and_anchor_terminal(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Optionally remap terminal distances to *_meananchor columns."""
    if mode != "mean":
        return df

    terminal_cols = sorted(
        c for c in df.columns
        if c.startswith("d_T_") and not c.endswith("_meananchor")
    )

    remapped = []
    for dst in terminal_cols:
        src = f"{dst}_meananchor"
        if src in df.columns:
            df[dst] = df[src]
            remapped.append(dst)

    required = ("d_T_mono", "d_T_c1", "d_T_c2")
    missing_required = [
        f"{c}_meananchor"
        for c in required
        if c in df.columns and f"{c}_meananchor" not in df.columns
    ]

    if remapped:
        print(
            "  AND anchor mode: mean — remapped terminal columns "
            f"to *_meananchor ({len(remapped)} columns)."
        )
    if missing_required:
        print(
            "Warning: requested mean AND anchor but missing "
            f"{missing_required}; using per-seed anchor for those columns."
        )
    return df


def load_terminal(
    data_dir: Path,
    monolithic_baseline: str = "auto",
    and_anchor: str = "seed",
) -> pd.DataFrame:
    p = _resolve_json(data_dir, "per_seed_distances.json")
    if not p.exists():
        sys.exit(f"Not found: {p}\nRun measure_composability_gap.py first.")
    df = pd.DataFrame(json.loads(p.read_text()))
    df = _apply_and_anchor_terminal(df, and_anchor)
    df = _apply_monolithic_baseline_terminal(df, monolithic_baseline)
    return enrich_taxonomy_dataframe(df)


def load_trajectory(data_dir: Path, monolithic_baseline: str = "auto") -> pd.DataFrame:
    p = _resolve_json(data_dir, "trajectory_distances.json")
    if not p.exists():
        sys.exit(f"Not found: {p}\nRun measure_composability_gap.py first.")
    df = pd.DataFrame(json.loads(p.read_text()))
    df = _apply_monolithic_baseline_trajectory(df, monolithic_baseline)
    return enrich_taxonomy_dataframe(df)


def load_all_pairs_gap(data_dir: Path, monolithic_baseline: str = "auto"):
    p = _resolve_json(data_dir, "all_pairs_gap.json")
    if not p.exists():
        return None
    rows = json.loads(p.read_text())
    rows = _apply_monolithic_baseline_gap(rows, monolithic_baseline)
    return enrich_taxonomy_records(rows)


def load_within_and(data_dir: Path):
    p = _resolve_json(data_dir, "within_and_distances.json")
    if not p.exists():
        return None
    return enrich_taxonomy_records(json.loads(p.read_text()))


def load_semantic_baseline_audit(data_dir: Path) -> dict | None:
    audit_path = resolve_semantic_audit_path(data_dir)
    if audit_path.exists():
        return json.loads(audit_path.read_text())

    joint_probe_path = resolve_joint_probe_path(data_dir)
    if joint_probe_path.exists():
        return build_audit_from_joint_probe_file(joint_probe_path)
    return None


def filter_semantic_baseline_scope(
    df: pd.DataFrame,
    data_dir: Path,
    scope: str = "full",
    mono_pass_threshold: float = 0.75,
    mono_seed_gate: str = "semantic",
) -> pd.DataFrame:
    if scope == "full" or df.empty:
        return df

    audit = load_semantic_baseline_audit(data_dir)
    if audit is None:
        print(
            "Warning: semantic-baseline audit is unavailable; "
            f"keeping full dataset for scope '{scope}'."
        )
        return df

    qualified_pairs, qualified_seed_pairs = select_qualified_pair_slugs(
        audit,
        scope=scope,
        mono_pass_threshold=mono_pass_threshold,
        mono_seed_gate=mono_seed_gate,
    )
    if scope == "pair_qualified":
        return df[df["pair_slug"].isin(qualified_pairs)].copy()
    if scope == "seed_qualified":
        qualified_pairs = set(qualified_pairs)
        qualified_seed_pairs = set(qualified_seed_pairs)
        mask = [
            (pair_slug in qualified_pairs) and ((pair_slug, seed) in qualified_seed_pairs)
            for pair_slug, seed in zip(df["pair_slug"], df["seed"])
        ]
        return df[pd.Series(mask, index=df.index)].copy()
    return df
