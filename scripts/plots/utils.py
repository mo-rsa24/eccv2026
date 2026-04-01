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
    # p* variants — ordered by increasing language expressiveness
    "d_T_pstar":     "#9467BD",   # purple  (backward-compat alias for inverter)
    "d_T_pstar_inv": "#9467BD",   # purple  — trained CLIP inverter
    "d_T_pstar_pez": "#17BECF",   # teal    — discrete token optimisation (PEZ / VGD)
    "d_T_pstar_z2t": "#E377C2",   # magenta — Zero2Text ridge regression
    "d_T_pstar_vlm": "#E8A838",   # amber   — VLM natural-language caption (LLaVA/BLIP-2)
}
TERM_LABEL = {
    "d_T_mono":      "Monolithic",
    "d_T_c1":        "Solo c₁",
    "d_T_c2":        "Solo c₂",
    "d_T_poe":       "PoE",
    "d_T_pstar":     "p* (inverter)",
    "d_T_pstar_inv": "p* (CLIP inverter)",
    "d_T_pstar_pez": "p* (token opt.)",
    "d_T_pstar_z2t": "p* (Zero2Text)",
    "d_T_pstar_vlm": "p* (VLM caption)",
}
TRAJ_COLOR = {
    "d_t_mono":      "#E15759",
    "d_t_c1":        "#4E79A7",
    "d_t_c2":        "#59A14F",
    "d_t_poe":       "#F28E2B",
    "d_t_pstar":     "#9467BD",
    "d_t_pstar_inv": "#9467BD",
    "d_t_pstar_pez": "#17BECF",
    "d_t_pstar_z2t": "#E377C2",
    "d_t_pstar_vlm": "#E8A838",
}
TRAJ_LABEL = {
    "d_t_mono":      "Monolithic",
    "d_t_c1":        "Solo c₁",
    "d_t_c2":        "Solo c₂",
    "d_t_poe":       "PoE",
    "d_t_pstar":     "p* (inverter)",
    "d_t_pstar_inv": "p* (CLIP inverter)",
    "d_t_pstar_pez": "p* (token opt.)",
    "d_t_pstar_z2t": "p* (Zero2Text)",
    "d_t_pstar_vlm": "p* (VLM caption)",
}

TERM_CONDITIONS = ["d_T_mono", "d_T_c1", "d_T_c2"]
TRAJ_CONDITIONS = ["d_t_mono", "d_t_c1", "d_t_c2"]

# Canonical priority for auto-detection: most expressive → least expressive.
# Primary path excludes legacy Z2T from defaults.
PSTAR_PRIORITY = [
    "d_T_pstar_vlm",
    "d_T_pstar_pez",
    "d_T_pstar_inv",
    "d_T_pstar",     # backward-compat alias — shown only if _inv is absent
]
# Legacy-only source kept for backward compatibility with older JSONs.
PSTAR_PRIORITY_LEGACY = ["d_T_pstar_z2t"]
# Parallel list for trajectory records
TRAJ_PSTAR_PRIORITY = [p.replace("d_T_", "d_t_") for p in PSTAR_PRIORITY]
TRAJ_PSTAR_PRIORITY_LEGACY = [p.replace("d_T_", "d_t_") for p in PSTAR_PRIORITY_LEGACY]

# Corresponding gap key in all_pairs_gap.json for each terminal pstar column
PSTAR_GAP_KEY = {
    "d_T_pstar_vlm": "gap_and_pstar_vlm",
    "d_T_pstar_pez": "gap_and_pstar_pez",
    "d_T_pstar_z2t": "gap_and_pstar_z2t",
    "d_T_pstar_inv": "gap_and_pstar_inv",
    "d_T_pstar":     "gap_and_pstar",
}

# Four visually distinct hues — one per concept pair
PAIR_PALETTE = ["#F28E2B", "#76B7B2", "#B07AA1", "#9C755F"]

# Time bins for stacked temporal bar charts
# 5 equal-width windows; sum of incremental heights = terminal MSE (since d_0 = 0)
STEP_BINS  = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
BIN_LABELS = ["0–10", "10–20", "20–30", "30–40", "40–50"]
BIN_ALPHAS = [0.92, 0.74, 0.55, 0.36, 0.18]   # opaque = early, transparent = late

# Axis-label math strings (shared across baseline/p* /distributional plots)
LABEL_D_T_MSE = (
    r"Per-element MSE  "
    r"$d_T=\frac{1}{N}\left\|z_T^{\mathrm{cond}}-z_T^{\mathrm{AND}}\right\|_2^2$"
)
LABEL_D_t_MSE = (
    r"Per-element MSE  "
    r"$d_t=\frac{1}{N}\left\|z_t^{\mathrm{cond}}-z_t^{\mathrm{AND}}\right\|_2^2$"
)
LABEL_CUM_DELTA_D_t = r"Cumulative $\Delta d_t$ from AND  (incremental per time bin)"
LABEL_ECDF = r"Cumulative probability  $P(d_T \leq x)$"
LABEL_JEFFREYS = r"$J(P,Q)=D_{\mathrm{KL}}(P\|Q)+D_{\mathrm{KL}}(Q\|P)$"
LABEL_JS2 = r"$\mathrm{JS}^2(P_{p^*}, P_{\mathrm{mono}})$  ($\downarrow$ better)"
LABEL_J_SOURCE_WITHIN = (
    r"$J(P_{\mathrm{source}}, P_{\mathrm{within\!-\!AND}})$  "
    r"($\downarrow$ closer to AND)"
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
    if not present:
        # Fallback for older runs that only contain legacy Z2T.
        present = [c for c in PSTAR_PRIORITY_LEGACY if c in df.columns]
    if "d_T_pstar_inv" in present and "d_T_pstar" in present:
        present.remove("d_T_pstar")
    return present


def get_present_poe(df: pd.DataFrame) -> list:
    """Return terminal PoE column if present."""
    return ["d_T_poe"] if "d_T_poe" in df.columns else []


def get_present_traj_pstar(df: pd.DataFrame) -> list:
    """Return trajectory pstar columns present in df, in canonical priority order.
    Suppresses the legacy 'd_t_pstar' alias when 'd_t_pstar_inv' is also present."""
    present = [c for c in TRAJ_PSTAR_PRIORITY if c in df.columns]
    if not present:
        # Fallback for older runs that only contain legacy Z2T.
        present = [c for c in TRAJ_PSTAR_PRIORITY_LEGACY if c in df.columns]
    if "d_t_pstar_inv" in present and "d_t_pstar" in present:
        present.remove("d_t_pstar")
    return present


def get_present_traj_poe(df: pd.DataFrame) -> list:
    """Return trajectory PoE column if present."""
    return ["d_t_poe"] if "d_t_poe" in df.columns else []


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


def kde_pmf(vals: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE (Scott bandwidth) normalised to a sum-to-one PMF."""
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
    return _apply_monolithic_baseline_terminal(df, monolithic_baseline)


def load_trajectory(data_dir: Path, monolithic_baseline: str = "auto") -> pd.DataFrame:
    p = _resolve_json(data_dir, "trajectory_distances.json")
    if not p.exists():
        sys.exit(f"Not found: {p}\nRun measure_composability_gap.py first.")
    df = pd.DataFrame(json.loads(p.read_text()))
    return _apply_monolithic_baseline_trajectory(df, monolithic_baseline)


def load_all_pairs_gap(data_dir: Path, monolithic_baseline: str = "auto"):
    p = _resolve_json(data_dir, "all_pairs_gap.json")
    if not p.exists():
        return None
    rows = json.loads(p.read_text())
    return _apply_monolithic_baseline_gap(rows, monolithic_baseline)


def load_within_and(data_dir: Path):
    p = _resolve_json(data_dir, "within_and_distances.json")
    if not p.exists():
        return None
    return json.loads(p.read_text())
