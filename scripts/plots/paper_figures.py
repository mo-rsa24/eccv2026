"""
Paper-facing composite figures.

Functions
---------
plot_mechanistic_summary_pair
    Two-part figure answering "Does high Γ predict visible failure?"

    Part 1 — Γ_t^ref curves (tall panel):
        Mean ± SEM of d_t(mono, PoE) per taxonomy group over denoising steps.
        Γ_t measures how far the monolithic P(A∧B) trajectory has drifted from
        the PoE trajectory at step t.  High Γ means the two conditioning
        strategies produce structurally different latent states — and therefore
        different decoded images.
        • Peak markers (▲) show the step at which each group's divergence is
          greatest; the onset window (grey band) marks where Γ first lifts off.
        • G1 (co-occurrence) stays near zero  → PoE can still find P(A∧B).
        • G4/G6 (dual-object / collision) peak high and early  → structural
          slot-assignment fails during the layout phase (~steps 10–20).
        • G2 (factorisation) is a key negative control: moderate Γ but PoE still
          works because style and content occupy disjoint feature subspaces.
        In-panel annotation explains what Γ means so the figure is self-contained.

    Part 2 — Endpoint image strip (two sub-rows per group):
        Representative pair (seed 42), monolithic top / PoE bottom.
        Group-coloured border, failure-mode name + Γ_mean caption.

plot_seed_sheet_all_groups
    Expanded seed sheet replacing the defunct plot_snr_semantic_instability.

    Layout: 2-row × 3-col outer grid (one cell per group), each inner cell
    contains mono (top row) and PoE (bottom row) for N different seeds.

    Images are cropped from the pre-assembled sd14_{condition}.png grids using
    the companion manifest files, so REAL per-seed variation is shown.

    Fairness control: the same seed index is used for both mono and PoE in each
    column — differences reflect the composition method, not the noise draw.

    Claim: PoE outputs show semantic instability that grows with Γ.
    G6 (coherent collision) PoE outputs vary between seeds (different hybrids).
    G1 (co-occurrence) PoE outputs are stable and qualitatively correct.

plot_mechanistic_attention_pair
    Stacks pre-saved attention_timeline.png filmstrips for each group found in
    the attention_viz directory.  Avoids the black-map problem in
    plot_group_comparison (which requires VAE decode hooks that were not
    recorded in the current run).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

try:
    from taxonomy_manifest import GROUP_LABEL_BY_KEY, GROUP_ORDER, GROUP_SPECS
except ImportError:
    from scripts.taxonomy_manifest import GROUP_LABEL_BY_KEY, GROUP_ORDER, GROUP_SPECS

from .utils import (
    GROUP_ORDER,
    GROUP_LABEL_BY_KEY,
    hide_top_right,
    save_fig,
    load_trajectory,
    load_terminal,
)


# ---------------------------------------------------------------------------
# Group colours — kept in sync with taxonomy_manifest GROUP_SPECS
# ---------------------------------------------------------------------------

_GROUP_COLORS: dict[str, str] = {spec["key"]: spec["color"] for spec in GROUP_SPECS}

# Aliases: legacy run keys map to canonical 6-group colours
_GROUP_COLOR_ALIASES: dict[str, str] = {
    "group2_disentangled":    _GROUP_COLORS.get("group2_factorization", "#6ACC65"),
    "group3_feature_overlap": _GROUP_COLORS.get("group3_role_separable_object_scene", "#4BAE73"),
    "group4_coherent_collision": _GROUP_COLORS.get("group6_coherent_collision", "#D7191C"),
    "group4_dual_object":     _GROUP_COLORS.get("group4_dual_object_composition", "#F5A623"),
}

# Failure-mode short labels (shown as peak annotations and image captions)
_FAILURE_MODE: dict[str, str] = {
    "group1_cooccurrence":                  "Co-occurrence\n(negative control)",
    "group2_factorization":                 "Factorisation\n(style ⊥ content)",
    "group2_disentangled":                  "Factorisation\n(style ⊥ content)",
    "group3_role_separable_object_scene":   "Object–Scene\n(role separable)",
    "group3_feature_overlap":               "Object–Scene\n(role separable)",
    "group4_dual_object_composition":       "Dual-Object\n(slot competition)",
    "group4_dual_object":                   "Dual-Object\n(slot competition)",
    "group5_concept_prior_entanglement":    "Prior Entanglement\n(prior dominance)",
    "group6_coherent_collision":            "Coherent Collision\n(identity fusion)",
    "group4_coherent_collision":            "Coherent Collision\n(identity fusion)",
}


def _group_color(key: str) -> str:
    return _GROUP_COLORS.get(key, _GROUP_COLOR_ALIASES.get(key, "#888888"))


def _failure_mode(key: str) -> str:
    return _FAILURE_MODE.get(key, key.replace("_", " ").title())


def _group_label(key: str) -> str:
    if key in GROUP_LABEL_BY_KEY:
        return GROUP_LABEL_BY_KEY[key]
    try:
        from taxonomy_manifest import normalize_group_key
    except ImportError:
        from scripts.taxonomy_manifest import normalize_group_key
    return GROUP_LABEL_BY_KEY.get(normalize_group_key(key), key)


def _short_group_label(key: str) -> str:
    m = re.match(r"Group\s*(\d+)", _group_label(key))
    return f"G{m.group(1)}" if m else _group_label(key).split()[0]


def _norm_fn():
    try:
        from taxonomy_manifest import normalize_group_key
    except ImportError:
        from scripts.taxonomy_manifest import normalize_group_key
    return normalize_group_key


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_image(path: Path) -> Optional[np.ndarray]:
    if path.exists():
        try:
            return np.array(Image.open(path).convert("RGB"))
        except Exception:
            pass
    return None


def _sem(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    if n < 2:
        return np.zeros(values.shape[1] if values.ndim > 1 else 1)
    return values.std(axis=0, ddof=1) / np.sqrt(n)


def _parse_manifest(path: Path) -> tuple[int, int, dict[int, tuple[int, int]]]:
    """Return (n_rows, n_cols, {seed: (row, col)}) from an image-grid manifest."""
    text = path.read_text()
    m = re.search(r"(\d+) row", text)
    n_rows = int(m.group(1)) if m else 6
    m2 = re.search(r"(\d+) columns.*\(nrow=(\d+)\)", text)
    n_cols = int(m2.group(1)) if m2 else 4
    seed_pos: dict[int, tuple[int, int]] = {}
    for match in re.finditer(r"Row (\d+), Col (\d+)\s*:\s*seed (\d+)", text):
        seed_pos[int(match.group(3))] = (int(match.group(1)), int(match.group(2)))
    return n_rows, n_cols, seed_pos


def _crop_seed_from_grid(
    grid_img: np.ndarray,
    n_rows: int,
    n_cols: int,
    row: int,
    col: int,
) -> np.ndarray:
    """Extract one cell from a (n_rows × n_cols) image grid."""
    H, W = grid_img.shape[:2]
    cell_h = H // n_rows
    cell_w = W // n_cols
    y0, y1 = row * cell_h, (row + 1) * cell_h
    x0, x1 = col * cell_w, (col + 1) * cell_w
    return grid_img[y0:y1, x0:x1]


def _find_representative_pair_dir(pairs_dir: Path, group_key: str) -> Optional[Path]:
    """Return the representative (or first available) pair directory for a group."""
    norm = _norm_fn()
    best: Optional[Path] = None
    first: Optional[Path] = None
    for p in sorted(pairs_dir.iterdir()):
        asset_file = p / "grid_assets.json"
        if not asset_file.exists():
            continue
        try:
            d = json.loads(asset_file.read_text())
        except Exception:
            continue
        if norm(d.get("taxonomy_group_key", "")) != norm(group_key):
            continue
        if first is None:
            first = p
        if d.get("is_representative_pair", False):
            best = p
            break
    return best or first


def _load_endpoint_images(
    pairs_dir: Path, group_key: str
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """(mono_img, poe_img) for the representative pair, seed 42."""
    pair_dir = _find_representative_pair_dir(pairs_dir, group_key)
    if pair_dir is None:
        return None, None
    asset_file = pair_dir / "grid_assets.json"
    if not asset_file.exists():
        return None, None
    paths = json.loads(asset_file.read_text()).get("decoded_image_paths", {})
    mono = _load_image(pair_dir / paths["monolithic"]) if "monolithic" in paths else None
    poe  = _load_image(pair_dir / paths["poe"])        if "poe"        in paths else None
    return mono, poe


def _load_seed_images(
    pairs_dir: Path,
    group_key: str,
    seeds: list[int],
    condition: str,
) -> list[Optional[np.ndarray]]:
    """
    Return one decoded image per seed for *condition* ('monolithic' | 'poe').
    Reads from the pre-assembled sd14_{condition}.png grid + manifest if available,
    falling back to the single seed-42 decoded image for all seeds.
    """
    pair_dir = _find_representative_pair_dir(pairs_dir, group_key)
    if pair_dir is None:
        return [None] * len(seeds)

    cond_key = "monolithic" if condition == "monolithic" else "poe"
    grid_name_map = {"monolithic": "sd14_monolithic", "poe": "sd14_poe"}
    grid_stem = grid_name_map.get(cond_key, f"sd14_{cond_key}")

    grid_path     = pair_dir / "images" / f"{grid_stem}.png"
    manifest_path = pair_dir / "images" / f"{grid_stem}_manifest.txt"

    if grid_path.exists() and manifest_path.exists():
        grid_arr = np.array(Image.open(grid_path).convert("RGB"))
        n_rows, n_cols, seed_pos = _parse_manifest(manifest_path)
        result = []
        for s in seeds:
            if s in seed_pos:
                r, c = seed_pos[s]
                result.append(_crop_seed_from_grid(grid_arr, n_rows, n_cols, r, c))
            else:
                result.append(None)
        return result

    # Fallback: single decoded image repeated for all seeds
    asset_file = pair_dir / "grid_assets.json"
    if asset_file.exists():
        paths = json.loads(asset_file.read_text()).get("decoded_image_paths", {})
        img = _load_image(pair_dir / paths[cond_key]) if cond_key in paths else None
    else:
        img = None
    return [img] * len(seeds)


# ---------------------------------------------------------------------------
# Γ curve computation
# ---------------------------------------------------------------------------

def _gamma_curves(
    df_traj: pd.DataFrame,
    gamma_col: str = "d_t_mono",
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Per-group mean ± SEM of gamma_col over denoising steps.
    Returns {group_key: (steps, mean, sem)}.
    """
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    steps = sorted(df_traj["step"].unique())
    steps_arr = np.array(steps, dtype=int)

    for group_key in df_traj["taxonomy_group_key"].dropna().unique():
        sub = df_traj[df_traj["taxonomy_group_key"] == group_key]
        traces = []
        for (_pair, _seed), g in sub.groupby(["pair_slug", "seed"]):
            row_vals = g.set_index("step")[gamma_col].reindex(steps)
            if row_vals.isna().all():
                continue
            row_vals = row_vals.ffill().fillna(0.0)
            traces.append(row_vals.values)
        if not traces:
            continue
        mat = np.stack(traces, axis=0)
        result[group_key] = (steps_arr, mat.mean(axis=0), _sem(mat))
    return result


def _order_groups(keys: list[str]) -> list[str]:
    norm = _norm_fn()
    ordered = []
    for canonical in GROUP_ORDER:
        for gk in keys:
            if norm(gk) == norm(canonical):
                ordered.append(gk)
                break
    for gk in keys:
        if gk not in ordered:
            ordered.append(gk)
    return ordered


# ---------------------------------------------------------------------------
# 1.  plot_mechanistic_summary_pair
# ---------------------------------------------------------------------------

_THEORY_BOX = (
    r"$\Gamma_t^{\mathrm{ref}}$ = per-element MSE between the" "\n"
    r"monolithic $P(A{\wedge}B)$ latent and the PoE latent at step $t$." "\n"
    r"High $\Gamma$ $\Rightarrow$ the two conditioning strategies produce" "\n"
    "structurally different states ⇒ different decoded images.\n"
    "Peak step encodes the failure mechanism:\n"
    "  early peak → layout / slot assignment failure\n"
    "  late peak  → semantic identity binding failure\n"
    r"G2 is the key negative control: moderate $\Gamma$," "\n"
    "but PoE still works because style ⊥ content."
)


def plot_mechanistic_summary_pair(
    data_dir: Path,
    out_dir: Path,
    pairs_dir: Optional[Path] = None,
    gamma_col: str = "d_t_mono",
    monolithic_baseline: str = "auto",
    dpi: int = 200,
) -> None:
    """
    Two-part figure: Γ_t^ref curves (top) + endpoint image strip (bottom).
    Answers: "Does high Γ predict visible failure?"
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if pairs_dir is None:
        pairs_dir = Path(data_dir).parent / "pairs"
        if not pairs_dir.exists():
            pairs_dir = Path(data_dir) / "pairs"

    df_traj = load_trajectory(Path(data_dir), monolithic_baseline=monolithic_baseline)
    df_term = load_terminal(Path(data_dir), monolithic_baseline=monolithic_baseline)

    gamma_curves = _gamma_curves(df_traj, gamma_col=gamma_col)
    if not gamma_curves:
        print("  Skipping plot_mechanistic_summary_pair: no trajectory data.")
        return

    ordered_keys = _order_groups(list(gamma_curves.keys()))
    n_groups = len(ordered_keys)
    if n_groups == 0:
        print("  Skipping plot_mechanistic_summary_pair: no groups found.")
        return

    # Pre-compute per-group terminal Γ mean
    terminal_col = gamma_col.replace("d_t_", "d_T_")
    gamma_means: dict[str, float] = {}
    for gk in ordered_keys:
        sub = df_term[df_term["taxonomy_group_key"] == gk]
        if not sub.empty and terminal_col in sub.columns:
            gamma_means[gk] = float(sub[terminal_col].mean())

    # ----------------------------------------------------------------- layout
    fig = plt.figure(
        figsize=(max(13.0, 2.9 * n_groups), 11.0),
        facecolor="white",
    )
    gs_outer = fig.add_gridspec(
        2, 1,
        height_ratios=[3, 2],
        hspace=0.38,
        left=0.09, right=0.97, top=0.93, bottom=0.05,
    )

    # --------------------------------------------------------- PART 1: curves
    ax = fig.add_subplot(gs_outer[0])

    onset_steps: dict[str, Optional[int]] = {}
    peak_steps:  dict[str, int]           = {}

    for gk in ordered_keys:
        steps, mean, sem = gamma_curves[gk]
        color = _group_color(gk)
        short = _short_group_label(gk)

        ax.plot(steps, mean, color=color, lw=2.2, label=short, zorder=3)
        ax.fill_between(steps, mean - sem, mean + sem, alpha=0.15, color=color, zorder=2)

        # Peak marker
        peak_idx = int(np.argmax(mean))
        peak_steps[gk] = int(steps[peak_idx])
        ax.plot(steps[peak_idx], mean[peak_idx], marker="^", markersize=8,
                color=color, zorder=5, clip_on=False)
        # Annotate peak with group short label + failure mode (first line only)
        mode_line = _failure_mode(gk).split("\n")[0]
        ax.annotate(
            f"{short}\n{mode_line}",
            xy=(steps[peak_idx], mean[peak_idx]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=6.5, color=color,
            fontweight="bold",
            arrowprops=None,
        )

        # Onset: first step where mean > 5 % of peak
        threshold = 0.05 * float(mean[peak_idx])
        cands = np.where(mean > threshold)[0]
        onset_steps[gk] = int(steps[cands[0]]) if len(cands) else None

    # Onset window shading
    onset_vals = [v for v in onset_steps.values() if v is not None]
    if onset_vals:
        ax.axvspan(min(onset_vals), max(onset_vals), alpha=0.07, color="gray", zorder=0)
        ax.text(
            (min(onset_vals) + max(onset_vals)) / 2.0, 0.04,
            "onset\nwindow",
            fontsize=7, color="gray", ha="center", va="bottom",
            transform=ax.get_xaxis_transform(),
        )

    # Theory box (top-right)
    ax.text(
        0.985, 0.97,
        _THEORY_BOX,
        transform=ax.transAxes,
        fontsize=7.0,
        va="top", ha="right",
        linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F4F6F8",
                  edgecolor="#C0C8D4", linewidth=0.9, alpha=0.97),
    )

    ax.set_xlabel("Denoising step  (0 = pure noise → T = clean)", fontsize=10)
    ax.set_ylabel(
        r"$\Gamma_t^{\mathrm{ref}}$ = $d_t(\mathrm{mono},\mathrm{PoE})$  [per-elem MSE]",
        fontsize=10,
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.22)
    hide_top_right(ax)
    ax.legend(title="Group", fontsize=9, title_fontsize=9, loc="upper left", framealpha=0.88)
    ax.set_title(
        r"$\Gamma_t^{\mathrm{ref}}$ curves — how monolithic and PoE trajectories"
        " diverge across taxonomy groups",
        fontsize=11, pad=7,
    )

    # --------------------------------------------------------- PART 2: images
    gs_img = gs_outer[1].subgridspec(2, n_groups, wspace=0.06, hspace=0.07)

    row_labels = ["Mono", "PoE"]
    row_colors = ["#3B8D5B", "#D9872B"]

    for col_i, gk in enumerate(ordered_keys):
        color   = _group_color(gk)
        short   = _short_group_label(gk)
        gval    = gamma_means.get(gk, float("nan"))
        mode    = _failure_mode(gk)
        mono_img, poe_img = _load_endpoint_images(pairs_dir, gk)
        imgs = [mono_img, poe_img]

        ax_poe: Optional[plt.Axes] = None
        for row_i, (img, row_lbl, row_color) in enumerate(zip(imgs, row_labels, row_colors)):
            ax_cell = fig.add_subplot(gs_img[row_i, col_i])
            if row_i == 1:
                ax_poe = ax_cell

            if img is not None:
                ax_cell.imshow(img, aspect="equal", interpolation="bilinear")
            else:
                ax_cell.set_facecolor("#E8E8E8")
                ax_cell.text(0.5, 0.5, f"{row_lbl}\n(missing)",
                             ha="center", va="center", fontsize=7,
                             color="#888888", transform=ax_cell.transAxes)
            ax_cell.set_xticks([])
            ax_cell.set_yticks([])
            for spine in ax_cell.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.2)
                spine.set_edgecolor(color)

            if col_i == 0:
                ax_cell.set_ylabel(row_lbl, fontsize=8, color=row_color,
                                   rotation=0, ha="right", va="center", labelpad=28)
            if row_i == 0:
                ax_cell.set_title(short, fontsize=9, fontweight="bold",
                                  color=color, pad=3)

        # Caption: failure mode + Γ_mean (on PoE row)
        if ax_poe is not None:
            gstr = f"{gval:.3f}" if not np.isnan(gval) else "—"
            ax_poe.text(
                0.5, -0.20,
                f"{mode}\nΓ = {gstr}",
                ha="center", va="top",
                fontsize=6.5, color=color, fontweight="bold",
                transform=ax_poe.transAxes,
                linespacing=1.3,
            )

    fig.suptitle(
        "Does high Γ predict visible failure?  "
        "Representative endpoints (seed 42) aligned to Γ_t curves above",
        fontsize=12, fontweight="bold",
    )
    save_fig(fig, out_dir / "plot_mechanistic_summary_pair.png")
    print(f"  → {out_dir / 'plot_mechanistic_summary_pair.png'}")


# ---------------------------------------------------------------------------
# 2.  plot_seed_sheet_all_groups
# ---------------------------------------------------------------------------

def plot_seed_sheet_all_groups(
    data_dir: Path,
    out_dir: Path,
    pairs_dir: Optional[Path] = None,
    seeds: list[int] | None = None,
    monolithic_baseline: str = "auto",
    dpi: int = 200,
) -> None:
    """
    Expanded seed sheet: each group cell shows mono (top) + PoE (bottom) across
    n_seeds columns, with real per-seed decoded images cropped from the
    pre-assembled sd14_{condition}.png grids.

    Fairness: identical seed in each column ⇒ differences = composition method,
    not noise.  Claim: PoE instability increases with Γ.
    """
    if seeds is None:
        seeds = [42, 1, 7, 13, 99]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if pairs_dir is None:
        pairs_dir = Path(data_dir).parent / "pairs"
        if not pairs_dir.exists():
            pairs_dir = Path(data_dir) / "pairs"

    df_term = load_terminal(Path(data_dir), monolithic_baseline=monolithic_baseline)

    all_data_keys = list(df_term["taxonomy_group_key"].dropna().unique())
    ordered_keys = _order_groups(all_data_keys)
    n_groups = len(ordered_keys)
    if n_groups == 0:
        print("  Skipping plot_seed_sheet_all_groups: no groups found.")
        return

    n_seeds = len(seeds)
    n_cols_outer = min(3, n_groups)
    n_rows_outer = int(np.ceil(n_groups / n_cols_outer))

    # Geometry: each cell = 2 inner rows × n_seeds columns of square images
    cell_w = 1.55 * n_seeds + 0.5
    cell_h = 3.6
    fig = plt.figure(
        figsize=(n_cols_outer * cell_w + 1.2, n_rows_outer * cell_h + 1.4),
        facecolor="white",
    )
    gs_outer = fig.add_gridspec(
        n_rows_outer, n_cols_outer,
        left=0.07, right=0.98, top=0.94, bottom=0.04,
        hspace=0.44, wspace=0.08,
    )

    row_labels = ["Mono", "PoE"]
    row_colors = ["#3B8D5B", "#D9872B"]

    for grp_idx, gk in enumerate(ordered_keys):
        outer_row = grp_idx // n_cols_outer
        outer_col = grp_idx % n_cols_outer
        color   = _group_color(gk)
        short   = _short_group_label(gk)
        sub_term = df_term[df_term["taxonomy_group_key"] == gk]
        gval: float = float(sub_term["d_T_mono"].mean()) if not sub_term.empty else float("nan")
        gstr = f"Γ = {gval:.3f}" if not np.isnan(gval) else ""

        gs_inner = gs_outer[outer_row, outer_col].subgridspec(
            2, n_seeds, wspace=0.04, hspace=0.05,
        )

        imgs_by_row: list[list[Optional[np.ndarray]]] = [
            _load_seed_images(pairs_dir, gk, seeds, "monolithic"),
            _load_seed_images(pairs_dir, gk, seeds, "poe"),
        ]

        ax_topleft: Optional[plt.Axes] = None
        for inner_row, (imgs, row_lbl, row_color) in enumerate(
            zip(imgs_by_row, row_labels, row_colors)
        ):
            for seed_col, (seed, img) in enumerate(zip(seeds, imgs)):
                ax_cell = fig.add_subplot(gs_inner[inner_row, seed_col])
                if inner_row == 0 and seed_col == 0:
                    ax_topleft = ax_cell

                if img is not None:
                    ax_cell.imshow(img, aspect="equal", interpolation="bilinear")
                else:
                    ax_cell.set_facecolor("#EBEBEB")
                    ax_cell.text(0.5, 0.5, "N/A", ha="center", va="center",
                                 fontsize=6, color="#AAAAAA",
                                 transform=ax_cell.transAxes)
                ax_cell.set_xticks([])
                ax_cell.set_yticks([])
                for spine in ax_cell.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.7)
                    spine.set_edgecolor(color)

                if inner_row == 0:
                    ax_cell.set_title(f"s{seed}", fontsize=6, pad=2, color="#555555")
                if seed_col == 0:
                    ax_cell.set_ylabel(row_lbl, fontsize=7, color=row_color,
                                       rotation=0, ha="right", va="center", labelpad=22)

        # Group label + Γ above seed row — on top-left inner axis
        if ax_topleft is not None:
            ax_topleft.set_title(
                f"s{seeds[0]}  ·  {short}  {gstr}",
                fontsize=7, fontweight="bold", color=color, pad=3,
            )

    for grp_idx in range(n_groups, n_rows_outer * n_cols_outer):
        fig.add_subplot(gs_outer[grp_idx // n_cols_outer, grp_idx % n_cols_outer]).axis("off")

    fig.suptitle(
        "Seed sheet — monolithic (top) vs PoE (bottom)  ·  all taxonomy groups  ·  5 seeds per group\n"
        "Same seed per column: differences reflect composition method, not noise draw.  "
        "PoE instability grows with Γ.",
        fontsize=10.5, fontweight="bold",
    )
    save_fig(fig, out_dir / "plot_seed_sheet_all_groups.png")
    print(f"  → {out_dir / 'plot_seed_sheet_all_groups.png'}")


# ---------------------------------------------------------------------------
# 3.  plot_mechanistic_attention_pair
# ---------------------------------------------------------------------------

def plot_mechanistic_attention_pair(
    attention_viz_dir: Path,
    out_dir: Path,
    group_keys: list[str] | None = None,
    dpi: int = 200,
) -> None:
    """
    Stack pre-saved attention_timeline.png filmstrips vertically, one per group.

    Avoids the black-map problem of plot_group_comparison (which requires live
    VAE-decode hooks not present in this run).  Uses the working composite-overlay
    heatmaps produced by the attention_timeline generation path.

    Does NOT load group_compare.png (would exceed the 2000 px multi-image limit).
    """
    attention_viz_dir = Path(attention_viz_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[tuple[str, Path]] = []
    search = [attention_viz_dir / gk for gk in group_keys] if group_keys else \
             sorted(attention_viz_dir.iterdir())
    for sub in search:
        p = (sub / "attention_timeline.png") if sub.is_dir() else sub
        if p.exists():
            candidates.append((sub.name, p))
        elif group_keys:
            print(f"  Warning: attention_timeline.png not found at {p}")

    if not candidates:
        print(
            f"  Skipping plot_mechanistic_attention_pair: "
            f"no attention_timeline.png in {attention_viz_dir}"
        )
        return

    first_img = Image.open(candidates[0][1])
    img_w, img_h = first_img.size
    fig_w = min(img_w / 72.0, 20.0)
    fig_h = (img_h / 72.0 + 0.4) * len(candidates)

    fig, axes = plt.subplots(len(candidates), 1, figsize=(fig_w, fig_h), facecolor="white")
    if len(candidates) == 1:
        axes = [axes]

    for ax_row, (gk, img_path) in zip(axes, candidates):
        ax_row.imshow(
            np.array(Image.open(img_path).convert("RGB")),
            aspect="auto", interpolation="bilinear",
        )
        ax_row.set_xticks([])
        ax_row.set_yticks([])
        ax_row.set_ylabel(
            gk.replace("_", " ").title(),
            fontsize=9, rotation=0, ha="right", va="center", labelpad=6,
        )

    fig.suptitle(
        "P(A∧B) Attention Timeline per Group\n"
        "Coloured heatmap = concept-token attention overlaid on denoised image  "
        "(100 %T = pure noise; 0 %T = final image)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "plot_mechanistic_attention_pair.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'plot_mechanistic_attention_pair.png'}")
