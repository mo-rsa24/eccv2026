"""
Multi-Model Taxonomy Grid Assembly
===================================
Assembles the 9-row × 4-column qualitative grids for Phase 1, one per
taxonomy group.  Each group contains 3 concept pairs; each pair contributes
3 rows (SD 1.4 → SDXL → SD 3.5); each row has 4 columns
(solo c1 | solo c2 | semantic | logical/PoE).

A thin horizontal separator is drawn between pairs to visually delimit
the three pair blocks.

Input directories
-----------------
SD 1.4   experiments/eccv2026/taxonomy_qualitative/          (already generated)
SDXL     experiments/eccv2026/taxonomy_qualitative_multimodel/sdxl/
SD 3.5   experiments/eccv2026/taxonomy_qualitative_multimodel/sd3/

Output
------
experiments/eccv2026/taxonomy_qualitative_multimodel/
    group1_multimodel_grid.png
    group2_multimodel_grid.png
    group3_multimodel_grid.png
    group4_multimodel_grid.png

Usage
-----
    python scripts/assemble_multimodel_taxonomy_grid.py
    python scripts/assemble_multimodel_taxonomy_grid.py --groups group4_collision
    python scripts/assemble_multimodel_taxonomy_grid.py --cell_size 384
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SD14_BASE  = PROJECT_ROOT / "experiments" / "eccv2026" / "taxonomy_qualitative"
MULTI_BASE = PROJECT_ROOT / "experiments" / "eccv2026" / "taxonomy_qualitative_multimodel"
SDXL_BASE  = MULTI_BASE / "sdxl"
SD3_BASE   = MULTI_BASE / "sd3"

# ---------------------------------------------------------------------------
# Taxonomy pairs (must match the generation scripts)
# ---------------------------------------------------------------------------
TAXONOMY_GROUPS = {
    "group1_cooccurrence": {
        "label": "Group 1: Co-occurring Concepts",
        "pairs": [
            ("a camel",             "a desert landscape"),
            ("a butterfly",         "a flower meadow"),
            ("a lion",              "a savanna at sunset"),
        ],
    },
    "group2_disentangled": {
        "label": "Group 2: Feature-Space Disentangled",
        "pairs": [
            ("a dog",               "oil painting style"),
            ("a lighthouse",        "watercolour style"),
            ("a bicycle",           "sketch style"),
        ],
    },
    "group3_ood": {
        "label": "Group 3: Feature Overlap (OOD)",
        "pairs": [
            ("a bathtub",           "a streetlamp"),
            ("a black grand piano", "a white vase"),
            ("a typewriter",        "a cactus"),
        ],
    },
    "group4_collision": {
        "label": "Group 4: Adversarial Collision",
        "pairs": [
            ("a cat",               "a dog"),
            ("a cat",               "an owl"),
            ("a cat",               "a bear"),
        ],
    },
}

CONDITIONS     = ["solo_a", "solo_b", "monolithic", "poe"]
MODEL_ORDER    = ["sd14", "sdxl", "sd3"]
MODEL_LABELS   = {"sd14": "SD 1.4", "sdxl": "SDXL", "sd3": "SD 3.5"}
COND_LABELS    = ["Solo $c_1$", "Solo $c_2$", "Semantic", "Logical (PoE)"]

# Layout constants (all in pixels)
SEPARATOR_PX   = 6    # thickness of the horizontal separator between pairs
LABEL_H        = 28   # height of row-label strip on the left
COL_LABEL_H    = 30   # height of column header strip at the top
FONT_SIZE      = 14
SEPARATOR_COL  = (180, 180, 180)   # light grey separator
BG_COL         = (240, 240, 240)   # label strip background
TEXT_COL       = (40, 40, 40)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace(",", "")
        .replace("'", "")
        .replace("/", "")
    )


def _pair_slug(a: str, b: str) -> str:
    return f"{_slugify(a)}__x__{_slugify(b)}"


def _get_image_path(model: str, group: str, a: str, b: str, cond: str) -> Path:
    slug = _pair_slug(a, b)
    if model == "sd14":
        return SD14_BASE / group / slug / f"{cond}.png"
    elif model == "sdxl":
        return SDXL_BASE / group / slug / f"{cond}.png"
    else:  # sd3
        return SD3_BASE / group / slug / f"{cond}.png"


def _load_cell(path: Path, cell_size: int) -> np.ndarray:
    """Load an image file and resize to cell_size × cell_size.

    Returns an (H, W, 3) uint8 array. If the file is missing, returns a
    placeholder grey cell with a red border and 'MISSING' text.
    """
    if path.exists():
        img = Image.open(path).convert("RGB").resize(
            (cell_size, cell_size), Image.LANCZOS,
        )
        return np.array(img)

    # Missing placeholder
    arr = np.full((cell_size, cell_size, 3), 220, dtype=np.uint8)
    arr[:4, :] = arr[-4:, :] = arr[:, :4] = arr[:, -4:] = [200, 60, 60]
    try:
        pil = Image.fromarray(arr)
        draw = ImageDraw.Draw(pil)
        draw.text((cell_size // 2 - 28, cell_size // 2 - 8), "MISSING",
                  fill=(180, 40, 40))
        arr = np.array(pil)
    except Exception:
        pass
    return arr


def _try_font(size: int):
    for name in ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Grid assembly
# ---------------------------------------------------------------------------

def assemble_group(group_name: str, cell_size: int, out_dir: Path) -> Path:
    group_info = TAXONOMY_GROUPS[group_name]
    pairs      = group_info["pairs"]
    n_cols     = len(CONDITIONS)           # 4
    n_models   = len(MODEL_ORDER)          # 3
    n_pairs    = len(pairs)                # 3
    n_rows     = n_pairs * n_models        # 9

    font = _try_font(FONT_SIZE)

    # Total canvas size
    total_w = LABEL_H + n_cols * cell_size
    total_h = COL_LABEL_H + n_rows * cell_size + (n_pairs - 1) * SEPARATOR_PX

    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)

    # Draw column headers
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)

    col_labels = ["Solo $c_1$", "Solo $c_2$", "Semantic", "Logical (PoE)"]
    for ci, lbl in enumerate(col_labels):
        x = LABEL_H + ci * cell_size + cell_size // 2
        draw.text((x, COL_LABEL_H // 2), lbl, fill=TEXT_COL, font=font, anchor="mm")

    canvas = np.array(pil_canvas)

    # Fill cells
    y_cursor = COL_LABEL_H
    for pi, (a, b) in enumerate(pairs):
        if pi > 0:
            # Horizontal separator between pairs
            canvas[y_cursor: y_cursor + SEPARATOR_PX, LABEL_H:] = SEPARATOR_COL
            y_cursor += SEPARATOR_PX

        for mi, model in enumerate(MODEL_ORDER):
            row_y = y_cursor + mi * cell_size

            # Row label strip
            canvas[row_y: row_y + cell_size, :LABEL_H] = BG_COL

            # Write model label vertically centred in the strip
            pil_canvas = Image.fromarray(canvas)
            draw = ImageDraw.Draw(pil_canvas)
            label_text = MODEL_LABELS[model]
            draw.text(
                (LABEL_H // 2, row_y + cell_size // 2),
                label_text, fill=TEXT_COL, font=font, anchor="mm",
            )
            canvas = np.array(pil_canvas)

            # Load and place each condition image
            for ci, cond in enumerate(CONDITIONS):
                img_path = _get_image_path(model, group_name, a, b, cond)
                cell = _load_cell(img_path, cell_size)
                x0 = LABEL_H + ci * cell_size
                canvas[row_y: row_y + cell_size, x0: x0 + cell_size] = cell

        y_cursor += n_models * cell_size

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{group_name}_multimodel_grid.png"
    Image.fromarray(canvas).save(out_path)
    print(f"  {group_name}  →  {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Assemble multi-model taxonomy grids (9 rows × 4 cols per group)."
    )
    parser.add_argument(
        "--groups", nargs="+",
        choices=sorted(TAXONOMY_GROUPS.keys()),
        default=list(TAXONOMY_GROUPS.keys()),
    )
    parser.add_argument(
        "--cell_size", type=int, default=512,
        help="Side length (px) to which each condition image is resized. "
             "Use 384 for a smaller but still high-quality grid. [512]",
    )
    parser.add_argument(
        "--out", type=str, default="",
        help="Output directory. Defaults to "
             "experiments/eccv2026/taxonomy_qualitative_multimodel/",
    )
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else MULTI_BASE

    print(f"\nAssembling multi-model taxonomy grids")
    print(f"  Cell size : {args.cell_size}px")
    print(f"  Groups    : {', '.join(args.groups)}")
    print(f"  Output    : {out_dir}\n")

    for group_name in args.groups:
        assemble_group(group_name, args.cell_size, out_dir)

    print(f"\nDone. Copy to proposal media directory:")
    print(f"  cp {out_dir}/*_multimodel_grid.png \\")
    print(f"     proposal/proposal_stage_3/chapters/research_method/media/")


if __name__ == "__main__":
    main()
