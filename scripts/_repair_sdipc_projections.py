"""One-shot repair: rewrite all real (non-symlinked) grid_assets.json in SDIPC dirs
with a correct joint MDS projection. Verifies x_T alignment for representative pairs."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sdxl_sdipc_utils import write_canonical_grid_assets  # noqa: E402

TARGET_PAIRS = {
    "a_butterfly__x__a_flower_meadow",
    "a_dog__x__oil_painting_style",
    "a_picnic_table__x__a_snowstorm",
    "a_typewriter__x__a_cactus",
    "fluffy__x__a_stone",
    "a_fox__x__a_wolf",
}

SDIPC_ROOTS = [
    "experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_frozen_sdipc",
    "experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_group6_edit_sdipc",
]

ok = errors = skipped = 0

for sdipc_dir in SDIPC_ROOTS:
    root = Path(sdipc_dir)
    if not root.exists():
        print(f"SKIP (not found): {sdipc_dir}")
        continue
    print(f"\n=== {root.name} ===")

    for pair_dir in sorted(root.glob("seed_*/*/*")):
        ga = pair_dir / "grid_assets.json"
        if not ga.exists():
            skipped += 1
            continue
        if ga.is_symlink():
            # Symlinked dirs are non-representative seeds with no SDIPC enrichment;
            # their base-flat trajectories live in SDXL_FINAL_DIR and won't have
            # pstar_sdipc. Skip — they don't affect paper figures.
            skipped += 1
            continue

        try:
            write_canonical_grid_assets(pair_dir=pair_dir, require_base_flats=True)
            ok += 1

            # Extra verification for the 6 representative pairs
            if pair_dir.name in TARGET_PAIRS:
                proj = json.loads(ga.read_text()).get("trajectory_projection", {}).get("projected", {})
                pa = proj.get("prompt_a")
                ps = proj.get("pstar_sdipc")
                if pa and ps:
                    aligned = pa[0] == ps[0]
                    tag = "x_T OK " if aligned else "x_T MISMATCH"
                else:
                    tag = f"missing prompt_a={bool(pa)} pstar_sdipc={bool(ps)}"
                seed = pair_dir.parent.parent.name
                print(f"  [{tag}] {seed}/{pair_dir.parent.name}/{pair_dir.name}")
        except Exception as e:
            errors += 1
            print(f"  ERROR {pair_dir.relative_to(root)}: {e}")

print(f"\nDone: {ok} reprojected, {errors} errors, {skipped} skipped (symlinks/missing)")
