"""Validate that an SDXL SD-IPC enrichment root is paper-ready."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdxl_sdipc_utils import BASE_CONDITIONS, SDIPC_CONDITION, canonicalize_grid_asset_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SDXL SD-IPC enrichment outputs against the paper-ready asset contract.",
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", default=None, help="Optional group filter.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Optional seed filter.")
    return parser.parse_args()


def _load_manifest(root: Path) -> dict:
    path = root / "sdxl_qualitative_run_manifest.json"
    if not path.exists():
        raise SystemExit(f"Missing manifest: {path}")
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    src = _load_manifest(source_dir)
    out = _load_manifest(output_dir)

    src_seeds = [int(s) for s in (src.get("seeds") or [])]
    seeds = [int(s) for s in (args.seeds if args.seeds is not None else src_seeds)]
    missing_seeds = [s for s in seeds if s not in set(src_seeds)]
    if missing_seeds:
        raise SystemExit(f"Requested seeds not in source manifest: {missing_seeds}")

    selected_pairs = list(src.get("selected_pairs") or [])
    if args.groups:
        wanted_groups = set(args.groups)
        selected_pairs = [row for row in selected_pairs if str(row.get("taxonomy_group_key")) in wanted_groups]

    failures: list[str] = []
    checked = 0

    for row in selected_pairs:
        group = str(row["taxonomy_group_key"])
        slug = str(row["qualitative_pair_slug"])
        for seed in seeds:
            pair_dir = output_dir / f"seed_{seed}" / group / slug
            asset_path = pair_dir / "grid_assets.json"
            if not asset_path.exists():
                failures.append(f"missing grid_assets.json: {asset_path}")
                continue
            try:
                payload = json.loads(asset_path.read_text())
            except Exception as exc:
                failures.append(f"bad json: {asset_path} ({exc})")
                continue

            required_files = [
                pair_dir / "summary.json",
                pair_dir / "trajectory_manifold.png",
                pair_dir / "decoded_images.png",
                pair_dir / "solo_a.png",
                pair_dir / "solo_b.png",
                pair_dir / "monolithic.png",
                pair_dir / "poe.png",
            ]
            for path in required_files:
                if not path.exists():
                    failures.append(f"missing file: {path}")

            try:
                canonical = canonicalize_grid_asset_payload(pair_dir=pair_dir, require_base_flats=True)
            except Exception as exc:
                failures.append(f"non-canonical pair assets: {pair_dir} ({exc})")
                continue

            decoded_paths = canonical.get("decoded_image_paths") or {}
            flat_paths = canonical.get("trajectory_flat_paths") or {}
            for cond in list(BASE_CONDITIONS) + [SDIPC_CONDITION]:
                rel = decoded_paths.get(cond)
                if not rel:
                    failures.append(f"missing decoded path for {cond}: {asset_path}")
                    continue
                if not (pair_dir / rel).exists():
                    failures.append(f"missing decoded file for {cond}: {pair_dir / rel}")

            for cond in list(BASE_CONDITIONS) + [SDIPC_CONDITION]:
                rel = flat_paths.get(cond)
                if not rel:
                    failures.append(f"missing flat path for {cond}: {asset_path}")
                    continue
                if not (pair_dir / rel).exists():
                    failures.append(f"missing flat file for {cond}: {pair_dir / rel}")

            projected = ((canonical.get("trajectory_projection") or {}).get("projected")) or {}
            for cond in list(BASE_CONDITIONS) + [SDIPC_CONDITION]:
                if cond not in projected:
                    failures.append(f"missing projected trajectory for {cond}: {asset_path}")
            checked += 1

    if failures:
        print(f"FAILED: {len(failures)} issues across {checked} records")
        for item in failures[:50]:
            print(f"  - {item}")
        if len(failures) > 50:
            print(f"  ... and {len(failures) - 50} more")
        raise SystemExit(1)

    print(f"OK: validated {checked} pair-seed records in {output_dir}")
    print(f"Output manifest: {output_dir / 'sdxl_qualitative_run_manifest.json'}")


if __name__ == "__main__":
    main()
