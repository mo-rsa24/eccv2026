#!/usr/bin/env python3
"""Render a paper-facing single-endpoint joint-probe explainer panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_joint_probe_pair import (  # noqa: E402
    build_probe_inspection_payload,
    render_probe_inspection_png,
    save_probe_inspection_artifacts,
    _resolve_pair_dir,
)


DEFAULT_DATA_DIR = PROJECT_ROOT / "experiments" / "eccv2026" / "sdxl_final" / "sdxl_six_group_seed42_steps50_cfg7p5_frozen"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "paper"
    / "neurips"
    / "Comparing Semantic and Logical Composition Using Latent Diffusion Models"
    / "figures"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one decoded endpoint plus structured joint-probe inspection.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--pair", required=True, help="Pair slug, e.g. a_typewriter__x__a_cactus")
    parser.add_argument("--group", default="", help="Optional group key for disambiguation.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--condition",
        choices=["monolithic", "poe", "prompt_a", "prompt_b", "pstar_sdipc"],
        default="monolithic",
        help="Decoded endpoint condition to inspect.",
    )
    parser.add_argument("--question-spec", default="", help="Optional JSON override bundle.")
    parser.add_argument("--out", default="", help="Output PNG path.")
    parser.add_argument("--output-dir", default="", help="Optional directory to also save JSON/bundle artifacts.")
    default_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    pair_dir = _resolve_pair_dir(data_dir, args.pair, args.group or None)
    payload = build_probe_inspection_payload(
        pair_dir=pair_dir,
        condition=args.condition,
        seed=args.seed,
        question_spec_path=Path(args.question_spec) if args.question_spec else None,
        device=args.device,
    )

    output_path = Path(args.out) if args.out else DEFAULT_OUTPUT_DIR / f"{args.condition}_probe_explainer.png"
    render_probe_inspection_png(payload, output_path)
    print(f"Saved -> {output_path}")

    if args.output_dir:
        save_probe_inspection_artifacts(payload, Path(args.output_dir))
        print(f"Saved artifacts -> {args.output_dir}")


if __name__ == "__main__":
    main()
