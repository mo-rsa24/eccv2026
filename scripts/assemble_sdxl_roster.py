#!/usr/bin/env python3
"""Legacy wrapper around the canonical manual SDXL freeze script."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from freeze_sdxl_taxonomy_roster import build_roster_manifest
    from sdxl_paper_audit_common import DEFAULT_AUDIT_MANIFEST_PATH
except ImportError:
    from scripts.freeze_sdxl_taxonomy_roster import build_roster_manifest
    from scripts.sdxl_paper_audit_common import DEFAULT_AUDIT_MANIFEST_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Legacy wrapper. Use freeze_sdxl_taxonomy_roster.py for the canonical six-group SDXL flow."
    )
    parser.add_argument("--orig-run-dir", type=Path, default=None, help="Unused legacy argument.")
    parser.add_argument("--replacements-run-dir", type=Path, default=None, help="Unused legacy argument.")
    parser.add_argument("--audit-manifest", type=Path, default=DEFAULT_AUDIT_MANIFEST_PATH, help="Manual audit manifest JSON path.")
    parser.add_argument("--output", type=Path, default=Path("/tmp/sdxl_paper_roster.json"), help="Output roster path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, _, _, errors = build_roster_manifest(args.audit_manifest)
    if errors:
        formatted = "\n".join(f"  - {msg}" for msg in errors)
        raise SystemExit(
            "assemble_sdxl_roster.py is deprecated and now delegates to the manual-only "
            "freeze flow.\n"
            f"{formatted}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))
    print("Saved manual-review roster via legacy wrapper ->", args.output)


if __name__ == "__main__":
    main()
