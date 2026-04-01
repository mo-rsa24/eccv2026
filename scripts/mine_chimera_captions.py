"""
Mine VLM manifest files for morphological chimera captions.

Recursively scans *_manifest.txt files under a search directory, applies
tiered regex patterns to each VLM caption, and reports all fusion / chimera
matches ranked by category strength.

Categories
----------
  STRONG   — morphological chimera: "a bird with a cat's face",
              "hybrid", "chimera", "cross between"
  MEDIUM   — anatomical anomaly: wrong body-part attributed to a species,
              unusual cross-attribute description
  WEAK     — coexistence / juxtaposition: two concepts present but separate

Usage
-----
# Scan all manifests in the default experiment directory:
python scripts/mine_chimera_captions.py

# Restrict to a specific run directory:
python scripts/mine_chimera_captions.py \\
    --search-dir experiments/inversion/gap_analysis/large_20260302_105630

# Only match manifests whose condition contains "vlm":
python scripts/mine_chimera_captions.py --condition vlm

# Show all tiers including weak coexistence:
python scripts/mine_chimera_captions.py --min-tier weak

# Save results to JSON:
python scripts/mine_chimera_captions.py --out chimeras.json

# Show a specific concept pair:
python scripts/mine_chimera_captions.py --pair "a_bird_a_cat"
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Fusion-pattern tiers
# ---------------------------------------------------------------------------

_BODY_PARTS = (
    r"face|head|body|eye|eyes|fur|feathers?|legs?|tail|beak|paws?|wings?|"
    r"ears?|nose|snout|muzzle|mouth|claw|claws|hooves?|horns?|scales?|skin|"
    r"neck|belly|coat|plumage|bill|talons?"
)

# Tier 1 – explicit morphological integration (anatomy borrowed across species)
_MORPHOLOGICAL = [
    # "a bird with a cat's face"  /  "X with Y's <body-part>"
    re.compile(
        rf"\b\w+\s*'s\s+(?:{_BODY_PARTS})\b",
        re.I,
    ),
    # "a hybrid bird-cat" / "hybrid" as standalone descriptor
    re.compile(r"\bhybrid\b", re.I),
    re.compile(r"\bchimera\b", re.I),
    re.compile(r"\bcross\s+between\b", re.I),
    # "part bird, part cat"
    re.compile(
        r"\bpart[\s-](?:bird|cat|dog|horse|fish|human|person|woman|man|animal)\b",
        re.I,
    ),
    # "bird-like cat" / "cat-like bird"
    re.compile(
        r"\b(?:bird|cat|dog|horse|fish|human|person)[\s-]like\s+"
        r"(?:bird|cat|dog|horse|fish|human|person|woman|man)\b",
        re.I,
    ),
    # generic "with <body-part>" where a non-native part is likely blended in
    re.compile(
        rf"\bwith\b.{{1,30}}\b(?:{_BODY_PARTS})\b",
        re.I,
    ),
]

# Tier 2 – anatomical anomaly: wrong body-part attributed to species
_ANATOMICAL = [
    # cat/dog/horse described as having beak/feathers/wings/talons (bird parts)
    re.compile(
        r"\b(?:cat|dog|horse|person|woman|man)\b.{1,50}"
        r"\b(?:beak|feathers?|wings?|talons?|plumage|bill)\b",
        re.I,
    ),
    # bird described as having fur/paws/hooves/mane (mammal parts)
    re.compile(
        r"\b(?:bird|parrot|sparrow|eagle|owl)\b.{1,50}"
        r"\b(?:fur|paws?|hooves?|mane|whiskers?)\b",
        re.I,
    ),
    # "a horse with long hair" counts as anatomical if it's crossed with person
    re.compile(r"\bwith\s+(?:long|short|curly|blonde|dark)\s+hair\b.{0,30}\bbird\b", re.I),
    re.compile(r"\bbird\b.{0,30}\bwith\s+(?:long|short|curly|blonde|dark)\s+hair\b", re.I),
]

# Tier 3 – coexistence / juxtaposition (two concepts present but not merged)
_COEXISTENCE_CONCEPTS = (
    r"cat|dog|bird|horse|person|woman|man|bus|car|truck|chair|umbrella|fish"
)
_JUXTAPOSITION = [
    re.compile(
        rf"\b(?:{_COEXISTENCE_CONCEPTS})\b.{{1,60}}\b(?:{_COEXISTENCE_CONCEPTS})\b",
        re.I,
    ),
]

_TIER_ORDER = ["strong", "medium", "weak"]
_TIER_LABEL = {
    "strong": "STRONG  ─ morphological chimera",
    "medium": "MEDIUM  ─ anatomical anomaly",
    "weak":   "WEAK    ─ coexistence / juxtaposition",
}
_TIER_MIN = {"strong": 0, "medium": 1, "weak": 2}


def classify_caption(caption: str) -> Optional[str]:
    """Return strongest tier matching caption, or None."""
    for pat in _MORPHOLOGICAL:
        if pat.search(caption):
            return "strong"
    for pat in _ANATOMICAL:
        if pat.search(caption):
            return "medium"
    for pat in _JUXTAPOSITION:
        if pat.search(caption):
            m = pat.search(caption)
            # Require the two matched concepts to be different
            if m:
                snippet = m.group(0).lower()
                concepts = re.findall(
                    rf"\b(?:{_COEXISTENCE_CONCEPTS})\b", snippet, re.I
                )
                uniq = set(c.lower() for c in concepts)
                if len(uniq) >= 2:
                    return "weak"
    return None


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------

# Matches: "Row R, Col C  :  seed S  →  prompt: "caption""
_ENTRY_RE = re.compile(
    r'Row\s+\d+,\s+Col\s+\d+\s*:\s*seed\s+(\d+)\s*→\s*prompt:\s*"([^"]+)"'
)
_CONDITION_RE = re.compile(r"^Condition\s*:\s*(.+)$", re.M)


def parse_manifest(path: Path) -> Tuple[str, List[Tuple[int, str]]]:
    """Return (condition_string, [(seed, caption), ...]) from a manifest file."""
    text = path.read_text(encoding="utf-8")

    m = _CONDITION_RE.search(text)
    condition = m.group(1).strip() if m else path.stem

    entries: List[Tuple[int, str]] = []
    for match in _ENTRY_RE.finditer(text):
        entries.append((int(match.group(1)), match.group(2)))

    return condition, entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Mine VLM manifest files for morphological chimera captions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--search-dir",
        default="experiments/inversion/gap_analysis",
        help="Root directory to recursively scan for *_manifest.txt files",
    )
    p.add_argument(
        "--condition",
        default=None,
        help="Only scan manifests whose condition string contains this substring "
             "(e.g. 'vlm', 'superdiff', 'monolithic'). Case-insensitive.",
    )
    p.add_argument(
        "--pair",
        default=None,
        help="Only scan manifests under a subdirectory whose name contains this "
             "substring (e.g. 'a_bird_a_cat').",
    )
    p.add_argument(
        "--min-tier",
        choices=["strong", "medium", "weak"],
        default="medium",
        help="Minimum fusion tier to report (default: medium).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional path to write results as JSON.",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output.",
    )
    args = p.parse_args()

    search_root = Path(args.search_dir)
    if not search_root.exists():
        print(f"[ERROR] search-dir not found: {search_root}", file=sys.stderr)
        sys.exit(1)

    min_rank = _TIER_MIN[args.min_tier]

    # ANSI helpers
    def _color(text, code):
        return text if args.no_color else f"\033[{code}m{text}\033[0m"

    tier_colors = {"strong": "1;31", "medium": "33", "weak": "36"}

    # Collect all manifest files
    manifests = sorted(search_root.rglob("*_manifest.txt"))
    if not manifests:
        print(f"No manifest files found under {search_root}")
        sys.exit(0)

    results = []

    for mf in manifests:
        # Filter by pair subdirectory name
        if args.pair and args.pair.lower() not in str(mf).lower():
            continue

        condition, entries = parse_manifest(mf)

        # Filter by condition string
        if args.condition and args.condition.lower() not in condition.lower():
            continue

        # Infer pair name from directory structure
        pair_name = mf.parent.parent.name  # …/pairs/<pair>/images/manifest.txt

        for seed, caption in entries:
            tier = classify_caption(caption)
            if tier is None:
                continue
            if _TIER_MIN[tier] > min_rank:
                continue

            results.append({
                "pair":      pair_name,
                "condition": condition,
                "seed":      seed,
                "caption":   caption,
                "tier":      tier,
                "manifest":  str(mf.relative_to(Path("."))),
            })

    if not results:
        print(f"No matches found at tier ≥ '{args.min_tier}'.")
        sys.exit(0)

    # Sort: tier first (strong → weak), then pair, then seed
    results.sort(key=lambda r: (_TIER_MIN[r["tier"]], r["pair"], r["seed"]))

    # ---- Terminal output ----
    print(f"\nFound {len(results)} match(es) "
          f"(min-tier: {args.min_tier}, condition filter: {args.condition or 'all'})\n")

    current_tier = None
    for r in results:
        if r["tier"] != current_tier:
            current_tier = r["tier"]
            print(_color(f"\n{'─'*70}", "90"))
            print(_color(f"  {_TIER_LABEL[current_tier]}", tier_colors[current_tier]))
            print(_color(f"{'─'*70}", "90"))

        print(
            f"  {_color(r['pair'], '1')}  "
            f"seed {_color(str(r['seed']), '33')}  │  "
            f"{_color(repr(r['caption']), tier_colors[r['tier']])}"
        )
        print(f"      ↳ {r['condition']}")

    print()

    # ---- Summary by tier ----
    print("Summary")
    print("-------")
    for tier in _TIER_ORDER:
        n = sum(1 for r in results if r["tier"] == tier)
        if n:
            print(f"  {_TIER_LABEL[tier]:45s}  {n:3d} match(es)")
    print()

    # ---- JSON output ----
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
