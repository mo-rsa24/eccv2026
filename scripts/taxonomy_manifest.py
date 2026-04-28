"""Canonical six-group taxonomy manifest with legacy compatibility helpers."""

from __future__ import annotations

import re
from typing import Any


TARGET_PAIRS_PER_GROUP = 10
DEFAULT_FINAL_SEEDS = [42] + list(range(1, 24))


GROUP_SPECS = [
    {
        "key": "group1_cooccurrence",
        "label": "Group 1 - Manifold-Supported Co-occurrence",
        "short_label": "G1\nCo-occurrence",
        "title": "Group 1: Manifold-Supported Co-occurrence",
        "color": "#4878CF",
        "representative_pair": ("a butterfly", "a flower meadow"),
        "pairs": [
            ("a butterfly", "a flower meadow"),
            ("a camel", "a desert landscape"),
            ("a deer", "a forest clearing"),
            ("a dolphin", "an ocean wave"),
            ("a duck", "a pond"),
            ("a flamingo", "a lagoon"),
            ("a horse", "a grassy field"),
            ("a lighthouse", "an ocean with stormy waves"),
            ("a polar bear", "an iceberg"),
            ("a sailboat", "a harbor"),
        ],
    },
    {
        "key": "group2_factorization",
        "label": "Group 2 - Feature-Space Factorization",
        "short_label": "G2\nFactorization",
        "title": "Group 2: Feature-Space Factorization",
        "color": "#6ACC65",
        "representative_pair": ("a dog", "oil painting style"),
        "pairs": [
            ("a dog", "oil painting style"),
            ("a lighthouse", "watercolour style"),
            ("a bicycle", "sketch style"),
            ("a teapot", "claymation style"),
            ("a barn", "pencil drawing style"),
            ("a cactus", "mosaic style"),
            ("a camera", "watercolor style"),
            ("a castle", "stained glass style"),
            ("a cat", "charcoal drawing style"),
            ("a train", "pixel art style"),
        ],
    },
    {
        "key": "group3_role_separable_object_scene",
        "label": "Group 3 - Role-Separable Object-Scene Composition",
        "short_label": "G3\nObject-Scene",
        "title": "Group 3: Role-Separable Object-Scene Composition",
        "color": "#4BAE73",
        "representative_pair": ("a picnic table", "a snowstorm"),
        "pairs": [
            ("a bookcase", "a glacier"),
            ("a candle", "a waterfall"),
            ("a fire hydrant", "a snowfield"),
            ("a lamppost", "a desert dune"),
            ("a lighthouse", "a desert dune"),
            ("a mailbox", "a snowfield"),
            ("a park bench", "a sand dune"),
            ("a phone booth", "a tropical beach"),
            ("a picnic table", "a snowstorm"),
            ("a rowboat", "a cactus garden"),
        ],
    },
    {
        "key": "group4_dual_object_composition",
        "label": "Group 4 - Dual-Object Composition",
        "short_label": "G4\nDual-Object",
        "title": "Group 4: Dual-Object Composition",
        "color": "#F5A623",
        "representative_pair": ("a typewriter", "a cactus"),
        "pairs": [
            ("a bathtub", "a streetlamp"),
            ("a birdcage", "a watering can"),
            ("a briefcase", "a ceramic bowl"),
            ("a chessboard", "a lantern"),
            ("a drum set", "a snowman"),
            ("a feather pillow", "a cast iron pan"),
            ("a lab microscope", "a hay bale"),
            ("a microwave", "a potted plant"),
            ("a suitcase", "a desk fan"),
            ("a typewriter", "a cactus"),
        ],
    },
    {
        "key": "group5_concept_prior_entanglement",
        "label": "Group 5 - Concept-Prior Entanglement",
        "short_label": "G5\nPrior Entanglement",
        "title": "Group 5: Concept-Prior Entanglement",
        "color": "#B279A2",
        "representative_pair": ("fluffy", "a stone"),
        "pairs": [
            ("fluffy", "a stone"),
            ("striped", "a sphere"),
            ("small", "an elephant"),
            ("a transparent glass", "a dog"),
            ("a fur coat", "a goldfish"),
            ("a winter coat", "a tropical parrot"),
            ("a wool scarf", "a jellyfish"),
            ("a ballerina", "a spacesuit"),
            ("a wedding dress", "a lobster"),
            ("a tuxedo", "a flamingo"),
        ],
    },
    {
        "key": "group6_coherent_collision",
        "label": "Group 6 - Coherent Collision",
        "short_label": "G6\nCollision",
        "title": "Group 6: Coherent Collision",
        "color": "#D7191C",
        "representative_pair": ("a fox", "a wolf"),
        "pairs": [
            ("a convertible", "a roadster"),
            ("a coupe", "a sedan"),
            ("a fox", "a wolf"),
            ("a goose", "a swan"),
            ("a leopard", "a cheetah"),
            ("a lion", "a leopard"),
            ("a pickup truck", "an SUV"),
            ("a raven", "a crow"),
            ("a sedan", "an SUV"),
            ("a wolf", "a husky"),
        ],
    },
]

TOTAL_PAIRS = len(GROUP_SPECS) * TARGET_PAIRS_PER_GROUP

GROUP_KEY_ALIASES = {
    "group2_disentangled": "group2_factorization",
    "group3_feature_overlap": "group3_role_separable_object_scene",
    "group3_ood": "group3_role_separable_object_scene",
    "group4_dual_object": "group4_dual_object_composition",
    "group4_coherent_collision": "group6_coherent_collision",
    "group4_collision": "group6_coherent_collision",
    "group5_entanglement": "group5_concept_prior_entanglement",
}

GROUP_DIR_ALIASES = {
    "group1_cooccurrence": ["group1_cooccurrence"],
    "group2_factorization": ["group2_factorization", "group2_disentangled"],
    "group3_role_separable_object_scene": [
        "group3_role_separable_object_scene",
        "group3_feature_overlap",
        "group3_ood",
    ],
    "group4_dual_object_composition": [
        "group4_dual_object_composition",
        "group3_feature_overlap",
        "group3_ood",
    ],
    "group5_concept_prior_entanglement": [
        "group5_concept_prior_entanglement",
        "group3_feature_overlap",
        "group3_ood",
    ],
    "group6_coherent_collision": [
        "group6_coherent_collision",
        "group4_coherent_collision",
        "group4_collision",
    ],
}


def normalize_group_key(group_key: str) -> str:
    return GROUP_KEY_ALIASES.get(group_key, group_key)


def slugify_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def pair_slug(c1: str, c2: str) -> str:
    return f"{slugify_text(c1)}_{slugify_text(c2)}"


def qualitative_pair_slug(c1: str, c2: str) -> str:
    return f"{slugify_text(c1)}__x__{slugify_text(c2)}"


def legacy_filesystem_pair_slug(c1: str, c2: str) -> str:
    clean_a = c1.lower().replace(" ", "_").replace("'", "")
    clean_b = c2.lower().replace(" ", "_").replace("'", "")
    return f"{clean_a}__x__{clean_b}"


GROUP_ORDER = [spec["key"] for spec in GROUP_SPECS]
GROUP_LABELS = [spec["label"] for spec in GROUP_SPECS]
GROUP_LABEL_BY_KEY = {spec["key"]: spec["label"] for spec in GROUP_SPECS}
GROUP_SHORT_LABEL_BY_KEY = {spec["key"]: spec["short_label"] for spec in GROUP_SPECS}
GROUP_TITLE_BY_KEY = {spec["key"]: spec["title"] for spec in GROUP_SPECS}
GROUP_COLOR_BY_KEY = {spec["key"]: spec["color"] for spec in GROUP_SPECS}


CANONICAL_PAIR_LOOKUP: dict[tuple[str, str], dict[str, Any]] = {}
CANONICAL_PAIR_LOOKUP_BY_SLUG: dict[str, dict[str, Any]] = {}
REPRESENTATIVE_PAIRS: list[tuple[str, str]] = []
REPRESENTATIVE_PAIR_SLUGS: list[str] = []
LARGE_REGIME_PAIRS: list[tuple[str, str]] = []

for group_index, spec in enumerate(GROUP_SPECS, start=1):
    representative = spec["representative_pair"]
    REPRESENTATIVE_PAIRS.append(representative)
    REPRESENTATIVE_PAIR_SLUGS.append(pair_slug(*representative))
    for pair_index_within_group, pair in enumerate(spec["pairs"], start=1):
        slug = pair_slug(*pair)
        if pair in CANONICAL_PAIR_LOOKUP:
            raise ValueError(f"Duplicate taxonomy pair: {pair}")
        meta = {
            "taxonomy_group_key": spec["key"],
            "taxonomy_group_label": spec["label"],
            "taxonomy_group_short_label": spec["short_label"],
            "taxonomy_group_title": spec["title"],
            "taxonomy_group_color": spec["color"],
            "group_index": group_index,
            "group_position": pair_index_within_group,
            "pair": pair,
            "pair_slug": slug,
            "qualitative_pair_slug": qualitative_pair_slug(*pair),
            "prompt_a": pair[0],
            "prompt_b": pair[1],
            "is_representative_pair": pair == representative,
            "is_canonical_pair": True,
        }
        CANONICAL_PAIR_LOOKUP[pair] = meta
        CANONICAL_PAIR_LOOKUP_BY_SLUG[slug] = meta
        CANONICAL_PAIR_LOOKUP_BY_SLUG[meta["qualitative_pair_slug"]] = meta
        CANONICAL_PAIR_LOOKUP_BY_SLUG[legacy_filesystem_pair_slug(*pair)] = meta
        LARGE_REGIME_PAIRS.append(pair)


LEGACY_GROUP_SPECS = [
    {
        "key": "group1_cooccurrence",
        "label": "Group 1 - Manifold-Supported Co-occurrence",
        "representative_pair": ("a butterfly", "a flower meadow"),
        "pairs": [
            ("a butterfly", "a flower meadow"),
            ("a camel", "a desert landscape"),
            ("a dolphin", "an ocean wave"),
            ("a lion", "a savanna at sunset"),
            ("a fox", "a snow-covered pine forest"),
            ("a lighthouse", "an ocean with stormy waves"),
        ],
    },
    {
        "key": "group2_disentangled",
        "label": "Group 2 - Feature-Space Disentangled",
        "representative_pair": ("a dog", "oil painting style"),
        "pairs": [
            ("a dog", "oil painting style"),
            ("a lighthouse", "watercolour style"),
            ("a bicycle", "sketch style"),
            ("a teapot", "claymation style"),
            ("a barn", "pencil drawing style"),
            ("a cactus", "mosaic style"),
        ],
    },
    {
        "key": "group3_feature_overlap",
        "label": "Group 3 - Feature Overlap / Low Co-occurrence",
        "representative_pair": ("a desk lamp", "a glacier"),
        "pairs": [
            ("a desk lamp", "a glacier"),
            ("a bathtub", "a streetlamp"),
            ("a lab microscope", "a hay bale"),
            ("a black grand piano", "a white vase"),
            ("a typewriter", "a cactus"),
            ("a bird", "a book"),
        ],
    },
    {
        "key": "group4_coherent_collision",
        "label": "Group 4 - Coherent Collision",
        "representative_pair": ("a cat", "a dog"),
        "pairs": [
            ("a cat", "a dog"),
            ("a cat", "a bear"),
            ("a cat", "an owl"),
            ("a tiger", "a lion"),
            ("a man with black hair and black shirt", "a red umbrella"),
            ("a red bmw", "a white canopy truck"),
        ],
    },
]

LEGACY_GROUP3_SUBGROUP_SPECS = [
    {
        "key": "group3a_missing_support",
        "label": "3a - Missing Joint Support",
        "failure_pattern": "One concept weakened or erased",
        "representative_pair": ("a penguin", "a desert landscape"),
        "pairs": [
            ("a penguin", "a desert landscape"),
            ("a snowman", "a tropical beach"),
            ("a cactus", "the Arctic tundra"),
        ],
    },
    {
        "key": "group3b_entanglement",
        "label": "3b - Feature Entanglement",
        "failure_pattern": "Attribute binding error; applying one attribute leaks into the other",
        "representative_pair": ("a red", "a cube"),
        "pairs": [
            ("a red", "a cube"),
            ("small", "an elephant"),
            ("striped", "a sphere"),
        ],
    },
    {
        "key": "group3c_interference",
        "label": "3c - Weak Interference",
        "failure_pattern": "Texture/material conflict; inconsistent rendering across seeds",
        "representative_pair": ("a wooden chair", "metallic texture"),
        "pairs": [
            ("a wooden chair", "metallic texture"),
            ("a transparent glass", "a dog"),
            ("fluffy", "a stone"),
        ],
    },
]


LEGACY_PAIR_LOOKUP: dict[tuple[str, str], dict[str, Any]] = {}
LEGACY_PAIR_LOOKUP_BY_SLUG: dict[str, dict[str, Any]] = {}

for spec in LEGACY_GROUP_SPECS:
    representative = spec["representative_pair"]
    for pair_index_within_group, pair in enumerate(spec["pairs"], start=1):
        meta = {
            "taxonomy_group_key": spec["key"],
            "taxonomy_group_label": spec["label"],
            "group_position": pair_index_within_group,
            "pair": pair,
            "pair_slug": pair_slug(*pair),
            "qualitative_pair_slug": qualitative_pair_slug(*pair),
            "prompt_a": pair[0],
            "prompt_b": pair[1],
            "is_representative_pair": pair == representative,
            "is_canonical_pair": False,
            "legacy_taxonomy": True,
        }
        LEGACY_PAIR_LOOKUP[pair] = meta
        LEGACY_PAIR_LOOKUP_BY_SLUG.setdefault(meta["pair_slug"], meta)
        LEGACY_PAIR_LOOKUP_BY_SLUG.setdefault(meta["qualitative_pair_slug"], meta)
        LEGACY_PAIR_LOOKUP_BY_SLUG.setdefault(legacy_filesystem_pair_slug(*pair), meta)


GROUP3_SUBGROUP_SPECS = LEGACY_GROUP3_SUBGROUP_SPECS
GROUP3_SUBGROUP_ORDER = [s["key"] for s in GROUP3_SUBGROUP_SPECS]
GROUP3_SUBGROUP_LABEL_BY_KEY = {s["key"]: s["label"] for s in GROUP3_SUBGROUP_SPECS}
GROUP3_REPRESENTATIVE_PAIRS = [s["representative_pair"] for s in GROUP3_SUBGROUP_SPECS]
GROUP3_REPRESENTATIVE_PAIR_SLUGS = [pair_slug(*s["representative_pair"]) for s in GROUP3_SUBGROUP_SPECS]
GROUP3_SUBGROUP_FAILURE_BY_KEY = {s["key"]: s["failure_pattern"] for s in GROUP3_SUBGROUP_SPECS}

GROUP3_SUBGROUP_LOOKUP: dict[tuple[str, str], dict[str, Any]] = {}
GROUP3_SUBGROUP_LOOKUP_BY_SLUG: dict[str, dict[str, Any]] = {}
for subgroup_index, spec in enumerate(GROUP3_SUBGROUP_SPECS, start=1):
    representative = spec["representative_pair"]
    for pair_index_within_group, pair in enumerate(spec["pairs"], start=1):
        meta = {
            "taxonomy_group_key": spec["key"],
            "taxonomy_group_label": spec["label"],
            "group_index": subgroup_index,
            "group_position": pair_index_within_group,
            "pair": pair,
            "pair_slug": pair_slug(*pair),
            "qualitative_pair_slug": qualitative_pair_slug(*pair),
            "prompt_a": pair[0],
            "prompt_b": pair[1],
            "is_representative_pair": pair == representative,
            "failure_pattern": spec["failure_pattern"],
            "is_canonical_pair": False,
            "legacy_taxonomy": True,
        }
        GROUP3_SUBGROUP_LOOKUP[pair] = meta
        GROUP3_SUBGROUP_LOOKUP_BY_SLUG[meta["pair_slug"]] = meta
        GROUP3_SUBGROUP_LOOKUP_BY_SLUG[meta["qualitative_pair_slug"]] = meta
        GROUP3_SUBGROUP_LOOKUP_BY_SLUG[legacy_filesystem_pair_slug(*pair)] = meta


PAIR_LOOKUP: dict[tuple[str, str], dict[str, Any]] = {}
PAIR_LOOKUP.update(LEGACY_PAIR_LOOKUP)
PAIR_LOOKUP.update(CANONICAL_PAIR_LOOKUP)

PAIR_LOOKUP_BY_SLUG: dict[str, dict[str, Any]] = {}
PAIR_LOOKUP_BY_SLUG.update(LEGACY_PAIR_LOOKUP_BY_SLUG)
PAIR_LOOKUP_BY_SLUG.update(GROUP3_SUBGROUP_LOOKUP_BY_SLUG)
PAIR_LOOKUP_BY_SLUG.update(CANONICAL_PAIR_LOOKUP_BY_SLUG)


def validate_taxonomy_manifest() -> None:
    if len(GROUP_SPECS) != 6:
        raise ValueError(f"Expected 6 taxonomy groups, got {len(GROUP_SPECS)}")
    for spec in GROUP_SPECS:
        if len(spec["pairs"]) != TARGET_PAIRS_PER_GROUP:
            raise ValueError(
                f"{spec['key']} expected {TARGET_PAIRS_PER_GROUP} pairs, got {len(spec['pairs'])}"
            )
        if spec["representative_pair"] not in spec["pairs"]:
            raise ValueError(
                f"Representative pair {spec['representative_pair']} is not in {spec['key']}"
            )
    if len(CANONICAL_PAIR_LOOKUP) != TOTAL_PAIRS:
        raise ValueError(
            f"Expected {TOTAL_PAIRS} canonical taxonomy pairs, got {len(CANONICAL_PAIR_LOOKUP)}"
        )


validate_taxonomy_manifest()


def get_pair_taxonomy_record(c1: str, c2: str) -> dict[str, Any] | None:
    return (
        CANONICAL_PAIR_LOOKUP.get((c1, c2))
        or GROUP3_SUBGROUP_LOOKUP.get((c1, c2))
        or LEGACY_PAIR_LOOKUP.get((c1, c2))
    )


def get_pair_taxonomy_from_slug(slug: str) -> dict[str, Any] | None:
    return (
        CANONICAL_PAIR_LOOKUP_BY_SLUG.get(slug)
        or GROUP3_SUBGROUP_LOOKUP_BY_SLUG.get(slug)
        or LEGACY_PAIR_LOOKUP_BY_SLUG.get(slug)
    )


def get_pair_taxonomy_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    c1 = row.get("c1")
    c2 = row.get("c2")
    if c1 and c2:
        meta = get_pair_taxonomy_record(str(c1), str(c2))
        if meta is not None:
            return meta

    pair_value = row.get("pair")
    if isinstance(pair_value, str) and " + " in pair_value:
        parts = pair_value.split(" + ", 1)
        return get_pair_taxonomy_record(parts[0], parts[1])
    if isinstance(pair_value, (list, tuple)) and len(pair_value) == 2:
        return get_pair_taxonomy_record(str(pair_value[0]), str(pair_value[1]))

    pair_slug_value = row.get("pair_slug") or row.get("slug") or row.get("qualitative_pair_slug")
    if pair_slug_value:
        return get_pair_taxonomy_from_slug(str(pair_slug_value))
    return None


def taxonomy_manifest_rows() -> list[dict[str, Any]]:
    rows = []
    for spec in GROUP_SPECS:
        for pair in spec["pairs"]:
            meta = CANONICAL_PAIR_LOOKUP[pair]
            rows.append(
                {
                    "taxonomy_group_key": meta["taxonomy_group_key"],
                    "taxonomy_group_label": meta["taxonomy_group_label"],
                    "pair_slug": meta["pair_slug"],
                    "qualitative_pair_slug": meta["qualitative_pair_slug"],
                    "prompt_a": meta["prompt_a"],
                    "prompt_b": meta["prompt_b"],
                    "is_representative": meta["is_representative_pair"],
                }
            )
    return rows


def taxonomy_manifest_payload() -> dict[str, Any]:
    return {
        "group_order": GROUP_ORDER,
        "target_pairs_per_group": TARGET_PAIRS_PER_GROUP,
        "total_pairs": TOTAL_PAIRS,
        "default_final_seeds": list(DEFAULT_FINAL_SEEDS),
        "groups": [
            {
                "taxonomy_group_key": spec["key"],
                "taxonomy_group_label": spec["label"],
                "taxonomy_group_short_label": spec["short_label"],
                "taxonomy_group_title": spec["title"],
                "taxonomy_group_color": spec["color"],
                "representative_pair": list(spec["representative_pair"]),
                "representative_pair_slug": pair_slug(*spec["representative_pair"]),
                "pairs": [
                    {
                        "pair": list(pair),
                        "pair_slug": pair_slug(*pair),
                        "qualitative_pair_slug": qualitative_pair_slug(*pair),
                        "is_representative_pair": pair == spec["representative_pair"],
                    }
                    for pair in spec["pairs"]
                ],
            }
            for spec in GROUP_SPECS
        ],
    }
