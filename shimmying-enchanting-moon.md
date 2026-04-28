# SDXL Pair Audit, Cleaning, and Roster Freeze Pipeline

## Context

Groups 3 and 4 of the 24-pair SDXL taxonomy are all provisional — no pair clears the `pair_pass_threshold=0.75` semantic baseline. A replacement screening run (`sdxl_g34_replacements_seed42_steps50_cfg7p5_20260413`) has 14 new candidates (8 for g3, only 4 for g4) but all are still provisional. The goal is to: (1) identify which probes are truly redundant vs genuinely independent, (2) programmatically clean the screening run by deleting bad pair folders and updating all JSON/CSV files, and (3) assemble and freeze a clean 24-pair roster for a fresh SDXL generation run.

**Scoring bottleneck:** `intentional_composition` (tier2 MIN probe) consistently scores 0.20–0.37 on g3/g4 pairs, triggering the semantic drift gate and collapsing `quality = tier2 alone`. This is why all g3/g4 pairs fail `semantic_pass=0.58`. The cleaning pipeline must work with this reality — provisional pairs will be accepted with `--allow-provisional`.

**Two screening runs:**
- `sdxl_taxonomy_seed42_steps50_cfg7p5_20260413/` — g1 (6/6 accepted), g2 (5/6 accepted), g3 (6/6 provisional), g4 (6/6 provisional)
- `sdxl_g34_replacements_seed42_steps50_cfg7p5_20260413/` — g3 (8 candidates, all provisional), g4 (4 candidates, all provisional — 2 short)

**Critical PAIR_LOOKUP_BY_SLUG gap:** `freeze_sdxl_taxonomy_roster.py` cannot be called directly for g3/g4 because it calls `PAIR_LOOKUP_BY_SLUG[slug]` unconditionally. G3 replacements are only in `GROUP3_SUBGROUP_LOOKUP_BY_SLUG`; novel g4 pairs (`teddy_bear×panda`, `otter×duck`) are in neither. `assemble_sdxl_roster.py` must handle this with a fallback lookup.

---

## Files to Create

| Script | Purpose |
|---|---|
| `scripts/analyze_probe_redundancy.py` | Probe correlation/redundancy report from inspection JSONs |
| `scripts/audit_and_clean_sdxl_pairs.py` | Report, visualize, and delete failing pair folders |
| `scripts/assemble_sdxl_roster.py` | Merge both runs → single `sdxl_paper_roster.json` |

## Critical Files to Reference (read-only)

- `scripts/semantic_baseline_common.py` — `build_semantic_baseline_audit()`, `_score_sort_key()`, `build_audit_from_joint_probe_file()` — **reuse these, do not reimplement**
- `scripts/freeze_sdxl_taxonomy_roster.py` — `build_roster_manifest()` — **follow its manifest JSON schema exactly**
- `scripts/eval_joint_probes.py` — `_probe_spec()`, `_aggregate_scores()`, scoring constants
- `scripts/taxonomy_manifest.py` — `PAIR_LOOKUP_BY_SLUG`, `GROUP3_SUBGROUP_LOOKUP_BY_SLUG`, `GROUP_SPECS`, `_score_sort_key`
- `scripts/run_taxonomy_qualitative_sdxl.py` — `_load_roster_manifest()` — Stage 3, **no changes needed**

---

## Script 1: `scripts/analyze_probe_redundancy.py`

**Purpose:** Read all `**/probe_inspection/monolithic_probe_inspection.json` files, compute pairwise probe correlation + disagreement rates, identify probes that fire on otherwise-good images (false penalization). Output a structured report. **Conservative by design** — only marks a probe as `REMOVE_CANDIDATE` if r > 0.85 AND disagreement < 10% AND false-pen < 20%. With N=7 currently, expect `INSUFFICIENT_DATA` for most pairs.

**Key functions:**

```python
def load_inspection_files(data_dirs: list[Path]) -> list[dict]:
    """Walk data_dirs rglob('**/probe_inspection/monolithic_probe_inspection.json')"""

def build_probe_matrix(
    payloads: list[dict],
    require_probes: list[str] | None = None,
) -> tuple[dict[str, list[float]], list[str]]:
    """Extract per-probe raw score vectors. Skip payloads missing any required probe."""

def compute_pairwise_correlation(
    scores_by_probe: dict[str, list[float]],
) -> dict[tuple[str, str], float]:
    """Pearson r for every unordered probe pair (use raw scores, not inverted)."""

def compute_disagreement_rate(
    scores_by_probe: dict[str, list[float]],
    *,
    pass_threshold: float = 0.58,
) -> dict[tuple[str, str], float]:
    """Fraction of images where probes disagree on pass/fail.
    Positive probe passes if score >= threshold; negative probe passes if score <= 1-threshold."""

def compute_false_penalization_rate(
    scores_by_probe: dict[str, list[float]],
    probe_polarity: dict[str, bool],
    payloads: list[dict],
    *,
    good_image_threshold: float = 0.60,
) -> dict[str, float]:
    """For images where aggregate joint_correctness_score >= good_image_threshold,
    fraction that are still penalized by each individual probe."""

def print_report(..., output_path: Path | None = None) -> None:
    """Print 3-section report:
    1. Per-probe summary (mean, stdev, false_pen_rate, n_images)
    2. User-suspected pairs: one_missing/both_present, merged_or_absorbed/distinct_entities,
       intentional_composition standalone, semantic_role_correct standalone
    3. All pairs with r > high_correlation_threshold
    Recommendation column: KEEP / REMOVE_CANDIDATE / INSUFFICIENT_DATA"""
```

**CLI:**
```bash
python scripts/analyze_probe_redundancy.py \
    --data-dirs experiments/eccv2026/sdxl_screening/sdxl_g34_replacements_seed42_steps50_cfg7p5_20260413 \
    [--output-report results/probe_redundancy_report.txt] \
    [--high-correlation-threshold 0.80] \
    [--pass-threshold 0.58]
```

**Note:** Warn prominently when N < 10 — correlations unreliable. Do NOT import from `eval_joint_probes.py`; read pre-computed JSON only.

---

## Script 2: `scripts/audit_and_clean_sdxl_pairs.py`

**Purpose:** Three-action tool for a single `--data-dir`. `report` is read-only. `show-visuals` prints PNG paths. `delete-failing` removes pair folders and rebuilds all JSON/CSV files.

**Key functions:**

```python
def collect_pair_dirs(data_dir: Path) -> dict[str, Path]:
    """Walk data_dir/group*/pair_slug/ → {pair_slug: path}"""

def collect_probe_inspection_paths(data_dir: Path) -> dict[str, Path]:
    """Walk **/probe_inspection/monolithic_probe_inspection.json → {pair_slug: json_path}"""

def build_report_table(data_dir: Path, pair_pass_threshold: float) -> list[dict]:
    """One row per pair. Fields: pair_slug, taxonomy_group_key, mono_mean_joint_correctness_score,
    mono_semantic_pass_rate, pair_eligible_for_semantic_baseline,
    tier2_quality_score (from inspection JSON if exists, else None),
    semantic_drift_flagged, has_inspection_file, pair_dir_exists,
    failure_modes: list from {low_overall, tier2_drift, hybridization_failure,
    omission_failure, no_inspection}
    Sorted by score descending."""

def delete_pair(pair_slug: str, data_dir: Path, *, dry_run: bool) -> None:
    """shutil.rmtree on the pair directory. On dry_run, only print."""

def filter_joint_probe_scores(payload: dict, pair_slugs_to_keep: set[str]) -> dict:
    """Return new payload with probe_records, image_scores, pair_summaries
    filtered to pair_slugs_to_keep. Preserve all top-level metadata."""

def rebuild_audit_files(data_dir: Path, pair_pass_threshold: float) -> None:
    """1. Load filtered joint_probe_scores.json
    2. Call build_semantic_baseline_audit() from semantic_baseline_common
    3. Overwrite semantic_baseline_audit.json + 4 CSV files (_seed, _pair, _group, _roster)"""
```

**Delete-failing logic:**
1. Must provide either `--pair-slugs` (explicit) or `--min-score` (auto-threshold)
2. If neither: print report and exit with message asking user to re-run with explicit slugs
3. `--dry-run` always available — prints what would be deleted without touching files
4. After deletion: filter JSON → overwrite → rebuild audit
5. Warn if any group drops below 6 candidates after deletion

**CLI:**
```bash
python scripts/audit_and_clean_sdxl_pairs.py \
    --data-dir experiments/eccv2026/sdxl_screening/sdxl_g34_replacements_seed42_steps50_cfg7p5_20260413 \
    --action {report|show-visuals|delete-failing} \
    [--min-score 0.490]
    [--pair-slugs slug1 slug2]
    [--pair-pass-threshold 0.75]
    [--dry-run]
```

**Table columns for `report`:**
`GROUP | PAIR_SLUG | JOINT_SCORE | PASS_RATE | DRIFT? | FAILURE_MODES | STATUS`
Status: `ACCEPTED` / `PROVISIONAL` / `FAILING`

---

## Script 3: `scripts/assemble_sdxl_roster.py`

**Purpose:** Cross-run merge. G1+G2 from original taxonomy run; G3+G4 from replacements run, supplemented from original run when replacements run is short (g4 currently has only 4 candidates). Output follows `freeze_sdxl_taxonomy_roster.py`'s manifest schema exactly.

**Key functions:**

```python
def _extended_pair_lookup(pair_slug: str) -> dict:
    """Try PAIR_LOOKUP_BY_SLUG → GROUP3_SUBGROUP_LOOKUP_BY_SLUG → synthesize from slug.
    Never returns None. Synthetic: split on __x__, derive qualitative_pair_slug."""

def load_pair_rows_for_group(
    data_dir: Path, group_key: str, pair_pass_threshold: float,
) -> list[dict]:
    """Load joint_probe_scores.json → build_semantic_baseline_audit()
    → filter pair_rows to group_key."""

def merge_pair_rows_for_group(
    rows_orig: list[dict], rows_repl: list[dict], *, prefer_replacements: bool = True,
) -> list[dict]:
    """Union by pair_slug. When prefer_replacements=True, keep replacements-run row
    for pairs appearing in both runs."""

def select_top_n(pair_rows: list[dict], n: int) -> list[dict]:
    """Sort by semantic_baseline_common._score_sort_key, return top n."""

def build_group_entry(
    group_key: str, selected_rows: list[dict], spec: dict,
) -> dict:
    """Build manifest groups[] entry using _extended_pair_lookup() for qualitative slugs.
    Mirror freeze_sdxl_taxonomy_roster.py build_roster_manifest() group entry structure."""

def assemble_roster(
    *, orig_run_dir: Path, replacements_run_dir: Path,
    pair_pass_threshold: float, allow_provisional: bool,
) -> tuple[dict, list[str]]:
    """Per-group strategy:
    - g1, g2: load_pair_rows_for_group(orig_run_dir, ...)
    - g3, g4: merge(orig, replacements), select_top_n(6)
    Returns (manifest_dict, errors_list).
    errors contains messages for provisional pairs and shortfalls."""
```

**CLI:**
```bash
python scripts/assemble_sdxl_roster.py \
    --orig-run-dir experiments/eccv2026/sdxl_screening/sdxl_taxonomy_seed42_steps50_cfg7p5_20260413 \
    --replacements-run-dir experiments/eccv2026/sdxl_screening/sdxl_g34_replacements_seed42_steps50_cfg7p5_20260413 \
    --output experiments/eccv2026/sdxl_screening/sdxl_paper_roster.json \
    [--pair-pass-threshold 0.75]
    [--allow-provisional]   # required until g3/g4 clear threshold
    [--strict]              # error on any provisional pair
```

---

## End-to-End Workflow

### Stage 1 — Pair-Type-Aware Semantic Audit

```bash
# 1a. Probe redundancy report (diagnostic)
python scripts/analyze_probe_redundancy.py \
    --data-dirs experiments/eccv2026/sdxl_screening/sdxl_g34_replacements_seed42_steps50_cfg7p5_20260413 \
    --output-report results/probe_redundancy_report.txt

# 1b. Report current state of both runs
python scripts/audit_and_clean_sdxl_pairs.py \
    --data-dir experiments/eccv2026/sdxl_screening/sdxl_taxonomy_seed42_steps50_cfg7p5_20260413 \
    --action report
python scripts/audit_and_clean_sdxl_pairs.py \
    --data-dir experiments/eccv2026/sdxl_screening/sdxl_g34_replacements_seed42_steps50_cfg7p5_20260413 \
    --action report

# 1c. Show paths to probe inspection PNGs for visual review
python scripts/audit_and_clean_sdxl_pairs.py \
    --data-dir experiments/eccv2026/sdxl_screening/sdxl_g34_replacements_seed42_steps50_cfg7p5_20260413 \
    --action show-visuals
# Open each PNG. Lowest two g3 scorers are:
#   a_bathtub__x__a_streetlamp (0.493) and a_snowman__x__a_tropical_beach (0.490)

# 1d. Dry run then commit deletions of visually confirmed failures
python scripts/audit_and_clean_sdxl_pairs.py \
    --data-dir experiments/eccv2026/sdxl_screening/sdxl_g34_replacements_seed42_steps50_cfg7p5_20260413 \
    --action delete-failing \
    --pair-slugs a_bathtub__x__a_streetlamp a_snowman__x__a_tropical_beach \
    --dry-run
# If satisfied:
python scripts/audit_and_clean_sdxl_pairs.py \
    --data-dir experiments/eccv2026/sdxl_screening/sdxl_g34_replacements_seed42_steps50_cfg7p5_20260413 \
    --action delete-failing \
    --pair-slugs a_bathtub__x__a_streetlamp a_snowman__x__a_tropical_beach
# g3 replacements now has exactly 6 candidates
```

### Stage 2 — Freeze The SDXL Paper Roster

```bash
python scripts/assemble_sdxl_roster.py \
    --orig-run-dir experiments/eccv2026/sdxl_screening/sdxl_taxonomy_seed42_steps50_cfg7p5_20260413 \
    --replacements-run-dir experiments/eccv2026/sdxl_screening/sdxl_g34_replacements_seed42_steps50_cfg7p5_20260413 \
    --output experiments/eccv2026/sdxl_screening/sdxl_paper_roster.json \
    --allow-provisional
```

Verify: `sdxl_paper_roster.json` has 24 pairs (6 per group), `summary.n_selected_pairs == 24`.

### Stage 3 — Fresh Final SDXL Run From The Frozen Roster

```bash
# Single GPU
python scripts/run_taxonomy_qualitative_sdxl.py \
    --roster-manifest experiments/eccv2026/sdxl_screening/sdxl_paper_roster.json \
    --output-dir experiments/eccv2026/taxonomy_qualitative_sdxl_final \
    --seed 42 --num-inference-steps 50 --guidance-scale 7.5

# OR 4-GPU parallel (existing worker pattern)
for i in 0 1 2 3; do
  python scripts/run_taxonomy_qualitative_sdxl.py \
      --roster-manifest experiments/eccv2026/sdxl_screening/sdxl_paper_roster.json \
      --output-dir experiments/eccv2026/taxonomy_qualitative_sdxl_final \
      --num-workers 4 --worker-index $i --gpu-id $i &
done; wait
```

No changes needed to `run_taxonomy_qualitative_sdxl.py` — it already supports `--roster-manifest`.

---

## Probe Redundancy Analysis: Key Expected Findings

Based on current data from `monolithic_probe_inspection.json` files (N=7):

| Suspected Redundant Pair | Relationship | Recommendation |
|---|---|---|
| `one_missing` vs `both_present` | both_present (positive) answers the same question as one_missing (negative). BUT: they use different linguistic frames and BLIP-VQA may respond differently to each. Disagreement rate determines if they're truly redundant. | KEEP BOTH (different frames) |
| `merged_or_absorbed` vs `distinct_entities` | distinct_entities asks the same thing positively. High correlation expected. | KEEP BOTH (polarity diversity) |
| `intentional_composition` | Consistently lowest tier2 probe (0.20–0.37 for g3). Drives semantic drift gate. | KEEP — it IS the signal |
| `concept_confusion` | Scores 0.30–0.59; unclear BLIP reliability. | KEEP — flag N too small to conclude |
| `semantic_role_correct` | Awkward question framing ("main subject is NOT x AND y"). | KEEP but note for paper methods |

The redundancy script will quantify these, but with N=7 the report serves primarily as documentation for the paper's methods section.

---

## Verification

After Stage 1 deletion:
- `report --action` on replacements run shows exactly 6 g3 pairs remaining
- `joint_probe_scores.json` has no records for deleted pair slugs

After Stage 2:
- `sdxl_paper_roster.json` has `summary.n_selected_pairs == 24`
- All 4 groups appear in `groups[]`
- `selected_qualitative_pair_slugs` has 24 entries — these are what Stage 3 uses

After Stage 3:
- `experiments/eccv2026/taxonomy_qualitative_sdxl_final/` has 24 pair subdirectories
- Each has `monolithic.png`, `poe.png`, `decoded_images.png`, `trajectory_manifold.png`
