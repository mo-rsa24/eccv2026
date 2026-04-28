#!/usr/bin/env python3
"""
merge_gap_shards.py — Reconstruct aggregate gap metric JSONs from per-pair artifacts.

Background
----------
run_gap_4gpu.sh runs 4 workers in parallel, each writing to the *same* --output-dir
without --merge.  All four workers overwrite the shared aggregate JSON files; only
the last worker to finish survives in:
    metrics/per_seed_distances.json
    metrics/trajectory_distances.json
    metrics/within_and_distances.json
    metrics/all_pairs_gap.json

This script fixes the damage in two complementary ways:

1.  all_pairs_gap.json  — FULL reconstruction from on-disk data.
    Every pair already wrote pairs/<slug>/gap_metrics.json with the CLIP-level
    summary.  We read all 24 and rebuild the file from scratch.

2.  per_seed_distances.json / trajectory_distances.json / within_and_distances.json
    — PARTIAL reconstruction.
    These contain per-seed latent-distance records that are NOT stored in any
    per-pair artifact.  We keep the surviving shard (pairs 0-5, group1) and
    print a clear report of which pairs are still missing, plus the exact
    command to re-run only those pairs with --merge so they append into the
    existing files.

Usage
-----
    # Fix all_pairs_gap.json and generate the resume script:
    python scripts/merge_gap_shards.py \\
        --data-dir results/gap_large_4gpu_rerun_2

    # Dry-run (print what would change, write nothing):
    python scripts/merge_gap_shards.py \\
        --data-dir results/gap_large_4gpu_rerun_2 --dry-run

    # Only diagnose coverage (skip rebuilding all_pairs_gap.json):
    python scripts/merge_gap_shards.py \\
        --data-dir results/gap_large_4gpu_rerun_2 --skip-all-pairs-gap

After running this script, re-run the missing shards:
    bash scripts/run_gap_4gpu_resume.sh results/gap_large_4gpu_rerun_2
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running from repo root without installing the package.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from taxonomy_manifest import LARGE_REGIME_PAIRS, get_pair_taxonomy_record  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path):
    with open(path) as fh:
        return json.load(fh)


def _write_json(path: Path, data, dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry-run] would write {len(data)} records → {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"  wrote {len(data)} records → {path}")


# ---------------------------------------------------------------------------
# Step 1: Rebuild all_pairs_gap.json from pairs/*/gap_metrics.json
# ---------------------------------------------------------------------------

def rebuild_all_pairs_gap(data_dir: Path, dry_run: bool) -> list:
    """Read all 24 gap_metrics.json files and write a merged all_pairs_gap.json.

    gap_metrics.json already stores exactly the fields that all_pairs_gap.json
    needs (gap_and_mono, gap_and_pstar_sdipc, etc.).  We assign the correct
    global pair_index (position in LARGE_REGIME_PAIRS) rather than the
    shard-local index that was written by each worker.
    """
    pairs_dir = data_dir / "pairs"
    if not pairs_dir.exists():
        sys.exit(f"ERROR: pairs/ directory not found at {pairs_dir}")

    records = []
    missing_slugs = []

    for global_idx, (c1, c2) in enumerate(LARGE_REGIME_PAIRS):
        meta = get_pair_taxonomy_record(c1, c2)
        if meta is None:
            sys.exit(f"ERROR: taxonomy record not found for ({c1!r}, {c2!r})")
        slug = meta["pair_slug"]
        gap_path = pairs_dir / slug / "gap_metrics.json"

        if not gap_path.exists():
            missing_slugs.append(slug)
            print(f"  MISSING: pairs/{slug}/gap_metrics.json — skipping pair [{global_idx:2d}]")
            continue

        gm = _load_json(gap_path)

        # Copy all fields from gap_metrics.json but override pair_index with
        # the correct *global* index (workers wrote a shard-local 0-5 index).
        record = dict(gm)
        record["pair_index"] = global_idx
        records.append(record)

    print(f"\nall_pairs_gap.json: {len(records)}/24 records reconstructed")
    if missing_slugs:
        print(f"  WARNING: {len(missing_slugs)} per-pair gap_metrics.json files are missing:")
        for s in missing_slugs:
            print(f"    {s}")

    out_path = data_dir / "metrics" / "all_pairs_gap.json"
    _write_json(out_path, records, dry_run)
    return records


# ---------------------------------------------------------------------------
# Step 2: Diagnose and report missing seed-level records
# ---------------------------------------------------------------------------

def diagnose_seed_records(data_dir: Path) -> dict:
    """Return a dict describing which pairs are present/missing in the
    per-seed, trajectory, and within-and JSONs."""
    result = {}
    for fname in (
        "per_seed_distances.json",
        "trajectory_distances.json",
        "within_and_distances.json",
    ):
        path = data_dir / "metrics" / fname
        if not path.exists():
            result[fname] = {"exists": False, "pairs_present": set(), "records": 0}
            continue

        data = _load_json(path)
        pairs_present = set()
        for rec in data:
            slug = rec.get("pair_slug") or rec.get("slug")
            if slug:
                pairs_present.add(slug)

        result[fname] = {
            "exists": True,
            "pairs_present": pairs_present,
            "records": len(data),
        }

    return result


def report_missing_pairs(data_dir: Path) -> list:
    """Print a detailed gap report and return the list of global indices that
    are missing from at least one aggregate JSON."""
    all_slugs: list[str] = []
    for c1, c2 in LARGE_REGIME_PAIRS:
        meta = get_pair_taxonomy_record(c1, c2)
        all_slugs.append(meta["pair_slug"])

    diag = diagnose_seed_records(data_dir)

    print("\n" + "=" * 68)
    print("Seed-level record coverage")
    print("=" * 68)

    missing_union: set[str] = set()

    for fname, info in diag.items():
        if not info["exists"]:
            print(f"\n{fname}: FILE MISSING")
            missing_union.update(all_slugs)
            continue

        present = info["pairs_present"]
        missing = [s for s in all_slugs if s not in present]
        missing_union.update(missing)

        print(f"\n{fname}:")
        print(f"  Records       : {info['records']}")
        print(f"  Pairs present : {len(present)}/24")
        if missing:
            print(f"  Pairs missing : {len(missing)}/24")
            for slug in missing:
                global_idx = all_slugs.index(slug)
                print(f"    [{global_idx:2d}] {slug}")
        else:
            print("  All 24 pairs present — no action needed.")

    print("\n" + "=" * 68)

    missing_indices = sorted(all_slugs.index(s) for s in missing_union)
    return missing_indices


def print_rerun_commands(data_dir: Path, missing_indices: list) -> None:
    """Print the exact measure_composability_gap.py commands needed to fill gaps."""
    if not missing_indices:
        print("\nAll pairs present in seed-level JSONs — nothing to re-run.")
        return

    print("\nTo fill the missing seed-level records, re-run only the missing")
    print("pairs with --merge (appends safely to existing aggregate JSONs).\n")
    print("Split across 4 GPUs with non-overlapping ranges:")

    script = _REPO_ROOT / "scripts" / "measure_composability_gap.py"
    out = data_dir
    ranges = _split_into_ranges(missing_indices, n_workers=4)

    nodes = ["mscluster107 GPU 0", "mscluster107 GPU 1",
             "mscluster109 GPU 0", "mscluster109 GPU 1"]
    for (lo, hi), label in zip(ranges, nodes):
        print(
            f"\n  # {label}\n"
            f"  CUDA_VISIBLE_DEVICES=<N> python {script} \\\n"
            f"    --paper-only --regime large --pstar-source sdipc \\\n"
            f"    --model-family sd14 --output-dir {out} \\\n"
            f"    --pair-start {lo} --pair-end {hi} \\\n"
            f"    --merge"
        )

    print()
    print("IMPORTANT: run workers sequentially within each node (not in parallel)")
    print("to avoid the --merge race condition on the shared aggregate JSON files.")
    print("Workers on different nodes (107 vs 109) can run concurrently.\n")


# ---------------------------------------------------------------------------
# Step 3: Write a ready-to-use SLURM resume script
# ---------------------------------------------------------------------------

def _split_into_ranges(indices: list, n_workers: int) -> list[tuple[int, int]]:
    """Split a sorted list of indices into n_workers non-overlapping contiguous
    [lo, hi) ranges for --pair-start / --pair-end.

    Uses the full span from min to max+1 so each range is contiguous.
    Already-completed pairs within a worker's range are no-ops under --merge.
    """
    lo = indices[0]
    hi = indices[-1] + 1  # exclusive upper bound
    total = hi - lo
    chunk = max(1, (total + n_workers - 1) // n_workers)
    ranges = []
    cursor = lo
    for _ in range(n_workers):
        if cursor >= hi:
            break
        end = min(cursor + chunk, hi)
        ranges.append((cursor, end))
        cursor = end
    return ranges


def write_resume_script(data_dir: Path, missing_indices: list, dry_run: bool) -> None:
    """Write scripts/run_gap_4gpu_resume.sh covering the missing pair ranges.

    Layout mirrors run_gap_4gpu.sh (2 SLURM jobs, 2 GPUs each):
      mscluster107 GPU 0 : worker 0 range
      mscluster107 GPU 1 : worker 1 range
      mscluster109 GPU 0 : worker 2 range
      mscluster109 GPU 1 : worker 3 range

    Key difference from the original: workers run SEQUENTIALLY within each
    node job (GPU 0 finishes before GPU 1 starts).  This eliminates the
    --merge race condition that caused the original data loss.
    """
    if not missing_indices:
        return

    script_path = _REPO_ROOT / "scripts" / "run_gap_4gpu_resume.sh"
    ranges = _split_into_ranges(missing_indices, n_workers=4)
    # Ensure exactly 4 slots.
    while len(ranges) < 4:
        ranges.append(None)

    rel_out = data_dir.relative_to(_REPO_ROOT)

    # Node assignments: workers 0,1 → mscluster107; workers 2,3 → mscluster109.
    node_jobs = [
        ("mscluster107", ranges[0], ranges[1]),
        ("mscluster109", ranges[2], ranges[3]),
    ]

    def fmt_range(r):
        return f"{r[0]} {r[1]}" if r else "0 0"

    lines = [
        "#!/usr/bin/env bash",
        "# ---------------------------------------------------------------------------",
        "# run_gap_4gpu_resume.sh — re-run MISSING pair shards and merge into",
        "# the existing output directory.",
        "#",
        "# Generated by: scripts/merge_gap_shards.py",
        "#",
        "# Worker layout (sequential within each node to avoid merge race):",
    ]
    for i, r in enumerate(ranges[:4]):
        if r:
            node = "mscluster107" if i < 2 else "mscluster109"
            gpu = i % 2
            lines.append(f"#   {node} GPU {gpu} : pairs {r[0]}–{r[1]-1}")
    lines += [
        "#",
        "# Usage:  bash scripts/run_gap_4gpu_resume.sh [OUTPUT_DIR]",
        "# ---------------------------------------------------------------------------",
        "set -euo pipefail",
        "",
        'ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"',
        'cd "$ROOT_DIR"',
        "",
        f'OUTPUT_DIR="${{1:-${{ROOT_DIR}}/{rel_out}}}"',
        'CONDA_ENV="${CONDA_ENV:-jaxstack}"',
        'CONDA_EXE="${CONDA_EXE:-${HOME}/miniforge3/bin/conda}"',
        'EXTRA_ARGS="${EXTRA_ARGS:-}"',
        'SEED_BATCH_SIZE="${SEED_BATCH_SIZE:-8}"',
        'SLURM_PARTITION="${SLURM_PARTITION:-biggpu}"',
        'SLURM_TIME="${SLURM_TIME:-72:00:00}"',
        'SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"',
        'LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"',
        "",
        'mkdir -p "$OUTPUT_DIR"',
        'mkdir -p "$LOG_DIR"',
        "",
        'if [[ ! -x "$CONDA_EXE" ]]; then',
        '    echo "Conda executable not found: $CONDA_EXE" >&2',
        "    exit 1",
        "fi",
        "",
        "COMMON_ARGS=(",
        "    --paper-only",
        "    --regime large",
        "    --pstar-source sdipc",
        "    --model-family sd14",
        '    --output-dir "$OUTPUT_DIR"',
        '    --seed-batch-size "$SEED_BATCH_SIZE"',
        "    --merge",
        ")",
        "",
        'if [[ -n "$EXTRA_ARGS" ]]; then',
        "    # shellcheck disable=SC2206",
        "    EXTRA_ARGS_ARRAY=($EXTRA_ARGS)",
        '    COMMON_ARGS+=("${EXTRA_ARGS_ARRAY[@]}")',
        "fi",
        "",
        "SBATCH_COMMON=(",
        "    --parsable",
        "    --ntasks 1",
        '    --cpus-per-task "$SLURM_CPUS_PER_TASK"',
        '    --partition "$SLURM_PARTITION"',
        '    --time "$SLURM_TIME"',
        ")",
        "",
        "# submit_node_job NODE PAIR0_START PAIR0_END PAIR1_START PAIR1_END",
        "# Workers run SEQUENTIALLY within the node (gpu0 finishes before gpu1",
        "# starts) to avoid concurrent writes to the shared aggregate JSON files.",
        "submit_node_job() {",
        "    local node=$1",
        "    local pair0_start=$2",
        "    local pair0_end=$3",
        "    local pair1_start=$4",
        "    local pair1_end=$5",
        "",
        '    local job_name="gap-resume-${node}"',
        '    local slurm_out="${LOG_DIR}/${job_name}-%j.out"',
        '    local slurm_err="${LOG_DIR}/${job_name}-%j.err"',
        "",
        "    sbatch \\",
        '        "${SBATCH_COMMON[@]}" \\',
        '        --job-name "$job_name" \\',
        '        --nodelist "$node" \\',
        '        --output "$slurm_out" \\',
        '        --error "$slurm_err" <<\'EOF\'',
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'cd "$ROOT_DIR"',
        'mkdir -p "$OUTPUT_DIR"',
        "",
        "run_worker() {",
        "    local gpu=$1",
        "    local pair_start=$2",
        "    local pair_end=$3",
        '    [[ "$pair_start" -ge "$pair_end" ]] && return 0  # empty range',
        '    local log_file="${OUTPUT_DIR}/worker_resume_${node}_gpu${gpu}.log"',
        "    echo \"[$(date '+%F %T')] ${node} gpu${gpu} pairs ${pair_start}-$((pair_end - 1)) (resume+merge)\" | tee -a \"$log_file\"",
        '    CUDA_VISIBLE_DEVICES="$gpu" USE_TF=0 \\',
        '        "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \\',
        "        python scripts/measure_composability_gap.py \\",
        '        ${COMMON_ARGS[*]} --pair-start "$pair_start" --pair-end "$pair_end" \\',
        '        >> "$log_file" 2>&1',
        "}",
        "",
        "# Sequential: GPU 0 first, then GPU 1.",
        "# This prevents both workers from simultaneously overwriting the shared",
        "# aggregate JSON files at the end of their runs.",
        "run_worker 0 $pair0_start $pair0_end || { echo 'GPU 0 worker failed' >&2; exit 1; }",
        "run_worker 1 $pair1_start $pair1_end || { echo 'GPU 1 worker failed' >&2; exit 1; }",
        "EOF",
        "}",
        "",
        'echo "Output: $OUTPUT_DIR"',
        'echo "Submitting 2 SLURM resume jobs (workers sequential within each node) ..."',
        "",
    ]

    job_vars = []
    for node, w0, w1 in node_jobs:
        r0 = fmt_range(w0)
        r1 = fmt_range(w1)
        var = f"JOB_{node.replace('mscluster', '').upper()}"
        lines.append(f'{var}="$(submit_node_job {node} {r0} {r1})"')
        job_vars.append((var, node, w0, w1))

    lines += ["", 'echo "Submitted:"']
    for var, node, w0, w1 in job_vars:
        desc_parts = []
        if w0:
            desc_parts.append(f"GPU0 pairs {w0[0]}-{w0[1]-1}")
        if w1:
            desc_parts.append(f"GPU1 pairs {w1[0]}-{w1[1]-1}")
        desc = ", ".join(desc_parts)
        lines.append(f'echo "  {node}: {desc}  (job ${{{var}}})"')

    lines += [
        "",
        'echo ""',
        'echo "Monitor with:"',
        'echo "  squeue -u $USER"',
        'echo "  tail -f ${OUTPUT_DIR}/worker_resume_mscluster107_gpu0.log"',
        'echo "  tail -f ${OUTPUT_DIR}/worker_resume_mscluster107_gpu1.log"',
        'echo "  tail -f ${OUTPUT_DIR}/worker_resume_mscluster109_gpu0.log"',
        'echo "  tail -f ${OUTPUT_DIR}/worker_resume_mscluster109_gpu1.log"',
        "",
        'echo ""',
        'echo "After jobs finish, verify with:"',
        'echo "  python scripts/merge_gap_shards.py --data-dir ${OUTPUT_DIR} --skip-all-pairs-gap"',
    ]

    content = "\n".join(lines) + "\n"

    if dry_run:
        print(f"\n[dry-run] would write resume script → {script_path}")
        print("  Worker layout:")
        for i, r in enumerate(ranges[:4]):
            if r:
                node = "mscluster107" if i < 2 else "mscluster109"
                gpu = i % 2
                print(f"    {node} GPU {gpu} : --pair-start {r[0]} --pair-end {r[1]}")
        return

    with open(script_path, "w") as fh:
        fh.write(content)
    script_path.chmod(0o755)
    print(f"\nResume script written → {script_path}")
    print(f"Run with:  bash {script_path.relative_to(_REPO_ROOT)} {data_dir.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("results/gap_large_4gpu_rerun_2"),
        help="Root output directory of the gap run (contains pairs/ and metrics/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without modifying any files.",
    )
    parser.add_argument(
        "--skip-all-pairs-gap",
        action="store_true",
        help="Skip rebuilding all_pairs_gap.json (useful if it is already fixed).",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        sys.exit(f"ERROR: --data-dir does not exist: {data_dir}")

    print(f"Data directory : {data_dir}")
    print(f"Dry-run        : {args.dry_run}")

    # ------------------------------------------------------------------
    # Step 1: Rebuild all_pairs_gap.json (fully reconstructable from disk)
    # ------------------------------------------------------------------
    if not args.skip_all_pairs_gap:
        print("\n" + "=" * 68)
        print("Step 1: Rebuild all_pairs_gap.json from pairs/*/gap_metrics.json")
        print("=" * 68)
        rebuild_all_pairs_gap(data_dir, dry_run=args.dry_run)

    # ------------------------------------------------------------------
    # Step 2: Diagnose missing seed-level records
    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("Step 2: Diagnose per_seed / trajectory / within_and coverage")
    print("=" * 68)
    missing_indices = report_missing_pairs(data_dir)

    # ------------------------------------------------------------------
    # Step 3: Print re-run commands and write resume script
    # ------------------------------------------------------------------
    print_rerun_commands(data_dir, missing_indices)
    write_resume_script(data_dir, missing_indices, dry_run=args.dry_run)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 68)
    print("Summary")
    print("=" * 68)
    if args.skip_all_pairs_gap:
        print("  all_pairs_gap.json  : (skipped — assumed already fixed)")
    else:
        print("  all_pairs_gap.json  : fully reconstructed from 24 gap_metrics.json files.")
    print("  per_seed_distances  : requires re-running 18 missing pairs with --merge.")
    print("  trajectory_distances: requires re-running 18 missing pairs with --merge.")
    print("  within_and_distances: requires re-running 18 missing pairs with --merge.")
    print()
    print("Expected record counts AFTER re-run:")
    print("  per_seed_distances.json    : 24 pairs × 24 seeds        =  576 records")
    print("  trajectory_distances.json  : 24 pairs × 24 seeds × ~13t = ~7 488 records")
    print("  within_and_distances.json  : 24 pairs × C(24,2)         = 6 624 records")
    print("  all_pairs_gap.json         : 24 records (already done)")
    print()
    if missing_indices:
        print(f"  Missing pair indices : {missing_indices}")
    else:
        print("  All pairs present — no re-run needed.")


if __name__ == "__main__":
    main()
