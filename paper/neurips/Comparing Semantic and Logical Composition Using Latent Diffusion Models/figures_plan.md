# Figure Plan — NeurIPS Paper

**Central claim:** the composability gap is structured across a six-group taxonomy rather than appearing as an unstructured failure pattern.

**Canonical taxonomy:** G1 co-occurrence, G2 factorization, G3 object-scene, G4 dual-object, G5 prior entanglement, G6 coherent collision.

**Interpretive overlay:** `easy = {G1, G2}`, `intermediate = {G3}`, `hard = {G4, G5, G6}`.

**Canonical source rule:** after freeze, all qualitative SDXL figures must be rendered from `experiments/eccv2026/sdxl_final/sdxl_six_group_seed42_steps50_cfg7p5_frozen`.

## Main Body

### Figure 1 — Six-Group Taxonomy Overview

- `scientific_claim`: the six groups differ qualitatively under a shared-noise protocol before any quantitative metric is introduced
- `source_artifacts`: seed-42 `grid_assets.json`, trajectory projections, and decoded images for the six representative pairs
- `renderer`: `scripts/render_taxonomy_paper_figure.py --mode figure1 --seed 42`
- `output_file`: `figures/trajectory_3x2_sdxl.png`
- `dependency_status`: `available_now`
- `placement`: `main_body`
- `narrative_role`: opens the paper’s scientific narrative and teaches the reader how to read the taxonomy

### Figure 2 — Representative Decoded Endpoints

- `scientific_claim`: the trajectory geometry corresponds to concrete endpoint behavior, not only latent-space separation
- `source_artifacts`: `solo_a.png`, `solo_b.png`, `monolithic.png`, `poe.png` for the six representative pairs at seed 42
- `renderer`: `scripts/render_taxonomy_paper_figure.py --mode figure1_endpoints --seed 42`
- `output_file`: `figures/representative_endpoints_3x2.png`
- `dependency_status`: `available_now`
- `placement`: `main_body`
- `narrative_role`: follows Figure 1 immediately so the reader sees geometry and decoded outputs back-to-back

### Figure 3 — Representative Seed Sheet

- `scientific_claim`: seed variation is part of the phenomenon; some groups are stable while others show systematic variability
- `source_artifacts`: the frozen final root across multiple seeds for the representative pairs
- `renderer`: `scripts/render_taxonomy_paper_figure.py --mode figure_seed_sheet --seeds 42 1 7 13 --seed-condition poe`
- `output_file`: `figures/representative_seed_sheet_poe.png`
- `dependency_status`: `available_now`
- `placement`: `main_body` if page budget allows, otherwise `appendix`
- `narrative_role`: establishes that representative examples are not single-seed accidents

### Figure 4a — BLIP-VQA Hybrid Summary

- `scientific_claim`: simple cue-presence can look superficially reasonable while decoded outputs reveal why naive marginal prompting is insufficient
- `source_artifacts`: groupwise `blip_vqa_scores.json` plus one explicit decoded endpoint strip per group
- `renderer`: `scripts/render_group_probe_endpoint_hybrid.py --metric blip_vqa`
- `output_file`: `figures/blip_vqa_endpoint_hybrid_3x2.png`
- `dependency_status`: `requires_additional_eval`
- `placement`: `main_body` or `appendix`
- `narrative_role`: teaches why BLIP-VQA alone cannot be the main gap metric

### Figure 4b — Joint-Probe Hybrid Summary

- `scientific_claim`: pair-type-aware probing aligns better with the decoded failure patterns than naive cue-presence bars
- `source_artifacts`: groupwise `joint_probe_scores.json` plus the same explicit decoded endpoint strips used in Figure 4a
- `renderer`: `scripts/render_group_probe_endpoint_hybrid.py --metric joint_probe`
- `output_file`: `figures/joint_probe_endpoint_hybrid_3x2.png`
- `dependency_status`: `requires_additional_eval`
- `placement`: `main_body`
- `narrative_role`: provides the corrected output-level bridge between groupwise metrics and qualitative examples

### Figure 4c — Single-Endpoint Probe Explainer

- `scientific_claim`: the joint-correctness score comes from structured sub-questions rather than one naive VQA prompt
- `source_artifacts`: one decoded endpoint and its structured probe inspection panel
- `renderer`: `scripts/render_probe_explainer_panel.py`
- `output_file`: `figures/monolithic_probe_explainer.png`
- `dependency_status`: `available_now`
- `placement`: `main_body` or `appendix`
- `narrative_role`: explains how the joint-probe score is constructed on one concrete image

### Figure 5 — Groupwise Quantitative Summary

- `scientific_claim`: the visual taxonomy aligns with measured behavior across all selected pairs
- `source_artifacts`: six-group evaluation products from the frozen final root
- `preferred_metric`: pair-type-aware joint correctness
- `secondary_metrics`: terminal gap, within-AND noise floor, temporal dynamics
- `output_file`: project-dependent
- `dependency_status`: `requires_additional_eval`
- `placement`: `main_body`
- `narrative_role`: first hard quantitative confirmation after the reader understands the qualitative taxonomy

### Figure 6 — Reachability / Recovery Analysis

- `scientific_claim`: some targets are language-recoverable, while harder groups remain unreachable
- `source_artifacts`: p* reruns plus a distributional summary from the six-group frozen final root
- `output_file`: project-dependent
- `dependency_status`: `requires_additional_eval`
- `placement`: `late_main_body`
- `narrative_role`: closes the paper after the failure landscape is already established

## Appendix

Appendix figures should absorb:

- full representative contact sheets beyond the seed-42 headline figure
- extra seed galleries per group
- any probe-endpoint hybrid variants too dense for the main body
- alternate density or distribution views once the main-body figure already makes the claim
- legacy subgroup material that no longer matches the canonical six-group story

Legacy Group 3 subgroup panels are appendix-only by default unless they are explicitly reframed as historical or diagnostic material.

## Wave Status

### Wave 1 — Available Now

- `trajectory_3x2_sdxl.png`
- `representative_endpoints_3x2.png`
- `representative_seed_sheet_poe.png`
- `monolithic_probe_explainer.png`

### Wave 2 — Requires Additional Evaluation

Blocked until the corresponding artifacts exist:

- `blip_vqa_scores.json`
- `joint_probe_scores.json`
- p* rerun images or metrics
- six-group groupwise quantitative exports
- `blip_vqa_endpoint_hybrid_3x2.png`
- `joint_probe_endpoint_hybrid_3x2.png`

Do not describe Wave 2 figures as already renderable from the current frozen final root.

## Default Choices

- headline qualitative seed: `42`
- representative pair set: one canonical representative pair per taxonomy group from `scripts/taxonomy_manifest.py`
- default seed-sheet seeds: `42 1 7 13`
- default seed-sheet condition: `poe`
- all figures preserve the canonical group order `G1 → G6`
- hybrid probe-endpoint figures require an explicit six-pair list in canonical group order
