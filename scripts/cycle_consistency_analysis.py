"""
Cycle Consistency Analysis: does SD-IPC close the loop?

Experiment A — Sanity check (should succeed):
  SD 1.4 "a cat lying on the bed"
      → SD-IPC projection → text embedding
      → SD 1.4 regeneration
  Expected: cycle closes visually (high CLIP similarity)

Experiment B — Composability gap probe (expected to fail):
  SuperDiff AND "a cat" ^ "a dog"  (hybrid/chimera image)
      → SD-IPC projection → text embedding
      → SD 1.4 regeneration
  Expected: cycle BREAKS — SD regeneration escapes the hybrid,
            lands on whichever concept dominates in CLIP space.

Experiment C — Monolithic baseline (should succeed):
  SD 1.4 "a cat and a dog"
      → SD-IPC projection → text embedding
      → SD 1.4 regeneration
  Expected: cycle closes (composition is on-manifold for SD)

The cycle-consistency gap between B and (A, C) is the composability gap.

Usage:
    conda activate attend_excite
    python scripts/cycle_consistency_analysis.py
"""
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "CompVis/stable-diffusion-v1-4"
CLIP_ID  = "openai/clip-vit-large-patch14"
OUT_DIR  = Path("experiments/composability_gap")
SEED     = 42
STEPS    = 50
SCALE    = 7.5

EXPERIMENTS = {
    "sanity"    : "a cat lying on the bed",
    "mono"      : "a cat and a dog",
    # AND hybrid is loaded from the previously generated file
}
AND_IMG_PATH = OUT_DIR / "superdiff_and_cat_dog.png"

OUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Load SD v1.4
# ---------------------------------------------------------------------------
print("Loading SD v1.4 (fp16) ...")
vae          = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae",         use_safetensors=True, torch_dtype=torch.float16).to(device)
tokenizer    = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
unet         = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", use_safetensors=True, torch_dtype=torch.float16).to(device)
scheduler    = DPMSolverMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
vae.eval(); text_encoder.eval(); unet.eval()

# Standalone CLIP ViT-L/14 for image encoding — on CPU during generation
print("Loading CLIP ViT-L/14 (CPU) ...")
clip_model     = CLIPModel.from_pretrained(CLIP_ID)
clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)
clip_model.eval()

# SD-IPC closed-form projection matrices
with torch.no_grad():
    inv_text    = torch.linalg.pinv(clip_model.text_projection.weight.float(), atol=0.3)
    visual_proj = clip_model.visual_projection.weight.float()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def sd_text_emb(prompt: str) -> torch.Tensor:
    """(1, 77, 768) SD cross-attention conditioning."""
    ids = tokenizer(prompt, padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").input_ids.to(device)
    return text_encoder(ids)[0]


@torch.no_grad()
def sdipc_project(image: Image.Image) -> torch.Tensor:
    """
    SD-IPC: "The CLIP Model is Secretly an Image-to-Prompt Converter"
    Ding et al., NeurIPS 2023, arXiv:2305.12716. Closed-form projection:
    image → CLIP pooler → visual_proj → pinv(text_proj) → L2-norm → ×27.5
    Returns (1, 768) in SD text-encoder hidden-state space.
    """
    clip_model.to(device)
    pv         = clip_processor(images=image, return_tensors="pt").pixel_values.to(device)
    pooler_out = clip_model.vision_model(pixel_values=pv).pooler_output.float()  # (1, 1024)
    joint      = pooler_out @ visual_proj.to(device).T   # (1, 768) — CLIP joint space
    text_space = joint @ inv_text.to(device).T           # (1, 768) — text encoder space
    text_space = text_space / text_space.norm(dim=-1, keepdim=True)
    return 27.5 * text_space                             # scale to match EOS token magnitude


def sdipc_to_seq(proj_vec: torch.Tensor) -> torch.Tensor:
    """
    Build a full (1, 77, 768) cross-attention sequence from a single SD-IPC vector.
    Faithful to SD-IPC inference_sd_ipc.py:
      seq[:, 0]  = null_text_embeddings[:, 0]   (keep BOS token)
      seq[:, 1:] = image_emb_proj               (fill all other positions)
    """
    null_seq = sd_text_emb("")                       # (1, 77, 768) fp16
    seq = torch.zeros_like(null_seq)
    seq[:, 0] = null_seq[:, 0]                      # keep BOS token from null prompt
    pv = proj_vec.to(dtype=seq.dtype, device=seq.device)
    seq[:, 1:] = pv.unsqueeze(1)                    # fill positions 1-76 with projection
    return seq


@torch.no_grad()
def clip_image_emb(image: Image.Image) -> torch.Tensor:
    """L2-normalised (1, D) CLIP joint-space image embedding."""
    clip_model.to(device)
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    e = clip_model.get_image_features(**inputs)
    if not isinstance(e, torch.Tensor):
        e = e.pooler_output if hasattr(e, "pooler_output") else e[0][:, 0]
    e = e.float()
    return e / e.norm(dim=-1, keepdim=True)


@torch.no_grad()
def clip_text_emb(prompt: str) -> torch.Tensor:
    """L2-normalised (1, D) CLIP joint-space text embedding."""
    clip_model.to(device)
    inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    e = clip_model.get_text_features(**inputs)
    if not isinstance(e, torch.Tensor):
        e = e.pooler_output if hasattr(e, "pooler_output") else e[0][:, 0]
    e = e.float()
    return e / e.norm(dim=-1, keepdim=True)


@torch.no_grad()
def decode(latents: torch.Tensor) -> Image.Image:
    x = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    x = (x / 2 + 0.5).clamp(0, 1)
    x = (x.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()[0]
    return Image.fromarray(x)


@torch.no_grad()
def run_sd(cond_emb: torch.Tensor, seed: int = SEED) -> Image.Image:
    """Standard CFG denoising given a (1, 77, 768) conditioning sequence."""
    uncond = sd_text_emb("")
    cond_emb = cond_emb.to(device=device, dtype=torch.float16)
    uncond   = uncond.to(device=device, dtype=torch.float16)

    scheduler.set_timesteps(STEPS)
    gen     = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn((1, unet.config.in_channels, 64, 64),
                          generator=gen, device=device, dtype=torch.float16)
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        inp        = scheduler.scale_model_input(latents, t)
        noise_c    = unet(inp, t, encoder_hidden_states=cond_emb).sample
        noise_u    = unet(inp, t, encoder_hidden_states=uncond   ).sample
        noise_pred = noise_u + SCALE * (noise_c - noise_u)
        latents    = scheduler.step(noise_pred, t, latents).prev_sample

    return decode(latents)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float(); b = b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))


# ---------------------------------------------------------------------------
# Taxonomy groups — mirrored from trajectory_dynamics_experiment.py
# (copied to avoid importing that file and triggering its model loading)
# ---------------------------------------------------------------------------
_TAXONOMY_GROUPS: dict[str, dict] = {
    "group1_cooccurrence": {
        "label": "G1 – Co-occurrence",
        "pairs": [
            ("a camel",             "a desert landscape"),
            ("a butterfly",         "a flower meadow"),
            ("a dolphin",           "an ocean wave"),
            ("a lion",              "a savanna at sunset"),
        ],
    },
    "group2_disentangled": {
        "label": "G2 – Disentangled",
        "pairs": [
            ("a dog",               "oil painting style"),
            ("a lighthouse",        "watercolour style"),
            ("a bicycle",           "sketch style"),
        ],
    },
    "group3_ood": {
        "label": "G3 – OOD",
        "pairs": [
            ("a desk lamp",         "a glacier"),
            ("a bathtub",           "a streetlamp"),
            ("a lab microscope",    "a hay bale"),
            ("a black grand piano", "a white vase"),
            ("a typewriter",        "a cactus"),
        ],
    },
    "group4_collision": {
        "label": "G4 – Collision",
        "pairs": [
            ("a cat",   "a dog"),
            ("a cat",   "an owl"),
            ("a cat",   "a bear"),
            ("a tiger", "a lion"),
        ],
    },
}
_GROUP_ORDER = ["group1_cooccurrence", "group2_disentangled", "group3_ood", "group4_collision"]
_GROUP_COLORS = {
    "group1_cooccurrence": "#4878CF",
    "group2_disentangled": "#6ACC65",
    "group3_ood":          "#F5A623",
    "group4_collision":    "#D7191C",
}
_CYCLE_BREAK_THRESHOLD = 0.80
_TAX_DATA_ROOT = Path("experiments/eccv2026/taxonomy_qualitative")


# ---------------------------------------------------------------------------
# Taxonomy batch mode — helpers
# ---------------------------------------------------------------------------

def _find_condition_image(pair_dir: Path, condition: str) -> Image.Image | None:
    """Load a condition PNG from a pair directory (handles single-seed and multi-seed layouts)."""
    # Direct path: pair_dir/condition.png
    direct = pair_dir / f"{condition}.png"
    if direct.exists():
        return Image.open(direct).convert("RGB")
    # Multi-seed: pair_dir/seed_NNN/condition.png — pick first seed
    seed_dirs = sorted(d for d in pair_dir.iterdir() if d.is_dir() and d.name.startswith("seed_"))
    for sd in seed_dirs:
        p = sd / f"{condition}.png"
        if p.exists():
            return Image.open(p).convert("RGB")
    return None


def _pair_dir_for(data_root: Path, group_key: str, prompt_a: str, prompt_b: str) -> Path:
    slug = f"{prompt_a}__x__{prompt_b}".replace(" ", "_")[:60]
    return data_root / group_key / slug


def run_taxonomy_cycle_test(
    data_root: Path,
    out_dir: Path,
    seeds: list[int],
) -> None:
    """
    E6 — Cycle consistency batch test over all taxonomy pairs.

    For each pair (per group) per seed:
      - Load PoE image (condition 'poe') and monolithic image (condition 'monolithic')
        from taxonomy_qualitative/ (generated by --taxonomy-grid).
      - Run SD-IPC → text embedding → SD regeneration.
      - Compute CLIP sim(source, regen).
      - Flag cycle broken if sim < CYCLE_BREAK_THRESHOLD.

    Outputs:
      cycle_consistency_taxonomy.json  — raw per-pair results
      cycle_consistency_taxonomy.png   — grouped bar chart (break rate PoE vs mono per group)
    """
    import json as _json

    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    text_encoder.to(device); unet.to(device); vae.to(device)
    clip_model.cpu()
    torch.cuda.empty_cache()

    for gkey in _GROUP_ORDER:
        ginfo = _TAXONOMY_GROUPS[gkey]
        pairs = ginfo["pairs"]
        print(f"\n{'='*55}")
        print(f"  {ginfo['label']}  ({len(pairs)} pairs)")
        print(f"{'='*55}")

        for prompt_a, prompt_b in pairs:
            pair_dir = _pair_dir_for(data_root, gkey, prompt_a, prompt_b)
            if not pair_dir.exists():
                print(f"  SKIP (no dir): {pair_dir.name}")
                continue

            for seed in seeds:
                # Try to load seed-specific image first, then fall back to pair root
                seed_dir = pair_dir / f"seed_{seed:03d}"
                search_dir = seed_dir if seed_dir.exists() else pair_dir

                for cond_name in ("poe", "monolithic"):
                    img = _find_condition_image(search_dir, cond_name)
                    if img is None:
                        img = _find_condition_image(pair_dir, cond_name)
                    if img is None:
                        print(f"  SKIP {cond_name} ({prompt_a} × {prompt_b}, seed {seed}): image not found")
                        continue

                    print(f"  [{gkey}] {prompt_a} × {prompt_b}  cond={cond_name}  seed={seed}")

                    # SD-IPC projection
                    clip_model.to(device)
                    proj = sdipc_project(img)          # (1, 768)
                    cond_seq = sdipc_to_seq(proj)       # (1, 77, 768)
                    clip_model.cpu()
                    torch.cuda.empty_cache()

                    # SD regeneration
                    regen = run_sd(cond_seq, seed=seed)

                    # CLIP similarity
                    clip_model.to(device)
                    src_emb   = clip_image_emb(img)
                    regen_emb = clip_image_emb(regen)
                    clip_model.cpu()

                    sim = cosine_sim(src_emb, regen_emb)
                    broken = sim < _CYCLE_BREAK_THRESHOLD

                    monolithic_prompt = f"{prompt_a} and {prompt_b}"
                    if cond_name == "poe":
                        prompt_label = f"PoE: \"{prompt_a}\" × \"{prompt_b}\""
                    else:
                        prompt_label = f"Monolithic: \"{monolithic_prompt}\""

                    print(f"    sim={sim:.4f}  {'✗ BROKEN' if broken else '✓ closed'}")

                    # Save regen
                    regen_fname = out_dir / f"regen_{gkey}_{prompt_a}__x__{prompt_b}_{cond_name}_s{seed}.png".replace(" ", "_")
                    regen.save(regen_fname)

                    records.append({
                        "taxonomy_group": gkey,
                        "prompt_a":       prompt_a,
                        "prompt_b":       prompt_b,
                        "condition":      cond_name,
                        "seed":           seed,
                        "sim_cycle":      float(sim),
                        "cycle_broken":   bool(broken),
                        "prompt_label":   prompt_label,
                    })

    # Save raw results
    json_out = out_dir / "cycle_consistency_taxonomy.json"
    with open(json_out, "w") as f:
        _json.dump(records, f, indent=2)
    print(f"\nRaw results → {json_out}")

    # Plot: grouped bar chart of cycle-break rate per group × condition
    _plot_taxonomy_cycle_bars(records, out_dir)


def _plot_taxonomy_cycle_bars(records: list[dict], out_dir: Path) -> None:
    """2-panel figure: break rate bars + cycle similarity KDE per group."""
    from collections import defaultdict

    # Aggregate by (group, condition)
    break_counts: dict[tuple, list[bool]] = defaultdict(list)
    sim_vals:     dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        key = (r["taxonomy_group"], r["condition"])
        break_counts[key].append(r["cycle_broken"])
        sim_vals[key].append(r["sim_cycle"])

    groups = [g for g in _GROUP_ORDER if any(g == k[0] for k in break_counts)]
    if not groups:
        print("No results to plot.")
        return

    conditions = ["poe", "monolithic"]
    cond_labels = {"poe": "PoE/AND", "monolithic": "Monolithic"}
    cond_hatch  = {"poe": "//", "monolithic": ""}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Cycle Consistency: SD-IPC breaks PoE off-manifold images more than Monolithic",
                 fontsize=11, y=1.02)

    # Left: break rate bars
    ax = axes[0]
    x = np.arange(len(groups))
    width = 0.35
    for i, cond in enumerate(conditions):
        rates = []
        errs  = []
        for g in groups:
            bools = break_counts.get((g, cond), [])
            if bools:
                rate = np.mean(bools)
                # Wilson CI approximate error
                n = len(bools)
                err = np.sqrt(rate * (1 - rate) / n) if n > 0 else 0.0
            else:
                rate, err = 0.0, 0.0
            rates.append(rate)
            errs.append(err)
        offset = (i - 0.5) * width
        color = _GROUP_COLORS.get(groups[0], "steelblue") if len(groups) == 1 else "#4878CF"
        # Use condition-specific neutral colour
        cond_color = "#D7191C" if cond == "poe" else "#6ACC65"
        ax.bar(x + offset, rates, width, yerr=errs, capsize=5,
               color=cond_color, alpha=0.80, hatch=cond_hatch[cond],
               edgecolor="white", label=cond_labels[cond],
               error_kw={"elinewidth": 1.2, "capthick": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_TAXONOMY_GROUPS[g]["label"] for g in groups], fontsize=8, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(_CYCLE_BREAK_THRESHOLD, linestyle="--", color="grey",
               linewidth=1, label=f"threshold ({_CYCLE_BREAK_THRESHOLD})")
    ax.set_ylabel("Cycle-break rate  (sim < 0.80)", fontsize=10)
    ax.set_title("Cycle-break rate per group × condition", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    # Right: KDE of cycle similarity
    ax2 = axes[1]
    from scipy.stats import gaussian_kde as _kde
    all_sims = [v for vv in sim_vals.values() for v in vv]
    x_min = max(0, min(all_sims) - 0.05)
    x_max = min(1, max(all_sims) + 0.05)
    xg = np.linspace(x_min, x_max, 300)

    linestyles = {"poe": "-", "monolithic": "--"}
    for cond in conditions:
        for g in groups:
            vals = np.array(sim_vals.get((g, cond), []))
            if len(vals) < 2:
                continue
            color = _GROUP_COLORS.get(g, "grey")
            ls    = linestyles[cond]
            density = _kde(vals, bw_method="scott")(xg)
            ax2.plot(xg, density, color=color, linewidth=1.5, linestyle=ls,
                     label=f"{_TAXONOMY_GROUPS[g]['label']} / {cond_labels[cond]}")
            ax2.fill_between(xg, density, alpha=0.06, color=color)

    ax2.axvline(_CYCLE_BREAK_THRESHOLD, linestyle="--", color="grey",
                linewidth=1, label=f"threshold ({_CYCLE_BREAK_THRESHOLD})")
    ax2.set_xlabel("CLIP sim(source, regen)", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.set_title("Cycle similarity KDE per group × condition", fontsize=10)
    ax2.legend(fontsize=6.5, framealpha=0.8, loc="upper left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax2.set_xlim(x_min, x_max)

    fig.tight_layout()
    out_path = out_dir / "cycle_consistency_taxonomy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)

    # Copy to proposal media dir
    media_dir = (
        Path(__file__).resolve().parent.parent
        / "proposal" / "proposal_stage_3"
        / "chapters" / "research_method" / "media"
    )
    if media_dir.exists():
        dest = media_dir / "cycle_consistency_multiseed.png"
        import shutil
        shutil.copy(out_path, dest)
        print(f"Copied → {dest}")


# ---------------------------------------------------------------------------
# Argparse — dispatch taxonomy-pairs mode before original flow
# ---------------------------------------------------------------------------
import argparse as _argparse
_ap = _argparse.ArgumentParser(
    description="Cycle consistency analysis (SD-IPC).  Default: original cat/dog experiments.  "
                "Use --taxonomy-pairs for Phase 1 batch cycle test.",
    add_help=True,
)
_ap.add_argument("--taxonomy-pairs", action="store_true",
                 help="Run cycle consistency test across all taxonomy groups (W4 / E6).")
_ap.add_argument("--tax-out-dir", type=str, default="",
                 help="Output directory for --taxonomy-pairs mode. "
                      "Default: experiments/eccv2026/cycle_consistency/")
_ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                 help="Seeds to test per pair (default: 42). E.g. --seeds 42 43 44")
_ap.add_argument("--groups", type=str, nargs="+", default=_GROUP_ORDER,
                 choices=_GROUP_ORDER,
                 help="Subset of taxonomy groups (default: all four).")
_args, _remaining = _ap.parse_known_args()

if _args.taxonomy_pairs:
    # Filter groups
    for gkey in list(_TAXONOMY_GROUPS.keys()):
        if gkey not in _args.groups:
            del _TAXONOMY_GROUPS[gkey]
    _GROUP_ORDER[:] = [g for g in _GROUP_ORDER if g in _TAXONOMY_GROUPS]

    _tax_out = (
        Path(_args.tax_out_dir)
        if _args.tax_out_dir
        else Path("experiments/eccv2026/cycle_consistency")
    )
    run_taxonomy_cycle_test(
        data_root=_TAX_DATA_ROOT,
        out_dir=_tax_out,
        seeds=_args.seeds,
    )
    import sys as _sys; _sys.exit(0)


# ---------------------------------------------------------------------------
# Step 1: Generate source images
# ---------------------------------------------------------------------------
clip_model.cpu()
torch.cuda.empty_cache()
print("\n=== Generating source images ===")

sources = {}

for name, prompt in EXPERIMENTS.items():
    print(f"  SD 1.4: '{prompt}'")
    emb  = sd_text_emb(prompt)
    img  = run_sd(emb, seed=SEED)
    sources[name] = {"prompt": prompt, "image": img}
    img.save(OUT_DIR / f"cycle_{name}_source.png")
    print(f"  Saved → cycle_{name}_source.png")

# Load pre-generated AND hybrid
print(f"  Loading AND hybrid from {AND_IMG_PATH}")
sources["and"] = {"prompt": "SuperDiff AND: 'a cat' ^ 'a dog'",
                  "image": Image.open(AND_IMG_PATH)}

# ---------------------------------------------------------------------------
# Step 2: SD-IPC projection → regenerate
# ---------------------------------------------------------------------------
print("\n=== SD-IPC projection + SD regeneration ===")

results = {}
for name, data in sources.items():
    print(f"\n  [{name}] {data['prompt']}")

    # Project image → text embedding space
    proj = sdipc_project(data["image"])          # (1, 768)
    cond = sdipc_to_seq(proj)                    # (1, 77, 768)

    # Move CLIP back to CPU, SD models are already on GPU
    clip_model.cpu()
    torch.cuda.empty_cache()

    # Regenerate from the projected embedding
    regen = run_sd(cond, seed=SEED)
    regen.save(OUT_DIR / f"cycle_{name}_regen.png")
    print(f"  Saved → cycle_{name}_regen.png")

    results[name] = {
        "prompt"  : data["prompt"],
        "source"  : data["image"],
        "regen"   : regen,
        "proj_emb": proj.cpu(),
    }

# ---------------------------------------------------------------------------
# Step 3: CLIP similarity measurements
# ---------------------------------------------------------------------------
print("\n=== CLIP similarity ===")
unet.cpu(); vae.cpu(); text_encoder.cpu()
torch.cuda.empty_cache()

for name, data in results.items():
    src_emb   = clip_image_emb(data["source"])
    regen_emb = clip_image_emb(data["regen"])
    sim_cycle = cosine_sim(src_emb, regen_emb)

    print(f"\n  [{name}] {data['prompt']}")
    print(f"    Source ↔ Regen  (cycle sim):  {sim_cycle:.4f}  {'✓ closed' if sim_cycle > 0.80 else '✗ broken'}")

    # Also compare regen to text prompts for AND experiment
    if name == "and":
        sim_cat = cosine_sim(regen_emb, clip_text_emb("a cat"))
        sim_dog = cosine_sim(regen_emb, clip_text_emb("a dog"))
        sim_cat_dog = cosine_sim(regen_emb, clip_text_emb("a cat and a dog"))
        print(f"    Regen → 'a cat' text:         {sim_cat:.4f}")
        print(f"    Regen → 'a dog' text:          {sim_dog:.4f}")
        print(f"    Regen → 'a cat and a dog' text:{sim_cat_dog:.4f}")
        print(f"    (AND source → 'a cat' text:    "
              f"{cosine_sim(src_emb, clip_text_emb('a cat')):.4f})")
        print(f"    (AND source → 'a dog' text:    "
              f"{cosine_sim(src_emb, clip_text_emb('a dog')):.4f})")

    results[name]["sim_cycle"] = sim_cycle
    results[name]["src_emb"]   = src_emb
    results[name]["regen_emb"] = regen_emb

# ---------------------------------------------------------------------------
# Step 4: Plot
# ---------------------------------------------------------------------------
print("\n=== Plotting ===")

order  = ["sanity", "mono", "and"]
titles = {
    "sanity": "Experiment A — SD sanity check\n(should cycle-close)",
    "mono"  : "Experiment B — SD monolithic\n(should cycle-close)",
    "and"   : "Experiment C — SuperDiff AND hybrid\n(expected to break)",
}

fig, axes = plt.subplots(3, 3, figsize=(13, 13))

for row, name in enumerate(order):
    data = results[name]
    sim  = data["sim_cycle"]

    # Column 0: source image
    axes[row, 0].imshow(np.array(data["source"].resize((256, 256))))
    axes[row, 0].set_title("Source image", fontsize=9)
    axes[row, 0].axis("off")

    # Column 1: arrow + sim score
    ax = axes[row, 1]
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.annotate("", xy=(0.9, 0.5), xytext=(0.1, 0.5),
                arrowprops=dict(arrowstyle="->", lw=2.5,
                                color="green" if sim > 0.80 else "red"))
    ax.text(0.5, 0.65, "SD-IPC →", ha="center", fontsize=9, color="grey")
    ax.text(0.5, 0.38, f"CLIP sim = {sim:.3f}", ha="center", fontsize=11,
            fontweight="bold", color="green" if sim > 0.80 else "red")
    status = "cycle closed ✓" if sim > 0.80 else "cycle broken ✗"
    ax.text(0.5, 0.22, status, ha="center", fontsize=9,
            color="green" if sim > 0.80 else "red")

    # Column 2: regenerated image
    axes[row, 2].imshow(np.array(data["regen"].resize((256, 256))))
    axes[row, 2].set_title("SD regen from SD-IPC embedding", fontsize=9)
    axes[row, 2].axis("off")

    # Row label
    axes[row, 0].set_ylabel(titles[name], fontsize=8, labelpad=6)

plt.suptitle(
    "Cycle Consistency Analysis\n"
    "Image → SD-IPC text embedding → SD 1.4 regeneration\n"
    "Monolithic SD is cycle-consistent; AND hybrid is not — the composability gap",
    fontsize=10, y=1.01
)
plt.tight_layout()
out = OUT_DIR / "cycle_consistency.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")

# ---------------------------------------------------------------------------
# Step 5: Embedding space plot — all source + regen embeddings together
# ---------------------------------------------------------------------------
from sklearn.decomposition import PCA

all_embs   = []
all_labels = []
all_colors = []
all_markers = []

palette = {
    "sanity_src" : ("#1565C0", "o"),   # blue circle
    "sanity_regen": ("#42A5F5", "o"),  # light blue circle
    "mono_src"   : ("#2E7D32", "s"),   # green square
    "mono_regen" : ("#81C784", "s"),   # light green square
    "and_src"    : ("#B71C1C", "^"),   # red triangle
    "and_regen"  : ("#EF9A9A", "^"),   # light red triangle
}

label_map = {
    "sanity_src"  : "Sanity source",
    "sanity_regen": "Sanity regen",
    "mono_src"    : "Mono source",
    "mono_regen"  : "Mono regen",
    "and_src"     : "AND source",
    "and_regen"   : "AND regen",
}

for name in order:
    d = results[name]
    all_embs.append(d["src_emb"].cpu().numpy())
    all_labels.append(f"{name}_src")
    all_embs.append(d["regen_emb"].cpu().numpy())
    all_labels.append(f"{name}_regen")

all_embs = np.vstack(all_embs)
pca    = PCA(n_components=2)
coords = pca.fit_transform(all_embs)

fig2, ax = plt.subplots(figsize=(8, 7))
for k, label in enumerate(all_labels):
    color, marker = palette[label]
    ax.scatter(coords[k, 0], coords[k, 1], c=color, marker=marker,
               s=220, edgecolors="black", linewidths=0.7, zorder=5)
    ax.annotate(label_map[label], (coords[k, 0], coords[k, 1]),
                textcoords="offset points", xytext=(7, 4), fontsize=8.5)

# Draw lines connecting source → regen for each experiment
for i in range(0, len(all_labels), 2):
    c = palette[all_labels[i]][0]
    ax.plot([coords[i, 0], coords[i+1, 0]],
            [coords[i, 1], coords[i+1, 1]],
            color=c, lw=1.5, linestyle="--", alpha=0.7)

ax.set_title("CLIP Embedding Space — Source vs Regen (PCA 2D)\n"
             "Short dashed lines = small cycle gap; long = cycle broken",
             fontsize=10, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
out2 = OUT_DIR / "cycle_consistency_pca.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved → {out2}")

# ---------------------------------------------------------------------------
# Step 6: Multi-seed regen grid  (5 seeds × 3 sources)
# ---------------------------------------------------------------------------
print("\n=== Multi-seed regen grid ===")
MULTI_SEEDS = [42, 43, 44, 45, 46]

# SD models were offloaded to CPU in Step 3 — bring them back for generation
clip_model.cpu()
text_encoder.to(device); unet.to(device); vae.to(device)
torch.cuda.empty_cache()

# Pre-compute SD-IPC conditionings once per source (independent of seed)
source_conds = {}
for name in order:
    proj = results[name]["proj_emb"].to(device=device)
    source_conds[name] = sdipc_to_seq(proj)   # (1, 77, 768)

# Generate regens for every (seed, source) pair
# Seed 42 already done — reuse from results
multi_regens = {}   # (name, seed) -> {"regen": Image, "regen_emb": tensor, "sim": float}

for seed in MULTI_SEEDS:
    clip_model.cpu()
    torch.cuda.empty_cache()
    for name in order:
        if seed == SEED:
            # Reuse already-computed result
            multi_regens[(name, seed)] = {
                "regen"    : results[name]["regen"],
                "regen_emb": results[name]["regen_emb"],
                "sim"      : results[name]["sim_cycle"],
            }
        else:
            regen     = run_sd(source_conds[name], seed=seed)
            clip_model.to(device)
            regen_emb = clip_image_emb(regen)
            sim       = cosine_sim(results[name]["src_emb"], regen_emb)
            multi_regens[(name, seed)] = {
                "regen"    : regen,
                "regen_emb": regen_emb,
                "sim"      : sim,
            }
            regen.save(OUT_DIR / f"cycle_{name}_regen_s{seed}.png")
    print(f"  Done seed {seed}")

# Plot 5 (rows=seeds) × 3 (cols=sources) grid of regen images
col_headers = {
    "sanity": "A · Sanity\n\"a cat on the bed\"",
    "mono"  : "B · Monolithic\n\"a cat and a dog\"",
    "and"   : "C · AND hybrid\ncat ^ dog",
}

fig3, axes3 = plt.subplots(len(MULTI_SEEDS), 3,
                            figsize=(10, 4 * len(MULTI_SEEDS)),
                            gridspec_kw={"hspace": 0.35, "wspace": 0.05})

for row, seed in enumerate(MULTI_SEEDS):
    for col, name in enumerate(order):
        data = multi_regens[(name, seed)]
        sim  = data["sim"]
        ax   = axes3[row, col]
        ax.imshow(np.array(data["regen"].resize((256, 256))))
        ax.axis("off")

        color = "green" if sim > 0.80 else "red"
        title = (col_headers[name] + f"\nseed {seed}" if row == 0
                 else f"seed {seed}")
        ax.set_title(f"{title}\nCLIP sim = {sim:.3f}",
                     fontsize=8, color=color, fontweight="bold")

plt.suptitle(
    "Multi-seed Regen Grid  (SD-IPC → SD 1.4)\n"
    "Green title = cycle closed (sim > 0.80)   Red = cycle broken",
    fontsize=11, y=1.01, fontweight="bold"
)
out3 = OUT_DIR / "cycle_consistency_multiseed.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved → {out3}")

# ---------------------------------------------------------------------------
# Step 7: Multi-seed PCA — sources (fixed) + all regen embeddings
# PCA is on CLIP image embeddings (768-dim), NOT on latents.
# ---------------------------------------------------------------------------
src_colors  = {"sanity": "#1565C0", "mono": "#2E7D32", "and": "#B71C1C"}
regen_colors = {"sanity": "#42A5F5", "mono": "#81C784", "and": "#EF9A9A"}
markers      = {"sanity": "o", "mono": "s", "and": "^"}

ms_embs   = []
ms_labels = []   # (name, "src"|"regen", seed)

# Add the 3 source embeddings first
for name in order:
    ms_embs.append(results[name]["src_emb"].cpu().numpy())
    ms_labels.append((name, "src", None))

# Add all regen embeddings (15 = 5 seeds × 3 sources)
for seed in MULTI_SEEDS:
    for name in order:
        ms_embs.append(multi_regens[(name, seed)]["regen_emb"].cpu().numpy())
        ms_labels.append((name, "regen", seed))

ms_embs  = np.vstack(ms_embs)
pca2     = PCA(n_components=2)
coords2  = pca2.fit_transform(ms_embs)

fig4, ax4 = plt.subplots(figsize=(10, 8))

for k, (name, kind, seed) in enumerate(ms_labels):
    if kind == "src":
        c = src_colors[name]; m = markers[name]; sz = 350; lw = 1.5; zorder = 8
        label = f"{name} source"
    else:
        c = regen_colors[name]; m = markers[name]; sz = 120; lw = 0.5; zorder = 4
        label = None  # avoid legend clutter for individual seeds

    ax4.scatter(coords2[k, 0], coords2[k, 1],
                c=c, marker=m, s=sz, edgecolors="black",
                linewidths=lw, zorder=zorder, alpha=0.85,
                label=label if kind == "src" else "_nolegend_")

    if kind == "src":
        ax4.annotate(f"{name}\nsource", (coords2[k, 0], coords2[k, 1]),
                     textcoords="offset points", xytext=(8, 5), fontsize=8,
                     fontweight="bold")
    else:
        sim = multi_regens[(name, seed)]["sim"]
        ax4.annotate(f"s{seed}\n{sim:.2f}", (coords2[k, 0], coords2[k, 1]),
                     textcoords="offset points", xytext=(4, 3), fontsize=6,
                     color=regen_colors[name])

# Draw lines from each source to its regens
src_idx = {name: i for i, (name, kind, _) in enumerate(ms_labels) if kind == "src"}
for k, (name, kind, seed) in enumerate(ms_labels):
    if kind == "regen":
        si = src_idx[name]
        ax4.plot([coords2[si, 0], coords2[k, 0]],
                 [coords2[si, 1], coords2[k, 1]],
                 color=regen_colors[name], lw=0.8, linestyle="--", alpha=0.5)

import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color=src_colors[n],   label=f"{n} source (large)") for n in order
] + [
    mpatches.Patch(color=regen_colors[n], label=f"{n} regens 5 seeds") for n in order
]
ax4.legend(handles=legend_handles, fontsize=8, loc="best")
ax4.set_title(
    "CLIP Embedding Space — 3 sources + 15 regens (5 seeds × 3) — PCA 2D\n"
    "Tight cluster = stable cycle; spread cluster = cycle unreliable",
    fontsize=10, fontweight="bold"
)
ax4.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% var)")
ax4.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% var)")
ax4.grid(True, alpha=0.3)
plt.tight_layout()
out4 = OUT_DIR / "cycle_consistency_multiseed_pca.png"
plt.savefig(out4, dpi=150, bbox_inches="tight")
print(f"Saved → {out4}")
print("\nDone.")
