"""
Cycle Consistency Analysis — SDXL + SD-IPC projection (SDXL adaptation)

SD-IPC for SDXL uses both CLIP branches:
  CLIP-L (ViT-L/14, 768-dim):  pinv(clip_l.text_projection)  → fills seq_l positions 1–76
  CLIP-G (ViT-bigG, 1280-dim): pinv(text_enc_g.text_projection) → fills seq_g positions 1–76
  prompt_embeds = cat([seq_l, seq_g], dim=-1)  → (1, 77, 2048)
  pooled_prompt_embeds = CLIP-G image features  → (1, 1280)  [already in CLIP-G joint space]

Experiments mirror cycle_consistency_analysis.py:
  A — Sanity:   SD "a cat lying on the bed"  → should cycle-close
  B — Mono:     SD "a cat and a dog"         → should cycle-close
  C — AND:      SuperDiff AND hybrid         → expected to break (composability gap)

Usage:
    conda activate compose_gligen
    pip install open-clip-torch -q   # if not already installed
    python scripts/cycle_consistency_analysis_sdxl.py
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA

from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SDXL_ID  = "stabilityai/stable-diffusion-xl-base-1.0"
CLIP_L_ID = "openai/clip-vit-large-patch14"
CLIP_G_ID = "ViT-bigG-14"          # open_clip model name
CLIP_G_PRETRAINED = "laion2b_s39b_b160k"  # matches SDXL text_encoder_2 weights

OUT_DIR  = Path("experiments/composability_gap")
SEED     = 42
STEPS    = 30    # SDXL quality at 30 DPM steps
SCALE    = 7.5
SIZE     = 1024  # SDXL native resolution

EXPERIMENTS = {
    "sanity": "a cat lying on the bed",
    "mono"  : "a cat and a dog",
}
AND_IMG_PATH = OUT_DIR / "superdiff_and_cat_dog.png"

OUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Load SDXL components (fp16)
# ---------------------------------------------------------------------------
print("Loading SDXL (fp16) ...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    SDXL_ID,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
# SDXL VAE is numerically unstable in fp16 (produces NaN → black images).
# upcast_vae() casts VAE weights to fp32 before moving to device.
pipe.upcast_vae()
pipe = pipe.to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

unet        = pipe.unet.eval()
vae         = pipe.vae.eval()   # fp32 after upcast_vae()
text_enc_l  = pipe.text_encoder.eval()   # CLIPTextModel            → (1, 77, 768)
text_enc_g  = pipe.text_encoder_2.eval() # CLIPTextModelWithProj    → (1, 77, 1280) + pooled (1, 1280)
tokenizer_l = pipe.tokenizer
tokenizer_g = pipe.tokenizer_2
scheduler   = pipe.scheduler

# ---------------------------------------------------------------------------
# Load CLIP-L full model (for visual projection — SD-IPC)
# ---------------------------------------------------------------------------
print("Loading CLIP-L ViT-L/14 (CPU, for SD-IPC projection) ...")
clip_l     = CLIPModel.from_pretrained(CLIP_L_ID)  # starts on CPU
proc_l     = CLIPProcessor.from_pretrained(CLIP_L_ID)
clip_l.eval()

# ---------------------------------------------------------------------------
# Load CLIP-G visual encoder via open_clip
# ---------------------------------------------------------------------------
print("Loading CLIP-G ViT-bigG-14 (CPU, for SD-IPC projection) ...")
import open_clip
clip_g, _, preproc_g = open_clip.create_model_and_transforms(
    CLIP_G_ID, pretrained=CLIP_G_PRETRAINED
)
clip_g.eval()

# ---------------------------------------------------------------------------
# Pre-compute projection matrices (float32 for numerical stability)
# ---------------------------------------------------------------------------
with torch.no_grad():
    # CLIP-L: text_projection (768, 768), maps hidden → joint space
    # pinv goes: joint space → hidden space
    W_l_pinv = torch.linalg.pinv(clip_l.text_projection.weight.float(), atol=0.3)   # (768, 768)

    # CLIP-G: text_projection from text_enc_g (1280, 1280)
    W_g_pinv = torch.linalg.pinv(text_enc_g.text_projection.weight.float(), atol=0.3)  # (1280, 1280)

SDIPC_SCALE = 27.5  # from SD-IPC paper (calibrated for CLIP-L; applied to both branches)

# ---------------------------------------------------------------------------
# Helpers — text encoding
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_l(prompt: str) -> torch.Tensor:
    """CLIP-L hidden states → (1, 77, 768) fp16."""
    toks = tokenizer_l(
        prompt, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt"
    ).to(device)
    return text_enc_l(**toks).last_hidden_state  # (1, 77, 768) fp16


@torch.no_grad()
def encode_g(prompt: str):
    """CLIP-G hidden states + pooled → (1, 77, 1280) fp16, (1, 1280) fp16."""
    toks = tokenizer_g(
        prompt, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt"
    ).to(device)
    out = text_enc_g(**toks)
    return out.last_hidden_state, out.text_embeds   # (1,77,1280), (1,1280)


@torch.no_grad()
def encode_sdxl(prompt: str):
    """Full SDXL conditioning: prompt_embeds (1,77,2048), pooled (1,1280)."""
    seq_l        = encode_l(prompt)              # (1, 77, 768)
    seq_g, pool  = encode_g(prompt)              # (1, 77, 1280), (1, 1280)
    return torch.cat([seq_l, seq_g], dim=-1), pool  # (1,77,2048), (1,1280)


# ---------------------------------------------------------------------------
# Helpers — SD-IPC SDXL projection
# ---------------------------------------------------------------------------
@torch.no_grad()
def sdipc_project_sdxl(image: Image.Image):
    """
    SD-IPC projection adapted for SDXL.
    Returns:
        prompt_embeds (1, 77, 2048) fp16
        pooled        (1, 1280)     fp16
    """
    # ── CLIP-L branch ────────────────────────────────────────────────────────
    clip_l.to(device)
    pv_l      = proc_l(images=image, return_tensors="pt").pixel_values.to(device)
    feat_l    = clip_l.visual_projection(
                    clip_l.vision_model(pixel_values=pv_l).pooler_output.float()
                )                                                        # (1, 768) joint
    feat_l    = feat_l / feat_l.norm(dim=-1, keepdim=True) * SDIPC_SCALE
    proj_l    = feat_l @ W_l_pinv.to(device).T                          # (1, 768) hidden
    clip_l.cpu()

    # Build seq_l: keep BOS from null prompt, fill positions 1–76
    null_l    = encode_l("")                                             # (1, 77, 768)
    seq_l     = torch.zeros_like(null_l)
    seq_l[:, 0]  = null_l[:, 0]
    seq_l[:, 1:] = proj_l.to(dtype=seq_l.dtype).unsqueeze(1)

    # ── CLIP-G branch ────────────────────────────────────────────────────────
    clip_g.to(device)
    img_t     = preproc_g(image).unsqueeze(0).to(device)
    feat_g    = clip_g.encode_image(img_t).float()                      # (1, 1280) joint
    feat_g    = feat_g / feat_g.norm(dim=-1, keepdim=True) * SDIPC_SCALE
    proj_g    = feat_g @ W_g_pinv.to(device).T                          # (1, 1280) hidden
    clip_g.cpu()

    # Build seq_g: keep BOS from null prompt, fill positions 1–76
    null_g, _ = encode_g("")                                             # (1, 77, 1280)
    seq_g     = torch.zeros_like(null_g)
    seq_g[:, 0]  = null_g[:, 0]
    seq_g[:, 1:] = proj_g.to(dtype=seq_g.dtype).unsqueeze(1)

    # ── Pooled: CLIP-G joint-space image features (same space as text_enc_g output) ──
    # feat_g / SDIPC_SCALE recovers the unit-normalized CLIP-G joint-space vector
    pooled = (feat_g / SDIPC_SCALE).to(dtype=torch.float16)             # (1, 1280)

    prompt_embeds = torch.cat([seq_l, seq_g], dim=-1)                   # (1, 77, 2048)
    return prompt_embeds, pooled


# ---------------------------------------------------------------------------
# Helpers — VAE decode, generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode(latents: torch.Tensor) -> Image.Image:
    # VAE is in fp32 (upcast_vae); pass fp32 latents to avoid dtype mismatch
    x = vae.decode(latents.float() / vae.config.scaling_factor, return_dict=False)[0]
    x = (x / 2 + 0.5).clamp(0, 1)
    x = (x.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()[0]
    return Image.fromarray(x)


def _add_time_ids(dtype=torch.float16):
    """Standard SDXL aesthetic time ids for 1024×1024, no crop offset."""
    return torch.tensor([[SIZE, SIZE, 0, 0, SIZE, SIZE]],
                        dtype=dtype, device=device)   # (1, 6)


@torch.no_grad()
def run_sdxl(prompt_embeds: torch.Tensor,
             pooled: torch.Tensor,
             seed: int = SEED) -> Image.Image:
    """CFG denoising given SDXL conditioning (1,77,2048) + pooled (1,1280)."""
    neg_pe, neg_pool = encode_sdxl("")

    # Batch uncond + cond for single UNet pass
    pe_in     = torch.cat([neg_pe,    prompt_embeds])   # (2, 77, 2048)
    pool_in   = torch.cat([neg_pool,  pooled])           # (2, 1280)
    tids      = _add_time_ids(dtype=prompt_embeds.dtype).repeat(2, 1)  # (2, 6)
    added_cond_kwargs = {"text_embeds": pool_in, "time_ids": tids}

    scheduler.set_timesteps(STEPS)
    gen     = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, unet.config.in_channels, SIZE // 8, SIZE // 8),
        generator=gen, device=device, dtype=torch.float16,
    ) * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        inp        = scheduler.scale_model_input(latents.repeat(2, 1, 1, 1), t)
        noise_pred = unet(
            inp, t,
            encoder_hidden_states=pe_in,
            added_cond_kwargs=added_cond_kwargs,
        ).sample
        noise_u, noise_c = noise_pred.chunk(2)
        noise_pred = noise_u + SCALE * (noise_c - noise_u)
        latents    = scheduler.step(noise_pred, t, latents).prev_sample

    return decode(latents)


# ---------------------------------------------------------------------------
# Helpers — CLIP similarity (CLIP-L for scoring, consistent with SD 1.4 script)
# ---------------------------------------------------------------------------
@torch.no_grad()
def clip_image_emb(image: Image.Image) -> torch.Tensor:
    """L2-normalised (1, 768) CLIP-L joint-space image embedding."""
    clip_l.to(device)
    pv = proc_l(images=image, return_tensors="pt").pixel_values.to(device)
    e  = clip_l.visual_projection(clip_l.vision_model(pixel_values=pv).pooler_output).float()
    clip_l.cpu()
    return F.normalize(e, dim=-1)


@torch.no_grad()
def clip_text_emb(prompt: str) -> torch.Tensor:
    """L2-normalised (1, 768) CLIP-L joint-space text embedding."""
    clip_l.to(device)
    inputs = proc_l(text=[prompt], return_tensors="pt", padding=True)
    toks   = {k: v.to(device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    e      = clip_l.text_projection(clip_l.text_model(**toks).pooler_output).float()
    clip_l.cpu()
    return F.normalize(e, dim=-1)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float(); b = b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))


# ---------------------------------------------------------------------------
# Step 1 — Generate source images
# ---------------------------------------------------------------------------
torch.cuda.empty_cache()
print("\n=== Step 1: Generate source images ===")

sources = {}
for name, prompt in EXPERIMENTS.items():
    print(f"  SDXL: '{prompt}'")
    pe, pool = encode_sdxl(prompt)
    img      = run_sdxl(pe, pool, seed=SEED)
    sources[name] = {"prompt": prompt, "image": img}
    img.save(OUT_DIR / f"sdxl_cycle_{name}_source.png")
    print(f"  Saved → sdxl_cycle_{name}_source.png")

# Load pre-generated AND hybrid (resize to SDXL resolution if needed)
print(f"  Loading AND hybrid from {AND_IMG_PATH}")
and_img = Image.open(AND_IMG_PATH).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
sources["and"] = {"prompt": "SuperDiff AND: 'a cat' ^ 'a dog'", "image": and_img}

# ---------------------------------------------------------------------------
# Step 2 — SD-IPC projection + SDXL regeneration
# ---------------------------------------------------------------------------
print("\n=== Step 2: SD-IPC SDXL projection + regeneration ===")

results = {}
for name, data in sources.items():
    print(f"\n  [{name}] {data['prompt']}")

    pe_proj, pool_proj = sdipc_project_sdxl(data["image"])

    torch.cuda.empty_cache()
    regen = run_sdxl(pe_proj, pool_proj, seed=SEED)
    regen.save(OUT_DIR / f"sdxl_cycle_{name}_regen.png")
    print(f"  Saved → sdxl_cycle_{name}_regen.png")

    results[name] = {
        "prompt"   : data["prompt"],
        "source"   : data["image"],
        "regen"    : regen,
        "pe_proj"  : pe_proj.cpu(),
        "pool_proj": pool_proj.cpu(),
    }

# ---------------------------------------------------------------------------
# Step 3 — CLIP similarity
# ---------------------------------------------------------------------------
print("\n=== Step 3: CLIP similarity ===")

for name, data in results.items():
    src_emb   = clip_image_emb(data["source"])
    regen_emb = clip_image_emb(data["regen"])
    sim       = cosine_sim(src_emb, regen_emb)

    print(f"\n  [{name}] {data['prompt']}")
    print(f"    Source ↔ Regen: {sim:.4f}  {'✓ closed' if sim > 0.80 else '✗ broken'}")

    if name == "and":
        sim_cat    = cosine_sim(regen_emb, clip_text_emb("a cat"))
        sim_dog    = cosine_sim(regen_emb, clip_text_emb("a dog"))
        sim_cd     = cosine_sim(regen_emb, clip_text_emb("a cat and a dog"))
        print(f"    Regen → 'a cat':          {sim_cat:.4f}")
        print(f"    Regen → 'a dog':          {sim_dog:.4f}")
        print(f"    Regen → 'a cat and a dog':{sim_cd:.4f}")

    results[name]["sim_cycle"] = sim
    results[name]["src_emb"]   = src_emb
    results[name]["regen_emb"] = regen_emb

# ---------------------------------------------------------------------------
# Step 4 — Plot: 3×3 grid (source | arrow+sim | regen)
# ---------------------------------------------------------------------------
print("\n=== Step 4: Plotting ===")
order  = ["sanity", "mono", "and"]
titles = {
    "sanity": "Experiment A — SDXL sanity check\n(should cycle-close)",
    "mono"  : "Experiment B — SDXL monolithic\n(should cycle-close)",
    "and"   : "Experiment C — SuperDiff AND hybrid\n(expected to break)",
}

fig, axes = plt.subplots(3, 3, figsize=(13, 13))
for row, name in enumerate(order):
    data = results[name]
    sim  = data["sim_cycle"]
    color = "green" if sim > 0.80 else "red"

    axes[row, 0].imshow(np.array(data["source"].resize((256, 256))))
    axes[row, 0].set_title("Source image", fontsize=9)
    axes[row, 0].axis("off")

    ax = axes[row, 1]
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.annotate("", xy=(0.9, 0.5), xytext=(0.1, 0.5),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=color))
    ax.text(0.5, 0.65, "SD-IPC (SDXL) →", ha="center", fontsize=9, color="grey")
    ax.text(0.5, 0.38, f"CLIP sim = {sim:.3f}", ha="center", fontsize=11,
            fontweight="bold", color=color)
    ax.text(0.5, 0.22, "cycle closed ✓" if sim > 0.80 else "cycle broken ✗",
            ha="center", fontsize=9, color=color)

    axes[row, 2].imshow(np.array(data["regen"].resize((256, 256))))
    axes[row, 2].set_title("SDXL regen from SD-IPC embedding", fontsize=9)
    axes[row, 2].axis("off")
    axes[row, 0].set_ylabel(titles[name], fontsize=8, labelpad=6)

plt.suptitle(
    "Cycle Consistency Analysis — SDXL + SD-IPC (CLIP-L + CLIP-G)\n"
    "Image → SD-IPC SDXL projection → SDXL regeneration\n"
    "Monolithic SD is cycle-consistent; AND hybrid is not — the composability gap",
    fontsize=10, y=1.01
)
plt.tight_layout()
out = OUT_DIR / "cycle_consistency_sdxl.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")

# ---------------------------------------------------------------------------
# Step 5 — Embedding PCA: source + regen
# ---------------------------------------------------------------------------
all_embs  = []
all_labels = []
palette = {
    "sanity_src"  : ("#1565C0", "o"),
    "sanity_regen": ("#42A5F5", "o"),
    "mono_src"    : ("#2E7D32", "s"),
    "mono_regen"  : ("#81C784", "s"),
    "and_src"     : ("#B71C1C", "^"),
    "and_regen"   : ("#EF9A9A", "^"),
}
label_map = {
    "sanity_src"  : "Sanity source",   "sanity_regen": "Sanity regen",
    "mono_src"    : "Mono source",     "mono_regen"  : "Mono regen",
    "and_src"     : "AND source",      "and_regen"   : "AND regen",
}

for name in order:
    d = results[name]
    all_embs.append(d["src_emb"].cpu().numpy())
    all_labels.append(f"{name}_src")
    all_embs.append(d["regen_emb"].cpu().numpy())
    all_labels.append(f"{name}_regen")

coords = PCA(n_components=2).fit_transform(np.vstack(all_embs))

fig2, ax2 = plt.subplots(figsize=(8, 7))
for k, label in enumerate(all_labels):
    color, marker = palette[label]
    ax2.scatter(coords[k, 0], coords[k, 1], c=color, marker=marker,
                s=220, edgecolors="black", linewidths=0.7, zorder=5)
    ax2.annotate(label_map[label], (coords[k, 0], coords[k, 1]),
                 textcoords="offset points", xytext=(7, 4), fontsize=8.5)
for i in range(0, len(all_labels), 2):
    c = palette[all_labels[i]][0]
    ax2.plot([coords[i, 0], coords[i+1, 0]], [coords[i, 1], coords[i+1, 1]],
             color=c, lw=1.5, linestyle="--", alpha=0.7)
ax2.set_title("CLIP Embedding Space — Source vs Regen (PCA 2D)", fontsize=10, fontweight="bold")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "cycle_consistency_sdxl_pca.png", dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_DIR / 'cycle_consistency_sdxl_pca.png'}")

# ---------------------------------------------------------------------------
# Step 6 — Multi-seed 5×3 grid
# ---------------------------------------------------------------------------
print("\n=== Step 6: Multi-seed regen grid ===")
MULTI_SEEDS = [42, 43, 44, 45, 46]

# Pre-compute projections once (seed-independent)
source_proj = {}
for name in order:
    d = results[name]
    source_proj[name] = (
        d["pe_proj"].to(device),
        d["pool_proj"].to(device),
    )

multi_regens = {}
for seed in MULTI_SEEDS:
    torch.cuda.empty_cache()
    for name in order:
        if seed == SEED:
            multi_regens[(name, seed)] = {
                "regen"    : results[name]["regen"],
                "regen_emb": results[name]["regen_emb"],
                "sim"      : results[name]["sim_cycle"],
            }
        else:
            pe, pool  = source_proj[name]
            regen     = run_sdxl(pe, pool, seed=seed)
            regen_emb = clip_image_emb(regen)
            sim       = cosine_sim(results[name]["src_emb"], regen_emb)
            multi_regens[(name, seed)] = {"regen": regen, "regen_emb": regen_emb, "sim": sim}
            regen.save(OUT_DIR / f"sdxl_cycle_{name}_regen_s{seed}.png")
    print(f"  Done seed {seed}")

col_headers = {
    "sanity": 'A · Sanity\n"a cat on the bed"',
    "mono"  : 'B · Monolithic\n"a cat and a dog"',
    "and"   : "C · AND hybrid\ncat ^ dog",
}
fig3, axes3 = plt.subplots(len(MULTI_SEEDS), 3,
                            figsize=(10, 4 * len(MULTI_SEEDS)),
                            gridspec_kw={"hspace": 0.35, "wspace": 0.05})
for row, seed in enumerate(MULTI_SEEDS):
    for col, name in enumerate(order):
        data  = multi_regens[(name, seed)]
        sim   = data["sim"]
        color = "green" if sim > 0.80 else "red"
        ax    = axes3[row, col]
        ax.imshow(np.array(data["regen"].resize((256, 256))))
        ax.axis("off")
        header = col_headers[name] + f"\nseed {seed}" if row == 0 else f"seed {seed}"
        ax.set_title(f"{header}\nCLIP sim = {sim:.3f}",
                     fontsize=8, color=color, fontweight="bold")
plt.suptitle(
    "Multi-seed Regen Grid  (SD-IPC SDXL → SDXL base)\n"
    "Green = cycle closed (sim > 0.80)   Red = cycle broken",
    fontsize=11, y=1.01, fontweight="bold"
)
plt.savefig(OUT_DIR / "cycle_consistency_sdxl_multiseed.png", dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_DIR / 'cycle_consistency_sdxl_multiseed.png'}")

# ---------------------------------------------------------------------------
# Step 7 — Multi-seed PCA: 3 sources + 15 regens
# ---------------------------------------------------------------------------
print("\n=== Step 7: Multi-seed PCA ===")
src_colors   = {"sanity": "#1565C0", "mono": "#2E7D32", "and": "#B71C1C"}
regen_colors = {"sanity": "#42A5F5", "mono": "#81C784", "and": "#EF9A9A"}
markers2     = {"sanity": "o", "mono": "s", "and": "^"}

ms_embs   = []
ms_labels = []
for name in order:
    ms_embs.append(results[name]["src_emb"].cpu().numpy())
    ms_labels.append((name, "src", None))
for seed in MULTI_SEEDS:
    for name in order:
        ms_embs.append(multi_regens[(name, seed)]["regen_emb"].cpu().numpy())
        ms_labels.append((name, "regen", seed))

pca2    = PCA(n_components=2)
coords2 = pca2.fit_transform(np.vstack(ms_embs))

fig4, ax4 = plt.subplots(figsize=(10, 8))
for k, (name, kind, seed) in enumerate(ms_labels):
    if kind == "src":
        c, m, sz, lw, zo = src_colors[name], markers2[name], 350, 1.5, 8
        ax4.scatter(coords2[k, 0], coords2[k, 1], c=c, marker=m, s=sz,
                    edgecolors="black", linewidths=lw, zorder=zo, alpha=0.85)
        ax4.annotate(f"{name}\nsource", (coords2[k, 0], coords2[k, 1]),
                     textcoords="offset points", xytext=(8, 5),
                     fontsize=8, fontweight="bold")
    else:
        sim = multi_regens[(name, seed)]["sim"]
        ax4.scatter(coords2[k, 0], coords2[k, 1], c=regen_colors[name],
                    marker=markers2[name], s=120, edgecolors="black",
                    linewidths=0.5, zorder=4, alpha=0.85)
        ax4.annotate(f"s{seed}\n{sim:.2f}", (coords2[k, 0], coords2[k, 1]),
                     textcoords="offset points", xytext=(4, 3),
                     fontsize=6, color=regen_colors[name])

src_idx = {name: i for i, (name, kind, _) in enumerate(ms_labels) if kind == "src"}
for k, (name, kind, seed) in enumerate(ms_labels):
    if kind == "regen":
        si = src_idx[name]
        ax4.plot([coords2[si, 0], coords2[k, 0]], [coords2[si, 1], coords2[k, 1]],
                 color=regen_colors[name], lw=0.8, linestyle="--", alpha=0.5)

legend_handles = (
    [mpatches.Patch(color=src_colors[n],   label=f"{n} source") for n in order] +
    [mpatches.Patch(color=regen_colors[n], label=f"{n} regens (5 seeds)") for n in order]
)
ax4.legend(handles=legend_handles, fontsize=8)
ax4.set_title(
    "CLIP Embedding Space — 3 sources + 15 regens (5 seeds × 3) — PCA 2D\n"
    "Tight cluster = stable cycle; spread = cycle unreliable",
    fontsize=10, fontweight="bold"
)
ax4.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% var)")
ax4.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% var)")
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "cycle_consistency_sdxl_multiseed_pca.png", dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_DIR / 'cycle_consistency_sdxl_multiseed_pca.png'}")
print("\nDone.")
