"""
Cycle Consistency Analysis — SD 3.5 Medium.

Experiment A — Sanity check (should succeed):
  SD3.5 "a cat lying on the bed"
      → SD-IPC projection (CLIP-L space) → SD3 conditioning
      → SD3.5 regeneration
  Expected: cycle closes visually (high CLIP similarity)

Experiment B — Composability gap probe (expected to fail):
  SuperDiff AND "a cat" ^ "a dog"  (hybrid/chimera image)
      → SD-IPC projection → SD3 conditioning
      → SD3.5 regeneration
  Expected: cycle BREAKS — regeneration escapes the hybrid

Experiment C — Monolithic baseline (should close):
  SD3.5 "a cat and a dog"
      → SD-IPC projection → SD3 conditioning
      → SD3.5 regeneration

SD-IPC injection into SD3:
  - Project image → 768-dim CLIP-L text space (same as SD1.4 approach)
  - Inject into CLIP-L branch of SD3 conditioning (BOS + proj broadcasted)
  - Use null (empty-prompt) for CLIP-G and T5 branches
  - pooled_projections: [proj_vec | null_clip_g_pool]

Usage:
    conda activate compose_gligen
    python scripts/cycle_consistency_analysis_sd3.py
"""
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from transformers import (
    CLIPTextModelWithProjection, CLIPTokenizer,
    T5EncoderModel, T5TokenizerFast,
    CLIPModel, CLIPProcessor,
)
from diffusers import (
    SD3Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKL,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
CLIP_ID  = "openai/clip-vit-large-patch14"
OUT_DIR  = Path("experiments/composability_gap")
SEED     = 42
STEPS    = 28
SCALE    = 4.5

AND_IMG_PATH = OUT_DIR / "sd3_superdiff_and_cat_dog.png"

EXPERIMENTS = {
    "sanity": "a cat lying on the bed",
    "mono"  : "a cat and a dog",
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.bfloat16
print(f"Device: {device}")
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Load SD3.5 components
# ---------------------------------------------------------------------------
MAX_LEN_CLIP = 77
MAX_LEN_T5   = 256

print(f"\nLoading SD3.5 ...")
transformer = SD3Transformer2DModel.from_pretrained(
    MODEL_ID, subfolder="transformer", torch_dtype=dtype).to(device)
vae = AutoencoderKL.from_pretrained(
    MODEL_ID, subfolder="vae", torch_dtype=dtype).to(device)
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    MODEL_ID, subfolder="scheduler")

tokenizer_l  = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
tokenizer_g  = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2")
tokenizer_t5 = T5TokenizerFast.from_pretrained(MODEL_ID, subfolder="tokenizer_3")
text_enc_l   = CLIPTextModelWithProjection.from_pretrained(
    MODEL_ID, subfolder="text_encoder",   torch_dtype=dtype).to(device)
text_enc_g   = CLIPTextModelWithProjection.from_pretrained(
    MODEL_ID, subfolder="text_encoder_2", torch_dtype=dtype).to(device)
text_enc_t5  = T5EncoderModel.from_pretrained(
    MODEL_ID, subfolder="text_encoder_3", torch_dtype=dtype).to(device)

transformer.eval(); vae.eval()
text_enc_l.eval(); text_enc_g.eval(); text_enc_t5.eval()

# Standalone CLIP ViT-L/14 — on CPU during generation
print(f"Loading CLIP ViT-L/14 (CPU) ...")
clip_model     = CLIPModel.from_pretrained(CLIP_ID)
clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)
clip_model.eval()

# SD-IPC closed-form projection matrices
with torch.no_grad():
    inv_text    = torch.linalg.pinv(clip_model.text_projection.weight.float(), atol=0.3)
    visual_proj = clip_model.visual_projection.weight.float()

# ---------------------------------------------------------------------------
# Helpers: SD3 text encoding
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_sd3_prompt(prompt: str):
    """Returns (prompt_embeds (1,410,4096), pooled_embeds (1,2048))."""
    ids_l = tokenizer_l(prompt, padding="max_length", max_length=MAX_LEN_CLIP,
                        truncation=True, return_tensors="pt").input_ids.to(device)
    out_l = text_enc_l(ids_l, output_hidden_states=True)
    emb_l    = out_l.hidden_states[-2]   # (1, 77, 768)
    pooled_l = out_l.text_embeds          # (1, 768)

    ids_g = tokenizer_g(prompt, padding="max_length", max_length=MAX_LEN_CLIP,
                        truncation=True, return_tensors="pt").input_ids.to(device)
    out_g = text_enc_g(ids_g, output_hidden_states=True)
    emb_g    = out_g.hidden_states[-2]   # (1, 77, 1280)
    pooled_g = out_g.text_embeds          # (1, 1280)

    ids_t5 = tokenizer_t5(prompt, padding="max_length", max_length=MAX_LEN_T5,
                          truncation=True, return_tensors="pt").input_ids.to(device)
    emb_t5 = text_enc_t5(ids_t5).last_hidden_state   # (1, 256, 4096)

    emb_l_pad = torch.nn.functional.pad(emb_l, (0, 4096 - 768))
    emb_g_pad = torch.nn.functional.pad(emb_g, (0, 4096 - 1280))
    clip_seq  = torch.cat([emb_l_pad, emb_g_pad], dim=1)          # (1, 154, 4096)
    prompt_embeds = torch.cat([clip_seq, emb_t5], dim=1)            # (1, 410, 4096)
    pooled_embeds = torch.cat([pooled_l, pooled_g], dim=-1)         # (1, 2048)
    return prompt_embeds, pooled_embeds


# Pre-compute null (empty-prompt) conditioning — needed for CFG and SD-IPC injection
print("Pre-computing null conditioning ...")
with torch.no_grad():
    uncond_embeds, uncond_pool = encode_sd3_prompt("")

    # Cache null CLIP-G pooled for SD-IPC injection
    ids_g_null = tokenizer_g("", padding="max_length", max_length=MAX_LEN_CLIP,
                             truncation=True, return_tensors="pt").input_ids.to(device)
    out_g_null = text_enc_g(ids_g_null, output_hidden_states=True)
    null_clip_g_seq    = out_g_null.hidden_states[-2]   # (1, 77, 1280)
    null_clip_g_pooled = out_g_null.text_embeds          # (1, 1280)

    # Cache null CLIP-L sequence (for BOS token)
    ids_l_null = tokenizer_l("", padding="max_length", max_length=MAX_LEN_CLIP,
                             truncation=True, return_tensors="pt").input_ids.to(device)
    out_l_null = text_enc_l(ids_l_null, output_hidden_states=True)
    null_clip_l_seq = out_l_null.hidden_states[-2]  # (1, 77, 768)

    # Cache null T5
    ids_t5_null = tokenizer_t5("", padding="max_length", max_length=MAX_LEN_T5,
                               truncation=True, return_tensors="pt").input_ids.to(device)
    null_t5_seq = text_enc_t5(ids_t5_null).last_hidden_state  # (1, 256, 4096)


# ---------------------------------------------------------------------------
# Helpers: decode, CLIP embeddings, SD-IPC
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode(latents: torch.Tensor) -> Image.Image:
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    x = vae.decode(latents, return_dict=False)[0]
    x = (x.float() / 2 + 0.5).clamp(0, 1)
    x = (x.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()[0]
    return Image.fromarray(x)


def _l2(e):
    if not isinstance(e, torch.Tensor):
        e = e.pooler_output if hasattr(e, "pooler_output") else e[0][:, 0]
    e = e.float()
    return e / e.norm(dim=-1, keepdim=True)


@torch.no_grad()
def clip_image_emb(image: Image.Image) -> torch.Tensor:
    """L2-normalised (1, 768) CLIP image embedding."""
    clip_model.to(device)
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    return _l2(clip_model.get_image_features(**inputs))


@torch.no_grad()
def clip_text_emb(prompt: str) -> torch.Tensor:
    clip_model.to(device)
    inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    return _l2(clip_model.get_text_features(**inputs))


@torch.no_grad()
def sdipc_project(image: Image.Image) -> torch.Tensor:
    """
    SD-IPC closed-form projection.
    Returns (1, 768) in CLIP-L text encoder space.
    """
    clip_model.to(device)
    pv         = clip_processor(images=image, return_tensors="pt").pixel_values.to(device)
    pooler_out = clip_model.vision_model(pixel_values=pv).pooler_output.float()  # (1, 1024)
    joint      = pooler_out @ visual_proj.to(device).T   # (1, 768)
    text_space = joint @ inv_text.to(device).T            # (1, 768)
    text_space = text_space / text_space.norm(dim=-1, keepdim=True)
    return 27.5 * text_space


@torch.no_grad()
def sdipc_to_sd3_cond(proj_vec: torch.Tensor):
    """
    Build SD3 conditioning (prompt_embeds, pooled_projections) from SD-IPC projection.

    Strategy:
      - CLIP-L sequence : BOS from null + proj_vec broadcasted to positions 1-76
      - CLIP-G sequence : null (empty prompt)
      - T5 sequence     : null (empty prompt)
      - pooled_l        : proj_vec  (it lives in CLIP-L joint space, same as text_embeds)
      - pooled_g        : from null CLIP-G

    Returns:
      prompt_embeds      : (1, 410, 4096)
      pooled_projections : (1, 2048)
    """
    pv = proj_vec.to(dtype=dtype, device=device)   # (1, 768)

    # CLIP-L sequence: BOS token + proj_vec broadcast
    seq_l = torch.zeros_like(null_clip_l_seq)       # (1, 77, 768)
    seq_l[:, 0] = null_clip_l_seq[:, 0]             # keep BOS
    seq_l[:, 1:] = pv.unsqueeze(1)                  # fill positions 1-76

    # Pad and concat with null CLIP-G and null T5
    seq_l_pad = torch.nn.functional.pad(seq_l, (0, 4096 - 768))         # (1, 77, 4096)
    seq_g_pad = torch.nn.functional.pad(null_clip_g_seq, (0, 4096 - 1280))  # (1, 77, 4096)
    clip_seq  = torch.cat([seq_l_pad, seq_g_pad], dim=1)                 # (1, 154, 4096)
    prompt_embeds = torch.cat([clip_seq, null_t5_seq], dim=1)             # (1, 410, 4096)

    # Pooled: proj_vec as CLIP-L pool + null CLIP-G pool
    pooled_g = null_clip_g_pooled.to(dtype=dtype, device=device)         # (1, 1280)
    pooled_projections = torch.cat([pv, pooled_g], dim=-1)               # (1, 2048)

    return prompt_embeds, pooled_projections


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float(); b = b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))


# ---------------------------------------------------------------------------
# SD3 denoising loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_sd3(prompt_embeds: torch.Tensor, pooled_projections: torch.Tensor,
            seed: int = SEED) -> Image.Image:
    """Standard CFG denoising with SD3.5."""
    scheduler.set_timesteps(STEPS, device=device)
    gen     = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, transformer.config.in_channels,
         transformer.config.sample_size, transformer.config.sample_size),
        generator=gen, device=device, dtype=dtype)
    # FlowMatch: no init_noise_sigma, latents are pure N(0,1)

    pe = prompt_embeds.to(device=device, dtype=dtype)
    pp = pooled_projections.to(device=device, dtype=dtype)

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]

        vel_c = transformer(hidden_states=latents, timestep=t.expand(latents.shape[0]),
                            encoder_hidden_states=pe,
                            pooled_projections=pp).sample
        vel_u = transformer(hidden_states=latents, timestep=t.expand(latents.shape[0]),
                            encoder_hidden_states=uncond_embeds,
                            pooled_projections=uncond_pool).sample
        vf      = vel_u + SCALE * (vel_c - vel_u)
        latents = latents + dsigma * vf   # clean ODE step

    return decode(latents)


# ---------------------------------------------------------------------------
# Step 1: Generate source images
# ---------------------------------------------------------------------------
clip_model.cpu()
torch.cuda.empty_cache()
print("\n=== Generating source images ===")

sources = {}
for name, prompt in EXPERIMENTS.items():
    print(f"  SD3.5: '{prompt}'")
    pe, pp = encode_sd3_prompt(prompt)
    img    = run_sd3(pe, pp, seed=SEED)
    sources[name] = {"prompt": prompt, "image": img}
    img.save(OUT_DIR / f"cycle_sd3_{name}_source.png")
    print(f"  Saved → cycle_sd3_{name}_source.png")

# Load pre-generated AND hybrid
if AND_IMG_PATH.exists():
    print(f"  Loading AND hybrid from {AND_IMG_PATH}")
    sources["and"] = {"prompt": "SuperDiff AND: 'a cat' ^ 'a dog'",
                      "image": Image.open(AND_IMG_PATH)}
else:
    # Generate AND on-the-fly with stochastic SuperDiff
    print("  AND hybrid not found — generating with stochastic SuperDiff ...")
    obj_emb, obj_pool   = encode_sd3_prompt("a cat")
    bg_emb,  bg_pool    = encode_sd3_prompt("a dog")

    scheduler.set_timesteps(STEPS, device=device)
    gen     = torch.Generator(device=device).manual_seed(SEED)
    latents = torch.randn(
        (1, transformer.config.in_channels,
         transformer.config.sample_size, transformer.config.sample_size),
        generator=gen, device=device, dtype=dtype)

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma  = scheduler.sigmas[i]

        vel_obj = transformer(hidden_states=latents, timestep=t.expand(1), encoder_hidden_states=obj_emb,      pooled_projections=obj_pool   ).sample
        vel_bg  = transformer(hidden_states=latents, timestep=t.expand(1), encoder_hidden_states=bg_emb,       pooled_projections=bg_pool    ).sample
        vel_unc = transformer(hidden_states=latents, timestep=t.expand(1), encoder_hidden_states=uncond_embeds, pooled_projections=uncond_pool).sample

        sigma_safe = sigma.clamp(min=1e-4)
        noise  = torch.sqrt(2 * dsigma.abs() * sigma_safe) * torch.randn_like(latents)
        dx_ind = dsigma * (vel_unc + SCALE * (vel_bg - vel_unc)) + noise
        denom  = (dsigma * SCALE * ((vel_obj - vel_bg)**2).sum((1, 2, 3))).clamp(min=1e-4)
        kappa  = (
            (dsigma.abs() * (vel_bg - vel_obj) * (vel_bg + vel_obj)).sum((1, 2, 3))
            - (dx_ind * (vel_obj - vel_bg)).sum((1, 2, 3))
        ) / denom
        kappa  = kappa.clamp(-2.0, 2.0)
        vf      = vel_unc + SCALE * ((vel_bg - vel_unc) + kappa[:, None, None, None] * (vel_obj - vel_bg))
        latents = latents + dsigma * vf

    img_and = decode(latents)
    img_and.save(AND_IMG_PATH)
    sources["and"] = {"prompt": "SuperDiff AND: 'a cat' ^ 'a dog'", "image": img_and}
    print(f"  Saved → {AND_IMG_PATH}")

# ---------------------------------------------------------------------------
# Step 2: SD-IPC projection → SD3 regen
# ---------------------------------------------------------------------------
# Offload T5 before CLIP ops (T5 is large; CLIP needs GPU briefly)
text_enc_t5.cpu()
torch.cuda.empty_cache()

print("\n=== SD-IPC projection + SD3 regeneration ===")

results = {}
for name, data in sources.items():
    print(f"\n  [{name}] {data['prompt']}")

    proj = sdipc_project(data["image"])          # (1, 768) on device
    pe, pp = sdipc_to_sd3_cond(proj)              # (1,410,4096), (1,2048)

    clip_model.cpu()
    torch.cuda.empty_cache()

    regen = run_sd3(pe, pp, seed=SEED)
    regen.save(OUT_DIR / f"cycle_sd3_{name}_regen.png")
    print(f"  Saved → cycle_sd3_{name}_regen.png")

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
transformer.cpu(); vae.cpu()
text_enc_l.cpu(); text_enc_g.cpu()
torch.cuda.empty_cache()

for name, data in results.items():
    src_emb   = clip_image_emb(data["source"])
    regen_emb = clip_image_emb(data["regen"])
    sim_cycle = cosine_sim(src_emb, regen_emb)

    print(f"\n  [{name}] {data['prompt']}")
    print(f"    Source ↔ Regen (cycle sim):  {sim_cycle:.4f}  {'✓ closed' if sim_cycle > 0.80 else '✗ broken'}")

    if name == "and":
        sim_cat     = cosine_sim(regen_emb, clip_text_emb("a cat"))
        sim_dog     = cosine_sim(regen_emb, clip_text_emb("a dog"))
        sim_cat_dog = cosine_sim(regen_emb, clip_text_emb("a cat and a dog"))
        print(f"    Regen → 'a cat' text:          {sim_cat:.4f}")
        print(f"    Regen → 'a dog' text:           {sim_dog:.4f}")
        print(f"    Regen → 'a cat and a dog' text: {sim_cat_dog:.4f}")
        print(f"    Source → 'a cat' text:          {cosine_sim(src_emb, clip_text_emb('a cat')):.4f}")
        print(f"    Source → 'a dog' text:          {cosine_sim(src_emb, clip_text_emb('a dog')):.4f}")

    results[name]["sim_cycle"] = sim_cycle
    results[name]["src_emb"]   = src_emb
    results[name]["regen_emb"] = regen_emb

# ---------------------------------------------------------------------------
# Step 4: Plot — 3×3 grid
# ---------------------------------------------------------------------------
print("\n=== Plotting ===")

order  = ["sanity", "mono", "and"]
titles = {
    "sanity": "Experiment A — SD3 sanity check\n(should cycle-close)",
    "mono"  : "Experiment B — SD3 monolithic\n(should cycle-close)",
    "and"   : "Experiment C — SuperDiff AND hybrid\n(expected to break)",
}

fig, axes = plt.subplots(3, 3, figsize=(13, 13))

for row, name in enumerate(order):
    data = results[name]
    sim  = data["sim_cycle"]

    axes[row, 0].imshow(np.array(data["source"].resize((256, 256))))
    axes[row, 0].set_title("Source image", fontsize=9)
    axes[row, 0].axis("off")

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

    axes[row, 2].imshow(np.array(data["regen"].resize((256, 256))))
    axes[row, 2].set_title("SD3 regen from SD-IPC embedding", fontsize=9)
    axes[row, 2].axis("off")

    axes[row, 0].set_ylabel(titles[name], fontsize=8, labelpad=6)

plt.suptitle(
    "Cycle Consistency Analysis — SD 3.5 Medium\n"
    "Image → SD-IPC (CLIP-L branch) → SD3.5 regeneration\n"
    "Monolithic SD3 is cycle-consistent; AND hybrid is not — the composability gap",
    fontsize=10, y=1.01
)
plt.tight_layout()
out = OUT_DIR / "cycle_consistency_sd3.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")

# ---------------------------------------------------------------------------
# Step 5: Embedding space PCA
# ---------------------------------------------------------------------------
from sklearn.decomposition import PCA

all_embs   = []
all_labels = []

palette = {
    "sanity_src" : ("#1565C0", "o"),
    "sanity_regen": ("#42A5F5", "o"),
    "mono_src"   : ("#2E7D32", "s"),
    "mono_regen" : ("#81C784", "s"),
    "and_src"    : ("#B71C1C", "^"),
    "and_regen"  : ("#EF9A9A", "^"),
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
out2 = OUT_DIR / "cycle_consistency_sd3_pca.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved → {out2}")
print("\nDone.")
