"""
Composability Gap Analysis — SD 3.5 Medium.

Pipeline:
  1. Generate SuperDiff AND image: "a cat" ^ "a dog"  (stochastic mode, SD3.5)
  2. Generate SD3 monolithic image: "a cat and a dog"  (standard CFG, same seed)
  3. Encode both images + text prompts with CLIP ViT-L/14 (image embeddings)
     AND with SD3's pooled text encoder (CLIP-L + CLIP-G, 2048-dim text embeddings)
  4. Cosine similarity matrix in both spaces
  5. PCA 2D projection + heatmap plot

Usage:
    conda activate compose_gligen
    python scripts/composability_gap_sd3_analysis.py
"""
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA
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
OBJ      = "a cat"
BG       = "a dog"
MONO     = "a cat and a dog"

OUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.bfloat16
print(f"Device: {device}")
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Load SD3.5 components
# ---------------------------------------------------------------------------
print(f"\nLoading SD3.5 from {MODEL_ID} ...")
transformer = SD3Transformer2DModel.from_pretrained(
    MODEL_ID, subfolder="transformer", torch_dtype=dtype).to(device)
vae = AutoencoderKL.from_pretrained(
    MODEL_ID, subfolder="vae", torch_dtype=dtype).to(device)
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    MODEL_ID, subfolder="scheduler")

# SD3 text encoders
tokenizer_l   = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
tokenizer_g   = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2")
tokenizer_t5  = T5TokenizerFast.from_pretrained(MODEL_ID, subfolder="tokenizer_3")
text_enc_l    = CLIPTextModelWithProjection.from_pretrained(
    MODEL_ID, subfolder="text_encoder",   torch_dtype=dtype).to(device)
text_enc_g    = CLIPTextModelWithProjection.from_pretrained(
    MODEL_ID, subfolder="text_encoder_2", torch_dtype=dtype).to(device)
text_enc_t5   = T5EncoderModel.from_pretrained(
    MODEL_ID, subfolder="text_encoder_3", torch_dtype=dtype).to(device)

transformer.eval(); vae.eval()
text_enc_l.eval(); text_enc_g.eval(); text_enc_t5.eval()

# Standalone CLIP ViT-L/14 for image embeddings — kept on CPU during generation
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
MAX_LEN_CLIP = 77
MAX_LEN_T5   = 256

@torch.no_grad()
def encode_sd3_prompt(prompt: str):
    """
    Returns:
        prompt_embeds  : (1, 333, 4096)  — sequence fed to cross-attention
        pooled_embeds  : (1, 2048)       — pooled conditioning (CLIP-L + CLIP-G)
    """
    # CLIP-L
    ids_l = tokenizer_l(prompt, padding="max_length", max_length=MAX_LEN_CLIP,
                        truncation=True, return_tensors="pt").input_ids.to(device)
    out_l = text_enc_l(ids_l, output_hidden_states=True)
    emb_l       = out_l.hidden_states[-2]          # (1, 77, 768)
    pooled_l    = out_l.text_embeds                # (1, 768)

    # CLIP-G
    ids_g = tokenizer_g(prompt, padding="max_length", max_length=MAX_LEN_CLIP,
                        truncation=True, return_tensors="pt").input_ids.to(device)
    out_g = text_enc_g(ids_g, output_hidden_states=True)
    emb_g       = out_g.hidden_states[-2]          # (1, 77, 1280)
    pooled_g    = out_g.text_embeds                # (1, 1280)

    # T5-XXL
    ids_t5 = tokenizer_t5(prompt, padding="max_length", max_length=MAX_LEN_T5,
                          truncation=True, return_tensors="pt").input_ids.to(device)
    emb_t5 = text_enc_t5(ids_t5).last_hidden_state  # (1, 256, 4096)

    # Pad CLIP embeddings to 4096-dim and concat with T5
    emb_l_pad = torch.nn.functional.pad(emb_l, (0, 4096 - 768))   # (1, 77, 4096)
    emb_g_pad = torch.nn.functional.pad(emb_g, (0, 4096 - 1280))  # (1, 77, 4096)
    clip_seq   = torch.cat([emb_l_pad, emb_g_pad], dim=1)          # (1, 154, 4096)
    prompt_embeds = torch.cat([clip_seq, emb_t5], dim=1)            # (1, 410, 4096)

    pooled_embeds = torch.cat([pooled_l, pooled_g], dim=-1)         # (1, 2048)
    return prompt_embeds, pooled_embeds


# ---------------------------------------------------------------------------
# Helpers: CLIP image/text embeddings  (CPU→GPU swap)
# ---------------------------------------------------------------------------
def _clip_l2(e):
    if not isinstance(e, torch.Tensor):
        e = e.pooler_output if hasattr(e, "pooler_output") else e[0][:, 0]
    e = e.float()
    return e / e.norm(dim=-1, keepdim=True)


@torch.no_grad()
def clip_text_emb(prompt: str):
    clip_model.to(device)
    inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    e = clip_model.get_text_features(**inputs)
    return _clip_l2(e)


@torch.no_grad()
def clip_image_emb(image: Image.Image):
    clip_model.to(device)
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    e = clip_model.get_image_features(**inputs)
    return _clip_l2(e)


@torch.no_grad()
def sdipc_project(image: Image.Image):
    clip_model.to(device)
    pv         = clip_processor(images=image, return_tensors="pt").pixel_values.to(device)
    pooler_out = clip_model.vision_model(pixel_values=pv).pooler_output.float()
    joint      = pooler_out @ visual_proj.to(device).T
    text_space = joint @ inv_text.to(device).T
    text_space = text_space / text_space.norm(dim=-1, keepdim=True)
    return 27.5 * text_space


# ---------------------------------------------------------------------------
# Helpers: decode SD3 latents
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode(latents: torch.Tensor) -> Image.Image:
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    x = vae.decode(latents, return_dict=False)[0]
    x = (x.float() / 2 + 0.5).clamp(0, 1)
    x = (x.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()[0]
    return Image.fromarray(x)


# ---------------------------------------------------------------------------
# Generation 1: SuperDiff AND  (stochastic, SD3.5)
# ---------------------------------------------------------------------------
clip_model.cpu()
torch.cuda.empty_cache()
print(f"\n=== SuperDiff AND (stochastic, SD3.5): '{OBJ}'  ^  '{BG}' ===")

obj_emb,    obj_pool    = encode_sd3_prompt(OBJ)
bg_emb,     bg_pool     = encode_sd3_prompt(BG)
uncond_emb, uncond_pool = encode_sd3_prompt("")

scheduler.set_timesteps(STEPS, device=device)
gen     = torch.Generator(device=device).manual_seed(SEED)
latents = torch.randn(
    (1, transformer.config.in_channels,
     transformer.config.sample_size, transformer.config.sample_size),
    generator=gen, device=device, dtype=dtype)
# FlowMatchEulerDiscreteScheduler has no init_noise_sigma; FM latents are pure N(0,1)

def _vel_sd3(x, emb, pool):
    return transformer(
        hidden_states=x,
        timestep=t.expand(x.shape[0]),
        encoder_hidden_states=emb,
        pooled_projections=pool,
    ).sample

with torch.no_grad():
    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma  = scheduler.sigmas[i]

        vel_obj = _vel_sd3(latents, obj_emb,    obj_pool)
        vel_bg  = _vel_sd3(latents, bg_emb,     bg_pool)
        vel_unc = _vel_sd3(latents, uncond_emb, uncond_pool)

        # SD3 stochastic: noise used only for kappa estimation; update is clean ODE
        sigma_safe = sigma.clamp(min=1e-4)
        noise  = torch.sqrt(2 * dsigma.abs() * sigma_safe) * torch.randn_like(latents)
        dx_ind = dsigma * (vel_unc + SCALE * (vel_bg - vel_unc)) + noise
        denom  = (dsigma * SCALE * ((vel_obj - vel_bg)**2).sum((1, 2, 3))).clamp(min=1e-4)
        kappa  = (
            (dsigma.abs() * (vel_bg - vel_obj) * (vel_bg + vel_obj)).sum((1, 2, 3))
            - (dx_ind * (vel_obj - vel_bg)).sum((1, 2, 3))
        ) / denom
        kappa = kappa.clamp(-2.0, 2.0)

        vf      = vel_unc + SCALE * ((vel_bg - vel_unc) + kappa[:, None, None, None] * (vel_obj - vel_bg))
        latents = latents + dsigma * vf   # clean ODE step

        if (i + 1) % 7 == 0:
            print(f"  AND step {i+1}/{STEPS}")

    img_and = decode(latents)

img_and.save(OUT_DIR / "sd3_superdiff_and_cat_dog.png")
print(f"Saved → {OUT_DIR}/sd3_superdiff_and_cat_dog.png")

# ---------------------------------------------------------------------------
# Generation 2: SD3 Monolithic  ("a cat and a dog", standard CFG)
# ---------------------------------------------------------------------------
print(f"\n=== SD3 Monolithic: '{MONO}' ===")

mono_emb, mono_pool = encode_sd3_prompt(MONO)

scheduler.set_timesteps(STEPS, device=device)
gen     = torch.Generator(device=device).manual_seed(SEED)
latents = torch.randn(
    (1, transformer.config.in_channels,
     transformer.config.sample_size, transformer.config.sample_size),
    generator=gen, device=device, dtype=dtype)

with torch.no_grad():
    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]

        vel_m  = _vel_sd3(latents, mono_emb,   mono_pool)
        vel_u  = _vel_sd3(latents, uncond_emb, uncond_pool)

        vf      = vel_u + SCALE * (vel_m - vel_u)
        latents = latents + dsigma * vf   # clean ODE step

        if (i + 1) % 7 == 0:
            print(f"  Mono step {i+1}/{STEPS}")

    img_mono = decode(latents)

img_mono.save(OUT_DIR / "sd3_mono_cat_dog.png")
print(f"Saved → {OUT_DIR}/sd3_mono_cat_dog.png")

# ---------------------------------------------------------------------------
# 3. Embeddings: CLIP joint space (image) + SD3 pooled text space
# Offload SD3 models first
# ---------------------------------------------------------------------------
print("\n=== Extracting embeddings ===")
transformer.cpu(); vae.cpu()
text_enc_l.cpu(); text_enc_g.cpu(); text_enc_t5.cpu()
torch.cuda.empty_cache()

# -- CLIP joint-space embeddings (L2-normalised, 768-dim) --
with torch.no_grad():
    clip_E = {
        "text 'a cat'"          : clip_text_emb(OBJ ).cpu().numpy(),
        "text 'a dog'"          : clip_text_emb(BG  ).cpu().numpy(),
        "text 'a cat and a dog'": clip_text_emb(MONO).cpu().numpy(),
        "img: SuperDiff AND"    : clip_image_emb(img_and ).cpu().numpy(),
        "img: SD3 monolithic"   : clip_image_emb(img_mono).cpu().numpy(),
    }

clip_labels   = list(clip_E.keys())
clip_all_embs = np.vstack(list(clip_E.values()))

# -- SD3 pooled text embeddings (2048-dim, L2-normalised) --
text_enc_l.to(device); text_enc_g.to(device)
with torch.no_grad():
    def sd3_pool(prompt):
        """SD3 pooled embedding (2048-dim): CLIP-L + CLIP-G only, no T5."""
        ids_l = tokenizer_l(prompt, padding="max_length", max_length=MAX_LEN_CLIP,
                            truncation=True, return_tensors="pt").input_ids.to(device)
        pooled_l = text_enc_l(ids_l, output_hidden_states=True).text_embeds   # (1, 768)

        ids_g = tokenizer_g(prompt, padding="max_length", max_length=MAX_LEN_CLIP,
                            truncation=True, return_tensors="pt").input_ids.to(device)
        pooled_g = text_enc_g(ids_g, output_hidden_states=True).text_embeds   # (1, 1280)

        pool = torch.cat([pooled_l, pooled_g], dim=-1).float().cpu().numpy()  # (1, 2048)
        return pool / np.linalg.norm(pool)

    sd3_text_cat    = sd3_pool(OBJ)
    sd3_text_dog    = sd3_pool(BG)
    sd3_text_catdog = sd3_pool(MONO)
text_enc_l.cpu(); text_enc_g.cpu()
torch.cuda.empty_cache()

# For image→SD3-text-space: use SD-IPC projection (maps to 768-dim SD text space;
# we pad to 2048 with zeros for distance comparison — or keep separate)
with torch.no_grad():
    proj_and  = sdipc_project(img_and ).cpu().numpy().flatten()
    proj_mono = sdipc_project(img_mono).cpu().numpy().flatten()

def cos(a, b):
    a = a.flatten(); b = b.flatten()
    return float(np.dot(a/np.linalg.norm(a), b/np.linalg.norm(b)))

# ---------------------------------------------------------------------------
# 4. Cosine similarity — CLIP space
# ---------------------------------------------------------------------------
clip_sims = clip_all_embs @ clip_all_embs.T
print("\n--- CLIP joint-space cosine similarities ---")
pairs = [
    (3, 2, "AND  → 'a cat and a dog' text"),
    (4, 2, "Mono → 'a cat and a dog' text"),
    (3, 4, "AND  ↔ Mono (image-image)"),
    (3, 0, "AND  → 'a cat' text"),
    (3, 1, "AND  → 'a dog' text"),
    (4, 0, "Mono → 'a cat' text"),
    (4, 1, "Mono → 'a dog' text"),
]
for i, j, desc in pairs:
    print(f"  {desc:42s}  {clip_sims[i,j]:.4f}")

print("\n--- SD-IPC projection (CLIP text-encoder space, 768-dim) ---")
print(f"  SD-IPC(AND)  vs SD-IPC(mono)          : {cos(proj_and,  proj_mono):.4f}")

print("\n--- SD3 pooled text space (CLIP-L+G, 2048-dim) ---")
print(f"  SD3 'a cat'  vs 'a dog'               : {cos(sd3_text_cat, sd3_text_dog):.4f}")
print(f"  SD3 'a cat'  vs 'a cat and a dog'      : {cos(sd3_text_cat, sd3_text_catdog):.4f}")
print(f"  SD3 'a dog'  vs 'a cat and a dog'      : {cos(sd3_text_dog, sd3_text_catdog):.4f}")

# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------
print("\n=== Plotting ===")
pca    = PCA(n_components=2)
coords = pca.fit_transform(clip_all_embs)

colors  = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
markers = ["s",       "s",       "s",       "o",       "o"      ]
sizes   = [180,       180,       180,       240,       240      ]

fig, axes = plt.subplots(1, 2, figsize=(17, 7))

# Left: PCA
ax = axes[0]
for k in range(len(clip_labels)):
    ax.scatter(coords[k, 0], coords[k, 1],
               c=colors[k], marker=markers[k], s=sizes[k],
               zorder=5, edgecolors="black", linewidths=0.8)
    ax.annotate(clip_labels[k], (coords[k, 0], coords[k, 1]),
                textcoords="offset points", xytext=(9, 5), fontsize=8.5)

composed_idx = clip_labels.index("text 'a cat and a dog'")
for k in [clip_labels.index("img: SuperDiff AND"), clip_labels.index("img: SD3 monolithic")]:
    ax.annotate("", xy=coords[composed_idx], xytext=coords[k],
                arrowprops=dict(arrowstyle="->", color=colors[k], lw=1.6, linestyle="dashed"))

ax.set_title("CLIP Embedding Space — PCA 2D  (SD3.5 images)", fontsize=11, fontweight="bold")
ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.grid(True, alpha=0.3)
ax.legend(handles=[
    mpatches.Patch(color="steelblue",  label="■  Text embeddings"),
    mpatches.Patch(color="darkorchid", label="●  Image embeddings (CLIP)"),
], fontsize=9)

# Right: similarity heatmap
ax2 = axes[1]
short = ["cat\n(txt)", "dog\n(txt)", "cat+dog\n(txt)", "AND\n(img)", "mono\n(img)"]
im   = ax2.imshow(clip_sims, cmap="RdYlGn", vmin=0.0, vmax=1.0)
ax2.set_xticks(range(5)); ax2.set_xticklabels(short, fontsize=9)
ax2.set_yticks(range(5)); ax2.set_yticklabels(short, fontsize=9)
for i in range(5):
    for j in range(5):
        ax2.text(j, i, f"{clip_sims[i,j]:.3f}", ha="center", va="center", fontsize=9,
                 color="black" if 0.25 < clip_sims[i,j] < 0.9 else "white")
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
ax2.set_title("Cosine Similarity Matrix (CLIP space)", fontsize=11, fontweight="bold")

plt.suptitle(
    f"Composability Gap  |  SuperDiff AND vs SD3.5 Monolithic\n"
    f"seed={SEED} · steps={STEPS} · guidance={SCALE}",
    fontsize=10, y=1.01
)
plt.tight_layout()
out_plot = OUT_DIR / "composability_gap_sd3.png"
plt.savefig(out_plot, dpi=150, bbox_inches="tight")
print(f"Saved plot → {out_plot}")
print("\nDone.")
