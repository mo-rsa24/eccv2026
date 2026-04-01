"""
Composability Gap Analysis via CLIP Embeddings.

Pipeline:
  1. Generate SuperDiff AND image: "a cat" ^ "a dog"  (deterministic mode, SD v1.4)
  2. Generate SD monolithic image: "a cat and a dog"   (standard CFG, same seed)
  3. Encode both images + text prompts with CLIP ViT-L/14
  4. Compute cosine similarity matrix
  5. PCA 2D projection + heatmap plot

Usage:
    conda activate attend_excite
    python scripts/composability_gap_clip_analysis.py
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
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "CompVis/stable-diffusion-v1-4"
CLIP_ID  = "openai/clip-vit-large-patch14"
OUT_DIR  = Path("experiments/composability_gap")
SEED     = 42
STEPS    = 50
SCALE    = 7.5
OBJ      = "a cat"
BG       = "a dog"
MONO     = "a cat and a dog"

OUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Load SD v1.4 components  (UNet in fp16 to save VRAM)
# ---------------------------------------------------------------------------
print(f"\nLoading SD v1.4 from {MODEL_ID} ...")
vae          = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae",          use_safetensors=True).to(device)
tokenizer    = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(device)
unet         = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet",  use_safetensors=True).to(device)
scheduler    = EulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
vae.eval(); text_encoder.eval(); unet.eval()

# ---------------------------------------------------------------------------
# Load standalone CLIP ViT-L/14 for image encoding — kept on CPU until needed
# ---------------------------------------------------------------------------
print(f"Loading CLIP from {CLIP_ID} (CPU) ...")
clip_model     = CLIPModel.from_pretrained(CLIP_ID)   # stays on CPU during generation
clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)
clip_model.eval()

# SD-IPC projection matrices (closed-form, no training needed)
#   image pooler_output → visual_projection → CLIP joint space
#   CLIP joint space    → pinv(text_projection) → text-encoder hidden space
with torch.no_grad():
    inv_text    = torch.linalg.pinv(clip_model.text_projection.weight.float(), atol=0.3)
    visual_proj = clip_model.visual_projection.weight.float()

# ---------------------------------------------------------------------------
# Helpers: SD text embeddings  (fp16 to match UNet)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sd_text_emb(prompt: str):
    """(1, 77, 768) fp32 hidden states for SD cross-attention."""
    ids = tokenizer(
        prompt, padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    ).input_ids.to(device)
    return text_encoder(ids)[0]


# ---------------------------------------------------------------------------
# Helpers: CLIP embeddings  (CLIP moved to GPU only for this block)
# ---------------------------------------------------------------------------
@torch.no_grad()
def clip_text_emb(prompt: str):
    """L2-normalised (1, D) fp32 CLIP joint-space text embedding."""
    clip_model.to(device)
    inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    e = clip_model.get_text_features(**inputs).float()
    return e / e.norm(dim=-1, keepdim=True)


@torch.no_grad()
def clip_image_emb(image: Image.Image):
    """L2-normalised (1, D) fp32 CLIP joint-space image embedding."""
    clip_model.to(device)
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    e = clip_model.get_image_features(**inputs).float()
    return e / e.norm(dim=-1, keepdim=True)


@torch.no_grad()
def sdipc_project(image: Image.Image):
    """
    SD-IPC closed-form projection (Ding et al., NeurIPS 2023):
        image → CLIP pooler → visual_proj → inv_text_proj → normalise → ×27.5
    Returns (1, 768) fp32 in SD text-encoder hidden-state space.
    """
    clip_model.to(device)
    pv         = clip_processor(images=image, return_tensors="pt").pixel_values.to(device)
    pooler_out = clip_model.vision_model(pixel_values=pv).pooler_output.float()  # (1, 1024)
    vp         = visual_proj.to(device)
    it         = inv_text.to(device)
    joint      = pooler_out @ vp.T    # (1, 768)
    text_space = joint @ it.T         # (1, 768)
    text_space = text_space / text_space.norm(dim=-1, keepdim=True)
    return 27.5 * text_space


# ---------------------------------------------------------------------------
# Helpers: decode latents
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode(latents: torch.Tensor) -> Image.Image:
    x = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    x = (x / 2 + 0.5).clamp(0, 1)
    x = (x.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()[0]
    return Image.fromarray(x)


# ---------------------------------------------------------------------------
# Generation 1: SuperDiff AND  ("a cat" ^ "a dog", stochastic mode)
# Stochastic mode uses 3 standard UNet forwards — no JVP, no memory explosion.
# CLIP stays on CPU during generation to keep VRAM free.
# ---------------------------------------------------------------------------
clip_model.cpu()
torch.cuda.empty_cache()
print(f"\n=== SuperDiff AND (stochastic): '{OBJ}'  ^  '{BG}' ===")

obj_emb    = sd_text_emb(OBJ)
bg_emb     = sd_text_emb(BG)
uncond_emb = sd_text_emb("")

scheduler.set_timesteps(STEPS)
gen     = torch.Generator(device=device).manual_seed(SEED)
latents = torch.randn((1, unet.config.in_channels, 64, 64), generator=gen, device=device)
latents = latents * scheduler.init_noise_sigma

with torch.no_grad():
    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma  = scheduler.sigmas[i]

        def _vel(x, e):
            return unet(x / ((sigma**2 + 1)**0.5), t, encoder_hidden_states=e).sample

        vel_obj = _vel(latents, obj_emb)
        vel_bg  = _vel(latents, bg_emb)
        vel_unc = _vel(latents, uncond_emb)

        noise = torch.sqrt(2 * sigma.abs() * dsigma.abs()) * torch.randn_like(latents)
        dx_ind = 2 * dsigma * (vel_unc + SCALE * (vel_bg - vel_unc)) + noise
        denom  = 2 * dsigma * SCALE * ((vel_obj - vel_bg)**2).sum((1, 2, 3)).clamp(min=1e-6)
        kappa  = (
            (dsigma.abs() * (vel_bg - vel_obj) * (vel_bg + vel_obj)).sum((1, 2, 3))
            - (dx_ind * (vel_obj - vel_bg)).sum((1, 2, 3))
        ) / denom
        kappa = kappa.clamp(-2.0, 2.0)

        vf      = vel_unc + SCALE * ((vel_bg - vel_unc) + kappa[:, None, None, None] * (vel_obj - vel_bg))
        latents = latents + 2 * dsigma * vf + noise

        if (i + 1) % 10 == 0:
            print(f"  AND step {i+1}/{STEPS}")

    img_and = decode(latents)
img_and.save(OUT_DIR / "superdiff_and_cat_dog.png")
print(f"Saved → {OUT_DIR}/superdiff_and_cat_dog.png")

# ---------------------------------------------------------------------------
# Generation 2: SD Monolithic  ("a cat and a dog", standard CFG)
# ---------------------------------------------------------------------------
print(f"\n=== SD Monolithic: '{MONO}' ===")

mono_emb = sd_text_emb(MONO)

scheduler.set_timesteps(STEPS)
gen     = torch.Generator(device=device).manual_seed(SEED)
latents = torch.randn((1, unet.config.in_channels, 64, 64), generator=gen, device=device)
latents = latents * scheduler.init_noise_sigma

with torch.no_grad():
    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma  = scheduler.sigmas[i]
        noise  = torch.sqrt(2 * sigma.abs() * dsigma.abs()) * torch.randn_like(latents)

        vel_m  = unet(latents / ((sigma**2 + 1)**0.5), t, encoder_hidden_states=mono_emb  ).sample
        vel_u  = unet(latents / ((sigma**2 + 1)**0.5), t, encoder_hidden_states=uncond_emb).sample

        vf      = vel_u + SCALE * (vel_m - vel_u)
        latents = latents + 2 * dsigma * vf + noise

        if (i + 1) % 10 == 0:
            print(f"  Mono step {i+1}/{STEPS}")

    img_mono = decode(latents)

img_mono.save(OUT_DIR / "sd_mono_cat_dog.png")
print(f"Saved → {OUT_DIR}/sd_mono_cat_dog.png")

# ---------------------------------------------------------------------------
# 3. CLIP embeddings (joint space, L2-normalised)
# Offload SD models to CPU first so CLIP fits on GPU
# ---------------------------------------------------------------------------
print("\n=== Extracting CLIP embeddings ===")
unet.cpu(); vae.cpu(); text_encoder.cpu()
torch.cuda.empty_cache()
with torch.no_grad():
    E = {
        "text 'a cat'"          : clip_text_emb(OBJ ).cpu().numpy(),
        "text 'a dog'"          : clip_text_emb(BG  ).cpu().numpy(),
        "text 'a cat and a dog'": clip_text_emb(MONO).cpu().numpy(),
        "img: SuperDiff AND"    : clip_image_emb(img_and ).cpu().numpy(),
        "img: SD monolithic"    : clip_image_emb(img_mono).cpu().numpy(),
    }

labels   = list(E.keys())
all_embs = np.vstack(list(E.values()))   # (5, D)

# ---------------------------------------------------------------------------
# 4. Cosine similarities
# ---------------------------------------------------------------------------
sims = all_embs @ all_embs.T  # already L2-normalised
print("\nFull cosine similarity matrix:")
header = "  " + "".join(f"{l[:14]:>16}" for l in labels)
print(header)
for i, li in enumerate(labels):
    row = f"{li[:14]:>14} " + "".join(f"{sims[i,j]:>16.4f}" for j in range(len(labels)))
    print(row)

print("\n--- Key composability gap metrics ---")
idx = {l: i for i, l in enumerate(labels)}
pairs = [
    ("img: SuperDiff AND",     "text 'a cat and a dog'", "AND  → composed text"),
    ("img: SD monolithic",     "text 'a cat and a dog'", "Mono → composed text"),
    ("img: SuperDiff AND",     "img: SD monolithic",     "AND  → Mono (image-image)"),
    ("img: SuperDiff AND",     "text 'a cat'",           "AND  → 'a cat' text"),
    ("img: SuperDiff AND",     "text 'a dog'",           "AND  → 'a dog' text"),
    ("img: SD monolithic",     "text 'a cat'",           "Mono → 'a cat' text"),
    ("img: SD monolithic",     "text 'a dog'",           "Mono → 'a dog' text"),
]
for a, b, desc in pairs:
    print(f"  {desc:40s}  {sims[idx[a], idx[b]]:.4f}")

# ---------------------------------------------------------------------------
# 5. Plot: PCA scatter + similarity heatmap
# ---------------------------------------------------------------------------
print("\n=== Plotting ===")
pca    = PCA(n_components=2)
coords = pca.fit_transform(all_embs)  # (5, 2)

colors  = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
markers = ["s",       "s",       "s",       "o",       "o"      ]
sizes   = [180,       180,       180,       240,       240      ]

fig, axes = plt.subplots(1, 2, figsize=(17, 7))

# ── Left: PCA scatter ──────────────────────────────────────────────────────
ax = axes[0]
for k in range(len(labels)):
    ax.scatter(coords[k, 0], coords[k, 1],
               c=colors[k], marker=markers[k], s=sizes[k],
               zorder=5, edgecolors="black", linewidths=0.8)
    ax.annotate(labels[k],
                (coords[k, 0], coords[k, 1]),
                textcoords="offset points", xytext=(9, 5), fontsize=8.5)

# Dashed arrows from each image to the composed text embedding
composed_idx = idx["text 'a cat and a dog'"]
for k in [idx["img: SuperDiff AND"], idx["img: SD monolithic"]]:
    ax.annotate("",
                xy=coords[composed_idx], xytext=coords[k],
                arrowprops=dict(arrowstyle="->", color=colors[k],
                                lw=1.6, linestyle="dashed"))

ax.set_title("CLIP Embedding Space — PCA 2D Projection", fontsize=11, fontweight="bold")
ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.grid(True, alpha=0.3)
ax.legend(handles=[
    mpatches.Patch(color="steelblue", label="■  Text embeddings"),
    mpatches.Patch(color="darkorchid",label="●  Image embeddings (CLIP)"),
], fontsize=9, loc="best")

# ── Right: cosine similarity heatmap ───────────────────────────────────────
ax2 = axes[1]
short = ["cat\n(txt)", "dog\n(txt)", "cat+dog\n(txt)", "AND\n(img)", "mono\n(img)"]
im  = ax2.imshow(sims, cmap="RdYlGn", vmin=0.0, vmax=1.0)
ax2.set_xticks(range(5)); ax2.set_xticklabels(short, fontsize=9)
ax2.set_yticks(range(5)); ax2.set_yticklabels(short, fontsize=9)
for i in range(5):
    for j in range(5):
        ax2.text(j, i, f"{sims[i,j]:.3f}",
                 ha="center", va="center", fontsize=9,
                 color="black" if 0.25 < sims[i,j] < 0.9 else "white")
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
ax2.set_title("Cosine Similarity Matrix (CLIP space)", fontsize=11, fontweight="bold")

plt.suptitle(
    f"Composability Gap  |  SuperDiff AND vs SD Monolithic\n"
    f"SD v1.4 · seed={SEED} · steps={STEPS} · guidance={SCALE}",
    fontsize=10, y=1.01
)
plt.tight_layout()
out_plot = OUT_DIR / "composability_gap_clip.png"
plt.savefig(out_plot, dpi=150, bbox_inches="tight")
print(f"Saved plot → {out_plot}")

# ---------------------------------------------------------------------------
# 6. SD-IPC projection (image → text-encoder hidden space) — for reference
# ---------------------------------------------------------------------------
print("\n=== SD-IPC projection (text-encoder hidden space) ===")
with torch.no_grad():
    proj_and  = sdipc_project(img_and ).cpu().numpy()   # (1, 768)
    proj_mono = sdipc_project(img_mono).cpu().numpy()

    # direct text-encoder EOS token embedding (the one SD-IPC was designed to match)
    text_encoder.to(device)
    ids_catdog = tokenizer(
        MONO, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    ).input_ids.to(device)
    eos_emb = text_encoder(ids_catdog)[0][:, -1, :].float().cpu().numpy()  # (1, 768) — EOS position
    text_encoder.cpu()

    # Cosine sim in text-encoder space (single-vector comparison)
    def cos(a, b):
        a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
        return float(np.dot(a.flatten(), b.flatten()))

print(f"  SD-IPC(AND)  vs EOS('a cat and a dog'): {cos(proj_and,  eos_emb):.4f}")
print(f"  SD-IPC(mono) vs EOS('a cat and a dog'): {cos(proj_mono, eos_emb):.4f}")
print(f"  SD-IPC(AND)  vs SD-IPC(mono)          : {cos(proj_and,  proj_mono):.4f}")

print("\nDone.")
