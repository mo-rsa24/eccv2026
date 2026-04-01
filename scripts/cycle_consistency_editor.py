"""
cycle_consistency_editor.py

Implements Steps 1 and 2 of EDITOR (Liu et al., arXiv:2506.03067), Algorithm 1.
Step 3 (Embedding Inversion / E2T) is intentionally omitted.

The output is an optimized contextual embedding c* ∈ R^{1×77×768} that can be
passed directly into SD as conditioning — finding the best point in SD's
conditioning space that regenerates the target image.

Algorithm (faithful to paper):
  ① Initialization
       p  ← BLIP_caption(x)                          # warm-start
       c  ← text_encoder(p).last_hidden_state         # (1, 77, 768)  ← AFTER transformer
  ② Reverse-engineering  (fixed noise n, fixed seed)
       for e = 1..max_epoch:
           x̂   = D(R_{ε_θ}(c, n))                   # regen: denoise c with n, then VAE-decode
           loss = L(x̂, x)                            # L2 + LPIPS reconstruction loss
           c   ← c − lr · ∇_c loss                   # gradient flows back through denoising loop
  (③ Embedding Inversion skipped — return c* as-is)

Key faithfulness details vs EDITOR figure:
  - c is the output of the text encoder transformer, NOT the token embeddings (Fig. 2b)
  - n (noise) is fixed for all optimization steps (same seed each epoch)
  - Gradient backpropagates through every UNet call via gradient checkpointing
    (avoids 40 GB graph storage while preserving correctness)

Usage:
    conda activate attend_excite
    pip install lpips -q        # recommended; falls back to L2-only if absent
    python scripts/cycle_consistency_editor.py
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
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import (
    CLIPTextModel, CLIPTokenizer,
    CLIPModel, CLIPProcessor,
    BlipProcessor, BlipForConditionalGeneration,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID  = "CompVis/stable-diffusion-v1-4"
CLIP_ID   = "openai/clip-vit-large-patch14"
BLIP_ID   = "Salesforce/blip-image-captioning-large"
OUT_DIR   = Path("experiments/composability_gap")
SEED      = 42
GEN_STEPS = 50     # denoising steps for final generation (quality)
OPT_STEPS = 20     # denoising steps inside the optimization loop (memory vs quality trade-off)
                   # paper uses full steps; reduce here if OOM (20 is a good balance)
N_OPT     = 200    # optimization epochs (Algorithm 1: max_epoch)
LR        = 5e-3   # Adam learning rate
SCALE     = 7.5    # CFG guidance scale

AND_IMG_PATH = OUT_DIR / "superdiff_and_cat_dog.png"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# LPIPS (optional — falls back to pure L2 if not installed)
# ---------------------------------------------------------------------------
try:
    import lpips
    loss_fn_lpips = lpips.LPIPS(net="vgg").to(device).eval()
    USE_LPIPS = True
    print("LPIPS available — using L2 + LPIPS loss")
except ImportError:
    USE_LPIPS = False
    print("LPIPS not found — using L2-only loss (pip install lpips for better results)")


def recon_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction loss L(x̂, x) from Algorithm 1.
    pred / target: (1, 3, H, W) in [-1, 1]  (VAE output range)
    """
    l2 = F.mse_loss(pred, target)
    if USE_LPIPS:
        lp = loss_fn_lpips(pred.clamp(-1, 1), target.clamp(-1, 1)).mean()
        return l2 + 0.5 * lp
    return l2


# ---------------------------------------------------------------------------
# Load SD v1.4 components
# ---------------------------------------------------------------------------
print("Loading SD v1.4 (fp16) ...")
vae          = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae",
               use_safetensors=True, torch_dtype=torch.float16).to(device)
tokenizer    = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder",
               torch_dtype=torch.float16).to(device)
unet         = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet",
               use_safetensors=True, torch_dtype=torch.float16).to(device)
# Use DDIM for guaranteed differentiability of the denoising step formula
scheduler    = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
scheduler.set_timesteps(GEN_STEPS)
vae.eval(); text_encoder.eval(); unet.eval()

# CLIP for similarity scoring (CPU to save VRAM during optimization)
print("Loading CLIP ViT-L/14 (CPU) ...")
clip_model     = CLIPModel.from_pretrained(CLIP_ID)
clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)
clip_model.eval()

# BLIP for caption initialization (CPU, offloaded after use)
print("Loading BLIP-large (CPU) ...")
blip_proc  = BlipProcessor.from_pretrained(BLIP_ID)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_ID)
blip_model.eval()

# Pre-compute projection matrices for SD-IPC comparison
with torch.no_grad():
    inv_text    = torch.linalg.pinv(clip_model.text_projection.weight.float(), atol=0.3)
    visual_proj = clip_model.visual_projection.weight.float()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def sd_text_emb(prompt: str) -> torch.Tensor:
    """(1, 77, 768) fp16 cross-attention conditioning."""
    ids = tokenizer(prompt, padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").input_ids.to(device)
    return text_encoder(ids).last_hidden_state


@torch.no_grad()
def caption_image(image: Image.Image) -> str:
    """BLIP caption for warm-start initialization (Algorithm 1, Step ①)."""
    blip_model.to(device)
    inputs = blip_proc(images=image, return_tensors="pt").to(device)
    out    = blip_model.generate(**inputs, max_new_tokens=50)
    cap    = blip_proc.decode(out[0], skip_special_tokens=True)
    blip_model.cpu()
    torch.cuda.empty_cache()
    return cap


def init_c_from_caption(caption: str) -> torch.Tensor:
    """
    Initialize c as text_encoder(caption).last_hidden_state — faithful to EDITOR Fig. 2b:
    c is the contextual embedding AFTER the transformer, not the token embeddings.
    Returns (1, 77, 768) float32, detached (optimization variable).
    """
    with torch.no_grad():
        ids = tokenizer(caption, padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").input_ids.to(device)
        c = text_encoder(ids).last_hidden_state  # (1, 77, 768) fp16
    return c.detach().float()  # float32 for optimizer stability


# ---------------------------------------------------------------------------
# Differentiable denoising loop (Algorithm 1, Step ②)
# ---------------------------------------------------------------------------
def _unet_checkpointed(inp, t, enc_hs):
    """
    UNet forward with gradient checkpointing.
    Recomputes activations during backward instead of storing them.
    Reduces memory from ~40 GB (20 steps × activations) to <2 GB.
    enc_hs must be float16 (UNet dtype); gradient flows through it.
    """
    def _fwd(inp, t, enc_hs):
        return unet(inp, t, encoder_hidden_states=enc_hs).sample
    return grad_checkpoint(_fwd, inp, t, enc_hs, use_reentrant=False)


def denoise_with_grad(c: torch.Tensor, fixed_latents: torch.Tensor,
                      n_steps: int) -> torch.Tensor:
    """
    Runs the full CFG denoising loop, keeping gradient w.r.t. c.

    c:             (1, 77, 768) float32  — optimization variable, requires_grad
    fixed_latents: (1, 4, 64, 64) float32 — fixed noise n (same every epoch)
    Returns final latents (1, 4, 64, 64) float32, attached to c's graph.
    """
    scheduler.set_timesteps(n_steps)
    latents = fixed_latents.clone()  # start from fixed noise each epoch

    uncond = sd_text_emb("").detach()  # (1, 77, 768) fp16, no grad

    for t in scheduler.timesteps:
        t_batch = t.unsqueeze(0).to(device)
        inp     = scheduler.scale_model_input(latents.half().detach(), t_batch)

        # Gradient flows through noise_c → latents → ... → c
        c_half  = c.half()  # cast inside loop so graph traces through c
        noise_c = _unet_checkpointed(inp, t_batch, c_half).float()

        with torch.no_grad():
            noise_u = unet(inp, t_batch, encoder_hidden_states=uncond).sample.float()

        noise_pred = noise_u + SCALE * (noise_c - noise_u)

        # DDIM step — pure tensor arithmetic, fully differentiable
        alpha_prod_t     = scheduler.alphas_cumprod[t].to(device).float()
        alpha_prod_t_prev = (
            scheduler.alphas_cumprod[scheduler.timesteps[
                (scheduler.timesteps == t).nonzero(as_tuple=True)[0][0] - 1
            ]].to(device).float()
            if t > scheduler.timesteps[-1]
            else torch.tensor(1.0, device=device)
        )
        beta_prod_t  = 1 - alpha_prod_t
        pred_x0      = (latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
        pred_x0      = pred_x0.clamp(-1, 1)
        latents      = (alpha_prod_t_prev.sqrt() * pred_x0
                        + (1 - alpha_prod_t_prev).sqrt() * noise_pred)

    return latents  # (1, 4, 64, 64) float32, gradient w.r.t. c intact


@torch.no_grad()
def decode(latents: torch.Tensor) -> Image.Image:
    x = vae.decode(latents.half() / vae.config.scaling_factor, return_dict=False)[0]
    x = (x / 2 + 0.5).clamp(0, 1)
    return Image.fromarray(
        (x.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()[0]
    )


def decode_tensor(latents: torch.Tensor) -> torch.Tensor:
    """Decode latents → (1, 3, H, W) in [-1, 1], keeping gradient graph."""
    return vae.decode(latents.half() / vae.config.scaling_factor, return_dict=False)[0].float()


# ---------------------------------------------------------------------------
# EDITOR optimization (Algorithm 1, Steps ① + ②)
# ---------------------------------------------------------------------------
def editor_invert(target_image: Image.Image,
                  n_opt: int = N_OPT,
                  lr: float = LR,
                  opt_steps: int = OPT_STEPS,
                  seed: int = SEED) -> tuple[torch.Tensor, list[float]]:
    """
    Returns:
        c_opt: (1, 77, 768) float32 — optimized contextual embedding
        losses: list of per-epoch loss values
    """
    # ① Caption → init c  (contextual embedding after transformer)
    cap = caption_image(target_image)
    print(f"    BLIP caption: '{cap}'")
    c   = init_c_from_caption(cap).requires_grad_(True)          # (1, 77, 768) float32

    # Fixed noise n (same every epoch — Algorithm 1 uses fixed n)
    gen            = torch.Generator(device=device).manual_seed(seed)
    fixed_latents  = torch.randn((1, unet.config.in_channels, 64, 64),
                                 generator=gen, device=device, dtype=torch.float32)
    fixed_latents  = fixed_latents * scheduler.init_noise_sigma

    # Target image as tensor in [-1, 1]
    from torchvision import transforms
    to_t    = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),                    # [0, 1]
        transforms.Lambda(lambda x: x * 2 - 1),  # [-1, 1]
    ])
    target_t = to_t(target_image).unsqueeze(0).to(device)  # (1, 3, 512, 512)

    optimizer = torch.optim.Adam([c], lr=lr)
    losses    = []

    # ② Reverse-engineering loop
    print(f"    Optimizing c* over {n_opt} epochs × {opt_steps} denoising steps ...")
    for epoch in range(n_opt):
        optimizer.zero_grad()

        # Regenerate: D(R_{ε_θ}(c, n))
        latents_opt = denoise_with_grad(c, fixed_latents, n_steps=opt_steps)  # grad w.r.t. c
        gen_t       = decode_tensor(latents_opt)                               # (1,3,512,512) [-1,1]

        loss = recon_loss(gen_t, target_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 25 == 0 or epoch == n_opt - 1:
            print(f"      epoch {epoch:3d}/{n_opt}  loss={loss.item():.5f}")

    return c.detach(), losses


# ---------------------------------------------------------------------------
# CLIP similarity helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def clip_image_emb(image: Image.Image) -> torch.Tensor:
    clip_model.to(device)
    pv = clip_processor(images=image, return_tensors="pt").pixel_values.to(device)
    e  = clip_model.visual_projection(
             clip_model.vision_model(pixel_values=pv).pooler_output
         ).float()
    clip_model.cpu()
    return F.normalize(e, dim=-1)


@torch.no_grad()
def clip_text_emb(prompt: str) -> torch.Tensor:
    clip_model.to(device)
    inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True)
    toks   = {k: v.to(device) for k, v in inputs.items()
              if k in ("input_ids", "attention_mask")}
    e      = clip_model.text_projection(
                 clip_model.text_model(**toks).pooler_output
             ).float()
    clip_model.cpu()
    return F.normalize(e, dim=-1)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float(); b = b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))


# ---------------------------------------------------------------------------
# SD-IPC projection (for side-by-side comparison)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sdipc_project(image: Image.Image) -> torch.Tensor:
    """Returns (1, 77, 768) conditioning sequence via SD-IPC closed-form projection."""
    clip_model.to(device)
    pv         = clip_processor(images=image, return_tensors="pt").pixel_values.to(device)
    pooler_out = clip_model.vision_model(pixel_values=pv).pooler_output.float()
    joint      = pooler_out @ visual_proj.to(device).T
    text_space = joint @ inv_text.to(device).T
    text_space = text_space / text_space.norm(dim=-1, keepdim=True) * 27.5
    clip_model.cpu()
    null_seq   = sd_text_emb("")
    seq        = torch.zeros_like(null_seq)
    seq[:, 0]  = null_seq[:, 0]
    seq[:, 1:] = text_space.to(dtype=seq.dtype).unsqueeze(1)
    return seq


@torch.no_grad()
def run_sd(cond_emb: torch.Tensor, seed: int = SEED) -> Image.Image:
    uncond   = sd_text_emb("")
    cond_emb = cond_emb.to(device=device, dtype=torch.float16)
    uncond   = uncond.to(device=device, dtype=torch.float16)
    scheduler.set_timesteps(GEN_STEPS)
    gen     = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn((1, unet.config.in_channels, 64, 64),
                          generator=gen, device=device, dtype=torch.float16)
    latents = latents * scheduler.init_noise_sigma
    for t in scheduler.timesteps:
        t_b        = t.unsqueeze(0).to(device)
        inp        = scheduler.scale_model_input(latents, t_b)
        noise_c    = unet(inp, t_b, encoder_hidden_states=cond_emb).sample
        noise_u    = unet(inp, t_b, encoder_hidden_states=uncond   ).sample
        noise_pred = noise_u + SCALE * (noise_c - noise_u)
        latents    = scheduler.step(noise_pred, t_b, latents).prev_sample
    return decode(latents)


# ---------------------------------------------------------------------------
# Step 1 — Load / generate source images
# ---------------------------------------------------------------------------
print("\n=== Step 1: Source images ===")
torch.cuda.empty_cache()

sources = {
    "sanity": {"prompt": "a cat lying on the bed"},
    "mono"  : {"prompt": "a cat and a dog"},
}
for name, d in sources.items():
    p    = OUT_DIR / f"cycle_{name}_source.png"
    img  = Image.open(p) if p.exists() else run_sd(sd_text_emb(d["prompt"]))
    if not p.exists():
        img.save(p)
    sources[name]["image"] = img
    print(f"  [{name}] loaded from {p}" if p.exists() else f"  [{name}] generated")

and_img = Image.open(AND_IMG_PATH).convert("RGB")
sources["and"] = {"prompt": "SuperDiff AND: 'a cat' ^ 'a dog'", "image": and_img}

# ---------------------------------------------------------------------------
# Step 2 — EDITOR inversion  (the main experiment)
# ---------------------------------------------------------------------------
print("\n=== Step 2: EDITOR inversion (Steps ① + ② of Algorithm 1) ===")
order = ["sanity", "mono", "and"]

results = {}
for name in order:
    src  = sources[name]["image"]
    print(f"\n  [{name}] {sources[name]['prompt']}")

    c_opt, losses = editor_invert(src)

    # Generate final image at full quality steps from c*
    torch.cuda.empty_cache()
    regen = run_sd(c_opt.half(), seed=SEED)
    regen.save(OUT_DIR / f"editor_cycle_{name}_regen.png")

    # Save loss curve
    np.save(OUT_DIR / f"editor_{name}_losses.npy", np.array(losses))

    results[name] = {
        "prompt" : sources[name]["prompt"],
        "source" : src,
        "regen"  : regen,
        "c_opt"  : c_opt.cpu(),
        "losses" : losses,
    }

# ---------------------------------------------------------------------------
# Step 3 — CLIP similarity
# ---------------------------------------------------------------------------
print("\n=== Step 3: CLIP similarity ===")
for name in order:
    d         = results[name]
    src_emb   = clip_image_emb(d["source"])
    regen_emb = clip_image_emb(d["regen"])
    sim       = cosine_sim(src_emb, regen_emb)
    d["sim"]  = sim
    d["src_emb"]   = src_emb
    d["regen_emb"] = regen_emb
    print(f"  [{name}]  source ↔ regen: {sim:.4f}  "
          f"{'✓ closed' if sim > 0.80 else '✗ broken'}")
    if name == "and":
        print(f"    regen → 'a cat':          {cosine_sim(regen_emb, clip_text_emb('a cat')):.4f}")
        print(f"    regen → 'a dog':          {cosine_sim(regen_emb, clip_text_emb('a dog')):.4f}")
        print(f"    regen → 'a cat and a dog':{cosine_sim(regen_emb, clip_text_emb('a cat and a dog')):.4f}")

# ---------------------------------------------------------------------------
# Step 4 — Side-by-side: EDITOR vs SD-IPC regenerations
# ---------------------------------------------------------------------------
print("\n=== Step 4: EDITOR vs SD-IPC comparison ===")
sdipc_regens = {}
for name in order:
    cond  = sdipc_project(sources[name]["image"])
    regen = run_sd(cond)
    regen.save(OUT_DIR / f"sdipc_cycle_{name}_regen.png")
    sdipc_emb = clip_image_emb(regen)
    sim       = cosine_sim(clip_image_emb(sources[name]["image"]), sdipc_emb)
    sdipc_regens[name] = {"regen": regen, "regen_emb": sdipc_emb, "sim": sim}
    print(f"  [{name}] SD-IPC sim: {sim:.4f}")

# Plot: 3 rows (sanity / mono / AND) × 4 cols (source | EDITOR regen | SD-IPC regen | loss curve)
fig, axes = plt.subplots(3, 4, figsize=(17, 13),
                         gridspec_kw={"width_ratios": [2, 2, 2, 2]})

col_titles = ["Source", "EDITOR regen (c*)", "SD-IPC regen", "Optimization loss"]
for col, t in enumerate(col_titles):
    axes[0, col].set_title(t, fontsize=10, fontweight="bold")

row_labels = {
    "sanity": "A · Sanity\n(should close)",
    "mono"  : "B · Monolithic\n(should close)",
    "and"   : "C · AND hybrid\n(expected to break)",
}

for row, name in enumerate(order):
    d     = results[name]
    sim_e = d["sim"]
    sim_s = sdipc_regens[name]["sim"]
    ce    = "green" if sim_e > 0.80 else "red"
    cs    = "green" if sim_s > 0.80 else "red"

    axes[row, 0].imshow(np.array(d["source"].resize((256, 256))))
    axes[row, 0].axis("off")
    axes[row, 0].set_ylabel(row_labels[name], fontsize=8, labelpad=6)

    axes[row, 1].imshow(np.array(d["regen"].resize((256, 256))))
    axes[row, 1].set_title(f"CLIP={sim_e:.3f}", fontsize=9, color=ce)
    axes[row, 1].axis("off")

    axes[row, 2].imshow(np.array(sdipc_regens[name]["regen"].resize((256, 256))))
    axes[row, 2].set_title(f"CLIP={sim_s:.3f}", fontsize=9, color=cs)
    axes[row, 2].axis("off")

    ax_loss = axes[row, 3]
    ax_loss.plot(d["losses"], lw=1.2, color="#1565C0")
    ax_loss.set_xlabel("epoch", fontsize=8); ax_loss.set_ylabel("loss", fontsize=8)
    ax_loss.tick_params(labelsize=7); ax_loss.grid(True, alpha=0.3)

plt.suptitle(
    "EDITOR (c* optimization) vs SD-IPC (closed-form projection) — Cycle Consistency\n"
    f"Opt: {N_OPT} epochs × {OPT_STEPS} denoise steps, lr={LR}",
    fontsize=11, y=1.01, fontweight="bold"
)
plt.tight_layout()
out = OUT_DIR / "cycle_consistency_editor.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")

# ---------------------------------------------------------------------------
# Step 5 — PCA: source, EDITOR regen, SD-IPC regen
# ---------------------------------------------------------------------------
all_embs  = []
all_labels = []
palette = {
    "sanity_src"   : ("#1565C0", "o"), "sanity_editor": ("#42A5F5", "o"), "sanity_sdipc": ("#BBDEFB", "o"),
    "mono_src"     : ("#2E7D32", "s"), "mono_editor"  : ("#81C784", "s"), "mono_sdipc"  : ("#C8E6C9", "s"),
    "and_src"      : ("#B71C1C", "^"), "and_editor"   : ("#EF9A9A", "^"), "and_sdipc"   : ("#FFCCBC", "^"),
}
for name in order:
    all_embs.append(results[name]["src_emb"].cpu().numpy())
    all_labels.append(f"{name}_src")
    all_embs.append(results[name]["regen_emb"].cpu().numpy())
    all_labels.append(f"{name}_editor")
    all_embs.append(sdipc_regens[name]["regen_emb"].cpu().numpy())
    all_labels.append(f"{name}_sdipc")

coords = PCA(n_components=2).fit_transform(np.vstack(all_embs))

fig2, ax = plt.subplots(figsize=(9, 7))
for k, label in enumerate(all_labels):
    color, marker = palette[label]
    ax.scatter(coords[k, 0], coords[k, 1], c=color, marker=marker,
               s=250, edgecolors="black", linewidths=0.7, zorder=5)
    ax.annotate(label, (coords[k, 0], coords[k, 1]),
                textcoords="offset points", xytext=(7, 4), fontsize=7.5)

legend_handles = [
    mpatches.Patch(color=palette[f"{n}_src"][0],    label=f"{n} source")          for n in order
] + [
    mpatches.Patch(color=palette[f"{n}_editor"][0], label=f"{n} EDITOR regen")    for n in order
] + [
    mpatches.Patch(color=palette[f"{n}_sdipc"][0],  label=f"{n} SD-IPC regen")    for n in order
]
ax.legend(handles=legend_handles, fontsize=7, ncol=3)
ax.set_title("CLIP Embedding Space — Source / EDITOR regen / SD-IPC regen (PCA 2D)", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "cycle_consistency_editor_pca.png", dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_DIR / 'cycle_consistency_editor_pca.png'}")
print("\nDone.")
