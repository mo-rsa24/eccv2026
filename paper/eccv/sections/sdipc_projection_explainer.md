# SD-IPC Projection: A Technical Explainer

## What is SD-IPC?

SD-IPC (Stable Diffusion Image-Prompt Conditioning) is a **closed-form, training-free method**
to convert a CLIP image embedding into a conditioning sequence that can be fed into SD's
cross-attention — the same slot normally occupied by a text prompt embedding.

**Paper**: "The CLIP Model is Secretly an Image-to-Prompt Converter"
Ding, Tian, Ding, Liu — NeurIPS 2023/2024, arXiv:2305.12716.
Zero-shot, closed-form method to convert any image into an SD text conditioning sequence.

---

## The Full Pipeline

```
Input image x
      │
      ▼
┌─────────────────────────────┐
│  CLIP Vision Encoder        │  ViT-L/14 — same model baked into SD 1.4's text encoder
│  vision_model(pixel_values) │
│  → pooler_output            │  (1, 1024)  raw ViT CLS-token features (pre-projection)
└─────────────────────────────┘
      │
      ▼  clip_model.visual_projection  [weight: (768, 1024)]
      │
      ▼
┌─────────────────────────────┐
│  CLIP joint space           │  (1, 768)  L2-normalised — same space as clip_text_emb("...")
│  image_emb = v / ‖v‖        │  This is the "CLIP image embedding" used for retrieval/similarity
└─────────────────────────────┘
      │
      ▼  × 27.5  (scale to match text-encoder EOS token magnitude)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  pinv(text_projection.weight)                        │
│  text_projection: Linear(768→768)  [weight: (768,768)]│
│  pinv(W) : (768, 768)                               │
│  proj_vec = image_emb_scaled @ pinv(W).T            │  (1, 768) in SD text-encoder hidden space
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Build full sequence (1, 77, 768)                   │
│  seq[:, 0]   = null_text_emb[:, 0]  ← BOS token    │
│  seq[:, 1:]  = proj_vec  (broadcast to 76 positions)│
└─────────────────────────────────────────────────────┘
      │
      ▼
Output:  (1, 77, 768)  — drop-in replacement for sd_text_emb("a cat")
```

---

## Supplementary: Why (1, 768) at the intermediate step, not (1, 77, 768)?

This is the most important architectural point to understand.

### Where the single vector comes from

The CLIP vision encoder is a Vision Transformer (ViT). It processes the image as a grid of
patches and produces one hidden vector per patch, plus a special `[CLS]` token at position 0.
The `pooler_output` is just that **single CLS token** — one 1024-dim vector representing the
entire image globally.

```
Input image (224×224)
  → split into 16×16 patches → 196 patch embeddings
  → + 1 [CLS] token prepended
  → 12 or 24 transformer layers
  → CLS token at output = pooler_output → shape (1, 1024)
```

The text encoder, by contrast, produces **one vector per token**:

```
"a cat lying on the bed"
  → tokenise → [BOS, "a", "cat", "lying", "on", "the", "bed", EOS, PAD, ..., PAD]
  → 12 transformer layers
  → output: (1, 77, 768)  ← 77 DIFFERENT vectors, one per token position
```

### What filling all 76 positions with the same vector means

A real text prompt conditioning sequence looks like this (values schematic):

```
Position 0  (BOS):   [ 0.12, -0.34,  0.91, ... ]   ← structural start token
Position 1  ("a"):   [ 0.03,  0.11, -0.02, ... ]   ← article
Position 2  ("cat"): [ 0.78, -0.12,  0.45, ... ]   ← primary subject concept
Position 3  ("on"):  [ 0.02,  0.55,  0.08, ... ]   ← spatial relation
Position 4  ("bed"): [ 0.31,  0.22, -0.67, ... ]   ← secondary object
Position 5  (EOS):   [ 0.88,  0.14,  0.31, ... ]   ← summary / end of meaning
Positions 6-76 (PAD): [ 0.00,  0.00,  0.00, ... ]  ← padding (no information)
```

Each position carries **different** semantic content. The UNet's cross-attention queries
different positions to "look up" different aspects of the scene:
- early layers often attend to subject tokens ("cat")
- mid layers often attend to attribute or spatial tokens ("on", "bed")

SD-IPC's sequence looks like this:

```
Position 0  (BOS):  [ 0.12, -0.34,  0.91, ... ]   ← kept from null prompt
Position 1:         [ v₁,    v₂,    v₃,   ... ]   ← proj_vec (cat+dog mixture)
Position 2:         [ v₁,    v₂,    v₃,   ... ]   ← SAME proj_vec
Position 3:         [ v₁,    v₂,    v₃,   ... ]   ← SAME proj_vec
...
Position 76:        [ v₁,    v₂,    v₃,   ... ]   ← SAME proj_vec
```

Every cross-attention query in the UNet sees **the same vector** regardless of which
position it queries. There is no positional semantic structure. The UNet must collapse
everything it generates into one "what does this image globally look like" signal.

### Concrete impact on composability

For a single object ("a cat"), the global CLS token is dominated by that one concept.
The single-vector conditioning works reasonably — every position says "cat" and the
UNet generates something cat-like.

For a composed image ("cat AND dog"), the CLS token is a weighted average:

```
CLS = α · cat_direction + (1-α) · dog_direction  (in CLIP joint space)
```

Where α is determined by how much visual area each concept occupies and how salient
CLIP finds each one. If the cat is more prominent, α > 0.5 and proj_vec ≈ cat.
If α ≈ 0.5, proj_vec is equidistant — and SD may generate either a chimera or collapse
to one concept, because the UNet was never trained to denoise from a blended conditioning.

This is the structural reason why SD-IPC underestimates cycle consistency for AND hybrids.

---

## Inputs and Preprocessing

| Input | Type | Details |
|-------|------|---------|
| `image` | `PIL.Image` | Any resolution; preprocessed internally |
| Preprocessing | `CLIPProcessor` | Resize to 224×224, normalize to `[-1,1]` with CLIP's mean/std |
| `pixel_values` | `(1, 3, 224, 224)` float32 | Sent to CLIP vision encoder |

No learnable parameters are trained. Everything is computed from frozen CLIP weights.

---

## Output

```
(1, 77, 768)  float16   — conditioning sequence in SD text-encoder hidden-state space
```

- **1**: batch size
- **77**: sequence length (CLIP/SD text encoder maximum token length)
- **768**: hidden dimension of CLIP-L / SD 1.4 text encoder

This is identical in shape and space to `text_encoder("a cat").last_hidden_state`.
It can be passed directly into SD's UNet cross-attention as `encoder_hidden_states`.

---

## Key Assumptions

### 1. CLIP alignment assumption
CLIP is trained so that `visual_projection(image_features) ≈ text_projection(text_features)`
for matching image-text pairs. This means the CLIP joint space is **shared** between modalities.
SD-IPC exploits this: given an image embedding `v` in joint space, the pseudo-inverse
`pinv(text_projection)` maps it back to the text-encoder hidden-state that would produce `v`.

Formally: if `T(h) = v` (text encoder hidden → joint), then `pinv(T)(v) ≈ h`.

**Worked example — the alignment in practice:**

```python
# These two embeddings end up near each other in joint space
v_text  = clip_text_emb("a cat")          # (1, 768), from text encoder path
v_image = clip_image_emb(photo_of_cat)    # (1, 768), from vision encoder path

cosine_sim(v_text, v_image) ≈ 0.28        # not 1.0, but directionally aligned

# The alignment is what makes the pinv meaningful:
# text_projection maps (1,768) hidden → (1,768) joint
# pinv(text_projection) maps (1,768) joint → (1,768) hidden
# If v_image ≈ text_projection(h_cat), then pinv(text_projection)(v_image) ≈ h_cat

h_via_text  = text_encoder("a cat").last_hidden_state[:, -1, :]  # EOS hidden (1, 768)
h_via_image = v_image_scaled @ pinv(text_projection).T           # projected (1, 768)
cosine_sim(h_via_text, h_via_image) ≈ 0.60–0.75  # similar direction, not identical
```

CLIP alignment is approximate (~0.28 cosine sim), so the pinv projection is also
approximate. The regen looks "cat-like" but not identical to the source.

**When this holds**: for concepts well-represented in CLIP's training distribution.
**When it breaks**: for rare, compositional, or abstract content that CLIP can't compress
into a single 768-dim vector.

### 2. Single-vector bottleneck assumption
The entire image's semantics are compressed into a **single 768-dim pooled vector**
(the ViT CLS token). All 76 non-BOS positions in the output sequence are filled with
the same projected vector — there is no spatial or per-token diversity.

Contrast with a text prompt embedding: each of the 77 tokens carries different information
(subject, attribute, spatial relation, etc.). SD-IPC collapses all of this into one vector.

**Worked example — what is lost in the bottleneck:**

```
Prompt:   "a cat lying on the bed"
Tokens:   [BOS] [a] [cat] [lying] [on] [the] [bed] [EOS] [PAD×69]

text_encoder output (1, 77, 768):
  dim 0 → encodes: start-of-sequence structure
  dim 2 → encodes: feline concept, fur texture, domestic animal
  dim 3 → encodes: horizontal/resting pose
  dim 6 → encodes: furniture, rectangular surface, bedroom context
  dims 8-76 → near-zero padding vectors

CLIP image embedding of the same scene (1, 768):
  → CLS token averages over: cat patches (~40% of image area)
                              bed patches (~50% of image area)
                              background (~10%)
  → Result: weighted mean pulled toward "bed" (larger area)
            "cat" and "bed" both partially represented
            "lying", "on" (spatial relation) LOST — no geometry in CLS

After projection to text-encoder space and broadcast (1, 77, 768):
  ALL 77 positions = same vector ≈ 0.6·cat + 0.4·bed (schematic)
  No position distinguishes cat from bed from their relationship.
```

**Consequence**: SD regen from the projected embedding may produce a cat OR a bed OR
a cat on a bed — but not reliably, and the spatial relationship "lying on" is not encoded.
In practice, CLIP's CLS token is biased toward the most distinctive visual concept (usually
the subject), so the cat dominates and the bed is often dropped in the regen.

### 3. Scale calibration for SD 1.4
The factor `27.5` was empirically calibrated for SD 1.4's text encoder — it matches the
L2 norm of the EOS token in typical prompt embeddings. This scale is **not transferable**
to other models (SD 2.x, SDXL, SD3) without recalibration.

For SDXL, the same factor is applied to both the CLIP-L and CLIP-G branches, but its
validity for CLIP-G's 1280-dim space is unverified and likely suboptimal.

**Worked example — why scale matters:**

```python
# After pinv projection, the vector is unit-normalised (‖proj_vec‖ = 1.0)
# But SD's cross-attention was trained with text embeddings of much larger magnitude:

text_emb = text_encoder("a cat").last_hidden_state  # (1, 77, 768)
eos_norm  = text_emb[0, -1].norm()                  # ≈ 27–28 for typical prompts
pad_norm  = text_emb[0, -2].norm()                  # ≈ 1–3 for padding tokens

# If we pass proj_vec with ‖proj_vec‖ = 1.0 (unnormalized):
#   → UNet cross-attention sees tiny activations
#   → conditioning is effectively ignored → image looks like unconditioned noise

# With × 27.5:
#   → ‖proj_vec‖ ≈ 27.5, matching the EOS token magnitude
#   → UNet cross-attention responds as it would to a real text prompt
```

This is why omitting the scale factor (or using the wrong scale) produces incoherent images
even though the direction of the vector in embedding space is correct.

### 4. BOS token preservation
Position 0 is always taken from the null prompt (`sd_text_emb("")[:, 0]`). This is
necessary because the BOS (beginning-of-sequence) token provides structural context
that SD's cross-attention expects. Overwriting it degrades generation quality severely.

---

## Why 77 × 768?

SD 1.4's UNet cross-attention layers were trained with the **full hidden-state sequence**
from CLIP-L's text encoder:
- 77 positions = CLIP's maximum tokenisation length (1 BOS + 75 tokens + 1 EOS)
- 768 = CLIP-L hidden dimension

The cross-attention keys and values are learned projections of this 77×768 matrix.
Passing a 768-dim vector directly (without expanding to 77×768) would fail
dimensionally. SD-IPC broadcasts the single projected vector across all 77 positions
to satisfy this interface, trading spatial richness for compatibility.

---

## What Can We Use SD-IPC For?

| Use case | Suitability | Notes |
|----------|-------------|-------|
| Regenerating a single-concept image | ✅ Good | Cat → projected emb → SD gen ≈ cat |
| CLIP similarity measurement (source ↔ regen) | ✅ Good | Use CLIP image embeddings directly, not the projected seq |
| Cycle consistency experiment | ⚠️ Partial | See detailed analysis below |
| Representing composed images (cat + dog) | ❌ Poor | Pooling discards the minority concept |
| Spatial / relational content | ❌ Poor | "cat on left, dog on right" → one pooled vector loses layout |
| Image editing / inpainting | ❌ Not designed for | Use textual inversion or IP-Adapter instead |

---

## Reliability for the Composability Gap

The composability gap is defined as the difference in cycle consistency between:
- A monolithic SD image (`"a cat and a dog"`) → should cycle-close
- A SuperDiff AND hybrid → should fail to cycle-close

### How SD-IPC handles composed images

When the input is a **SuperDiff AND hybrid** (e.g., cat + dog):

```
AND hybrid image
      │
      ▼  CLIP ViT-L/14 CLS token
      │
      ▼  visual_projection
      │
    (1, 768) in CLIP joint space
      │
      │  ← This single vector must represent BOTH cat and dog
      │  ← CLIP's pooler takes a weighted average over patch tokens
      │  ← The dominant concept (e.g., cat) will dominate the CLS token
      │
      ▼  pinv(text_projection) × 27.5
      │
    (1, 77, 768) — all 76 non-BOS positions identical
      │
      ▼  SD UNet generates...
      │
    Image of whichever concept dominated the CLIP pooler
```

**The gap is real but partially artificially amplified** by SD-IPC's bottleneck:

1. **True composability gap component**: Even if CLIP could perfectly encode both concepts,
   SD's text cross-attention was trained on single-concept or simply-composed text prompts.
   The AND hybrid occupies a region of conditioning space SD was not trained to denoise from.

2. **SD-IPC artefact component**: The pooler already discards the minority concept *before*
   SD ever sees the conditioning. The gap is therefore larger than the "true" gap.

**Consequence for your paper**: SD-IPC measures a **lower bound** on cycle consistency for
composed images. The true cycle consistency (with a richer embedding method like EDITOR)
would be higher, but still below the monolithic baseline if the gap is structural.

### Quantitative reliability

| Experiment | SD-IPC reliable? | Reason |
|------------|-----------------|--------|
| Sanity (single concept) | ✅ Yes | Single pooled vector sufficient |
| Monolithic "cat and dog" | ⚠️ Partial | CLIP may represent both if prompt is short |
| AND hybrid | ❌ Limited | Pooler collapses dual-concept to dominant one |
| Multi-object spatial layout | ❌ No | Spatial info lost in CLS token |

---

## Is SD-IPC Good for Cycle Consistency?

### Strengths
- **Fast**: closed-form, no optimisation, <100ms per image
- **Consistent**: deterministic given the same image and seed
- **Comparable**: all three experiments (sanity / mono / AND) go through the
  same projection, so differences in cycle sim are meaningful *relative to each other*

### Weaknesses
- **Semantically lossy**: minor/secondary objects often vanish in the regen
  (e.g., "cat lying on the bed" → cat survives, bed may not)
- **No spatial encoding**: layout information is completely absent
- **Scale not validated for non-SD1.4**: results on SDXL / SD3 are heuristic

### What cycle consistency via SD-IPC actually measures

```
High CLIP sim (source ↔ regen)
    → The image's dominant concept survives projection and regeneration
    → Does NOT mean full semantic content was preserved

Low CLIP sim (AND hybrid source ↔ regen)
    → The composed representation is not recoverable via this bottleneck
    → Consistent with composability gap, but causation is ambiguous
```

### Comparison with EDITOR for cycle consistency

| Property | SD-IPC | EDITOR (c* optimised) |
|----------|--------|----------------------|
| Embedding richness | Single 768-dim pooled vector | Full 77×768 contextual sequence |
| Spatial/relational info | ❌ Lost | ✅ Partially preserved |
| Minority concept in AND | ❌ Often lost | ✅ Better preserved |
| Speed | Fast (closed-form) | Slow (200 opt epochs × UNet passes) |
| Faithfulness to SD's conditioning space | Approximate | Near-optimal (gradient-optimised) |
| Cycle sim expected for AND hybrid | Lower (artefact + true gap) | Higher but still < mono (true gap only) |

**Key claim for the paper**: If EDITOR also shows a cycle-consistency gap for AND hybrids
(lower sim than for monolithic prompts), then the gap is structural — it exists in SD's
conditioning space itself, not just in SD-IPC's bottleneck.

---

## Summary

SD-IPC is a **convenient, fast proxy** for measuring whether an image's content is
representable in SD's conditioning space. For single-concept images it works well.
For composed (AND hybrid) images it **underestimates** cycle consistency because the
pooled CLIP vector is a lossy summary. This means:

- SD-IPC cycle-consistency results are **necessary but not sufficient** evidence of
  the composability gap.
- To strengthen the claim, pair SD-IPC results with EDITOR results:
  - If EDITOR also fails to cycle-close the AND hybrid, the gap is in SD's conditioning
    manifold itself — the most powerful version of the composability gap argument.
  - If EDITOR succeeds where SD-IPC fails, the gap is in the projection method, not SD.
