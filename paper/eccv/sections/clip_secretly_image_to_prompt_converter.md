# The CLIP Model is Secretly an Image-to-Prompt Converter
## A Pedagogical Scaffold for Understanding SD-IPC

**Paper**: "The CLIP Model is Secretly an Image-to-Prompt Converter"
**Authors**: Yuxuan Ding, Chunna Tian, Haoxuan Ding, Lingqiao Liu
**Venue**: NeurIPS 2023/2024
**arXiv**: 2305.12716

---

## 1. What Problem Does This Paper Solve?

Stable Diffusion (SD) is a text-to-image model. Its UNet denoiser expects a **text
conditioning sequence** — a $(77 \times 768)$ matrix produced by CLIP's text encoder from a
written prompt. If you want to condition generation on an *image* instead of text, you
face a shape and space mismatch:

```
What the UNet expects:   (1, 77, 768) from text_encoder("a cat")
What you have:           a PIL Image
```

Existing workarounds at the time of the paper:
- **Textual Inversion** — optimise a pseudo-token for many steps (slow, requires training)
- **DreamBooth** — fine-tune the whole model (very slow, requires GPU)
- **CLIP re-scoring** — use CLIP cosine similarity, not regeneration (doesn't produce images)

**SD-IPC's answer**: there is already a closed-form, zero-shot bridge between image
embeddings and text conditioning — it comes from the shared structure of CLIP itself.
No training required. One matrix multiply and a pseudo-inverse.

---

## 2. Background: Two Models That Share a Space

Understanding SD-IPC requires understanding one key fact about CLIP.

### 2.1 How CLIP Was Trained

CLIP was trained with a symmetric contrastive objective. Given a batch of $N$ image-text
pairs $\{(x_i, t_i)\}_{i=1}^N$, let $\mathbf{v}_i^I$ and $\mathbf{v}_i^T$ be the
normalised image and text embeddings respectively. The InfoNCE loss is:

$$
\mathcal{L}_\text{CLIP} = -\frac{1}{2N} \sum_{i=1}^{N}
\left[
  \log \frac{\exp(\mathbf{v}_i^I \cdot \mathbf{v}_i^T / \tau)}
            {\sum_{j=1}^{N} \exp(\mathbf{v}_i^I \cdot \mathbf{v}_j^T / \tau)}
  +
  \log \frac{\exp(\mathbf{v}_i^T \cdot \mathbf{v}_i^I / \tau)}
            {\sum_{j=1}^{N} \exp(\mathbf{v}_i^T \cdot \mathbf{v}_j^I / \tau)}
\right]
$$

where $\tau$ is a learned temperature. The contrastive objective pushes matched pairs
together and unmatched pairs apart in a shared embedding space.

```
CLIP Training:
  image  →  Vision Encoder  →  W_V  →  v^I  ┐
  text   →  Text Encoder    →  W_T  →  v^T  ┘
                              ↑
                    L_CLIP: v^I ≈ v^T for matched pairs
```

The key outcome: both encoders project into a **shared joint space** $\mathcal{V} \subset \mathbb{R}^{768}$
where semantically matched images and texts are nearby.

**Answer to "How was SD-IPC trained?"**: It was not trained at all. SD-IPC is a
closed-form mathematical consequence of CLIP's joint space. The only learned components
are CLIP's weights (frozen) and SD's weights (frozen). No gradient updates happen.

### 2.2 The Joint Space Geometry

Let $f_I : \mathbb{R}^{3 \times 224 \times 224} \to \mathbb{R}^{1024}$ be the ViT vision
encoder and $f_T : \mathbb{R}^{77} \to \mathbb{R}^{768}$ be the CLIP text encoder. The
projection matrices are:

$$
\mathbf{W}_V \in \mathbb{R}^{768 \times 1024}, \qquad \mathbf{W}_T \in \mathbb{R}^{768 \times 768}
$$

The normalised embeddings in the joint space are:

$$
\mathbf{v}^I = \frac{\mathbf{W}_V \, f_I(x)}{\|\mathbf{W}_V \, f_I(x)\|_2}, \qquad
\mathbf{v}^T = \frac{\mathbf{W}_T \, f_T(t)}{\|\mathbf{W}_T \, f_T(t)\|_2}
$$

After training, for a matched image-text pair:
$\cos(\mathbf{v}^I, \mathbf{v}^T) \approx 0.28$ — aligned but not equal.

```
Joint space (768-dim):
                          × "tabby cat"
           × "a cat"
                      ×  [photo of cat]
                                         × "a dog"
     × "automobile"                              × [photo of dog]
```

The alignment is **approximate** — not perfect — but sufficient for the pseudo-inverse
trick to work, as detailed in §4.

---

## 3. Step-by-Step: What Happens to Your Input

### 3.1 Input

```
Input:  PIL.Image of any resolution
        e.g., a 512×512 RGB image
```

No text prompt is needed. No captions. Just the image.

### 3.2 Preprocessing (CLIPProcessor)

Each pixel channel $c \in \{R, G, B\}$ is normalised as:

$$
\tilde{x}_c = \frac{x_c - \mu_c}{\sigma_c}
$$

with CLIP's fixed per-channel constants:
$\boldsymbol{\mu} = [0.48145,\ 0.45782,\ 0.40821]$,
$\boldsymbol{\sigma} = [0.26862,\ 0.26130,\ 0.27577]$.

```
PIL.Image (any H×W×3)
      │
      ▼  Resize to 224×224 (bicubic) → per-channel normalize
      │
pixel_values:  (1, 3, 224, 224)  float32
```

### 3.3 Vision Encoder — ViT-L/14

The image is divided into $P = 196$ non-overlapping $16 \times 16$ patches. Each patch $p_k \in \mathbb{R}^{768}$
is obtained by flattening and linearly projecting pixel values. A learnable $[\text{CLS}]$
token $\mathbf{e}_0 \in \mathbb{R}^{768}$ is prepended, and positional embeddings
$\mathbf{pos}_k$ are added:

$$
\mathbf{z}_k^{(0)} = p_k + \mathbf{pos}_k, \quad k = 0, \ldots, 196
$$

The sequence $[\mathbf{z}_0^{(0)}, \ldots, \mathbf{z}_{196}^{(0)}]$ passes through $L = 24$
transformer layers. Each layer $\ell$ computes:

$$
\mathbf{Z}^{(\ell)} = \text{MLP}\!\left(\text{LN}\!\left(\mathbf{Z}^{(\ell-1)} + \text{MHSA}\!\left(\text{LN}(\mathbf{Z}^{(\ell-1)})\right)\right)\right)
$$

where MHSA is multi-head self-attention and LN is layer normalisation. The CLS token
at the final layer is taken as the global image representation:

$$
\mathbf{p} = \text{LN}\!\left(\mathbf{z}_0^{(L)}\right) \in \mathbb{R}^{1024}
$$

```
pixel_values: (1, 3, 224, 224)
      │
      ▼  196 patch embeddings + [CLS] → 24 transformer layers → CLS output
      │
pooler_output  p:  (1, 1024)  float32
```

**Shape at this point**: $(1, 1024)$ — the raw ViT hidden space, not yet in CLIP's joint space.

### 3.4 Visual Projection → CLIP Joint Space

$$
\mathbf{v}^I_\text{raw} = \mathbf{W}_V \, \mathbf{p} \in \mathbb{R}^{768}, \qquad
\mathbf{v}^I = \frac{\mathbf{v}^I_\text{raw}}{\|\mathbf{v}^I_\text{raw}\|_2}
$$

where $\mathbf{W}_V \in \mathbb{R}^{768 \times 1024}$ is `clip_model.visual_projection.weight`.

```
pooler_output p: (1, 1024)
      │
      ▼  W_V ∈ ℝ^{768×1024}  (visual_projection)   FROZEN
      │
      ▼  L2 normalise
      │
image_emb  v^I:  (1, 768)  unit vector
```

**Shape at this point**: $(1, 768)$ — the standard CLIP image embedding, in the shared
joint space $\mathcal{V}$.

**Answer to "Is this a CLIP embedding?"**: Yes. $\mathbf{v}^I$ is exactly the embedding
returned by `clip_model.get_image_features(...)`, used for retrieval and zero-shot
classification. SD-IPC takes this and goes further.

### 3.5 Scale to Match SD Text Embedding Magnitude

After normalisation, $\|\mathbf{v}^I\|_2 = 1$. But SD's UNet cross-attention was trained
with text embeddings of much larger magnitude. Empirically, the EOS token of SD 1.4's
text encoder satisfies:

$$
\|\mathbf{h}_\text{EOS}\|_2 \approx 27\text{–}28 \quad \text{for typical prompts}
$$

SD-IPC rescales the image embedding to match:

$$
\tilde{\mathbf{v}}^I = \lambda \cdot \mathbf{v}^I, \qquad \lambda = 27.5
$$

```
image_emb v^I: (1, 768)  ‖v‖ = 1.0
      │
      ▼  × 27.5
      │
image_emb_scaled  ṽ^I:  (1, 768)  ‖v‖ ≈ 27.5
```

**Why $\lambda = 27.5$?** Cross-attention keys and values are linear projections of the
conditioning sequence. If $\|\mathbf{h}\|_2 \approx 1$, the resulting attention logits
are far smaller than those produced by real text embeddings — the conditioning is
effectively ignored. Matching the EOS norm restores appropriate attention activation.

```python
# Verification
emb = text_encoder("a cat").last_hidden_state  # (1, 77, 768)
emb[0, -1].norm()   # EOS token: ≈ 27.3
emb[0, -2].norm()   # PAD token: ≈ 1.8
```

**$\lambda$ is model-specific** — calibrated for SD 1.4. Not valid for SDXL or SD3 without recalibration.

### 3.6 Pseudo-Inverse Projection → SD Text-Encoder Hidden Space

This is the theoretical core of the paper. Full derivation in §4.

$$
\hat{\mathbf{h}} = \mathbf{W}_T^+ \, \tilde{\mathbf{v}}^I \in \mathbb{R}^{768}
$$

where $\mathbf{W}_T^+ = \texttt{pinv}(\mathbf{W}_T,\ \texttt{atol}=0.3)$ is the
Moore-Penrose pseudo-inverse of the text projection weight.

```
image_emb_scaled ṽ^I: (1, 768)  in CLIP joint space
      │
      ▼  W_T^+ ∈ ℝ^{768×768}  (pinv of text_projection)   computed once
      │
proj_vec  ĥ:  (1, 768)  in SD text-encoder hidden space
```

**Shape at this point**: $(1, 768)$ — now in the space that SD's UNet cross-attention
was trained to receive as hidden states.

### 3.7 Build the Full Conditioning Sequence

The SD UNet requires $\mathbf{C} \in \mathbb{R}^{77 \times 768}$. SD-IPC constructs it as:

$$
\mathbf{C}_{[k]} = \begin{cases}
  \mathbf{h}_\text{BOS} & k = 0 \\
  \hat{\mathbf{h}} & k = 1, \ldots, 76
\end{cases}
$$

where $\mathbf{h}_\text{BOS} = \texttt{text\_encoder}(\text{``"})_{[0]}$ is the BOS
hidden state from the null prompt.

```
proj_vec ĥ: (1, 768)
      │
      ▼  prepend h_BOS at position 0
      ▼  broadcast ĥ to positions 1–76
      │
encoder_hidden_states  C:  (1, 77, 768)  float16
```

**Output shape**: $(1, 77, 768)$ — identical interface to `text_encoder("a cat").last_hidden_state`.
Passed directly to the SD UNet as `encoder_hidden_states`.

---

## 4. The Mathematics: Why the Pseudo-Inverse Works

### 4.1 What is $\mathbf{W}_T$?

$\mathbf{W}_T$ is the weight matrix of CLIP's `text_projection` layer — a single linear
layer with no bias, shape $(768 \times 768)$. Its job is to map the text encoder's final
hidden state into the CLIP joint space:

$$
\mathbf{v}^T = \mathbf{W}_T \mathbf{h}_\text{text}, \qquad \mathbf{h}_\text{text} \in \mathbb{R}^{768},\ \mathbf{v}^T \in \mathbb{R}^{768}
$$

It is not the text encoder itself — it is the single projection head that sits on top of
it, after all 12 transformer layers have run. In PyTorch it is stored as
`clip_model.text_projection.weight` with shape $(768, 768)$; the forward pass computes
$\mathbf{v}^T = \mathbf{h}^\top \mathbf{W}_T^\top$, equivalently
$\mathbf{v}^T = \mathbf{W}_T \mathbf{h}$ in column-vector convention.

### 4.2 Why Does the Pseudo-Inverse Work?

CLIP contrastive training forces:

$$
\mathbf{W}_T \mathbf{h}_\text{text} \approx \mathbf{W}_V \mathbf{p}_\text{image} \qquad \text{for matched pairs}
$$

Both sides land in the same joint space. This means image embeddings
$\mathbf{v}^I = \mathbf{W}_V \mathbf{p}$ approximately lie in $\text{col}(\mathbf{W}_T)$
— the set of vectors that $\mathbf{W}_T$ can produce from some hidden state $\mathbf{h}$.

Given that, the pseudo-inverse asks: *what $\mathbf{h}$ would $\mathbf{W}_T$ have needed
to produce $\mathbf{v}^I$?* This is the inversion problem:

$$
\hat{\mathbf{h}} = \mathbf{W}_T^+ \mathbf{v}^I = \underset{\mathbf{h}}{\arg\min}\ \|\mathbf{h}\|_2
\quad \text{s.t.} \quad \mathbf{W}_T \mathbf{h} \approx \mathbf{v}^I
$$

It works **because** training made the image embedding look like a text projection output.
If that alignment didn't exist — if $\mathbf{v}^I$ were random with respect to
$\text{col}(\mathbf{W}_T)$ — the pseudo-inverse would return a meaningless $\hat{\mathbf{h}}$
with large residual. **The alignment is the entire justification.**

If $\mathbf{W}_T$ were invertible, the solution would simply be $\mathbf{h} = \mathbf{W}_T^{-1} \mathbf{v}^I$.
In practice $\mathbf{W}_T$ may be rank-deficient, so the **Moore-Penrose pseudo-inverse**
gives the minimum-norm least-squares solution instead.

### 4.3 Singular Value Decomposition and the Pseudo-Inverse

Via the SVD, $\mathbf{W}_T = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$, the
pseudo-inverse is:

$$
\mathbf{W}_T^+ = \mathbf{V} \boldsymbol{\Sigma}^+ \mathbf{U}^\top
$$

where $\boldsymbol{\Sigma}^+$ replaces each diagonal entry $\sigma_i$ with $1/\sigma_i$
if $\sigma_i > \texttt{atol}$, else $0$. Setting $\texttt{atol} = 0.3$ zeroes out the
smallest singular values, suppressing noise amplification:

$$
(\boldsymbol{\Sigma}^+)_{ii} = \begin{cases} 1/\sigma_i & \text{if } \sigma_i > 0.3 \\ 0 & \text{otherwise} \end{cases}
$$

The full SD-IPC projection, including scale, is therefore:

$$
\boxed{
\hat{\mathbf{h}} = \mathbf{W}_T^+ \left( \lambda \cdot \frac{\mathbf{W}_V \mathbf{p}}{\|\mathbf{W}_V \mathbf{p}\|_2} \right), \qquad \lambda = 27.5
}
$$

### 4.4 Why the Pseudo-Inverse Is Approximately Correct

The solution $\hat{\mathbf{h}} = \mathbf{W}_T^+ \mathbf{v}^I$ satisfies $\mathbf{W}_T \hat{\mathbf{h}} = \mathbf{v}^I$
exactly if and only if $\mathbf{v}^I \in \text{col}(\mathbf{W}_T)$ (the column space
of $\mathbf{W}_T$). CLIP contrastive training makes this approximately true:

$$
\mathbf{v}^I \approx \mathbf{v}^T = \mathbf{W}_T \mathbf{h}_\text{matched} \in \text{col}(\mathbf{W}_T)
$$

The approximation error is bounded by the CLIP alignment gap:

$$
\|\mathbf{W}_T \hat{\mathbf{h}} - \mathbf{v}^I\|_2 = \|\mathbf{v}^T - \mathbf{v}^I\|_2 = \sqrt{2(1 - \cos(\mathbf{v}^I, \mathbf{v}^T))}
$$

For typical image-text pairs, $\cos(\mathbf{v}^I, \mathbf{v}^T) \approx 0.28$, giving a residual of approximately $\sqrt{2(1-0.28)} \approx 1.20$. The reconstruction is approximate, not exact — which is why SD-IPC regenerations resemble but do not perfectly reproduce the source.

### 4.5 Connection to the PyTorch Implementation

```python
# CLIP text_projection stores weight as (out_features=768, in_features=768)
# Forward: v = h @ W_T.T   (row vector convention)
# Inverse: h = v @ pinv(W_T).T

W_T    = clip_model.text_projection.weight          # (768, 768)
W_pinv = torch.linalg.pinv(W_T, atol=0.3)           # (768, 768) = W_T^+
proj_vec = image_emb_scaled @ W_pinv.T               # (1, 768) = ĥ
```

The `.T` on `W_pinv` converts from column-vector convention (used in the math above)
to row-vector convention (used in PyTorch's linear layers).

---

## 5. Architecture Summary (Abstracted)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           SD-IPC PIPELINE                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: PIL.Image  (any H × W × 3)                                      │
│              │                                                           │
│              ▼  CLIPProcessor: resize 224×224, normalize per channel     │
│         pixel_values:  (1, 3, 224, 224)                                 │
│              │                                                           │
│              ▼  CLIP ViT-L/14: 196 patches + CLS, 24 transformer layers │
│              │  Output: CLS token only  ← FROZEN                        │
│         p = pooler_output:  (1, 1024)                                   │
│              │                                                           │
│              ▼  v^I = W_V p / ‖W_V p‖   visual_projection  ← FROZEN    │
│         v^I = image_emb:  (1, 768)   ← CLIP joint space, unit vector    │
│              │                                                           │
│              ▼  ṽ^I = 27.5 · v^I                                        │
│         ṽ^I = image_emb_scaled:  (1, 768)                               │
│              │                                                           │
│              ▼  ĥ = W_T^+ ṽ^I   pinv(text_projection)  ← computed once │
│         ĥ = proj_vec:  (1, 768)   ← SD text-encoder hidden space        │
│              │                                                           │
│              ▼  C[0] = h_BOS,  C[1:] = ĥ  (broadcast)                  │
│         C = encoder_hidden_states:  (1, 77, 768)                        │
│              │                                                           │
│              ▼  SD UNet cross-attention: Q from latents, K,V from C     │
│         Generated image                                                  │
│                                                                          │
│  No gradients. No training. No optimisation loop.                        │
│  All weights (CLIP + SD) remain frozen.                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Visual Illustrations

### 6.1 The Three Spaces and Projection Chain

SD-IPC passes through three distinct vector spaces. The two projection matrices act as
bridges; the pseudo-inverse reverses one of them.

```
 ╔══════════════════════╗       ╔═══════════════════════════════════════╗       ╔══════════════════════╗
 ║  ViT HIDDEN SPACE    ║       ║         CLIP JOINT SPACE              ║       ║  SD TEXT HIDDEN      ║
 ║  (1024-dim)          ║       ║         (768-dim)                     ║       ║  SPACE  (768-dim)    ║
 ║                      ║       ║                                       ║       ║                      ║
 ║  Raw patch features  ║       ║  Image and text embeddings            ║       ║  What SD's UNet      ║
 ║  after 24 ViT layers ║       ║  are pulled together here             ║       ║  cross-attention     ║
 ║                      ║       ║  by contrastive training              ║       ║  was trained on      ║
 ║                      ║       ║                                       ║       ║                      ║
 ║    p (CLS token)     ║       ║   v^I  ●  ← image embedding          ║       ║   ĥ  ●               ║
 ║        ●             ║       ║         ╲                             ║       ║      ▲               ║
 ║        │             ║  W_V  ║          ╲  cos ≈ 0.28               ║ W_T^+ ║      │               ║
 ║        │  (768×1024) ╠══════►║           ●  v^T ← text embedding    ╠══════►║      │               ║
 ║        │             ║       ║          ("a cat")                    ║       ║      │               ║
 ║        │             ║       ║                                       ║       ║  h_text  ●           ║
 ║        │             ║       ║   v^I ≈ v^T (alignment, not exact)   ║       ║  (what "a cat"       ║
 ║        │             ║       ║                                       ║       ║   text token gives)  ║
 ╚════════╪═════════════╝       ╚═══════════════════════════════════════╝       ╚══════════════════════╝
          │                                                                               ▲
          │ pooler_output                     W_T (forward direction, text encoder):     │
          │ shape: (1,1024)                   h  ──── W_T ────► v^T  (text → joint)      │
          │                                                                               │
          │                                   W_T^+ (SD-IPC, inverse direction):         │
          └──────────────────────────────────  v^I ──── W_T^+ ───► ĥ  (joint → hidden) ──┘
                                               shape: (1,768)          shape: (1,768)
```

**Key insight**: $\mathbf{W}_T^+$ is the left-inverse of $\mathbf{W}_T$. It maps a point
in joint space back to the text hidden state that would have produced it — but only exactly
if the point was already on $\text{col}(\mathbf{W}_T)$. Because $\mathbf{v}^I \approx \mathbf{v}^T$
(CLIP alignment), the inversion is approximate. The better the alignment, the better the projection.

---

### 6.2 The Full Projection Chain With Shapes

```
                              SD-IPC: image → conditioning sequence

  ┌─────────────────┐
  │  Input Image    │  PIL.Image, any resolution
  └────────┬────────┘
           │
           ▼  CLIPProcessor: resize 224×224, normalise
  ┌─────────────────────────────────────────────────┐
  │  pixel_values                                    │  shape: (1, 3, 224, 224)
  └────────┬────────────────────────────────────────┘
           │
           ▼  ViT-L/14: 196 patches → 24 transformer layers → CLS token
  ┌─────────────────────────────────────────────────┐
  │  p  =  pooler_output                            │  shape: (1, 1024)
  │        raw ViT CLS token                        │  space: ViT hidden
  └────────┬────────────────────────────────────────┘
           │
           ▼  visual_projection:  W_V ∈ ℝ^{768×1024}
           │  v_raw = p @ W_V.T
           ▼  L2 normalise:  v^I = v_raw / ‖v_raw‖
  ┌─────────────────────────────────────────────────┐
  │  v^I  =  image_emb                              │  shape: (1, 768)
  │          CLIP image embedding (unit vector)      │  space: CLIP joint  ← same space as
  └────────┬────────────────────────────────────────┘             clip_text_emb("a cat")
           │
           ▼  scale:  ṽ^I = 27.5 × v^I
  ┌─────────────────────────────────────────────────┐
  │  ṽ^I  =  image_emb_scaled                       │  shape: (1, 768)
  │          ‖ṽ^I‖ ≈ 27.5  (matches EOS norm in SD) │  space: CLIP joint (scaled)
  └────────┬────────────────────────────────────────┘
           │
           ▼  pseudo-inverse:  W_T^+ = pinv(W_T, atol=0.3)
           │  ĥ = ṽ^I @ W_T^+.T
  ┌─────────────────────────────────────────────────┐
  │  ĥ  =  proj_vec                                 │  shape: (1, 768)
  │        projected text-encoder hidden state       │  space: SD text hidden  ← same space as
  └────────┬────────────────────────────────────────┘             text_encoder("a cat")[:,EOS,:]
           │
           ▼  broadcast to 76 positions + prepend BOS
  ┌─────────────────────────────────────────────────┐
  │  C  =  encoder_hidden_states                    │  shape: (1, 77, 768)
  │                                                  │
  │   pos  0:  h_BOS  (from null prompt "")         │
  │   pos  1:  ĥ      ─┐                            │
  │   pos  2:  ĥ       │  all identical             │
  │   pos  3:  ĥ       │  same vector               │
  │        ⋮   ĥ      ─┘  repeated 76 times         │
  │   pos 76:  ĥ                                    │
  └────────┬────────────────────────────────────────┘
           │
           ▼  passed to SD UNet as encoder_hidden_states
  ┌─────────────────────────────────────────────────┐
  │  Generated image                                │
  └─────────────────────────────────────────────────┘
```

---

### 6.3 Cross-Attention: How C Conditions the UNet

The SD UNet is a stack of residual blocks and transformer blocks. Each transformer block
contains a **cross-attention** layer that reads from $\mathbf{C}$ (the conditioning sequence).

```
                        ONE CROSS-ATTENTION LAYER IN THE SD UNET

  ┌────────────────────────────────────────────────────────────────────────────┐
  │                                                                            │
  │   Spatial feature map (from noisy latent x_t being denoised)              │
  │   shape: (B, H·W, d_model)   e.g. (1, 64·64, 320) at one resolution      │
  │             │                                                              │
  │             ▼  Linear W_Q                                                  │
  │   Q  =  queries    shape: (1, 4096, 64)  ← one query per spatial position │
  │                                                                            │
  │                                                                            │
  │   Conditioning sequence C   shape: (1, 77, 768)   ← from SD-IPC          │
  │             │                                                              │
  │             ├──► Linear W_K ──► K  shape: (1, 77, 64)  ← 77 keys         │
  │             └──► Linear W_V ──► V  shape: (1, 77, 64)  ← 77 values       │
  │                                                                            │
  │                                                                            │
  │              ┌─── Attention weights ───────────────────────────────────┐  │
  │              │                                                          │  │
  │              │  A = softmax( Q K^T / √64 )   shape: (1, 4096, 77)      │  │
  │              │                                                          │  │
  │              │  A[b, i, j]  =  how much spatial position i             │  │
  │              │                 attends to conditioning token j          │  │
  │              │                                                          │  │
  │              └──────────────────────────────────────────────────────── ┘  │
  │                                                                            │
  │              Output  =  A · V    shape: (1, 4096, 64)                     │
  │              (weighted sum of value vectors from C)                        │
  │                                                                            │
  └────────────────────────────────────────────────────────────────────────────┘
```

**What this means for SD-IPC specifically**:

Because all 76 non-BOS keys $\mathbf{K}_{[1]}, \ldots, \mathbf{K}_{[76]}$ are **identical**
(projected from the same $\hat{\mathbf{h}}$), the attention weights across those positions
are uniform. The output collapses to a single effective conditioning vector:

```
  Real text prompt "a cat":                    SD-IPC conditioning:
  ┌─────────────────────────────┐              ┌─────────────────────────────┐
  │  K[0]  = W_K · h_BOS       │              │  K[0]  = W_K · h_BOS       │
  │  K[1]  = W_K · h["a"]      │              │  K[1]  = W_K · ĥ  ─┐       │
  │  K[2]  = W_K · h["cat"]    │              │  K[2]  = W_K · ĥ   │       │
  │  K[3]  = W_K · h["on"]     │              │  K[3]  = W_K · ĥ   │ all   │
  │  K[4]  = W_K · h["the"]    │              │  K[4]  = W_K · ĥ   │ equal │
  │  K[5]  = W_K · h["bed"]    │              │  K[5]  = W_K · ĥ  ─┘       │
  │  K[6]  = W_K · h_EOS       │              │  K[6]  = W_K · ĥ  ─┐       │
  │  ...                        │              │  ...                 │       │
  │  K[76] ≈ 0 (padding)        │              │  K[76] = W_K · ĥ  ─┘       │
  └─────────────────────────────┘              └─────────────────────────────┘
          │                                              │
          ▼                                              ▼
  Each spatial position in the UNet           Every spatial position sees
  can selectively attend to specific          the same effective key →
  token concepts ("cat" vs "bed")             no selective attention possible
  → rich, structured conditioning             → globally blended conditioning
```

The UNet's cross-attention was trained to use **selective attention** across diverse token
positions. SD-IPC forces it into a degenerate regime where it can only use the global
image summary, which is why generations are semantically similar to the source but lack
structural detail.

---

### 6.4 The Composability Gap in the Joint Space

This illustration shows why AND hybrid images fail cycle consistency.

```
  CLIP Joint Space (768-dim, shown projected to 2D)

         v^T("a cat")                 v^T("a dog")
              ●                             ●
             /                              \
            /  cos ≈ 0.28                    \  cos ≈ 0.28
           /                                  \
  v^I([photo of cat])                 v^I([photo of dog])
          ●                                   ●
           \                                 /
            \                               /
             ╲                             ╱
              ╲   SD-IPC inversion works  ╱
               ╲  (single concept →      ╱
                ╲  ĥ ≈ h_text)           ╱


  ─────────────────────────────────────────────────────────────

  What happens for a SuperDiff AND hybrid (cat + dog image):

         v^T("a cat")       midpoint        v^T("a dog")
              ●  ·  ·  ·  ·  ·  ●  ·  ·  ·  ·  ●
                              ↑
                         v^I([AND hybrid])
                              ●
                              │
                     "which text embedding
                      does this map back to?"
                              │
                              ▼
                    ĥ = W_T^+ · ṽ^I
                              │
                              ▼
                  ĥ  ≈  midpoint in hidden space
                  SD generates neither cat nor dog clearly,
                  OR collapses to whichever concept has
                  larger α in: v^I = α·v^I_cat + (1-α)·v^I_dog
```

The AND hybrid's CLIP embedding lands **between** the two concept directions in joint
space. The pseudo-inverse maps this midpoint back to a hidden state that SD has never
been trained to condition on — producing a degraded or collapsed generation, measured
as low cycle consistency.

---

## 7. What Information Is Lost vs. Preserved

### 6.1 Information Preserved

| What | Why preserved |
|------|---------------|
| Dominant concept (primary object) | CLIP CLS token strongly represents the salient subject |
| Global style and texture | ViT attends to texture patterns across all 196 patches |
| Approximate colour palette | Embedded in CLIP patch features |
| Category-level semantics | Core of CLIP's contrastive training target |

### 6.2 Information Lost

| What | Why lost | Impact |
|------|----------|--------|
| Spatial layout | CLS is permutation-invariant over patches; no positional info in $\mathbf{p}$ | "cat on left, dog on right" → layout gone |
| Secondary / minority objects | CLS is a weighted average biased toward salient area | Cat+dog image → $\mathbf{v}^I \approx \alpha \mathbf{v}^I_\text{cat}$, dog lost |
| Precise counts | "three cats" vs "one cat" produce similar $\mathbf{v}^I$ | SD regen may have wrong count |
| Spatial relations | "above", "below" require positional contrast; CLS averages this away | "cat on the bed" → "on" may not survive |
| Fine-grained attributes | Subtle textures, colours of small objects | Averaged into global CLS |
| Per-token diversity | $\mathbf{C}$ has 76 identical vectors vs 76 distinct token vectors in real text | UNet cannot look up different concepts at different positions |

### 6.3 The Information Bottleneck — Quantified

For a $k$-token prompt, the text encoder produces $k$ distinct vectors, each in $\mathbb{R}^{768}$.
SD-IPC produces 1 vector broadcast to all 76 non-BOS positions. The **capacity ratio** is:

$$
\rho = \frac{768}{k \times 768} = \frac{1}{k}
$$

For "a cat lying on the bed" ($k = 7$ content tokens):

$$
\rho = \frac{1}{7} \approx 14.3\%
$$

This 7$\times$ compression explains why SD-IPC regenerations are recognisable but not faithful
reproductions of the source.

---

## 7. Supervisor Q&A Reference

### Q: "How was SD-IPC trained?"

**A**: It was **not trained**. SD-IPC is a closed-form method using two pretrained frozen
models: CLIP (for the vision encoder and both projections) and Stable Diffusion (for
the text encoder and UNet). No gradient updates occur. The only computation at inference
time is: one ViT forward pass, two matrix multiplies, an L2 normalise, and a scalar
multiply. The pseudo-inverse $\mathbf{W}_T^+$ is computed once from the frozen CLIP
weights and cached.

### Q: "Is this a CLIP embedding?"

**A**: Partially. The intermediate $\mathbf{v}^I \in \mathbb{R}^{768}$ after `visual_projection`
**is** a standard CLIP image embedding — the same one used for retrieval, classification,
and similarity search. SD-IPC then continues: it applies $\mathbf{W}_T^+$ to map
$\mathbf{v}^I$ back into the SD text-encoder's hidden space. The final output
$(1, 77, 768)$ is **not** a CLIP embedding — it is a synthesised SD text conditioning
sequence derived from a CLIP embedding.

### Q: "What is the shape of the output?"

**A**: $(1, 77, 768)$ — identical to `text_encoder("any prompt").last_hidden_state`.
- $1$: batch dimension
- $77$: sequence length (CLIP/SD max: 1 BOS + 75 tokens + 1 EOS)
- $768$: hidden dimension of CLIP-L (the text encoder used in SD 1.4)

All 76 non-BOS positions contain **the same 768-dim vector** $\hat{\mathbf{h}}$.
Position 0 (BOS) is taken from the null prompt.

### Q: "How do they project from image embedding to text embedding?"

**A**: Via the Moore-Penrose pseudo-inverse of CLIP's text projection. The text
projection is a linear map $\mathbf{W}_T : \mathbb{R}^{768} \to \mathbb{R}^{768}$.
Given $\mathbf{v}^I$ in joint space, SD-IPC solves:

$$
\hat{\mathbf{h}} = \underset{\mathbf{h}}{\arg\min}\|\mathbf{h}\|_2
\quad \text{s.t.} \quad \mathbf{h} = \underset{\mathbf{h}'}{\arg\min} \|\mathbf{W}_T \mathbf{h}' - \tilde{\mathbf{v}}^I\|_2
$$

which has the closed-form solution $\hat{\mathbf{h}} = \mathbf{W}_T^+ \tilde{\mathbf{v}}^I$.
This works because CLIP training ensures $\mathbf{v}^I \approx \mathbf{v}^T = \mathbf{W}_T \mathbf{h}_\text{matched}$,
so $\mathbf{v}^I$ lies approximately in $\text{col}(\mathbf{W}_T)$.

### Q: "What information is lost?"

**A**: Everything that cannot be expressed in a single 768-dim vector — spatial layout,
minority concepts in composed images, precise counts, spatial relations ("on", "beside"),
and per-token semantic diversity. The capacity ratio is $\rho = 1/k$ for a $k$-token prompt.

### Q: "What information is preserved?"

**A**: The dominant concept's category, approximate style, global texture, and colour
palette — whatever CLIP's CLS token captures as the most salient image-level summary.

### Q: "Is this the same as a text prompt embedding?"

**A**: It has the same shape and is passed through the same cross-attention interface,
but it is not semantically equivalent. A real text embedding has $77$ distinct vectors
with structured token-level meaning. SD-IPC's $\mathbf{C}$ has 76 identical vectors —
a degenerate case that SD's UNet was never explicitly trained on. Cross-attention is a
weighted average:

$$
\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}
$$

When all keys $\mathbf{K}$ are identical (as in SD-IPC), the softmax is uniform and the
output is simply $\mathbf{V}$ (also identical), i.e., the same conditioning regardless of
which query position attends. The UNet degrades to a globally-conditioned denoiser.

### Q: "Why not use CLIP image embeddings directly?"

**A**: SD's UNet cross-attention requires $(1, 77, 768)$ tensors from CLIP's text-encoder
hidden space. A raw CLIP image embedding $\mathbf{v}^I \in \mathbb{R}^{768}$ is wrong
shape and in the joint space, not the text-encoder hidden space. These are related by
$\mathbf{v}^T = \mathbf{W}_T \mathbf{h}$, so $\mathbf{v}^I \neq \mathbf{h}$. SD-IPC
provides $\mathbf{W}_T^+$ to bridge the gap, then broadcasts to fill the 77-token interface.

### Q: "Why do the regenerated images not look exactly like the source?"

**A**: Three additive sources of approximation. Letting the total reconstruction error be
$\epsilon = \|\hat{\mathbf{h}} - \mathbf{h}_\text{ideal}\|_2$:

1. **CLIP alignment gap** ($\epsilon_1$): $\|\mathbf{v}^I - \mathbf{v}^T\|_2 = \sqrt{2(1-0.28)} \approx 1.20$
2. **Pseudo-inverse residual** ($\epsilon_2$): propagates $\epsilon_1$ through $\mathbf{W}_T^+$
3. **Uniform-broadcast bottleneck** ($\epsilon_3$): 76 identical positions cannot encode
   the per-token diversity that the UNet was trained to use

### Q: "Does SD-IPC work for SDXL or SD3?"

**A**: Not without adaptation. SDXL uses a dual encoder:
- CLIP-L: $\mathbf{W}_T^{(L)} \in \mathbb{R}^{768 \times 768}$, output contributes to $(1, 77, 768)$
- CLIP-G: $\mathbf{W}_T^{(G)} \in \mathbb{R}^{1280 \times 1280}$, output contributes to $(1, 77, 1280)$
- Concatenated: `prompt_embeds` $(1, 77, 2048)$ + `pooled_prompt_embeds` $(1, 1280)$

The pinv trick applies separately to each encoder's $\mathbf{W}_T$, but $\lambda = 27.5$
must be recalibrated for CLIP-G's different norm distribution. SD3 adds a T5-XXL encoder with no obvious pinv analogue.

---

## 8. The Theoretical Insight (the "Secret")

The paper's title claim refers to this algebraic observation:

> CLIP's text projection $\mathbf{W}_T$ is a linear map from text hidden states to the
> joint space. CLIP training places image embeddings approximately in the same joint
> space. Therefore $\mathbf{W}_T^+$ is a ready-made image-to-text-hidden-state converter
> — baked into every pretrained CLIP model, requiring no additional training.

The complete SD-IPC formula, from raw image to SD conditioning sequence, is:

$$
\mathbf{C}_{[k]} = \begin{cases}
  \mathbf{h}_\text{BOS} & k = 0 \\[4pt]
  \mathbf{W}_T^+ \left( \lambda \cdot \dfrac{\mathbf{W}_V \, f_I(x)}{\|\mathbf{W}_V \, f_I(x)\|_2} \right) & k = 1, \ldots, 76
\end{cases}
$$

with $\lambda = 27.5$, $\mathbf{W}_T^+ = \texttt{pinv}(\mathbf{W}_T, \texttt{atol}=0.3)$,
and $f_I$ the ViT-L/14 vision encoder.

This is fully closed-form: no optimisation, no training data, no gradient flow.

---

## 9. How This Enables Cycle Consistency Measurement

SD-IPC enables the following measurable cycle. Let $\phi : \text{Image} \to \mathbb{R}^{768}$
denote the CLIP image encoder. The cycle consistency score for an image $x$ is:

$$
s(x) = \cos\!\left(\phi(x),\ \phi\!\left(\text{SD}\!\left(\mathbf{C}(x),\ \epsilon\right)\right)\right)
$$

where $\mathbf{C}(x)$ is the SD-IPC conditioning derived from $x$ and $\epsilon$ is a
fresh noise sample (new seed). In code:

```
x  (PIL.Image)
│
▼  C = sdipc_project(x)             # (1, 77, 768) — §3
│
▼  x̂ = SD_UNet(C, seed=new_seed)   # regenerated image
│
▼  s = cos(φ(x), φ(x̂))             # CLIP cosine similarity
```

**High $s(x)$** → dominant concept survived the projection–regeneration round-trip.
Image content is representable in SD's conditioning space via a pooled embedding.

**Low $s(x)$** → content could not be faithfully encoded through the single-vector
bottleneck. For AND hybrid images (two concepts), $s$ is expected to be lower than for
monolithic SD images — the measurable signature of the **composability gap**.

$$
\Delta_\text{gap} = \mathbb{E}_{x \sim p_\text{mono}}[s(x)] - \mathbb{E}_{x \sim p_\text{AND}}[s(x)] > 0
$$

The gap $\Delta_\text{gap}$ is a lower bound on the true composability gap (SD-IPC amplifies
it via the pooling bottleneck). Pairing SD-IPC with EDITOR (full $77\times768$ optimised
conditioning) distinguishes artefact from structural gap.

---

## 10. Quick Reference

| Question | Answer |
|----------|--------|
| Was it trained? | No — closed-form, all weights frozen |
| Input shape | $(1, 3, 224, 224)$ after CLIPProcessor normalisation |
| After ViT | $(1, 1024)$ — CLS token in raw ViT hidden space |
| After $\mathbf{W}_V$ + L2 norm | $(1, 768)$ — standard CLIP image embedding |
| After $\times 27.5$ | $(1, 768)$ — $\|\mathbf{v}\|_2 \approx 27.5$ |
| After $\mathbf{W}_T^+$ | $(1, 768)$ — SD text-encoder hidden space |
| Output | $(1, 77, 768)$ — 1 BOS + 76 identical projected vectors |
| Projection formula | $\hat{\mathbf{h}} = \mathbf{W}_T^+\!\left(27.5 \cdot \mathbf{v}^I\right)$ |
| Capacity ratio vs real text | $1/k$ where $k$ = number of content tokens |
| Alignment residual | $\sqrt{2(1 - \cos(\mathbf{v}^I, \mathbf{v}^T))} \approx 1.20$ |
| Primary limitation | Single CLS token = no spatial or compositional structure |
| Model compatibility | SD 1.4 natively; $\lambda$ and interface differ for SDXL, SD3 |
| Speed | $<100$ ms per image (closed-form inference only) |

---

## 11. Research Hypotheses

### Hypotheses

**RH1 — Characterisation**
> Score-space AND composition produces images that lie outside the text-reachable manifold of CLIP joint space. We characterise this composability gap as a function of the semantic distance between constituent concepts.

**RH2 — Conditional Application**
> Score-space AND composition faithfully represents the logical conjunction of independent factors when their latent representations are orthogonal. Co-morbid disease synthesis in chest X-rays serves as a structured testbed for this condition, where clinical independence provides a natural orthogonality prior.

**RH3 — General Solution**
> When orthogonality does not hold, manifold-preserving composition via geodesic traversal keeps the composed representation on the data manifold, closing the composability gap in the general case.

### Connecting Narrative

Logical composition in diffusion models works only when the concepts being composed are orthogonal in latent space — otherwise the result falls off the manifold that text and semantics can reach. We first characterise this composability gap (RH1), then show that the orthogonality condition is naturally satisfied for independent disease co-occurrences in X-rays, enabling faithful medical image synthesis (RH2). Where orthogonality cannot be assumed, we close the gap by composing along geodesics on the data manifold rather than in raw score space (RH3).
