# The PoE Interaction Term: Derivation, Intractability, and What We Can Do Instead

## 1. Problem Statement

Standard **Product of Experts (PoE)** composes two conditional score functions as:

$$s_{\text{PoE}} = s_1 + s_2 - s_\emptyset$$

where $s_i = \nabla_{x_t} \log p(x_t \mid c_i)$ and $s_\emptyset = \nabla_{x_t} \log p(x_t)$.

This approximation holds exactly only when the two concepts are **conditionally independent given $x_t$**:

$$p(c_1, c_2 \mid x_t) = p(c_1 \mid x_t) \cdot p(c_2 \mid x_t)$$

For spatially competing concepts (e.g. "a cat" + "a dog") this condition fails: if the model infers a cat is present at some pixel, the probability of a dog at the same pixel is suppressed. The concepts are deeply conditionally **dependent**.

---

## 2. Derivation: Bayesian Decomposition → Interaction Term

Starting from Bayes' rule:

$$\nabla_{x_t} \log p(x_t \mid c_1, c_2) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(c_1, c_2 \mid x_t)$$

Decomposing the joint condition via the PMI identity:

$$\log p(c_1, c_2 \mid x_t) = \log p(c_1 \mid x_t) + \log p(c_2 \mid x_t) + \underbrace{\log \frac{p(c_1, c_2 \mid x_t)}{p(c_1 \mid x_t) \cdot p(c_2 \mid x_t)}}_{\text{log PMI}}$$

Substituting $\log p(c_i \mid x_t) = \log p(x_t \mid c_i) - \log p(x_t) + \text{const}$:

$$\nabla_{x_t} \log p(x_t \mid c_1, c_2) = \underbrace{s_1 + s_2 - s_\emptyset}_{s_{\text{PoE}}} + \underbrace{\nabla_{x_t} \log \frac{p(c_1, c_2 \mid x_t)}{p(c_1 \mid x_t) \cdot p(c_2 \mid x_t)}}_{\Delta(x_t)}$$

The **interaction term** is therefore:

$$\boxed{\Delta(x_t) = \nabla_{x_t} \log \operatorname{PMI}(c_1, c_2 \,;\, x_t)}$$

This is the gradient of the pointwise mutual information of the conditions given the noisy image.

---

## 3. Why the Interaction Term Is Intractable

**$\Delta(x_t)$ cannot be estimated from marginal models alone.**

Computing $\Delta(x_t)$ requires $p(c_1, c_2 \mid x_t)$, the posterior probability of the *joint* condition given the noisy image. This in turn requires the joint conditional model $p(x_t \mid c_1, c_2)$, which is exactly the model conditioned on "a cat and a dog" simultaneously — the joint prompt we are trying to avoid using.

The marginal models $p(x_t \mid c_1)$ and $p(x_t \mid c_2)$ provide:
- $\nabla_{x_t} \log p(c_1 \mid x_t)$ (up to a constant)
- $\nabla_{x_t} \log p(c_2 \mid x_t)$ (up to a constant)

But their product $p(c_1 \mid x_t) \cdot p(c_2 \mid x_t)$ is **not** $p(c_1, c_2 \mid x_t)$ unless the concepts are independent.

> **The PMI framing is tautological.** Writing $\Delta = \nabla_{x_t} \log \text{PMI}$ renames the unknown quantity — it does not make it estimable. Any method that claims to "recover $\Delta(x_t)$" using only marginal prompts is making an approximation, not computing the true interaction term.

The gap diagnostic data confirms this: the cosine similarity between the true joint velocity and the PoE velocity is close to $-1.0$ at large noise timesteps (early denoising), meaning PoE points in nearly the *opposite direction* to the true composed score.

---

## 4. What Existing Methods Did Wrong

### 4.1 Method 11 (before fix): Agreement term had the wrong sign

The "agreement-seeking" correction was:

$$\Delta x_0^{\text{agree}} = -(\hat{x}_0^{(1)} - \hat{x}_0^{(2)})$$

This drives both Tweedie predictions toward their **mean**. For cat + dog:

$$\hat{x}_0^{(1)} \approx \text{(cat-like image)}, \quad \hat{x}_0^{(2)} \approx \text{(dog-like image)}$$

$$-(\hat{x}_0^{(1)} - \hat{x}_0^{(2)}) \text{ pushes both toward } \frac{\hat{x}_0^{(1)} + \hat{x}_0^{(2)}}{2} \quad \Rightarrow \textbf{chimera}$$

The sign should be **positive** — pushing the predictions *apart* at contested regions, not averaging them.

**Fix:** Replace with spatial divergence push:

$$\Delta x_0^{\text{diverge}} = +w_{\text{overlap}}(x_t) \odot (\hat{x}_0^{(1)} - \hat{x}_0^{(2)})$$

where $w_{\text{overlap}} = \frac{||\hat{x}_0^{(1)}||_C \cdot ||\hat{x}_0^{(2)}||_C}{\max(\cdot)}$ is a per-pixel weight that is high where both concepts are simultaneously active.

### 4.2 Method 11 (before fix): Spatial corrections active at pure noise

The schedule peak was $\alpha_{\text{peak}} = 0.15$ (i.e., the correction was strongest at $t_{\text{frac}} \approx 0.15$, just after pure noise).

At pure noise ($t_{\text{frac}} \approx 0$), $\hat{x}_0 \approx \mathcal{N}(0, I)$ — the Tweedie prediction is near-uniform Gaussian and carries **no spatial information**. Applying a spatially-informed correction here is equivalent to applying it to random noise.

**Fix:** Add a phase gate: corrections are zero for $t_{\text{frac}} < 0.20$.

### 4.3 Method 12 (before fix): Activation window missed the layout phase

The overlap penalty was configured with `apply_from_frac=0.70`, meaning it activated at 70% through denoising.

However, **spatial layout** (which concept occupies which region) is committed in the **first 30% of the trajectory** — the high-noise steps where the coarse structure is established. By step 35 of 50, the latent's coarse spatial organisation is frozen; only texture and fine detail remain malleable.

Penalising overlap at steps 35–50 reshapes texture, not layout.

**Fix:** `apply_from_frac=0.05, apply_to_frac=0.60` — the correction now covers the critical early layout-formation phase.

### 4.4 Method 12 (before fix): Penalty was negligible in magnitude

With `step_size=0.05` and measured `correction_grad_norm ≈ 0.027`:

$$\|\Delta x_t\| = \text{step\_size} \times \text{grad\_norm} = 0.05 \times 0.027 \approx 0.00135$$

Latent norms are $O(1)$–$O(10)$, so the effective relative correction was $\approx 10^{-4}$. The penalty was geometrically too small to reshape the latent trajectory.

**Fix:** `step_size=0.30`, `lambda_overlap_start=2.0` → expected effective correction $\approx 0.03$–$0.30$ (2–3 orders of magnitude larger).

### 4.5 The gradient cosine as a collision detector: blind to constructive interference

The prediction was:

> "A highly negative cosine similarity $\cos(S_1, S_2) \ll 0$ is an early warning for mode collision."

This is only true for *destructive* interference (concepts pulling in opposite directions → one erases the other).

For **cat + dog**, both concepts natively want to occupy the **centre** of the image. Their score deltas $\Delta_1, \Delta_2$ point in *similar* directions (constructive interference). The cosine is **positive** — falsely suggesting "no problem" — while the generation collapses to a chimera.

**Revised diagnostic:** Track both the cosine AND $||\hat{x}_0^{(1)} - \hat{x}_0^{(2)}||$ (Tweedie disagreement). A chimera failure shows: high $\cos(\Delta_1, \Delta_2) > 0$ (constructive) AND low $||\hat{x}_0^{(1)} - \hat{x}_0^{(2)}||$ (predictions not separating). A successful composition shows: moderate cosine AND growing disagreement norm.

### 4.6 The $\mathcal{M}_{\text{gap}}$ metric conflates separation with competition

$$\mathcal{M}_{\text{gap}} = ||\hat{x}_0^{(1)} - \hat{x}_0^{(2)}||_2^2$$

High $\mathcal{M}_{\text{gap}}$ could mean:
- **(a) Good:** The two concepts are spatially separated (high disagreement norm, low overlap) — exactly what we want.
- **(b) Bad:** The two concepts are destructively competing (both trying to predict conflicting content at the same location).

The scalar norm cannot distinguish these. The **spatial distribution** of the disagreement is what matters: high disagreement in *different* regions is good; high disagreement in the *same* region is bad.

---

## 5. The Spatial Routing Approach (Method 13)

Rather than approximating $\Delta(x_t)$, we **enforce the condition under which PoE is exact**: spatial disjointness.

When $\text{supp}(c_1) \cap \text{supp}(c_2) = \emptyset$, the interaction term vanishes: $\Delta(x_t) = 0$ and PoE is the correct composition. We engineer this condition directly.

### 5.1 The Softmax Router

At each timestep, the model's Tweedie predictions reveal where each concept is "claiming" spatial territory. The **channel L2 norm** at each position measures the prediction intensity:

$$n_1^{(b,h,w)} = ||\hat{x}_0^{(1)}[b, :, h, w]||_2, \quad n_2^{(b,h,w)} = ||\hat{x}_0^{(2)}[b, :, h, w]||_2$$

A **competitive softmax** over the concept axis resolves competition:

$$\text{stack} = [n_1, n_2] \in \mathbb{R}^{B \times 2 \times H \times W}$$

$$[w_1, w_2] = \operatorname{softmax}\left(\frac{\text{stack}}{T}, \; \dim=1\right) \quad \Rightarrow \quad w_1 + w_2 = 1 \text{ per pixel}$$

> **Why `dim=1` (concept axis), not `dim=(2,3)` (spatial)?**
> Softmax over spatial dims produces an *attention map* — "where does this concept attend?" — spreading probability over space. That is a different operation. Softmax over the concept axis produces a *per-pixel assignment* — "which concept owns this pixel?" — which is what enforces disjointness.

The composed Tweedie mean:

$$\hat{x}_0^{\text{comp}} = w_1 \odot \hat{x}_0^{(1)} + w_2 \odot \hat{x}_0^{(2)}$$

Map back to velocity space and apply CFG:

$$v^{\text{comp}} = \frac{x_t - \hat{x}_0^{\text{comp}}}{\sigma}, \quad v_f = v_\emptyset + \lambda \cdot (v^{\text{comp}} - v_\emptyset)$$

### 5.2 Why No Background Subtraction

The standard PoE formula uses $\hat{x}_0^{\text{PoE}} = \hat{x}_0^{(1)} + \hat{x}_0^{(2)} - \hat{x}_0^{(\emptyset)}$, subtracting the unconditional prediction to remove baseline content.

In the spatial routing formulation:
- The weights enforce $w_1 + w_2 = 1$ — the combination is already normalised.
- Subtracting $\hat{x}_0^{(\emptyset)}$ would remove background content from *both* concept regions and introduce a bias.
- The CFG wrapper $v_f = v_\emptyset + \lambda(v^{\text{comp}} - v_\emptyset)$ provides the unconditional baseline.

### 5.3 Temperature Schedule

$$T(t_{\text{frac}}) = \begin{cases}
T_{\text{start}} & t_{\text{frac}} \leq t_1 \\
T_{\text{start}} + \frac{t_{\text{frac}} - t_1}{t_2 - t_1}(T_{\text{end}} - T_{\text{start}}) & t_1 < t_{\text{frac}} < t_2 \\
T_{\text{end}} & t_{\text{frac}} \geq t_2
\end{cases}$$

Defaults: $T_{\text{start}} = 1.0$, $T_{\text{end}} = 0.2$, $t_1 = 0.25$, $t_2 = 0.70$.

**Interpretation:**
- **High $T$ (early):** Both $n_1, n_2 \approx \text{const}$ (Gaussian noise) → $w_1 \approx w_2 \approx 0.5$ everywhere → soft blending, no premature commitment. Correct, since spatial structure has not yet formed.
- **Low $T$ (mid-late):** The softmax sharpens → regions where $n_1 > n_2$ strongly prefer concept 1, and vice versa → near-binary masks → effective spatial disjointness.
- **Phase 3** ($t_{\text{frac}} > 0.70$): Masks are near-binary and stable. Standard PoE is applied, with optional PCGrad conflict projection to remove residual cross-concept interference in the score deltas.

---

## 6. Diagnostics: What to Track Per Step

| Metric | Computation | What it means | Target |
|--------|-------------|---------------|--------|
| `cos_d1_d2` | $\cos(\Delta_1, \Delta_2)$ | Constructive (+) vs destructive (−) interference | Should trend toward 0 or negative as concepts separate |
| `spatial_iou` | Soft IoU$(w_1, w_2)$ | How much both concepts claim the same pixels | Should decrease from ~0.5 to <0.1 |
| `x0_disagree` | $\|\|\hat{x}_0^{(1)} - \hat{x}_0^{(2)}\|\|$ | Concept separation in Tweedie space | Should **grow** early then stabilise |
| `correction_grad_norm` | $\|\|\nabla_{x_t} E\|\|$ (Method 12) | Strength of the overlap penalty | Should be $O(0.1)$–$O(1)$, not $O(0.001)$ |
| `lambda_overlap_t` | Scheduled penalty weight (Method 12) | Whether penalty is active | Must be nonzero in first 60% of steps |
| `phase` | 1, 2, or 3 (Method 13) | Which routing regime is active | Should transition 1→2→3 |
| `temperature` | $T$ at step $i$ (Method 13) | Softmax sharpness | Should decrease from 1.0 to 0.2 |
| `dominance_ratio` | $\|\|\Delta_1\|\| / \|\|\Delta_2\|\|$ | One concept drowning the other | Should stay near 1.0; far from 1 → concept drop |

---

## 7. Expected Outcomes

### Successful composition (cat + dog, two distinct animals)

- **Visual:** Both animals present, spatially separated (left/right, foreground/background)
- `x0_disagree` is large ($> 5$) and **growing** at $t_{\text{frac}} \in [0.1, 0.4]$
- `cos_d1_d2` starts near $+1$ (both want the centre) then trends toward $0$ as the router separates them
- `spatial_iou` decreases from $\approx 0.5$ (random) to $< 0.15$ by $t_{\text{frac}} = 0.5$
- `dominance_ratio` stays near $1.0$ throughout

### Chimera failure (merged cat-dog entity)

- **Visual:** Single animal with mixed features (e.g. dog body with cat ears)
- `x0_disagree` remains **small** throughout ($< 3$)
- `cos_d1_d2` stays large positive ($> 0.5$) — constructive interference not broken
- `spatial_iou` stays near $0.5$ — spatial assignment never differentiates
- For Method 12: `correction_grad_norm` near zero (penalty too weak) or `lambda_overlap_t = 0` (window missed)

### Entity missing failure (only one concept rendered)

- **Visual:** Only a cat or only a dog
- One of `occupancy_mass_a` or `occupancy_mass_b` collapses to near $0$
- `dominance_ratio` $\gg 1$ or $\ll 1$ persistently — one score delta overwhelms the other
- `preservation_penalty > 0` (Method 12) — preservation constraint triggered but insufficient

---

## 8. Summary of Changes Made to Methods 11 and 12

### Method 11 — `method_11_tweedie_poe_corrector.py`

| Change | Old | New | Reason |
|--------|-----|-----|--------|
| Correction A | `_delta_agreement = -(x̂₀¹-x̂₀²)` | `_delta_divergence = +overlap_w⊙(x̂₀¹-x̂₀²)` | Wrong sign drove toward chimera |
| `alpha_peak` | 0.15 | 0.20 | Correction should peak at layout-formation, not pure noise |
| `alpha_width` | 0.15 | 0.20 | Wider coverage of layout window |
| `gamma_peak` | 0.45 | 0.35 | Spatial repulsion earlier in layout phase |
| `gamma_max` | 0.20 | 0.25 | Slightly stronger spatial push |
| Phase gate | none | `spatial_phase_gate=0.20` | No spatial signal at pure noise |
| Outer CFG | always on | `apply_outer_cfg=False` | Already-composed velocity incorrectly re-wrapped |
| Diagnostic | missing | `cos_d1_d2` in `info` | Need to detect constructive interference |

### Method 12 — `method_12_overlap_penalty_corrector.py`

| Change | Old | New | Reason |
|--------|-----|-----|--------|
| `apply_from_frac` | 0.70 | 0.05 | Must cover early layout-formation phase |
| `apply_to_frac` | 0.25 | 0.60 | Active window now `[0.05, 0.60]` |
| `step_size` | 0.05 | 0.30 | Effective correction was $\approx 10^{-4}$ (negligible) |
| `lambda_overlap_start` | 0.45 | 2.0 | Penalty strength needed to be $\approx 50\times$ larger |
| `lambda_overlap_end` | 0.10 | 0.50 | Stronger tail |
| Diagnostics | — | `cos_delta1_delta2`, `dominance_ratio`, `x0_disagree_norm` | Directional interference visibility |
