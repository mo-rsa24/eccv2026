Here’s a compact “mental map” of compositionality in generative modeling, organized around methods, assumptions, and what they would mean for TB ∧ silicosis.

---

## 1. Big picture: what “compositionality” means here

**Semantic compositionality** = we want the composed distribution to reflect a *conceptual* AND/OR/NOT:

* “dog AND hat” → dog wearing a hat
* “TB AND silicosis” → realistic co-morbidity in the *same* lungs
* “red cube left of blue sphere” (CLEVR) → correct objects + relations

Key question:

> When we combine models or densities, *when does that algebra in log-p space actually correspond to this semantic composition?*

The different papers answer this with different tools and assumptions.

---

## 2. Score / energy combination methods

### 2.1 Composable Diffusion (Liu et al., ECCV 2022) ([arXiv][1])

**Method**

* View diffusion models as *implicit EBMs*: each model has score (\nabla_x \log p_i(x_t)).
* Compose by **linear score arithmetic**:

  * AND (conjunction): sum scores of relevant models
  * NOT: subtract a model’s score
* Works with conditional text/image models (e.g. Stable Diffusion) and object-level models; no explicit log-density tracking.

**Assumptions / implicit requirements**

* There exists a feature space where *Factorized Conditionals* roughly hold (cf. Bradley): each model captures independent “features” that can be added.
* The reverse SDE sampler is good enough to follow these combined scores.

**Strengths**

* Simple; works with off-the-shelf diffusion models.
* Enabled attribute binding and multi-object scenes beyond training distribution.

**Limitations**

* No explicit control of likelihood or density along the trajectory.
* Can fail badly when concepts are entangled (e.g. “dog AND horse”).
* Composition is heuristic: no guarantee composed score corresponds to true composite density.

**TB ∧ silicosis**

* You could train TB and silicosis conditional scores and sum them.
* But without guarantees on *latent disentanglement and alignment*, score-sum may put incompatible lesions in anatomically odd places, with no clear way to detect or correct it.

---

### 2.2 Compositional EBMs (Du et al. 2020) ([proceedings.neurips.cc][2])

**Method**

* Classical **Energy-Based Models**: (p(x) \propto \exp(-E_\theta(x))).
* Composition via **logical operators on energies**:

  * AND: sum energies of concepts (PoE)
  * OR, NOT via other energy transforms
* Sampling via MCMC / Langevin dynamics.

**Assumptions**

* Concept distributions correspond to *independent factors* in the true generative process.
* Combined energy still defines a well-behaved distribution you can sample from with MCMC.

**Strengths**

* Very clean semantics: PoE = intersection of high-probability regions.
* Logically expressive (AND, OR, NOT, nesting).

**Limitations**

* Training EBMs is hard and unstable.
* MCMC is expensive and slow; mixing is hard.
* No diffusion-specific structure; doesn’t exploit SDE machinery.

**TB ∧ silicosis**

* In principle ideal: TB and silicosis could be EBMs over lung images and AND = PoE = co-morbidity.
* Practically, you’d prefer diffusion-parameterized energies (RRR) to avoid training raw EBMs.

---

### 2.3 Reduce, Reuse, Recycle (RRR: energy-based diffusion + MCMC) ([arXiv][3])

**Method**

* Reparameterize diffusion as an **explicit energy function (f_\theta(x,t))** whose gradient is the score.
* Use this to:

  * Compose energies (as in EBMs),
  * Design **MCMC-style samplers** (MALA, HMC, Metropolis corrections) that better approximate the intended composite distribution.

**Key insight**

* Many failures of compositional diffusion come from the **sampler**, not the model: naive ancestral samplers don’t follow the correct composite density.

**Assumptions**

* Energy parameterization is expressive enough.
* MCMC steps are tuned and mix reasonably well.
* Underlying distributions still satisfy some factorization structure when you compose them.

**Strengths**

* More faithful sampling of composite densities than naive score-sum.
* Can implement AND/OR/NOT in a principled PoE fashion with Metropolis corrections.

**Limitations**

* Heavier sampling; per-step MCMC cost.
* Still inherits semantic assumptions from EBMs (independent factors, etc.).
* No explicit log-p along the SDE; you get better samples but not a closed-form density estimator.

**TB ∧ silicosis**

* Attractive if you want **very accurate sampling** of PoE densities for TB and silicosis models.
* Still doesn’t tell you *when* PoE equals “true co-morbidity”; that’s a structural question, not a sampling question.

---

## 3. Density-based composition: SUPERDIFF

### 3.1 Itô density estimator + superposition (Skreta et al. 2024) ([arXiv][4])

**Method**

* Derive an **Itô SDE for the log-density (\log q_t(x_t))** along the reverse diffusion trajectory.
* This gives a **pathwise estimator of (\log q_t(x_t))** that:

  * uses only forward drift and scores,
  * avoids explicit divergence computation (unlike probability-flow ODEs).
* For multiple pretrained models (q^{(i)}), maintain:

  * a **shared trajectory** (x_t),
  * an estimated log-density (\log q^{(i)}_t(x_t)) for each model.

**Composition operators**

* **OR (mixture)**: choose mixture weights via softmax over log-densities.
* **AND (equal-density)**: solve for weights / control so that log-densities are equal (or lie on a target manifold of equal density).

**Assumptions / preconditions**

* A **shared state / latent space** where all models define densities.
* SDE discretisation and Itô estimator are accurate enough (variance-bias trade-offs).
* For semantic AND, your equal-density set must correspond to meaningful co-occurrence in that space.

**Strengths**

* First compositional diffusion method to **explicitly track density** along an SDE.
* Gives **fine-grained control** over each model’s contribution at each step.
* Enables **density-aware AND/OR**, not just heuristic score arithmetic.
* Efficient: avoids expensive divergence computations.

**Limitations**

* Gives you a mathematically clean composite density, but **does not by itself guarantee semantic correctness**; you still need structural conditions (disentanglement, factorization).
* Sensitive to shared-space alignment: if models live in misaligned latent spaces, equal-density constraints may be meaningless.
* AND is defined via *density equality*, which is only a good proxy for semantic intersection if densities are structured sensibly.

**TB ∧ silicosis**

* Natural fit to your PhD setup:

  * Train TB and silicosis LDMs with a **shared VAE latent**.
  * Use SUPERDIFF’s AND to generate co-morbidity.
* Works best if:

  * latent space is **aligned & shared** (H1),
  * pathology factors are **disentangled from anatomy** (H2),
  * TB and silicosis behave like independent factors given shared lung structure (H3).

---

## 4. Theoretical analysis: Bradley et al. “Mechanisms of Projective Composition” ([arXiv][5])

**What they do**

* Define a precise target notion: **projective composition** (e.g. “dog wearing hat” as the projection of a high-dimensional joint onto image space).
* Analyze **when linear score combination** (Liu, Du, etc.) **actually achieves this**.

**Key findings**

* Introduce **Factorized Conditionals (FC)**: a structural condition on the data distribution that, roughly, says there exists a feature space where each concept controls independent coordinates.
* Show:

  * Under FC, linear score composition can be *exactly correct* for projective composition.
  * When FC fails (e.g. entangled attributes), score-combination can be arbitrarily wrong.
* Explain previously mysterious successes/failures (e.g. why style+content sometimes works but dog+horse doesn’t).

**Implications for *all* methods**

* Composition is not “free”: you need **disentangled features / factorization** for any algebra on log-p to match semantics.
* This theory doesn’t depend on SuperDiff vs RRR vs Liu — it’s about the structure of the underlying distribution.

**TB ∧ silicosis**

* FC translates to:

  * there exists a representation where **anatomy (lung structure)** and **pathology factors (TB lesions, silicosis nodules)** are cleanly factorized.
* If that’s false, *no* composition method can be guaranteed to give realistic co-morbidity without additional constraints.

---

## 5. Semantic compositionality: cross-cutting requirements

Across these methods, a few **structural preconditions** keep showing up:

1. **Shared / aligned representation (H1)**

   * All models must “see” data in a compatible coordinate system (shared latent or well-behaved pixel space).
   * Misalignment → density/score algebra behaves like adding apples and oranges.

2. **Local disentanglement of factors (H2)**

   * Concept dimensions (e.g. TB vs silicosis vs anatomy) should control roughly independent directions.
   * Without this, AND/PoE/equal-density constraints distort multiple factors at once.

3. **Conditional factorization for PoE semantics (H3)**

   * For PoE to equal “true joint concept”, concepts should be independent conditioned on shared structure.
   * For TB∧silicosis: lesions should be largely independent given lung structure.

4. **Compatible log-density geometry (H4)**

   * Composed densities inherit the **sum of log-p Hessians**; wildly different curvature can yield jagged, implausible modes.
   * Smoothness and curvature compatibility play into whether composition looks anatomically reasonable.

These are **method-agnostic**: they apply no matter whether you use SuperDiff, composable diffusion, EBMs, or RRR + MCMC.

---

## 6. TB ∧ silicosis: how the methods stack up

### 6.1 Composable diffusion (Liu et al., score sum) ([arXiv][1])

* **Setup**: two conditional models (TB, silicosis) → sum scores.
* **Pros**: easy to implement; leverages existing LDMs.
* **Cons**:

  * Heuristic; no explicit density control.
  * Strongly relies on FC-like disentanglement which is unlikely in raw radiographs.
  * No explicit mechanism to control contribution of each model beyond simple weights.
* **Risk**: high chance of anatomically implausible co-morbidity unless latent structure is unusually clean.

---

### 6.2 EBMs / RRR (energy-based diffusion + MCMC) ([arXiv][3])

* **Setup**: parameterize TB and silicosis densities as energies; compose via PoE; sample with MCMC-corrected diffusion samplers.
* **Pros**:

  * Principled PoE semantics; samplers can, in principle, converge to the true composite density.
  * Can be very faithful if energy parameterization and MCMC are good.
* **Cons**:

  * Expensive sampling; tricky MCMC tuning.
  * Still needs structural assumptions (factorization) for PoE to match real co-morbidity.
  * Harder to integrate with your existing LDM infrastructure.

---

### 6.3 SuperDiff (Itô log-density + equal-density AND) ([arXiv][4])

* **Setup**:

  * Two LDMs (TB, silicosis) sharing an autoencoder latent.
  * Use Itô log-density estimator to track each model’s log-p along a shared trajectory.
  * Define AND as equal-density or controlled weighting, OR as mixture.

* **Pros**:

  * Direct access to **log-density** along the path → fine-grained control.
  * Efficient; avoids divergence estimation.
  * Naturally matches your TB+silicosis latent architecture.
  * Can be extended with RRR-style samplers or FK-correctors if you later want more exactness.

* **Cons**:

  * Still requires **aligned & disentangled latent factors** for semantic correctness.
  * Equal-density ≠ true co-morbidity unless those structural assumptions hold.
  * Needs careful evaluation in medical setting (radiologist review, anatomical constraints).

---

## 7. So which method is best for TB & silicosis?

**Short answer:**

> For your silico-TB problem, **SuperDiff in a shared, well-aligned latent space is the most natural core method**, potentially augmented with RRR-style or FK-style corrections *and* robust property guidance.

**Why:**

* You already plan **separate TB and silicosis LDMs with a shared VAE latent**, which is exactly the regime where SuperDiff shines.
* It gives you explicit **log-density control** needed for:

  * logical AND (co-morbidity),
  * OR / mixtures,
  * controllable relative weighting (e.g., TB: 0.3, silicosis: 0.7).
* Combined with your hypotheses (H1–H4):

  * you can experimentally *test* when composition is plausible vs broken,
  * and tie failures back to latent misalignment or entanglement rather than “just bad sampling.”

**How the others fit in:**

* **Composable diffusion (Liu)** is a good *baseline* and a sanity check: if linear score-sum already fails badly on TB∧silicosis, you can show why your density-aware approach plus structural conditions does better.
* **Du 2020 + RRR** are more about *sampling correctness* once the target composite density is defined; they can be layered on top of SuperDiff’s density modeling if you later want even stricter adherence to PoE semantics, at the cost of more computation.

So your thesis story can be:

1. Use **SuperDiff** as the main compositional engine on TB + silicosis with a shared latent space.
2. Use **Bradley’s theory + your hypotheses** to reason about when this composition is structurally valid.
3. Use **RRR / FK-style samplers and robust guidance** as optional enhancements for sampling and control, especially when you add severity/location guidance.

That gives you a coherent, theoretically motivated position in the current landscape of compositional generative modeling.

[1]: https://arxiv.org/abs/2206.01714?utm_source=chatgpt.com "Compositional Visual Generation with Composable Diffusion Models"
[2]: https://proceedings.neurips.cc/paper_files/paper/2020/hash/49856ed476ad01fcff881d57e161d73f-Abstract.html?utm_source=chatgpt.com "Compositional Visual Generation with Energy Based Models"
[3]: https://arxiv.org/abs/2302.11552?utm_source=chatgpt.com "Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC"
[4]: https://arxiv.org/abs/2412.17762?utm_source=chatgpt.com "The Superposition of Diffusion Models Using the Itô ..."
[5]: https://arxiv.org/abs/2502.04549?utm_source=chatgpt.com "Mechanisms of Projective Composition of Diffusion Models"
