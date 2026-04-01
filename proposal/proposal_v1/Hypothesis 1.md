Paper: Compositional Generative Modeling A Single Model is Not All You Need
Below is a **precise, paper-aligned classification** of the silico-TB setting, showing whether it belongs to *learned* compositional structure or *explicit / a-priori* compositional structure, and how you should construct the factors.

Everything is grounded in the compositional operators described in *Compositional Generative Modeling: A Single Model Is Not All You Need*. 

Propose - Medical images are not naturally factorized into disease latents—so learning compositional structure is unreliable.

---

# **1. Where does the silico-TB problem fit in the framework of the paper?**

## **Short answer:**

**Silico-TB should be modeled as *explicit / a-priori* compositional structure**, *not* as learned compositional structure.

This is because:

1. **We already know the meaningful factors** (pathologies, anatomy).
2. **We do not want the model to discover arbitrary latent factors** (orientation, marker artifacts, texture statistics) because these are *not* the medically relevant axes of variation.
3. **Incorrect latent discovery could entangle or miss pathologies**, leading to compositional errors.

Thus **silico-TB falls under the “Factorization / Structured Composition” category, not “Learned Compositional Structure.”**

This matches Section 2 of the paper, where factorization is motivated by **a known independence structure**:

> “If we *know* that a distribution exhibits an independence structure… we can substantially reduce the data requirements by learning factors and composing them.” — Section 2. 

---

# **2. How the paper would classify your proposed factorization**

Your decomposition:

* **A** = Silicosis
* **B** = TB
* **C, D** = anatomical structure + background variation (lungs, ribs, soft tissue, device artifacts, etc.)

fits the **Exact form of the factorized distribution in the paper’s Section 2**:

[
p(A, B, C, D) \propto p(A), p(B), p(C,D)
]

The paper calls this a **product factorization over independent or weakly coupled variables**.

Compare with the paper’s exemplar:

[
p(A,B,C,D) \propto p(A), p(B), p(C,D)
]

> “We can substantially reduce data requirements by learning these factors and composing them.” (Section 2). 

This is exactly your case:

* **Pathologies are independent factors**
* **Background/anatomy is a separate factor**

---

# **3. Why not learned compositional structure?**

Section 4 of the paper describes **unsupervised discovery**:

* objects,
* relational forces,
* classes,
* or other latent components.

But these discovered components are **not guaranteed** to align with clinically meaningful factors.

They often correspond to:

* pose,
* lighting,
* noise/artifacts,
* view angle,
* dominant texture directions,
* machine-specific variation,
* irrelevant confounders.

The paper itself warns about this:

> “Current work on discovering compositional structure assumes data is naturally factorized… real data often exhibits *spurious correlations*… causing algorithms to fail to discover the correct structure.” (Section 6, limitations). 

This is exactly what you want to **avoid** in a medical setting.

---

# **4. So what operator would you actually use for silico-TB?**

From the taxonomies in the paper, the correct operator is:

## **→ Product Composition of Conditional EBMs or Classifier Guidance (explicit constraint factors)**

This corresponds to Section 3: **Generalization via product of densities**:

[
q(x) \propto p_{\text{base}}(x) ; r_{\text{constraint}}(x)
]

In your setting:

### **Base model**

(p_{\text{anatomy}}(x)) — a generative model for normal CXR anatomy.

### **Pathology models**

Define two “constraint” energy factors:

* (r_{\text{TB}}(x))
* (r_{\text{Silicosis}}(x))

Then compose:

[
q(x) \propto p_{\text{anatomy}}(x); r_{\text{TB}}(x); r_{\text{Silicosis}}(x)
]

This is **exactly the planning composition form**:

[
q(\tau) \propto p(\tau), r(\tau, g)
]

from Figure 7.
Here, the pathologies **play the role of explicit constraints**.

---

# **5. Is this equivalent to classifier guidance modeled as EBMs?**

### **Yes.**

This is the *most direct and paper-aligned* way to implement explicit composition.

The paper uses EBMs to encode constraint-like factors:

> “Each constraint can be represented as a low-dimensional factor… sampling from the composition corresponds to solving the constraint problem.” (Section 3). 

Your pathology labels (y_{TB}, y_{Silicosis}) can be represented as:

[
r_{TB}(x) = \exp(-E_{TB}(x)), \qquad r_{Silicosis}(x) = \exp(-E_{Sil}(x))
]

Which is identical to **classifier guidance** in diffusion models / EBMs.

Then the composed target distribution is:

[
p(x \mid TB, Silicosis) \propto p_{\text{base}}(x)\exp(-E_{TB}(x))\exp(-E_{Sil}(x))
]

This is literally Equation (2) in the paper:

[
e^{-(E_1(x)+E_2(x))}
]

i.e., **product of densities = sum of energy functions**.

---

# **6. Why explicit composition is the right choice for pathology modeling**

Because:

* We know the clinically meaningful axes (pathologies).
* We know they can co-occur (TB + silicosis comorbidity).
* We want explicit control (generate with TB only, silicosis only, both, or neither).
* Unsupervised factor discovery would identify *nonclinical* factors.
* Discovered latents would not align with pathology boundaries.

This matches the paper’s key point:

> “In many real-world settings, gathered data will often exhibit **spurious correlations**… causing algorithms to fail to discover the correct structure.” (Section 6). 

Medical images are *not* naturally factorized into disease latents—so *learning* compositional structure is unreliable.

---

# **7. Final classification**

| Approach                               | Paper category | Appropriate for silico-TB? | Reason                                                            |
| -------------------------------------- | -------------- | -------------------------- | ----------------------------------------------------------------- |
| **Learned compositional structure**    | Section 4      | ❌ NO                       | Learns object-like or texture-like factors unrelated to pathology |
| **Explicit factorization**             | Section 2      | ✔ YES                      | You know the pathology factors a priori                           |
| **Constraint-based composition**       | Section 3      | ✔ YES                      | TB and silicosis act as explicit constraints                      |
| **Classifier-guidance as EBM factors** | Section 5      | ✔ YES                      | Direct implementation of pathology factors in product form        |

---

# **TL;DR (most important sentence)**

**Silico-TB belongs to explicit factorized composition (Section 2 + Section 3). You should not use learned compositional structure; instead, represent each pathology as its own explicit EBM/classifier-guided constraint and compose them multiplicatively.**

---

If you'd like, I can also help you construct:

* the exact energy functions for TB and silicosis,
* the full generative pipeline diagram,
* the factor graph representation,
* a recommended architecture (diffusion + EBM constraints).
