# Slide Title

BLIP-VQA: What It Is and What It Measures

## Slide Subtitle

Why it was the first output-level diagnostic

## On-Slide Text

- `BLIP-VQA` is a vision-language question answering model.
- In this repo, it is used as a concept-presence diagnostic, not as the final compositionality metric.
- For each image, we ask two yes/no questions:
  - `Is there A in the image?`
  - `Is there B in the image?`
- The evaluation records:
  - `p_c1 = P(A present)`
  - `p_c2 = P(B present)`
- Ordered conditions used in the script:
  - `c1`, `c2`, `mono`, `poe`, `pstar_sdipc`
- Slide order used in the grouped bars:
  - `Solo A`, `Solo B`, `Mono (A∧B)`, `PoE`, optional `PoE p*`

## Suggested Slide Layout

- Two-column explainer slide.
- Left: `What BLIP-VQA is` and the scoring equation.
- Right: prompt template, ordered conditions, and optional architecture figure reference.
- If space allows, add a small thumbnail of `blip_vqa_grouped_bar.png` at the bottom-right.

## Figures To Include

- Optional local figure:
  - `paper/neurips/Comparing Semantic and Logical Composition Using Latent Diffusion Models/figures/blip_vqa_grouped_bar.png`
- Optional external architecture figure:
  - `BLIP paper, Figure 2`
  - Source: `https://proceedings.mlr.press/v162/li22n/li22n.pdf`
  - Path to verify: external paper figure, not stored in this repo

## Figure Usage Notes

- If using the local grouped-bar figure:
  - keep it small
  - label it `output-level cue-presence diagnostic`
- If using the BLIP paper figure:
  - use Figure 2 only as a model overview
  - cite `Li et al., 2022`
  - verify image reuse requirements before copying it into the final deck
- Do not let this slide become a BLIP paper summary. Keep the focus on how BLIP-VQA is used in this project.

## Mathematical Notation / Equations

$$
p_{\mathrm{yes}}(x,q)=
\frac{\exp(\ell_{\mathrm{yes}}(x,q))}
{\exp(\ell_{\mathrm{yes}}(x,q))+\exp(\ell_{\mathrm{no}}(x,q))}
$$

where $q$ is a yes/no question and $\ell_{\mathrm{yes}}, \ell_{\mathrm{no}}$ are the first-token logits for `yes` and `no`.

For this evaluation:

$$
p_{c1}(x)=p_{\mathrm{yes}}(x,\text{``Is there }c_1\text{ in the image?''})
$$

$$
p_{c2}(x)=p_{\mathrm{yes}}(x,\text{``Is there }c_2\text{ in the image?''})
$$

Optional summary score for presentation purposes:

$$
s_{\mathrm{cue}}(x; A,B)=\frac{1}{2}\big(p_{c1}(x)+p_{c2}(x)\big)
$$

Exact question template used in the repo:

$$
\texttt{"Is there \{concept\} in the image? Answer yes or no."}
$$

## Speaker Notes

- Repo implementation:
  - model id: `Salesforce/blip-vqa-base`
  - script: `scripts/eval_blip_vqa.py`
  - output file: `blip_vqa_scores.json`
- The code computes `P("yes")` by taking a softmax over the first-token logits for `yes` vs `no`.
- This makes BLIP-VQA a clean cue-presence probe.
- The limitation, explained on the next slide, is that cue presence is not the same as `joint correctness`.
