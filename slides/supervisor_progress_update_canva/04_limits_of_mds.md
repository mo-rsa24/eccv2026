# Slide Title

What MDS Can and Cannot Tell Us

## Slide Subtitle

Useful mechanistic evidence, but not a sufficient interpretive endpoint

## On-Slide Text

- The trajectory panels are built from the full denoising trajectories under shared noise.
- Quantitatively, the paper tracks the per-step latent gap between Mono and PoE.
- Visually, MDS gives a 2D projection of those high-dimensional trajectories.
- That makes MDS useful for geometry.
- It does not directly answer whether both concepts are jointly and correctly realised.
- It does not cleanly separate:
  - successful joint realisation,
  - benign displacement,
  - merged hybrids,
  - concept dominance.

## Suggested Slide Layout

- Left: one equation block for the actual latent-gap metric and one short line for the MDS view.
- Right: bullets under `What MDS cannot disambiguate`.

## Figures To Include

- None required.

## Figure Usage Notes

- If desired, add a simple custom schematic:
  - full trajectories in latent space,
  - pairwise gap tracked by $d_t^{(p,s)}$,
  - MDS used only for 2D visualisation.
- Avoid another heavy plot here. This is a clarification slide about what the figure is and is not doing.

## Mathematical Notation / Equations

$$
d_t^{(p,s)}=\frac{1}{N}\left\|z_t^{\mathrm{mono},(p,s)}-z_t^{\mathrm{poe},(p,s)}\right\|_2^2
$$

$$
d_T^{(p,s)} = d_t^{(p,s)}\big|_{t=T}
$$

For the trajectory figure itself:

$$
\{z_t^{A}, z_t^{B}, z_t^{A \wedge B}, z_t^{\mathrm{poe}}\}
\;\longrightarrow\;
\{\tilde{z}_t^{A}, \tilde{z}_t^{B}, \tilde{z}_t^{A \wedge B}, \tilde{z}_t^{\mathrm{poe}}\}\in\mathbb{R}^2
$$

Takeaway line:

$$
\text{MDS visualises trajectory geometry; } d_t^{(p,s)} \text{ is the actual latent-space metric.}
$$

## Speaker Notes

- Use the paper's notation here.
- The key distinction is:
  - the quantitative analysis is based on latent distances,
  - the MDS panel is a qualitative projection of the trajectories.
- The issue is not that MDS is wrong.
- The issue is that MDS alone cannot adjudicate joint correctness.
