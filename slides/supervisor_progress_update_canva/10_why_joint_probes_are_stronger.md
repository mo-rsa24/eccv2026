# Slide Title

Why the New Result Is Easier to Interpret

## Slide Subtitle

`blip_vqa_grouped_bar.png` vs `joint_probe_grouped_bar.png`

## On-Slide Text

- `blip_vqa_grouped_bar.png` measures cue presence.
- `joint_probe_grouped_bar.png` measures `joint correctness`.
- The new probe bundle explicitly penalises:
  - collapse,
  - merged-object behaviour,
  - concept dominance.
- Hard cases therefore look less artificially successful.
- This makes the result easier to defend.

## Suggested Slide Layout

- Side-by-side figure comparison.
- Short caption underneath spanning the slide.
- Keep bullets above or to the left.

## Figures To Include

- `paper/neurips/Comparing Semantic and Logical Composition Using Latent Diffusion Models/figures/blip_vqa_grouped_bar.png`
- `paper/neurips/Comparing Semantic and Logical Composition Using Latent Diffusion Models/figures/joint_probe_grouped_bar.png`

## Figure Usage Notes

- Left figure label: `Cue presence`
- Right figure label: `Joint correctness`
- Add caption:
  - `Presence-based scoring overstates hard cases; pair-type-aware scoring tracks interpretable joint success more faithfully`
- Highlight Groups 3 and 4 in both plots.

## Mathematical Notation / Equations

$$
s_{\mathrm{cue}}(x) \quad \text{vs} \quad s_{\mathrm{joint}}(x; A,B,\tau)
$$

## Speaker Notes

- This should be one of the most important slides in the deck.
- The contrast should look immediate even before you speak.
