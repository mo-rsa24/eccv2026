# Slide Title

New Result: Pair-Type-Aware Joint Probe Analysis

## Slide Subtitle

The key correction was in the measurement protocol, not the backbone model

## On-Slide Text

- I moved to `pair-type-aware joint probes`, which write `joint_probe_scores.json`.
- The BLIP-VQA backbone stays the same.
- The important change is the probe bundle.
- The score now asks whether the image satisfies the right notion of `joint correctness` for that pair type.
- This aligns much better with the taxonomy and the actual failure modes.

## Suggested Slide Layout

- Large figure centered or right-aligned.
- Five bullets on the left.
- Add a small badge under the figure: `Main corrected output-level figure`.

## Figures To Include

- `paper/neurips/Comparing Semantic and Logical Composition Using Latent Diffusion Models/figures/joint_probe_grouped_bar.png`

## Figure Usage Notes

- Highlight the clearer group-wise ordering.
- Add a note near the figure: `pair-type-aware joint probes`.

## Mathematical Notation / Equations

$$
\text{joint correctness score} = s_{\mathrm{joint}}(x; A,B,\tau)
$$

where $\tau$ denotes pair type.

## Speaker Notes

- Say explicitly: the extension corrected what was being measured.
- This is the main new result slide.
