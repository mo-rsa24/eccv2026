# Slide Title

What the Probe Bundle Actually Measures

## Slide Subtitle

Positive probes plus anti-collapse probes

## On-Slide Text

- For factorised pairs, ask:
  - is the base content preserved?
  - is the second factor actually applied?
- For object-object, overlap, and collision pairs, ask:
  - are both concepts present as distinct realisations?
  - is either concept merged, absorbed, or dominated?
- The bundle combines:
  - positive probes,
  - `anti-collapse` probes,
  - a final aggregated `joint correctness` score.

## Suggested Slide Layout

- Two columns.
- Left: factorised pair probes.
- Right: hard-pair probes.
- Put the final score formula across the bottom.

## Figures To Include

- None required.

## Figure Usage Notes

- Use the following probe examples as slide text boxes:
  - `Does the image depict A?`
  - `Does the image also realize B?`
  - `Are there two distinct things, one A and one B, rather than a single merged object?`
  - `Is one of A or B merged into or absorbed by the other?`

## Mathematical Notation / Equations

$$
s_{\mathrm{joint}}(x) =
\frac{1}{K}
\left(
\sum_{i \in \mathcal{P}^{+}} s_i(x)

+ \sum_{j \in \mathcal{P}^{-}} \big(1 - s_j(x)\big)
\right)
$$

where $\mathcal{P}^{+}$ are positive probes and $\mathcal{P}^{-}$ are `anti-collapse` probes.

## Speaker Notes

- This is the slide that explains the logic of `joint_probe_scores.json`.
- Make clear that `anti-collapse` is what cue-presence scoring was missing.
