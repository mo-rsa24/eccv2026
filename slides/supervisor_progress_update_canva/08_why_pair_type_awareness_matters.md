# Slide Title

Why Pair-Type Awareness Matters

## Slide Subtitle

Different pair types fail differently, so they should not share one probe template

## On-Slide Text

- Different pair types have different valid notions of success.
- The probe bundle adapts to the compositional structure of the pair.
- Singleton conditions:
  - score intended marginal presence
- Factorised pairs:
  - score `content_preserved` + `factor_applied`
- Non-factorised / hard pairs:
  - score joint realisation + `anti-collapse`

## Suggested Slide Layout

- Left: compact taxonomy-to-probe table.
- Right: four bullets and one takeaway sentence.

## Figures To Include

- None required.

## Figure Usage Notes

- Build the table as the main visual:
  - `Factorized pairs -> content_preserved + factor_applied`
  - `Object/object or hard pairs -> a_present + b_present + distinct_entities + anti-collapse probes`

## Mathematical Notation / Equations

$$
s_{\mathrm{joint}}(x; A,B,\tau)
=
\frac{1}{|Q_\tau|}\sum_{q \in Q_\tau} s_q(x)
$$

where $Q_\tau$ is the probe bundle chosen for pair type $\tau$.

## Speaker Notes

- This slide should feel methodological, not defensive.
- It explains why the probe bundle is principled rather than ad hoc.
