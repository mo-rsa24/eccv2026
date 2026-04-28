# Slide Title

Why Cue-Presence Was Not Enough

## Slide Subtitle

Presence is not the same as joint correctness

## On-Slide Text

- BLIP-VQA concept-presence scoring asks:
  - `Is there A in the image?`
  - `Is there B in the image?`
- That is adequate for marginal presence.
- It is not adequate for joint compositional correctness.
- A hybrid image can answer `yes` to both questions and still fail composition.
- This matters most in hard regimes:
  - merged objects,
  - absorbed attributes,
  - one concept dominating the other.

## Suggested Slide Layout

- Left: mini two-row comparison table.
- Right: five bullets.

## Figures To Include

- None required.

## Figure Usage Notes

- Use a simple text box or table:
  - `Cue present? yes`
  - `Jointly realised correctly? not necessarily`

## Mathematical Notation / Equations

$$
\big(P(A \mid x)\ \text{high} \;\wedge\; P(B \mid x)\ \text{high}\big)
\centernot\implies
\text{joint correctness}(x; A,B)
$$

## Speaker Notes

- This is the bridge from BLIP-VQA to `pair-type-aware joint probes`.
- Stress that the failure is specification-level, not implementation-level.
