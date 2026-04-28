# Supervisor Progress Update Presentation Plan

## Summary

This version should minimise the timeline justification and put the intellectual weight on the new result. The delay itself should be handled briefly and professionally in one bullet only. The main narrative should explain why the unexpected Group 2 trajectory forced a methodological check, why a cue-presence metric was not enough, and why the pair-type-aware joint-probe result is now the more convincing interpretive layer for the paper.

The central line of argument should be: the extension was needed because `group_2_trajectory_2x2_sdipc.png` exposed a mismatch between geometric separation and qualitative compositional success, and the new joint-probe analysis resolves that mismatch by measuring the right thing.

## Slide 1 — Title and message

**Title:** `Progress Update: Interpreting the Compositionality Gap More Reliably`

**Opening message**
- I paused the write-up because one key result exposed an unresolved interpretability problem in the evidence, and I needed to resolve that before locking the paper narrative.

**Speaker note**
- Keep this to one bullet only, then move on.

## Slide 2 — Where the issue appeared

**Section heading:** `Unexpected Result in Group 2`

**Key talking points**
- The original expectation was `G1 ≈ G2 < G3 < G4`.
- In `group_2_trajectory_2x2_sdipc.png`, `A ^ B` appeared substantially farther from PoE than expected.
- This was surprising because Group 2 should be comparatively well behaved under the factorised/disentangled regime.

**Insert**
- `group_2_trajectory_2x2_sdipc.png`
- Annotate the unexpected `A ^ B` versus PoE separation.

**Speaker note**
- Present this as the empirical trigger for the extension.

## Slide 3 — Why this mattered scientifically

**Section heading:** `Why This Could Not Be Ignored`

**Key talking points**
- The trajectory plot suggested strong geometric separation.
- But qualitatively, PoE still looked reasonably successful.
- This created a mismatch between what the 2D trajectory visualisation suggested and what the images appeared to show.
- That forced a methodological question:
  - are MDS trajectory plots sufficient on their own to expose the compositionality gap we actually care about?

**Insert**
- Short contrast box:
  - `MDS view: large separation`
  - `Qualitative view: PoE still plausible`

**Speaker note**
- This is the pivot of the talk. Emphasise that the extension came from measurement validity, not scope drift.

## Slide 4 — Why MDS alone was insufficient

**Section heading:** `What MDS Can and Cannot Tell Us`

**Key talking points**
- MDS is useful for visualising geometry, not for directly adjudicating semantic correctness.
- A large 2D separation does not by itself tell us whether both concepts are jointly and correctly realised.
- It does not cleanly distinguish:
  - successful joint realisation,
  - benign displacement,
  - merged hybrids,
  - concept dominance.
- Therefore MDS remains useful as mechanistic evidence, but not as a sufficient interpretive endpoint.

**Insert**
- Compact notation: `z_t in high-dimensional latent space -> 2D MDS projection`
- Optional one-line equation or note about dimensionality reduction.

**Speaker note**
- The phrasing should be careful: the issue is not that MDS is wrong, but that it is incomplete for this question.

## Slide 5 — First follow-up and its limitation

**Section heading:** `First Attempt: BLIP-VQA Cue-Presence Analysis`

**Key talking points**
- I first looked at `blip_vqa_grouped_bar.png` because it adds an output-level signal.
- This metric asks whether the relevant concepts are present.
- That was directionally useful, but it turned out to be too weak for hybrid failure cases.
- In practice, it appears to inflate compositionality in Groups 3 and 4.

**Insert**
- `blip_vqa_grouped_bar.png`

**Speaker note**
- Keep this slide short and transitional. It sets up why a second extension was needed.

## Slide 6 — Why cue presence was the wrong target

**Section heading:** `Why Cue-Presence Was Not Enough`

**Key talking points**
- BLIP-VQA concept-presence scoring answers questions of the form:
  - "Is there A in the image?"
  - "Is there B in the image?"
- That is adequate for marginal presence, but not for joint compositional correctness.
- A hybrid image can still trigger "yes" for both concepts while failing the real compositional test.
- This is especially problematic in difficult regimes, where failure often appears as:
  - merged objects,
  - absorbed attributes,
  - one concept dominating the other.

**Insert**
- Very small example table:
  - `Cue present? yes`
  - `Jointly realised correctly? not necessarily`

**Speaker note**
- This should prepare the supervisors for why the new result is not just another metric, but a better-specified one.

## Slide 7 — New result: pair-type-aware probe bundle

**Section heading:** `New Result: Pair-Type-Aware Joint Probe Analysis`

**Key talking points**
- I then moved to the pair-type-aware probe bundle, which writes `joint_probe_scores.json`.
- The key change is not the BLIP backbone itself, but the measurement protocol.
- Instead of asking only whether concepts are present, it asks whether the image satisfies the appropriate notion of joint correctness for that pair type.
- This makes the metric align much more closely with the taxonomy and the actual failure modes.

**Insert**
- `joint_probe_grouped_bar.png`
- One sentence under figure: `Main corrected output-level figure`

**Speaker note**
- Make clear that the extension is a correction in what is being measured.

## Slide 8 — Rationale of the pair-type-aware probe bundle

**Section heading:** `Why Pair-Type Awareness Matters`

**Key talking points**
- Different pair types fail in different ways, so they should not all be evaluated with the same probe template.
- The probe bundle adapts the questions to the compositional structure of the pair:
  - singleton conditions are scored for intended marginal presence,
  - factorised pairs are scored for preserved content plus applied factor,
  - non-factorised pairs are scored for joint realisation and anti-collapse behaviour.
- This means the evaluation respects the difference between:
  - factor application,
  - distinct entity co-realisation,
  - merged or absorbed failure.

**Insert**
- A compact taxonomy-to-probe table:
  - `Factorized pairs -> content_preserved + factor_applied`
  - `Object/object or hard pairs -> a_present + b_present + distinct_entities + anti-collapse probes`

**Speaker note**
- This slide should carry the methodological rationale in detail.

## Slide 9 — How the probe bundle is constructed

**Section heading:** `What the Probe Bundle Actually Measures`

**Key talking points**
- For factorised pairs, the relevant question is:
  - is the base content preserved, and is the second factor actually applied?
- For object-object, overlap, and collision pairs, the relevant question is stronger:
  - are both concepts present as distinct realisations?
  - has one concept been merged into, absorbed by, or dominated by the other?
- The probe bundle therefore includes both positive probes and anti-collapse probes.
- The final image-level score aggregates these into a joint-correctness score.

**Insert**
- Suggested probe examples:
  - `Does the image depict A?`
  - `Does the image also realize B?`
  - `Are there two distinct things, one A and one B, rather than a single merged object?`
  - `Is one of A or B merged into or absorbed by the other?`

**Speaker note**
- This is the slide where you explain the thinking, not just the result.

## Slide 10 — Why this is stronger than BLIP-VQA grouped bars

**Section heading:** `Why the New Result Is Easier to Interpret`

**Key talking points**
- `blip_vqa_grouped_bar.png` measures cue presence.
- `joint_probe_grouped_bar.png` measures joint correctness.
- Because the new probe bundle explicitly penalises collapse, merged-object behaviour, and dominance, it is much harder for difficult cases to look artificially successful.
- This is why the new result aligns better with the expected behaviour pattern and is easier to defend.

**Insert**
- Side-by-side comparison:
  - `blip_vqa_grouped_bar.png`
  - `joint_probe_grouped_bar.png`
- Caption line:
  - `Presence-based scoring overstates hard cases; pair-type-aware scoring tracks interpretable joint success more faithfully`

**Speaker note**
- This should be one of the most important slides in the meeting.

## Slide 11 — What the new result says substantively

**Section heading:** `Interpretation of the New Result`

**Key talking points**
- The joint-probe result recovers a behaviour pattern that is much closer to the expected taxonomy.
- It supports the claim that the earlier ambiguity came from the insufficiency of the original measurement layer, not from the collapse of the overall theory.
- It therefore reconciles the tension between:
  - trajectory geometry,
  - qualitative outputs,
  - group-level compositional interpretation.

**Insert**
- `joint_probe_grouped_bar.png` again if needed, with the expected ordering highlighted.

**Speaker note**
- State plainly that this result is now the preferred output-level evidence.

## Slide 12 — Revised paper narrative

**Section heading:** `How This Changes the Paper Narrative`

**Key talking points**
- The paper should treat MDS trajectories as mechanistic evidence about geometry and divergence.
- It should treat the pair-type-aware joint probes as the main output-level interpretive check.
- The two together give a stronger argument:
  - trajectories show where behaviour diverges,
  - joint probes show whether that divergence corresponds to meaningful compositional success or failure.

**Insert**
- Two-layer schematic:
  - `Trajectory analysis -> geometric mechanism`
  - `Joint probes -> interpretable compositional correctness`

**Speaker note**
- This slide turns the extension into a cleaner paper architecture.

## Slide 13 — Close

**Section heading:** `Takeaway`

**Key talking points**
- The extension was necessary because the original evidence was not yet decision-safe.
- The important outcome is not simply that there is another figure, but that the new figure measures the right semantic question.
- I can now resume the write-up with a more defensible explanation of the compositionality gap.

**Speaker note**
- End on confidence in the revised evidence chain, not on delay.

## Test Plan

- Check that the delay explanation appears only once and only as one bullet.
- Check that most of the time is spent on:
  - the Group 2 anomaly,
  - the failure of cue-presence scoring,
  - the rationale of the pair-type-aware probe bundle,
  - why `joint_probe_grouped_bar.png` is the stronger result.
- Check that the plan clearly distinguishes:
  - BLIP-VQA backbone,
  - measurement protocol,
  - joint-correctness aggregation.
- Check that a supervisor could answer this question after the talk:
  - "Why is `joint_probe_grouped_bar.png` more convincing than `blip_vqa_grouped_bar.png`?"
- Check that the pair-type-aware probe slide makes the reasoning explicit enough that it does not sound like an ad hoc metric swap.

## Assumptions

- Assume the supervisors do not need a long project-management explanation for the delay.
- Assume the main audience need is a serious methodological justification and a clear interpretation of the new result.
- Assume the presentation should use the repo's own language:
  - `pair-type-aware joint probes`
  - `joint_probe_scores.json`
  - `joint correctness`
  - `anti-collapse`
- Default emphasis:
  - one brief bullet on delay,
  - several slides on why the new result was needed and why it is stronger.
