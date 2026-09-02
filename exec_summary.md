# SKELETON — not the submission

Every `[?]` is a number or claim I do not have yet. Two rules for using this file:

1. **Scope guard.** If an experiment I am tempted to run does not fill in a `[?]`
   somewhere below, I do not run it. I have 12-20 hours.
2. **Voice.** The prose here is placeholder. I rewrite every paragraph in my own
   words before submitting. A write-up that reads like model output gets rejected.

Target length when filled: 1-2 pages. It has to stand alone — assume it is the
only thing that gets read, by someone with mech interp experience but zero
context on my project.

---

## Title

[?] Working version: "The refusal probe survives detection but not compliance:
stress-testing prefill refusal readouts under adversarial attack in Qwen3-4B"

---

## What problem am I trying to solve, and why is it interesting?

In reasoning models, whether the model will end up refusing is already linearly
decodable from the hidden state at the first thinking token — 0.84-0.95 AUROC
across Qwen3-8B, Olmo-3-7B-Think, Phi-4-Reasoning and GPT-OSS-20B
("Do Thinking Tokens Help with Safety?", arXiv 2606.25013). The decision appears
to be made before any visible deliberation.

That result is measured on plain harmful and benign prompts. The paper says so in
its own limitations: *"our evaluation is also limited to refusal/compliance
behavior on harmful and benign prompt sets."*

So: nobody has checked whether the probe still works when someone is actively
trying to break the model.

Why I care about the answer:

- **Practical.** An internal probe is nearly free — one dot product against a
  forward pass I am already running, versus standing up a separate guard model.
  If it holds under attack it is the cheapest guardrail available. If it does
  not, a lot of activation-monitoring proposals are resting on an untested
  assumption.
- **Mechanistic.** *Where* the probe fails tells me which stage of refusal a
  given attack targets. Refusal is at least two steps: notice the request is
  harmful, then act on it. An attack can defeat either one, and the probe
  distinguishes them.

My question, in one sentence: [?]

---

## Setup

- **Model.** Qwen3-4B, 36 layers, d_model 2560. Picked specifically because it
  has a hard `enable_thinking` switch, so I can toggle deliberation on
  *identical weights* rather than comparing different models with different
  post-training.
- **Direction.** Difference-in-means, harmful minus harmless, at the last
  prefill token. Layer [?] of 36, selected by [?]. Cross-validated Cohen's
  d = [?] +/- [?] on held-out folds.
- **Attacks.** [?] families x [?] targets = [?] attempts, each run with thinking
  ON and OFF. Families: roleplay, prefix injection, pseudo-suffix (hand-written,
  *not* GCG — see limitations).
- **Outcome labels.** LLM judge, StrongREJECT-style rubric, applied to the final
  answer only (post-`</think>`). Cohen's kappa = [?] against 50 hand labels.

---

## High-level takeaways

Written as predictions now so that filling them in is honest rather than
retrofitted. If a prediction is wrong I say so and keep the original.

1. **[?] Headline.** Predicted: probe AUROC falls from [?] unattacked to [?]
   under attack, and the fall is uneven across families.
2. **[?] Mechanism.** Predicted: most successful attacks are *overpower*, not
   *stealth* — the probe fires, the model complies anyway. Pilot data pointed at
   84-100% overpower. If that holds, detection is robust and the thing that
   breaks is the link from detection to refusal.
3. **[?] Deliberation.** Predicted: toggling thinking leaves prefill AUROC
   roughly unchanged, consistent with thinking being prefix completion rather
   than deliberation.
4. **[?] Causality.** Predicted: adding the refusal direction rescues refusal in
   stealth cases but not overpower cases.

Most interesting thing I found: [?]

Most surprising thing I found: [?] — and if nothing surprised me, say that
plainly. A project where everything went as predicted is information about the
prediction being cheap.

---

## Key experiments

One paragraph and one graph each. For each, I write down in advance what the
boring explanation is, and what result would change my mind.

### E1. Is there a refusal direction in this model at all?

- **Why.** Table stakes. If harmful and harmless prompts do not separate, nothing
  downstream is interpretable.
- **Found.** [?]
- **Graph.** Projection histograms, harmful vs harmless, layer [?].
- **Boring explanation.** It is a prompt-length or topic direction, not a refusal
  direction. Ruled out by [?].
- **Note.** This is not a finding. "Safety concept is linearly represented" is a
  known result and I am not claiming it. It exists to license E2-E5.

### E2. Does the probe still predict outcomes under attack?

- **Why.** The core question.
- **Found.** [?]
- **Graph — this is the key figure.** AUROC by attack family, thinking ON vs OFF,
  with my own unattacked AUROC drawn as a horizontal reference line so the
  degradation is visible without reading numbers.
- **Baselines.** Chance = 0.50. My unattacked condition = [?]. Prompt-length
  confound: correlation of projection with token count, r = [?].
- **What would change my mind.** If AUROC is flat across families and close to
  the unattacked number, H1 is dead and the story becomes "the probe is
  surprisingly robust" — which is a fine result and I report it as such.

### E3. When it fails, how does it fail — stealth or overpower?

- **Why.** Separates the two stages of refusal.
- **Found.** [?]
- **Graph.** Prefill projection vs outcome, colored by family, with the
  calibrated threshold marked.
- **Threshold.** Youden-J from held-out folds, *not* the harmful/harmless
  midpoint. The midpoint sat at -26.2 in my pilot, far enough below zero that
  almost everything counted as "detected" and the overpower rate was inflated by
  construction.
- **What would change my mind.** A roughly even stealth/overpower split, or
  stealth dominating, would invert takeaway 2.

### E4. Is the direction causally doing the work — and differentially?

- **Why.** A probe is correlational. This is the causal test.
- **Design.** (a) Ablate the direction on clean harmful prompts; expect
  compliance to rise, replicating known results and validating the direction.
  (b) *Add* the direction during attacks and measure the rescue rate separately
  for stealth and overpower cases.
- **Found.** [?] rescue on stealth, [?] on overpower.
- **Graph.** Rescue rate by pre-intervention class.
- **Why differential matters.** The two-stage story predicts a *gap*: if the
  probe already fired, pushing detection harder should not help, because
  detection was never the broken part. A single global steering number cannot
  distinguish that from a plain dose-response.
- **What would change my mind.** Equal rescue in both classes means the
  two-stage framing is wrong and I should describe it as monotone steering
  strength instead.

### E5. Does thinking help?

- **Why.** Within-model deliberation ablation, identical weights.
- **Found.** [?] attack success rate with thinking ON vs [?] OFF; prefill AUROC
  [?] vs [?].
- **Graph.** Mean projection trajectory across normalized CoT position, grouped
  by family and eventual outcome.
- **What would change my mind.** If projection moves substantially *during* the
  CoT and the final outcome tracks that movement rather than the prefill value,
  then deliberation is real here and takeaway 3 is wrong.

---

## Controls and sanity checks

- **Attack-wrapped harmless prompts.** [?] Necessary because attack wrappers are
  far longer than my extraction prompts — without this control I cannot separate
  "the attack changed the projection" from "long wrapper text changed the
  projection." The guardrail interpretation of the whole result depends on this
  coming back clean.
- **Prompt length vs projection.** r = [?]
- **Judge vs 50 hand labels.** kappa = [?]. Judge vs old keyword labeler: [?]
- **Truncated generations.** [?] rows never emitted `</think>`; reported
  separately, not silently labeled.
- **Direction stability across thinking modes.** cosine similarity = [?]

---

## What went wrong (keeping this in on purpose)

My first run labeled outcomes by keyword-matching the first 50 generated tokens.
Qwen3 thinks by default, so those 50 tokens were almost entirely the reasoning
trace — I was labeling the model's private deliberation instead of its answer.
36 of 150 rows came back "ambiguous" and my first instinct was to expand the
keyword list, which papered over the symptom. The `<think>` tags were visible in
the output I printed to inspect those rows; I had the evidence and did not follow
it for [?] hours.

Before: [?]. After: [?].

Lesson: [?]

---

## Limitations

- One model, one size. No claim about generalization.
- The "adversarial suffix" family is hand-written compliance bait, not
  GCG-optimized. This is the weakest part of the design, and it matters
  specifically because my original hypothesis predicted that *suffixes* would be
  the stealthy family. I cannot test that claim with these strings, and I say so
  rather than letting the family name imply otherwise.
- One sample per row at temperature 0.6. No variance estimate over sampling.
- Judge reliability is kappa = [?]; all outcome-dependent numbers inherit that.
- Direction fit on extreme-harm prompts. Probably does not transfer to borderline
  or dual-use content, which is where real deployments live.
- n per family is small; all AUROCs carry bootstrap CIs and I do not read
  differences smaller than those CIs.

---

## Dual use

[?] Short and honest. Something that predicts which attacks work is useful for
building defenses and useful for building attacks. State the asymmetry I think
applies here — [?] — and note that I am not releasing optimized attack strings.

---

## Where I would go next

The measurement in this project is not really about refusal. It is: does the
model internally represent X while outputting not-X? Refusal is just the case
where I have clean ground truth for both halves. The same instrument should
apply to sandbagging, evaluation-awareness, and resistance to secret
elicitation — [?] one paragraph on which of these I would try first and what the
cheapest version of that experiment looks like.
