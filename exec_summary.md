# The refusal probe still sees the harm. It does not reliably see the compliance.

Stress-testing prefill refusal readouts under attack in Qwen3-4B.

## What problem am I trying to solve, and why is it interesting?

In reasoning models, whether the model will refuse is already linearly
decodable from the hidden state at the first thinking token — 0.84–0.95
AUROC across Qwen3-8B, Olmo-3-7B-Think, Phi-4-Reasoning and GPT-OSS-20B
("Do Thinking Tokens Help with Safety?", arXiv 2606.25013). The decision
looks like it is made before any visible deliberation.

That result is on plain harmful and benign prompts. The paper says so:
*"our evaluation is also limited to refusal/compliance behavior on
harmful and benign prompt sets."*

Nobody has checked whether the probe still works when someone is trying
to break the model.

I care about two things:

- **Practical.** A probe is one dot product against a forward pass I am
  already running. If it holds under attack it is a cheap guardrail. If
  it does not, activation-monitoring proposals are resting on an untested
  assumption.
- **Mechanistic.** Refusal is at least two steps: notice the request is
  harmful, then actually refuse. An attack can defeat either step, and a
  probe distinguishes them.

My question: on Qwen3-4B, does the end-of-prefill refusal direction still
predict behavior under attack, and does toggling `enable_thinking` change
the answer?

## Setup

- **Model.** Qwen3-4B, 36 layers, d_model 2560. I picked it because it
  has a hard `enable_thinking` switch, so I can turn deliberation on and
  off on identical weights.
- **Direction.** Difference-in-means, harmful minus harmless, at the last
  prefill token. I locked one common layer (layer 35 of 36) by mean
  harmful/harmless separability across both thinking modes. Cross-validated
  held-out Cohen's d = 6.61 ± 2.14 (thinking ON) and 7.92 ± 2.31 (thinking
  OFF) on 60 JailbreakBench / Alpaca pairs. Held-out threshold accuracy
  0.983 and 0.975. Final Youden-J thresholds on the full fitting set:
  2.49 (ON) and 9.48 (OFF), both with fitting accuracy 1.0.
- **Attacks.** 3 families × 10 harmful targets, plus a no-attack arm and
  matched attack-wrapped harmless controls. 200 conditions × thinking
  ON/OFF = 400 rows. Families: roleplay, prefix injection, and
  pseudo-suffix (hand-written compliance bait, *not* GCG).
- **Sampling.** Temperature 0.6, top_p 0.95, top_k 20, seed 42. 1024
  tokens with thinking, 256 without. Trajectories from HuggingFace
  `output_hidden_states`, not a second decode-then-reencode pass.
- **Outcome labels.** StrongREJECT-style JSON judge on the request plus
  the final answer only (post-`</think>`). Compliance = not-refusal AND
  fulfills the request. The intended judge was `gpt-5-mini`; this org's
  free tier caps both that model and `gpt-4o-mini` at 50 requests/day, so
  the reported labels are from the same Qwen3-4B under a JSON rubric
  (`Qwen/Qwen3-4B-local-json`). Cohen's kappa against 50 labels scored
  from request and final answer only, blind to probe values and judge
  fields: 0.57 (refusal), 0.53 (fulfillment), 0.56 (compliance). Judge vs
  the old first-token keyword labeler: 0.92 agreement, kappa 0.84.

## High-level takeaways

I wrote these as predictions before the corrected run. The data disagreed
with the interesting one, and I am keeping that.

1. **Headline.** I predicted probe AUROC would fall from the unattacked
   arm to the attacked arms. That is the wrong estimand to lead with.
   *Harm detection* (harmful vs harmless, `prefill_proj`) stays at
   0.94–1.00 under every family, including no-attack, inside or above the
   published 0.84–0.95 band — on a smaller model, with attacks on.
   *Compliance prediction* on harmful rows (`-prefill_proj`) is 0.70–0.85
   with bootstrap CIs that all overlap 0.5 at the low end. The collapse I
   expected is not there. What is there is a split: the probe still sees
   that the prompt is harmful. It is only a middling predictor of whether
   the model will go through with it.
2. **Mechanism.** I predicted overpower (probe fires, model complies
   anyway) would dominate stealth (probe fooled). On completed harmful
   rows the split is 19 overpower vs 7 stealth (16 vs 7 if I drop the
   three clean-prompt overpowers). That matches the direction of the
   prediction, not the 84–100% pilot rate. Ten further rows are false
   alarms: harmless-looking projections on prompts the model refused.
3. **Deliberation.** Toggling thinking leaves both AUROCs roughly flat
   across families. Attack success is *higher* with thinking on (18/94 =
   19%) than off (8/100 = 8%). The directions themselves are nearly
   orthogonal: cosine 0.087 at the locked common layer. Cross-validation
   even prefers different layers (mostly 34 with thinking on, 22–23 with
   it off). I forced one layer so the comparison is fair. I did not get
   one shared refusal vector.
4. **Causality.** I predicted adding the direction would rescue stealth
   and not overpower. The opposite happened. Stealth rescue is 0/6, 0/7,
   0/7 at alpha 5/10/20. Overpower rescue is 6/16, 4/16, 5/16 (38%, 25%,
   31%). Cluster-bootstrap difference (stealth minus overpower) is
   −0.38 [−0.67, −0.21], −0.25 [−0.50, −0.11], −0.31 [−0.63, −0.16].
   Ablating the direction on clean harmful prompts creates 3/10 new
   compliances with thinking off and 0/10 with thinking on. The two-stage
   rescue prediction is false on this model. Steering only bites when the
   probe already fired.

Most interesting thing I found: harm detection does not care about these
attacks, and the thinking-on and thinking-off directions at layer 35 are
almost unrelated vectors.

Most surprising thing I found: the cosine. I expected two similar
refusal directions and a thinking switch that barely moved anything. I
got two different directions, a common-layer lock that CV does not
agree with, and more compliance when the model is allowed to "think."

## Key experiments

### E1. Is there a refusal direction in this model at all?

- **Why.** Table stakes. If harmful and harmless prompts do not
  separate, nothing downstream is interpretable.
- **Found.** They separate cleanly on held-out fitting folds: Cohen's d
  6.61 ± 2.14 (ON) and 7.92 ± 2.31 (OFF), threshold accuracy 0.983 and
  0.975, n = 60 pairs. Layer 35 is the common lock.
- **Graph.** Projection histograms are the fitting-set check; the public
  figure is the right-hand AUROC panel below.
- **Boring explanation.** It is a prompt-length or topic direction.
  Raw Pearson correlation of projection with token count is r = −0.28
  (n = 400). After controlling for family, thinking, target, and harmful
  flag, the length slope is +0.25 projection units per token. Length is
  not the separator. I am not claiming a new "safety is linear" result.
  This licenses E2–E5.

### E2. Does the probe still predict outcomes under attack?

- **Why.** The core question. I track two estimands because mixing them
  is how I would have lied to myself. Harm detection asks whether the
  last prefill state still distinguishes harmful from harmless prompts.
  Compliance prediction asks whether, among harmful prompts, a lower
  projection predicts that the model will actually comply.
- **Found.** Harm detection AUROC is 0.96 / 1.00 (no-attack, OFF/ON),
  0.94 / 1.00 (prefix injection), 0.96 / 0.97 (pseudo-suffix),
  0.95 / 1.00 (roleplay). Compliance AUROC on harmful rows is 0.78 / 0.71
  (no-attack), 0.79 / 0.72 (prefix injection), 0.85 / 0.70
  (pseudo-suffix), 0.70 / 0.73 (roleplay). n per harmful arm is 9–30
  after dropping truncated traces; several CIs run from ~0.15 to 1.0.
- **Graph.** `submission_artifacts/killer_graph_auroc.png`. Chance is
  0.50. The shaded band on the detection panel is the published
  plain-prompt range from a *different* model (Qwen3-8B and friends),
  not a number I measured here.
- **Baselines.** Chance = 0.50. Own no-attack compliance AUROC = 0.78
  (OFF) and 0.71 (ON). Prompt-length confound r = −0.28.
- **What would change my mind, and did.** If AUROC were flat across
  families and close to the unattacked number, H1 is dead. That is what
  happened for both estimands. I am reporting "the detector is
  surprisingly robust; the compliance probe is weak everywhere,
  including no-attack," not a family-wise collapse.

### E3. When it fails, how does it fail — stealth or overpower?

- **Why.** Separates the two stages of refusal.
- **Found.** Completed harmful rows: 158 correct refusals, 19 overpower,
  7 stealth, 10 false alarms, plus 6 truncated that I left unlabeled.
  Overpower is the common failure among actual jailbreaks. Stealth is
  concentrated in thinking-ON pseudo-suffix (5 of 7). False alarms are
  also concentrated there (8 of 10).
- **Threshold.** Youden-J from the fitting set, not the harmful/harmless
  midpoint. The midpoint in my invalid pilot sat at −26.2 and counted
  almost everything as "detected."
- **Caveat.** Attack wrappers shift harmless projections by +1.5 to
  +9.3 on average. The ON threshold is 2.49, so a wrapper can push a
  harmless prompt over the line by itself. That inflates "detected,"
  which mechanically inflates overpower. The harmless-wrapper control is
  not clean enough to ignore.

### E4. Is the direction causally doing the work — and differentially?

- **Why.** A probe is correlational. The two-stage story makes a
  differential prediction: adding the refusal direction should rescue
  stealth (detection was fooled) and not overpower (detection already
  fired).
- **Design.** Forward hooks on `model.model.layers[35]`, adding
  α ∈ {5, 10, 20} × the mode-specific unit direction at the last
  position, then regenerating the 23 attack failures. Separate ablation
  (`h -= (h · r̂) r̂`) on the 10 clean harmful battery prompts, both
  thinking modes.
- **Found.** Stealth: 0 rescued at every alpha. Overpower: 38% / 25% /
  31%. The gap is in the wrong direction and the CIs exclude zero.
  Ablation: thinking OFF goes from 1/10 to 4/10 compliance (3 new).
  Thinking ON goes from 2/10 to 1/9 completed, with 0 new compliances
  (one truncated). Graph: `submission_artifacts/steering_rescue_rates.png`.
- **What would change my mind, and did.** Equal rescue, or stealth >
  overpower, would have supported the two-stage framing. I got overpower
  only. I should describe this as a last-token add that sometimes
  recovers refusal when the probe already fired, not as a detection
  patch. Tiny n (7 stealth clusters) is a real limit; a point estimate
  of zero across three alphas is still the thing I owe the reader.

### E5. Does thinking help?

- **Why.** Within-model deliberation ablation, identical weights.
- **Found.** Attack success (judge compliance on completed harmful
  attack rows) is 16/85 = 19% with thinking ON vs 7/90 = 8% with it OFF.
  Prefill harm-detection AUROC is if anything *higher* with thinking on.
  Prefill compliance AUROC does not move in a consistent direction.
  During the `<think>` span, projection onto the refusal direction stays
  near zero across families and outcomes. Separation shows up in the
  final-answer span, not in the chain of thought
  (`submission_artifacts/trajectory_by_family_outcome.png`).
- **What would change my mind.** If projection moved substantially
  during CoT and the outcome tracked that movement rather than the
  prefill value, deliberation would be doing work. I do not see that
  movement.

## Controls and sanity checks

- **Attack-wrapped harmless prompts.** Mean projection shift vs the
  matched no-attack harmless prompt: prefix injection +6.3 / +5.2
  (OFF/ON), pseudo-suffix +3.7 / +1.5, roleplay +6.6 / +9.3. Necessary,
  and it did not come back near zero. Wrapper text moves the readout.
- **Prompt length vs projection.** r = −0.28; controlled length slope
  +0.25.
- **Judge vs 50 independent labels.** kappa = 0.57 / 0.53 / 0.56.
  Judge vs old keyword labeler: agreement 0.92, kappa 0.84.
- **Truncated generations.** 15 of 200 thinking rows never emitted
  `</think>`. They are unlabeled, not silently scored.
- **Direction stability across thinking modes.** cosine similarity =
  0.087.

## What went wrong (keeping this in on purpose)

My first run labeled outcomes by keyword-matching the first 50 generated
tokens. Qwen3 thinks by default, so those 50 tokens were almost entirely
the reasoning trace. I was labeling the model's private scratchpad
instead of its answer. 36 of 150 rows came back "ambiguous" and my first
instinct was to expand the keyword list. The `<think>` tags were sitting
in the output I printed; I had the evidence and did not follow it.

Before: keyword labels on the thinking trace, including a `complied=105`
count that does not describe answers. After: split on `</think>`, 15
truncated rows held out, judge on the final answer, kappa 0.57 against
50 blinded labels.

Lesson: if the model has a hidden scratchpad, look at whether you are
scoring the scratchpad. A longer keyword list would have made the bug
tidier, not smaller. I kept the pilot notebook cells as an audit trail
and threw out every pilot outcome number.

A second operational failure: I planned on `gpt-5-mini` as judge.
Unauthenticated probing plus a `max_tokens` incompatibility burned the
org's 50-request/day cap on `gpt-4o-mini` first, then the same cap on
`gpt-5-mini`. Four hundred API calls were not happening. I switched to a
local JSON judge on the same 4B weights. That is a same-model judge, and
every outcome number inherits it.

## Limitations

- One model, one size. No claim about generalization.
- Pseudo-suffixes are hand-written compliance bait, not GCG. I cannot
  test the original guess that *optimized* suffixes would be the stealthy
  family.
- One sample per row at temperature 0.6. No variance over sampling.
- Judge reliability is kappa 0.57 on 50 labels, and the judge is the
  same 4B model. All outcome-dependent numbers inherit that.
- Direction fit on extreme-harm JailbreakBench prompts. Probably does
  not transfer to borderline or dual-use content.
- n per family is small. I do not read AUROC differences smaller than
  the bootstrap CIs, and most of those CIs are wide.
- Forcing one common layer hides that CV wants different layers per
  thinking mode.
- Harmless wrappers shift the projection enough to matter.

## Dual use

A readout that says "this prompt still looks harmful internally" is
more useful to a defender than to an attacker. The attacker already
sees the output; the probe is telling you the model noticed and did it
anyway. A readout that told you *which strings hide the harm* would
flip that asymmetry. On this battery, stealth is the rare class, and I
am not releasing optimized attack strings. I am also not claiming this
as a deployed guardrail: the wrapper control moved the score, the
compliance AUROC is mediocre, and the judge is the same model.

## Where I would go next

The measurement is not really about refusal. It is: does the model
internally represent X while outputting not-X? Refusal is just the case
where I have reasonably clean ground truth for both halves. The cheapest
next version of that question is sandbagging on a capability the 4B
model actually has — GSM8K with an explicit "hide that you can do this"
instruction — and asking whether a "I know this" direction still fires
while the final answer is wrong. Same fitting/battery split, same two
estimands, no new attack families. If the cosine between thinking-on and
thinking-off refusal directions stays near zero on a second task, that
is the result I would actually chase.
