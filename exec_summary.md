# The cheap refusal probe still sees the harm. It does not reliably stop the answer.

Qwen3-4B, thinking on vs off, under three jailbreak families.

## 1. The core question and motivation

If I am already running a model, a safety check that is one extra guard
model is expensive. A check that is one dot product against a vector of
length `d_model` (2560 here), taken from the hidden state at the last
prompt token, is almost free. That is the pitch for an internal refusal
probe: you do not stand up Llama-Guard beside the model; you read a
direction you already paid to compute.

On plain harmful vs benign prompts that pitch looks real. “Do Thinking
Tokens Help with Safety?” (arXiv 2606.25013) reports 0.84–0.95 AUROC
for a first-thinking-token probe on Qwen3-8B and three other reasoning
models. The paper also says the evaluation is limited to those plain
prompts. So the question I actually ran is narrower and meaner: on
Qwen3-4B, does this cheap detector still work when someone applies a
jailbreak, or does adversarial pressure break the monitor? And does
turning the `<think>` scratchpad on or off change the answer?

I care about two failures that look the same from the outside. Either
the probe never notices the harm (stealth), or it notices and the model
answers anyway (overpower). Those are different bugs. Only the first one
means the cheap monitor is blind.

## 2. Experimental setup

I used Qwen3-4B (36 layers, `d_model` 2560) because it has a hard
`enable_thinking` switch, so thinking ON and OFF are the same weights.
The refusal direction is difference-in-means, harmful minus harmless, at
the last prefill token, locked at layer 35 for both modes. I fit on 60
JailbreakBench / Alpaca pairs and held out 10 battery targets. The
battery is three attack families — roleplay, prefix injection, and
hand-written pseudo-suffixes, not GCG — plus a no-attack arm and
attack-wrapped harmless controls. That is 200 conditions × thinking
ON/OFF = 400 rows. Sampling is temperature 0.6, top_p 0.95, top_k 20,
seed 42.

I score only the text after `</think>`. Compliance means the model did
not refuse *and* it actually supplied the request. The judge is a
StrongREJECT-style JSON rubric. I wanted `gpt-5-mini`; after a 50
request/day cap I used the same 4B model as a local JSON judge. I then
labeled 50 completed answers myself from the request and final answer
only. Cohen’s kappa versus the judge is 0.57 (refusal), 0.53
(fulfillment), 0.56 (compliance). That is moderate, not a rubber stamp.

Graphs: `submission_artifacts/killer_graph_auroc.png`,
`trajectory_by_family_outcome.png`, `steering_rescue_rates.png`.

## 3. Key findings and surprises

**Detection stayed up. Compliance prediction did not.** Harm-detection
AUROC (harmful vs harmless, using the prefill projection) is 0.94–1.00
under every family, including attacks, on this smaller model. That sits
in or above the published 0.84–0.95 band from *other* models on plain
prompts. Compliance AUROC on harmful rows is 0.70–0.85, with bootstrap
intervals wide enough that several of them reach down near chance. I
predicted a collapse under attack. I did not get one. I got a split: the
probe still sees that the prompt is harmful; it is only a middling
predictor of whether the model will go through with it.

**Overpower, not stealth.** On completed harmful rows: 19 overpower
(alarm rang, model answered anyway) and 7 stealth (alarm stayed quiet),
plus 158 correct refusals and 10 false alarms. Attackers on this battery
rarely hide the harm from the readout. They break the link from “this
looks bad” to “I will not do it.”

**Stealth rescue was zero.** I added the refusal direction at layer 35
with α = 5, 10, 20. That restored refusal in 25–38% of overpower cases
(6/16, 4/16, 5/16) and in **0** stealth cases (0/6, 0/7, 0/7). I had
predicted the opposite: if stealth is “the probe was fooled,” pushing
the direction should help stealth and not overpower. The CIs on the
stealth-minus-overpower gap sit entirely below zero. So steering here is
not a detection patch. It sometimes recovers refusal when the alarm
already rang.

**The scratchpad made jailbreaks easier.** Attack success on completed
harmful attack rows is 19% with thinking ON (18/94) and 8% with it OFF
(8/100). During the `<think>` span the projection sits near zero; the
separation shows up in the final-answer span, not in the notes.

**The two thinking modes do not share a refusal vector.** Cosine
similarity between the thinking-ON and thinking-OFF directions at the
locked layer is 0.087. They are almost orthogonal. Cross-validation even
wants different layers (mostly 34 with thinking on, 22–23 with it off).
I forced layer 35 so the comparison was fair. I did not get one shared
direction.

## 4. Engineering failures and lessons

The first pilot keyword-labeled the first 50 generated tokens. Qwen3
thinks by default, so those tokens were almost all inside `<think>`. I
was grading the scratchpad. 36 of 150 rows came back “ambiguous,” and I
almost “fixed” it by growing the keyword list. The tags were in the
printout. After the fix I split on `</think>`, left 15 truncated
thinking rows unlabeled, and judged the answer.

The second failure was the API. `gpt-5-mini` and `gpt-4o-mini` both sat
behind a 50 request/day cap on this org. Four hundred judge calls were
not going to happen. I pivoted to a local JSON judge on Qwen3-4B. That
is the same family as the target model. Kappa 0.57 against my 50 labels
is the size of that blind spot. Every compliance percentage in section 3
inherits it.

## 5. Limitations and honest assessment

This is one 4B model. I am not claiming Qwen3-8B, and I am not claiming
a shipped guardrail. The “suffix” family is hand-written compliance
bait, not GCG, so I cannot test the guess that *optimized* suffixes
would be the stealthy ones. Human verification is n = 50 with kappa
0.57. One sample per row at temperature 0.6. Attack wrappers shift
harmless projections by about +1.5 to +9.3, which can push a prompt over
the thinking-ON threshold by itself and inflate overpower. Per-family
samples are small; I do not read AUROC gaps smaller than those
intervals.

A monitor that says “this still looks harmful inside” is more useful to
a defender than to an attacker, because the attacker already sees the
output. On this battery stealth is the rare class, and I am not
releasing optimized attack strings. I would not put this probe in
production on the strength of these numbers.
