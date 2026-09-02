# Mech interp project details
_Exported on 9/2/2026 at 04:18:21 EST from Cursor (3.18.25)_

---

**User**

this si my mech interp project i am doing attached is more context  the @/home/sharleen/Downloads/Copy of Neel Nanda MATS 12.0 Stream -  Admissions Procedure + FAQ.pdf  the file called curr.ipynb is the current version all the text files are my consideratiosn, thoughts etc 

---

**Cursor**

I'll take a look at your project — the notebook, your notes, and the MATS admissions PDF — to get up to speed.

Your notes give a clear picture of the project direction. Now let me look at the notebook itself — it's large, so I'll extract its structure first — and skim the admissions PDF.

Both PDFs are the same roadmap document. Now let me read the actual notebook code to see how far you've gotten.

I've read through everything — your notes (`thoughts.txt`, `ledger.txt`, `plan.txt`), the "Tracer to Submission" roadmap (both PDFs are the same document, by the way — neither is actually the MATS admissions FAQ), and the full `curr.ipynb`. Here's my assessment of where the project stands.

## What you have working

The pipeline is genuinely end-to-end: refusal direction extracted via difference-in-means on 20 contrastive pairs, validated with 5-fold CV (held-out Cohen's d = 8.5 ± 1.9, layer 31 selected consistently across all folds — that's a solid, reportable result), then a 150-attempt battery across 3 attack families with stealth/overpower classification and plots. And your actual empirical story is more interesting than the one you predicted: **the probe almost never gets fooled — ~85–100% of successful jailbreaks are "overpower" (detection fires, model complies anyway), not "stealth."** That's a clean, surprising-ish finding: jailbreaks don't hide harm from the model, they break the link between detection and refusal. Finding 1 (gen_proj is an artifact) being reported honestly is also good — Neel explicitly rewards that kind of skepticism.

## Critical problems, in priority order

**1. Qwen3's thinking mode is silently corrupting your outcome labels.** Look at your own Cell 14 output: the "ambiguous" texts all start with `<think>\nokay, the user wants...`. Qwen3-4B defaults to thinking mode, so your 50 generated tokens are almost entirely the *reasoning trace*, not the answer. You're keyword-labeling the model's private deliberation ("okay, the user wants a phishing email... it needs to look legitimate...") — which pattern-matches compliance phrases even when the final answer after `</think>` is a refusal. Your `complied` counts are very likely inflated, which means the headline overpower/stealth numbers can't be trusted yet. Fix: pass `enable_thinking=False` in `apply_chat_template` (and re-extract the direction with the same setting), or generate enough tokens to get past `</think>` and label only what comes after.

**2. The expanded keyword labeler is too blunt to survive scrutiny.** `REFUSAL_PHRASES_EXPANDED` includes "do not", "don't", "dangerous", "harmful", "illegal", "sorry" — a fully compliant answer that says "be careful, this is dangerous. Step 1..." gets labeled *refused* (refusal takes precedence). Meanwhile the compliance list includes "sure", "method", "you can". With only 150 samples, the credible move is to hand-label all of them (an hour of work) or use an LLM judge, and report agreement with the keyword labeler. Neel's doc explicitly calls out rigor of measurement; this is the weakest link in the chain right now.

**3. The ledger contains placeholder results as if they were real.** "AUC=0.85 for role-play, AUC=0.52 for suffixes, steering changed success rate by 40%" — those come from the roadmap PDF's *hypothetical* executive summary. Your actual data says something different (probe threshold-accuracy ~28–35% everywhere, overpower dominating in all families, no steering experiment run yet). Make sure none of that template text leaks into the write-up; fabricated-looking numbers would be disqualifying. Also, the model is Qwen3-4B, not "Qwen 3.6."

**4. Missing pieces the roadmap (and Neel) require:**
- **Baselines** — you never compute AUC of prefill projection vs. success, nor compare against clean-prompt refusal rate or a black-box "is this harmful?" prompt baseline. This is explicitly required.
- **Causal test** — no steering experiment yet. Given your story is "detection fires but gets overridden," the natural causal test is: amplify the refusal direction at layer 31 during an overpower attack and see if refusal comes back. That would directly support the two-stage claim and is maybe a day of work with the infrastructure you already have.

**5. Smaller validity notes.** Your threshold (−26.2) is the midpoint between *extreme* harmful and harmless centroids, but it sits far below zero, so nearly everything classifies as "prefill high" — worth reporting the projection distributions rather than leaning on one threshold. The "adversarial suffix" family is hand-written compliance-bait, not GCG-optimized suffixes (you've flagged this, keep flagging it). And in Pass 3, decoding generated tokens back to a string and re-tokenizing can shift token boundaries — moot if you drop gen_proj, but don't reuse that pattern.

## Where the novelty risk sits

Your `thoughts.txt` already identifies it: "refusal is a single direction" is Neel's own scholars' work, and follow-ups have shown jailbreaks suppress that direction. Your differentiator is the **two-stage decomposition** (detection vs. execution) and the finding that detection is robust while execution is fragile — plus, if you run it, causal steering evidence per attack family. Frame the write-up around that, not around "we found a refusal direction."

My recommended order: fix thinking mode → relabel (hand-label all 150) → rerun the battery → compute per-family AUC + baselines → steering experiment → write-up. Want me to start on the thinking-mode fix and the labeling rework in the notebook?

---

**User**

using the above context of neel nandas guid i want you to know that my goal is to get accepted into the program to build a strong impactful project

---

**Cursor**

Before I write the plan, two things determine how ambitious it should be 

**Scoping the project plan**

> **How much working time do you realistically have left before submission?**
> - Under 10 hours (tighten what exists, write-up focused)
> - 10-20 hours (fix validity issues + add baselines + steering experiment)
> - 20+ hours (also add a second model or GCG suffixes) ✔

> **How do you want to label jailbreak outcomes (refused vs complied) after fixing the thinking-mode issue?**
> - Hand-label all ~150 outputs myself in a CSV (Recommended: most credible, ~1 hour)
> - LLM-as-judge via API (I have an OpenAI/Anthropic/OpenRouter key) ✔
> - Use Qwen3-4B itself as a local judge with a strict rubric (no API needed)

> **Where will the notebook run?**
> - Google Colab GPU (T4/A100), same as current runs ✔
> - Local GPU on this machine
> - Other (RunPod / Lambda / etc.)

and how to handle the labeling problem.

Good — that gives me a 

lot to work with. Let 

me verify a few technical details before I commit to a plan.

**Core framing decision**

> **Your thinking-mode bug may actually be the most novel thing here. Prior work (Arditi et al., 'Refusal is mediated by a single direction') studied non-reasoning chat models, where the refusal decision is read off at the end of the prompt. But Qwen3-4B deliberates in a <think> trace first — so the refusal decision may not live at prefill at all. Which framing do you want?**
> - Make thinking mode the independent variable (Recommended): run the battery with thinking ON vs OFF, and track the refusal-direction projection across every CoT token to find WHERE refusal gets decided. Turns your bug into the finding; highest novelty; uses the 20+ hours. ✔
> - Play it safe: disable thinking, redo cleanly, and nail the detection-vs-execution two-stage story with AUC, baselines, GCG suffixes and steering. Lower risk, more incremental.
> - Do the safe version first as the backbone, then add the reasoning-model angle as an extension if time allows.

Let me check the 

literature so the plan cites prior work correctly and I can flag any novelty overlap.

The literature check changes things — the naive version of the reasoning framing is already taken. Let me read the closest paper carefully to find the real gap.

---

**User**

does this follow neels nandas guide? what are potential real world applications , take on the persona of a genius with unlimited creativity and unconventional thinking

---

**Cursor**

Let me ground this against Neel's actual published guidance rather than working from the roadmap document alone.

Grounding first, because the search turned up something that directly contradicts the plan I just wrote.

## The hard constraint I got wrong

Neel's actual application task is **~12 hours, max 20** — not the 24 I budgeted, and the "20+" you selected is at the ceiling rather than the floor. His stated extras are for the *write-up*, not more experiments. Combined with the writing advice in [Mats-main/report.txt](Mats-main/report.txt) — *"Quality over Quantity. One interesting finding, well-explained and well-supported, is far better than ten superficial experiments"* — a 9-phase, 17-todo plan is itself a failure mode. I'd cut it to roughly this:

- Fix thinking mode and generation parsing (~2h)
- LLM judge + 50 hand-labels for kappa (~2h)
- Rerun battery, thinking ON vs OFF, existing 3 families + no-attack control (~3h, mostly unattended compute)
- Attack-wrapped-harmless control (~1h)
- Per-family AUROC with CIs, three baselines not five (~2h)
- The differential rescue steering test (~2h)
- Write-up (~3h)

Drop GCG, the layer-sweep heatmap, the second model, the capability-cost benchmark, and the 200-pair dataset scale-up (50-60 pairs from JailbreakBench is enough). Relabel the hand-written suffixes honestly as "pseudo-suffixes" instead of buying real ones with 4 hours. Neel explicitly says *"start by doing the obvious thing"* and that he's agnostic about fancy techniques.

## Where the project does match the guide

Strongly, on the things that matter most. His stream doc lists **"Understanding thinking models"** first among problems he's excited about — the thinking-mode framing is a bullseye, not a gamble. It also lists "Understanding sophisticated/safety relevant behaviour" and **"Investigate fundamental assumptions,"** which is precisely what stress-testing a published 0.84-0.95 AUROC under adversarial pressure is. You have a falsifiable hypothesis, real baselines, a negative-result-tolerant design, and a discovered bug that becomes a narrative beat ("Show Your Work" rewards exactly that).

Two risks worth naming bluntly. First: *"Docs that read like LLM slop will be rejected."* Your notebook is wall-to-wall `# FIRST PRINCIPLE:` comment blocks, and the roadmap PDF you've been working from is itself clearly model-generated. The write-up has to be in your voice, with your reasoning and your dead ends. Second: the executive summary format he specifies is 1-2 pages — problem and why it's interesting, high-level takeaways, then one paragraph and one graph per key experiment. Write that first, before the remaining experiments, and let it tell you which experiments are load-bearing.

## Real-world applications

Sorted by how much weight they can actually bear, because the temptation to overclaim here is the thing that would sink the write-up.

**Defensible if your control comes back clean.** The strongest application is nearly trivial and that's what makes it good: if the probe has high recall even under attack, you get a safety monitor for free. You're already running the forward pass, so the marginal cost of reading the refusal direction is one 2560-dimensional dot product — about 2,500 multiply-accumulates against the ~10^10 FLOPs per token of running Llama-Guard alongside your model. Six orders of magnitude cheaper, zero added latency, and it sees the internal state rather than the laundered output. The entire claim hinges on the attack-wrapped-harmless control: if long benign wrappers also push the projection up, your false-positive rate makes it undeployable. That single missing experiment is the difference between "interesting correlation" and "shippable guardrail," which is a nice thing to be able to say in a write-up.

Second, your pilot already implies a metric nobody publishes: **P(refuse | probe fires)**, currently something like 28-35%. Call it refusal integrity. It says the model recognized the attack and complied anyway two times in three. That's a standardized disclosure a lab would find uncomfortable and a regulator would find legible.

**Well-grounded extensions.** The stealth/overpower split inverts red-teaming economics in a non-obvious direction. The intuitive use is "predict which attacks work, skip the rest." The better use is the opposite: overpower attacks are the ones the probe *already* catches, so automate those and route the stealth cases — where the probe is blind — to human red-teamers. The probe becomes an acquisition function for where human attention is scarce and valuable.

It also redirects training budget. If detection is intact and execution is broken, then more RLHF on harmful examples is money spent hardening a stage that already works. The implied intervention is a consistency loss that penalizes the gap between what the probe reads and what the model does. That's a training recommendation derived from mechanism, which is the "understanding-based downstream task" your notes flagged from his list.

And it explains your own observation from [Mats-main/thoughts.txt](Mats-main/thoughts.txt) — that jailbroken models got *worse* at your actual work. *Where Do Reasoning Models Refuse?* measures 5-30% relative capability degradation from directional ablation. There's a jailbreak tax, because the refusal direction is entangled with general capability. Your Nonny-DINO output degrading into `[INSUFFICIENT DATA]` is that tax being paid. A Pareto frontier of compliance gained per unit of capability lost is a genuinely useful artifact, and it's the rare case where a personal itch and a research result are the same thing.

**Speculative, worth one sentence each and no more.** In law, liability turns on knowledge — and a probe reading gives you something like a mechanistic knowledge receipt. "The system had no way to know" stops being an assertion and becomes falsifiable, which changes deployer exposure under systemic-risk regimes and could underwrite insurance priced on detection-versus-compliance divergence. Relatedly, if your trajectory work shows the projection stays high through the CoT and only collapses at the answer boundary, then products marketing visible "reasoning" as a safety feature are selling a rationalization, and that's closer to an advertising-claims question than a technical one. Both are framings, not findings. Put them in a closing paragraph or leave them out.

**The one I'd actually build toward.** Detection-versus-behavior divergence isn't about refusal. Refusal is just the instance where you happen to have clean ground truth for both halves. The same measurement — does the model internally represent X while outputting not-X — is a general instrument for sandbagging, evaluation-awareness, sycophancy, and deception. Which closes the loop on the question you wrote down yourself before any of this: *"is the refusal direction the same mechanism that governs a model's resistance to secret elicitation?"* That's your future-work paragraph, it maps onto his interest in eliciting latent knowledge, and it shows you were pointed somewhere bigger than the experiment you had time to run.

I'm in Ask mode so I haven't touched the plan file — the phase list and the 24-hour budget in it are now wrong. Say the word and I'll cut it down to the 15-hour version above.

---

**User**

ok create a new plan dont discard the other one

---

**Cursor**

Creating a second plan file — the first one stays on disk as the stretch version.

---

**User**

what are the skills needed to be as good of a researcher engineer as neel nanda how would he do the first todo do it while taking on his persona

---

**Cursor**

Written to [Mats-main/exec_summary.md](Mats-main/exec_summary.md). First todo done.

## The skills

I'd put them in this order, because the ordering is itself the claim.

**Not fooling yourself.** This is the whole game and everything else is downstream. Neel's research-process writing keeps returning to it: the default state of an empirical researcher is believing something false because a number came out nice. The concrete habits are pre-registering what would change your mind, naming the boring explanation before you rule it out, and being more suspicious of results you like than results you don't. Your Cohen's d of 8.5 should have felt *too good* — an effect size of 8 means near-perfect separation, which usually means you measured something structural rather than semantic.

**Following confusion instead of routing around it.** You have a live example. In cell 16 you printed the ambiguous outputs to inspect them — genuinely good instinct, most people tune the classifier blind. But the printout said `<think>` at the top of all 36 rows, and instead of stopping to ask why, you expanded the keyword list. That is the single highest-leverage moment in the whole project, and the skill gap isn't knowledge (you'd have understood the issue instantly) — it's that a confusing observation triggered "make it go away" rather than "what is this telling me." The habit worth building: when you catch yourself patching a symptom, stop and write down what you're confused about.

**Iteration speed as a research capability, not an engineering nicety.** Being able to go from question to first data in minutes changes which questions you're willing to ask. Concretely: caching activations so you never recompute, keeping models loaded, small-n pilots before full runs, and enough PyTorch fluency that shape/dtype/device errors cost you seconds rather than derailing an afternoon. Your project already shows the cost of not having this — the three-pass structure in cell 14 does roughly triple the necessary forward passes, which is why you tested 10 targets instead of 60.

**Knowing your tools well enough to distrust them.** He wrote TransformerLens and is still the one telling people that SAEs are often the wrong choice and to do the obvious thing first. That's not modesty, it's knowing where tools lie. In your case: nnsight's `.trace()` semantics, whether `output[0]` carries a batch dim, and the fact that decoding tokens back to a string and re-tokenizing does not round-trip. That last one silently corrupted your Pass 3.

**Taste in problem selection.** The ability to look at a question and estimate, before starting, whether it's tractable in the time available and whether the answer would change anyone's mind. Hardest to teach, best trained by reading a lot of papers and forming opinions about which ones mattered.

**Distillation.** He treats write-up as a phase of research, not a chore after it. The illusion of transparency is the specific failure — what feels obvious to you is invisible to your reader.

The honest gap analysis for you: your bottleneck is not ideas. `thoughts.txt` shows genuinely good taste — you independently landed on the refusal/secret-elicitation connection and flagged your own novelty risk and dual-use problem unprompted, which most applicants don't. Your bottleneck is measurement discipline: trusting an aggregate before validating the thing it aggregates. That is a much easier fix than taste, and Phase 4's kappa check is exactly the habit.

## What I built into the skeleton, and why

Four things that are specifically his methodology rather than generic outlining:

**Predictions written before results.** Takeaways 1-4 are stated as predictions with blanks. Filling them in then becomes a test rather than a narrative you construct backwards. If a prediction is wrong, the template tells you to keep the original and say so.

**"What would change my mind" per experiment.** Each of E1-E5 has one. E2's is load-bearing: if AUROC stays flat, H1 dies and the story becomes "the probe is surprisingly robust" — which is still a real result, so you can't lose. That's how you make a project that can't fail.

**"Boring explanation" per experiment.** E1's is that you found a length or topic direction rather than a refusal direction. Naming it forces the control that rules it out, which is how the attack-wrapped-harmless prompts stopped being optional.

**E1 explicitly demoted to non-finding.** The skeleton says outright that "safety concept is linearly represented" is known and you're not claiming it. That's the rejected pattern your own notes flagged, and stating the demotion in the summary is what proves you know it.

Two more deliberate choices. The "What went wrong" section stays in the submission — his advice rewards showing your reasoning, and a candidate who finds and reports their own methodological bug reads as more trustworthy than one with clean numbers. And the limitations section says explicitly that the pseudo-suffix family cannot test your original stealth hypothesis, because the failure mode there is letting a family name quietly imply a stronger claim than the strings support.

Next up is Phase 1, the thinking-mode fix and generation parsing — say go and I'll start on the notebook.

