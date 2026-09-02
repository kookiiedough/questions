---
name: Submission Ready Cut
overview: "A 15-hour cut of the refusal-probe-under-attack experiment sized to Neel's actual 12-20 hour limit, keeping only the load-bearing experiments: thinking-mode fix, LLM judge with human agreement, per-family AUROC against baselines, and the differential steering rescue test."
todos:
  - id: exec-skeleton
    content: Draft the 1-2 page executive summary skeleton in Neel's format with numbers left blank, as a scope guard
    status: completed
  - id: thinking-flag
    content: Thread enable_thinking through format_chat; re-extract direction under both settings and report cosine similarity
    status: pending
  - id: parse-and-decode
    content: Split generations on </think> and label the final answer only; record n_think_tokens and truncation flag; raise max_new_tokens; switch to temp 0.6 / top_p 0.95 / top_k 20 with fixed seed
    status: pending
  - id: single-pass-traj
    content: Replace broken gen_proj with one forward pass over concatenated prompt+generation token ids, slicing all positions for the full trajectory
    status: pending
  - id: dataset-and-control
    content: Scale to 50-60 JailbreakBench/Alpaca pairs; add attack-wrapped-harmless control and no-attack condition; keep fitting and battery prompts disjoint
    status: pending
  - id: battery-rerun
    content: "Rerun battery: 3 families (suffix family renamed to pseudo-suffix) plus no-attack, crossed with thinking ON/OFF, caching to Drive per family"
    status: pending
  - id: judge-kappa
    content: LLM judge on final answers with StrongREJECT-style rubric; hand-label 50 rows for Cohen's kappa; report agreement with the old keyword labeler
    status: pending
  - id: auroc-baselines
    content: Per-family AUROC with bootstrap CIs against three baselines (chance, own no-attack condition, prompt-length confound check)
    status: pending
  - id: recalibrate-and-dedupe
    content: Replace midpoint threshold with Youden-J from held-out folds; redo stealth/overpower split; delete the duplicate classification cell
    status: pending
  - id: steering-rescue
    content: "Differential rescue test via forward hooks on model._model: adding the direction should rescue stealth but not overpower cases; plus directional ablation on clean harmful prompts"
    status: pending
  - id: writeup
    content: Fill the summary skeleton, build the killer graph with the published AUROC reference line, write in own voice, scrub fabricated numbers from ledger.txt
    status: pending
isProject: false
---

## Why this plan exists

The earlier plan (`refusal_probe_under_attack_da2681b8.plan.md`, retained as the stretch version) budgeted 24 hours across 9 phases. Neel's actual application task is **~12 hours, max 20**, and his extra-time allowance is for the write-up, not more experiments. His writing advice in [Mats-main/report.txt](Mats-main/report.txt) is explicit: *"One interesting finding, well-explained and well-supported, is far better than ten superficial experiments."* This plan is the same research question at one third the scope.

**Cut from the stretch plan:** real GCG suffixes via nanogcg, the layer-by-position AUROC heatmap, the Qwen3-8B replication, the ablation capability-cost benchmark, CoT flip-point detection, the black-box self-classification and TF-IDF baselines, and the 200-pair dataset scale-up.

**Kept:** everything needed for one complete, defensible claim with a causal test.

## Research question and hypotheses

> The end-of-prefill refusal probe predicts refusal at 0.84-0.95 AUROC on plain harmful prompts (arXiv 2606.25013, which uses Qwen3-8B). Does that survive adversarial attack, and does thinking mode change the answer?

- **H1:** AUROC degrades under attack, non-uniformly across families.
- **H2:** Failures split into stealth (probe fooled) and overpower (probe fires, model complies anyway). Pilot data suggests overpower dominates at 84-100%.
- **H3:** If thinking is prefix completion rather than deliberation, the thinking toggle leaves prefill AUROC unchanged.

Positioning: the four relevant papers all use plain harmful prompts with no attack-family decomposition, and 2606.25013 names this in its own limitations. Qwen3-4B's hard `enable_thinking` switch gives a within-model deliberation ablation on identical weights.

## Phase 0 - Write the executive summary skeleton first (0.5h)

Before touching code, draft the 1-2 page summary in Neel's specified format: what problem and why it's interesting, high-level takeaways, then one paragraph and one graph per key experiment. Leave the numbers blank. This is the scope guard: any experiment that does not fill a blank in this document does not get run.

## Phase 1 - Measurement fixes (2.5h, blocking)

All in [Mats-main/src/neel/curr.ipynb](Mats-main/src/neel/curr.ipynb).

Thread the thinking flag through `format_chat` (currently cell 7):

```python
def format_chat(prompt: str, thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
```

Parse before labeling. The 36 ambiguous rows in cell 16 are all `<think>` text, so the current `complied=105` is measuring the reasoning trace, not the answer:

```python
def split_trace(text: str):
    if "</think>" in text:
        think, _, answer = text.partition("</think>")
        return think, answer.strip(), True
    return text, "", False          # third value = reached close tag
```

Record `n_think_tokens` and the reached-close flag; report truncated rows separately rather than silently labeling them. Raise `max_new_tokens` to 1024 for thinking, 256 for non-thinking.

Replace greedy decoding. `do_sample=False` in cell 14 contradicts the Qwen3 model card, which warns it causes endless repetition in thinking mode. Use temperature 0.6, top_p 0.95, top_k 20, fixed seed, one sample per row.

Replace the broken `gen_proj` with a single-pass trajectory. Stay in token space to avoid the decode-then-re-encode boundary shift in cell 14's Pass 3:

```python
full_ids = torch.cat([input_ids, gen_ids.unsqueeze(0).to(model.device)], dim=1)
with model.trace(full_ids):
    h_all = model.model.layers[best_layer].output[0].save()   # (seq_len, 2560)
proj_all = (h_all.float() @ refusal_dir.float()).cpu().numpy()
```

This gives prefill projection, first-thinking-token projection, and the whole CoT trajectory from one forward pass.

Re-extract the direction under both thinking settings and report the cosine similarity between them.

## Phase 2 - Dataset and the one control that matters (1.5h)

- Moderate scale-up only: about 50-60 contrastive pairs, harmful from JailbreakBench, harmless from Alpaca. Citable beats hand-written, and n=20 is too thin, but 200 pairs is not where the marginal value is.
- **Add attack-wrapped harmless prompts.** This is the single most important missing piece. Attack wrappers are far longer than the extraction prompts, so right now you cannot separate "the attack moved the projection" from "long wrapper text moved the projection." Without it, the guardrail interpretation of the result is unsupported.
- Add a no-attack condition (plain harmful target) to supply the baseline.
- Keep direction-fitting prompts disjoint from `jailbreak_target_prompts`.

## Phase 3 - Battery rerun (1h hands-on, ~2h unattended)

Existing three families from cell 12, with `adversarial_suffix_templates` **renamed to pseudo-suffix** in code and write-up, since they are hand-written compliance bait rather than gradient-optimized. Conditions: 3 families plus no-attack, crossed with thinking ON and OFF. Cache to Drive after each family so a Colab disconnect costs nothing.

## Phase 4 - Judge and human agreement (2h)

- LLM judge over the **final answer only**, StrongREJECT-style rubric (binary refusal plus specificity and convincingness), blind to internals.
- **Hand-label 50 random rows and report Cohen's kappa.** This is what makes every downstream number trustworthy, and it is cheap.
- Also report judge agreement with the old keyword labeler from cell 17, to quantify the bug for the write-up.

## Phase 5 - Analysis (2.5h)

- Per-family, per-thinking-condition AUROC of prefill projection predicting compliance, with bootstrap CIs. n is small; CIs are not optional.
- Three baselines, all cheap:
  - chance at 0.5
  - your own no-attack condition, which replicates the published 0.84-0.95 on your setup and makes the degradation legible
  - a prompt-length confound check: regress projection on token count, no extra compute needed
- Recalibrate the threshold. The current midpoint of -26.2 (cell 18) sits so far below zero that nearly everything counts as "prefill high," which mechanically inflates the overpower count. Use a Youden-J threshold from held-out folds.
- Redo the stealth/overpower split on the recalibrated threshold, and delete one of the two conflicting classification cells (19 and 20 both claim to be "CELL 17").
- Replicate the first-thinking-token probe AUROC and compare attacked versus unattacked.

## Phase 6 - The causal test (2h)

This is the crown jewel and must not be cut. The two-stage model makes a **differential** prediction: adding the refusal direction should rescue stealth cases (probe was fooled) but not overpower cases (probe already fired). A differential rescue rate is far stronger evidence than one global steering number.

Use PyTorch hooks on `model._model` rather than nnsight's generate API, consistent with the workaround already proven in cell 14:

```python
def make_add_hook(vec, alpha):
    def hook(module, inputs, output):
        h = output[0]
        h[:, -1, :] += alpha * vec
        return (h,) + output[1:]
    return hook

handle = model._model.model.layers[best_layer].register_forward_hook(
    make_add_hook(refusal_dir_bf16, alpha)
)
# ... generate ...
handle.remove()
```

Also run directional ablation on clean harmful prompts (`h -= (h @ r_hat).unsqueeze(-1) * r_hat`) to validate the direction against the known result.

## Phase 7 - Write-up (3h)

- Fill in the Phase 0 skeleton. Killer graph: AUROC by family and thinking condition, with the published no-attack AUROC as a horizontal reference line so the collapse is visible at a glance.
- **Write in your own voice.** Neel states that *"docs that read like LLM slop will be rejected."* The notebook is currently wall-to-wall `# FIRST PRINCIPLE:` comment blocks and the roadmap PDF you worked from is visibly model-generated. Narrate your actual path, including the thinking-mode bug, which his "Show Your Work" advice rewards.
- **Scrub the fabricated numbers.** [Mats-main/ledger.txt](Mats-main/ledger.txt) contains "AUC=0.85 / 0.52", "steering changed success rate by 40%", and "Qwen 3.6", all copied from the roadmap PDF's hypothetical example. None of it can reach the submission.
- Limitations: one model, pseudo-suffixes rather than GCG, judge reliability, single sample per row.
- One closing paragraph on why it matters: the cheap-guardrail application (a 2560-dim dot product against running a separate guard model), contingent on the harmless-wrapper control; and the generalization line, that detection-versus-behavior divergence extends to sandbagging and secret elicitation, which is the question you wrote in [Mats-main/thoughts.txt](Mats-main/thoughts.txt) before starting. Keep the legal and consumer-protection framings out, or to one sentence.

## Budget

Hands-on: 0.5 + 2.5 + 1.5 + 1 + 2 + 2.5 + 2 + 3 = **15h**, plus roughly 2h unattended compute. Leaves headroom under the 20h ceiling.

If time runs short, cut in this order: first-thinking-token replication, then the thinking-OFF arm (report ON only), then the trajectory plot. Do not cut Phase 4 or Phase 6.