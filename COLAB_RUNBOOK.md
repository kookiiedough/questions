# Submission rerun in Google Colab

Use this only for the submission-ready section at the end of
`src/neel/curr.ipynb`. The earlier cells are the invalid pilot and are retained
only as an audit trail.

## 1. Start the runtime

1. Create a Colab runtime with a T4 GPU.
2. Clone this repository and check out the implementation branch (or
   `fable-rev` after the pull request is merged):

   ```bash
   !git clone https://github.com/kookiiedough/questions.git
   %cd questions
   !git checkout cursor/execute-submission-plan-c3b0
   ```

3. Open `src/neel/curr.ipynb` from that clone.
4. Add `OPENAI_API_KEY` to the Colab secrets/environment. Set `JUDGE_MODEL` to
   the exact judge model snapshot you intend to report.

The setup cell selects FP16 on a T4 and BF16 only on GPUs that support it.

## 2. Run the corrected experiment

Run cells 26–30 in order. They:

- mount Drive;
- install the pinned project environment;
- load Qwen3-4B;
- load 60 disjoint JailbreakBench/Alpaca fitting pairs and separate battery
  controls;
- extract both thinking-mode directions at one common layer;
- run and cache every family × thinking arm;
- judge only completed final answers, with per-answer resume caching.

Results are written to
`/content/drive/MyDrive/refusal_probe_submission`. Do not delete this directory
when reconnecting. Cache fingerprints prevent stale results from being reused
after prompts, dependencies, model revisions, settings, layers, or directions
change.

## 3. Complete the independent labels

The first run of cell 31 creates `human_labels_50.csv` and intentionally stops.
Without viewing judge fields or probe values, fill both columns for all rows:

- `human_refusal`: `true` or `false`
- `human_fulfills_request`: `true` or `false`

Do not edit the request, answer, fingerprint, or SHA-256 columns. Upload the
completed file to the same Drive path and rerun cell 31. It rejects stale text,
wrong-run fingerprints, missing rows, and invalid labels.

## 4. Finish analysis and interventions

Run cells 32–35. Cell 33 performs the compute-intensive matched steering and
ablation runs. The final cell must print a JSON report with
`"status": "complete"`.

Before running the final cell, replace every `[?]` in `exec_summary.md` using
only corrected-run artifacts. Rewrite the skeleton in your own voice. Preserve
limitations: one model, pseudo-suffixes rather than GCG, one sample per row,
judge reliability, and small per-family samples.

## 5. Return evidence

Keep the complete Drive results directory, the filled `exec_summary.md`, and the
cell outputs. Send those artifacts back to this task so the repository-level
completion audit can be rerun before the goal is marked complete.

## RunPod / local GPU

The notebook still needs nnsight in Colab. On a CUDA 12.8 driver (including
RunPod RTX Ada cards), do not install Torch 2.14 CUDA 13 wheels; they will
report `cuda False`. Use a CUDA 12.8 Torch build, then:

```bash
source /root/.openai_env   # OPENAI_API_KEY, not committed
uv run --no-sync python scripts/run_submission_experiment.py \
  --cache-dir results \
  --stages extract,battery,judge,human-sheet
```

Family caches resume mid-run. Stop after `human-sheet`, fill
`results/human_labels_50.csv`, then run `agreement,analysis,steering,figures,audit`.
