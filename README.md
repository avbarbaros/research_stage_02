# research_stage_02 — Reasoning Methods in Language Models

**Stage 2 of the Multimodal Reasoning Thesis Roadmap**
Schedule: 12 Oct – 25 Oct 2026 | Hardware: single L4 GPU (Google Colab)

This repository contains implementations for Stage 2 experiments: mastering the reasoning
toolkit in its original text-only setting before porting ideas to the multimodal setting in Stage 3.

---

## Repository structure

```
research_stage_02/
├── config.yaml                  # All settings for E2.1–E2.4 (prompts, generation, models)
├── few_shot_exemplars.jsonl     # 8 hand-written GSM8K-style CoT exemplars (edit freely)
├── few_shot_exemplars_wei.jsonl # the standard 8 GSM8K CoT exemplars from Wei et al. (2022)
│
├── reasoning_utils.py           # shared: GSM8K subset, answer extraction, statistics
├── e2_1_cot_from_scratch.py     # E2.1 — direct / zero-shot CoT / two few-shot CoT variants
├── e2_2_self_consistency.py     # E2.2 — self-consistency (majority voting over sampled CoT)
├── plot_e2_2.py                 # E2.2 — scaling-curve figure + table from the results JSON
├── setup.py                     # installs packages, checks GPU, validates config.yaml
│                                #        (scripts for E2.3–E2.4 to be added)
│
└── outputs/                     # Created at runtime (also copied to drive_output_dir)
    ├── e2_1_results_<timestamp>.json
    ├── e2_2_checkpoint.jsonl              # in drive_output_dir when set
    ├── e2_2_results_<timestamp>.json
    └── e2_2_results_<timestamp>_scaling.png / .pdf / _table.md
```

---

## Running the experiments (start here)

Everything runs on Google Colab with an **L4 GPU** runtime. Run the steps in this order,
from the repo root. The commands are shown as terminal commands; in a notebook cell, prefix
them with `!` and use `%cd` instead of `cd`.

Both E2.1 and E2.2 use `n_eval: 300` with the same seed, so they evaluate **the same 300
GSM8K test questions**. That is what makes the paired comparisons between them valid.

### Step 1 — Setup (every new Colab session)

```bash
python setup.py
```

This installs packages, checks the GPU, and validates `config.yaml` and both exemplar files.

### Step 2 — Smoke tests (a few minutes)

```bash
python e2_1_cot_from_scratch.py --n_eval 5
python e2_2_self_consistency.py --n_eval 2
```

Check that both finish, that accuracies print, and that result files appear in `outputs/`.
The E2.2 smoke test writes the first 2 questions to the checkpoint. The full run in Step 4
reuses them, so this is not wasted work.

### Step 3 — E2.1 (greedy baselines)

E2.1 has **no checkpoint**: if Colab disconnects mid-run, that run is lost. Running the
conditions one at a time keeps each run short, and each run writes its own results file.
Run `zero_shot_cot` first, because E2.2 compares against it.

```bash
python e2_1_cot_from_scratch.py --condition zero_shot_cot      # ~70 min  ← needed by E2.2
python e2_1_cot_from_scratch.py --condition few_shot_cot       # ~40 min
python e2_1_cot_from_scratch.py --condition few_shot_cot_wei   # ~40 min
python e2_1_cot_from_scratch.py --condition direct             # ~1 min
```

The times are for 300 questions, estimated from the 20-question run (about 14 s per question
for zero-shot CoT, 8 s for few-shot). You can also run everything at once (~2.5 h) with
`python e2_1_cot_from_scratch.py`. The McNemar tests between conditions are only computed
for conditions in the same run.

### Step 4 — E2.2 (self-consistency)

Point `--greedy_ref` at the results file from the `zero_shot_cot` run in Step 3:

```bash
python e2_2_self_consistency.py --greedy_ref outputs/e2_1_results_YYYYMMDD_HHMM.json
```

This takes an estimated 2–3 h for 300 questions × 32 samples; the log prints an ETA. **If the
session disconnects, run Step 1 again and then the same command.** It continues from the
checkpoint (`e2_2_checkpoint.jsonl`, in `drive_output_dir` when set, otherwise in
`outputs/`). If you run out of GPU memory, set `generation.samples_per_batch: 16` in
`config.yaml` (this does not invalidate the checkpoint).

If you ran E2.2 without `--greedy_ref`, add the comparison afterwards without a GPU:

```bash
python e2_2_self_consistency.py \
    --analyze_only <checkpoint dir>/e2_2_checkpoint.jsonl \
    --greedy_ref   outputs/e2_1_results_YYYYMMDD_HHMM.json
```

### Step 5 — Plot E2.2

```bash
python plot_e2_2.py outputs/e2_2_results_YYYYMMDD_HHMM.json
```

This writes the figure (`_scaling.png` / `.pdf`) and table (`_table.md`) next to the results
file and prints the "clean scaling curve" checks. Plotting needs no GPU.

> Results are written to `outputs/` and copied to `drive_output_dir` when it is set. In a
> new Colab session the local `outputs/` folder is empty, so use the copies in
> `drive_output_dir` for the paths in Steps 4–5.

### Order at a glance

| # | Command | GPU | Time (est.) | Resumable | Needs |
|---|---------|-----|-------------|-----------|-------|
| 1 | `python setup.py` | yes | 5–10 min | – | – |
| 2 | smoke tests | yes | 5 min | – | 1 |
| 3 | `e2_1 … --condition zero_shot_cot` | yes | ~70 min | no | 1 |
| 3 | `e2_1 …` other three conditions | yes | ~80 min total | no | 1 |
| 4 | `e2_2_self_consistency.py --greedy_ref …` | yes | ~2–3 h | **yes** | 3 (zero-shot file) |
| 5 | `plot_e2_2.py …` | no | seconds | – | 4 |

---

## E2.1 — CoT from Scratch

**Goal:** implement zero-shot CoT, few-shot CoT, and direct prompting on GSM8K with a 7B
model using nothing beyond the inference engine. Reproduce the qualitative ordering of the
three conditions and document every answer-extraction pitfall.

**Papers:** Wei et al. (2022) arXiv:2201.11903 · Kojima et al. (2022) arXiv:2205.11916

### Conditions

| Condition | Prompt |
|-----------|--------|
| `direct` | Answer only, no reasoning |
| `zero_shot_cot` | "Let's think step by step." (Kojima et al.) |
| `few_shot_cot` | 8 hand-written exemplars (`few_shot_exemplars.jsonl`) |
| `few_shot_cot_wei` | The standard 8 GSM8K exemplars from Wei et al. (`few_shot_exemplars_wei.jsonl`) |

The two few-shot conditions share the same system prompt, so any difference between them
comes from the exemplars alone.

### Statistics

- Each condition's accuracy is reported with a **Wilson 95% confidence interval**.
- Every other condition is compared with `zero_shot_cot` using an **exact McNemar test** on
  paired per-question outcomes (all conditions see the same questions).
- With small `n_eval` the intervals are wide. For example, 17/20 gives [64.0, 94.8] %.
  Use `n_eval >= 300`, or `null` for all 1319, before drawing conclusions.

### Run

```bash
# All four conditions, 300 examples (default)
python e2_1_cot_from_scratch.py

# Single condition
python e2_1_cot_from_scratch.py --condition direct
python e2_1_cot_from_scratch.py --condition zero_shot_cot
python e2_1_cot_from_scratch.py --condition few_shot_cot
python e2_1_cot_from_scratch.py --condition few_shot_cot_wei

# Quick smoke test
python e2_1_cot_from_scratch.py --n_eval 20

# Override config without editing the file
python e2_1_cot_from_scratch.py --n_eval 200 --condition few_shot_cot
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config.yaml` | Path to config file |
| `--condition` | `all` | `all` / `direct` / `zero_shot_cot` / `few_shot_cot` / `few_shot_cot_wei` |
| `--n_eval` | value from config | Number of GSM8K test examples to use |

### Configuration

All knobs live in `config.yaml` under the `e2_1` section — no code edits needed:

| Key | Default | What it controls |
|-----|---------|-----------------|
| `e2_1.n_eval` | 300 | Examples evaluated per condition |
| `e2_1.prompts.few_shot_cot.exemplars_file` | `few_shot_exemplars.jsonl` | Exemplars for `few_shot_cot` |
| `e2_1.prompts.few_shot_cot_wei.exemplars_file` | `few_shot_exemplars_wei.jsonl` | Exemplars for `few_shot_cot_wei` |
| `e2_1.prompts.direct.system` | see file | System prompt for direct condition |
| `e2_1.prompts.zero_shot_cot.system` | see file | System prompt for zero-shot CoT |
| `e2_1.prompts.zero_shot_cot.user_suffix` | `"\n\nLet's think step by step."` | Appended to each question |
| `e2_1.prompts.few_shot_cot.system` | see file | System prompt for few-shot CoT |
| `e2_1.generation.direct.max_new_tokens` | 64 | Token budget for direct condition |
| `e2_1.generation.cot.max_new_tokens` | 512 | Token budget for CoT conditions |
| `e2_1.generation.*.repetition_penalty` | 1.0 | Overrides Qwen's hidden default of 1.05, so decoding is plain greedy |
| `models.main` | `Qwen/Qwen2.5-7B-Instruct` | Inference model |

### Few-shot exemplars

Edit `few_shot_exemplars.jsonl` to change or extend the hand-written exemplars. Keep
`few_shot_exemplars_wei.jsonl` unchanged, since it is the published baseline. One JSON object per line:

```jsonl
{"question": "...", "answer": "Step 1 ...\nStep 2 ...\nThe answer is 42."}
```

The script loads this file at startup; no code changes required.

### Output format

Results are written to `outputs/e2_1_results_<timestamp>.json` (and copied to
`drive_output_dir`). Each record contains the full prompt and response with no truncation:

```json
{
  "experiment": "E2.1",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "n_eval": 300,
  "summary": {
    "direct":           {"accuracy": 72.40, "ci95_low": 68.3, "ci95_high": 76.1, "extraction_failures": 3, ...},
    "zero_shot_cot":    {"accuracy": 84.20, "ci95_low": 80.7, "ci95_high": 87.2, ...},
    "few_shot_cot":     {"accuracy": 86.60, ...},
    "few_shot_cot_wei": {"accuracy": 85.80, ...}
  },
  "comparisons": {
    "few_shot_cot_vs_zero_shot_cot": {
      "a_only_correct": 14, "b_only_correct": 26, "diff_pp": 2.4, "mcnemar_p": 0.0807
    },
    ...
  },
  "records": {
    "direct": [
      {
        "index":       0,
        "question":    "...",
        "gt_raw":      "... #### 42",
        "gt":          "42",
        "messages":    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
        "prompt_text": "<|im_start|>system\n...",
        "response":    "42",
        "pred":        "42",
        "correct":     true
      }
    ]
  }
}
```

### Answer extraction pitfalls (documented)

| # | Pitfall | Handled by |
|---|---------|-----------|
| 1 | Numbers with commas (`1,234`) | `.replace(",", "")` before `float()` |
| 2 | Numbers with units (`42 apples`) | regex does not require end-of-string |
| 3 | Multiple number candidates | trigger-phrase match wins; last number is fallback |
| 4 | Negative / decimal numbers | pattern includes `-` and `.` |
| 5 | No number in output | returns `None`; counted as wrong, no crash |

---

## E2.2 — Self-consistency

**Goal:** sample several chain-of-thought answers per question, take the majority answer, and
measure how accuracy grows with the number of samples k.

**Paper:** Wang et al. (2022) arXiv:2203.11171

### How it works

- The prompt is the same as E2.1 `zero_shot_cot`, so the only change is sampling + voting.
- For each question, `max(k_values)` = 32 samples are drawn once (T=0.7, top_p=0.95).
- For each smaller k, the reported accuracy is the **expected** majority-vote accuracy over
  `n_resamples` random k-subsets of those 32 samples. This gives a smoother curve than
  using only the first k samples. The 95% CI is a bootstrap over questions.
- **Voting:** answers are compared numerically ("18.0" = "18"). Samples with no number do
  not vote, and ties go to the answer seen first.
- **Any-correct rate:** the share of questions where at least one of the k samples is right.
  This is the upper bound for any answer-selection method, including the verifiers in E2.4.
- **Agreement bins:** accuracy split by how many of the 32 samples back the winning answer.
- E2.1 and E2.2 use the same seed and `n_eval: 300`, so they evaluate the same 300 questions.
  Pass an E2.1 results file as `--greedy_ref` to get an exact McNemar test of greedy CoT
  vs. self-consistency at k=32.

**Sampling settings:** Qwen2.5's `generation_config.json` adds `top_k=20` and
`repetition_penalty=1.05` unless they are overridden. E2.2 sets `top_k: 0` and
`repetition_penalty: 1.0` in the config, so only temperature and top_p apply.

### Run

```bash
# Full run (300 questions × 32 samples). Resumes automatically after a disconnect.
python e2_2_self_consistency.py

# Smoke test
python e2_2_self_consistency.py --n_eval 5

# Compare with E2.1 greedy zero-shot CoT
python e2_2_self_consistency.py --greedy_ref outputs/e2_1_results_YYYYMMDD_HHMM.json

# Start over, discarding the checkpoint
python e2_2_self_consistency.py --fresh

# Re-analyse a checkpoint without loading the model (CPU is fine)
python e2_2_self_consistency.py --analyze_only <checkpoint dir>/e2_2_checkpoint.jsonl
```

### Plot the scaling curve

```bash
python plot_e2_2.py                                   # newest outputs/e2_2_results_*.json
python plot_e2_2.py outputs/e2_2_results_XXXX.json
```

This writes `<results>_scaling.png/.pdf` and `<results>_table.md`, and prints the
"clean curve" checks:
- accuracy never drops as k grows
- the CI at k=32 sits above the CI at k=1
- the gain per doubling of k shrinks
- SC@32 beats greedy CoT at p < 0.05 (needs `greedy_ref`)

**Checkpointing:** each finished question is appended to `e2_2_checkpoint.jsonl`, on Google
Drive when `drive_output_dir` is set. Re-running the same command continues from the last
finished question. If the sampling settings in `config.yaml` have changed, the script stops
and asks you to use `--fresh`. Increasing `n_eval` later is fine, because the question
order doesn't depend on `n_eval`.

### Configuration (`e2_2` in `config.yaml`)

| Key | Default | What it controls |
|-----|---------|-----------------|
| `n_eval` | 300 | Questions evaluated |
| `k_values` | `[1, 4, 8, 16, 32]` | Vote sizes; the largest sets samples per question |
| `n_resamples` | 100 | Random subsets per (question, k) |
| `n_bootstrap` | 1000 | Bootstrap resamples for the CI |
| `generation.temperature` / `top_p` | 0.7 / 0.95 | Sampling |
| `generation.top_k` / `repetition_penalty` | 0 / 1.0 | Disable Qwen's hidden defaults |
| `generation.samples_per_batch` | 32 | Samples generated in parallel; lower on OOM |
| `greedy_ref` | `""` | E2.1 results JSON for the McNemar comparison |
| `checkpoint_file` | `e2_2_checkpoint.jsonl` | Checkpoint file name |

---

## Planned experiments

| ID | Script | Status |
|----|--------|--------|
| E2.1 | `e2_1_cot_from_scratch.py` | Ready |
| E2.2 | `e2_2_self_consistency.py` | Ready |
| E2.3 | `e2_3_grpo_by_hand.py` | Planned |
| E2.4 | `e2_4_verifier_study.py` | Planned |

All experiments share `config.yaml` and write results to `outputs/`.

---

## Related repositories

| Repo | Stage |
|------|-------|
| `research_stage_00` | Stage 0 — Setup and Orientation |
| `research_stage_01` | Stage 1 — Vision-Language Foundations |
| `research_stage_02` | Stage 2 — Reasoning Methods in LLMs ← you are here |
