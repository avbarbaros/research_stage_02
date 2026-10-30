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
│
├── e2_1_cot_from_scratch.py     # E2.1 — direct / zero-shot CoT / few-shot CoT on GSM8K
│                                #        (scripts for E2.2–E2.4 to be added)
│
└── outputs/                     # Created at runtime; all result JSONs land here
    └── e2_1_results.json
```

---

## Setup (Colab CLI)

```bash
pip install transformers datasets accelerate pyyaml
```

Model weights are downloaded automatically from Hugging Face on first run.
Default model: `Qwen/Qwen2.5-7B-Instruct` (~14 GB in bfloat16, fits on L4).

---

## E2.1 — CoT from Scratch

**Goal:** implement zero-shot CoT, few-shot CoT, and direct prompting on GSM8K with a 7B
model using nothing beyond the inference engine. Reproduce the qualitative ordering of the
three conditions and document every answer-extraction pitfall.

**Papers:** Wei et al. (2022) arXiv:2201.11903 · Kojima et al. (2022) arXiv:2205.11916

### Run

```bash
# All three conditions, 500 examples (default)
python e2_1_cot_from_scratch.py

# Single condition
python e2_1_cot_from_scratch.py --condition direct
python e2_1_cot_from_scratch.py --condition zero_shot_cot
python e2_1_cot_from_scratch.py --condition few_shot_cot

# Quick smoke test
python e2_1_cot_from_scratch.py --n_eval 20

# Override config without editing the file
python e2_1_cot_from_scratch.py --n_eval 200 --condition few_shot_cot
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config.yaml` | Path to config file |
| `--condition` | `all` | `all` / `direct` / `zero_shot_cot` / `few_shot_cot` |
| `--n_eval` | value from config | Number of GSM8K test examples to use |

### Configuration

All knobs live in `config.yaml` under the `e2_1` section — no code edits needed:

| Key | Default | What it controls |
|-----|---------|-----------------|
| `e2_1.n_eval` | 500 | Examples evaluated per condition |
| `e2_1.few_shot_file` | `few_shot_exemplars.jsonl` | Path to exemplar file |
| `e2_1.prompts.direct.system` | see file | System prompt for direct condition |
| `e2_1.prompts.zero_shot_cot.system` | see file | System prompt for zero-shot CoT |
| `e2_1.prompts.zero_shot_cot.user_suffix` | `"\n\nLet's think step by step."` | Appended to each question |
| `e2_1.prompts.few_shot_cot.system` | see file | System prompt for few-shot CoT |
| `e2_1.generation.direct.max_new_tokens` | 64 | Token budget for direct condition |
| `e2_1.generation.cot.max_new_tokens` | 512 | Token budget for CoT conditions |
| `models.main` | `Qwen/Qwen2.5-7B-Instruct` | Inference model |

### Few-shot exemplars

Edit `few_shot_exemplars.jsonl` to change or extend the exemplars — one JSON object per line:

```jsonl
{"question": "...", "answer": "Step 1 ...\nStep 2 ...\nThe answer is 42."}
```

The script loads this file at startup; no code changes required.

### Output format

Results are written to `outputs/e2_1_results.json`. Each record contains the full prompt and
response with no truncation:

```json
{
  "experiment": "E2.1",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "n_eval": 500,
  "summary": {
    "direct":        {"accuracy": 72.40, "extraction_failures": 3, "resp_len_mean": 18, ...},
    "zero_shot_cot": {"accuracy": 84.20, "extraction_failures": 1, "resp_len_mean": 312, ...},
    "few_shot_cot":  {"accuracy": 86.60, "extraction_failures": 0, "resp_len_mean": 287, ...}
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

## Planned experiments

| ID | Script | Status |
|----|--------|--------|
| E2.1 | `e2_1_cot_from_scratch.py` | Ready |
| E2.2 | `e2_2_self_consistency.py` | Planned |
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
