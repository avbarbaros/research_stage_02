"""
E2.1 — CoT from Scratch
=======================
Stage 2, Experiment 2.1 · Multimodal Reasoning Thesis Roadmap

Three prompting conditions on GSM8K with a 7B model:
  direct        — answer immediately, no reasoning
  zero_shot_cot — "Let's think step by step."  (Kojima et al., 2022)
  few_shot_cot  — exemplars from few_shot_exemplars.jsonl  (Wei et al., 2022)

All configuration (prompts, generation params, dataset) lives in config.yaml.
Exemplars live in few_shot_exemplars.jsonl.

Dependencies:
  pip install transformers datasets accelerate pyyaml

Usage:
  python e2_1_cot_from_scratch.py
  python e2_1_cot_from_scratch.py --condition direct
  python e2_1_cot_from_scratch.py --n_eval 100
"""

import re
import json
import time
import random
import argparse
import statistics
from datetime import datetime
from pathlib import Path

import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config",    default="config.yaml")
parser.add_argument("--n_eval",    type=int, default=None, help="Override e2_1.n_eval")
parser.add_argument("--condition", default="all",
                    choices=["all", "direct", "zero_shot_cot", "few_shot_cot"])
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open(args.config) as f:
    cfg = yaml.safe_load(f)

common   = cfg["common"]
e21      = cfg["e2_1"]
model_id = cfg["models"]["main"]
prompts  = e21["prompts"]
gen_cfg  = e21["generation"]

n_eval  = args.n_eval if args.n_eval is not None else e21["n_eval"]
seed    = common["seed"]
out_dir = Path(common["output_dir"])
out_dir.mkdir(exist_ok=True)

drive_out_dir = common.get("drive_output_dir", "")
if drive_out_dir:
    drive_out_dir = Path(drive_out_dir)
    drive_out_dir.mkdir(parents=True, exist_ok=True)
else:
    drive_out_dir = None

random.seed(seed)
torch.manual_seed(seed)

dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
dtype = dtype_map[common["dtype"]]

# ---------------------------------------------------------------------------
# Few-shot exemplars
# ---------------------------------------------------------------------------
few_shot_path = Path(e21["few_shot_file"])
with open(few_shot_path) as f:
    FEW_SHOT = [json.loads(line) for line in f if line.strip()]
print(f"Loaded {len(FEW_SHOT)} few-shot exemplars from {few_shot_path}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"E2.1  CoT from Scratch")
print(f"Model : {model_id}")
if torch.cuda.is_available():
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print(f"Loaded {sum(p.numel() for p in model.parameters())/1e9:.2f}B params "
      f"in {time.time()-t0:.1f}s")
if torch.cuda.is_available():
    print(f"VRAM used : {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
print(f"\nLoading {e21['dataset']} ...")
ds = load_dataset(e21["dataset"], e21["dataset_config"])
test_data = list(ds[e21["split"]])
if n_eval is not None:
    random.shuffle(test_data)
    test_data = test_data[:n_eval]
print(f"Using {len(test_data)} / {len(ds[e21['split']])} examples")

# ---------------------------------------------------------------------------
# Answer extraction
#
# Documented pitfalls (recorded here for the Stage 2 gate-check):
#   1. Numbers with commas  (1,234)  → strip before float()
#   2. Numbers with units   ("42 apples") → regex doesn't require end-of-string
#   3. Multiple candidates  → trigger phrase wins; else last number in text
#   4. Negative / decimal   → pattern includes '-' and '.'
#   5. No number at all     → return None (counts as wrong, no crash)
# ---------------------------------------------------------------------------

def extract_gt(raw: str) -> str:
    """Ground-truth integer after '####' in GSM8K answer field."""
    m = re.search(r"####\s*([\-\d,\.]+)", raw)
    if m:
        return m.group(1).replace(",", "").strip()
    raise ValueError(f"No '####' in: {raw!r}")


_TRIGGERS = [
    r"the answer is[:\s]+",
    r"the final answer is[:\s]+",
    r"therefore[,\s]+the answer is[:\s]+",
    r"so the answer is[:\s]+",
    r"####[\s]+",
    r"answer[:\s]+",
]
_NUM_PAT = r"(-?[\d,]+(?:\.\d+)?)"


def extract_pred(text: str) -> str | None:
    """
    Extract numeric answer from model output.
    Priority: last trigger-phrase match → last number in text.
    """
    low = text.lower()
    last_match = None
    for trigger in _TRIGGERS:
        for m in re.finditer(trigger + _NUM_PAT, low):
            last_match = m
    if last_match:
        return last_match.group(1).replace(",", "").strip()
    nums = re.findall(_NUM_PAT, text)
    if nums:
        return nums[-1].replace(",", "").strip()
    return None


def is_correct(pred: str | None, gt: str) -> bool:
    if pred is None:
        return False
    try:
        return float(pred) == float(gt)
    except ValueError:
        return False


# Sanity checks
assert extract_gt("steps\n#### 42") == "42"
assert extract_gt("steps\n#### 1,234") == "1234"
assert extract_pred("The answer is 42 apples.") == "42"
assert extract_pred("Therefore, the answer is -5.") == "-5"
assert extract_pred("She earns $315.") == "315"
assert extract_pred("No numbers here") is None
print("Extraction sanity checks passed.")

# ---------------------------------------------------------------------------
# Prompt builders  (system prompts and user suffixes read from config.yaml)
# ---------------------------------------------------------------------------

def build_direct(q: str) -> list[dict]:
    return [
        {"role": "system", "content": prompts["direct"]["system"]},
        {"role": "user",   "content": q},
    ]


def build_zscot(q: str) -> list[dict]:
    suffix = prompts["zero_shot_cot"].get("user_suffix", "")
    return [
        {"role": "system", "content": prompts["zero_shot_cot"]["system"]},
        {"role": "user",   "content": q + suffix},
    ]


def build_fscot(q: str) -> list[dict]:
    msgs = [{"role": "system", "content": prompts["few_shot_cot"]["system"]}]
    for ex in FEW_SHOT:
        msgs.append({"role": "user",      "content": ex["question"]})
        msgs.append({"role": "assistant", "content": ex["answer"]})
    msgs.append({"role": "user", "content": q})
    return msgs


# Print prompt token counts once for transparency
sample_q = test_data[0]["question"]
print("\nPrompt token counts (sample question):")
for name, fn in [("direct", build_direct), ("zero_shot_cot", build_zscot), ("few_shot_cot", build_fscot)]:
    t = tokenizer.apply_chat_template(fn(sample_q), tokenize=False, add_generation_prompt=True)
    print(f"  {name:<16}: {len(tokenizer.encode(t)):,} tokens")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate(messages: list[dict], max_new_tokens: int) -> tuple[str, str]:
    """Returns (prompt_text, response_text) — both stored verbatim in results."""
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    response_text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    return prompt_text, response_text


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

CONDITIONS = [
    {"name": "direct",        "build": build_direct, "max_tokens": gen_cfg["direct"]["max_new_tokens"]},
    {"name": "zero_shot_cot", "build": build_zscot,  "max_tokens": gen_cfg["cot"]["max_new_tokens"]},
    {"name": "few_shot_cot",  "build": build_fscot,  "max_tokens": gen_cfg["cot"]["max_new_tokens"]},
]
if args.condition != "all":
    CONDITIONS = [c for c in CONDITIONS if c["name"] == args.condition]

all_records: dict[str, list] = {}
summary: dict[str, dict]     = {}

for cond in CONDITIONS:
    cname = cond["name"]
    print(f"\n{'='*60}  {cname}")

    records  = []
    n_ok     = 0
    n_exfail = 0
    t0       = time.time()

    for i, ex in enumerate(test_data):
        gt                   = extract_gt(ex["answer"])
        messages             = cond["build"](ex["question"])
        prompt_text, response = generate(messages, cond["max_tokens"])
        pred                 = extract_pred(response)
        ok                   = is_correct(pred, gt)

        if pred is None:
            n_exfail += 1
        if ok:
            n_ok += 1

        records.append({
            "index":        i,
            "question":     ex["question"],
            "gt_raw":       ex["answer"],          # full GSM8K answer with steps + ####
            "gt":           gt,                    # extracted integer string
            "messages":     messages,              # full chat turns sent to model
            "prompt_text":  prompt_text,           # rendered prompt (after chat template)
            "response":     response,              # model output verbatim
            "pred":         pred,                  # extracted numeric prediction
            "correct":      ok,
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(test_data):
            elapsed = time.time() - t0
            speed   = (i + 1) / elapsed
            eta     = max(0, (len(test_data) - i - 1) / speed)
            print(f"  [{i+1:4d}/{len(test_data)}]  acc={n_ok/(i+1)*100:.1f}%  "
                  f"extr_fail={n_exfail}  {speed:.1f} ex/s  ETA={eta:.0f}s")

    elapsed = time.time() - t0
    lengths = [len(r["response"]) for r in records]
    meta = {
        "accuracy":            round(n_ok / len(test_data) * 100, 4),
        "n_correct":           n_ok,
        "n_total":             len(test_data),
        "extraction_failures": n_exfail,
        "elapsed_s":           round(elapsed, 1),
        "resp_len_mean":       round(statistics.mean(lengths), 1),
        "resp_len_median":     round(statistics.median(lengths), 1),
        "resp_len_max":        max(lengths),
    }
    print(f"  DONE  acc={meta['accuracy']:.2f}%  extr_fail={n_exfail}  {elapsed:.0f}s")

    all_records[cname] = records
    summary[cname]     = meta

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f"\n{'='*65}")
print(f"  SUMMARY  —  {model_id}")
print(f"  GSM8K test ({len(test_data)} examples, greedy decoding)")
print(f"{'='*65}")
print(f"  {'Condition':<18}  {'Accuracy':>10}  {'Extr.fail':>10}  {'AvgRespLen':>11}")
print(f"  {'-'*55}")
for cname, m in summary.items():
    print(f"  {cname:<18}  {m['accuracy']:>9.2f}%  "
          f"{m['extraction_failures']:>10}  {m['resp_len_mean']:>11.0f}")
print(f"{'='*65}")

# ---------------------------------------------------------------------------
# Error analysis: divergent cases between conditions
# ---------------------------------------------------------------------------
def show_divergent(a: str, b: str, n: int = 3):
    if a not in all_records or b not in all_records:
        return
    a_only = [(ra, rb) for ra, rb in zip(all_records[a], all_records[b])
               if ra["correct"] and not rb["correct"]]
    b_only = [(ra, rb) for ra, rb in zip(all_records[a], all_records[b])
               if not ra["correct"] and rb["correct"]]
    print(f"\n  {a} ✓  &  {b} ✗  :  {len(a_only)} cases")
    print(f"  {b} ✓  &  {a} ✗  :  {len(b_only)} cases")
    print(f"\n  --- Sample: {a} correct, {b} wrong ---")
    for ra, rb in a_only[:n]:
        print(f"  Q  : {ra['question'][:100]}...")
        print(f"  GT : {ra['gt']}")
        print(f"  {a:20s}: {ra['pred']!r}")
        print(f"  {b:20s}: {rb['pred']!r}  (response: {rb['response'][:100]!r})")
        print()

if "direct" in all_records and "zero_shot_cot" in all_records:
    show_divergent("direct", "zero_shot_cot")
if "zero_shot_cot" in all_records and "few_shot_cot" in all_records:
    show_divergent("zero_shot_cot", "few_shot_cot")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
output = {
    "experiment": "E2.1",
    "model":      model_id,
    "n_eval":     len(test_data),
    "seed":       seed,
    "summary":    summary,
    "records":    all_records,
}
ts = datetime.now().strftime("%Y%m%d_%H%M")
filename = f"e2_1_results_{ts}.json"

out_path = out_dir / filename
out_path.write_text(json.dumps(output, indent=2))
print(f"\nResults saved → {out_path}")

if drive_out_dir:
    drive_path = drive_out_dir / filename
    drive_path.write_text(json.dumps(output, indent=2))
    print(f"Results saved → {drive_path}  (Google Drive)")

print("Done.")
