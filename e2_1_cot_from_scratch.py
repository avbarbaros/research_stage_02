"""
E2.1 — CoT from Scratch
=======================
Stage 2, Experiment 2.1 · Multimodal Reasoning Thesis Roadmap

Four prompting conditions on GSM8K with a 7B model:
  direct           — answer immediately, no reasoning
  zero_shot_cot    — "Let's think step by step."  (Kojima et al., 2022)
  few_shot_cot     — hand-written exemplars (few_shot_exemplars.jsonl)
  few_shot_cot_wei — the standard 8 GSM8K exemplars from Wei et al. (2022)
                     (few_shot_exemplars_wei.jsonl)

All configuration (prompts, generation params, dataset, exemplar files) lives
in config.yaml.

Statistics reported per condition: accuracy with a Wilson 95% CI.
Each CoT condition is compared to zero_shot_cot (and direct to zero_shot_cot)
with an exact McNemar test on the paired per-question outcomes.

Dependencies:
  pip install transformers datasets accelerate pyyaml

Usage:
  python e2_1_cot_from_scratch.py
  python e2_1_cot_from_scratch.py --condition direct
  python e2_1_cot_from_scratch.py --n_eval 100
"""

import re
import json
import math
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
                    choices=["all", "direct", "zero_shot_cot", "few_shot_cot", "few_shot_cot_wei"])
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
FEW_SHOT_CONDITIONS = ["few_shot_cot", "few_shot_cot_wei"]


def load_exemplars(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


EXEMPLARS: dict[str, list[dict]] = {}
for _name in FEW_SHOT_CONDITIONS:
    _path = prompts[_name]["exemplars_file"]
    EXEMPLARS[_name] = load_exemplars(_path)
    print(f"Loaded {len(EXEMPLARS[_name])} exemplars for {_name} from {_path}")

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
# Statistics (stdlib only)
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion, in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half   = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round(max(0.0, centre - half) * 100, 2), round(min(1.0, centre + half) * 100, 2))


def mcnemar_exact(b: int, c: int) -> float:
    """
    Exact two-sided McNemar p-value.
    b = questions only A got right, c = questions only B got right.
    Under H0 the discordant pairs split 50/50, so this is a binomial test.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_compare(recs_a: list[dict], recs_b: list[dict]) -> dict:
    """Paired comparison of two conditions evaluated on the same questions."""
    b = sum(ra["correct"] and not rb["correct"] for ra, rb in zip(recs_a, recs_b))
    c = sum(rb["correct"] and not ra["correct"] for ra, rb in zip(recs_a, recs_b))
    n = len(recs_a)
    acc_a = sum(r["correct"] for r in recs_a) / n * 100
    acc_b = sum(r["correct"] for r in recs_b) / n * 100
    return {
        "a_only_correct": b,
        "b_only_correct": c,
        "diff_pp":        round(acc_b - acc_a, 2),   # B minus A, percentage points
        "mcnemar_p":      round(mcnemar_exact(b, c), 4),
    }


assert wilson_ci(17, 20) == (63.96, 94.76)
assert mcnemar_exact(3, 0) == 0.25
assert mcnemar_exact(0, 0) == 1.0
print("Statistics sanity checks passed.")

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


def make_fscot_builder(cname: str):
    """Few-shot builder for one condition (system prompt + its own exemplars)."""
    def build(q: str) -> list[dict]:
        msgs = [{"role": "system", "content": prompts[cname]["system"]}]
        for ex in EXEMPLARS[cname]:
            msgs.append({"role": "user",      "content": ex["question"]})
            msgs.append({"role": "assistant", "content": ex["answer"]})
        msgs.append({"role": "user", "content": q})
        return msgs
    return build


build_fscot     = make_fscot_builder("few_shot_cot")
build_fscot_wei = make_fscot_builder("few_shot_cot_wei")

BUILDERS = [
    ("direct",           build_direct),
    ("zero_shot_cot",    build_zscot),
    ("few_shot_cot",     build_fscot),
    ("few_shot_cot_wei", build_fscot_wei),
]

# Print prompt token counts once for transparency
sample_q = test_data[0]["question"]
print("\nPrompt token counts (sample question):")
for name, fn in BUILDERS:
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
    {"name":       name,
     "build":      fn,
     "max_tokens": gen_cfg["direct" if name == "direct" else "cot"]["max_new_tokens"]}
    for name, fn in BUILDERS
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

        elapsed = time.time() - t0
        speed   = (i + 1) / elapsed
        eta     = max(0, (len(test_data) - i - 1) / speed)
        mark    = "✓" if ok else "✗"
        print(f"  [{i+1:4d}/{len(test_data)}] {mark}  pred={str(pred):>8}  gt={gt:<8}  "
              f"acc={n_ok/(i+1)*100:.1f}%  {speed:.1f} ex/s  ETA={eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    lengths = [len(r["response"]) for r in records]
    ci_lo, ci_hi = wilson_ci(n_ok, len(test_data))
    meta = {
        "accuracy":            round(n_ok / len(test_data) * 100, 4),
        "ci95_low":            ci_lo,
        "ci95_high":           ci_hi,
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
print(f"\n{'='*80}")
print(f"  SUMMARY  —  {model_id}")
print(f"  GSM8K test ({len(test_data)} examples, greedy decoding)")
print(f"{'='*80}")
print(f"  {'Condition':<18}  {'Accuracy':>9}  {'95% CI (Wilson)':>17}  "
      f"{'Extr.fail':>9}  {'AvgRespLen':>10}")
print(f"  {'-'*72}")
for cname, m in summary.items():
    ci = f"[{m['ci95_low']:.1f}, {m['ci95_high']:.1f}]"
    print(f"  {cname:<18}  {m['accuracy']:>8.2f}%  {ci:>17}  "
          f"{m['extraction_failures']:>9}  {m['resp_len_mean']:>10.0f}")
print(f"{'='*80}")

# ---------------------------------------------------------------------------
# Paired significance tests (exact McNemar), all against zero_shot_cot
# ---------------------------------------------------------------------------
comparisons: dict[str, dict] = {}
BASELINE = "zero_shot_cot"
if BASELINE in all_records:
    others = [c for c in all_records if c != BASELINE]
    if others:
        print(f"\n  Paired comparison vs {BASELINE} (exact McNemar)")
        print(f"  {'Condition':<18}  {'Δacc (pp)':>9}  {'only base ✓':>11}  "
              f"{'only this ✓':>11}  {'p-value':>8}")
        print(f"  {'-'*66}")
    for cname in others:
        cmp = paired_compare(all_records[BASELINE], all_records[cname])
        comparisons[f"{cname}_vs_{BASELINE}"] = cmp
        print(f"  {cname:<18}  {cmp['diff_pp']:>+9.2f}  {cmp['a_only_correct']:>11}  "
              f"{cmp['b_only_correct']:>11}  {cmp['mcnemar_p']:>8.4f}")

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
if "zero_shot_cot" in all_records and "few_shot_cot_wei" in all_records:
    show_divergent("zero_shot_cot", "few_shot_cot_wei")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
output = {
    "experiment": "E2.1",
    "model":      model_id,
    "n_eval":     len(test_data),
    "seed":       seed,
    "summary":    summary,
    "comparisons": comparisons,
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
