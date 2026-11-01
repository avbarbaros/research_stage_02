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

Base-model ablation (--model base):
  Runs the same conditions on models.base (Qwen2.5-7B, no instruction tuning).
  A base model has no chat template, so prompts are plain "Q: ... A:" text as in
  the original papers, generation stops at the next "Q:", and zero-shot CoT uses
  Kojima et al.'s two-stage answer extraction. See e2_1.base_prompts in config.yaml.

Usage:
  python e2_1_cot_from_scratch.py
  python e2_1_cot_from_scratch.py --condition direct
  python e2_1_cot_from_scratch.py --n_eval 100
  python e2_1_cot_from_scratch.py --model base
"""

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

from reasoning_utils import (
    select_subset, extract_gt, extract_pred, extract_first_number, is_correct,
    wilson_ci, paired_compare, run_self_tests,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config",    default="config.yaml")
parser.add_argument("--n_eval",    type=int, default=None, help="Override e2_1.n_eval")
parser.add_argument("--condition", default="all",
                    choices=["all", "direct", "zero_shot_cot", "few_shot_cot", "few_shot_cot_wei"])
parser.add_argument("--model", default="main", choices=["main", "base"],
                    help="'main' = instruction-tuned model with chat prompts; "
                         "'base' = models.base with plain-text Q/A prompts")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open(args.config) as f:
    cfg = yaml.safe_load(f)

common   = cfg["common"]
e21      = cfg["e2_1"]
model_id = cfg["models"][args.model]
prompts  = e21["prompts"]
gen_cfg  = e21["generation"]

BASE        = args.model == "base"        # plain-text prompts instead of the chat template
base_cfg    = e21["base_prompts"]
STOP        = base_cfg["stop_strings"] if BASE else None

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
print(f"Model : {model_id}  ({'base, plain-text prompts' if BASE else 'chat template'})")
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
test_data = select_subset(ds[e21["split"]], n_eval, seed)
print(f"Using {len(test_data)} / {len(ds[e21['split']])} examples")

# ---------------------------------------------------------------------------
# Answer extraction + statistics: shared with other experiments
# (see reasoning_utils.py for the documented extraction pitfalls)
# ---------------------------------------------------------------------------
run_self_tests()

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

# ---------------------------------------------------------------------------
# Plain-text prompts for the base model (--model base)
# Format follows Wei et al. / Kojima et al.: "Q: <question>\nA: <answer>"
# ---------------------------------------------------------------------------

def base_direct_prompt(q: str) -> str:
    return f"Q: {q}\nA: {base_cfg['direct']['answer_prefix']}"


def base_zscot_prompt(q: str) -> str:
    return f"Q: {q}\nA: {base_cfg['zero_shot_cot']['reasoning_prefix']}"


def base_fscot_prompt(cname: str, q: str) -> str:
    shots = "".join(f"Q: {ex['question']}\nA: {ex['answer']}\n\n" for ex in EXEMPLARS[cname])
    return shots + f"Q: {q}\nA:"


def render_chat(messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def preview_prompt(cname: str, q: str) -> str:
    """The (first) prompt text sent to the model for this condition."""
    if not BASE:
        return render_chat(dict(BUILDERS)[cname](q))
    if cname == "direct":
        return base_direct_prompt(q)
    if cname == "zero_shot_cot":
        return base_zscot_prompt(q)
    return base_fscot_prompt(cname, q)


# Print prompt token counts once for transparency
sample_q = test_data[0]["question"]
print("\nPrompt token counts (sample question):")
for name, _ in BUILDERS:
    print(f"  {name:<16}: {len(tokenizer.encode(preview_prompt(name, sample_q))):,} tokens")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate(prompt_text: str, max_new_tokens: int, repetition_penalty: float,
             stop_strings: list[str] | None = None) -> str:
    """
    Greedy decoding of an already-rendered prompt. Returns the response text.

    Qwen2.5's generation_config.json sets do_sample/temperature/top_p/top_k and
    repetition_penalty=1.05, which generate() applies unless overridden. All of
    them are set explicitly here so decoding is plain greedy.

    stop_strings (base model only) end generation at e.g. the next "Q:", and are
    cut from the returned text.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    stop_kwargs = {"stop_strings": stop_strings, "tokenizer": tokenizer} if stop_strings else {}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        **stop_kwargs,
    )
    text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    for s in stop_strings or []:
        text = text.split(s)[0]
    return text.strip()


def answer(cname: str, q: str, gen: dict) -> dict:
    """
    Run one condition on one question. Returns the fields stored per record:
    messages (chat mode only), prompt_text, response, pred.
    """
    max_new, rep = gen["max_new_tokens"], gen["repetition_penalty"]

    # Instruction-tuned model: chat template, extract from the whole response.
    if not BASE:
        messages = dict(BUILDERS)[cname](q)
        prompt_text = render_chat(messages)
        response = generate(prompt_text, max_new, rep)
        return {"messages": messages, "prompt_text": prompt_text,
                "response": response, "pred": extract_pred(response)}

    # Base model: plain text.
    if cname == "direct":
        # The prompt ends with "The answer (arabic numerals) is", so the first
        # number in the completion is the answer.
        prompt_text = base_direct_prompt(q)
        response = generate(prompt_text, max_new, rep, STOP)
        pred = extract_first_number(response)

    elif cname == "zero_shot_cot":
        # Kojima et al.'s two stages: (1) reasoning, (2) append the answer
        # trigger after the reasoning and read the first number that follows.
        zs = base_cfg["zero_shot_cot"]
        prompt_text = base_zscot_prompt(q)
        reasoning = generate(prompt_text, max_new, rep, STOP)
        stage2 = f"{prompt_text} {reasoning}\n{zs['answer_trigger']}"
        final = generate(stage2, zs["answer_max_new_tokens"], rep, STOP)
        response = f"{reasoning}\n{zs['answer_trigger']} {final}"
        pred = extract_first_number(final)

    else:
        # Few-shot: exemplars end with "The answer is N.", so the normal
        # trigger-phrase extraction applies.
        prompt_text = base_fscot_prompt(cname, q)
        response = generate(prompt_text, max_new, rep, STOP)
        pred = extract_pred(response)

    return {"messages": None, "prompt_text": prompt_text, "response": response, "pred": pred}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

CONDITIONS = [
    {"name":       name,
     "build":      fn,
     "gen":        gen_cfg["direct" if name == "direct" else "cot"]}
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
        gt   = extract_gt(ex["answer"])
        res  = answer(cname, ex["question"], cond["gen"])
        pred = res["pred"]
        ok   = is_correct(pred, gt)

        if pred is None:
            n_exfail += 1
        if ok:
            n_ok += 1

        records.append({
            "index":        i,
            "question":     ex["question"],
            "gt_raw":       ex["answer"],          # full GSM8K answer with steps + ####
            "gt":           gt,                    # extracted integer string
            "messages":     res["messages"],       # chat turns sent to model (None for base)
            "prompt_text":  res["prompt_text"],    # rendered prompt (first stage for base zero-shot CoT)
            "response":     res["response"],       # model output verbatim
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
        cmp = paired_compare([r["correct"] for r in all_records[BASELINE]],
                             [r["correct"] for r in all_records[cname]])
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
    "prompt_format": "completion" if BASE else "chat",
    "n_eval":     len(test_data),
    "seed":       seed,
    "generation": gen_cfg,
    **({"base_prompts": base_cfg} if BASE else {}),
    "summary":    summary,
    "comparisons": comparisons,
    "records":    all_records,
}
ts = datetime.now().strftime("%Y%m%d_%H%M")
filename = f"e2_1_results_{ts}{'_base' if BASE else ''}.json"

out_path = out_dir / filename
out_path.write_text(json.dumps(output, indent=2))
print(f"\nResults saved → {out_path}")

if drive_out_dir:
    drive_path = drive_out_dir / filename
    drive_path.write_text(json.dumps(output, indent=2))
    print(f"Results saved → {drive_path}  (Google Drive)")

print("Done.")
