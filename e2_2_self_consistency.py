"""
E2.2 — Self-consistency (majority voting)
==========================================
Stage 2, Experiment 2.2 · Multimodal Reasoning Thesis Roadmap

Paper: Wang et al. (2022), "Self-Consistency Improves Chain of Thought
Reasoning in Language Models", arXiv:2203.11171

Method
  For each GSM8K question, sample K = max(k_values) chain-of-thought answers
  with temperature sampling, using the same prompt as E2.1 zero_shot_cot.
  For each k in k_values, the answer is the majority vote over k samples.

  Samples are drawn once. Accuracy at a smaller k is the *expected* accuracy
  over `n_resamples` random k-subsets of the K samples, which gives a much
  less noisy curve than using only the first k samples.

Reported per k
  vote accuracy     — majority-vote answer is correct (with bootstrap 95% CI)
  any-correct rate  — at least one of the k samples is correct; an upper
                      bound for any answer-selection method (relevant for E2.4)

Also reported
  - accuracy by vote agreement (how often the majority is right when samples
    agree strongly vs. weakly)
  - optional exact McNemar test vs. E2.1 greedy zero-shot CoT (`greedy_ref`)

Voting rules
  - answers are compared in canonical numeric form ("18.0" == "18")
  - samples with no extractable number do not vote
  - ties go to the answer that appeared first among the voting samples

Checkpointing
  Every finished question is appended to a JSONL checkpoint. Re-running the
  script resumes where it stopped. Use --fresh to start over.

Usage
  python e2_2_self_consistency.py
  python e2_2_self_consistency.py --n_eval 50
  python e2_2_self_consistency.py --greedy_ref outputs/e2_1_results_XXXX.json
  python e2_2_self_consistency.py --analyze_only outputs/e2_2_checkpoint.jsonl   # no GPU needed
  python e2_2_self_consistency.py --model base \
      --greedy_ref outputs/e2_1_results_XXXX_base.json                          # base-model run

Base model (--model base)
  Samples from models.base with the plain-text zero-shot CoT prompt of the E2.1
  base-model run ("Q: ... A: Let's think step by step.", e2_1.base_prompts),
  stopping at the stop strings there. The answer is the trigger phrase / last
  number in each sample (no Kojima second stage, which would double the cost).
  Uses its own checkpoint and results file (suffix "_base").
"""

import json
import time
import random
import argparse
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

from reasoning_utils import (
    select_subset, extract_gt, extract_pred, canonical, is_correct,
    paired_compare, bootstrap_ci, run_self_tests,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config",       default="config.yaml")
parser.add_argument("--n_eval",       type=int, default=None, help="Override e2_2.n_eval")
parser.add_argument("--greedy_ref",   default=None, help="Override e2_2.greedy_ref (E2.1 results JSON)")
parser.add_argument("--fresh",        action="store_true", help="Delete the checkpoint and start over")
parser.add_argument("--analyze_only", default=None, metavar="CHECKPOINT",
                    help="Skip generation; analyse an existing checkpoint JSONL")
parser.add_argument("--model", default="main", choices=["main", "base"],
                    help="'main' = instruction-tuned model with the chat template; "
                         "'base' = models.base with plain-text prompts")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open(args.config) as f:
    cfg = yaml.safe_load(f)

common   = cfg["common"]
e22      = cfg["e2_2"]
model_id = cfg["models"][args.model]
gen_cfg  = e22["generation"]

BASE   = args.model == "base"
SUFFIX = "_base" if BASE else ""                       # keeps base and Instruct files apart
base_cfg = cfg["e2_1"]["base_prompts"]                 # shared with the E2.1 base-model run
STOP     = base_cfg["stop_strings"] if BASE else None

n_eval      = args.n_eval if args.n_eval is not None else e22["n_eval"]
seed        = common["seed"]
k_values    = sorted(e22["k_values"])
max_k       = k_values[-1]
n_resamples = e22["n_resamples"]
n_bootstrap = e22["n_bootstrap"]
greedy_ref  = args.greedy_ref if args.greedy_ref is not None else e22.get("greedy_ref", "")

out_dir = Path(common["output_dir"])
out_dir.mkdir(exist_ok=True)
drive_out_dir = common.get("drive_output_dir", "")
if drive_out_dir:
    drive_out_dir = Path(drive_out_dir)
    drive_out_dir.mkdir(parents=True, exist_ok=True)
else:
    drive_out_dir = None

# Checkpoint lives on Drive when available: Colab's local disk is lost on disconnect.
_ckpt = Path(e22["checkpoint_file"])
ckpt_path = (drive_out_dir or out_dir) / f"{_ckpt.stem}{SUFFIX}{_ckpt.suffix}"

run_self_tests()

# Settings that change what the samples mean. A checkpoint made with different
# values cannot be resumed (samples_per_batch is recorded but not checked).
RUN_HEADER = {
    "experiment":         "E2.2",
    "model":              model_id,
    "seed":               seed,
    "max_k":              max_k,
    "system":             e22["prompts"]["system"],
    "user_suffix":        e22["prompts"].get("user_suffix", ""),
    "temperature":        gen_cfg["temperature"],
    "top_p":              gen_cfg["top_p"],
    "top_k":              gen_cfg["top_k"],
    "repetition_penalty": gen_cfg["repetition_penalty"],
    "max_new_tokens":     gen_cfg["max_new_tokens"],
}
if BASE:
    # Only added for base runs, so existing Instruct checkpoints still resume.
    RUN_HEADER.update({
        "prompt_format":    "completion",
        "system":           None,
        "user_suffix":      None,
        "reasoning_prefix": base_cfg["zero_shot_cot"]["reasoning_prefix"],
        "stop_strings":     STOP,
    })


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def read_checkpoint(path: Path) -> tuple[dict | None, list[dict]]:
    if not path.exists():
        return None, []
    header, records = None, []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Last line may be cut off if the runtime died mid-write.
                print(f"  (skipping unreadable line in {path})")
                continue
            if "_header" in obj:
                header = obj["_header"]
            else:
                records.append(obj)
    return header, records


def append_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Voting + analysis (pure Python; used by both generation and --analyze_only)
# ---------------------------------------------------------------------------

def majority_vote(preds: list[str | None]) -> tuple[str | None, int]:
    """
    Majority answer among canonical predictions, ignoring None.
    Ties → the answer seen first (Counter keeps insertion order and
    most_common() is a stable sort).
    """
    votes = Counter(p for p in preds if p is not None)
    if not votes:
        return None, 0
    ans, count = votes.most_common(1)[0]
    return ans, count


assert majority_vote(["5", "3", "3", None]) == ("3", 2)
assert majority_vote(["5", "3", None]) == ("5", 1)     # tie → first seen
assert majority_vote([None, None]) == (None, 0)


def analyse(records: list[dict]) -> dict:
    """Vote accuracy / any-correct curves over k, agreement bins, sample stats."""
    rng = random.Random(seed)
    n_q = len(records)

    # Per question: canonical predictions and per-sample correctness.
    qs = []
    for rec in records:
        preds = [canonical(s["pred"]) for s in rec["samples"]]
        ok    = [is_correct(p, rec["gt"]) for p in preds]
        qs.append((preds, ok, rec["gt"]))

    n_samples = min(len(p) for p, _, _ in qs)
    usable_k  = [k for k in k_values if k <= n_samples]

    curve = {}
    for k in usable_k:
        vote_scores, any_scores = [], []
        for preds, ok, gt in qs:
            n = len(preds)
            if k == 1:
                subsets = [[j] for j in range(n)]              # exact: every sample
            elif k >= n:
                subsets = [list(range(n))]                     # only one subset
            else:
                subsets = [rng.sample(range(n), k) for _ in range(n_resamples)]
            v = a = 0
            for sub in subsets:
                ans, _ = majority_vote([preds[j] for j in sub])
                v += is_correct(ans, gt)
                a += any(ok[j] for j in sub)
            vote_scores.append(v / len(subsets))
            any_scores.append(a / len(subsets))

        acc = statistics.mean(vote_scores) * 100
        lo, hi = bootstrap_ci(vote_scores, n_bootstrap, seed)
        curve[k] = {
            "vote_accuracy":    round(acc, 2),
            "ci95_low":         lo,
            "ci95_high":        hi,
            "any_correct_rate": round(statistics.mean(any_scores) * 100, 2),
        }

    # Agreement at the full sample set: share of all samples that back the winner.
    bins = [(0.0, 0.5, "<50%"), (0.5, 0.8, "50–80%"), (0.8, 1.01, "≥80%")]
    by_bin = {label: [] for _, _, label in bins}
    per_question = []
    for preds, ok, gt in qs:
        ans, count = majority_vote(preds)
        agreement  = count / len(preds)
        correct    = is_correct(ans, gt)
        per_question.append({"vote": ans, "agreement": round(agreement, 4), "correct": correct})
        for lo_b, hi_b, label in bins:
            if lo_b <= agreement < hi_b:
                by_bin[label].append(correct)
    agreement_bins = {
        label: {"n": len(v), "accuracy": round(sum(v) / len(v) * 100, 2) if v else None}
        for label, v in by_bin.items()
    }

    all_samples = [s for rec in records for s in rec["samples"]]
    sample_stats = {
        "n_samples_total":     len(all_samples),
        "extraction_failures": sum(s["pred"] is None for s in all_samples),
        "truncated":           sum(s["truncated"] for s in all_samples),
        "tokens_mean":         round(statistics.mean(s["n_tokens"] for s in all_samples), 1),
        "tokens_median":       statistics.median(s["n_tokens"] for s in all_samples),
        "distinct_answers_mean": round(statistics.mean(
            len({p for p in preds if p is not None}) for preds, _, _ in qs), 2),
    }

    return {
        "n_questions":      n_q,
        "samples_per_q":    n_samples,
        "curve":            curve,
        "agreement_bins":   agreement_bins,
        "sample_stats":     sample_stats,
        "per_question":     per_question,
    }


def compare_with_greedy(records: list[dict], per_question: list[dict], ref_path: str,
                        records_model: str | None = None) -> dict | None:
    """
    Exact McNemar: E2.1 greedy zero-shot CoT vs. self-consistency at max k.
    Both must come from the same model (records_model, default: this run's model).
    """
    path = Path(ref_path)
    if not path.exists():
        print(f"  greedy_ref not found: {path} — skipping comparison")
        return None
    ref = json.loads(path.read_text())
    sc_model = records_model or model_id
    if ref.get("model") != sc_model:
        print(f"  greedy_ref model ({ref.get('model')}) ≠ self-consistency model ({sc_model}) "
              f"— skipping comparison: it would mix two models")
        return None
    greedy = {r["question"]: r["correct"] for r in ref["records"].get("zero_shot_cot", [])}
    pairs = [(greedy[rec["question"]], pq["correct"])
             for rec, pq in zip(records, per_question) if rec["question"] in greedy]
    if not pairs:
        print("  greedy_ref has no zero_shot_cot records matching these questions — skipping")
        return None
    cmp = paired_compare([g for g, _ in pairs], [s for _, s in pairs])
    cmp["greedy_accuracy"] = round(sum(g for g, _ in pairs) / len(pairs) * 100, 2)
    cmp["sc_accuracy"]     = round(sum(s for _, s in pairs) / len(pairs) * 100, 2)
    cmp["ref_file"]        = str(path)
    return cmp


def print_report(res: dict, cmp: dict | None) -> None:
    print(f"\n{'='*72}")
    print(f"  E2.2 SELF-CONSISTENCY  —  {model_id}")
    print(f"  GSM8K test, {res['n_questions']} questions × {res['samples_per_q']} samples  "
          f"(T={gen_cfg['temperature']}, top_p={gen_cfg['top_p']})")
    print(f"{'='*72}")
    print(f"  {'k':>3}  {'Vote acc':>9}  {'95% CI (bootstrap)':>19}  {'Any-correct':>11}")
    print(f"  {'-'*50}")
    for k, m in res["curve"].items():
        ci = f"[{m['ci95_low']:.1f}, {m['ci95_high']:.1f}]"
        print(f"  {k:>3}  {m['vote_accuracy']:>8.2f}%  {ci:>19}  {m['any_correct_rate']:>10.2f}%")
    print(f"{'='*72}")

    print(f"\n  Accuracy by vote agreement (k={res['samples_per_q']})")
    for label, b in res["agreement_bins"].items():
        acc = f"{b['accuracy']:.1f}%" if b["accuracy"] is not None else "—"
        print(f"    agreement {label:>7}:  n={b['n']:<4}  accuracy={acc}")

    s = res["sample_stats"]
    print(f"\n  Samples: {s['n_samples_total']}  |  no answer extracted: {s['extraction_failures']}  "
          f"|  hit max_new_tokens: {s['truncated']}")
    print(f"  Tokens per sample: mean {s['tokens_mean']}, median {s['tokens_median']}  "
          f"|  distinct answers per question: {s['distinct_answers_mean']}")

    if cmp:
        print(f"\n  vs. E2.1 greedy zero-shot CoT ({cmp['n']} matched questions, exact McNemar)")
        print(f"    greedy {cmp['greedy_accuracy']:.2f}%  →  SC@{res['samples_per_q']} "
              f"{cmp['sc_accuracy']:.2f}%   Δ = {cmp['diff_pp']:+.2f} pp")
        print(f"    only greedy ✓: {cmp['a_only_correct']}   only SC ✓: {cmp['b_only_correct']}   "
              f"p = {cmp['mcnemar_p']:.4f}")


def save_results(records: list[dict], res: dict, cmp: dict | None, header: dict) -> None:
    for rec, pq in zip(records, res["per_question"]):
        rec["vote_at_max_k"] = pq["vote"]
        rec["agreement"]     = pq["agreement"]
        rec["correct"]       = pq["correct"]
    output = {
        "experiment":     "E2.2",
        "model":          header.get("model", model_id),
        "n_eval":         res["n_questions"],
        "seed":           header.get("seed", seed),
        "run_settings":   header,
        "k_values":       list(res["curve"].keys()),
        "n_resamples":    n_resamples,
        "summary":        res["curve"],
        "agreement_bins": res["agreement_bins"],
        "sample_stats":   res["sample_stats"],
        "vs_greedy":      cmp,
        "records":        records,
    }
    tag = "_base" if header.get("prompt_format") == "completion" else ""
    filename = f"e2_2_results_{datetime.now().strftime('%Y%m%d_%H%M')}{tag}.json"
    text = json.dumps(output, indent=2)
    (out_dir / filename).write_text(text)
    print(f"\nResults saved → {out_dir / filename}")
    if drive_out_dir:
        (drive_out_dir / filename).write_text(text)
        print(f"Results saved → {drive_out_dir / filename}  (Google Drive)")


# ---------------------------------------------------------------------------
# Analyse-only mode: no model, no GPU
# ---------------------------------------------------------------------------
if args.analyze_only:
    header, records = read_checkpoint(Path(args.analyze_only))
    if not records:
        raise SystemExit(f"No records in {args.analyze_only}")
    records.sort(key=lambda r: r["index"])
    if header and header.get("seed") != seed:
        print(f"  Note: checkpoint seed {header.get('seed')} ≠ config seed {seed}")
    res = analyse(records)
    # Compare against the model that made the checkpoint, not the --model flag.
    ckpt_model = (header or {}).get("model")
    cmp = (compare_with_greedy(records, res["per_question"], greedy_ref, ckpt_model)
           if greedy_ref else None)
    print_report(res, cmp)
    save_results(records, res, cmp, header or {})
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# Model (imported here so --analyze_only works without torch installed)
# ---------------------------------------------------------------------------
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

random.seed(seed)
dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[common["dtype"]]

print(f"\n{'='*60}")
print(f"E2.2  Self-consistency")
print(f"Model : {model_id}  ({'base, plain-text prompt' if BASE else 'chat template'})")
print(f"Checkpoint file: {ckpt_path}")
if torch.cuda.is_available():
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s")

# Every token that ends a sequence (Qwen: <|im_end|> and <|endoftext|>).
_eos = model.generation_config.eos_token_id
EOS_IDS = set(_eos if isinstance(_eos, list) else [_eos]) | {tokenizer.eos_token_id}
PAD_ID  = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

# ---------------------------------------------------------------------------
# Dataset (same subset logic and seed as E2.1)
# ---------------------------------------------------------------------------
ds = load_dataset(e22["dataset"], e22["dataset_config"])
test_data = select_subset(ds[e22["split"]], n_eval, seed)
print(f"Using {len(test_data)} / {len(ds[e22['split']])} examples, {max_k} samples each")

# ---------------------------------------------------------------------------
# Checkpoint: resume or start
# ---------------------------------------------------------------------------
if args.fresh and ckpt_path.exists():
    ckpt_path.unlink()
    print(f"Deleted checkpoint {ckpt_path}")

header, done = read_checkpoint(ckpt_path)
if header is not None:
    diffs = {k: (header.get(k), v) for k, v in RUN_HEADER.items() if header.get(k) != v}
    if diffs:
        print("Checkpoint was made with different settings:")
        for k, (old, new) in diffs.items():
            print(f"  {k}: checkpoint={old!r}  config={new!r}")
        raise SystemExit(f"Re-run with --fresh to discard {ckpt_path}, or restore the old settings.")
    print(f"Resuming from {ckpt_path}: {len(done)} questions already done")
else:
    append_jsonl(ckpt_path, {"_header": {**RUN_HEADER,
                                         "samples_per_batch": gen_cfg["samples_per_batch"],
                                         "started": datetime.now().isoformat(timespec="seconds")}})
    header = RUN_HEADER
    print(f"Checkpoint → {ckpt_path}")

done_by_index = {r["index"]: r for r in done}
for i, r in done_by_index.items():
    if i < len(test_data) and r["question"] != test_data[i]["question"]:
        raise SystemExit(f"Checkpoint question {i} does not match the dataset subset. "
                         f"Use --fresh.")

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def build_prompt(q: str) -> str:
    """Chat-template prompt (Instruct), or the plain-text E2.1 base zero-shot CoT prompt."""
    if BASE:
        return f"Q: {q}\nA: {base_cfg['zero_shot_cot']['reasoning_prefix']}"
    messages = [
        {"role": "system", "content": e22["prompts"]["system"]},
        {"role": "user",   "content": q + e22["prompts"].get("user_suffix", "")},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def sample_answers(prompt_text: str, n: int) -> list[dict]:
    """Draw n sampled completions for one prompt, in batches of samples_per_batch."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    samples, remaining = [], n
    while remaining > 0:
        b = min(gen_cfg["samples_per_batch"], remaining)
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            top_k=gen_cfg["top_k"],
            repetition_penalty=gen_cfg["repetition_penalty"],
            max_new_tokens=gen_cfg["max_new_tokens"],
            num_return_sequences=b,
            pad_token_id=PAD_ID,
            **({"stop_strings": STOP, "tokenizer": tokenizer} if STOP else {}),
        )
        for seq in out[:, prompt_len:].tolist():
            # A sequence ends at EOS, or (base model) is padded after a stop string.
            end = next((j for j, t in enumerate(seq) if t in EOS_IDS or t == PAD_ID), None)
            text_full = tokenizer.decode(seq if end is None else seq[:end], skip_special_tokens=True)
            text = text_full
            for s in STOP or []:
                text = text.split(s)[0]
            stopped = text != text_full                    # cut at a stop string
            truncated = end is None and not stopped
            n_tokens  = len(seq) if end is None else end
            text = text.strip()
            samples.append({
                "response":  text,
                "pred":      extract_pred(text),
                "n_tokens":  n_tokens,
                "truncated": truncated,
            })
        remaining -= b
    return samples


records: list[dict] = []
t_start = time.time()
n_new = 0
for i, ex in enumerate(test_data):
    if i in done_by_index:
        records.append(done_by_index[i])
        continue

    # Seed per question so a resumed run draws the same samples as an
    # uninterrupted one (up to GPU non-determinism).
    torch.manual_seed(seed + i)

    gt          = extract_gt(ex["answer"])
    prompt_text = build_prompt(ex["question"])
    tq          = time.time()
    samples     = sample_answers(prompt_text, max_k)

    rec = {
        "index":       i,
        "question":    ex["question"],
        "gt_raw":      ex["answer"],
        "gt":          gt,
        "prompt_text": prompt_text,
        "samples":     samples,
    }
    append_jsonl(ckpt_path, rec)
    records.append(rec)
    n_new += 1

    preds      = [canonical(s["pred"]) for s in samples]
    vote, cnt  = majority_vote(preds)
    sample_acc = sum(is_correct(p, gt) for p in preds) / len(preds) * 100
    run_vote   = sum(is_correct(majority_vote([canonical(s["pred"]) for s in r["samples"]])[0], r["gt"])
                     for r in records) / len(records) * 100
    per_q      = (time.time() - t_start) / n_new
    eta        = per_q * (len(test_data) - i - 1)
    mark       = "✓" if is_correct(vote, gt) else "✗"
    print(f"  [{i+1:4d}/{len(test_data)}] {mark}  vote={str(vote):>8}  gt={gt:<8}  "
          f"agree={cnt}/{len(preds):<3} sample_acc={sample_acc:5.1f}%  "
          f"SC@{max_k}={run_vote:5.1f}%  {time.time()-tq:5.1f}s  ETA={eta/60:.0f}min", flush=True)

print(f"\nGeneration done: {n_new} new questions in {(time.time()-t_start)/60:.1f} min")

# ---------------------------------------------------------------------------
# Analysis + save
# ---------------------------------------------------------------------------
res = analyse(records)
cmp = compare_with_greedy(records, res["per_question"], greedy_ref) if greedy_ref else None
print_report(res, cmp)
save_results(records, res, cmp, header)
print("Done.")
