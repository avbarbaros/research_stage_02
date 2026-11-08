"""
E2.4 — Verifier Study
======================
Stage 2, Experiment 2.4 · Multimodal Reasoning Thesis Roadmap

Paper: Lightman et al. (2023), Let's Verify Step by Step, arXiv:2305.20050

Trains two reward models on GSM8K traces and compares best-of-n reranking accuracy:
  ORM — outcome reward model: scores the full solution as correct / incorrect
  PRM — process reward model: scores each reasoning step, aggregates by minimum
        step label strategy: outcome_propagation (all steps inherit the final answer label)

Best-of-n for n ∈ {1, 4, 8, 16, 32}:
  ORM:    pick the solution with the highest ORM score among n candidates
  PRM:    pick the solution with the highest PRM min-step score among n candidates
  Oracle: pick any correct solution among n candidates (upper bound)
  k=1:    single sample, no reranking (flat baseline)

Success criterion (roadmap §2.2): process supervision outperforms outcome supervision
at matched n, replicating the qualitative finding of Lightman et al. (2023).

Phases
  generate_train — sample CPQ_TRAIN completions per training question; create/overwrite
                   the traces file with train records only
  generate_eval  — sample N_EVAL_COMPLETIONS per eval question; append to the traces file
                   (creates it if it does not exist yet).  Uses at most 8 completions per
                   model.generate() call and 1 question per batch to stay within the 24 GB
                   GPU budget (naive 32 × 4 = 128-sequence batches cause OOM on L4).
  generate       — run generate_train then generate_eval in one session
  train_orm      — fine-tune an RM backbone on (question + solution → 0/1) labels
  train_prm      — fine-tune on (question + partial_solution → 0/1) with outcome
                   propagation: each step prefix inherits the final answer label
  evaluate       — load both RMs, re-score n eval completions per question, compute
                   accuracy at each n in best_of_n_values
  all            — run all phases in sequence

Outputs (in common.output_dir or drive_output_dir)
  e2_4_traces_<stem>.jsonl      train + eval traces with correctness labels
  e2_4_orm_<stem>/              fine-tuned ORM checkpoint
  e2_4_prm_<stem>/              fine-tuned PRM checkpoint
  e2_4_results_<stem>.json      accuracy tables and stats

Usage
  python e2_4_verifier_study.py --phase all
  python e2_4_verifier_study.py --phase generate_train
  python e2_4_verifier_study.py --phase generate_eval  --traces_file outputs/e2_4_traces_STEM.jsonl
  python e2_4_verifier_study.py --phase train_orm --traces_file outputs/e2_4_traces_STEM.jsonl
  python e2_4_verifier_study.py --phase train_prm  --traces_file outputs/e2_4_traces_STEM.jsonl
  python e2_4_verifier_study.py --phase evaluate \\
      --traces_file outputs/e2_4_traces_STEM.jsonl \\
      --orm_path outputs/e2_4_orm_STEM \\
      --prm_path outputs/e2_4_prm_STEM
  python e2_4_verifier_study.py --phase all --n_smoke 20   # quick sanity check
"""

import json
import time
import random
import argparse
from datetime import datetime
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset

from reasoning_utils import (
    select_subset, extract_gt, extract_pred, is_correct, wilson_ci, paired_compare,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config",       default="config.yaml")
parser.add_argument("--phase",        required=True,
                    choices=["generate", "generate_train", "generate_eval",
                             "train_orm", "train_prm", "evaluate", "all"])
parser.add_argument("--traces_file",  default=None, help="Traces JSONL for phases 2–4")
parser.add_argument("--orm_path",     default=None, help="Trained ORM dir for evaluate")
parser.add_argument("--prm_path",     default=None, help="Trained PRM dir for evaluate")
parser.add_argument("--stem",         default=None, help="Output stem (default: datestamp)")
parser.add_argument("--n_smoke",      type=int, default=None,
                    help="Smoke-test: use at most this many questions per split")
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))
common, c4 = cfg["common"], cfg["e2_4"]
seed = common["seed"]
random.seed(seed)
torch.manual_seed(seed)

BEST_OF_N          = c4["best_of_n_values"]          # [1, 4, 8, 16, 32]
N_TRAIN_TRACES     = c4["n_train_traces"]              # 5000
N_EVAL             = c4["n_eval"]                      # 300
N_EVAL_COMPLETIONS = max(BEST_OF_N)                    # 32 — generated once, subsetted at eval
CPQ_TRAIN          = 5                                 # completions per training question
N_TRAIN_Q          = N_TRAIN_TRACES // CPQ_TRAIN       # 1000

if args.n_smoke:
    N_TRAIN_Q          = min(N_TRAIN_Q, args.n_smoke)
    N_EVAL             = min(N_EVAL, args.n_smoke)
    N_EVAL_COMPLETIONS = min(N_EVAL_COMPLETIONS, 4)
    BEST_OF_N          = [n for n in BEST_OF_N if n <= N_EVAL_COMPLETIONS] or [1]

orm_cfg = c4["outcome_rm_training"]
prm_cfg = c4["process_rm_training"]

out_dir  = Path(common["output_dir"]); out_dir.mkdir(exist_ok=True)
drive    = Path(common["drive_output_dir"]) if common.get("drive_output_dir") else None
save_dir = drive or out_dir
if drive:
    drive.mkdir(parents=True, exist_ok=True)

stem         = args.stem or datetime.now().strftime("%Y%m%d_%H%M")
traces_path  = Path(args.traces_file) if args.traces_file else save_dir / f"e2_4_traces_{stem}.jsonl"
orm_dir_path = Path(args.orm_path)    if args.orm_path    else save_dir / f"e2_4_orm_{stem}"
prm_dir_path = Path(args.prm_path)    if args.prm_path    else save_dir / f"e2_4_prm_{stem}"
results_path = save_dir / f"e2_4_results_{stem}.json"

GEN_MODEL_ID = cfg["models"]["main"]
RM_MODEL_ID  = cfg["models"].get("outcome_rm") or cfg["models"]["grpo_policy"]  # 1.5B fallback
dtype        = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}[common["dtype"]]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_gsm8k(split: str) -> list[dict]:
    ds = load_dataset(c4["dataset"], c4["dataset_config"], split=split)
    return [{"question": ex["question"], "gt": extract_gt(ex["answer"])} for ex in ds]


def format_gen_prompt(question: str, tok) -> str:
    msgs = [
        {"role": "system", "content": c4["prompts"]["generator_system"]},
        {"role": "user",   "content": question + c4["prompts"]["generator_user_suffix"]},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def format_rm_input(question: str, solution: str) -> str:
    return f"Question: {question}\n\nSolution: {solution}"


def split_steps(text: str) -> list[str]:
    """Split a CoT completion into reasoning steps, preferring paragraph breaks."""
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(parts) >= 2:
        return parts
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    return parts if len(parts) >= 2 else [text]


# ---------------------------------------------------------------------------
# Phase 1: Generate traces
# ---------------------------------------------------------------------------
def _load_generator():
    """Load the generator LLM. Caller is responsible for del + torch.cuda.empty_cache()."""
    print(f"[generate] Loading generator: {GEN_MODEL_ID}", flush=True)
    gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
    gen_tok.padding_side = "left"
    gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_ID, torch_dtype=dtype, device_map="auto")
    dev = gen_model.device

    eos_id = gen_model.generation_config.eos_token_id
    EOS = torch.tensor(
        sorted(set(eos_id if isinstance(eos_id, list) else [eos_id]) | {gen_tok.eos_token_id}),
        device=dev,
    )
    PAD = gen_tok.pad_token_id or gen_tok.eos_token_id

    @torch.no_grad()
    def gen_batch(questions: list[str], n: int) -> list[str]:
        """Return n completions per question, ordered question-major."""
        prompts = [format_gen_prompt(q, gen_tok) for q in questions]
        enc = gen_tok(prompts, return_tensors="pt", padding=True).to(dev)
        P = enc["input_ids"].shape[1]
        out = gen_model.generate(
            **enc, num_return_sequences=n, max_new_tokens=512,
            do_sample=True, temperature=0.7, top_p=0.95, top_k=0,
            repetition_penalty=1.0, pad_token_id=PAD,
        )
        texts = []
        for row in out:
            comp = row[P:]
            is_eos = torch.isin(comp, EOS)
            L = (is_eos.int().argmax().item() + 1) if is_eos.any() else len(comp)
            texts.append(gen_tok.decode(comp[:L], skip_special_tokens=True))
        return texts

    return gen_model, gen_tok, gen_batch


def _write_split(split: str, questions: list[dict], n_per_q: int, f, gen_batch) -> None:
    """Write completions to open file f; flushes after every question batch.

    Uses at most CHUNK completions per model.generate() call to stay within the 24 GB
    GPU budget.  For train (n_per_q=5) BATCH_Q=4 keeps throughput up.  For eval
    (n_per_q=32) BATCH_Q=1 and four 8-completion calls per question avoid OOM.
    """
    CHUNK   = 8
    BATCH_Q = 4 if n_per_q <= CHUNK else 1

    for i in range(0, len(questions), BATCH_Q):
        batch = questions[i:i + BATCH_Q]
        per_q: list[list[str]] = [[] for _ in batch]

        remaining = n_per_q
        while remaining > 0:
            n_chunk = min(CHUNK, remaining)
            texts   = gen_batch([ex["question"] for ex in batch], n_chunk)
            for j in range(len(batch)):
                per_q[j].extend(texts[j * n_chunk:(j + 1) * n_chunk])
            remaining -= n_chunk

        for j, ex in enumerate(batch):
            for text in per_q[j]:
                f.write(json.dumps({
                    "split": split, "question": ex["question"], "gt": ex["gt"],
                    "text": text,
                    "correct": is_correct(extract_pred(text), ex["gt"]),
                }) + "\n")
        f.flush()  # survive an OOM or disconnect on the next iteration

        done = min(i + BATCH_Q, len(questions))
        if done % (BATCH_Q * 20) == 0 or done == len(questions):
            print(f"  {split} {done}/{len(questions)}", flush=True)


def run_generate_train() -> None:
    """Phase generate_train: generate train traces; creates / overwrites the traces file."""
    gen_model, gen_tok, gen_batch = _load_generator()
    print(f"[generate_train] GSM8K train: {N_TRAIN_Q} questions × {CPQ_TRAIN} completions",
          flush=True)
    train_qs = select_subset(load_gsm8k("train"), N_TRAIN_Q, seed)
    with open(traces_path, "w") as f:
        _write_split("train", train_qs, CPQ_TRAIN, f, gen_batch)
    print(f"[generate_train] Train traces → {traces_path}", flush=True)
    del gen_model, gen_tok
    torch.cuda.empty_cache()


def run_generate_eval() -> None:
    """Phase generate_eval: generate eval traces; appends to an existing traces file."""
    gen_model, gen_tok, gen_batch = _load_generator()
    print(f"[generate_eval] GSM8K test: {N_EVAL} questions × {N_EVAL_COMPLETIONS} completions",
          flush=True)
    eval_qs = select_subset(load_gsm8k("test"), N_EVAL, seed)
    # Append when train traces already exist in the file, create otherwise
    mode = "a" if traces_path.exists() else "w"
    with open(traces_path, mode) as f:
        _write_split("eval", eval_qs, N_EVAL_COMPLETIONS, f, gen_batch)
    print(f"[generate_eval] Eval traces → {traces_path}", flush=True)
    del gen_model, gen_tok
    torch.cuda.empty_cache()


def run_generate() -> None:
    """Phase generate: generate train traces then eval traces in one go."""
    run_generate_train()
    run_generate_eval()


# ---------------------------------------------------------------------------
# Phase 2 & 3: Train reward models
# ---------------------------------------------------------------------------
class RMDataset(Dataset):
    """(text, label) pairs; label = final-answer correctness (0 or 1)."""
    def __init__(self, records: list[dict], rm_type: str):
        self.items: list[tuple[str, float]] = []
        for rec in records:
            label = float(rec["correct"])
            if rm_type == "orm":
                self.items.append((format_rm_input(rec["question"], rec["text"]), label))
            else:
                steps = split_steps(rec["text"])
                for k in range(1, len(steps) + 1):
                    prefix = "\n\n".join(steps[:k])
                    self.items.append((format_rm_input(rec["question"], prefix), label))

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


def make_collate(tok, max_len: int):
    def collate(batch):
        texts  = [x[0] for x in batch]
        labels = torch.tensor([x[1] for x in batch])
        enc = tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        return {**enc, "labels": labels}
    return collate


def train_rm(rm_type: str, traces: list[dict], save_path: Path, train_cfg: dict) -> None:
    print(f"[train_{rm_type}] Loading backbone: {RM_MODEL_ID}", flush=True)
    tok = AutoTokenizer.from_pretrained(RM_MODEL_ID)
    tok.padding_side = "right"           # sequence classification reads the last real token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        RM_MODEL_ID, num_labels=1, torch_dtype=dtype, device_map="auto",
    )
    model.config.pad_token_id = tok.pad_token_id
    model.train()
    dev = model.device

    max_len = train_cfg["max_seq_len"]
    dataset = RMDataset(traces, rm_type)
    loader  = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True,
                         collate_fn=make_collate(tok, max_len), num_workers=0)
    opt     = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    n_items = len(dataset)
    n_pos   = sum(1 for _, lab in dataset.items if lab > 0.5)
    print(f"[train_{rm_type}] {n_items} items  ({n_pos} pos / {n_items - n_pos} neg)  "
          f"{len(loader)} batches × {train_cfg['epochs']} epochs", flush=True)

    for epoch in range(1, train_cfg["epochs"] + 1):
        total_loss, n_correct, n_total = 0.0, 0, 0
        t0 = time.time()
        for batch in loader:
            ids    = batch["input_ids"].to(dev)
            attn   = batch["attention_mask"].to(dev)
            labels = batch["labels"].to(dev).float()
            logits = model(input_ids=ids, attention_mask=attn).logits.squeeze(-1)
            loss   = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()
            n_correct  += ((torch.sigmoid(logits) > 0.5) == labels.bool()).sum().item()
            n_total    += len(labels)
        print(f"  epoch {epoch}/{train_cfg['epochs']}  "
              f"loss {total_loss / len(loader):.4f}  "
              f"acc {n_correct / n_total * 100:.1f}%  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tok.save_pretrained(save_path)
    print(f"[train_{rm_type}] Saved → {save_path}", flush=True)
    del model, tok
    torch.cuda.empty_cache()


def run_train_orm(traces: list[dict]) -> None:
    train_rm("orm", [t for t in traces if t["split"] == "train"], orm_dir_path, orm_cfg)


def run_train_prm(traces: list[dict]) -> None:
    train_rm("prm", [t for t in traces if t["split"] == "train"], prm_dir_path, prm_cfg)


# ---------------------------------------------------------------------------
# Phase 4: Evaluate best-of-n
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_orm(model, tok, questions: list[str], solutions: list[str], max_len: int) -> list[float]:
    texts  = [format_rm_input(q, s) for q, s in zip(questions, solutions)]
    enc    = tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    logits = model(**enc.to(model.device)).logits.squeeze(-1)
    return torch.sigmoid(logits).tolist()


@torch.no_grad()
def score_prm_question(model, tok, question: str, solutions: list[str], max_len: int) -> list[float]:
    """Score every solution for one question; returns one score (min over steps) per solution."""
    all_texts: list[str] = []
    ranges: list[tuple[int, int]] = []          # (start, end) in all_texts for each solution
    for sol in solutions:
        steps = split_steps(sol)
        start = len(all_texts)
        for k in range(1, len(steps) + 1):
            all_texts.append(format_rm_input(question, "\n\n".join(steps[:k])))
        ranges.append((start, len(all_texts)))

    # Batch inference over all step prefixes at once
    all_logits: list[float] = []
    BATCH = 16
    for i in range(0, len(all_texts), BATCH):
        enc    = tok(all_texts[i:i + BATCH], truncation=True, max_length=max_len,
                     padding=True, return_tensors="pt")
        logits = model(**enc.to(model.device)).logits.squeeze(-1)
        all_logits.extend(torch.sigmoid(logits).tolist())

    return [min(all_logits[s:e]) for s, e in ranges]  # pessimistic: weakest step governs


def run_evaluate(traces: list[dict]) -> None:
    # Group eval completions by question
    by_q: dict[str, dict] = {}
    for t in traces:
        if t["split"] != "eval":
            continue
        q = t["question"]
        if q not in by_q:
            by_q[q] = {"gt": t["gt"], "solutions": []}
        by_q[q]["solutions"].append({"text": t["text"], "correct": t["correct"]})

    questions = list(by_q.keys())
    print(f"[evaluate] {len(questions)} questions, up to {N_EVAL_COMPLETIONS} completions each",
          flush=True)

    # --- ORM scoring ---
    print(f"[evaluate] Loading ORM: {orm_dir_path}", flush=True)
    orm_tok = AutoTokenizer.from_pretrained(orm_dir_path)
    orm_tok.padding_side = "right"
    if orm_tok.pad_token is None:
        orm_tok.pad_token = orm_tok.eos_token
    orm = AutoModelForSequenceClassification.from_pretrained(
        orm_dir_path, num_labels=1, torch_dtype=dtype, device_map="auto"
    )
    orm.eval()

    orm_scores: dict[str, list[float]] = {}
    BATCH_ORM = 32
    for i in range(0, len(questions), BATCH_ORM):
        batch_qs = questions[i:i + BATCH_ORM]
        q_list, s_list = [], []
        sol_counts = []
        for q in batch_qs:
            sols = by_q[q]["solutions"]
            q_list.extend([q] * len(sols))
            s_list.extend(s["text"] for s in sols)
            sol_counts.append(len(sols))
        scores_flat = score_orm(orm, orm_tok, q_list, s_list, orm_cfg["max_seq_len"])
        offset = 0
        for q, cnt in zip(batch_qs, sol_counts):
            orm_scores[q] = scores_flat[offset:offset + cnt]
            offset += cnt
        if (i // BATCH_ORM + 1) % 5 == 0:
            print(f"  ORM {min(i + BATCH_ORM, len(questions))}/{len(questions)}", flush=True)

    del orm, orm_tok
    torch.cuda.empty_cache()

    # --- PRM scoring ---
    print(f"[evaluate] Loading PRM: {prm_dir_path}", flush=True)
    prm_tok = AutoTokenizer.from_pretrained(prm_dir_path)
    prm_tok.padding_side = "right"
    if prm_tok.pad_token is None:
        prm_tok.pad_token = prm_tok.eos_token
    prm = AutoModelForSequenceClassification.from_pretrained(
        prm_dir_path, num_labels=1, torch_dtype=dtype, device_map="auto"
    )
    prm.eval()

    prm_scores: dict[str, list[float]] = {}
    for i, q in enumerate(questions):
        sols = by_q[q]["solutions"]
        prm_scores[q] = score_prm_question(prm, prm_tok, q, [s["text"] for s in sols],
                                           prm_cfg["max_seq_len"])
        if (i + 1) % 25 == 0:
            print(f"  PRM {i + 1}/{len(questions)}", flush=True)

    del prm, prm_tok
    torch.cuda.empty_cache()

    # --- Best-of-n accuracy ---
    rows = {"n": [], "k1_baseline": [], "orm": [], "prm": [], "oracle": []}
    per_q: dict[int, dict[str, list[bool]]] = {n: {"orm": [], "prm": [], "oracle": []} for n in BEST_OF_N}

    for n in BEST_OF_N:
        n_orm = n_prm = n_oracle = n_k1 = 0
        for q in questions:
            sols   = by_q[q]["solutions"][:n]
            correct = [s["correct"] for s in sols]
            n_k1    += correct[0]                          # single-sample baseline
            n_oracle += any(correct)
            best_orm = max(range(len(sols)), key=lambda i: orm_scores[q][i])
            best_prm = max(range(len(sols)), key=lambda i: prm_scores[q][i])
            n_orm += correct[best_orm]
            n_prm += correct[best_prm]
            per_q[n]["orm"].append(bool(correct[best_orm]))
            per_q[n]["prm"].append(bool(correct[best_prm]))
            per_q[n]["oracle"].append(any(correct))
        N_q = len(questions)
        rows["n"].append(n)
        rows["k1_baseline"].append(round(n_k1    / N_q * 100, 2))
        rows["orm"].append(         round(n_orm   / N_q * 100, 2))
        rows["prm"].append(         round(n_prm   / N_q * 100, 2))
        rows["oracle"].append(      round(n_oracle / N_q * 100, 2))

    # Statistical test: PRM vs ORM at the largest n
    n_ref = BEST_OF_N[-1]
    cmp   = paired_compare(per_q[n_ref]["orm"], per_q[n_ref]["prm"])

    results = {
        "best_of_n": rows,
        "paired_prm_vs_orm": {"n": n_ref, **cmp},
        "orm_model":  str(orm_dir_path),
        "prm_model":  str(prm_dir_path),
        "n_questions": len(questions),
        "n_completions": N_EVAL_COMPLETIONS,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[evaluate] Results → {results_path}", flush=True)

    print("\n── Best-of-n reranking accuracy (%) ───────────────────────────────")
    print(f"{'n':>4}  {'k=1 base':>9}  {'ORM':>8}  {'PRM':>8}  {'oracle':>8}")
    for i, n in enumerate(rows["n"]):
        print(f"{n:4d}  {rows['k1_baseline'][i]:9.1f}  {rows['orm'][i]:8.1f}  "
              f"{rows['prm'][i]:8.1f}  {rows['oracle'][i]:8.1f}")
    print(f"\nPRM vs ORM (n={n_ref}): {cmp['diff_pp']:+.2f} pp  McNemar p = {cmp['mcnemar_p']:.4f}")
    success = cmp["diff_pp"] > 0
    print(f"Success criterion {'✅ met' if success else '⚠ not met'}: "
          f"PRM {'>' if success else '<='} ORM at n={n_ref}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if args.phase == "generate_train":
    run_generate_train()
elif args.phase == "generate_eval":
    run_generate_eval()
elif args.phase in ("generate", "all"):
    run_generate()

traces: list[dict] | None = None
if args.phase in ("train_orm", "train_prm", "evaluate", "all"):
    if not traces_path.exists():
        raise FileNotFoundError(f"Traces file not found: {traces_path}")
    with open(traces_path) as f:
        traces = [json.loads(line) for line in f if line.strip()]
    n_train = sum(1 for t in traces if t["split"] == "train")
    n_eval  = sum(1 for t in traces if t["split"] == "eval")
    print(f"Loaded {len(traces)} traces ({n_train} train, {n_eval} eval) from {traces_path}",
          flush=True)

if args.phase in ("train_orm", "all"):
    run_train_orm(traces)

if args.phase in ("train_prm", "all"):
    run_train_prm(traces)

if args.phase in ("evaluate", "all"):
    run_evaluate(traces)
