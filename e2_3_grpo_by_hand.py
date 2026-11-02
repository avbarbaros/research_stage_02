"""
E2.3 — GRPO by hand
===================
Stage 2, Experiment 2.3 · Multimodal Reasoning Thesis Roadmap

Papers: Shao et al. (2024), DeepSeekMath, arXiv:2402.03300 (GRPO)
        Liu et al. (2025), Understanding R1-Zero-Like Training, arXiv:2503.20783 (Dr. GRPO)

For each step: sample G completions for each of B prompts, score them with a
verifiable reward (exact answer match), compute group-relative advantages, and
take a clipped policy-gradient step with a KL penalty to the reference model.

  A_i      = r_i − mean(r_group)                [÷ std(r_group) in GRPO]
  loss_tok = −min(ρ·A_i, clip(ρ, 1±ε)·A_i) + β·KL_k3(π_ref ‖ π_θ)
  loss     = mean_i (1/|o_i|) Σ_t loss_tok      (GRPO)
           = (1/(N·L_max))  Σ_i Σ_t loss_tok    (Dr. GRPO, L_max = max_new_tokens)

The policy is Qwen2.5-1.5B-Instruct with a LoRA adapter; the reference model is
the same network with the adapter disabled. All settings: e2_3 in config.yaml.

Usage
  python e2_3_grpo_by_hand.py --variant grpo
  python e2_3_grpo_by_hand.py --variant dr_grpo
  python e2_3_grpo_by_hand.py --variant grpo --steps 5        # smoke test
"""

import json
import time
import random
import argparse
from datetime import datetime
from pathlib import Path

import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from reasoning_utils import extract_pred, is_correct

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
parser.add_argument("--variant", required=True, choices=["grpo", "dr_grpo"])
parser.add_argument("--steps", type=int, default=None, help="Override e2_3.grpo.steps")
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))
common, c3 = cfg["common"], cfg["e2_3"]
g, variant = c3["grpo"], c3["variants"][args.variant]
NORM_LEN, SCALE_STD = variant["normalize_by_length"], variant["scale_by_std"]
STEPS, B, G = args.steps or g["steps"], g["prompts_per_step"], g["group_size"]
L_MAX, BETA, EPS, MU = g["max_new_tokens"], g["kl_coef"], g["clip_eps"], g["ppo_epochs"]
seed = common["seed"]
random.seed(seed)
torch.manual_seed(seed)

# ---------------------------------------------------------------------------
# Task: generated arithmetic with an exactly verifiable answer
# ---------------------------------------------------------------------------
def make_problems(n: int, rng: random.Random, exclude: set) -> list[dict]:
    t, out = c3["task"], []
    while len(out) < n:
        nums = [rng.randint(1, t["max_operand"]) for _ in range(t["n_operands"])]
        ops = [rng.choice(t["ops"]) for _ in nums[1:]]
        expr = str(nums[0]) + "".join(f" {o} {x}" for o, x in zip(ops, nums[1:]))
        if expr not in exclude:
            exclude.add(expr)
            out.append({"question": f"Compute: {expr}", "answer": str(eval(expr))})  # ints and + - * only
    return out

seen: set = set()
eval_set = make_problems(c3["n_eval"], random.Random(seed + 1), seen)
train_set = make_problems(c3["n_train"], random.Random(seed), seen)   # disjoint from eval_set

def reward(text: str, answer: str) -> float:
    return float(is_correct(extract_pred(text), answer))

# ---------------------------------------------------------------------------
# Model: policy = base + LoRA, reference = base (adapter disabled)
# ---------------------------------------------------------------------------
model_id = cfg["models"]["grpo_policy"]
tok = AutoTokenizer.from_pretrained(model_id)
tok.padding_side = "left"                                  # prompts end where generation starts
dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[common["dtype"]]
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
if g["gradient_checkpointing"]:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
lc = c3["lora"]
model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                         target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=g["lr"], weight_decay=0.0)
dev = model.device
_eos = model.generation_config.eos_token_id
EOS = torch.tensor(sorted(set(_eos if isinstance(_eos, list) else [_eos]) | {tok.eos_token_id}), device=dev)
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

def render(q: str) -> str:
    msgs = [{"role": "system", "content": c3["prompts"]["system"]}, {"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout(problems: list[dict], n: int, sample: bool) -> dict:
    """n completions per problem. Rows are ordered problem-major: p0×n, p1×n, …"""
    model.eval()
    enc = tok([render(p["question"]) for p in problems], return_tensors="pt", padding=True).to(dev)
    out = model.generate(**enc, max_new_tokens=L_MAX, num_return_sequences=n, pad_token_id=PAD,
                         do_sample=sample, repetition_penalty=1.0,
                         **({"temperature": g["temperature"], "top_p": 1.0, "top_k": 0} if sample
                            else {"temperature": None, "top_p": None, "top_k": None}))
    P = enc["input_ids"].shape[1]
    comp = out[:, P:]
    is_eos = torch.isin(comp, EOS)
    has_eos = is_eos.any(1)
    lengths = torch.where(has_eos, is_eos.int().argmax(1) + 1, comp.shape[1])   # incl. the EOS token
    cmask = torch.arange(comp.shape[1], device=dev)[None] < lengths[:, None]
    texts = [tok.decode(c[:l], skip_special_tokens=True) for c, l in zip(comp, lengths.tolist())]
    answers = [p["answer"] for p in problems for _ in range(n)]
    return {"ids": out, "attn": torch.cat([enc["attention_mask"].repeat_interleave(n, 0), cmask.long()], 1),
            "cmask": cmask.float(), "P": P, "texts": texts, "lengths": lengths.float(),
            "truncated": ~has_eos, "rewards": torch.tensor([reward(t, a) for t, a in zip(texts, answers)], device=dev)}

def token_logps(ids: torch.Tensor, attn: torch.Tensor, P: int) -> torch.Tensor:
    """log π(o_t | q, o_<t) for every completion token. Positions skip left padding."""
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    logits = model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False).logits[:, P - 1:-1]
    return torch.log_softmax(logits.float(), -1).gather(-1, ids[:, P:, None]).squeeze(-1)

# ---------------------------------------------------------------------------
# GRPO step
# ---------------------------------------------------------------------------
def grpo_step(batch: list[dict]) -> dict:
    ro = rollout(batch, G, sample=True)
    r = ro["rewards"].view(B, G)
    A = r - r.mean(1, keepdim=True)
    if SCALE_STD:
        A = A / (r.std(1, keepdim=True) + 1e-4)
    A = A.reshape(-1)
    N, mb = B * G, g["micro_batch_size"]
    chunks = [slice(i, i + mb) for i in range(0, N, mb)]
    with torch.no_grad():                                  # reference / old-policy log-probs
        ref = None
        if BETA:
            with model.disable_adapter():
                ref = torch.cat([token_logps(ro["ids"][s], ro["attn"][s], ro["P"]) for s in chunks])
        old = torch.cat([token_logps(ro["ids"][s], ro["attn"][s], ro["P"]) for s in chunks]) if MU > 1 else None
    model.train()
    stats = {"loss": 0.0, "kl": 0.0}
    for _ in range(MU):
        opt.zero_grad()
        for s in chunks:
            logp, m = token_logps(ro["ids"][s], ro["attn"][s], ro["P"]), ro["cmask"][s]
            ratio = torch.exp(logp - (old[s] if old is not None else logp.detach()))
            a = A[s, None]
            per_tok = -torch.min(ratio * a, ratio.clamp(1 - EPS, 1 + EPS) * a)
            if BETA:
                d = ref[s] - logp
                kl = torch.exp(d) - d - 1                  # k3 estimator, ≥ 0
                per_tok = per_tok + BETA * kl
                stats["kl"] += ((kl * m).sum() / m.sum()).item() / len(chunks) / MU
            if NORM_LEN:   # GRPO: average within each response → divide by |o_i|
                loss = ((per_tok * m).sum(1) / m.sum(1)).sum() / N
            else:          # Dr. GRPO: one constant for every token
                loss = (per_tok * m).sum() / (N * L_MAX)
            loss.backward()
            stats["loss"] += loss.item() / MU
        stats["grad_norm"] = torch.nn.utils.clip_grad_norm_(model.parameters(), g["max_grad_norm"]).item()
        opt.step()
    return {**stats, **summarise(ro), "zero_signal_groups": (r.std(1) == 0).float().mean().item()}

def summarise(ro: dict) -> dict:
    """Accuracy and response lengths (tokens), split by correct / incorrect."""
    ok, ln = ro["rewards"].bool(), ro["lengths"]
    mean = lambda x: x.mean().item() if x.numel() else None
    return {"reward": ro["rewards"].mean().item(), "len": ln.mean().item(), "len_correct": mean(ln[ok]),
            "len_incorrect": mean(ln[~ok]), "truncated": ro["truncated"].float().mean().item()}

def evaluate() -> dict:
    bs, parts = g["eval_batch_size"], []
    for i in range(0, len(eval_set), bs):
        parts.append(rollout(eval_set[i:i + bs], 1, sample=False))
    merged = {k: torch.cat([p[k] for p in parts]) for k in ("rewards", "lengths", "truncated")}
    return {f"eval_{k}": v for k, v in summarise(merged).items()}

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
out_dir = Path(common["output_dir"]); out_dir.mkdir(exist_ok=True)
drive = Path(common["drive_output_dir"]) if common.get("drive_output_dir") else None
if drive:
    drive.mkdir(parents=True, exist_ok=True)
stem = f"e2_3_{args.variant}_{datetime.now().strftime('%Y%m%d_%H%M')}"
log_path = (drive or out_dir) / f"{stem}_log.jsonl"         # on Drive: survives a disconnect
header = {"experiment": "E2.3", "variant": args.variant, "model": model_id, "seed": seed,
          "variant_settings": variant, "grpo": {**g, "steps": STEPS}, "task": c3["task"], "lora": lc}
with open(log_path, "w") as f:
    f.write(json.dumps({"_header": header}) + "\n")

def log(row: dict) -> None:
    with open(log_path, "a") as f:
        f.write(json.dumps(row) + "\n")

order = list(range(len(train_set)))
random.Random(seed).shuffle(order)
ev0 = evaluate()
log({"step": 0, **ev0})
print(f"[{args.variant}] step 0  eval acc {ev0['eval_reward']:.3f}  len {ev0['eval_len']:.1f}", flush=True)
t0 = time.time()
for step in range(1, STEPS + 1):
    batch = [train_set[order[((step - 1) * B + j) % len(order)]] for j in range(B)]   # cycles epochs
    ts = time.time()
    row = {"step": step, **grpo_step(batch), "sec": round(time.time() - ts, 1)}
    if step % g["eval_every"] == 0 or step == STEPS:
        row.update(evaluate())
    log(row)
    lc_, li = row["len_correct"], row["len_incorrect"]
    print(f"[{args.variant}] step {step:4d}/{STEPS}  reward {row['reward']:.3f}  "
          f"len ✓ {lc_ or 0:5.1f} ✗ {li or 0:5.1f}  kl {row['kl']:.4f}  "
          f"zero-signal {row['zero_signal_groups']:.2f}  {row['sec']:.0f}s"
          + (f"  | eval acc {row['eval_reward']:.3f} len {row['eval_len']:.1f}" if "eval_reward" in row else ""),
          flush=True)

adapter_dir = (drive or out_dir) / f"{stem}_adapter"
model.save_pretrained(adapter_dir)
print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Log → {log_path}  Adapter → {adapter_dir}")
