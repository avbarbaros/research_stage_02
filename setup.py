"""
setup.py — Stage 2 environment setup
=====================================
Installs all required packages, configures the environment, validates
config.yaml, and verifies the GPU before any experiment is run.

Run once at the start of each Colab session:
  python setup.py
  python setup.py --hf_token YOUR_TOKEN   # if model requires authentication
  python setup.py --skip_model_check       # skip tokenizer connectivity test
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config",           default="config.yaml")
parser.add_argument("--hf_token",         default=None,
                    help="Hugging Face access token (or set HF_TOKEN env var)")
parser.add_argument("--hf_cache_dir",     default=None,
                    help="Override HF_HOME cache directory (useful to point at /content/drive)")
parser.add_argument("--skip_model_check", action="store_true",
                    help="Skip tokenizer download check (saves time if already cached)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RESET  = "\033[0m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg):  print(f"  {YELLOW}!{RESET}  {msg}")
def fail(msg):  print(f"  {RED}✗{RESET}  {msg}"); sys.exit(1)
def section(title): print(f"\n{BOLD}{title}{RESET}\n" + "-" * len(title))

def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)

# ---------------------------------------------------------------------------
# 1. Python version
# ---------------------------------------------------------------------------
section("1. Python version")
major, minor = sys.version_info[:2]
print(f"  Python {major}.{minor}  ({sys.executable})")
if (major, minor) < (3, 10):
    fail("Python 3.10+ required (type-hint syntax used in experiment scripts).")
ok(f"Python {major}.{minor} meets requirement.")

# ---------------------------------------------------------------------------
# 2. Install packages
# ---------------------------------------------------------------------------
section("2. Package installation")

PACKAGES = [
    # Core inference
    "torch>=2.2.0",
    "transformers>=4.47.0",
    "accelerate>=0.30.0",
    # Dataset access
    "datasets>=2.20.0",
    # Config parsing
    "pyyaml>=6.0",
    # Hugging Face Hub (model download, login)
    "huggingface_hub>=0.23.0",
    # Experiment tracking (used from E2.3 onward)
    "wandb>=0.17.0",
    # PEFT — LoRA/QLoRA for reward model fine-tuning (E2.4)
    "peft>=0.11.0",
    # TRL — reference for verifier training (E2.4); not used as a black box
    "trl>=0.9.0",
]

for pkg in PACKAGES:
    pkg_name = pkg.split(">=")[0].split("==")[0]
    try:
        result = run([sys.executable, "-m", "pip", "install", "-q", pkg])
        ok(f"{pkg}")
    except subprocess.CalledProcessError as e:
        fail(f"Failed to install {pkg}:\n{e.stderr}")

# ---------------------------------------------------------------------------
# 3. Verify torch + CUDA
# ---------------------------------------------------------------------------
section("3. GPU / CUDA verification")

import importlib
torch = importlib.import_module("torch")

print(f"  PyTorch  : {torch.__version__}")
print(f"  CUDA     : {torch.version.cuda}")

if not torch.cuda.is_available():
    warn("No CUDA device found. Experiments will run on CPU — expect very long runtimes.")
else:
    dev = torch.cuda.get_device_properties(0)
    vram_gb = dev.total_memory / 1e9
    print(f"  GPU      : {dev.name}")
    print(f"  VRAM     : {vram_gb:.1f} GB")

    if vram_gb < 12:
        warn(f"Only {vram_gb:.1f} GB VRAM detected. "
             "Qwen2.5-7B-Instruct needs ~14 GB in bfloat16. "
             "Consider setting dtype: float16 or using a smaller model.")
    else:
        ok(f"{dev.name}  ({vram_gb:.1f} GB) — sufficient for 7B in bfloat16.")

# ---------------------------------------------------------------------------
# 4. Hugging Face token
# ---------------------------------------------------------------------------
section("4. Hugging Face authentication")

hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")

if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    try:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        ok("Logged in to Hugging Face Hub.")
    except Exception as e:
        warn(f"HF login failed: {e}")
else:
    warn("No HF_TOKEN provided. This is fine for public models (Qwen2.5-7B-Instruct is public). "
         "Pass --hf_token YOUR_TOKEN if you need gated models.")

# ---------------------------------------------------------------------------
# 5. Cache directory
# ---------------------------------------------------------------------------
section("5. Cache / storage configuration")

if args.hf_cache_dir:
    cache_dir = Path(args.hf_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"]         = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_DATASETS_CACHE"]  = str(cache_dir / "datasets")
    ok(f"Cache redirected to {cache_dir}")
    print(f"     HF_HOME            = {os.environ['HF_HOME']}")
    print(f"     TRANSFORMERS_CACHE = {os.environ['TRANSFORMERS_CACHE']}")
    print(f"     HF_DATASETS_CACHE  = {os.environ['HF_DATASETS_CACHE']}")
else:
    hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface")
    ok(f"Using default cache: {hf_home}")
    warn("On Colab, the default cache is on the ephemeral disk (not Drive). "
         "Pass --hf_cache_dir /content/drive/MyDrive/hf_cache to persist across sessions.")

# ---------------------------------------------------------------------------
# 6. Validate config.yaml
# ---------------------------------------------------------------------------
section("6. config.yaml validation")

import yaml

config_path = Path(args.config)
if not config_path.exists():
    fail(f"{config_path} not found. Make sure you are running setup.py from the repo root.")

with open(config_path) as f:
    cfg = yaml.safe_load(f)

REQUIRED_TOP = ["common", "models", "e2_1", "e2_2", "e2_3", "e2_4"]
REQUIRED_COMMON = ["seed", "device", "dtype", "output_dir"]
REQUIRED_MODELS = ["main", "grpo_policy"]
FEW_SHOT_CONDITIONS = ["few_shot_cot", "few_shot_cot_wei"]

errors = []
for key in REQUIRED_TOP:
    if key not in cfg:
        errors.append(f"Missing top-level key: '{key}'")

if "common" in cfg:
    for key in REQUIRED_COMMON:
        if key not in cfg["common"]:
            errors.append(f"Missing common.{key}")

if "models" in cfg:
    for key in REQUIRED_MODELS:
        if key not in cfg["models"]:
            errors.append(f"Missing models.{key}")

if "e2_1" in cfg:
    e21 = cfg["e2_1"]

    REQUIRED_PROMPTS = ["direct", "zero_shot_cot", "few_shot_cot", "few_shot_cot_wei"]
    for p in REQUIRED_PROMPTS:
        if p not in e21.get("prompts", {}):
            errors.append(f"Missing e2_1.prompts.{p}")
        elif "system" not in e21["prompts"][p]:
            errors.append(f"Missing e2_1.prompts.{p}.system")

    for p in FEW_SHOT_CONDITIONS:
        pcfg = e21.get("prompts", {}).get(p, {})
        if "exemplars_file" not in pcfg:
            errors.append(f"Missing e2_1.prompts.{p}.exemplars_file")
        elif not Path(pcfg["exemplars_file"]).exists():
            errors.append(f"e2_1.prompts.{p}.exemplars_file '{pcfg['exemplars_file']}' not found")

if errors:
    print()
    for e in errors:
        print(f"  {RED}✗{RESET}  {e}")
    fail("config.yaml has errors — fix them before running experiments.")

ok(f"{config_path} is valid.")
print(f"     model        : {cfg['models']['main']}")
print(f"     grpo_policy  : {cfg['models']['grpo_policy']}")
print(f"     dtype        : {cfg['common']['dtype']}")
print(f"     output_dir   : {cfg['common']['output_dir']}")

# ---------------------------------------------------------------------------
# 7. Few-shot exemplars
# ---------------------------------------------------------------------------
section("7. Few-shot exemplars")

for cond in FEW_SHOT_CONDITIONS:
    few_shot_path = Path(cfg["e2_1"]["prompts"][cond]["exemplars_file"])
    with open(few_shot_path) as f:
        exemplars = [json.loads(line) for line in f if line.strip()]

    if len(exemplars) == 0:
        fail(f"{few_shot_path} is empty.")

    missing_keys = [i for i, ex in enumerate(exemplars)
                    if "question" not in ex or "answer" not in ex]
    if missing_keys:
        fail(f"Exemplars at lines {missing_keys} in {few_shot_path} "
             f"are missing 'question' or 'answer' keys.")

    ok(f"{cond}: {len(exemplars)} exemplars loaded from {few_shot_path}.")
    for i, ex in enumerate(exemplars):
        print(f"     [{i}] Q: {ex['question'][:70]}...")

# ---------------------------------------------------------------------------
# 8. Output directory
# ---------------------------------------------------------------------------
section("8. Output directory")

out_dir = Path(cfg["common"]["output_dir"])
out_dir.mkdir(exist_ok=True)
ok(f"Local output directory ready: {out_dir.resolve()}")

drive_out = cfg["common"].get("drive_output_dir", "")
if drive_out:
    drive_out_path = Path(drive_out)
    if not drive_out_path.exists():
        try:
            drive_out_path.mkdir(parents=True, exist_ok=True)
            ok(f"Drive output directory created: {drive_out_path}")
        except Exception as e:
            warn(f"Could not create Drive output dir '{drive_out_path}': {e}\n"
                 "     Make sure Google Drive is mounted at /content/drive before running setup.")
    else:
        ok(f"Drive output directory ready: {drive_out_path}")
else:
    warn("drive_output_dir is blank — results will only be saved locally.\n"
         "     Set common.drive_output_dir in config.yaml to persist results to Drive.")

# ---------------------------------------------------------------------------
# 9. Tokenizer connectivity check (optional)
# ---------------------------------------------------------------------------
section("9. Model tokenizer check")

if args.skip_model_check:
    warn("Skipped (--skip_model_check). Run without this flag to verify model access.")
else:
    model_id = cfg["models"]["main"]
    print(f"  Downloading tokenizer for {model_id} ...")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        sample = tok.apply_chat_template(
            [{"role": "user", "content": "What is 2+2?"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ok(f"Tokenizer loaded. Chat template renders {len(tok.encode(sample))} tokens "
           f"for a minimal prompt.")
        print(f"     vocab size : {tok.vocab_size:,}")
    except Exception as e:
        fail(f"Could not load tokenizer for '{model_id}': {e}\n"
             "Check your internet connection or HF_TOKEN.")

# ---------------------------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------------------------
section("Setup complete")

print("""
  All checks passed. You can now run:

    python e2_1_cot_from_scratch.py               # all four conditions
    python e2_1_cot_from_scratch.py --n_eval 20   # quick smoke test
    python e2_2_self_consistency.py --n_eval 5    # E2.2 smoke test

  Results will be written to:
    outputs/e2_1_results_<timestamp>.json
    outputs/e2_2_results_<timestamp>.json

  To persist model weights across Colab sessions, mount Google Drive and run:
    python setup.py --hf_cache_dir /content/drive/MyDrive/hf_cache
""")
