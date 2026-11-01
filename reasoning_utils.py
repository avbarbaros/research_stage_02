"""
Shared helpers for Stage 2 experiments (E2.1 – E2.4).

  - GSM8K subset selection (same seed → same questions in every experiment)
  - Answer extraction and correctness
  - Statistics: Wilson CI, exact McNemar, bootstrap CI

Stdlib only, so it imports quickly and is easy to test without a GPU.
"""

import re
import math
import random

# ---------------------------------------------------------------------------
# Dataset subset
# ---------------------------------------------------------------------------

def select_subset(examples: list, n_eval: int | None, seed: int) -> list:
    """
    Shuffle with `seed` and take the first `n_eval` examples (all if None).

    Uses the same RNG stream as `random.seed(seed); random.shuffle(...)`, so a
    run with n_eval=300 uses exactly the first 300 questions of a run with
    n_eval=500 and the same seed. That makes results pairable across experiments.
    """
    examples = list(examples)
    if n_eval is not None:
        random.Random(seed).shuffle(examples)
        examples = examples[:n_eval]
    return examples


# ---------------------------------------------------------------------------
# Answer extraction
#
# Documented pitfalls (recorded here for the Stage 2 gate-check):
#   1. Numbers with commas  (1,234)  → strip before float()
#   2. Numbers with units   ("42 apples") → regex doesn't require end-of-string
#   3. Multiple candidates  → trigger phrase wins; else last number in text
#   4. Negative / decimal   → pattern includes '-' and '.'
#   5. No number at all     → return None (counts as wrong, no crash)
#   6. Lone comma           → a number must start with a digit; the old pattern
#                             matched a bare "," and returned "" (fixed)
#   7. Multi-part answers   → "The answer is 4 blue and 6 red" yields 4, not the
#                             total; not handled, visible as minority votes in E2.2
#   8. Base models continue → after answering they invent new problems; handled by
#                             stop strings in the callers (e2_1.base_prompts)
# See docs/E2.1_cot_from_scratch.md §5 for examples.
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
# Must start with a digit: "[\d,]+" alone also matched a bare "," (e.g. in
# "\boxed{52}, done"), which became the empty string "" after comma removal.
_NUM_PAT = r"(-?\d[\d,]*(?:\.\d+)?)"


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


def extract_first_number(text: str) -> str | None:
    """
    First number in the text. Used for answer-only completions from a base model
    (Kojima et al.'s "answer cleansing"): the answer comes first, and anything the
    model writes after it is noise.
    """
    m = re.search(_NUM_PAT, text)
    return m.group(1).replace(",", "").strip() if m else None


def canonical(pred: str | None) -> str | None:
    """
    Canonical form for comparing / voting on answers: "18", "18.0", "018" → "18".
    Returns None for missing or unparseable predictions.
    """
    if pred is None:
        return None
    try:
        f = float(pred)
    except ValueError:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return str(int(f)) if f.is_integer() else str(f)


def is_correct(pred: str | None, gt: str) -> bool:
    if pred is None:
        return False
    try:
        return float(pred) == float(gt)
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Statistics
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


def paired_compare(correct_a: list[bool], correct_b: list[bool]) -> dict:
    """Paired comparison of two conditions evaluated on the same questions."""
    assert len(correct_a) == len(correct_b) and len(correct_a) > 0
    b = sum(a and not o for a, o in zip(correct_a, correct_b))
    c = sum(o and not a for a, o in zip(correct_a, correct_b))
    n = len(correct_a)
    return {
        "n":              n,
        "a_only_correct": b,
        "b_only_correct": c,
        "diff_pp":        round((sum(correct_b) - sum(correct_a)) / n * 100, 2),  # B − A
        "mcnemar_p":      round(mcnemar_exact(b, c), 4),
    }


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """
    Percentile bootstrap 95% CI of the mean, resampling questions. In percent.
    `values` are per-question scores in [0, 1] (can be fractional, e.g. an
    expected accuracy averaged over sample subsets).
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    lo = means[int(0.025 * n_boot)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return (round(lo * 100, 2), round(hi * 100, 2))


# ---------------------------------------------------------------------------
# Self-tests (no GPU needed). Scripts call this at startup.
# ---------------------------------------------------------------------------

def run_self_tests() -> None:
    assert extract_gt("steps\n#### 42") == "42"
    assert extract_gt("steps\n#### 1,234") == "1234"
    assert extract_pred("The answer is 42 apples.") == "42"
    assert extract_pred("Therefore, the answer is -5.") == "-5"
    assert extract_pred("She earns $315.") == "315"
    assert extract_pred("No numbers here") is None
    assert extract_pred("The result is \\boxed{52}, done.") == "52"     # lone comma is not a number
    assert extract_pred("Total: 1,234, as shown.") == "1234"
    assert extract_first_number(": 7 apples, then 9") == "7"

    assert canonical("18.0") == canonical("18") == canonical("018") == "18"
    assert canonical("2.50") == "2.5"
    assert canonical(None) is None and canonical("1.2.3") is None

    assert wilson_ci(17, 20) == (63.96, 94.76)
    assert mcnemar_exact(3, 0) == 0.25
    assert mcnemar_exact(0, 0) == 1.0
    assert bootstrap_ci([1.0] * 10) == (100.0, 100.0)

    a = select_subset(range(1319), 300, 42)
    b = select_subset(range(1319), 500, 42)
    assert a == b[:300]
    print("reasoning_utils self-tests passed.")


if __name__ == "__main__":
    run_self_tests()
