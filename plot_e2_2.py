"""
Plot E2.2 self-consistency results.

Reads an e2_2_results_*.json file and writes:
  <stem>_scaling.png / .pdf   two-panel figure
      (a) accuracy vs. k: majority vote with 95% CI band, any-sample-correct
          ceiling, and the E2.1 greedy CoT baseline (if the run had greedy_ref)
      (b) accuracy by vote agreement at the largest k
  <stem>_table.md             the same numbers as a Markdown table

It also prints a few checks for "is this a clean scaling curve?".

Usage:
  python plot_e2_2.py                                  # newest outputs/e2_2_results_*.json
  python plot_e2_2.py outputs/e2_2_results_XXXX.json
"""

import sys
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colours: reference categorical palette, slots 1–2, plus neutral ink.
BLUE    = "#2a78d6"   # majority vote
ORANGE  = "#eb6834"   # any-correct ceiling
NEUTRAL = "#6b6a66"   # greedy baseline
INK     = "#0b0b0b"
INK_2   = "#52514e"
GRID    = "#e4e3df"


def load(path_arg: str | None) -> tuple[Path, dict]:
    if path_arg:
        path = Path(path_arg)
    else:
        files = sorted(Path("outputs").glob("e2_2_results_*.json"))
        if not files:
            raise SystemExit("No outputs/e2_2_results_*.json found; pass a path.")
        path = files[-1]
    return path, json.loads(path.read_text())


def style_axis(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    path, res = load(sys.argv[1] if len(sys.argv) > 1 else None)
    curve  = {int(k): v for k, v in res["summary"].items()}
    ks     = sorted(curve)
    vote   = [curve[k]["vote_accuracy"] for k in ks]
    lo     = [curve[k]["ci95_low"] for k in ks]
    hi     = [curve[k]["ci95_high"] for k in ks]
    anyc   = [curve[k]["any_correct_rate"] for k in ks]
    greedy = res.get("vs_greedy")
    run    = res.get("run_settings", {})
    n      = res["n_eval"]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [2.1, 1]},
        layout="constrained",
    )
    fig.patch.set_facecolor("white")

    # ---- (a) scaling curve -------------------------------------------------
    style_axis(ax)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.minorticks_off()

    ax.fill_between(ks, lo, hi, color=BLUE, alpha=0.15, linewidth=0, label="95% CI (bootstrap)")
    ax.plot(ks, anyc, color=ORANGE, lw=1.5, ls="--", marker="o", ms=5,
            mec="white", mew=1.2, label="Any sample correct (ceiling)")
    ax.plot(ks, vote, color=BLUE, lw=2, marker="o", ms=6,
            mec="white", mew=1.2, label="Majority vote")

    if greedy:
        g = greedy["greedy_accuracy"]
        ax.axhline(g, color=NEUTRAL, lw=1.2, ls=":", label="Greedy CoT (E2.1)")
        # Label sits between the first two k ticks, clear of the data points.
        ax.annotate(f"greedy {g:.1f}%", xy=(math.sqrt(ks[0] * ks[1]), g), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK_2)

    # Selective direct labels: vote at both ends, ceiling at the end only
    # (at k=1 the two lines coincide).
    for x, y, dy in ((ks[0], vote[0], -14), (ks[-1], vote[-1], -14), (ks[-1], anyc[-1], 7)):
        ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(0, dy),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)

    ymin = min(min(lo), greedy["greedy_accuracy"] if greedy else 100)
    ax.set_ylim(max(0, ymin - 4), 101)
    ax.set_xlim(ks[0] / 1.25, ks[-1] * 1.25)
    ax.set_xlabel("Samples per question (k)", fontsize=9.5, color=INK_2)
    ax.set_ylabel("Accuracy (%)", fontsize=9.5, color=INK_2)
    ax.set_title("(a) Accuracy vs. number of samples", loc="left", fontsize=10.5, color=INK)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK_2)

    # ---- (b) agreement bins ------------------------------------------------
    style_axis(ax2)
    bins   = res["agreement_bins"]
    labels = list(bins)
    accs   = [bins[b]["accuracy"] or 0 for b in labels]
    ns     = [bins[b]["n"] for b in labels]
    xs     = range(len(labels))
    ax2.bar(xs, accs, width=0.55, color=BLUE)
    for x, a, c in zip(xs, accs, ns):
        txt = f"{a:.0f}%\nn={c}" if c else "n=0"
        ax2.annotate(txt, xy=(x, a), xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, color=INK)
    ax2.set_xticks(list(xs), labels)
    ax2.set_ylim(0, 118)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_xlabel(f"Share of {ks[-1]} samples backing the winner", fontsize=9.5, color=INK_2)
    ax2.set_title("(b) Accuracy by vote agreement", loc="left", fontsize=10.5, color=INK)

    model = res.get("model", "")
    fig.suptitle(
        f"Self-consistency on GSM8K — {model}, {n} questions, "
        f"T={run.get('temperature', '?')}, top_p={run.get('top_p', '?')}",
        x=0.01, ha="left", fontsize=11, color=INK,
    )

    stem = path.with_suffix("")
    for ext in ("png", "pdf"):
        out = Path(f"{stem}_scaling.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved → {out}")

    # ---- Markdown table ----------------------------------------------------
    rows = ["| k | Majority vote | 95% CI | Any sample correct | Gain over k=1 |",
            "|---:|---:|---:|---:|---:|"]
    for k, v, l, h, a in zip(ks, vote, lo, hi, anyc):
        rows.append(f"| {k} | {v:.1f}% | [{l:.1f}, {h:.1f}] | {a:.1f}% | {v - vote[0]:+.1f} pp |")
    if greedy:
        rows.append("")
        rows.append(f"Greedy CoT (E2.1) on the same {greedy['n']} questions: "
                    f"{greedy['greedy_accuracy']:.1f}%. Self-consistency at k={ks[-1]}: "
                    f"{greedy['sc_accuracy']:.1f}% (Δ {greedy['diff_pp']:+.1f} pp, "
                    f"exact McNemar p = {greedy['mcnemar_p']:.4f}).")
    table = "\n".join(rows)
    Path(f"{stem}_table.md").write_text(table + "\n")
    print(f"Saved → {stem}_table.md\n\n{table}")

    # ---- "Clean curve" checks ---------------------------------------------
    print("\nCurve checks")
    mono = all(b >= a - 0.5 for a, b in zip(vote, vote[1:]))
    print(f"  {'✓' if mono else '✗'} vote accuracy never drops by more than 0.5 pp as k grows")
    sep = lo[-1] > hi[0]
    print(f"  {'✓' if sep else '✗'} CI at k={ks[-1]} lies fully above the CI at k=1")
    # Gain per doubling of k (k=1→4 is two doublings), so steps are comparable.
    per_dbl = [(b - a) / math.log2(k2 / k1)
               for a, b, k1, k2 in zip(vote, vote[1:], ks, ks[1:])]
    sat = len(per_dbl) >= 2 and per_dbl[-1] <= per_dbl[0]
    print(f"  {'✓' if sat else '✗'} gain per doubling of k shrinks (diminishing returns): "
          + ", ".join(f"{k1}→{k2}: {g:+.1f}" for g, k1, k2 in zip(per_dbl, ks, ks[1:])))
    if greedy:
        sig = greedy["mcnemar_p"] < 0.05 and greedy["diff_pp"] > 0
        print(f"  {'✓' if sig else '✗'} SC@{ks[-1]} beats greedy CoT at p < 0.05 "
              f"(p = {greedy['mcnemar_p']:.4f})")
    gap = anyc[-1] - vote[-1]
    print(f"  Selection gap at k={ks[-1]}: {gap:.1f} pp "
          f"(room for a better selector than voting, e.g. E2.4 verifiers)")


if __name__ == "__main__":
    main()
