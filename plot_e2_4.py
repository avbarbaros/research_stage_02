"""
Plot E2.4: best-of-n reranking accuracy for ORM, PRM, oracle and k=1 baseline.

Writes next to the results JSON:
  e2_4_results_<stem>_scaling.png / .pdf   single panel: accuracy vs. n
  e2_4_results_<stem>_table.md             markdown summary table

Usage:
  python plot_e2_4.py                               # newest e2_4_results_*.json in outputs/
  python plot_e2_4.py outputs/e2_4_results_STEM.json
"""

import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palette: same construction as plot_e2_2 / plot_e2_3 (CVD-safe, contrast ≥ 3:1 on white).
# baseline = neutral grey, oracle = soft green, ORM = blue, PRM = orange
COLOR = {
    "k1_baseline": "#a0a09a",
    "orm":         "#2a78d6",
    "prm":         "#eb6834",
    "oracle":      "#3baa5c",
}
LABEL = {
    "k1_baseline": "k = 1 (no reranking)",
    "orm":         "ORM",
    "prm":         "PRM",
    "oracle":      "Oracle (any correct)",
}
MARKER = {
    "k1_baseline": "none",
    "orm":         "o",
    "prm":         "s",
    "oracle":      "^",
}
LINE = {
    "k1_baseline": "--",
    "orm":         "-",
    "prm":         "-",
    "oracle":      ":",
}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def find_results() -> Path:
    cands = sorted(Path("outputs").glob("e2_4_results_*.json"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError("No e2_4_results_*.json found in outputs/")
    return cands[-1]


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot(data: dict, out_stem: Path) -> None:
    bon  = data["best_of_n"]
    ns   = bon["n"]
    keys = ["k1_baseline", "orm", "prm", "oracle"]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for key in keys:
        ys = bon[key]
        mk = MARKER[key]
        ax.plot(ns, ys,
                color=COLOR[key], linestyle=LINE[key],
                marker=mk if mk != "none" else None,
                markersize=6, linewidth=1.8,
                label=LABEL[key], zorder=3)
        # Annotate peak value for ORM and PRM
        if key in ("orm", "prm"):
            peak_i = ys.index(max(ys))
            ax.annotate(
                f"{ys[peak_i]:.1f}%",
                xy=(ns[peak_i], ys[peak_i]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7.5, color=COLOR[key],
            )

    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Best-of-n  (n)", fontsize=10, color=INK)
    ax.set_ylabel("Accuracy (%)", fontsize=10, color=INK)
    ax.set_title("E2.4 — Verifier study: best-of-n reranking on GSM8K", fontsize=10.5, color=INK)
    ax.tick_params(colors=INK_2, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Annotate n_questions
    n_q = data.get("n_questions", "?")
    ax.text(0.98, 0.04, f"n = {n_q} questions",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=INK_2)

    leg = ax.legend(fontsize=8.5, framealpha=0.95, edgecolor=GRID,
                    loc="upper left", handlelength=1.8)
    for txt in leg.get_texts():
        txt.set_color(INK)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved → {p}")
    plt.close(fig)


def table(data: dict, out_stem: Path) -> None:
    bon  = data["best_of_n"]
    ns   = bon["n"]
    cmp  = data.get("paired_prm_vs_orm", {})
    n_q  = data.get("n_questions", "?")

    lines = [
        "## E2.4 — Best-of-n reranking accuracy (%)",
        "",
        f"**{n_q} eval questions · 32 completions per question**",
        "",
        "| n | k=1 baseline | ORM | PRM | Oracle |",
        "|---:|---:|---:|---:|---:|",
    ]
    for i, n in enumerate(ns):
        mark = lambda v, col: f"**{v}**" if v == max(bon[col]) else str(v)
        lines.append(
            f"| {n} | {bon['k1_baseline'][i]} | {mark(bon['orm'][i],'orm')} | "
            f"{mark(bon['prm'][i],'prm')} | {bon['oracle'][i]} |"
        )

    lines += [
        "",
        "Bold = peak for that method.",
        "",
        "**Gains over k=1 baseline at peak:**",
        "",
        f"| Method | Peak n | Accuracy | Gain |",
        f"|---|---:|---:|---:|",
        f"| ORM | {ns[bon['orm'].index(max(bon['orm']))]} | {max(bon['orm']):.2f}% "
            f"| +{max(bon['orm']) - bon['k1_baseline'][0]:.2f} pp |",
        f"| PRM | {ns[bon['prm'].index(max(bon['prm']))]} | {max(bon['prm']):.2f}% "
            f"| +{max(bon['prm']) - bon['k1_baseline'][0]:.2f} pp |",
        f"| Oracle | {ns[bon['oracle'].index(max(bon['oracle']))]} | {max(bon['oracle']):.2f}% "
            f"| +{max(bon['oracle']) - bon['k1_baseline'][0]:.2f} pp |",
        "",
    ]

    if cmp:
        lines += [
            f"**PRM vs ORM at n = {cmp.get('n', '?')}:** "
            f"+{cmp.get('diff_pp', 0):.2f} pp  "
            f"(McNemar p = {cmp.get('mcnemar_p', '?')}) — "
            f"{cmp.get('b_only_correct', '?')} PRM-only vs "
            f"{cmp.get('a_only_correct', '?')} ORM-only correct",
        ]

    md_path = out_stem.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"  Saved → {md_path}")


def checks(data: dict) -> None:
    bon = data["best_of_n"]
    print("\n── Scaling-curve checks ──────────────────────────────────────────")

    # ORM: does it degrade?
    orm = bon["orm"]
    orm_peak = max(orm)
    orm_at_32 = orm[-1]
    if orm_at_32 < orm_peak:
        print(f"  ⚠  ORM degrades after n={bon['n'][orm.index(orm_peak)]}: "
              f"{orm_peak:.2f}% → {orm_at_32:.2f}% (overoptimisation)")
    else:
        print(f"  ✓  ORM does not degrade")

    # PRM: monotone up to peak?
    prm = bon["prm"]
    prm_diffs = [prm[i] - prm[i - 1] for i in range(1, len(prm))]
    if all(d >= 0 for d in prm_diffs):
        print(f"  ✓  PRM is non-decreasing (peak {max(prm):.2f}% at n={bon['n'][prm.index(max(prm))]})")
    else:
        first_drop = next(i for i, d in enumerate(prm_diffs) if d < 0)
        print(f"  ⚠  PRM drops at n={bon['n'][first_drop + 1]}")

    # PRM > ORM at n=32?
    prm_32, orm_32 = prm[-1], orm[-1]
    diff = prm_32 - orm_32
    cmp  = data.get("paired_prm_vs_orm", {})
    sig  = "✓ p < 0.05" if cmp.get("mcnemar_p", 1) < 0.05 else f"⚠  p = {cmp.get('mcnemar_p', '?')} (not significant)"
    sym  = ">" if diff > 0 else "<="
    print(f"  {'✓' if diff > 0 else '✗'}  PRM@32 ({prm_32:.2f}%) {sym} ORM@32 ({orm_32:.2f}%): "
          f"{diff:+.2f} pp  {sig}")

    # Oracle gap
    baseline = bon["k1_baseline"][0]
    oracle   = max(bon["oracle"])
    best_rm  = max(max(prm), max(orm))
    print(f"  ·  Oracle gap at peak: {oracle - best_rm:.2f} pp  "
          f"({best_rm:.2f}% best RM vs {oracle:.2f}% oracle)")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_results()
    print(f"Results: {path}")
    data     = load(path)
    out_stem = path.parent / (path.stem + "_scaling")
    plot(data, out_stem)
    table(data, out_stem)
    checks(data)
    print("\nDone.")


if __name__ == "__main__":
    main()
