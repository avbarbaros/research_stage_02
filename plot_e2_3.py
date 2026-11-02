"""
Plot E2.3: GRPO vs. Dr. GRPO from two training logs.

Writes next to the GRPO log:
  e2_3_compare_<ts>.png / .pdf   four small multiples over training steps
      (a) training reward (rolling mean)      (b) held-out accuracy (greedy)
      (c) length of INCORRECT responses       (d) length of CORRECT responses
  e2_3_compare_<ts>.md           summary table: start vs. end of training

and prints the length-bias check: how much the length of incorrect responses
changes over training under each variant.

Usage:
  python plot_e2_3.py                                   # newest grpo + dr_grpo logs in outputs/
  python plot_e2_3.py outputs/e2_3_grpo_X_log.jsonl outputs/e2_3_dr_grpo_Y_log.jsonl
"""

import sys
import json
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same palette as plot_e2_2.py (validated: CVD ΔE 24.7, contrast ≥ 3:1 on white).
COLOR   = {"grpo": "#2a78d6", "dr_grpo": "#eb6834"}
LABEL   = {"grpo": "GRPO", "dr_grpo": "Dr. GRPO"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WINDOW  = 10        # rolling-mean window (steps) for noisy per-step metrics
EDGE    = 25        # steps averaged for "start" and "end" in the summary


def load(path: Path) -> tuple[dict, list[dict]]:
    header, rows = {}, []
    for line in path.read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            header = obj["_header"] if "_header" in obj else header
            if "_header" not in obj:
                rows.append(obj)
    return header, rows


def find_logs() -> list[Path]:
    logs = []
    for v in ("grpo", "dr_grpo"):
        found = sorted(Path("outputs").glob(f"e2_3_{v}_*_log.jsonl"))
        if not found:
            raise SystemExit(f"No outputs/e2_3_{v}_*_log.jsonl found; pass both log paths.")
        logs.append(found[-1])
    return logs


def series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    pts = [(r["step"], r[key]) for r in rows if r.get(key) is not None]
    return [s for s, _ in pts], [v for _, v in pts]


def rolling(xs: list[int], ys: list[float], w: int) -> tuple[list[int], list[float]]:
    return xs[w - 1:], [statistics.mean(ys[i - w + 1:i + 1]) for i in range(w - 1, len(ys))]


def edge_mean(rows: list[dict], key: str, end: bool) -> float | None:
    train = [r for r in rows if r["step"] > 0]
    part = train[-EDGE:] if end else train[:EDGE]
    vals = [r[key] for r in part if r.get(key) is not None]
    return statistics.mean(vals) if vals else None


def style(ax, title: str, ylabel: str) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK_2)


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:3]] if len(sys.argv) >= 3 else find_logs()
    runs = {}
    for p in paths:
        header, rows = load(p)
        runs[header["variant"]] = (header, rows, p)
    if set(runs) != {"grpo", "dr_grpo"}:
        raise SystemExit(f"Need one grpo and one dr_grpo log, got {sorted(runs)}")

    panels = [("reward", "(a) Training reward (rolling mean)", "Reward", True),
              ("eval_reward", "(b) Held-out accuracy (greedy)", "Accuracy", False),
              ("len_incorrect", "(c) Length of incorrect responses", "Tokens", True),
              ("len_correct", "(d) Length of correct responses", "Tokens", True)]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True, layout="constrained")
    fig.patch.set_facecolor("white")
    for ax, (key, title, ylabel, smooth) in zip(axes.flat, panels):
        style(ax, title, ylabel)
        ends = {}
        for v in ("grpo", "dr_grpo"):
            xs, ys = series(runs[v][1], key)
            if smooth and len(xs) >= WINDOW:
                xs, ys = rolling(xs, ys, WINDOW)
            ax.plot(xs, ys, color=COLOR[v], lw=2, label=LABEL[v],
                    marker=None if smooth else "o", ms=6, mec="white", mew=1.2)
            if ys:
                ends[v] = (xs[-1], ys[-1])
        # Selective direct labels: final value only, nudged apart when the two ends are close.
        lo, hi = ax.get_ylim()
        close = len(ends) == 2 and abs(ends["grpo"][1] - ends["dr_grpo"][1]) < 0.05 * (hi - lo)
        for v, (x, y) in ends.items():
            other = [e[1] for w, e in ends.items() if w != v]
            higher = not other or y > other[0] or (y == other[0] and v == "grpo")
            dy = (6 if higher else -6) if close else 0
            ax.annotate(f"{y:.2f}" if key in ("reward", "eval_reward") else f"{y:.0f}",
                        xy=(x, y), xytext=(4, dy), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK)
    for ax in axes[1]:
        ax.set_xlabel("Training step", fontsize=9.5, color=INK_2)
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_2)
    h = runs["grpo"][0]
    fig.suptitle(f"E2.3 — GRPO vs. Dr. GRPO on {h['model']}, arithmetic task, "
                 f"G={h['grpo']['group_size']}, β={h['grpo']['kl_coef']}",
                 x=0.01, ha="left", fontsize=11, color=INK)

    stem = runs["grpo"][2].parent / f"e2_3_compare_{datetime.now().strftime('%Y%m%d_%H%M')}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
        print(f"Saved → {stem}.{ext}")

    # ---- summary table: first vs. last EDGE training steps --------------
    metrics = [("reward", "Training reward"), ("len_incorrect", "Length, incorrect (tokens)"),
               ("len_correct", "Length, correct (tokens)"), ("truncated", "Truncated share"),
               ("zero_signal_groups", "Zero-signal groups"), ("kl", "KL to reference")]
    fmt = lambda x: "—" if x is None else (f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}")
    lines = [f"| Metric (mean of first / last {EDGE} steps) | GRPO start | GRPO end | Dr. GRPO start | Dr. GRPO end |",
             "|---|---:|---:|---:|---:|"]
    for key, name in metrics:
        vals = [edge_mean(runs[v][1], key, end) for v in ("grpo", "dr_grpo") for end in (False, True)]
        lines.append(f"| {name} | " + " | ".join(fmt(x) for x in vals) + " |")
    ev = {v: series(runs[v][1], "eval_reward")[1] for v in runs}
    lines.append(f"| Held-out accuracy (first / last eval) | {fmt(ev['grpo'][0])} | {fmt(ev['grpo'][-1])} | "
                 f"{fmt(ev['dr_grpo'][0])} | {fmt(ev['dr_grpo'][-1])} |")
    table = "\n".join(lines)
    Path(f"{stem}.md").write_text(table + "\n")
    print(f"Saved → {stem}.md\n\n{table}")

    # ---- the two success-criterion checks ------------------------------
    print("\nChecks")
    for v in ("grpo", "dr_grpo"):
        r0, r1 = edge_mean(runs[v][1], "reward", False), edge_mean(runs[v][1], "reward", True)
        print(f"  {'✓' if r1 > r0 else '✗'} {LABEL[v]}: training reward {r0:.3f} → {r1:.3f}")
    d = {}
    for v in ("grpo", "dr_grpo"):
        a, b = edge_mean(runs[v][1], "len_incorrect", False), edge_mean(runs[v][1], "len_incorrect", True)
        d[v] = None if a is None or b is None else b - a
        print(f"  {LABEL[v]}: incorrect-response length {fmt(a)} → {fmt(b)} tokens"
              + ("" if d[v] is None else f"  (Δ {d[v]:+.1f})"))
    if None not in d.values():
        grows = d["grpo"] > d["dr_grpo"]
        print(f"  {'✓' if grows else '✗'} length bias visible: incorrect responses grow more under GRPO "
              f"than under Dr. GRPO ({d['grpo']:+.1f} vs {d['dr_grpo']:+.1f} tokens)")


if __name__ == "__main__":
    main()
