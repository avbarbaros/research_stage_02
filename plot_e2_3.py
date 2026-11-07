"""
Plot E2.3: GRPO vs. Dr. GRPO from two training logs.

Writes next to the GRPO log:
  e2_3_compare_<ts>.png / .pdf   six small multiples over training steps
      (a) training reward (rolling mean)      (b) held-out accuracy (greedy)
      (c) length of INCORRECT responses       (d) length of CORRECT responses
      (e) KL divergence (k3 estimator)        (f) policy gradient loss
  e2_3_compare_<ts>.md           summary table: start vs. end of training

Epoch boundaries are drawn as vertical dashed lines on every panel.
A secondary x-axis showing epoch count is added to the bottom row.

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
EPOCH_COLOR = "#b0aeaa"


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


def find_epoch_boundaries(rows: list[dict]) -> list[tuple[int, int]]:
    """Return (step, epoch_int) pairs where a new integer epoch is first reached."""
    seen: set[int] = set()
    boundaries = []
    for r in rows:
        ep = r.get("epoch")
        if ep is not None:
            ep_int = int(ep)
            if ep_int > 0 and ep_int not in seen:
                seen.add(ep_int)
                boundaries.append((r["step"], ep_int))
    return boundaries


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


def add_epoch_markers(ax, boundaries: list[tuple[int, int]], show_labels: bool = True) -> None:
    """Draw vertical dashed lines and optionally epoch labels at boundaries."""
    ymin, ymax = ax.get_ylim()
    label_y = ymin + 0.96 * (ymax - ymin)
    for step, ep_int in boundaries:
        ax.axvline(x=step, color=EPOCH_COLOR, linestyle="--", linewidth=0.9, zorder=0)
        if show_labels:
            ax.text(step, label_y, f"E{ep_int}", ha="center", va="top",
                    fontsize=7.5, color=EPOCH_COLOR)


def add_epoch_xaxis(ax, boundaries: list[tuple[int, int]], max_step: int) -> None:
    """Add a secondary x-axis at the top of ax showing epoch numbers."""
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks = [s for s, _ in boundaries if s <= max_step]
    labels = [f"E{e}" for _, e in boundaries if _ <= max_step]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels, fontsize=8, color=EPOCH_COLOR)
    ax2.tick_params(colors=EPOCH_COLOR, length=3)
    for side in ("top", "right", "left", "bottom"):
        ax2.spines[side].set_visible(False)
    ax2.set_xlabel("Epoch", fontsize=8.5, color=EPOCH_COLOR)


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:3]] if len(sys.argv) >= 3 else find_logs()
    runs = {}
    for p in paths:
        header, rows = load(p)
        runs[header["variant"]] = (header, rows, p)
    if set(runs) != {"grpo", "dr_grpo"}:
        raise SystemExit(f"Need one grpo and one dr_grpo log, got {sorted(runs)}")

    # Epoch boundaries from the GRPO run (same schedule for both variants).
    epoch_boundaries = find_epoch_boundaries(runs["grpo"][1])
    all_steps = [r["step"] for r in runs["grpo"][1]]
    max_step = max(all_steps) if all_steps else 0

    panels = [
        ("reward",       "(a) Training reward (rolling mean)",    "Reward",   True,  False),
        ("eval_reward",  "(b) Held-out accuracy (greedy)",        "Accuracy", False, False),
        ("len_incorrect","(c) Length of incorrect responses",     "Tokens",   True,  False),
        ("len_correct",  "(d) Length of correct responses",       "Tokens",   True,  False),
        ("kl",           "(e) KL divergence (k3 estimator)",      "KL",       True,  True),
        ("loss",         "(f) Policy gradient loss",              "Loss",     True,  True),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), layout="constrained")
    fig.patch.set_facecolor("white")

    for ax, (key, title, ylabel, smooth, is_bottom) in zip(axes.flat, panels):
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

        # Epoch markers (draw after data so ylim is set).
        add_epoch_markers(ax, epoch_boundaries, show_labels=True)

        # Secondary epoch x-axis on bottom-row panels.
        if is_bottom and epoch_boundaries:
            add_epoch_xaxis(ax, epoch_boundaries, max_step)

        # Selective direct labels: final value only, nudged apart when close.
        lo, hi = ax.get_ylim()
        close = len(ends) == 2 and abs(ends["grpo"][1] - ends["dr_grpo"][1]) < 0.05 * (hi - lo)
        for v, (x, y) in ends.items():
            other = [e[1] for w, e in ends.items() if w != v]
            higher = not other or y > other[0] or (y == other[0] and v == "grpo")
            dy = (6 if higher else -6) if close else 0
            fmt_v = f"{y:.4f}" if key == "kl" else (f"{y:.2f}" if key in ("reward", "eval_reward", "loss") else f"{y:.0f}")
            ax.annotate(fmt_v, xy=(x, y), xytext=(4, dy), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK)

    # x-axis label on the four non-bottom panels (bottom panels have epoch twin axis).
    for ax, (_, _, _, _, is_bottom) in zip(axes.flat, panels):
        ax.set_xlabel("Training step", fontsize=9.5, color=INK_2)

    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_2)
    h = runs["grpo"][0]
    n_train = h.get("n_train", "?")
    epochs_total = h.get("epochs_total", "?")
    fig.suptitle(
        f"E2.3 — GRPO vs. Dr. GRPO · {h['model']}, arithmetic task, "
        f"G={h['grpo']['group_size']}, β={h['grpo']['kl_coef']}, "
        f"B={h['grpo']['prompts_per_step']}, {n_train} train, {epochs_total} epochs",
        x=0.01, ha="left", fontsize=10.5, color=INK,
    )

    stem = runs["grpo"][2].parent / f"e2_3_compare_{datetime.now().strftime('%Y%m%d_%H%M')}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
        print(f"Saved → {stem}.{ext}")

    # ---- summary table: first vs. last EDGE training steps --------------
    metrics = [("reward",             "Training reward"),
               ("len_incorrect",      "Length, incorrect (tokens)"),
               ("len_correct",        "Length, correct (tokens)"),
               ("truncated",          "Truncated share"),
               ("zero_signal_groups", "Zero-signal groups"),
               ("kl",                 "KL to reference"),
               ("loss",               "Policy gradient loss")]
    fmt = lambda x: "—" if x is None else (f"{x:.4f}" if x != 0 and abs(x) < 0.01 else
                                            (f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"))
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
