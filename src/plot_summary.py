"""Summary figures for the multi-step instruction results.

  system_diagram.png     how the fast and slow paths share one frozen Qwen
  time_per_instruction.png   seconds per instruction for each setup (log scale)
  executor_cost.png      executor success (oracle plans) vs. executor compute, 1.7B depths and 0.6B
  step_types.png         planner accuracy by step type: reasoning vs. bag-of-words vs. direct

Reads outputs/compute_cost.json and outputs/chain_planner/*.json.
Usage: python src/plot_summary.py [--figures figures]
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BLUE, ORANGE, GRAY, LIGHT_GRAY = "#2a78d6", "#eb6834", "#8a8984", "#c9c8c2"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 1. System diagram
# ---------------------------------------------------------------------------

def box(ax, x, y, w, h, title, sub="", fc="white", ec=INK2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.4))
    ty = y + h / 2 + (0.13 if sub else 0)
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=10.5, color=INK,
            fontweight="bold" if bold else "normal")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", fontsize=8.5, color=INK2)


def arrow(ax, a, b, color=INK2, style="-|>", rad=0.0, lw=1.5):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def path_arrow(ax, pts, color, lw=1.6):
    """Polyline through pts with an arrowhead at the end (for loops routed around boxes)."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw)


def system_diagram(cost, out):
    fast_ms = cost["fast_path"]["16"]["ms"]
    think = cost["planner_reasoning"]
    ours = cost["setups"][0]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    # One frozen model, drawn as its layer stack.
    sx, sw = 4.3, 2.2
    ax.add_patch(FancyBboxPatch((sx, 0.9), sw, 2.1, boxstyle="round,pad=0.02,rounding_size=0.06", fc="#fdeee7", ec=ORANGE, lw=1.6))
    ax.add_patch(FancyBboxPatch((sx, 3.05), sw, 1.55, boxstyle="round,pad=0.02,rounding_size=0.06", fc="#e6f0fb", ec=BLUE, lw=1.6))
    ax.text(sx + sw / 2, 1.95, "layers 1–16", ha="center", va="center", fontsize=11, color=INK)
    ax.text(sx + sw / 2, 3.83, "layers 17–28", ha="center", va="center", fontsize=11, color=INK)
    ax.text(sx + sw / 2, 5.08, "One frozen Qwen3-1.7B", ha="center", fontsize=11.5, color=INK, fontweight="bold")
    ax.text(sx + sw / 2, 4.93, "(never fine-tuned)", ha="center", va="top", fontsize=8.5, color=INK2)

    # Inputs (left), both entering at layer 1.
    box(ax, 0.3, 3.4, 2.9, 1.0, "Planner prompt", "instruction, markers, steps done")
    box(ax, 0.3, 1.2, 2.9, 1.0, "Move prompt", "hand, gripper, goal = current place")
    arrow(ax, (3.22, 3.9), (sx - 0.02, 1.25), color=BLUE)
    arrow(ax, (3.22, 1.7), (sx - 0.02, 1.15), color=ORANGE)

    # Slow path exits at layer 28; fast path branches off at layer 16.
    box(ax, 7.3, 3.4, 1.9, 1.0, "Reasons", f"~{think['new_tokens_mean']:,} tokens")
    box(ax, 9.7, 3.4, 1.5, 1.0, "Next place", "e.g. “yellow”")
    arrow(ax, (sx + sw + 0.02, 4.3), (7.28, 3.9), color=BLUE)
    arrow(ax, (9.22, 3.9), (9.68, 3.9), color=INK2)
    box(ax, 7.3, 1.2, 1.9, 1.0, "Small head", "1 of 54 moves")
    box(ax, 9.7, 1.2, 1.5, 1.0, "Robot arm", "Meta-World")
    arrow(ax, (sx + sw + 0.02, 2.85), (7.28, 1.7), color=ORANGE)
    arrow(ax, (9.22, 1.7), (9.68, 1.7), color=INK2)

    # Timing labels
    ax.text(9.55, 3.05, f"SLOW PATH  ·  {think['ms'] / 1000:.0f} s per call  ·  ~{ours['planner_calls']:.1f}× per instruction",
            ha="center", fontsize=9, color=BLUE, fontweight="bold")
    ax.text(9.55, 2.45, f"FAST PATH  ·  {fast_ms:.0f} ms per move  ·  ~{ours['env_steps']:.0f}× per instruction",
            ha="center", fontsize=9, color=ORANGE, fontweight="bold")

    # Loops around the outside: next place -> goal of the move prompt; arm -> new state.
    path_arrow(ax, [(10.45, 4.42), (10.45, 5.45), (0.12, 5.45), (0.12, 2.0), (0.28, 2.0)], BLUE)
    ax.text(5.4, 5.55, "the chosen place becomes the goal of every move until it is reached; then the planner is asked again",
            ha="center", fontsize=8.5, color=INK2)
    path_arrow(ax, [(10.45, 1.18), (10.45, 0.2), (1.75, 0.2), (1.75, 1.18)], INK2)
    ax.text(6.1, 0.3, "new robot state after each move", ha="center", fontsize=8.5, color=INK2)

    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# 2. Time per instruction
# ---------------------------------------------------------------------------

def fmt_time(sec):
    return f"{sec / 60:.0f} min" if sec >= 120 else f"{sec:.0f} s" if sec >= 2 else f"{sec:.1f} s"


def time_per_instruction(cost, out):
    s = {x["setup"].split("  [")[0]: x for x in cost["setups"]}
    rows = [  # label, setup key, color
        ("Reasoning before every move\n(no fast path)", "reasoning planner at every step", LIGHT_GRAY),
        ("Reasoning planner + full model\nas executor (layers 1–28)", "reasoning planner + full-depth executor (k=28)", GRAY),
        ("Reasoning planner + fast path\n(layers 1–16)  — ours", "reasoning planner + fast path (k=16)", BLUE),
        ("Direct-answer planner + fast path", "direct planner + fast path (k=16)", ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for i, (label, key, color) in enumerate(rows):
        r = s[key]
        ax.barh(i, r["seconds"], color=color, height=0.55)
        succ = "success not measured" if r["success"] is None else f"{r['success']:.0%} of instructions completed"
        ax.text(r["seconds"] * 1.15, i, f"{fmt_time(r['seconds'])}   ·   {succ}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.3, 60000)
    ax.set_xlabel("Seconds per instruction (log scale; one request at a time on one GPU)")
    clean(ax)
    ax.grid(axis="y", visible=False)
    ours = s["reasoning planner + fast path (k=16)"]
    every = s["reasoning planner at every step"]
    ax.set_title(f"The fast path makes reasoning affordable: {every['seconds'] / ours['seconds']:.0f}× less time "
                 f"than reasoning before every move", fontsize=10.5, loc="left", color=INK)
    fig.text(0.01, -0.09, f"Ours: {ours['planner_share_of_time']:.0%} of the time is the ~{ours['planner_calls']:.1f} "
             "reasoning calls; the fast path's moves take about 1 s.", fontsize=8.5, color=INK2)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# 3. Executor quality vs. cost
# ---------------------------------------------------------------------------

def executor_cost(cost, out):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for model, color, label in [("1.7B", BLUE, "Qwen3-1.7B, first k layers (one model)"),
                                ("0.6B", ORANGE, "Qwen3-0.6B, first k layers (separate small model)")]:
        pts = sorted((d["gflops"] / 1000, 100 * d["success"], d["k"]) for d in cost["depth"] if d["model"] == model)
        xs, ys, ks = zip(*pts)
        ax.plot(xs, ys, "-o", color=color, lw=2, ms=6, label=label, mec=SURFACE, mew=1.5)
        for x, y, k in pts:
            below = {("1.7B", 4), ("1.7B", 28), ("0.6B", 8)}
            if (model, k) in below | {("1.7B", 16), ("0.6B", 16), ("0.6B", 28)}:
                dy = -14 if (model, k) in below else 7
                ax.annotate(f"{k}", (x, y), xytext=(0, dy), textcoords="offset points", ha="center",
                            fontsize=8.5, color=INK2)
    ax.set_xlabel("Executor compute per instruction (TFLOPs)")
    ax.set_ylabel("Instructions completed with perfect plans (%)")
    ax.set_ylim(80, 100)
    ax.set_xlim(0, 26)
    clean(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Middle layers execute best; the full model is the costliest and worst.\n"
                 "A separate 0.6B executor matches at about a third of the compute.",
                 fontsize=10.5, loc="left", color=INK)
    fig.text(0.01, -0.03, "Numbers next to points = layers used. 200 test instructions per point (±~3 points); "
             "recovery data was collected with 1.7B layers 1–16 driving.", fontsize=8, color=INK2)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# 4. Accuracy by step type
# ---------------------------------------------------------------------------

STEP_TYPES = [("cond", "if … otherwise"), ("relative", "closest to the\nlast one touched"),
              ("back", "go back to\nstart / first"), ("direct", "plain color"), ("farthest", "farthest from\na color"),
              ("extreme", "leftmost /\nfrontmost ..."), ("nearest", "closest to\na color")]


def step_types(planner_dir, out):
    think = json.load(open(os.path.join(planner_dir, "think_Qwen3-1.7B.json")))["records"]
    keys = {(r["seed"], r["step"]) for r in think}  # reasoning ran on 200 of the 400 test chains
    series = [("Qwen3-1.7B, reasoning", think, BLUE),
              ("Bag-of-words MLP", json.load(open(os.path.join(planner_dir, "bow.json")))["records"], GRAY),
              ("Qwen3-1.7B, direct answer", json.load(open(os.path.join(planner_dir, "direct_Qwen3-1.7B.json")))["records"], ORANGE)]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    w = 0.26
    for j, (label, recs, color) in enumerate(series):
        acc, n = defaultdict(int), defaultdict(int)
        for r in recs:
            if (r["seed"], r["step"]) in keys:
                n[r["family"]] += 1
                acc[r["family"]] += r["answer"] == r["truth"]
        ys = [100 * acc[f] / n[f] for f, _ in STEP_TYPES]
        xs = [i + (j - 1) * (w + 0.02) for i in range(len(STEP_TYPES))]
        ax.bar(xs, ys, width=w, color=color, label=label)
        if j == 0:
            for x, y in zip(xs, ys):
                ax.text(x, y + 1.5, f"{y:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(STEP_TYPES)), [t for _, t in STEP_TYPES], fontsize=9)
    ax.set_ylabel("Steps planned correctly (%)")
    ax.set_ylim(0, 108)
    ax.axhline(20, color=INK2, lw=0.8, ls=":")
    ax.text(-0.5, 21.5, "random guess (1 in 5)", fontsize=8, color=INK2, ha="left", zorder=5,
            bbox=dict(fc=SURFACE, ec="none", pad=1.5))
    clean(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.text(0, 1.2, "Reasoning wins where a step needs logic or memory of earlier steps; bag-of-words edges it on plain distance",
            transform=ax.transAxes, fontsize=10.5, color=INK)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost", default="outputs/compute_cost.json")
    parser.add_argument("--planner", default="outputs/chain_planner")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()
    os.makedirs(args.figures, exist_ok=True)
    cost = json.load(open(args.cost))
    system_diagram(cost, os.path.join(args.figures, "system_diagram.png"))
    time_per_instruction(cost, os.path.join(args.figures, "time_per_instruction.png"))
    executor_cost(cost, os.path.join(args.figures, "executor_cost.png"))
    step_types(args.planner, os.path.join(args.figures, "step_types.png"))


if __name__ == "__main__":
    main()
