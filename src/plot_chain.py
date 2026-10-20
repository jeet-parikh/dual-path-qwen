"""Figures for multi-step instructions.

1. chain_success.png: success vs. number of instruction steps, one line per planner.
   Reads closed-loop results (outputs/chain_rollouts/*.json) or offline planner results
   (outputs/chain_planner/*.json, --offline).
2. chain_trajectory.png: top-down hand paths for one instruction, one panel per planner.

Usage: python src/plot_chain.py [--offline] [--panels think_Qwen3-1.7B__fast_k16 direct_Qwen3-1.7B__fast_k16]
"""

import argparse
import glob
import json
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLOR_HEX = {"red": "#d62728", "blue": "#1f77b4", "green": "#2ca02c", "yellow": "#e6b800"}
LABELS = {"keyword": "Keyword rule", "bow": "Bag-of-words MLP", "direct": "direct answer",
          "think": "step-by-step reasoning", "oracle": "Oracle plan"}
STYLE = {"keyword": ("#9e9e9e", ":"), "bow": ("#7f7f7f", "--"), "direct": ("#ff7f0e", "-"),
         "think": ("#1f77b4", "-"), "oracle": ("#000000", "-.")}


def kind(name):
    return name.split("_")[0]


def success_chart(files, offline, out):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    order = ["oracle", "think", "direct", "bow", "keyword"]
    entries = []
    for path in files:
        d = json.load(open(path))
        name = os.path.basename(path)[:-5]
        res = d["results"] if offline else d["summary"]
        curve = res["chain_accuracy_by_length"] if offline else res["success_by_length"]
        n = res["n_by_length"]
        entries.append((order.index(kind(name)) if kind(name) in order else 9, name, curve, n))
    for _, name, curve, n in sorted(entries):
        xs = sorted(int(k) for k in curve)
        ys = [100 * curve[str(x)] for x in xs]
        # 95% binomial interval as a light band
        err = [196 * np.sqrt(curve[str(x)] * (1 - curve[str(x)]) / n[str(x)]) for x in xs]
        color, ls = STYLE.get(kind(name), ("#555", "-"))
        label = LABELS.get(kind(name), name)
        if "Qwen" in name:
            label = f"{name.split('_')[1]}, {label}"
        ax.plot(xs, ys, ls, color=color, lw=2.2, marker="o", label=label)
        ax.fill_between(xs, np.array(ys) - err, np.array(ys) + err, color=color, alpha=0.08, lw=0)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Steps in the instruction")
    ax.set_ylabel("Instructions completed (%)" if not offline else "Instructions planned correctly (%)")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("Closed loop: planner + fast path" if not offline else "Offline: next-place accuracy, whole chain")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


def draw_panel(ax, r, title):
    for c, x, y in r["markers"]:
        ax.scatter(x, y, s=260, color=COLOR_HEX[c], edgecolor="black", zorder=3)
    sx, sy = r["start"][:2]
    ax.scatter(sx, sy, s=160, marker="s", color="white", edgecolor="black", zorder=3)
    ax.annotate("start", (sx, sy), xytext=(6, -12), textcoords="offset points", fontsize=8)
    path = np.array(r["path"])
    ax.plot(path[:, 0], path[:, 1], color="black", lw=1.6, alpha=0.8, zorder=2)
    # Label each step where the hand arrived (the next planner call, or the end of the path).
    # Steps that end at the same place share one label ("1, 3: yellow").
    arrivals = [c["t"] for c in r["calls"][1:]] + [len(path) - 1]
    groups = []  # [xy, answer, step numbers]
    for i, (call, t) in enumerate(zip(r["calls"], arrivals)):
        xy = path[t, :2]
        for g in groups:
            if g[1] == call["answer"] and np.linalg.norm(g[0] - xy) < 0.06:
                g[2].append(i + 1)
                break
        else:
            groups.append([xy, call["answer"], [i + 1]])
    for xy, answer, steps in groups:
        ax.scatter(*xy, s=60, marker="*", color="black", zorder=4)
        ax.annotate(f"{', '.join(map(str, steps))}: {answer}", xy, xytext=(6, 6), textcoords="offset points",
                    fontsize=8, fontweight="bold")
    verdict = "correct" if r["success"] else f"wrong (went to {', '.join(r['visited'])})"
    ax.set_title(f"{title}\n{verdict}", fontsize=10, color="#2ca02c" if r["success"] else "#d62728")
    ax.set_aspect("equal")
    ax.set_xlim(-0.32, 0.32)
    ax.set_ylim(0.44, 0.9)
    ax.set_xlabel("x (m, right →)")
    ax.set_ylabel("y (m, away from robot →)")
    ax.grid(alpha=0.2)


def trajectory_figure(paths, out, seed=None):
    runs = [json.load(open(p)) for p in paths]
    by_seed = [{r["seed"]: r for r in d["results"]} for d in runs]
    common = set.intersection(*(set(b) for b in by_seed))
    if seed is None:
        # Prefer a long instruction where the first run succeeds and the others fail.
        cands = [s for s in common if by_seed[0][s]["success"] and not any(b[s]["success"] for b in by_seed[1:])]
        cands = sorted(cands, key=lambda s: (-by_seed[0][s]["length"], s)) or sorted(common)
        seed = cands[0]
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 5.6))
    axes = np.atleast_1d(axes)
    for ax, p, b in zip(axes, paths, by_seed):
        name = os.path.basename(p)[:-5].split("__")[0]
        label = LABELS.get(kind(name), name)
        draw_panel(ax, b[seed], f"{name.split('_')[1]}, {label}" if "Qwen" in name else label)
    r = by_seed[0][seed]
    fig.suptitle("\n".join(textwrap.wrap(f"“{r['instruction']}”", 110)) + f"\ncorrect places: {', '.join(r['places'])}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"Saved {out} (seed {seed})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dir", default=None)
    parser.add_argument("--panels", nargs="*", default=None, help="rollout file names (no .json) for the trajectory figure")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()
    d = args.dir or ("outputs/chain_planner" if args.offline else "outputs/chain_rollouts")
    os.makedirs(args.figures, exist_ok=True)
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    success_chart(files, args.offline, os.path.join(args.figures, "chain_success" + ("_offline" if args.offline else "") + ".png"))
    if args.panels and not args.offline:
        trajectory_figure([os.path.join(d, p + ".json") for p in args.panels],
                          os.path.join(args.figures, "chain_trajectory.png"), args.seed)


if __name__ == "__main__":
    main()
