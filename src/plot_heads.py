"""Plot head accuracy vs. layer from outputs/heads/<model>/results.json files.

Writes to <out>/: accuracy_vs_layer.png, accuracy_by_task.png, subtask_heatmap.png

Usage: python src/plot_heads.py [--models Qwen3-0.6B Qwen3-1.7B] [--heatmap-model Qwen3-1.7B] [--out figures]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Palette: categorical slots 1-2, neutral inks, and a one-hue sequential ramp (light -> dark blue).
MODEL_COLORS = ["#eb6834", "#2a78d6"]  # assigned in --models order: smaller model orange, larger blue
HEAD_STYLE = {"mlp": dict(ls="-", marker="o"), "linear": dict(ls="--", marker="s")}
INK, INK_2, REF, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e8e7e3", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
REFERENCES = [  # (baseline key, label)
    ("state_mlp", "MLP on full-precision state"),
    ("prompt_ceiling", "best possible from prompt text"),
    ("prompt_mlp", "MLP on the prompt's numbers"),
]
TASK_NAMES = {"reach-v3": "reach", "push-v3": "push", "pick-place-v3": "pick-place"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": REF, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
})


def load(model):
    with open(os.path.join("outputs/heads", model, "results.json")) as f:
        r = json.load(f)
    r["layers"] = {int(k): v for k, v in r["layers"].items()}
    return r


def lower_limit(results, baselines, get_line, get_ref):
    """Y-axis floor: 5 points below the lowest line or reference, rounded down to a multiple of 5."""
    vals = [100 * get_line(r["layers"][k][h]) for r in results.values() for k in r["layers"] for h in r["layers"][k]]
    vals += [100 * get_ref(baselines[key]) for key, _ in REFERENCES if key in baselines]
    return 5 * int((min(vals) - 5) // 5)


def plot_lines(ax, results, get, label_models=True):
    """Draw accuracy-vs-layer lines for every model x head. get(entry) -> accuracy in [0, 1]."""
    for color, (model, r) in zip(MODEL_COLORS, results.items()):
        layers = sorted(r["layers"])
        for head, style in HEAD_STYLE.items():
            if head not in r["layers"][layers[0]]:
                continue
            acc = [100 * get(r["layers"][k][head]) for k in layers]
            label = f"{model.replace('Qwen3-', 'Qwen3 ')} · {head}" if label_models else None
            ax.plot(layers, acc, color=color, label=label, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    zorder=3, **style)
        ax.set_xticks(layers)


def plot_references(ax, baselines, get, label=True):
    for key, text in REFERENCES:
        y = 100 * get(baselines[key])
        ax.axhline(y, color=REF, lw=1, ls=":", zorder=1)
        if label:
            ax.annotate(f"{text} ({y:.1f}%)", xy=(1, y), xycoords=("axes fraction", "data"),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK_2)


def fig_accuracy_vs_layer(results, out):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    base = next(iter(results.values()))["baselines"]
    plot_lines(ax, results, lambda e: e["overall"])
    plot_references(ax, base, lambda b: b["overall"])
    ax.set_ylim(lower_limit(results, base, lambda e: e["overall"], lambda b: b["overall"]), 100)
    ax.set_xlabel("Qwen layers used (of 28)")
    ax.set_ylabel("Test accuracy, 54 action classes (%)")
    ax.set_title("Action accuracy vs. Qwen depth", loc="left", fontsize=12, fontweight="bold",
                 pad=22)
    ax.text(0, 1.02, f"Held-out episodes, all three tasks. Off-scale: always guessing the most common class "
            f"{100 * base['majority']['overall']:.0f}%, per task {100 * base['task_majority']['overall']:.0f}%.",
            transform=ax.transAxes, fontsize=8.5, color=INK_2)
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "accuracy_vs_layer.png"), dpi=200)
    plt.close(fig)


def fig_accuracy_by_task(results, out):
    tasks = list(TASK_NAMES)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    base = next(iter(results.values()))["baselines"]
    for i, (ax, task) in enumerate(zip(axes, tasks)):
        plot_lines(ax, results, lambda e: e["per_task"][task], label_models=(i == 0))
        y = 100 * base["prompt_ceiling"]["per_task"][task]
        ax.axhline(y, color=REF, lw=1, ls=":", zorder=1)
        ax.text(0.02, y + 0.4, f"best possible from prompt ({y:.1f}%)", transform=ax.get_yaxis_transform(),
                fontsize=8.5, color=INK_2)
        ax.set_title(TASK_NAMES[task], loc="left", fontweight="bold")
        ax.set_xlabel("Qwen layers used")
    axes[0].set_ylim(min(lower_limit(results, {}, lambda e, t=t: e["per_task"][t], None) for t in tasks), 100)
    axes[0].set_ylabel("Test accuracy (%)")
    fig.legend(loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Accuracy vs. layer, per task",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(out, "accuracy_by_task.png"), dpi=200)
    plt.close(fig)


def fig_subtask_heatmap(r, head, out):
    layers = sorted(r["layers"])
    rows = list(r["layers"][layers[0]][head]["per_subtask"])
    ceiling = r["baselines"]["prompt_ceiling"]["per_subtask"]
    data = np.array([[100 * r["layers"][k][head]["per_subtask"][s] for k in layers] + [100 * ceiling[s]]
                     for s in rows])
    cols = [str(k) for k in layers] + ["prompt\nceiling"]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    vmin = 10 * int(data.min() // 10)
    ax.imshow(data, cmap=BLUES, vmin=vmin, vmax=100, aspect="auto")
    for (i, j), v in np.ndenumerate(data):
        dark = (v - vmin) / (100 - vmin) >= 0.6
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5, color=SURFACE if dark else INK)
    ax.axvline(len(layers) - 0.5, color=SURFACE, lw=4)  # gap before the ceiling column
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), [s.replace("-v3 | ", ": ") for s in rows])
    ax.set_xlabel("Qwen layers used")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"Test accuracy (%) by subtask · {r['model']} {head} head", loc="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "subtask_heatmap.png"), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["Qwen3-0.6B", "Qwen3-1.7B"], help="smallest first")
    parser.add_argument("--heatmap-model", default="Qwen3-1.7B")
    parser.add_argument("--heatmap-head", default="mlp")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = {m: load(m) for m in args.models}
    fig_accuracy_vs_layer(results, args.out)
    fig_accuracy_by_task(results, args.out)
    fig_subtask_heatmap(results[args.heatmap_model], args.heatmap_head, args.out)
    print(f"Saved figures to {args.out}/")


if __name__ == "__main__":
    main()
