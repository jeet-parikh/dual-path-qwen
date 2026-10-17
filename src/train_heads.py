"""Train one action head per layer on cached Qwen features, plus baselines; report test accuracy.

Split is by episode (70% train / 10% val / 20% test within each task, fixed seed), identical
for every layer and model. One shared head covers all three tasks. Each head keeps the epoch
with the best validation accuracy, so the test set is only used for the final number.

Outputs in <out>/: <head>_layer<k>.pt (trained heads, for rollouts) and results.json.

Baselines: most common class (overall / per task), and MLPs on numbers instead of Qwen features:
  state_mlp:  full-precision observation + task id (what a non-LLM policy could see)
  prompt_mlp: the exact rounded numbers shown in the prompt + task id (same information Qwen gets)
  prompt_ceiling: best possible accuracy from the prompt text (identical rounded prompts can have
                  different expert actions; this predicts each prompt's most common test label)

Usage: python src/train_heads.py --model Qwen3-1.7B [--layers 4 8 ...] [--epochs 200]
"""

import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import torch
from torch import nn

from common import NUM_CLASSES, class_to_parts

# Lookup table: class index -> (dx, dy, dz, gripper) for per-component accuracy.
PARTS = np.array([class_to_parts(c) for c in range(NUM_CLASSES)], dtype=int)


def split_episodes(episode_ids, tasks, seed):
    """Return boolean masks (train, val, test), splitting each task's episodes 70/10/20."""
    rng = np.random.default_rng(seed)
    test_eps, val_eps = set(), set()
    for task in np.unique(tasks):
        eps = np.unique(episode_ids[tasks == task])
        rng.shuffle(eps)
        n_test, n_val = round(0.2 * len(eps)), round(0.1 * len(eps))
        test_eps.update(eps[:n_test])
        val_eps.update(eps[n_test : n_test + n_val])
    test = np.isin(episode_ids, list(test_eps))
    val = np.isin(episode_ids, list(val_eps))
    return ~(test | val), val, test


def prompt_ceiling(keys, y):
    """Oracle: for each distinct key, predict its most common label among the given rows."""
    counts = {}
    for k, label in zip(map(tuple, keys), y):
        counts.setdefault(k, Counter())[label] += 1
    return np.array([counts[k].most_common(1)[0][0] for k in map(tuple, keys)])


def make_head(kind, dim):
    if kind == "linear":
        return nn.Linear(dim, NUM_CLASSES)
    return nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, NUM_CLASSES))


def train_head(kind, x, y, train, val, epochs, device, seed):
    """Standardize features, train with AdamW, keep the best-validation epoch.

    Returns (head, mean, std); apply as head((x - mean) / std).
    """
    torch.manual_seed(seed)
    mean = x[train].mean(0)
    std = x[train].std(0)
    std[std < 1e-6] = 1.0  # constant features (e.g. unused obs slots)
    xs = torch.tensor((x - mean) / std, dtype=torch.float32, device=device)
    ys = torch.tensor(y, device=device)
    tr, va = torch.tensor(np.where(train)[0], device=device), torch.tensor(np.where(val)[0], device=device)

    head = make_head(kind, x.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    best_acc, best_state = -1.0, None
    for _ in range(epochs):
        head.train()
        for idx in tr[torch.randperm(len(tr), device=device)].split(512):
            loss = nn.functional.cross_entropy(head(xs[idx]), ys[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            acc = (head(xs[va]).argmax(1) == ys[va]).float().mean().item()
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, mean, std


@torch.no_grad()
def predict(head, mean, std, x, device):
    head.eval()
    return head(torch.tensor((x - mean) / std, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()


def report(pred, y, mask, meta):
    """Test accuracy overall, per task, per subtask, and per action component."""
    p, t = pred[mask], y[mask]
    tasks, subtasks = meta["task"][mask], meta["subtask"][mask]
    correct = p == t
    return {
        "overall": float(correct.mean()),
        "per_task": {k: float(correct[tasks == k].mean()) for k in np.unique(tasks)},
        "per_subtask": {f"{tk} | {st}": float(correct[(tasks == tk) & (subtasks == st)].mean())
                        for tk, st in sorted(set(zip(tasks, subtasks)))},
        "per_component": {name: float((PARTS[p, i] == PARTS[t, i]).mean())
                          for i, name in enumerate(["x", "y", "z", "gripper"])},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3-1.7B", help="folder name under outputs/hidden/")
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="default: all cached layers")
    parser.add_argument("--heads", nargs="+", default=["linear", "mlp"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--baseline-epochs", type=int, default=1000, help="small-input MLPs learn slowly")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="default: outputs/heads/<model>")
    args = parser.parse_args()
    feat_dir = os.path.join("outputs/hidden", args.model)
    args.out = args.out or os.path.join("outputs/heads", args.model)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = dict(np.load(os.path.join(feat_dir, "meta.npz")))
    y = meta["action_class"]
    train, val, test = split_episodes(meta["episode_id"], meta["task"], args.seed)
    layers = args.layers or sorted(int(f[5:-4]) for f in os.listdir(feat_dir) if f.startswith("layer"))
    print(f"{args.model}: {train.sum()} train / {val.sum()} val / {test.sum()} test steps, device {device}")

    results = {"model": args.model, "args": vars(args),
               "n": {"train": int(train.sum()), "val": int(val.sum()), "test": int(test.sum())},
               "baselines": {}, "layers": {}}

    # Baselines: most common class overall / per task, and MLPs on numbers instead of Qwen features.
    majority = np.bincount(y[train], minlength=NUM_CLASSES).argmax()
    results["baselines"]["majority"] = report(np.full_like(y, majority), y, test, meta)
    task_major = {t: np.bincount(y[train & (meta["task"] == t)], minlength=NUM_CLASSES).argmax()
                  for t in np.unique(meta["task"])}
    results["baselines"]["task_majority"] = report(np.array([task_major[t] for t in meta["task"]]), y, test, meta)
    obs = meta["obs"]
    task_id = np.stack([meta["task"] == t for t in np.unique(meta["task"])], 1).astype(np.float32)
    prompt_numbers = np.round(np.concatenate([obs[:, 0:7], obs[:, 36:39]], 1), 2)  # hand, gripper, object, goal
    for name, x in [("state_mlp", np.concatenate([obs, task_id], 1)),
                    ("prompt_mlp", np.concatenate([prompt_numbers, task_id], 1))]:
        head, mean, std = train_head("mlp", x, y, train, val, args.baseline_epochs, device, args.seed)
        results["baselines"][name] = report(predict(head, mean, std, x, device), y, test, meta)
    keys = np.concatenate([prompt_numbers, task_id], 1)
    oracle = np.zeros_like(y)
    oracle[test] = prompt_ceiling(keys[test], y[test])
    results["baselines"]["prompt_ceiling"] = report(oracle, y, test, meta)
    for name, r in results["baselines"].items():
        print(f"  baseline {name:<14} test acc {r['overall']:.3f}")

    print(f"\n{'layer':>5}" + "".join(f"{k + ' test':>13}{k + ' train':>13}" for k in args.heads))
    for k in layers:
        x = np.load(os.path.join(feat_dir, f"layer{k}.npy")).astype(np.float32)
        results["layers"][k] = {}
        line = f"{k:>5}"
        for kind in args.heads:
            start = time.time()
            head, mean, std = train_head(kind, x, y, train, val, args.epochs, device, args.seed)
            pred = predict(head, mean, std, x, device)
            r = report(pred, y, test, meta)
            r["train_acc"] = float((pred[train] == y[train]).mean())
            results["layers"][k][kind] = r
            torch.save({"kind": kind, "layer": k, "model": args.model, "state_dict": head.state_dict(),
                        "mean": mean, "std": std}, os.path.join(args.out, f"{kind}_layer{k}.pt"))
            line += f"{r['overall']:>13.3f}{r['train_acc']:>13.3f}"
        print(line + f"   ({time.time() - start:.0f}s/head)")

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved heads and results.json to {args.out}")


if __name__ == "__main__":
    main()
