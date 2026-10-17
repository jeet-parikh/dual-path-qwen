"""Learned router: detect when the subtask in the fast path's prompt is out of date.

A confidence router misses stale plans: the fast head is confident while executing the wrong
subtask. This router reads the fast path's own layer-k features (already computed every step,
so it costs nothing extra) and predicts whether the prompt's subtask still matches the state.

Training data: each state paired with its true subtask (fresh, label 0) and with a different
subtask of the same task (stale, label 1). Split by episode, as everywhere else.

Outputs: <out>/stale_layer<k>.pt and results.json.

Usage: python src/stale_router.py [--model Qwen/Qwen3-1.7B] [--layers 4 16] [--states 40000]
Needs a GPU.
"""

import argparse
import json
import os
import warnings

import numpy as np
import torch
from torch import nn

from common import obs_to_prompt
from planner import SUBTASKS
from qwen import layer_features_bucketed, load_qwen
from train_heads import split_episodes

warnings.filterwarnings("ignore")
TASKS_WITH_STAGES = ["push-v3", "pick-place-v3"]


def make_stale_head(dim):
    return nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1))


def train_stale(x, y, train, val, device, epochs=30, seed=0):
    """Binary classifier, best-val epoch. Returns (head, mean, std)."""
    torch.manual_seed(seed)
    mean, std = x[train].mean(0), x[train].std(0)
    std[std < 1e-6] = 1.0
    xs = torch.tensor((x - mean) / std, dtype=torch.float32, device=device)
    ys = torch.tensor(y, dtype=torch.float32, device=device)
    tr, va = torch.tensor(np.where(train)[0], device=device), torch.tensor(np.where(val)[0], device=device)
    head = make_stale_head(x.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = -1.0, None
    for _ in range(epochs):
        head.train()
        for idx in tr[torch.randperm(len(tr), device=device)].split(512):
            loss = nn.functional.binary_cross_entropy_with_logits(head(xs[idx])[:, 0], ys[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            acc = ((head(xs[va])[:, 0] > 0).float() == ys[va]).float().mean().item()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--data", default="outputs/data_d1")
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 16])
    parser.add_argument("--states", type=int, default=40000, help="states sampled (x2 prompts)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="default: outputs/routers/<model>")
    args = parser.parse_args()
    args.out = args.out or os.path.join("outputs/routers", args.model.split("/")[-1])
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    rng = np.random.default_rng(args.seed)

    rows = [json.loads(line) for t in TASKS_WITH_STAGES for line in open(os.path.join(args.data, f"{t}.jsonl"))]
    rows = [rows[i] for i in rng.choice(len(rows), size=min(args.states, len(rows)), replace=False)]
    prompts, labels, episodes, tasks = [], [], [], []
    for r in rows:
        wrong = str(rng.choice([s for s in SUBTASKS[r["task"]] if s != r["subtask"]]))
        for subtask, label in [(r["subtask"], 0), (wrong, 1)]:
            prompts.append(obs_to_prompt(r["obs"], r["task"], subtask))
            labels.append(label)
            episodes.append(r["episode_id"])
            tasks.append(r["task"])
    y, episodes, tasks = np.array(labels), np.array(episodes), np.array(tasks)
    train, val, test = split_episodes(episodes, tasks, args.seed)
    print(f"{len(prompts)} prompts ({train.sum()} train / {val.sum()} val / {test.sum()} test)")

    tok, model = load_qwen(args.model)
    feats = layer_features_bucketed(model, tok, prompts, args.layers)
    results = {}
    for k in args.layers:
        x = feats[k].cpu().numpy()
        head, mean, std = train_stale(x, y, train, val, device)
        with torch.no_grad():
            p = torch.sigmoid(head(torch.tensor((x - mean) / std, dtype=torch.float32, device=device))[:, 0]).cpu().numpy()
        pred = p > 0.5
        results[k] = {
            "test_acc": float((pred[test] == y[test]).mean()),
            "test_recall_stale": float(pred[test & (y == 1)].mean()),
            "test_false_alarm": float(pred[test & (y == 0)].mean()),
        }
        torch.save({"layer": k, "state_dict": head.state_dict(), "mean": mean, "std": std},
                   os.path.join(args.out, f"stale_layer{k}.pt"))
        r = results[k]
        print(f"layer {k:>2}: test acc {r['test_acc']:.1%}, catches {r['test_recall_stale']:.1%} of stale prompts, "
              f"false alarms on {r['test_false_alarm']:.1%} of fresh prompts")
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
