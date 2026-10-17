"""Closed loop: the trained head drives the robot.

state -> prompt -> first k Qwen layers -> head -> action class -> continuous action -> env.step

Per episode we record success and steps; per decision the time (tokenize + Qwen + head), the
head's top probability, and whether it matched the expert's class at that same state.

Usage: python src/rollout.py [--model Qwen/Qwen3-1.7B] [--layers 4 16 28] [--head mlp] [--episodes 20]
Needs a GPU. Episode i uses env seed (seed + i), default 10000+, so layouts never overlap with
the training data (seeds 0..499).
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import torch

from common import TASKS, class_to_continuous, continuous_to_class, episode_env, get_expert, obs_to_prompt
from qwen import fast_features, load_qwen
from train_heads import make_head

warnings.filterwarnings("ignore")


def load_head(path, device):
    """Load a head saved by train_heads.py. Returns a function: features [B, d] -> probabilities [B, 54]."""
    ckpt = torch.load(path, weights_only=False)
    mean = torch.tensor(ckpt["mean"], device=device)
    std = torch.tensor(ckpt["std"], device=device)
    head = make_head(ckpt["kind"], len(ckpt["mean"])).to(device).eval()
    head.load_state_dict(ckpt["state_dict"])
    return lambda feats: torch.softmax(head((feats - mean) / std), dim=-1)


def fast_policy(model, tok, head, k):
    """Policy that runs the first k layers + head. policy(prompt) -> (class, probabilities)."""
    @torch.no_grad()
    def policy(prompt):
        probs = head(fast_features(model, tok, [prompt], k))[0]
        return int(probs.argmax()), probs.cpu().numpy()
    return policy


def run_episode(task, seed, policy, expert, max_steps):
    """Run one episode (layout given by seed) with `policy`. Returns episode stats and per-step logs."""
    env, obs = episode_env(task, seed)
    steps = []
    success = False
    for t in range(max_steps):
        prompt = obs_to_prompt(obs, task)
        torch.cuda.synchronize()
        start = time.perf_counter()
        cls, probs = policy(prompt)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000
        expert_cls = continuous_to_class(expert.get_action(obs))
        steps.append({"step": t, "class": cls, "expert_class": expert_cls, "top_prob": float(probs.max()), "ms": ms})
        obs, _, terminated, truncated, info = env.step(class_to_continuous(cls))
        if info["success"]:
            success = True
            break
        if terminated or truncated:
            break
    env.close()
    return {
        "success": success,
        "steps": len(steps),
        "ms": float(np.mean([s["ms"] for s in steps])),
        "expert_agreement": float(np.mean([s["class"] == s["expert_class"] for s in steps])),
        "top_prob": float(np.mean([s["top_prob"] for s in steps])),
        "log": steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 16, 28])
    parser.add_argument("--head", default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--out", default=None, help="default: outputs/rollouts/<model>")
    args = parser.parse_args()
    short = args.model.split("/")[-1]
    args.out = args.out or os.path.join("outputs/rollouts", short)
    os.makedirs(args.out, exist_ok=True)

    tok, model = load_qwen(args.model)
    summary = []
    print(f"{'layer':>5}  {'task':<14}{'success':>9}{'steps':>7}{'ms/dec':>8}{'agree':>7}{'conf':>6}")
    for k in args.layers:
        head = load_head(os.path.join("outputs/heads", short, f"{args.head}_layer{k}.pt"), model.device)
        policy = fast_policy(model, tok, head, k)
        for task in args.tasks:
            expert = get_expert(task)  # same seeds for every layer -> same layouts
            episodes = [run_episode(task, args.seed + i, policy, expert, args.max_steps)
                        for i in range(args.episodes)]
            row = {
                "layer": k, "task": task, "head": args.head,
                "success": float(np.mean([e["success"] for e in episodes])),
                "steps": float(np.mean([e["steps"] for e in episodes])),
                "ms": float(np.mean([e["ms"] for e in episodes])),
                "expert_agreement": float(np.mean([e["expert_agreement"] for e in episodes])),
                "top_prob": float(np.mean([e["top_prob"] for e in episodes])),
            }
            summary.append(row)
            with open(os.path.join(args.out, f"{args.head}_layer{k}_{task}.jsonl"), "w") as f:
                for e in episodes:
                    f.write(json.dumps(e) + "\n")
            n_ok = sum(e["success"] for e in episodes)
            print(f"{k:>5}  {task:<14}{f'{n_ok}/{args.episodes}':>9}{row['steps']:>7.0f}{row['ms']:>8.2f}"
                  f"{row['expert_agreement']:>7.0%}{row['top_prob']:>6.2f}")

    with open(os.path.join(args.out, f"summary_{args.head}.json"), "w") as f:
        json.dump({"args": vars(args), "results": summary}, f, indent=2)
    print(f"\nSaved per-episode logs and summary to {args.out}")


if __name__ == "__main__":
    main()
