"""Closed loop: the trained head drives the robot.

state -> prompt -> first k Qwen layers -> head -> action class -> continuous action -> env.step

Per episode we record success and steps; per decision the time (tokenize + Qwen + head), the
head's top probability, and whether it matched the expert's class at that same state.

With --subtask oracle, the prompt includes the expert's current stage (an upper bound for a planner);
heads are then loaded from outputs/heads/<model>-subtask.
With --subtask planner, the SAME Qwen model (full depth, thinking mode) decides the subtask: at the
start, every --plan-every steps, whenever the fast head's top probability drops below
--router-threshold, and/or whenever the learned staleness router (--stale-router, reading the fast
path's own layer-k features) says the current subtask is out of date with probability above
--stale-threshold (with at least --cooldown steps between calls).

Usage: python src/rollout.py [--model Qwen/Qwen3-1.7B] [--layers 4 16 28] [--head mlp] [--episodes 20] [--subtask oracle]
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

from common import (TASKS, class_to_continuous, continuous_to_class, episode_env, expert_subtask, get_expert,
                    obs_to_prompt)
from planner import SUBTASKS, plan
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
        feats = fast_features(model, tok, [prompt], k)
        policy.last_features = feats
        probs = head(feats)[0]
        return int(probs.argmax()), probs.cpu().numpy()
    return policy


def load_stale_router(path, device):
    """Staleness router from stale_router.py: features [1, d] -> probability the subtask is stale."""
    from stale_router import make_stale_head
    ckpt = torch.load(path, weights_only=False)
    mean, std = torch.tensor(ckpt["mean"], device=device), torch.tensor(ckpt["std"], device=device)
    head = make_stale_head(len(ckpt["mean"])).to(device).eval()
    head.load_state_dict(ckpt["state_dict"])
    return lambda feats: float(torch.sigmoid(head((feats - mean) / std))[0, 0])


class PlannerSubtask:
    """Stateful subtask source for one episode: asks the full model (thinking) when needed."""

    def __init__(self, model, tok, every=0, threshold=None, cooldown=5, max_new_tokens=1536,
                 stale_router=None, stale_threshold=0.8):
        self.model, self.tok = model, tok
        self.stale_router, self.stale_threshold, self.last_stale = stale_router, stale_threshold, 0.0
        self.every, self.threshold, self.cooldown, self.max_new = every, threshold, cooldown, max_new_tokens
        self.current, self.t, self.last_call, self.last_conf = None, 0, -10**9, 1.0
        self.calls, self.seconds, self.log = 0, 0.0, []

    def observe(self, probs, features=None):
        self.last_conf = float(probs.max())
        if self.stale_router is not None and features is not None:
            self.last_stale = self.stale_router(features)

    def __call__(self, task, obs):
        due = self.current is None or (self.every and self.t - self.last_call >= self.every)
        cooled = self.t - self.last_call >= self.cooldown
        unsure = cooled and (
            (self.threshold is not None and self.last_conf < self.threshold)
            or (self.stale_router is not None and self.last_stale > self.stale_threshold))
        if due or unsure:
            start = time.perf_counter()
            answer = plan(self.model, self.tok, task, [obs], True, self.max_new)[0][0]
            self.seconds += time.perf_counter() - start
            self.calls += 1
            self.last_call = self.t
            truth = expert_subtask(task, obs)
            self.log.append({"step": self.t, "answer": answer, "truth": truth, "trigger": "due" if due else "unsure"})
            self.current = answer or self.current or SUBTASKS[task][0]
        self.t += 1
        return self.current


def run_episode(task, seed, policy, expert, max_steps, subtask_fn=None):
    """Run one episode (layout given by seed) with `policy`. Returns episode stats and per-step logs.

    subtask_fn(task, obs) -> subtask text for the prompt, or None for no subtask line.
    """
    env, obs = episode_env(task, seed)
    steps = []
    success = False
    for t in range(max_steps):
        prompt = obs_to_prompt(obs, task, subtask_fn(task, obs) if subtask_fn else None)
        torch.cuda.synchronize()
        start = time.perf_counter()
        cls, probs = policy(prompt)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000
        if hasattr(subtask_fn, "observe"):
            subtask_fn.observe(probs, getattr(policy, "last_features", None))
        expert_cls = continuous_to_class(expert.get_action(obs))
        steps.append({"step": t, "class": cls, "expert_class": expert_cls, "top_prob": float(probs.max()), "ms": ms})
        obs, _, terminated, truncated, info = env.step(class_to_continuous(cls))
        if info["success"]:
            success = True
            break
        if terminated or truncated:
            break
    env.close()
    planner = {}
    if isinstance(subtask_fn, PlannerSubtask):
        planner = {"planner_calls": subtask_fn.calls, "planner_seconds": subtask_fn.seconds,
                   "planner_accuracy": float(np.mean([c["answer"] == c["truth"] for c in subtask_fn.log])),
                   "planner_log": subtask_fn.log}
    return {
        **planner,
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
    parser.add_argument("--subtask", default="none", choices=["none", "oracle", "planner"])
    parser.add_argument("--plan-every", type=int, default=0, help="planner call every N steps (0 = off)")
    parser.add_argument("--router-threshold", type=float, default=None, help="call planner if top prob < this")
    parser.add_argument("--cooldown", type=int, default=5)
    parser.add_argument("--heads-dir", default=None, help="override the heads folder")
    parser.add_argument("--stale-router", default=None, help="path to stale_layer<k>.pt (must match --layers)")
    parser.add_argument("--stale-threshold", type=float, default=0.8)
    parser.add_argument("--out", default=None, help="default: outputs/rollouts/<model>[-subtask]")
    args = parser.parse_args()
    short = args.model.split("/")[-1]
    uses_subtask = args.subtask != "none"
    heads_dir = args.heads_dir or os.path.join("outputs/heads", short + ("-subtask" if uses_subtask else ""))
    tag = {"none": "", "oracle": "-subtask", "planner": f"-planner-every{args.plan_every}-thr{args.router_threshold}"
           + ("-stale" if args.stale_router else "")}
    args.out = args.out or os.path.join("outputs/rollouts", short + tag[args.subtask])
    os.makedirs(args.out, exist_ok=True)

    tok, model = load_qwen(args.model)
    stale = load_stale_router(args.stale_router, model.device) if args.stale_router else None
    summary = []
    print(f"{'layer':>5}  {'task':<14}{'success':>9}{'steps':>7}{'ms/dec':>8}{'agree':>7}{'conf':>6}")
    for k in args.layers:
        head = load_head(os.path.join(heads_dir, f"{args.head}_layer{k}.pt"), model.device)
        policy = fast_policy(model, tok, head, k)
        for task in args.tasks:
            expert = get_expert(task)  # same seeds for every layer -> same layouts
            episodes = []
            for i in range(args.episodes):
                if args.subtask == "planner":
                    subtask_fn = PlannerSubtask(model, tok, args.plan_every, args.router_threshold, args.cooldown,
                                                stale_router=stale, stale_threshold=args.stale_threshold)
                else:
                    subtask_fn = expert_subtask if args.subtask == "oracle" else None
                episodes.append(run_episode(task, args.seed + i, policy, expert, args.max_steps, subtask_fn))
            row = {
                "layer": k, "task": task, "head": args.head,
                "success": float(np.mean([e["success"] for e in episodes])),
                "steps": float(np.mean([e["steps"] for e in episodes])),
                "ms": float(np.mean([e["ms"] for e in episodes])),
                "expert_agreement": float(np.mean([e["expert_agreement"] for e in episodes])),
                "top_prob": float(np.mean([e["top_prob"] for e in episodes])),
            }
            if args.subtask == "planner":
                row["planner_calls"] = float(np.mean([e["planner_calls"] for e in episodes]))
                row["planner_seconds"] = float(np.mean([e["planner_seconds"] for e in episodes]))
                row["planner_accuracy"] = float(np.mean([e["planner_accuracy"] for e in episodes]))
            summary.append(row)
            with open(os.path.join(args.out, f"{args.head}_layer{k}_{task}.jsonl"), "w") as f:
                for e in episodes:
                    f.write(json.dumps(e) + "\n")
            n_ok = sum(e["success"] for e in episodes)
            print(f"{k:>5}  {task:<14}{f'{n_ok}/{args.episodes}':>9}{row['steps']:>7.0f}{row['ms']:>8.2f}"
                  f"{row['expert_agreement']:>7.0%}{row['top_prob']:>6.2f}"
                  + (f"  planner: {row['planner_calls']:.1f} calls, {row['planner_seconds']:.0f}s, "
                     f"{row['planner_accuracy']:.0%} right" if args.subtask == "planner" else ""))

    with open(os.path.join(args.out, f"summary_{args.head}.json"), "w") as f:
        json.dump({"args": vars(args), "results": summary}, f, indent=2)
    print(f"\nSaved per-episode logs and summary to {args.out}")


if __name__ == "__main__":
    main()
