"""Closed loop for multi-step instructions: planner picks each place, the fast path drives there.

All test chains run in lockstep so planner calls and fast-path steps are batched. When a chain
needs its next place (at the start and after each arrival), the planner is asked; the place's xyz
goes in the prompt's Goal slot and the executor moves the hand until it is within 5 cm. A chain
succeeds when it visits exactly the instructed places in order.

Executors:
  expert   the scripted expert, discretized (CPU; isolates planner errors)
  fast     first k layers of Qwen + a head trained on chain step data (GPU)

Usage: python src/chain_rollout.py --planner bow --executor expert --limit 100
       python src/chain_rollout.py --planner think --executor fast --head outputs/heads/Qwen3-1.7B-chain/mlp_layer16.pt
"""

import argparse
import json
import os
import time
import warnings

import numpy as np

from chain import COLORS, SEGMENT_STEPS, TASK, expert_policy, place_xyz
from chain_planner import load_chains, make_planner
from common import class_to_continuous, episode_env, obs_to_prompt
from lang import SUCCESS_RADIUS, with_target

warnings.filterwarnings("ignore")


def fast_executor(tok, model, head_path, k):
    import torch
    from qwen import fast_features
    from rollout import load_head

    head = load_head(head_path, model.device)

    @torch.no_grad()
    def act(observations):
        prompts = [obs_to_prompt(o, TASK) for o in observations]
        return head(fast_features(model, tok, prompts, k)).argmax(-1).tolist()
    return act


def run(chains, plan, act):
    """Run all chains in lockstep. Returns one result dict per chain."""
    states = []
    for c in chains:
        env, obs = episode_env(TASK, c["seed"])
        env.unwrapped.max_path_length = 10 ** 6
        states.append({"chain": c, "env": env, "obs": obs, "start": obs[0:3].copy(), "visited": [],
                       "target": None, "seg_t": 0, "path": [obs[0:3].tolist()], "calls": [], "texts": [],
                       "done": False, "failed": None})
    planner_secs, n_calls, exec_steps = 0.0, 0, 0
    while True:
        need = [s for s in states if not s["done"] and s["target"] is None]
        if need:
            t0 = time.time()
            answers, texts = plan([(s["chain"], len(s["visited"]), s["obs"][0:3]) for s in need])
            planner_secs += time.time() - t0
            n_calls += len(need)
            for s, a, t in zip(need, answers, texts):
                s["calls"].append({"t": len(s["path"]) - 1, "answer": a, "text": t[-1500:]})
                if a not in COLORS + ["start"]:
                    s["done"], s["failed"] = True, "unparseable answer"
                    continue
                s["place"] = a
                s["target"] = place_xyz(a, s["chain"]["markers"], s["start"])
                s["seg_t"] = 0
        active = [s for s in states if not s["done"]]
        if not active:
            break
        classes = act([with_target(s["obs"], s["target"]) for s in active])
        exec_steps += len(active)
        for s, cls in zip(active, classes):
            s["obs"], *_ = s["env"].unwrapped.step(class_to_continuous(cls))
            s["path"].append(s["obs"][0:3].tolist())
            s["seg_t"] += 1
            if np.linalg.norm(s["obs"][0:3] - s["target"]) <= SUCCESS_RADIUS:
                s["visited"].append(s["place"])
                s["target"] = None
                if len(s["visited"]) == s["chain"]["length"]:
                    s["done"] = True
            elif s["seg_t"] >= SEGMENT_STEPS:
                s["done"], s["failed"] = True, "did not reach the place"
    results = []
    for s in states:
        s["env"].close()
        c = s["chain"]
        results.append({"seed": c["seed"], "length": c["length"], "instruction": c["instruction"],
                        "markers": c["markers"], "places": c["places"], "visited": s["visited"],
                        "success": s["visited"] == c["places"], "failed": s["failed"],
                        "start": s["start"].tolist(), "path": s["path"], "calls": s["calls"]})
    return results, {"planner_sec_per_call": planner_secs / max(n_calls, 1), "planner_calls": n_calls,
                     "executor_steps": exec_steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=["keyword", "bow", "direct", "think", "oracle"], required=True)
    parser.add_argument("--executor", choices=["expert", "fast"], default="expert")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--head", default="outputs/heads/Qwen3-1.7B-chain/mlp_layer16.pt")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--train", default="outputs/chain_train")
    parser.add_argument("--test", default="outputs/chain_test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default="outputs/chain_rollouts")
    args = parser.parse_args()

    loaded = None
    if args.executor == "fast" or args.planner in ("direct", "think"):
        from qwen import load_qwen
        loaded = load_qwen(args.model)
    plan = make_planner(args, loaded)
    act = fast_executor(*loaded, args.head, args.k) if args.executor == "fast" else \
        (lambda observations: [expert_policy(o) for o in observations])

    chains = load_chains(args.test, args.limit)
    t0 = time.time()
    results, timing = run(chains, plan, act)
    by_len = {}
    for r in results:
        by_len.setdefault(r["length"], []).append(r["success"])
    summary = {"success_by_length": {k: float(np.mean(v)) for k, v in sorted(by_len.items())},
               "n_by_length": {k: len(v) for k, v in sorted(by_len.items())},
               "success": float(np.mean([r["success"] for r in results])),
               "minutes": (time.time() - t0) / 60, **timing}
    print(f"{args.planner} + {args.executor}: success {summary['success']:.0%} | by length "
          + ", ".join(f"{k}: {v:.0%}" for k, v in summary["success_by_length"].items())
          + f" | {summary['minutes']:.1f} min")

    os.makedirs(args.out, exist_ok=True)
    planner = args.planner if args.planner in ("keyword", "bow", "oracle") else f"{args.planner}_{args.model.split('/')[-1]}"
    name = f"{planner}__{args.executor}" + (f"_k{args.k}" if args.executor == "fast" else "")
    with open(os.path.join(args.out, f"{name}.json"), "w") as f:
        json.dump({"args": vars(args), "summary": summary, "results": results}, f)
    print(f"Saved {args.out}/{name}.json")


if __name__ == "__main__":
    main()
