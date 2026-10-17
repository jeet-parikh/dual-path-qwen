"""DAgger data collection: let the trained head drive, label every visited state with the expert.

Imitation policies drift into states the expert never visits and then don't know how to recover.
DAgger fixes this: run the learned policy, record each state it visits together with the action
the expert *would* take there (and the expert's subtask), and add these rows to the training data.

Rows use the same format as data_gen.py (episode_id prefix "dagger<iter>_"). Env seeds start at
--seed (default 20000), disjoint from training (0..499) and evaluation (10000+) layouts.

Usage: python src/dagger.py [--model Qwen/Qwen3-1.7B] [--layer 16] [--episodes 200] [--iter 1]
       [--heads outputs/heads/Qwen3-1.7B-subtask] [--out outputs/data_dagger1]
Needs a GPU. The prompt includes the expert's subtask (oracle), as in --subtask oracle rollouts.
"""

import argparse
import json
import os
import warnings

import numpy as np

from common import (TASKS, class_to_continuous, continuous_to_class, episode_env, expert_subtask, get_expert,
                    obs_to_prompt)
from qwen import load_qwen
from rollout import fast_policy, load_head

warnings.filterwarnings("ignore")


def collect(task, seed, policy, expert, max_steps, iteration):
    env, obs = episode_env(task, seed)
    rows, success = [], False
    short = task.removesuffix("-v3")
    for step in range(max_steps):
        subtask = expert_subtask(task, obs)
        cls, _ = policy(obs_to_prompt(obs, task, subtask))
        action = expert.get_action(obs)
        rows.append({
            "episode_id": f"dagger{iteration}_{short}_{seed}", "task": task, "step": step,
            "prompt": obs_to_prompt(obs, task), "action_class": continuous_to_class(action),
            "action_continuous": [round(float(a), 4) for a in action], "subtask": subtask,
            "obs": [round(float(o), 4) for o in obs],
        })
        obs, _, terminated, truncated, info = env.step(class_to_continuous(cls))  # the learned policy acts
        if info["success"]:
            success = True
            break
        if terminated or truncated:
            break
    env.close()
    return rows, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--heads", default="outputs/heads/Qwen3-1.7B-subtask")
    parser.add_argument("--head", default="mlp")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--out", default="outputs/data_dagger1")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    tok, model = load_qwen(args.model)
    head = load_head(os.path.join(args.heads, f"{args.head}_layer{args.layer}.pt"), model.device)
    policy = fast_policy(model, tok, head, args.layer)
    for task in args.tasks:
        expert = get_expert(task)
        n_rows, n_success, agree = 0, 0, []
        with open(os.path.join(args.out, f"{task}.jsonl"), "w") as f:
            for i in range(args.episodes):
                rows, success = collect(task, args.seed + i, policy, expert, args.max_steps, args.iter)
                n_success += success
                n_rows += len(rows)
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        print(f"{task}: policy succeeded {n_success}/{args.episodes}; {n_rows} labeled states -> {args.out}/{task}.jsonl")


if __name__ == "__main__":
    main()
