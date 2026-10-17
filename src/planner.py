"""Slow path: the full Qwen model as planner, deciding which subtask the robot should do now.

The planner sees the task, the task's procedure in plain language, and the current state as text,
and must answer with one of the task's subtasks. Scored against the expert's actual stage
(`expert_subtask`). Qwen3 can answer directly or "think" step by step first (--thinking).

Usage: python src/planner.py [--model Qwen/Qwen3-1.7B] [--per-subtask 30] [--thinking]
Needs a GPU. Evaluates on states from held-out episodes (same split as train_heads.py).
"""

import argparse
import json
import os
import re
import time
import warnings
from collections import defaultdict

import numpy as np
import torch

from common import TASKS
from qwen import load_qwen
from train_heads import split_episodes

warnings.filterwarnings("ignore")

# Subtasks in order, and the procedure each expert follows (thresholds from the expert scripts).
SUBTASKS = {
    "reach-v3": ["move to the goal"],
    "push-v3": ["move above the puck", "lower onto the puck", "push the puck to the goal"],
    "pick-place-v3": ["move above the puck", "lower onto the puck", "grasp the puck", "carry the puck to the goal"],
}
PROCEDURES = {
    "reach-v3": "Move the hand to the goal.",
    "push-v3": ("1. If the hand is more than 2 cm away from the puck horizontally (x, y), move above the puck. "
                "2. Otherwise, if the hand is more than 4 cm above the puck, lower onto the puck. "
                "3. Otherwise, push the puck to the goal."),
    "pick-place-v3": ("1. If the hand is more than 2 cm away from the puck horizontally (x, y), move above the puck. "
                      "2. Otherwise, if the hand is more than 5 cm above the puck and the puck is still on the table "
                      "(puck height below 0.04), lower onto the puck. "
                      "3. Otherwise, if the gripper opening is above 0.73, grasp the puck. "
                      "4. Otherwise, carry the puck to the goal."),
}


def state_text(task, obs):
    f = lambda v: "(" + ", ".join(f"{x:.3f}" for x in v) + ")"
    return (f"Hand: {f(obs[0:3])}. Gripper opening: {obs[3]:.2f}.\n"
            f"Puck: {f(obs[4:7])}. Goal: {f(obs[36:39])}.")


def planner_messages(task, obs):
    options = "; ".join(f'"{s}"' for s in SUBTASKS[task])
    return [{"role": "user", "content": (
        f"You are the planner for a robot arm. Task: {TASKS[task][0]}\n"
        f"Procedure: {PROCEDURES[task]}\n"
        f"Positions are in meters (x, y, z).\n{state_text(task, obs)}\n"
        f"Which subtask should the robot do right now? Options: {options}. "
        f"Answer with the option text only.")}]


def parse_answer(text, task):
    text = text.split("</think>")[-1].lower()
    hits = [(text.rfind(s), s) for s in SUBTASKS[task] if s in text]
    return max(hits)[1] if hits else None


@torch.no_grad()
def plan(model, tok, task, observations, thinking, max_new_tokens):
    """Batched planner calls. Returns (answers, raw texts)."""
    prompts = [tok.apply_chat_template(planner_messages(task, o), tokenize=False, add_generation_prompt=True,
                                       enable_thinking=thinking) for o in observations]
    batch = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
    texts = tok.batch_decode(out[:, batch.input_ids.shape[1]:], skip_special_tokens=True)
    return [parse_answer(t, task) for t in texts], texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--features", default="outputs/hidden/Qwen3-1.7B", help="for meta.npz (states + split)")
    parser.add_argument("--per-subtask", type=int, default=30)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/planner")
    args = parser.parse_args()
    max_new = 2048 if args.thinking else 24

    meta = dict(np.load(os.path.join(args.features, "meta.npz")))
    _, _, test = split_episodes(meta["episode_id"], meta["task"], args.seed)
    rng = np.random.default_rng(args.seed)
    tok, model = load_qwen(args.model)

    results, examples = {}, []
    for task in ["push-v3", "pick-place-v3"]:
        correct, n_by = defaultdict(int), defaultdict(int)
        idx = [i for s in SUBTASKS[task]
               for i in rng.permutation(np.where(test & (meta["task"] == task) & (meta["subtask"] == s))[0])[: args.per_subtask]]
        start = time.time()
        for b in range(0, len(idx), args.batch_size):
            rows = idx[b : b + args.batch_size]
            answers, texts = plan(model, tok, task, [meta["obs"][i] for i in rows], args.thinking, max_new)
            for i, a, t in zip(rows, answers, texts):
                truth = str(meta["subtask"][i])
                n_by[truth] += 1
                correct[truth] += a == truth
                if len(examples) < 6:
                    examples.append({"task": task, "truth": truth, "answer": a, "text": t[-400:]})
        secs = time.time() - start
        acc = sum(correct.values()) / len(idx)
        results[task] = {"accuracy": acc, "per_subtask": {s: correct[s] / n_by[s] for s in n_by},
                         "sec_per_call_batched": secs / len(idx)}
        print(f"{task}: accuracy {acc:.0%} ({len(idx)} states, {secs / len(idx) * 1000:.0f} ms/call batched) "
              + ", ".join(f"{s}: {correct[s]}/{n_by[s]}" for s in n_by))

    os.makedirs(args.out, exist_ok=True)
    name = f"{args.model.split('/')[-1]}_{'think' if args.thinking else 'direct'}.json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump({"args": vars(args), "results": results, "examples": examples}, f, indent=2)
    print(f"Saved {args.out}/{name}")


if __name__ == "__main__":
    main()
