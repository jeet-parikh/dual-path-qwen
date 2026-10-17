"""Generate expert demonstrations: one JSON line per step in <out>/<task>.jsonl.

At each step we record the expert's action as the label, but the robot executes the
discretized version, so the recorded states are the ones a perfect 54-class policy visits.
Only successful episodes are saved (a few discretized pick-place episodes get stuck).

Usage: python src/data_gen.py [--episodes 200] [--tasks reach-v3 ...] [--seed 0] [--out outputs/data]
"""

import argparse
import json
import os
import warnings
from collections import Counter

from common import (
    TASKS,
    class_to_continuous,
    continuous_to_class,
    expert_subtask,
    get_expert,
    make_env,
    obs_to_prompt,
)

warnings.filterwarnings("ignore")


def generate(task, episodes, seed, max_steps):
    """Yield (rows for one episode, success) for each episode."""
    env = make_env(task, seed=seed)
    expert = get_expert(task)
    short = task.removesuffix("-v3")
    for ep in range(episodes):
        obs, _ = env.reset()
        rows, success = [], False
        for step in range(max_steps):
            action = expert.get_action(obs)
            cls = continuous_to_class(action)
            rows.append({
                "episode_id": f"{short}_{ep:04d}",
                "task": task,
                "step": step,
                "prompt": obs_to_prompt(obs, task),
                "action_class": cls,
                "action_continuous": [round(float(a), 4) for a in action],
                "subtask": expert_subtask(task, obs),
                "obs": [round(float(o), 4) for o in obs],
            })
            obs, _, terminated, truncated, info = env.step(class_to_continuous(cls))
            if info["success"]:
                success = True
                break
            if terminated or truncated:
                break
        yield rows, success
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--out", default="outputs/data")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    for task in args.tasks:
        path = os.path.join(args.out, f"{task}.jsonl")
        n_steps, n_success, classes, subtasks = 0, 0, Counter(), Counter()
        with open(path, "w") as f:
            for rows, success in generate(task, args.episodes, args.seed, args.max_steps):
                if not success:
                    continue
                n_success += 1
                for row in rows:
                    f.write(json.dumps(row) + "\n")
                    classes[row["action_class"]] += 1
                    subtasks[row["subtask"]] += 1
                n_steps += len(rows)

        print(f"\n{task}: kept {n_success}/{args.episodes} episodes (failures dropped), {n_steps} steps "
              f"({n_steps / n_success:.0f}/episode) -> {path}")
        print(f"  classes used: {len(classes)}/54; most common: "
              + ", ".join(f"{c} ({n / n_steps:.0%})" for c, n in classes.most_common(5)))
        print("  subtasks: " + ", ".join(f"{s} ({n / n_steps:.0%})" for s, n in subtasks.items()))


if __name__ == "__main__":
    main()
