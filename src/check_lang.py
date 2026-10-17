"""Expert feasibility sweep for language-specified targets.

For each episode: sample markers + an instruction (train wording), compute the target, and run the
scripted expert (plain and discretized) with obs[36:39] overwritten by the target every step.
Success = hand (reach) or puck (push) within 5 cm of the target. Reported per family and per
offset direction, to find target regions where the expert is reliable.

Usage: python src/check_lang.py [--episodes 300] [--tasks reach-v3 push-v3] [--seed 0]
"""

import argparse
import warnings
from collections import defaultdict

import numpy as np

from common import discretize, episode_env, get_expert
from lang import lang_success, sample_episode, with_target

warnings.filterwarnings("ignore")


def run(task, seed, squash, max_steps=500):
    env, obs = episode_env(task, seed)
    ep = sample_episode(np.random.default_rng(seed), task, "train", obs[0:2], obs[4:6])
    expert = get_expert(task)
    success = False
    for t in range(max_steps):
        action = expert.get_action(with_target(obs, ep["target"]))
        if squash:
            action = discretize(action)
        obs, _, terminated, truncated, _ = env.step(action)
        if lang_success(task, obs, ep["target"]):
            success = True
            break
        if terminated or truncated:
            break
    env.close()
    return ep, success, t + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--tasks", nargs="+", default=["reach-v3", "push-v3"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for task in args.tasks:
        for name, squash in [("expert", False), ("discretized", True)]:
            groups = defaultdict(list)
            steps = []
            for i in range(args.episodes):
                ep, ok, n = run(task, args.seed + i, squash)
                spec = ep["spec"]
                groups[spec["family"]].append(ok)
                if spec["family"] == "offset":
                    groups[f"offset:{spec['direction']}"].append(ok)
                steps.append(n)
            total = np.mean([ok for k, v in groups.items() if ":" not in k for ok in v])
            print(f"\n{task} {name}: {total:.1%} success, mean {np.mean(steps):.0f} steps")
            for k in sorted(groups):
                print(f"  {k:<16} {sum(groups[k])}/{len(groups[k])}")


if __name__ == "__main__":
    main()
