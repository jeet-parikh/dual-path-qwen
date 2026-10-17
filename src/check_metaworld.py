"""Check that the scripted expert, and the expert squashed through the 54 classes, solve each task.

A classifier that imitates the expert can at best match the discretized expert,
so both should succeed on (nearly) every episode.

Usage: python src/check_metaworld.py [--episodes 30] [--tasks reach-v3 push-v3] [--seed 0]
"""

import argparse
import time
import warnings

import numpy as np

from common import TASKS, discretize, episode_env, get_expert

warnings.filterwarnings("ignore")  # Gymnasium/Meta-World warnings are harmless noise here


def run_episodes(task, episodes, seed, squash):
    """Run episodes with the expert (optionally discretized). Returns (successes, steps per episode)."""
    expert = get_expert(task)
    successes, steps = 0, []
    for ep in range(episodes):
        env, obs = episode_env(task, seed + ep)  # one unique layout per episode
        for t in range(1, 501):
            action = expert.get_action(obs)
            if squash:
                action = discretize(action)
            obs, _, terminated, truncated, info = env.step(action)
            if info["success"]:
                successes += 1
                break
            if terminated or truncated:
                break
        steps.append(t)
        env.close()
    return successes, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"{'task':<15}{'policy':<22}{'success':>10}{'mean steps':>12}")
    for task in args.tasks:
        for name, squash in [("expert", False), ("discretized expert", True)]:
            start = time.time()
            wins, steps = run_episodes(task, args.episodes, args.seed, squash)
            print(f"{task:<15}{name:<22}{f'{wins}/{args.episodes}':>10}{np.mean(steps):>12.1f}"
                  f"   ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
