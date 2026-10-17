"""Shared helpers: task list, env creation, prompt formatting, action conversion.

Every other script imports from here so that the prompt format and the
discretization settings are defined in exactly one place.

Run `python src/common.py` for a quick self-test.
"""

import gymnasium as gym
import metaworld  # noqa: F401  (importing registers the "Meta-World/..." envs)
import numpy as np
from metaworld.policies import (
    SawyerPickPlaceV3Policy,
    SawyerPushV3Policy,
    SawyerReachV3Policy,
)

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

# task name -> (sentence used in the prompt, scripted expert policy class)
TASKS = {
    "reach-v3": ("reach the goal.", SawyerReachV3Policy),
    "push-v3": ("push the puck to the goal.", SawyerPushV3Policy),
    "pick-place-v3": ("pick up the puck and place it at the goal.", SawyerPickPlaceV3Policy),
}


def make_env(task, seed=None):
    """Create a single Meta-World MT1 env. Each reset() samples new object/goal positions."""
    return gym.make("Meta-World/MT1", env_name=task, seed=seed)


def get_expert(task):
    """Return the scripted expert for a task. Use expert.get_action(obs)."""
    return TASKS[task][1]()


# ---------------------------------------------------------------------------
# Observation -> text prompt
# ---------------------------------------------------------------------------

def _fmt(values):
    # Adding 0.0 turns -0.0 into 0.0 so we never print "-0.00".
    return "(" + ", ".join(f"{round(float(v), 2) + 0.0:.2f}" for v in values) + ")"


def obs_to_prompt(obs, task):
    """Turn a 39-number Meta-World observation into the text prompt Qwen sees.

    Layout: [0:3] hand xyz, [3] gripper opening, [4:7] object xyz, [36:39] goal xyz.
    """
    obs = np.asarray(obs)
    gripper = round(float(obs[3]), 2) + 0.0
    return (
        f"Task: {TASKS[task][0]}\n"
        f"Hand: {_fmt(obs[0:3])}. Gripper: {gripper:.2f}.\n"
        f"Object: {_fmt(obs[4:7])}. Goal: {_fmt(obs[36:39])}.\n"
        f"Action:"
    )


# ---------------------------------------------------------------------------
# Continuous action <-> discrete class
# ---------------------------------------------------------------------------

# Tuned and verified (30/30 on all three tasks). Change these only here.
DEAD_ZONE = 0.05       # |dx| below this counts as "don't move" on that axis
MOVE_SIZE = 0.5        # size of a move when converting a class back to an action
GRIP_THRESHOLD = 0.3   # grip effort above this counts as "closed"
GRIP_OPEN = 0.0        # continuous grip value sent for "open" (what the experts use)
GRIP_CLOSED = 1.0      # continuous grip value sent for "closed"

NUM_CLASSES = 3 * 3 * 3 * 2  # x, y, z in {-1, 0, +1}; gripper in {open, closed}


def continuous_to_class(action):
    """Map a continuous action [dx, dy, dz, grip] to a class index in [0, 54)."""
    action = np.asarray(action, dtype=float)
    dirs = [0 if abs(a) < DEAD_ZONE else int(np.sign(a)) for a in action[:3]]
    ix, iy, iz = (d + 1 for d in dirs)  # -1/0/+1 -> 0/1/2
    ig = int(action[3] > GRIP_THRESHOLD)
    return ((ix * 3 + iy) * 3 + iz) * 2 + ig


def class_to_parts(cls):
    """Split a class index into (dx, dy, dz, gripper_closed), with d* in {-1, 0, +1}."""
    cls = int(cls)
    ig = cls % 2
    rest = cls // 2
    iz = rest % 3
    iy = (rest // 3) % 3
    ix = rest // 9
    return ix - 1, iy - 1, iz - 1, bool(ig)


def class_to_continuous(cls):
    """Map a class index back to a continuous action the env can execute."""
    dx, dy, dz, closed = class_to_parts(cls)
    grip = GRIP_CLOSED if closed else GRIP_OPEN
    return np.array([dx * MOVE_SIZE, dy * MOVE_SIZE, dz * MOVE_SIZE, grip], dtype=np.float32)


def discretize(action):
    """Squash an action through the 54 classes and back (the "discretized expert")."""
    return class_to_continuous(continuous_to_class(action))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Every class survives a round trip class -> continuous -> class.
    for c in range(NUM_CLASSES):
        assert continuous_to_class(class_to_continuous(c)) == c, c
    print(f"Round trip OK for all {NUM_CLASSES} classes.")

    # 2. A few hand-checked conversions.
    assert continuous_to_class([0.0, 0.0, 0.0, 0.0]) == 26        # no move, open
    assert continuous_to_class([0.4, -0.1, 0.01, 0.6]) == 39      # +x, -y, z still, closed
    assert class_to_parts(39) == (1, -1, 0, True)
    print("Hand-checked conversions OK.")

    # 3. A real observation, prompt, and expert action for each task.
    for task in TASKS:
        env = make_env(task, seed=0)
        obs, _ = env.reset()
        action = get_expert(task).get_action(obs)
        c = continuous_to_class(action)
        print(f"\n--- {task} (obs has {len(obs)} numbers) ---")
        print(obs_to_prompt(obs, task))
        print(f"expert action {np.round(action, 2)} -> class {c} {class_to_parts(c)} "
              f"-> {class_to_continuous(c)}")
        env.close()
