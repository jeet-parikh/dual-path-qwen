"""Multi-step instructions in the reach scene: the plan must be worked out from the words.

Four colored (text-only) markers lie on the table. An instruction lists 1-4 steps, e.g.
"First, touch the leftmost marker. Then go to the marker closest to the one you just touched.
After that, go back to where you started." Each step resolves to one place (a marker color or
"start"), but only by reasoning over the scene and the earlier steps:

  direct     the red marker
  extreme    the leftmost / rightmost / closest-to-the-robot / farthest-from-the-robot marker
  nearest    the marker closest to the red one          (excluding red itself)
  farthest   the marker farthest from the red one
  relative   the marker closest to the one you just touched   (depends on the previous step)
  back       back to where you started / back to the first marker you touched
  cond       if the red marker is left of the blue one, the green marker; otherwise the yellow one

The planner sees the instruction, the scene, and how many steps are done, and names the next
place. The fast path then drives to it: the place's xyz goes in the prompt's Goal slot, so its
prompt has the same format as the original reach data (obs_to_prompt).

Episodes step the raw env (env.unwrapped): MT1's wrappers stop at 500 steps and end the
episode when the hand passes Meta-World's own (unused) goal.

Usage:
  python src/chain.py                               self-test + expert check on 50 chains
  python src/chain.py --episodes 1000 --out outputs/chain_data   chains + fast-path step data
"""

import argparse
import json
import os
import warnings

import numpy as np

from common import class_to_continuous, continuous_to_class, episode_env, get_expert, obs_to_prompt
from lang import REACH_Z, SUCCESS_RADIUS, with_target

warnings.filterwarnings("ignore")

TASK = "reach-v3"
COLORS = ["red", "blue", "green", "yellow"]
MARKER_REGION = (np.array([-0.25, 0.50]), np.array([0.25, 0.85]))
MIN_MARKER_GAP = 0.10
FAMILIES = ["direct", "extreme", "nearest", "farthest", "relative", "back", "cond"]
EXTREMES = {  # phrase -> (axis, sign): pick the marker with the largest sign * coordinate
    "the leftmost marker": (0, -1),
    "the rightmost marker": (0, 1),
    "the marker closest to the robot": (1, -1),
    "the marker farthest from the robot": (1, 1),
}
COMPARISONS = {  # phrase -> (axis, sign): "a <phrase> b" is true when sign * (a - b) > 0
    "left of": (0, -1),
    "right of": (0, 1),
    "closer to the robot than": (1, -1),
}
VERBS = ["go to", "touch", "move to", "visit"]
CONNECTORS = ["First,", "Then", "Next,", "After that,"]
SEGMENT_STEPS = 200      # expert/fast-path step budget per instruction step
MIN_MOVE = 0.08          # consecutive places must be at least this far apart


# ---------------------------------------------------------------------------
# Scene and instruction
# ---------------------------------------------------------------------------

def sample_markers(rng):
    """All four colors at random positions, MIN_MARKER_GAP apart, listed in shuffled order."""
    colors = list(rng.permutation(COLORS))
    while True:
        xy = rng.uniform(*MARKER_REGION, size=(4, 2))
        if min(np.linalg.norm(xy[i] - xy[j]) for i in range(4) for j in range(i + 1, 4)) >= MIN_MARKER_GAP:
            return [(str(c), float(x), float(y)) for c, (x, y) in zip(colors, xy)]


def place_xyz(place, markers, start):
    if place == "start":
        return np.asarray(start, dtype=float)
    x, y = next((x, y) for c, x, y in markers if c == place)
    return np.array([x, y, REACH_Z])


def _nearest(xy, ref, exclude, sign=1):
    """Color whose marker is nearest to (sign=1) or farthest from (sign=-1) ref, skipping exclude."""
    options = [(sign * np.linalg.norm(p - ref), c) for c, p in xy.items() if c not in exclude]
    return min(options)[1]


def resolve(step, markers, history):
    """The place a step refers to. history = places visited so far, starting with "start"."""
    xy = {c: np.array([x, y]) for c, x, y in markers}
    fam = step["family"]
    if fam == "direct":
        return step["color"]
    if fam == "extreme":
        axis, sign = EXTREMES[step["phrase"]]
        return max(xy, key=lambda c: sign * xy[c][axis])
    if fam == "nearest":
        return _nearest(xy, xy[step["color"]], {step["color"]})
    if fam == "farthest":
        return _nearest(xy, xy[step["color"]], {step["color"]}, sign=-1)
    if fam == "relative":
        return _nearest(xy, xy[history[-1]], {history[-1]})
    if fam == "back":
        return "start" if step["which"] == "start" else history[1]
    if fam == "cond":
        axis, sign = COMPARISONS[step["phrase"]]
        holds = sign * (xy[step["a"]][axis] - xy[step["b"]][axis]) > 0
        return step["then"] if holds else step["else"]
    raise ValueError(fam)


def step_text(step):
    fam = step["family"]
    if fam == "direct":
        return f"the {step['color']} marker"
    if fam == "extreme":
        return step["phrase"]
    if fam == "nearest":
        return f"the marker closest to the {step['color']} one"
    if fam == "farthest":
        return f"the marker farthest from the {step['color']} one"
    if fam == "relative":
        return "the marker closest to the one you just touched"
    if fam == "back":
        return "back to where you started" if step["which"] == "start" else "back to the first marker you touched"
    if fam == "cond":
        return (f"the {step['then']} marker if the {step['a']} marker is {step['phrase']} the {step['b']} one, "
                f"otherwise the {step['else']} marker")
    raise ValueError(fam)


def sample_step(rng, history):
    fam = str(rng.choice(FAMILIES))
    if fam in ("relative", "back") and len(history) < 2 or fam == "relative" and history[-1] == "start":
        fam = "direct"
    step = {"family": fam, "verb": str(rng.choice(VERBS))}
    c = list(rng.choice(COLORS, size=4, replace=False))
    if fam in ("direct", "nearest", "farthest"):
        step["color"] = str(c[0])
    elif fam == "extreme":
        step["phrase"] = str(rng.choice(list(EXTREMES)))
    elif fam == "back":
        step["which"] = str(rng.choice(["start", "first"]))
    elif fam == "cond":
        step.update(a=str(c[0]), b=str(c[1]), then=str(c[2]), phrase=str(rng.choice(list(COMPARISONS))))
        step["else"] = str(c[3])
    return step


def sample_chain(rng, length, start):
    """Markers + steps whose resolved places are all distinct from the place before them."""
    for _ in range(1000):
        markers = sample_markers(rng)
        steps, history = [], ["start"]
        for _ in range(length):
            for _ in range(50):
                step = sample_step(rng, history)
                place = resolve(step, markers, history)
                prev = place_xyz(history[-1], markers, start)
                if np.linalg.norm(place_xyz(place, markers, start)[:2] - prev[:2]) >= MIN_MOVE:
                    break
            else:
                break
            steps.append(step)
            history.append(place)
        if len(steps) == length:
            return {"markers": markers, "steps": steps, "places": history[1:],
                    "instruction": instruction_text(steps)}
    raise RuntimeError("could not sample a chain")


def instruction_text(steps):
    if len(steps) == 1:
        s = steps[0]
        text = step_text(s)
        return text[0].upper() + text[1:] + "." if s["family"] == "back" else f"{s['verb'].capitalize()} {text}."
    parts = []
    for i, s in enumerate(steps):
        conn = CONNECTORS[min(i, len(CONNECTORS) - 1)]
        verb = "go" if s["family"] == "back" else s["verb"]
        parts.append(f"{conn} {verb} {step_text(s)}.")
    return " ".join(parts)


def scene_text(markers):
    return ", ".join(f"{c} ({x:.2f}, {y:.2f})" for c, x, y in markers)


# ---------------------------------------------------------------------------
# Running a chain with a low-level policy
# ---------------------------------------------------------------------------

def run_chain(chain, seed, policy, next_place, record=False):
    """Drive the hand through the places that next_place picks, one instruction step at a time.

    policy(obs_with_target) -> action class; next_place(chain, obs, n_done, history) -> place.
    A step ends when the hand is within SUCCESS_RADIUS of its place (or after SEGMENT_STEPS).
    Returns dict with visited places, success, hand path, planner-call indices, and step rows.
    """
    env, obs = episode_env(TASK, seed)
    raw = env.unwrapped
    raw.max_path_length = 10 ** 6
    start = obs[0:3].copy()
    history, path, calls, rows = ["start"], [obs[0:3].tolist()], [], []
    reached_all = True
    for n_done in range(len(chain["steps"])):
        place = next_place(chain, obs, n_done, history)
        calls.append(len(path) - 1)
        target = place_xyz(place, chain["markers"], start) if place in COLORS + ["start"] else None
        if target is None:  # unparseable planner answer
            reached_all = False
            history.append(place)
            break
        for t in range(SEGMENT_STEPS):
            if np.linalg.norm(obs[0:3] - target) <= SUCCESS_RADIUS:
                break
            o = with_target(obs, target)
            cls = policy(o)
            if record:
                rows.append({"episode_id": f"chain_{seed}", "task": TASK, "step": len(rows),
                             "prompt": obs_to_prompt(o, TASK), "action_class": int(cls),
                             "action_continuous": [round(float(a), 4) for a in expert_action(o)],
                             "subtask": "move to the goal", "segment": n_done,
                             "obs": [round(float(v), 4) for v in o]})
            obs, *_ = raw.step(class_to_continuous(cls))
            path.append(obs[0:3].tolist())
        else:
            reached_all = False
        history.append(place)
        if not reached_all:
            break
    env.close()
    visited = history[1:]
    return {"visited": visited, "success": reached_all and visited == chain["places"],
            "path": path, "calls": calls, "rows": rows, "start": start.tolist()}


_EXPERT = get_expert(TASK)


def expert_action(obs_with_target):
    return _EXPERT.get_action(obs_with_target)


def expert_policy(obs_with_target):
    return continuous_to_class(expert_action(obs_with_target))


def oracle_planner(chain, obs, n_done, history):
    return chain["places"][n_done]


# ---------------------------------------------------------------------------
# Self-test / data generation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=20000, help="episode i uses env seed (seed + i)")
    parser.add_argument("--out", default=None, help="write chains.jsonl and reach-v3.jsonl (step data) here")
    args = parser.parse_args()

    # Hand-checked resolution on a fixed scene.
    markers = [("red", -0.20, 0.60), ("blue", 0.00, 0.80), ("green", 0.15, 0.55), ("yellow", -0.05, 0.62)]
    h = ["start", "green"]
    assert resolve({"family": "extreme", "phrase": "the leftmost marker"}, markers, h) == "red"
    assert resolve({"family": "extreme", "phrase": "the marker farthest from the robot"}, markers, h) == "blue"
    assert resolve({"family": "nearest", "color": "red"}, markers, h) == "yellow"
    assert resolve({"family": "farthest", "color": "red"}, markers, h) == "green"
    assert resolve({"family": "relative"}, markers, h) == "yellow"
    assert resolve({"family": "back", "which": "first"}, markers, ["start", "green", "red"]) == "green"
    cond = {"family": "cond", "a": "red", "b": "blue", "phrase": "left of", "then": "green", "else": "yellow"}
    assert resolve(cond, markers, h) == "green"
    print("Resolution math OK.")

    rng = np.random.default_rng(args.seed)
    start = np.array([0.0, 0.6, 0.2])
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        f_chains = open(os.path.join(args.out, "chains.jsonl"), "w")
        f_steps = open(os.path.join(args.out, f"{TASK}.jsonl"), "w")
    n_ok, n_steps, by_len = 0, 0, {}
    for i in range(args.episodes):
        length = int(args.lengths[i % len(args.lengths)])
        seed = args.seed + i
        chain = sample_chain(rng, length, start)
        chain.update(seed=seed, length=length)
        result = run_chain(chain, seed, expert_policy, oracle_planner, record=bool(args.out))
        ok = result["success"]
        n_ok += ok
        by_len.setdefault(length, []).append(ok)
        if i < 3:
            print(f"\n[{length} steps] {chain['instruction']}\n  markers: {scene_text(chain['markers'])}"
                  f"\n  places: {chain['places']}  expert success: {ok}  ({len(result['path']) - 1} env steps)")
        if args.out and ok:
            f_chains.write(json.dumps(chain) + "\n")
            for row in result["rows"]:
                f_steps.write(json.dumps(row) + "\n")
            n_steps += len(result["rows"])
    print(f"\nExpert + oracle planner: {n_ok}/{args.episodes} chains completed. By length: "
          + ", ".join(f"{k}: {sum(v)}/{len(v)}" for k, v in sorted(by_len.items())))
    if args.out:
        f_chains.close()
        f_steps.close()
        print(f"Wrote {n_ok} chains and {n_steps} fast-path steps to {args.out}")


if __name__ == "__main__":
    main()
