"""Language-specified goals: markers, instructions, target math, prompts, and splits.

The target is never shown to the model. The prompt lists colored (text-only) markers, and an
instruction describes the target relative to them. Our code computes the true target and hands
it to the unchanged scripted expert by overwriting obs[36:39].

Frame convention (fixed, never varied): left = -x, right = +x, behind = +y (away from the
robot), in front of = -y (toward the robot).

Splits (by instruction, not by episode):
  train / test_iid   seen sentence frames, phrases, and combinations
  test_paraphrase    at least one held-out piece of wording (frame, direction phrase, or
                     distance written as a number word)
  test_compose       seen wording, but a combination never seen in training

Run `python src/lang.py` for a self-test.
"""

import numpy as np

from common import TASKS

COLORS = ["red", "blue", "green", "yellow"]
DIRECTIONS = {"left": (-1, 0), "right": (1, 0), "front": (0, -1), "behind": (0, 1)}
DISTANCES_CM = [5, 10, 15]
NUMBER_WORDS = {5: "five", 10: "ten", 15: "fifteen"}

# Sentence frames per family: (train frames, held-out paraphrase frames).
FRAMES = {
    "direct": (
        ["go to the {c} marker", "move the gripper to the {c} marker", "reach the {c} marker"],
        ["bring your hand over to the {c} one", "head for the marker colored {c}"],
    ),
    "offset": (
        ["move {d} {r} the {c} marker", "go {d} {r} the {c} marker", "position the gripper {d} {r} the {c} marker"],
        ["I want the hand {d} {r} the {c} marker", "end up {d} {r} the {c} one"],
    ),
    "between": (
        ["go halfway between the {c} and {c2} markers", "move to the midpoint of the {c} and {c2} markers"],
        ["split the difference between the {c} marker and the {c2} marker",
         "go to the spot centered between {c} and {c2}"],
    ),
}
# Direction phrases: (train, held-out).
DIRECTION_PHRASES = {
    "left": (["to the left of", "left of"], ["on the left-hand side of"]),
    "right": (["to the right of", "right of"], ["on the right-hand side of"]),
    "front": (["in front of"], ["nearer to the robot than"]),
    "behind": (["behind"], ["farther from the robot than"]),
}
# Combinations never used in train (only in test_compose).
HELD_OUT_COMBOS = [
    ("offset", "left", "green"),   # direction x color
    ("offset", "behind", 15),      # direction x distance
    ("between", "red", "yellow"),  # color pair (either order)
]

MARKER_REGION = (np.array([-0.25, 0.50]), np.array([0.25, 0.85]))
MIN_MARKER_GAP = 0.08
REACH_Z = 0.10
TARGET_REGION = {  # xy box the target must fall in (from the expert feasibility sweep)
    "reach-v3": (np.array([-0.35, 0.45]), np.array([0.35, 0.90])),
    "push-v3": (np.array([-0.25, 0.50]), np.array([0.25, 0.85])),
}
SUCCESS_RADIUS = 0.05


def is_held_out_combo(spec):
    for combo in HELD_OUT_COMBOS:
        if spec["family"] != combo[0]:
            continue
        if combo[0] == "offset" and combo[1] == spec["direction"] and combo[2] in (spec["color"], spec["cm"]):
            return True
        if combo[0] == "between" and {combo[1], combo[2]} == {spec["color"], spec["color2"]}:
            return True
    return False


def sample_markers(rng):
    """Three distinct colors at random table positions, at least MIN_MARKER_GAP apart, shuffled order."""
    colors = list(rng.choice(COLORS, size=3, replace=False))
    while True:
        xy = rng.uniform(*MARKER_REGION, size=(3, 2))
        gaps = [np.linalg.norm(xy[i] - xy[j]) for i in range(3) for j in range(i + 1, 3)]
        if min(gaps) >= MIN_MARKER_GAP:
            return [(c, float(x), float(y)) for c, (x, y) in zip(colors, xy)]


def sample_spec(rng, markers, split):
    """Sample what the instruction means (family, colors, direction, distance) and its wording choices."""
    colors = [m[0] for m in markers]
    family = rng.choice(list(FRAMES))
    spec = {"family": str(family), "color": str(rng.choice(colors))}
    if family == "offset":
        spec["direction"] = str(rng.choice(list(DIRECTIONS)))
        spec["cm"] = int(rng.choice(DISTANCES_CM))
    if family == "between":
        spec["color2"] = str(rng.choice([c for c in colors if c != spec["color"]]))

    # Wording: test_paraphrase must use at least one held-out piece; other splits use none.
    pieces = ["frame"] + (["direction", "distance"] if family == "offset" else [])
    held = set()
    if split == "test_paraphrase":
        held = {p for p in pieces if rng.random() < 0.5} or {str(rng.choice(pieces))}
    frames = FRAMES[family][1 if "frame" in held else 0]
    spec["frame"] = str(rng.choice(frames))
    if family == "offset":
        phrases = DIRECTION_PHRASES[spec["direction"]][1 if "direction" in held else 0]
        spec["direction_phrase"] = str(rng.choice(phrases))
        spec["distance_text"] = (f"{NUMBER_WORDS[spec['cm']]} centimeters" if "distance" in held
                                 else f"{spec['cm']} cm")
    spec["held_out_wording"] = sorted(held)
    return spec


def instruction_text(spec):
    return spec["frame"].format(c=spec["color"], c2=spec.get("color2", ""), d=spec.get("distance_text", ""),
                                r=spec.get("direction_phrase", ""))


def target_xy(spec, markers):
    pos = {c: np.array([x, y]) for c, x, y in markers}
    if spec["family"] == "direct":
        return pos[spec["color"]]
    if spec["family"] == "offset":
        return pos[spec["color"]] + np.array(DIRECTIONS[spec["direction"]]) * spec["cm"] / 100
    if spec["family"] == "between":
        return (pos[spec["color"]] + pos[spec["color2"]]) / 2
    raise ValueError(spec["family"])


def sample_episode(rng, task, split, hand_xy, puck_xy):
    """Sample markers + instruction for one episode such that the target is reachable.

    Returns dict with markers, instruction, spec, target (xyz).
    """
    lo, hi = TARGET_REGION[task]
    for _ in range(1000):
        markers = sample_markers(rng)
        spec = sample_spec(rng, markers, split)
        if is_held_out_combo(spec) != (split == "test_compose"):
            continue
        xy = target_xy(spec, markers)
        if not (np.all(xy >= lo) and np.all(xy <= hi)):
            continue
        if task == "reach-v3" and np.linalg.norm(xy - hand_xy) < 0.07:
            continue
        if task == "push-v3" and not 0.10 <= np.linalg.norm(xy - puck_xy) <= 0.25:
            continue
        z = REACH_Z if task == "reach-v3" else 0.02
        return {"markers": markers, "spec": spec, "instruction": instruction_text(spec),
                "target": [float(xy[0]), float(xy[1]), z]}
    raise RuntimeError("could not sample a valid episode")


def _fmt(values):
    return "(" + ", ".join(f"{round(float(v), 2) + 0.0:.2f}" for v in values) + ")"


def lang_prompt(obs, instruction, markers):
    """Prompt for language-specified goals. Static part (instruction, markers) first."""
    obs = np.asarray(obs)
    gripper = round(float(obs[3]), 2) + 0.0
    marker_text = ", ".join(f"{c} {_fmt((x, y))}" for c, x, y in markers)
    return (
        f"Instruction: {instruction}.\n"
        f"Markers: {marker_text}.\n"
        f"Hand: {_fmt(obs[0:3])}. Gripper: {gripper:.2f}.\n"
        f"Object: {_fmt(obs[4:7])}.\n"
        f"Action:"
    )


def with_target(obs, target):
    """Copy of obs with the goal slot replaced by the language target (what the expert reads)."""
    obs = np.array(obs, dtype=np.float64)
    obs[36:39] = target
    return obs


def lang_success(task, obs, target):
    obs = np.asarray(obs)
    pos = obs[0:3] if task == "reach-v3" else obs[4:7]
    return bool(np.linalg.norm(pos - np.asarray(target)) <= SUCCESS_RADIUS)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    markers = [("blue", -0.05, 0.62), ("red", 0.12, 0.70), ("green", 0.20, 0.55)]
    spec = {"family": "offset", "color": "red", "direction": "left", "cm": 10}
    assert np.allclose(target_xy(spec, markers), [0.02, 0.70])          # left = -x
    spec = {"family": "offset", "color": "blue", "direction": "behind", "cm": 5}
    assert np.allclose(target_xy(spec, markers), [-0.05, 0.67])         # behind = +y
    spec = {"family": "between", "color": "blue", "color2": "green"}
    assert np.allclose(target_xy(spec, markers), [0.075, 0.585])
    print("Target math OK.")

    rng = np.random.default_rng(0)
    held_frames = {f for fam in FRAMES.values() for f in fam[1]}
    held_phrases = {p for ph in DIRECTION_PHRASES.values() for p in ph[1]}
    for split in ["train", "test_iid", "test_paraphrase", "test_compose"]:
        examples = [sample_episode(rng, "reach-v3", split, np.array([0, 0.6]), np.array([0, 0.65]))
                    for _ in range(500)]
        for e in examples:
            s = e["spec"]
            uses_held = (s["frame"] in held_frames or s.get("direction_phrase") in held_phrases
                         or "centimeters" in s.get("distance_text", ""))
            assert uses_held == (split == "test_paraphrase"), (split, e["instruction"])
            assert is_held_out_combo(s) == (split == "test_compose"), (split, e["instruction"])
        fams = {f: sum(e["spec"]["family"] == f for e in examples) for f in FRAMES}
        print(f"{split:<16} OK  families {fams}  e.g. \"{examples[0]['instruction']}\"")

    e = sample_episode(np.random.default_rng(1), "reach-v3", "test_paraphrase", np.array([0, 0.6]), np.array([0, 0.65]))
    obs = np.zeros(39)
    obs[0:3] = [0, 0.6, 0.2]
    obs[3] = 1.0
    obs[4:7] = [0, 0.65, 0.02]
    print("\nExample prompt:\n" + lang_prompt(obs, e["instruction"], e["markers"]))
    print("target:", np.round(e["target"], 3), "| tasks known:", [t for t in TARGET_REGION if t in TASKS])
