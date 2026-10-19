"""Planners for multi-step instructions (chain.py): which place should the robot go to next?

Every planner gets the same information: the instruction, the marker positions, the hand's
position, and how many steps are done. It answers with one place: red, blue, green, yellow, or
start. It is not told which places were visited, so steps like "the marker closest to the one you
just touched" or "back to the first marker" must be worked out from the instruction.

Planners:
  keyword   first color word in the current sentence ("start" if none). No learning.
  bow       MLP on bag-of-words of the current sentence + marker/hand positions + step index,
            trained on outputs/chain_train. The cheap learned baseline.
  direct    Qwen answering immediately.
  think     Qwen reasoning step by step first (thinking mode).
  oracle    the true place (upper bound).

Offline scoring: for each test chain and step, the hand sits at the previous true place and the
planner names the next one. A chain counts as correct when every step is right, which equals
closed-loop success with a perfect low-level controller.

Usage: python src/chain_planner.py --planner bow [--test outputs/chain_test] [--limit 200]
       python src/chain_planner.py --planner think --model Qwen/Qwen3-1.7B   (needs a GPU)
"""

import argparse
import json
import os
import re
import time
import warnings
from collections import defaultdict

import numpy as np

from chain import COLORS, place_xyz

warnings.filterwarnings("ignore")

PLACES = COLORS + ["start"]
START = np.array([0.0, 0.6, 0.2])


def load_chains(path, limit=None):
    with open(os.path.join(path, "chains.jsonl")) as f:
        chains = [json.loads(line) for line in f]
    return chains[:limit] if limit else chains


def sentences(instruction):
    return [s for s in re.split(r"(?<=\.)\s+", instruction.strip()) if s]


def hand_before(chain, n_done):
    prev = chain["places"][n_done - 1] if n_done else "start"
    return place_xyz(prev, chain["markers"], START)


# ---------------------------------------------------------------------------
# Qwen planner
# ---------------------------------------------------------------------------

def chain_messages(chain, n_done, hand):
    markers = ", ".join(f"{c} ({x:.2f}, {y:.2f})" for c, x, y in chain["markers"])
    n = len(chain["steps"])
    return [{"role": "user", "content": (
        "You are the planner for a robot arm. Positions are (x, y) in meters on a table: larger x is "
        "further right, larger y is farther from the robot.\n"
        f"Markers: {markers}.\n"
        f"The hand started at ({START[0]:.2f}, {START[1]:.2f}) and is now at ({hand[0]:.2f}, {hand[1]:.2f}).\n"
        f"Instruction: {chain['instruction']}\n"
        f"The robot has finished {n_done} of the {n} steps. Where should it go for step {n_done + 1}? "
        "Answer with one word: red, blue, green, yellow, or start.")}]


def parse_place(text):
    text = text.split("</think>")[-1].lower()
    hits = [(m.start(), m.group(0)) for m in re.finditer(r"\b(red|blue|green|yellow|start)\b", text)]
    return hits[-1][1] if hits else None


def qwen_planner(model_name, thinking, batch_size, loaded=None):
    """loaded = (tok, model) to share an already-loaded Qwen with the fast path."""
    import torch
    from qwen import load_qwen

    tok, model = loaded or load_qwen(model_name)
    # Direct mode: the answer is begun for the model ("The robot should go to the") and its next
    # word is read off, restricted to the five places. Asked for one word, it otherwise starts a
    # sentence ("The robot is currently at ...") and names no place within a short budget.
    prefix = "The robot should go to the"
    option_ids = [tok.encode(" " + p, add_special_tokens=False)[0] for p in PLACES]
    assert len(set(option_ids)) == len(PLACES)

    @torch.no_grad()
    def plan(queries):
        """queries: list of (chain, n_done, hand). Returns (answers, raw texts)."""
        answers, texts = [], []
        for b in range(0, len(queries), batch_size):
            prompts = [tok.apply_chat_template(chain_messages(*q), tokenize=False, add_generation_prompt=True,
                                               enable_thinking=thinking) for q in queries[b:b + batch_size]]
            if not thinking:
                prompts = [p + prefix for p in prompts]
            batch = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            if not thinking:
                logits = model(**batch).logits[:, -1, option_ids]
                picks = [PLACES[i] for i in logits.argmax(-1).tolist()]
                answers += picks
                texts += picks
                continue
            out = model.generate(**batch, max_new_tokens=3072, do_sample=False)
            t = tok.batch_decode(out[:, batch.input_ids.shape[1]:], skip_special_tokens=True)
            texts += t
            answers += [parse_place(x) for x in t]
        return answers, texts

    return plan


# ---------------------------------------------------------------------------
# Cheap baselines
# ---------------------------------------------------------------------------

def keyword_planner(queries):
    answers = []
    for chain, n_done, _ in queries:
        sent = sentences(chain["instruction"])[n_done].lower()
        hits = re.findall(r"\b(red|blue|green|yellow)\b", sent)
        answers.append(hits[0] if hits else "start")
    return answers, [""] * len(answers)


def bow_features(queries, vocab):
    rows = []
    for chain, n_done, hand in queries:
        words = re.findall(r"[a-z]+", sentences(chain["instruction"])[n_done].lower())
        bow = np.zeros(len(vocab))
        for w in words:
            if w in vocab:
                bow[vocab[w]] += 1
        pos = {c: (x, y) for c, x, y in chain["markers"]}
        coords = [v for c in COLORS for v in pos[c]]
        step = np.eye(4)[min(n_done, 3)]
        rows.append(np.concatenate([bow, coords, hand[:2], step]))
    return np.array(rows)


def all_queries(chains):
    return [(c, i, hand_before(c, i)) for c in chains for i in range(len(c["steps"]))]


def bow_planner(train_path, extra=20000, seed=0):
    """Trained on the train chains plus `extra` sampled chains (no simulation needed to label them)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    from chain import sample_chain

    chains = load_chains(train_path)
    rng = np.random.default_rng(seed + 1)
    chains += [sample_chain(rng, int(rng.integers(1, 5)), START) for _ in range(extra)]
    train = all_queries(chains)
    words = sorted({w for c, i, _ in train for w in re.findall(r"[a-z]+", sentences(c["instruction"])[i].lower())})
    vocab = {w: j for j, w in enumerate(words)}
    x = bow_features(train, vocab)
    y = [c["places"][i] for c, i, _ in train]
    scaler = StandardScaler().fit(x)
    clf = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=500, early_stopping=True, random_state=seed)
    clf.fit(scaler.transform(x), y)
    print(f"bow: trained on {len(train)} steps, vocab {len(vocab)}, train acc {clf.score(scaler.transform(x), y):.0%}")

    def plan(queries):
        answers = list(clf.predict(scaler.transform(bow_features(queries, vocab))))
        return answers, [""] * len(answers)

    return plan


def oracle_planner(queries):
    return [c["places"][i] for c, i, _ in queries], [""] * len(queries)


def make_planner(args, loaded=None):
    if args.planner == "keyword":
        return keyword_planner
    if args.planner == "bow":
        return bow_planner(args.train)
    if args.planner == "oracle":
        return oracle_planner
    return qwen_planner(args.model, args.planner == "think", args.batch_size, loaded)


# ---------------------------------------------------------------------------
# Offline evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=["keyword", "bow", "direct", "think", "oracle"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train", default="outputs/chain_train")
    parser.add_argument("--test", default="outputs/chain_test")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N test chains")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="outputs/chain_planner")
    args = parser.parse_args()

    chains = load_chains(args.test, args.limit)
    queries = all_queries(chains)
    plan = make_planner(args)
    t0 = time.time()
    answers, texts = plan(queries)
    secs = time.time() - t0

    step_ok, by_family, records = defaultdict(list), defaultdict(list), []
    for (chain, i, _), a, t in zip(queries, answers, texts):
        ok = a == chain["places"][i]
        step_ok[id(chain)].append(ok)
        by_family[chain["steps"][i]["family"]].append(ok)
        records.append({"seed": chain["seed"], "step": i, "truth": chain["places"][i], "answer": a,
                        "family": chain["steps"][i]["family"], "text": t[-1500:]})
    by_length = defaultdict(list)
    for chain in chains:
        by_length[chain["length"]].append(all(step_ok[id(chain)]))

    name = args.planner if args.planner in ("keyword", "bow", "oracle") else \
        f"{args.planner}_{args.model.split('/')[-1]}"
    results = {
        "planner": name, "n_chains": len(chains), "n_steps": len(queries),
        "step_accuracy": float(np.mean([a == c["places"][i] for (c, i, _), a in zip(queries, answers)])),
        "chain_accuracy_by_length": {k: float(np.mean(v)) for k, v in sorted(by_length.items())},
        "n_by_length": {k: len(v) for k, v in sorted(by_length.items())},
        "step_accuracy_by_family": {k: float(np.mean(v)) for k, v in sorted(by_family.items())},
        "sec_per_step_batched": secs / len(queries),
    }
    print(f"\n{name}: step accuracy {results['step_accuracy']:.0%} ({len(queries)} steps, "
          f"{secs / len(queries) * 1000:.0f} ms/step batched)")
    print("  chain accuracy by length: " + ", ".join(f"{k}: {v:.0%}" for k, v in results["chain_accuracy_by_length"].items()))
    print("  step accuracy by family:  " + ", ".join(f"{k} {v:.0%}" for k, v in results["step_accuracy_by_family"].items()))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{name}.json"), "w") as f:
        json.dump({"args": vars(args), "results": results, "records": records}, f, indent=2)
    print(f"Saved {args.out}/{name}.json")


if __name__ == "__main__":
    main()
