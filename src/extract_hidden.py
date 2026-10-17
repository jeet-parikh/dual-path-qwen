"""Run every prompt in the dataset through frozen Qwen once and cache layer-k features.

Outputs in <out>/ (row i of every file is the same dataset step):
  layer<k>.npy  float16 [N, hidden]  last-token features at layer k (see qwen.py)
  meta.npz      action_class, episode_id, task, subtask, step, obs
  args.json

Batches hold prompts of equal token length (bucketed), so there is no padding.
With --subtask, prompts are rebuilt from each row's obs with the expert's subtask line added.

Usage: python src/extract_hidden.py [--model Qwen/Qwen3-0.6B] [--limit 200] [--out outputs/hidden/<model>]
Needs a GPU.
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import torch

from common import TASKS, obs_to_prompt
from qwen import DEFAULT_LAYERS, layer_features_bucketed, load_qwen, num_layers

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data", default="outputs/data")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="only the first N steps per task (for testing)")
    parser.add_argument("--subtask", action="store_true", help="add the expert's subtask line to each prompt")
    parser.add_argument("--out", default=None, help="default: outputs/hidden/<model name>[-subtask]")
    args = parser.parse_args()
    args.out = args.out or os.path.join("outputs/hidden", args.model.split("/")[-1] + ("-subtask" if args.subtask else ""))

    tok, model = load_qwen(args.model)
    assert max(args.layers) <= num_layers(model), f"model has only {num_layers(model)} layers"

    rows = []
    for task in args.tasks:
        with open(os.path.join(args.data, f"{task}.jsonl")) as f:
            task_rows = [json.loads(line) for line in f]
        rows += task_rows[: args.limit]

    feats = {k: [] for k in args.layers}
    start = time.time()
    for task in args.tasks:
        task_rows = [r for r in rows if r["task"] == task]
        prompts = [obs_to_prompt(r["obs"], task, r["subtask"]) if args.subtask else r["prompt"] for r in task_rows]
        out = layer_features_bucketed(model, tok, prompts, args.layers, args.batch_size)
        for k in args.layers:
            feats[k].append(out[k].half().cpu().numpy())
        print(f"{task}: {len(prompts)} prompts done ({time.time() - start:.0f}s)")

    os.makedirs(args.out, exist_ok=True)
    for k in args.layers:
        np.save(os.path.join(args.out, f"layer{k}.npy"), np.concatenate(feats[k]))
    np.savez(
        os.path.join(args.out, "meta.npz"),
        action_class=np.array([r["action_class"] for r in rows]),
        episode_id=np.array([r["episode_id"] for r in rows]),
        task=np.array([r["task"] for r in rows]),
        subtask=np.array([r["subtask"] for r in rows]),
        step=np.array([r["step"] for r in rows]),
        obs=np.array([r["obs"] for r in rows], dtype=np.float32),
    )
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    size_gb = sum(os.path.getsize(os.path.join(args.out, f"layer{k}.npy")) for k in args.layers) / 1e9
    print(f"\nSaved {len(rows)} rows x {len(args.layers)} layers "
          f"(hidden {model.config.hidden_size}, {size_gb:.2f} GB) to {args.out} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
