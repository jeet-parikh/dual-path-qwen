"""Check that Qwen loads, features have the expected shapes, and time first-k layers vs. full depth.

Usage: python src/check_qwen.py [--model Qwen/Qwen3-0.6B] [--reps 50]
Needs a GPU.
"""

import argparse
import time
import warnings

import torch

from common import get_expert, make_env, obs_to_prompt
from qwen import DEFAULT_LAYERS, fast_features, layer_features, load_qwen, num_layers

warnings.filterwarnings("ignore")


def time_ms(fn, reps):
    for _ in range(3):  # warm-up: the first GPU calls are slow
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--reps", type=int, default=50)
    args = parser.parse_args()

    tok, model = load_qwen(args.model)
    n = num_layers(model)
    print(f"{args.model}: {n} layers, hidden size {model.config.hidden_size}, "
          f"GPU {torch.cuda.get_device_name(0)}")

    # A real prompt from a push-v3 episode.
    env = make_env("push-v3", seed=0)
    obs, _ = env.reset()
    prompt = obs_to_prompt(obs, "push-v3")
    print(f"\nPrompt ({len(tok(prompt).input_ids)} tokens):\n{prompt}\n")

    # 1. Feature shapes from one full forward pass.
    layers = [k for k in DEFAULT_LAYERS if k <= n]
    feats = layer_features(model, tok, [prompt], layers)
    for k in layers:
        print(f"layer {k:>2}: feature shape {tuple(feats[k].shape)}")

    # 2. Running only the first k layers gives the same features as the full pass.
    for k in layers:
        diff = (fast_features(model, tok, [prompt], k) - feats[k]).abs().max().item()
        assert diff == 0.0, f"layer {k}: fast path differs from cached features by {diff}"
    print("Fast-path features match full-pass features at every layer.")

    # 3. Latency per decision (batch of 1, like the robot loop), including tokenization.
    print(f"\n{'layers':>6}{'ms/decision':>13}{'vs full':>9}")
    full_ms = time_ms(lambda: fast_features(model, tok, [prompt], n), args.reps)
    for k in layers:
        ms = full_ms if k == n else time_ms(lambda: fast_features(model, tok, [prompt], k), args.reps)
        print(f"{k:>6}{ms:>13.2f}{ms / full_ms:>8.0%}")

    # Sanity check that the expert works on this prompt's state too.
    print(f"\nExpert action for this state: {get_expert('push-v3').get_action(obs).round(2)}")


if __name__ == "__main__":
    main()
