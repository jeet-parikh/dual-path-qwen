"""Compute cost of each setup on multi-step instructions: latency and FLOPs per instruction.

1. Measures, one request at a time on the GPU (no batching, like a real robot):
   - fast path: first k layers on a step prompt, for each k
   - planner, direct answer: one forward pass of the full model
   - planner, reasoning: a full generate() call; also counts the tokens it writes
2. Estimates FLOPs per call from parameter and token counts (2 FLOPs per weight per token, plus
   attention; approximate, embeddings excluded).
3. Combines these with the closed-loop runs (env steps and planner calls per instruction) to give
   cost per instruction for each setup, next to its measured success.

Also times the split-model executor (a separate Qwen3-0.6B) for the baseline.

Usage: python src/compute_cost.py [--think-samples 24] [--reps 50]   (needs an idle GPU)
"""

import argparse
import json
import os
import statistics
import time
import warnings

import numpy as np
import torch

from chain import TASK, with_target
from chain_planner import PLACES, all_queries, chain_messages, load_chains
from common import obs_to_prompt
from qwen import fast_features, load_qwen

warnings.filterwarnings("ignore")


def timed(fn, reps):
    """Median seconds of fn() over reps calls (after one warm-up)."""
    fn()
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def flops(model, n_layers, prompt_tokens, new_tokens=0, logits=False):
    """Approximate FLOPs: prefill of prompt_tokens, then new_tokens decoded one at a time."""
    cfg = model.config
    p_layer = sum(p.numel() for p in model.model.layers[0].parameters())
    p_head = cfg.hidden_size * cfg.vocab_size if logits else 0
    d_attn = cfg.num_attention_heads * cfg.head_dim
    # Weights: 2 per token per parameter. Attention: ~2 * d * (context) per token per layer, twice (QK, AV).
    prefill = 2 * prompt_tokens * (n_layers * p_layer + p_head) + 2 * n_layers * d_attn * prompt_tokens ** 2
    decode = sum(2 * (n_layers * p_layer + p_head) + 4 * n_layers * d_attn * (prompt_tokens + t)
                 for t in range(new_tokens))
    return prefill + decode


def per_chain(path):
    """(env steps, planner calls, success, length) per chain from a chain_rollout.py result file."""
    d = json.load(open(path))
    return [(len(r["path"]) - 1, len(r["calls"]), r["success"], r["length"]) for r in d["results"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24, 28])
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--think-samples", type=int, default=24)
    parser.add_argument("--test", default="outputs/chain_test")
    parser.add_argument("--rollouts", default="outputs/chain_rollouts")
    parser.add_argument("--depth-rollouts", default="outputs/chain_rollouts_depth")
    parser.add_argument("--split-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--split-layers", type=int, nargs="+", default=[8, 12, 16, 20, 24, 28])
    parser.add_argument("--split-rollouts", default="outputs/chain_rollouts_split")
    parser.add_argument("--out", default="outputs/compute_cost.json")
    args = parser.parse_args()

    tok, model = load_qwen(args.model)
    n = len(model.model.layers)
    chains = load_chains(args.test, 50)

    # --- 1. Latency and FLOPs per call -------------------------------------------------------
    step_prompts = []
    for c in chains[:20]:
        obs = np.zeros(39)
        obs[0:3], obs[3] = [0.0, 0.6, 0.2], 1.0
        step_prompts.append(obs_to_prompt(with_target(obs, [c["markers"][0][1], c["markers"][0][2], 0.1]), TASK))
    step_tokens = int(np.median([len(ids) for ids in tok(step_prompts).input_ids]))
    fast = {}
    for k in args.layers:
        sec = timed(lambda: fast_features(model, tok, [step_prompts[0]], k), args.reps)
        fast[k] = {"ms": sec * 1000, "gflops": flops(model, k, step_tokens) / 1e9}
        print(f"fast path k={k:>2}: {sec * 1000:6.2f} ms, {fast[k]['gflops']:6.2f} GFLOPs per step")

    queries = all_queries(chains)
    rng = np.random.default_rng(0)
    sample = [queries[i] for i in rng.choice(len(queries), size=args.think_samples, replace=False)]
    texts = {t: [tok.apply_chat_template(chain_messages(*q), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=t) for q in sample] for t in (False, True)}
    plan_tokens = int(np.median([len(ids) for ids in tok(texts[True]).input_ids]))

    direct_prompt = texts[False][0] + "The robot should go to the"
    direct_tokens = len(tok(direct_prompt).input_ids)
    sec = timed(lambda: model(**tok([direct_prompt], return_tensors="pt").to(model.device)), args.reps)
    direct = {"ms": sec * 1000, "gflops": flops(model, n, direct_tokens, logits=True) / 1e9}
    print(f"planner direct:     {direct['ms']:6.2f} ms, {direct['gflops']:6.2f} GFLOPs per call")

    think_secs, think_new = [], []
    for text in texts[True]:
        batch = tok([text], return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**batch, max_new_tokens=3072, do_sample=False)
        torch.cuda.synchronize()
        think_secs.append(time.perf_counter() - t0)
        think_new.append(out.shape[1] - batch.input_ids.shape[1])
    new_tokens = int(np.mean(think_new))
    think = {"ms": float(np.mean(think_secs)) * 1000, "new_tokens_mean": new_tokens,
             "new_tokens_median": int(np.median(think_new)),
             "gflops": flops(model, n, plan_tokens, new_tokens, logits=True) / 1e9}
    print(f"planner reasoning:  {think['ms'] / 1000:6.1f} s, {new_tokens} tokens written, "
          f"{think['gflops']:6.0f} GFLOPs per call (mean of {len(sample)})")

    # --- 2. Cost per instruction for each setup ------------------------------------------------
    def setup(name, rows, step_cost, call_cost, note="", measured=True):
        steps = np.mean([r[0] for r in rows])
        calls = np.mean([r[1] for r in rows])
        sec = (steps * step_cost["ms"] + calls * call_cost["ms"]) / 1000
        gf = steps * step_cost["gflops"] + calls * call_cost["gflops"]
        succ = {L: float(np.mean([r[2] for r in rows if r[3] == L])) if measured else None for L in (1, 2, 3, 4)}
        return {"setup": name, "env_steps": float(steps), "planner_calls": float(calls), "seconds": float(sec),
                "gflops": float(gf), "planner_share_of_time": float(calls * call_cost["ms"] / 1000 / sec),
                "success": float(np.mean([r[2] for r in rows])) if measured else None, "success_by_length": succ,
                "note": note}

    none = {"ms": 0.0, "gflops": 0.0}
    think_rows = per_chain(os.path.join(args.rollouts, "think_Qwen3-1.7B__fast_k16.json"))
    direct_rows = per_chain(os.path.join(args.rollouts, "direct_Qwen3-1.7B__fast_k16.json"))
    setups = [
        setup("reasoning planner + fast path (k=16)  [ours]", think_rows, fast[16], think),
        setup("direct planner + fast path (k=16)", direct_rows, fast[16], direct),
        # Same instructions and plans, but the full model picks every move.
        setup("reasoning planner + full-depth executor (k=28)", think_rows, fast[28], think,
              "cost only: env steps and calls from the k=16 run", measured=False),
        # Reason before every single move (no fast path).
        setup("reasoning planner at every step", [(s, s, ok, L) for s, _, ok, L in think_rows], none, think,
              "cost only: one reasoning call per env step", measured=False),
    ]
    depth = []
    for k in args.layers:
        path = os.path.join(args.depth_rollouts if k != 16 else args.rollouts, f"oracle__fast_k{k}.json")
        if os.path.exists(path):
            r = setup(f"oracle plan + 1.7B layers 1-{k}", per_chain(path), fast[k], none, "execution only")
            depth.append({"model": "1.7B", "k": k, **r})

    # Split-model baseline: a separate small model as executor (the planner would be the same 1.7B).
    del model
    torch.cuda.empty_cache()
    tok_s, small = load_qwen(args.split_model)
    small_fast = {}
    for k in args.split_layers:
        path = os.path.join(args.split_rollouts, f"oracle__fast_k{k}.json")
        if not os.path.exists(path):
            continue
        sec = timed(lambda: fast_features(small, tok_s, [step_prompts[0]], k), args.reps)
        small_fast[k] = {"ms": sec * 1000, "gflops": flops(small, k, step_tokens) / 1e9}
        r = setup(f"oracle plan + 0.6B layers 1-{k}", per_chain(path), small_fast[k], none, "execution only")
        depth.append({"model": "0.6B", "k": k, **r})

    print(f"\n{'setup':<50}{'success':>8}{'4-step':>8}{'env steps':>10}{'calls':>7}{'seconds':>9}{'TFLOPs':>8}{'planner %':>10}")
    for s in setups + depth:
        pct = lambda v: f"{v:>8.0%}" if v is not None else f"{'-':>8}"
        print(f"{s['setup']:<50}{pct(s['success'])}{pct(s['success_by_length'][4])}{s['env_steps']:>10.0f}"
              f"{s['planner_calls']:>7.1f}{s['seconds']:>9.1f}{s['gflops'] / 1000:>8.1f}{s['planner_share_of_time']:>10.0%}")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "step_prompt_tokens": step_tokens, "planner_prompt_tokens": plan_tokens,
                   "fast_path": fast, "planner_direct": direct, "planner_reasoning": think,
                   "think_new_tokens": think_new, "split_fast_path": small_fast, "setups": setups,
                   "depth": depth}, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
