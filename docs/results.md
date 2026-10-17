# Results Log

One entry per run: date, command, settings, numbers, takeaway. Newest at the bottom.

## 2026-10-17 — Meta-World expert check

`python src/check_metaworld.py --episodes 30 --seed 0` (Meta-World 3.1.1, CPU, ~10 s total)

| task | expert | discretized expert | mean steps (expert → discretized) |
|---|---|---|---|
| reach-v3 | 30/30 | 30/30 | 51.7 → 57.3 |
| push-v3 | 30/30 | 30/30 | 62.7 → 115.3 |
| pick-place-v3 | 30/30 | 30/30 | 53.8 → 73.8 |

Takeaway: discretization settings (`DEAD_ZONE=0.05`, `MOVE_SIZE=0.5`, gripper open/closed → 0.0/1.0) lose no successes. Discretized push takes ~2× the steps, so push episodes will be longer in the dataset.

## 2026-10-18 — Qwen check (Qwen3-0.6B)

`python src/check_qwen.py --model Qwen/Qwen3-0.6B` (RTX PRO 6000 Blackwell 96 GB, bf16, batch 1, 80-token prompt, incl. tokenization)

| layers | 4 | 8 | 12 | 16 | 20 | 24 | 28 |
|---|---|---|---|---|---|---|---|
| ms/decision | 1.80 | 3.05 | 4.29 | 5.52 | 6.74 | 7.94 | 9.19 |
| vs. full | 20% | 33% | 47% | 60% | 73% | 86% | 100% |

Takeaway: features are 1024-dim; fast-path (first-k) features exactly match cached full-pass features at all layers. Latency is ~linear: ~0.31 ms/layer + ~0.6 ms fixed overhead. At 0.6B / batch 1 the GPU is underused, so re-time on 1.7B and 8B.

## 2026-10-18 — Qwen check (Qwen3-1.7B)

`python src/check_qwen.py --model Qwen/Qwen3-1.7B` (same GPU/settings; hidden size 2048)

| layers | 4 | 8 | 12 | 16 | 20 | 24 | 28 |
|---|---|---|---|---|---|---|---|
| ms/decision | 1.83 | 3.13 | 4.38 | 5.66 | 6.88 | 8.15 | 9.42 |

Takeaway: nearly identical to 0.6B despite ~3× the parameters. At batch 1 on this GPU, time is dominated by the per-layer cost of launching GPU kernels, not arithmetic. Wall-clock savings scale with layers skipped; FLOPs is the metric that will show model-size differences.

## 2026-10-18 — Dataset v1

`python src/data_gen.py --episodes 200 --seed 0` (~25 s on CPU) → `outputs/data/` (30 MB)

| task | episodes kept | steps | steps/episode | classes used | most common class |
|---|---|---|---|---|---|
| reach-v3 | 200/200 | 11,099 | 55 | 9/54 | 32 (66%) |
| push-v3 | 200/200 | 22,985 | 115 | 30/54 | 25 (25%) |
| pick-place-v3 | 194/200 | 14,138 | 73 | 34/54 | 53 (16%) |

Subtasks: push = above puck 13% / lower 50% / push 37%; pick-place = above 21% / lower 21% / grasp 6% / carry 53%.

Takeaways:
- Discretized expert fails 6/200 pick-place episodes (plain expert 200/200), so the pick-place ceiling is ~97%. The 30-episode check missed this. Failed episodes are dropped from the data.
- Reach is dominated by one class (32 = +y only, 66%), so its majority baseline is high. Compare per-task accuracy to per-task majority baselines.
- Push spends half its steps in "lower onto the puck" (zig-zagging from fixed-size moves).
