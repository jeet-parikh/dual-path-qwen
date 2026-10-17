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
