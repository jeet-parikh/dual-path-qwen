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
