# Dual Path Qwen

CPSC 4880 final project. One frozen Qwen3 model controls a robot arm in Meta-World (reach, push, pick-place):

- **Fast path:** the first k layers + a small trained head pick one of 54 discrete actions each step.
- **Slow path:** the full model reasons about which subtask is active.
- **Router:** decides when to call the slow path (confidence router, learned staleness router).

Robot state is given to Qwen as a short text prompt. Qwen is never fine-tuned.

## Pipeline (`src/`)

| Step | Script |
| --- | --- |
| Generate expert data | `data_gen.py` (+ `dagger.py` for recovery data) |
| Cache hidden states per layer | `extract_hidden.py` |
| Train action heads per layer | `train_heads.py`, `plot_heads.py` |
| Closed-loop rollouts | `rollout.py` |
| Planner (full model) | `planner.py` |
| Staleness router | `stale_router.py` |

Shared prompt format and action conversion: `common.py`. Results so far: `docs/results.md`.

## Setup

Python env `dpq` with `torch`, `transformers`, `metaworld`. Run scripts from the repo root, e.g. `python src/rollout.py --help`.
