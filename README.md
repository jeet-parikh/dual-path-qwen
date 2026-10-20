# Dual Path Qwen

CPSC 4880 final project. One frozen Qwen3 model controls a robot arm in Meta-World, with robot state given as text:

- **Fast path:** the first 16 layers + a small trained head pick one of 54 discrete moves each step.
- **Slow path:** the full model reasons step by step to choose the next subtask or target.
- **Router:** decides when to call the slow path.

**Current result:** on 1–4 step instructions ("visit the marker closest to blue, then go back to the first marker you touched, ..."), the reasoning planner + fast path completes 64% of 4-step instructions vs. 36% for a bag-of-words planner and 2% for Qwen answering directly (`figures/chain_success.png`). Details: `docs/results.md`.

## Code (`src/`, run from the repo root, e.g. `python src/chain.py --help`)

| What | Scripts |
| --- | --- |
| Shared prompt/action code, model loading | `common.py`, `qwen.py` |
| Expert data, hidden-state cache, per-layer heads | `data_gen.py`, `dagger.py`, `extract_hidden.py`, `train_heads.py` |
| Closed loop on reach / push / pick-place, subtask planner, staleness router | `rollout.py`, `planner.py`, `stale_router.py` |
| Multi-step instructions: generator, planners, closed loop, figures | `chain.py`, `chain_planner.py`, `chain_rollout.py`, `plot_chain.py` |
| Setup checks and plots | `check_metaworld.py`, `check_qwen.py`, `plot_heads.py` |

Requires `torch`, `transformers`, `metaworld==3.1.1`, `scikit-learn`, `matplotlib`; a GPU for anything that runs Qwen. Outputs go to `outputs/` (not committed).
