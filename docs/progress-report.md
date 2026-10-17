# Dual Path Qwen — Progress Report (draft, 2026-10-18)

## Goal

One frozen, general-purpose LLM (Qwen3) acts as both a **fast executor** (first k layers + a small trained head picking one of 54 discrete robot actions every step) and a **slow planner** (full depth, reasoning about which subtask to pursue), with a **router** deciding when to use the slow path. Benchmark: Meta-World reach / push / pick-place, with state given as text.

## What we built

- Scripted-expert data pipeline: 1,472 episodes, one unique layout each, 119k labeled steps (+78k DAgger steps).
- Feature cache: frozen Qwen3-0.6B / 1.7B hidden states at layers 4–28.
- Heads per layer, a closed-loop rollout harness, a thinking-mode planner, and a planner-in-the-loop controller with a confidence router. The same model plans (full depth) and acts (first k layers).

## Key results so far

1. **Shallow layers suffice for reactive control, and the final layer is never best.** On unseen layouts, action accuracy peaks at layers 12–20 (~72%; 77% with subtask text). Closed-loop success: layer 16 beats the full model (72% vs. 62%) at 61% of the cost per decision (5.7 vs. 9.4 ms).
2. **System 1 / system 2 split, measured.** Asked which subtask is active, the full model answering directly is at chance (25–33%). With step-by-step reasoning it reaches 86–93%, at ~100× the cost. Reactive actions need ms-level decisions; subtask decisions need reasoning. This is why a fast path plus an occasional slow path is the right design.
3. **The planner's output helps the executor.** Adding the subtask to the fast path's prompt raises action accuracy by 5 points (push: +9).
4. **DAgger closes the gap for the cheapest executor.** Recovery data lifts layer-4 success from 48% to 63% (push 30 → 50%) at 2.1 ms per decision.
5. **Fast-head confidence predicts errors.** Below 0.4 top probability, the head is wrong 96% of the time vs. 68% otherwise, so confidence works as a router signal.
6. **Confidence routers miss stale plans; a learned staleness router fixes it.** In the loop, triggering the planner on low fast-head confidence fails completely on push (0/10): after a correct switch to "lower onto the puck", the executor stays *confident* while following the outdated subtask. A tiny probe on the fast path's own layer-16 features detects stale subtasks with ~95% accuracy offline (93% caught, 4% false alarms) at zero extra inference cost. In the loop (push, 20 episodes each), it matches planning every 25 steps (9/20 vs. 8/20) with **~45% fewer planner calls** (8.2 vs. 14.9 per episode); a hybrid with a planner call every 50 steps as a safety net reaches 10/20, close to the oracle-subtask reference (53%). Planning is no longer the bottleneck; the executor is.

## Honest negatives (also findings)

- **An MLP reading the numbers directly beats the LLM fast path** (76% vs. 72% on the same rounded numbers; 97% on full precision). LLM hidden states don't support precise coordinate arithmetic. For purely numeric control a specialist wins; the LLM's value is in the integrated planner + executor.
- A data bug (Meta-World reuses 50 layouts per seed) first inflated accuracy to ~90%. Fixed; all numbers above use unique layouts.
- Side branch: language-specified goals. Small encoders / bag-of-words matched or beat Qwen on held-out phrasings, so not pursued.

## Next

- Router comparison at scale (more episodes; hybrid staleness + periodic fallback).
- Baselines: entropy-based early exit; split models (Qwen3-0.6B executor + 1.7B/8B planner).
- Resume-from-layer-k escalation; FLOPs accounting.
- Chained multi-step tasks so the planner matters more; scale to Qwen3-8B.
- More episodes per cell (≥ 50) for tighter success estimates.
