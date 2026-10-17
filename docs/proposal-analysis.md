# Proposal → Plan Analysis

How the original proposal ([`proposal.md`](proposal.md), verbatim) maps onto the current plan ([`project-overview.md`](project-overview.md)). Written 2026-10-17. The overview is the source of truth for locked decisions; this file records what the proposal promised, the risks, and recommended additions.

## 1. What the proposal commits to

**Three research questions** (from its Related Work section). The final report should answer each directly:

| # | Question | How we answer it |
|---|---|---|
| Q1 | How reliable are the early (layer-k) actions? | Offline action accuracy vs. layer depth; closed-loop task success vs. depth |
| Q2 | Does training the head affect the final output? | No, by construction: Qwen is frozen. One sentence in the report. Optional extension: LoRA-tune the first k layers and measure how much the full model's output drifts |
| Q3 | How much faster is stopping early? | Latency and FLOPs vs. depth; fast/slow usage ratio once a router is in |

**Components promised:**
- Fast path: first k layers (proposal says 8–12 of an 8B model), frozen, plus a trained classifier head over a constrained action space.
- Slow path: full depth, used for long-horizon reasoning / task decomposition (a planner), not just a deeper classifier.
- Router, two versions compared: (1) a learned lightweight router "before the first layer"; (2) escalate to full depth when the head's probabilities have no clear winner.
- Stretch: efficient inference techniques.

**Evaluation promised:**
- Benchmarks: LIBERO and Meta-World. LIBERO was dropped (needs images + a VLM); the report must justify this.
- Metrics: inference latency, task success rate, FLOPs, slow/fast usage ratio, fast-path probability distributions per task/domain.
- Baselines: existing early-exit methods (DeeBERT/LayerSkip style); split models (separate fast and slow models, not one combined model).
- Ablations: fast-path depth k, routing mechanism.
- Model: Qwen 8B, possibly several sizes. We develop on 1.7B/0.6B, so headline results should be on 8B (or the choice justified).
- Desired outcome: similar task performance with far less compute.

**Motivating example** (useful for the report intro): "clean the dishes". The planner splits it into subtasks; the fast module picks low-level actions per subtask; the planner is polled periodically to check progress and redirect.

## 2. Phased approach

1. **Week 1: fast path works end to end.** The current MVP plan in the overview (Section 4). Answers Q1 and Q3 at small scale.
2. **Week 2: make closed-loop success decent.** Closed-loop success will likely be poor due to covariate shift. Fix with **DAgger**: let the trained head drive the robot, label every visited state with the scripted expert's action, add these to the dataset, retrain, and repeat. Cheap because the expert can be queried anywhere. Also run the early-exit baseline.
3. **Week 3: slow path as a real planner.** The scripted experts already have explicit stages (e.g. the push expert's `_desired_pos`: move above the puck → lower onto it → push to the goal). These give **free ground-truth subtask labels**. The full model generates/advances subtasks; the fast head sees the current subtask in its prompt; planner decisions can be scored against the true stage at every step. Also: both routers, split baseline, resume-from-layer-k.
4. **Week 4: scale to 8B,** latency/FLOPs across everything, write-up.

If time runs short, protect the planner loop and the two-router comparison; these are what make the project distinct from plain early exit.

## 3. Risks to plan around

**R1. "Why use an LLM at all?"** The expert's actions are simple threshold rules on numbers (sign of goal − hand, etc.). A tiny MLP on the raw 39-number observation will probably reach near-perfect accuracy, and graders will ask about it. Responses:
- Add a raw-state MLP baseline. It gives the best accuracy any head could reach.
- Pitch what the LLM adds: one model for many tasks, tasks/subtasks specified in language, a planner whose text output feeds the fast path directly.
- Train one shared head across all three tasks so the task sentence matters.

**R2. The split-model baseline may win on raw compute.** The first 8 of Qwen3-8B's 36 layers cost roughly as much as running all of Qwen3-0.6B (~1.8B vs. ~0.6B parameters' worth of compute). The single-model advantages to measure, not assume:
- Resume-from-layer-k on escalation (no wasted fast-path compute).
- One set of weights in GPU memory instead of two.
- Shared representation between planner and fast path.

**R3. The week-1 router is just early exit.** "Use the full-depth head when the fast head is unsure" is essentially DeeBERT. Fine for the MVP, but novelty comes once the slow path does planning (Phase 3).

**Note on the "router before layer 1":** at that point it sees only token embeddings, which barely encode the numeric values. Expect it to underperform the confidence-based router; that is a legitimate finding. The overview's fallback (layer 1–2 features) is sensible.

## 4. Recommended additions (proposed, not yet agreed)

Not yet locked into the overview. Move each into `project-overview.md` once agreed.

- [x] Raw-state MLP baseline in `train_heads.py` (week 1). **Done 2026-10-18**: full-precision state + task id → 97.7%; plus an MLP on the prompt's rounded numbers (86.7%) and a prompt-text oracle ceiling (94.4%).
- [x] One shared head across all three tasks, alongside or instead of per-task heads. **Done 2026-10-18** (shared head only).
- [ ] DAgger loop in week 2.
- [x] Subtask labels taken from the expert policies' stages, recorded during data generation. **Agreed and done 2026-10-18** (`subtask` field).
- [x] A dated results log, `docs/results.md`: command, settings, and number for every run. **Done.**
- [ ] Optional: LoRA ablation for Q2.

## 4b. Novelty check and pivot (2026-10-18)

Prior work found in a quick search: [DeeR-VLA](https://arxiv.org/abs/2411.02359) (NeurIPS 2024; multi-exit MLLM for robot control, 5–6× compute savings on CALVIN), [FiS-VLA](https://arxiv.org/pdf/2506.01953) (VLM's final blocks repurposed as a fast controller, trained jointly), and UniFS (cited in the proposal). All are end-to-end trained VLAs, so "early exit for control" and "fast+slow in one model" are not new by themselves.

R1 materialized: on numbers-only prompts an MLP beats Qwen. Pivot to language-specified goals (overview Section 3b). Novelty claim to test: frozen LLM early layers understand *unseen* instructions better than a dedicated small text encoder, at a fraction of full-model cost, with the full model as planner sharing compute. The decisive baseline is a small text encoder (MiniLM) + raw numbers, i.e. the proposal's split-model baseline.

## 5. Reference baselines for every result

A number means little without its floor and ceiling:
- Floor: majority-class baseline.
- Head ceiling: raw-state MLP.
- Policy ceiling: discretized expert (30/30 on all three tasks).
- Depth comparison: full-depth (final-layer) head.
