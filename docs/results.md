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

> **SUPERSEDED (layout leak).** Meta-World MT1 pre-samples only 50 layouts per seed and cycles through them on reset, so these 200 episodes per task were ~49 layouts repeated ~4×. See "Layout leak" below.

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

## 2026-10-18 — Hidden-state extraction

`python src/extract_hidden.py --model Qwen/Qwen3-{0.6B,1.7B}` (interactive on gpu_devel, batch 128, bf16) → `outputs/hidden/<model>/`

| model | hidden size | rows | layers | size | time |
|---|---|---|---|---|---|
| Qwen3-0.6B | 1024 | 48,222 | 4–28 (7) | 0.69 GB | 30 s |
| Qwen3-1.7B | 2048 | 48,222 | 4–28 (7) | 1.38 GB | 57 s |

Notes:
- Prompt length is fixed per task (reach 77, push 80, pick-place 84 tokens), so per-task batches need no padding.
- Batched vs. single-prompt features differ by ~1% (relative L2) from bf16 rounding with different batch sizes, even without padding. Harmless noise for the heads.

## 2026-10-18 — Action heads v1 (INVALID: layout leak)

> **INVALID.** Trained on dataset v1, where train and test episodes shared the same ~49 layouts per task (near-identical deterministic trajectories), so test accuracy is inflated by memorization. First closed-loop test on new layouts failed (reach 0/2, ~0% agreement with the expert). Kept for the record; superseded by v2 below.

`python src/train_heads.py --model Qwen3-{0.6B,1.7B}` then `python src/plot_heads.py` → `figures/`
Split by episode within each task: 33,686 train / 4,793 val / 9,743 test steps. One shared head for all tasks. 200 epochs (AdamW, lr 1e-3), best-val epoch kept. Baseline MLPs: 1000 epochs.

Test accuracy (%), 54 classes:

| layer | 0.6B linear | 0.6B MLP | 1.7B linear | 1.7B MLP |
|---|---|---|---|---|
| 4 | 83.0 | 89.2 | 88.1 | 90.3 |
| 8 | 86.2 | 90.9 | 88.6 | 90.8 |
| 12 | 87.3 | 90.9 | 89.3 | 91.3 |
| 16 | 88.5 | 91.3 | 89.2 | 90.8 |
| 20 | 85.8 | 91.4 | 88.9 | 90.2 |
| 24 | 82.3 | 90.1 | 86.5 | 89.7 |
| 28 | 80.8 | 90.3 | 86.2 | 89.4 |

| baseline | test acc |
|---|---|
| always most common class | 15.8 |
| most common class per task | 32.0 |
| MLP on the prompt's rounded numbers + task id | 86.7 (still slowly improving at 1000 epochs) |
| best possible from prompt text (oracle) | 94.4 |
| MLP on full-precision state + task id | 97.7 |

Takeaways:
- **Layer 4 of 28 is already within ~1 point of the best layer** (MLP heads ~90–91% everywhere). This is the core hypothesis holding offline.
- Middle layers (8–20) are best; the final layer is slightly worse, sharply so for linear heads (0.6B linear: 88.5% at layer 16 → 80.8% at 28). Action-relevant info is most linearly accessible mid-network.
- 0.6B ≈ 1.7B with MLP heads; 1.7B features are more linearly decodable.
- **The prompt format sets the ceiling.** 2-decimal rounding makes identical prompts map to different expert actions: only 8,249 distinct prompts in 48k steps, best possible 94.4% (push 91.5%). Qwen MLP heads train to 94.0%, i.e. at that ceiling. Full-precision state reaches 97.7%. Rounding costs ~3 points.
- Qwen features beat an MLP reading the same rounded numbers (91 vs. 87%), with the caveat that this baseline was still slowly improving.
- Hardest subtask: pick-place "lower onto the puck" (78–82%, ceiling 87%).
- Training budget mattered: 40 epochs understated linear heads by up to 5 points and the state baseline by 15. Undertrained baselines would have flattered Qwen.

## 2026-10-18 — Layout leak found

First rollout (`src/rollout.py`, layer 4, seed 1000) failed on reach (0/2) with 0% expert agreement while the head was 97% confident, despite 96% offline reach accuracy. Live and cached features matched (<1% diff) and predicted correctly on training prompts, so the head was fine. Cause: **Meta-World MT1 envs pre-sample 50 object/goal layouts per seed and cycle through them on `reset()`.** Dataset v1 had only 48–49 distinct layouts per task, so the episode split leaked layouts into test, and seed 1000 meant all-new layouts.

Fix: `episode_env(task, seed)` in `common.py` creates a fresh env per episode (one unique layout per seed). Data uses seeds 0..N-1; rollouts use 10000+. Creating an env costs ~0.4 s.

Expert re-check on unique layouts (`python src/check_metaworld.py --episodes 200`):

| task | expert | discretized expert | mean steps (expert → discretized) |
|---|---|---|---|
| reach-v3 | 200/200 | 200/200 | 48.4 → 55.6 |
| push-v3 | 200/200 | 200/200 | 61.4 → 114.5 |
| pick-place-v3 | 200/200 | 193/200 | 53.3 → 88.4 |

## 2026-10-18 — Dataset v2 + action heads v2 (valid)

Data: `python src/data_gen.py` (500 episodes per task, one unique layout each; ~10 min because each episode builds a fresh env). Kept 500 / 500 / 472 episodes; 119,405 steps (reach 27,773, push 56,902, pick-place 34,730). Features re-extracted (0.6B 1.71 GB in 74 s; 1.7B 3.42 GB in 141 s). Heads: same settings as v1. Split 83,844 train / 12,040 val / 23,521 test steps.

Test accuracy (%):

| layer | 0.6B linear | 0.6B MLP | 1.7B linear | 1.7B MLP |
|---|---|---|---|---|
| 4 | 59.6 | 63.1 | 63.7 | 68.6 |
| 8 | 66.8 | 69.5 | 68.2 | 71.5 |
| 12 | 68.9 | 71.3 | 70.1 | 71.1 |
| 16 | 68.9 | 71.9 | 71.4 | 72.4 |
| 20 | 68.9 | 71.2 | 70.9 | 71.6 |
| 24 | 67.5 | 70.0 | 69.1 | 70.4 |
| 28 | 66.2 | 68.3 | 68.6 | 70.2 |

| baseline | test acc |
|---|---|
| always most common class | 15.3 |
| most common class per task | 31.2 |
| MLP on the prompt's rounded numbers + task id | 76.3 |
| best possible from prompt text (oracle) | 94.6 |
| MLP on full-precision state + task id | 96.7 |

Takeaways:
- On truly unseen layouts the best heads reach ~72% (layer 16), far above guessing (31%) but far below the prompt ceiling (94.6%).
- Middle layers are still best. Early layers now lag more: 1.7B MLP layer 4 = 68.6% vs. 72.4% at 16; 0.6B layer 4 = 63.1%.
- **Qwen features trail a plain MLP on the same rounded numbers (72 vs. 76%).** The v1 advantage was the leak. This is the "why an LLM?" risk materializing.
- Large train/test gap (MLP train 80–90%, test ~70%): heads overfit to layouts. Candidates: more episodes, stronger regularization, mean-pooled features.
- Per task (1.7B MLP, best layer): reach ~91%, push ~65%, pick-place ~69%. Weakest stage: "lower onto the puck" (41–57%, ceiling 88–92%). It needs xy alignment within 0.02, while the prompt resolves 0.01, so it is where rounding hurts most.

## 2026-10-18 — First closed-loop rollouts (end-to-end MVP)

`python src/rollout.py --model Qwen/Qwen3-1.7B --layers 4 16 28 --episodes 20` (MLP heads from v2; env seeds 10000–10019, unseen layouts; same layouts for every layer). ms/decision = tokenize + first-k layers + head, batch 1. "agree" = fraction of visited states where the head's class equals the expert's class at that state.

| layer | task | success | mean steps | ms/decision | agree | mean top prob |
|---|---|---|---|---|---|---|
| 4 | reach | 19/20 | 77 | 2.15 | 48% | 0.94 |
| 4 | push | 4/20 | 456 | 2.03 | 4% | 0.87 |
| 4 | pick-place | 10/20 | 303 | 2.04 | 28% | 0.76 |
| 16 | reach | **20/20** | 56 | 5.74 | 54% | 0.95 |
| 16 | push | **11/20** | 287 | 5.72 | 27% | 0.91 |
| 16 | pick-place | 12/20 | 243 | 5.74 | 39% | 0.87 |
| 28 | reach | 16/20 | 145 | 9.39 | 40% | 0.95 |
| 28 | push | 6/20 | 402 | 9.42 | 15% | 0.84 |
| 28 | pick-place | **15/20** | 192 | 9.40 | 30% | 0.75 |

Totals: layer 4 = 33/60 (55%), layer 16 = 43/60 (72%), layer 28 = 37/60 (62%). Reference ceiling (discretized expert): 100% / 100% / ~97%.

Takeaways:
- **The full loop works.** A frozen Qwen + small head drives the robot to success on unseen layouts, at 2–9 ms per decision.
- Layer 16 (57% of the model's time) beats the full model (28) overall, matching the offline trend. Layer 4 runs at 23% of full-depth time and still solves reach 19/20.
- Push is hardest (4–11/20), consistent with its weak offline "lower onto the puck" accuracy.
- Agreement with the expert in closed loop (4–54%) is far below offline accuracy (~70%), yet success is often decent. Two effects: (1) covariate shift, as the robot drifts into states the expert never visits and errors compound, so failing episodes contribute hundreds of low-agreement steps; (2) many disagreements are harmless (e.g. moving on two axes vs. one).
- Confidence stays high (0.75–0.95) even when the policy is wrong. The top probability alone may be a weak router signal; check per-step calibration when building the router.
- Note: 20 episodes per cell gives roughly ±10–20 points of noise on each success rate.

## 2026-10-18 — M1a: language-goal target probe (go/no-go) → NO-GO as designed

`python src/lang_probe.py` (no simulator; 6,500 step-0 reach scenes: 4,000 train / 500 val / 500 test_iid / 1,000 test_paraphrase / 500 test_compose; MLP regression probe to target xy, best-val epoch). Metric: fraction of targets predicted within 5 cm. Qwen rows show the layer with the best paraphrase score, an optimistic choice for Qwen.

| representation | iid | paraphrase | compose | paraphrase mean err (cm) |
|---|---|---|---|---|
| numbers only (no language) | 14% | 14% | 17% | 13.0 |
| bag-of-words + numbers | 93% | **67%** | **65%** | 4.6 |
| MiniLM + numbers | **96%** | 13% | 57% | 11.6 |
| Qwen3-0.6B full prompt (best L4) | 16% | 7% | 19% | 18.5 |
| Qwen3-0.6B full prompt L20 + numbers | 59% | 20% | 46% | 10.8 |
| Qwen3-0.6B instruction-only L8 + numbers | 90% | 25% | 57% | 9.6 |
| Qwen3-1.7B full prompt (best L16) | 19% | 6% | 24% | 18.0 |
| Qwen3-1.7B full prompt L20 + numbers | 36% | 17% | 32% | 12.4 |
| Qwen3-1.7B instruction-only L4 + numbers | 91% | 18% | 61% | 11.2 |

Instruction-only features *without* numbers: 7–8% everywhere, so there is no leak from workspace rejection sampling.

Takeaways:
- **Go criterion not met.** On held-out paraphrases, Qwen does not beat MiniLM by a clear margin (25 vs. 13% at best), and bag-of-words beats both (67%).
- **Why bag-of-words wins:** the held-out paraphrases still contain the key words (colors, "left"/"right"), and a word-count model ignores unfamiliar frames. Dense embeddings (MiniLM, Qwen) shift on new sentence frames, and a probe trained on only 2–3 frames per family doesn't extrapolate. The test measured keyword overlap more than understanding.
- **Qwen can't do the coordinate arithmetic in its hidden states.** Full-prompt features alone stay at 15–20% even on seen wording (vs. 90%+ when the numbers go to the probe directly). Putting the numbers into the LLM *hurts*: full prompt + numbers < instruction-only + numbers. Design rule: the LLM should handle language, and numbers should go straight to the head.
- Qwen instruction features ≈ MiniLM on seen wording (90–91 vs. 96%).

## 2026-10-18 — Expert feasibility with arbitrary (marker-defined) targets

`python src/check_lang.py --episodes 300` (target overwritten into `obs[36:39]` each step; success = within 5 cm)

| task | expert | discretized expert | mean steps (discretized) |
|---|---|---|---|
| reach-v3 | 300/300 | 300/300 | 30 |
| push-v3 | 286/300 (95.3%) | **300/300** | 99 |

All families and directions ≥ 93% for the plain expert, 100% discretized. Takeaway: the scripted experts can be steered to arbitrary table targets by overwriting the goal slot, so chained multi-step tasks (e.g. push the puck to a spot, then reach a marker) are feasible with free expert data. The discretized push expert was even more reliable than the plain one here.

## 2026-10-18 — Fast path v2: subtask conditioning and direct numbers (Qwen3-1.7B, MLP heads)

`extract_hidden.py --subtask` (prompt gains `Subtask: <expert stage>.`), then `train_heads.py --heads mlp [--features-suffix=-subtask] [--numbers prompt|state]`. Same split and settings as heads v2. Test accuracy (%):

| layer | prompt (v2) | + raw prompt numbers | **+ subtask line** | + subtask + prompt numbers | + subtask + full-precision state |
|---|---|---|---|---|---|
| 4 | 68.6 | 72.3 | 73.1 | 76.8 | 82.1 |
| 8 | 71.5 | 72.4 | 75.3 | 77.6 | 82.7 |
| 12 | 71.1 | 73.1 | **77.4** | 77.7 | 81.2 |
| 16 | 72.4 | 72.5 | 77.2 | 77.7 | 80.4 |
| 20 | 71.6 | 71.6 | 77.3 | 78.0 | 80.9 |
| 24 | 70.4 | 72.5 | 75.3 | 77.6 | 83.9 |
| 28 | 70.2 | 72.1 | 75.1 | 77.4 | 84.6 |

Reference (no LLM): MLP on rounded prompt numbers + task id 76.3%; on full-precision state + task id 96.7%.

Takeaways:
- **Telling the executor its subtask helps: +5 points** at the best layer (72.4 → 77.4%), mostly on push (65.6 → 74.9%, +9). The planner's output has a measurable effect on execution, which is the proposal's core interaction.
- Giving the head the numbers directly **flattens the depth curve.** With subtask + prompt numbers, layer 4 (76.8%) ≈ layer 20 (78.0%). Once numbers bypass the LLM, the executor's LLM features only need to carry task/subtask meaning, and 4 layers suffice.
- Full-precision numbers + Qwen features (80–85%) do *worse* than full-precision numbers alone (96.7%). The 2,048-dim features let the head overfit (train 93–95%). A head design that keeps numbers primary (e.g. a small projection of the LLM features) is needed before claiming anything here.

## 2026-10-18 — Closed loop with oracle subtask (Qwen3-1.7B, MLP heads on subtask prompts)

`python src/rollout.py --subtask oracle --layers 4 16 28 --episodes 20` (same seeds 10000–10019 as the no-subtask run)

| layer | subtask in prompt | reach | push | pick-place | total |
|---|---|---|---|---|---|
| 4 | no | 19 | 4 | 10 | 33/60 |
| 4 | oracle | 13 | 6 | 10 | 29/60 |
| 16 | no | 20 | 11 | 12 | 43/60 |
| 16 | oracle | 15 | 13 | 16 | 44/60 |
| 28 | no | 16 | 6 | 15 | 37/60 |
| 28 | oracle | 16 | 8 | 12 | 36/60 |

Takeaway: no clear closed-loop gain at 20 episodes. Push/pick-place improve at layer 16, but reach (single subtask, so unaffected by subtask text) drops 20 → 15, which shows how much heads retrained on the same data, and 20-episode estimates, vary. Need ≥ 50 episodes per cell to resolve a few-point effect.

## 2026-10-18 — Planner (slow path): subtask decisions by the full model

`python src/planner.py --per-subtask 30 [--thinking]` (Qwen3-1.7B; given the task procedure in plain language + state with 3-decimal coordinates; must name the active subtask; scored against the expert's stage on held-out-episode states, 30 per subtask; batched, batch 32)

| mode | push | pick-place | time per call (batched) |
|---|---|---|---|
| direct answer | 33% (always "move above the puck") | 25% (same) | 6–9 ms |
| **thinking** | **93%** | **86%** | 0.5–0.9 s |

Takeaways:
- **Without reasoning, the full model is at chance;** it always names the first option. **With step-by-step reasoning it reaches 86–93%.** Its thinking computes the horizontal and vertical distances explicitly and applies the procedure.
- This is the system-1/system-2 split in action: the reactive policy needs ms-level decisions, while correct subtask decisions need hundreds of reasoning tokens. So the slow path can only run occasionally (every N steps or when triggered), which is the proposal's design.
- Thinking costs ~100× more than a direct answer, and unbatched calls will be slower still (to measure in the loop).

## 2026-10-18 — DAgger iteration 1 (Qwen3-1.7B, subtask prompts, MLP heads)

`python src/dagger.py --episodes 200` (layer-16 head drives, oracle subtask in prompt, max 300 steps, env seeds 20000+; every visited state labeled with the expert's action and subtask) → merged with dataset v2 (198k rows) → `extract_hidden.py --subtask` → `train_heads.py --features-suffix=-subtask-d1` → `rollout.py --subtask oracle --episodes 30`.

During collection the pre-DAgger policy (layer 16) succeeded on 163/200 reach, 123/200 push, 168/200 pick-place (82 / 62 / 84%; 300-step cap). That is the best success estimate so far for the pre-DAgger policy (10× more episodes).

Closed-loop success with oracle subtask, before (20 episodes, seeds 10000–10019) vs. after DAgger (30 episodes, seeds 10000–10029):

| layer | reach | push | pick-place | total |
|---|---|---|---|---|
| 4 before | 65% | 30% | 50% | 48% |
| **4 after** | **80%** | **50%** | **60%** | **63%** |
| 16 before | 75% | 65% | 80% | 73% |
| 16 after | 87% | 53% | 83% | 74% |
| 28 before | 80% | 40% | 60% | 60% |
| 28 after | 70% | 53% | 63% | 62% |

ms/decision after: 2.1 / 6.0 / 9.6 at layers 4 / 16 / 28.

Takeaways:
- **DAgger helps the shallowest executor most** (+15 points at layer 4, push 30 → 50%), even though the recovery data was collected by the layer-16 policy. Layers 16 and 28 are unchanged within noise (±9 points at 30 episodes).
- Offline test accuracy fell (77 → 69%) because the test split now contains hard drifted states. Not comparable to earlier offline numbers.
- **The fast head's confidence predicts its errors** (push, layer 16): steps with top prob < 0.4 (5% of steps) disagree with the expert 96% of the time vs. 68% otherwise; < 0.5 (14% of steps): 92% vs. 65%. So confidence is a usable router signal.

## 2026-10-18 — Planner in the loop: periodic planning vs. confidence router (push, layer 16, DAgger heads)

`rollout.py --subtask planner --heads-dir outputs/heads/Qwen3-1.7B-subtask-d1 --layers 16 --tasks push-v3 --episodes 10` with `--plan-every 25` or `--router-threshold 0.4 --cooldown 10`. The planner is the same Qwen3-1.7B (full depth, thinking), called unbatched (~7–10 s per call).

| subtask source | success | planner calls / episode | planner answers correct | planner time / episode |
|---|---|---|---|---|
| oracle (reference, 30 episodes) | 16/30 (53%) | — | — | — |
| planner every 25 steps | 5/10 | 12.9 (5.8 in successful episodes) | 87% | 96 s |
| planner when top prob < 0.4 | **0/10** | 5.8 | 86% | 41 s |

Takeaways:
- **Periodic planning works about as well as the oracle subtask** (5/10 vs. 53%), with ~6 planner calls in successful episodes.
- **The confidence router fails, and the failure is instructive.** The planner correctly switches to "lower onto the puck" around step 30, but then the fast head stays confident (~0.85) while executing that stale subtask, so the router never re-plans and the robot hovers for the rest of the episode. **Confidence catches action uncertainty, not plan staleness.**
- Motivates a learned router that detects staleness: a small classifier on the fast path's own layer-k features predicting "the subtask in my prompt no longer fits", at no extra inference cost.

## 2026-10-18 — Learned staleness router (offline)

`python src/stale_router.py` (Qwen3-1.7B; 40k push/pick-place states from dataset v2 + DAgger, each paired with its true subtask (fresh) and a random other subtask of the same task (stale) → 80k prompts; small MLP on the fast path's own layer-k features; split by episode)

| layer | test accuracy | stale prompts caught | false alarms on fresh prompts |
|---|---|---|---|
| 4 | 92.5% | 92.2% | 7.1% |
| 16 | 94.7% | 93.3% | 3.9% |

Takeaway: the fast path's own features already encode whether its subtask fits the state. A tiny probe detects stale plans with ~95% accuracy at zero extra inference cost (the features are computed anyway). Contrast: the full model asked directly is at chance on the related subtask question and needs ~10 s of thinking to reach ~90%. So the cheap probe is the trigger, and the expensive planner fixes the plan.
