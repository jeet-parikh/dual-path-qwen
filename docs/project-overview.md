# Dual Path Qwen — Project Overview

> **Instructions for Claude Code (read first):**
> 1. Save this entire document as `docs/project-overview.md` in the repo root (`~/dual-path-qwen`).
> 2. Create a short `CLAUDE.md` in the repo root that summarizes the "Rules for Claude" section below and points to `docs/project-overview.md` for full context.
> 3. Then help me build the pipeline following the week 1 plan in Section 4. I'm new to ML research, so go one file at a time: explain what the file does and why, create it, run it if it's meant to be run, and wait for me before moving on. Start with `src/common.py` (shared prompt formatting and action conversion) and a Meta-World check script.
> 4. Keep this document up to date when we make or change a decision.

---

## 1. What the project is

Final project for **CPSC 4880 (Advances in Frontier AI Models), Yale**. The proposal was written by a team of four (Jeet Parikh (jnp32), Farhan Baig, Kashvi Pundir, Hari Viswanathan); **right now Jeet is working on it solo.** Original proposal title: *"Dual Path Qwen: Using General Purpose LLMs in High-Level Control and Reactive Settings."* Full proposal text (verbatim): [`docs/proposal.md`](proposal.md). How it maps onto this plan, risks, and recommended additions: [`docs/proposal-analysis.md`](proposal-analysis.md).

**Core idea.** Use one general-purpose LLM (Qwen) as both a fast "system 1" action selector and a slow "system 2" reasoner for robot control:

- **Fast path:** run only the first *k* layers of a **frozen** Qwen model, take the hidden state, and feed it to a small trained classifier head that outputs probabilities over a fixed, discrete set of robot actions.
- **Slow path:** run the full model for harder decisions (and, later, for high-level planning / task decomposition).
- **Router:** decides which path to use for each decision.

**Hypothesis.** Early layers already contain enough information to pick good low-level actions, so the fast path can match much of the full model's accuracy at a fraction of the latency and compute.

**Headline result we want:** action accuracy and task success versus number of layers used, plus the latency/FLOPs saved, and how often the router needs the slow path.

**Inspiration / related work** (from the proposal): Jev (TypeSafe AI) and its open-source counterpart Kev, which return structured choices over constrained action spaces; DeeBERT and LayerSkip (early-exit inference); UniFS (fast-to-slow hierarchy for VLAs). Note: details about Jev/Kev have not been independently verified; check their docs before relying on specific claims.

## 2. Scope decisions

- **Benchmark: Meta-World only.** LIBERO has been dropped (it needs images and a vision-language model, too much for the timeline).
- **Observations are state vectors written as text**, not images. This keeps everything in text-only Qwen and avoids rendering on the cluster.
- **Training data comes from Meta-World's built-in scripted expert policies**, so we can generate unlimited demonstrations.
- **Current goal: a basic working MVP of the whole fast-path pipeline by the progress report (about one week).**
- **Solo for now:** one person, working directly on `main`.

## 3. Locked design decisions (week 1 MVP)

| Decision | Choice |
|---|---|
| Tasks | Meta-World `reach-v3`, `push-v3`, `pick-place-v3` (MT1 environments) |
| Model | Qwen3-1.7B (28 layers) for development; Qwen3-0.6B for quick debugging; Qwen3-8B (36 layers) later |
| Input | Observation serialized as a text prompt, numbers rounded to 2 decimals (format below) |
| Actions | x, y, z each discretized to {-1, 0, +1}; gripper to {open, closed} → **54 classes** |
| Discretization settings | `DEAD_ZONE = 0.05`, `MOVE_SIZE = 0.5`, `GRIP_THRESHOLD = 0.3` (tuned and verified, see below) |
| Class index | `class = ((ix*3 + iy)*3 + iz)*2 + ig`, where `ix, iy, iz = direction + 1` (0, 1, 2) and `ig` = 0 open / 1 closed. Defined in `src/common.py` |
| Class → continuous action | xyz = direction × `MOVE_SIZE`; gripper open → 0.0, closed → 1.0 (the experts use 0.0 for open and 0.6–1.0 for closed) |
| Features | Last-token hidden state at layers 4, 8, 12, 16, 20, 24, 28, **with the model's final RMSNorm applied** (= what a model cut to k layers outputs; raw residuals have outlier values ~6500, normed < 100). Defined in `src/qwen.py`; verified identical between cached full-pass features and the live first-k-layers fast path |
| Qwen | **Frozen.** Never fine-tuned. Hidden states are extracted once and cached; heads train on the cache |
| Train/test split | **By episode, never by step** (neighboring steps are nearly identical, so a step split inflates accuracy). **Every episode must have its own layout:** Meta-World MT1 pre-samples only 50 layouts per seed and cycles through them on `reset()`, so always use `episode_env(task, seed)` with a distinct seed per episode. Data: seeds 0..N-1; rollouts: 10000+ |

**Prompt format** (produced by `obs_to_prompt` in `src/common.py`):

```
Task: push the puck to the goal.
Hand: (0.00, 0.60, 0.20). Gripper: 1.00.
Object: (0.00, 0.69, 0.02). Goal: (0.08, 0.88, 0.02).
Action:
```

**Meta-World observation layout (39 numbers):** `[0:3]` hand xyz, `[3]` gripper opening, `[4:7]` object xyz, `[7:18]` object rotation / second object, `[18:36]` previous frame, `[36:39]` goal xyz. The goal is visible in the observation in the MT1 environments we use (verified).

**Discretization was validated** in an earlier prototype. A check script ran the expert normally and with its actions squashed into the 54 classes and back ("discretized expert"). The first setting (dead zone 0.15) broke reach (12/20 successes) because the robot overshot near the goal. A sweep found `DEAD_ZONE = 0.05`, `MOVE_SIZE = 0.5` works: **30/30 successes on all three tasks** for both the expert and the discretized expert (Meta-World 3.1.1). This matters because a classifier imitating the expert can at best match the discretized expert. **Caveat (2026-10-18):** over 200 episodes the discretized expert gets 194/200 on pick-place (plain expert 200/200): fixed-size moves occasionally knock the puck and the gripper closes too early. So the pick-place ceiling is ~97%.

**Data format** (handoff from data generation to everything else), `data/<task>.jsonl`, one line per step:

```json
{"episode_id": "push_0042", "task": "push-v3", "step": 17, "prompt": "...", "action_class": 39, "action_continuous": [0.4, -0.1, 0.0, 1.0], "subtask": "push the puck to the goal", "obs": [... 39 numbers ...]}
```

- `action_class` = `continuous_to_class(action_continuous)`; `action_continuous` is the raw expert action (the label).
- `subtask` = the expert's active stage, from `expert_subtask` in `src/common.py` (mirrors the expert's if/else branches). Used for planner work later.
- `obs` = raw observation, for the raw-state MLP baseline.
- **The robot executes the discretized action during data generation**, so recorded states match what a perfect 54-class policy visits.
- **Only successful episodes are saved.**

Current dataset: v2, `python src/data_gen.py` (500 episodes per task, one unique layout each). v1 (200 episodes, only ~49 layouts per task) is superseded. See `docs/results.md`.

## 4. Week 1 plan (MVP for the progress report)

**Done means:** one command takes Meta-World states, routes them through a partial Qwen and a trained head, and drives the robot; plus a plot of action accuracy versus layer depth.

- **Day 1 — Setup and checks.** Environment working (done). Build `src/common.py`, then two check scripts: `src/check_metaworld.py` (expert and discretized expert should succeed on all three tasks) and `src/check_qwen.py`, run on `gpu_devel` (loads Qwen, prints hidden-state shapes, times the first k layers vs. the full model).
- **Day 2 — Generate data.** `src/data_gen.py`: ~200 expert episodes per task, saved in the data format above. Check the class distribution (some of the 54 classes will be rare or unused; that's fine but worth knowing).
- **Day 3 — Extract hidden states.** `src/extract_hidden.py`: one forward pass of Qwen over every prompt with `output_hidden_states=True`, save last-token vectors for the chosen layers. Run via `sbatch`. A couple of GB at 1.7B. No training of Qwen. **Done 2026-10-18:** ran interactively instead (about 1 min per model on one GPU); features in `outputs/hidden/<model>/` (`layer<k>.npy` + `meta.npz`).
- **Day 4 — Train heads, make the plot.** `src/train_heads.py`: one head per layer (logistic regression first, then a 2-layer MLP). Split by episode. Include a majority-class baseline. Plot test accuracy vs. layer. This is the centerpiece of the progress report. **Done 2026-10-18:** `src/train_heads.py` + `src/plot_heads.py` → `figures/`. One shared head for all tasks; split 70/10/20 by episode; 200 epochs, best-val epoch. Baselines: majority, per-task majority, MLP on prompt numbers, MLP on full-precision state, prompt-text oracle ceiling. Results in `docs/results.md`.
- **Day 5 — Closed loop.** `src/rollout.py`: state → prompt → first k layers → head → action class → continuous action → environment. ~20 episodes per task; record success rate and ms per decision. Repeat with the full-depth head. Expect lower success than offline accuracy (covariate shift: small errors compound as the robot drifts into states the expert never visited). The goal this week is that the loop runs, not that it performs well. **Done 2026-10-18:** `src/rollout.py`; 1.7B MLP heads succeed 55% / 72% / 62% of episodes at layers 4 / 16 / 28 (2.1 / 5.7 / 9.4 ms per decision). See `docs/results.md`.
- **Day 6 — Simple router + buffer.** `src/router.py`: if the fast head's top probability is below a threshold, use the full-depth head. Log the fast/slow usage ratio. Rest of the day is buffer.
- **Day 7 — Progress report.** Layer-vs-accuracy plot, closed-loop success and latency table, router usage ratio, what we learned, what's next.

**Open decision (raised 2026-10-18): prompt precision.** At 2 decimals, identical prompts map to different expert actions, capping accuracy at 94.4% (full-precision state reaches 97.7%). Showing 3 decimals would raise the ceiling but adds ~10 tokens per prompt (slightly slower). Not changed yet.

## 5. After the MVP (to honor the proposal)

Firm commitments from the proposal still to do after week 1:

- **Planner role for the slow path.** The full model generates a subtask sequence from the overall goal; the fast head is conditioned on the current subtask (subtask text in its prompt); every N steps the planner checks progress and advances or redirects. In Meta-World, chain single tasks (e.g. reach the object, then push it to the goal).
- **Two routers, compared.** (1) Confidence threshold on the fast head (above). (2) A learned lightweight router. Labels: run both paths over the dataset and mark states "needs slow" where the fast head is wrong but the full-depth path is right; train a small classifier on input-embedding (or layer 1–2) features to predict that.
- **Baselines.** (a) Entropy-based early exit (DeeBERT-style: heads at several layers, exit at the first whose prediction entropy is below a threshold); nearly free since we train heads at multiple depths. (b) Split models: a separate small model (e.g. Qwen3-0.6B) as the fast path paired with 8B as the slow path. This is the baseline the "single model" pitch must beat.
- **Resume-from-layer-k escalation.** When the router escalates, continue the forward pass from the layer-k hidden state instead of restarting, so fast-path compute is never wasted. This is the main argument for one model over two; show the FLOPs savings vs. the split baseline.
- **Metrics:** inference latency, task success rate, FLOPs, slow/fast usage ratio, fast-path probability distributions per task. Log probabilities in every closed-loop run.
- **Vary fast-path depth and routing mechanism.**

Optional (hedged in the proposal): multiple model sizes, efficient inference techniques (stretch goal).

Rough timeline: Week 2 closed-loop eval + early-exit baseline; Week 3 planner loop, both routers, split baseline, resume-from-k; Week 4 scale to 8B, latency/FLOPs across everything, write-up. If something slips, keep the planner loop and both routers (the distinctive contributions).

Note: because Qwen is frozen, training the head **cannot** change the full model's output. The proposal lists "whether training them affects the final output" as an open question; the answer is no by construction.

## 6. Cluster setup (Yale Bouchet) — current state

General cluster reference (partitions, storage, `salloc`/`sbatch`, SSH): [`docs/cluster.md`](cluster.md). This section covers our specific setup.

- **Username:** `cpsc4880_jnp32`. Home: `/home/cpsc4880_jnp32`. Repo: `~/dual-path-qwen`.
- **Login nodes** (`login1`/`login2`): editing, git, submitting jobs only. **Never run Python, pip, or heavy work there.**
- **Compute nodes:** reserved via a job. All nodes share the same home directory, so files, the repo, and the conda env persist across sessions. Running processes and `/tmp` do not.
- **Conda env:** `dpq` (Python 3.11) at `/home/cpsc4880_jnp32/.conda/envs/dpq`. Installed: torch, transformers>=4.51, accelerate, metaworld==3.1.1, gymnasium, numpy, scikit-learn, matplotlib, tqdm. Activate with `module load miniconda && conda activate dpq`.
- **Claude Code** is installed at `~/.local/bin/claude`.
- **GitHub:** cluster SSH key added; repo cloned via SSH (`git@github.com:jeet-parikh/dual-path-qwen.git`). Solo for now, so work happens on `main`; still ask before pushing.
- **Partitions:** `devel` (CPU, 6 h), `gpu_devel` (GPU, 6 h, interactive debugging), `gpu` (RTX 5000 32 GB, 48 h, batch jobs via `sbatch`), `gpu_h200` (141 GB), `day` (CPU, 24 h). The class shares limits: max 6 concurrent GPUs on `gpu`/`gpu_rtx6000`. Request 1 GPU unless told otherwise.
- **Storage:** home 125 GB private; `~/scratch_cpsc4880` shared by the class, purged after 60 days (big regenerable files go here, in a personal subfolder); `~/project_cpsc4880` shared and permanent (small things we can't lose). Folder names verified with `ls ~`.
- **Scratch setup (done):** `~/scratch_cpsc4880/$USER/hf_cache` (Hugging Face model downloads; `HF_HOME` is set in `~/.bashrc`) and `~/scratch_cpsc4880/$USER/outputs`, which the repo's `outputs/` symlink points to.

**My workflow:** Start a Code Server session in OnDemand (this reserves the compute node; don't cancel it). In iTerm2: `ssh <node>.bouchet.ycrc.yale.edu`, then `tmux new -s work` (reattach with `tmux attach -t work` after a disconnect), `cd ~/dual-path-qwen`, `module load miniconda`, `conda activate dpq`, `claude`. Code Server in the browser is for viewing files. The node name changes every session.

## 7. Repo layout

Kept deliberately simple: no package install, no test framework.

```
docs/      project-overview.md, proposal.md, proposal-analysis.md, results.md (dated results log)
src/       common.py (prompts/actions/envs), qwen.py (model loading/features) + one script per pipeline step (run as `python src/<script>.py ...` from the repo root)
slurm/     sbatch scripts for heavy jobs
outputs/   symlink to ~/scratch_cpsc4880/$USER/outputs (gitignored): data/ hidden/ heads/ rollouts/ logs/
```

- Scripts import shared code with `from common import ...` (works because Python adds the script's folder to its import path).
- Every script takes command-line arguments, defaults to writing under `outputs/`, and saves its arguments as `args.json` next to its outputs.
- `common.py`'s self-test (`python src/common.py`) and the `check_*.py` scripts are the only tests.

## 8. Rules for Claude

- I'm new to AI research and cluster computing. Explain what you're doing and why, briefly and in plain language. Go step by step.
- Always confirm you're on a compute node (hostname is not `login1`/`login2`) before running Python.
- **Ask before:** submitting any `sbatch` job, requesting GPUs, deleting files, pushing to `main`, or downloading large models.
- Never commit data, hidden-state caches, model weights, checkpoints, or logs (see `.gitignore`).
- Test on small subsets and small models before any full run.
- Scripts should be runnable from the repo root with command-line arguments, not hard-coded paths. Large outputs go to scratch.
- All prompt formatting and action conversion goes through `src/common.py`; change discretization settings only there.
- Keep train/test splits by episode.
