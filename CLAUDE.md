# CLAUDE.md

Dual Path Qwen (CPSC 4880 final project, currently solo): a frozen Qwen's first k layers + a small trained head pick discrete robot actions in Meta-World (fast path); the full model handles harder decisions (slow path); a router chooses between them.

**Full context, locked decisions, and the week-by-week plan: [`docs/project-overview.md`](docs/project-overview.md).** Keep it up to date whenever a decision is made or changed.

- Original proposal (verbatim; the user won't paste it again): [`docs/proposal.md`](docs/proposal.md)
- Proposal → plan mapping, risks, and recommended-but-not-yet-agreed additions: [`docs/proposal-analysis.md`](docs/proposal-analysis.md)

## Rules

- The user is new to ML research and cluster computing: explain briefly and in plain language, go one file at a time, and wait for the user before moving on.
- Before running Python, confirm you're on a compute node (`hostname` is not `login1`/`login2`). Activate the env with `module load miniconda && conda activate dpq`.
- **Ask before:** submitting `sbatch` jobs, requesting GPUs, deleting files, pushing to `main`, or downloading large models.
- Never commit data, hidden-state caches, model weights, checkpoints, or logs (see `.gitignore`).
- Test on small subsets and small models (Qwen3-0.6B) before any full run.
- Scripts run from the repo root with command-line arguments, not hard-coded paths. Large outputs go under `outputs/` (symlink to scratch). Repo layout: overview Section 7. Cluster reference (partitions, sbatch template, storage): `docs/cluster.md`.
- All prompt formatting and action conversion go through `src/common.py`; change discretization settings only there.
- Train/test splits are always by episode, never by step.
- Qwen is frozen; never fine-tune it.
- Prefer the simplest approach that works (no packaging, frameworks, or extra structure unless needed).
