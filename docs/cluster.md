# Yale Bouchet HPC Cluster — Reference

Condensed from the course guide "Yale HPC Cluster Setup and Use" (Aakash Patel, written for CPSC 4770, Spring 2026), adapted to our account. The guide says `cpsc4770`; for us it is **`cpsc4880`** (group verified with `groups`).

## Links
- Cluster docs: https://docs.ycrc.yale.edu/clusters-at-yale/
- SLURM (job scheduling): https://docs.ycrc.yale.edu/clusters-at-yale/job-scheduling
- OnDemand guide: https://docs.ycrc.yale.edu/clusters-at-yale/access/ood/
- Bouchet overview / getting started: https://docs.ycrc.yale.edu/clusters/bouchet/
- OnDemand dashboard (Yale wifi or VPN only): https://ood-bouchet.ycrc.yale.edu/pun/sys/dashboard/
- SSH setup: https://docs.ycrc.yale.edu/clusters-at-yale/access/ssh/

## Nodes
- **Login nodes** (`login1`/`login2`): logging in, launching jobs, checking storage. No running code or opening big files.
- **Compute nodes** (e.g. `a1130u05n01`): everything else. You get one by starting a job (`salloc`, a Code Server session, or `sbatch`).

## Storage
| Location | Size | Notes |
|---|---|---|
| `~` (`/home/cpsc4880_jnp32`) | 125 GB, private | Code, conda env |
| `~/scratch_cpsc4880` | 10 TB, **shared by the whole class** | Files auto-deleted after 60 days. Big regenerable outputs (our `outputs/` symlink points here) |
| `~/project_cpsc4880` | 4 TB, shared by the class, permanent | Small things we can't lose |

Be mindful of usage on the shared spaces. Check usage with `du -sh <dir>` (home) and `getquota -g cpsc4880` (scratch/project).

## Partitions
| Partition | Max time | GPU (memory) | Use |
|---|---|---|---|
| `devel` | 6 h | none | Interactive CPU work, debugging |
| `day` | 24 h | none | Long CPU jobs (data processing, evals) |
| `gpu_devel` | 6 h | RTX 5000 (32 GB) / H200 (141 GB) | Interactive GPU debugging |
| `gpu` | 48 h | RTX 5000 (32 GB) | Batch training/eval |
| `gpu_rtx6000` | 48 h | RTX 6000 (96 GB) | Batch training/eval |
| `gpu_h200` | 48 h | H200 (141 GB) | Batch training/eval |

- GPUs are shared across all of Yale, so expect queue waits.
- Class-wide limits: **6 concurrent GPUs** across `gpu` + `gpu_rtx6000`; 32 on `gpu_h200`.
- Rule of thumb: `devel`/`gpu_devel` for interactive work, `day`/`gpu` for background jobs.

## Interactive jobs
- `salloc` with no arguments gives 1 CPU, 1 hour, 5 GB on a compute node. Add options (below) for more, e.g. `salloc -p gpu_devel --gpus=1 -t 2:00:00`.
- Or use OnDemand → Interactive Apps → **Code Server** (VS Code in the browser on a compute node).
- Check jobs: `squeue --me` (`R` = running, `PD` = pending). Cancel: `scancel <jobid>`.

## Batch jobs (`sbatch`)
Template (our scripts go in `slurm/`; logs go to `outputs/logs/`):

```bash
#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --output=outputs/logs/%x_%J.log   # %x = job name, %J = job ID
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL                   # optional: email on start/end/fail

date; hostname; pwd
module load miniconda
conda deactivate
conda activate dpq
python src/<script>.py ...
```

Submit with `sbatch slurm/<job>.sh` from the repo root; watch with `squeue --me` and `tail -f outputs/logs/<log>`.

**Common options:** `--job-name`, `--output` (default `slurm-%j.out`), `--partition`, `--time` (`D-HH:MM:SS`, e.g. `4:00:00`), `--cpus-per-task` (default 1), `--mem-per-cpu` (default 5G) or `--mem` (per node), `--gpus`, `--constraint` (node type), `--mail-type`.

**Requesting resources:**
- Request only what the code uses. Bigger requests queue longer and don't run faster.
- More than 1 CPU only if the code is parallel; more than 1 GPU only if the code is multi-GPU.
- Memory: about 2× the data size (16 GB data → 32 GB). Don't take a whole GPU node's memory, or its other GPUs become unusable for everyone. `bigmem` is for datasets of hundreds of GB or more.

## SSH from a local machine
1. Generate a key pair and upload the public key (Yale SSH guide). Test with `ssh <netid>@bouchet.ycrc.yale.edu` (Duo push).
2. `~/.ssh/config` on the local machine:

```
Host bouchet
    HostName bouchet.ycrc.yale.edu
    User <netid>
    ForwardAgent yes
    IdentityFile ~/.ssh/id_ed25519
    ForwardX11 yes
    ControlMaster auto
    ControlPath /tmp/%h_%p_%r
    ControlPersist 2h

Host *.bouchet.ycrc.yale.edu
    User <netid>
    ProxyJump bouchet
    IdentityFile ~/.ssh/id_ed25519
    ControlMaster auto
    ControlPath /tmp/%h_%p_%r
    ControlPersist 2h
```

3. Start a job (`salloc` or Code Server), note its node, then `ssh <node>.bouchet.ycrc.yale.edu` (or VS Code / Cursor "Remote-SSH: Connect to Host" with that name). This is how Claude Code is run here; see the workflow in `project-overview.md` Section 6.
