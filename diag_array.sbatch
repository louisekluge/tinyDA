#!/bin/bash
# Predator-prey sampler sweep: 9 configurations x 20 repetitions.
#
# Every task keeps its full chain, so the estimator-formulation work can be
# done offline without resampling. Sizing is in the comment block below.
#
#   sbatch diag_array.sbatch
#
#SBATCH --job-name=mlda_diag
#SBATCH --partition=single
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-179%20
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

REPO="$HOME/tinyDA_vr"            # <-- directory holding mlda_diagnostics.py
OUTDIR="$(ws_find mlda_diag)/sweep"

# One thread per task. numpy and scipy each default to grabbing every core on
# the node; with 20 concurrent array tasks that oversubscribes it and every
# task slows down together.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

module purge
module load devel/python/3.11        # <-- adjust to the module you used
source "$HOME/venv_tinyda/bin/activate"   # <-- adjust

mkdir -p logs "$OUTDIR"
cd "$REPO"

echo "task=$SLURM_ARRAY_TASK_ID host=$(hostname) started=$(date -Is)"
python mlda_diagnostics.py --task-id "$SLURM_ARRAY_TASK_ID" --outdir "$OUTDIR"
echo "finished=$(date -Is)"

# ---------------------------------------------------------------------------
# Sizing, from measured bytes-per-link (30-40 compressed) and a measured
# cost of 1.75e-3 s per model-evaluation unit, where a unit weights each
# level by how far it integrates (level 0 to t=3.5, level 1 to t=7.5,
# level 2 to t=12, so roughly 0.29 : 0.63 : 1).
#
# A three-level run stores N(2 + 2J + J^2) links, a two-level run N(2 + J),
# single level N. Cost per top-level sample is J^2*0.29 + J*0.63 + 1.
#
#   config          J   iters    links     chain file    runtime (local)
#   base           10   20000   2.44e6        98 MB          21 min
#   nsub1           1   20000   1.00e5         4 MB           1 min
#   nsub2           2   20000   2.00e5         8 MB           2 min
#   nsub5           5   20000   7.40e5        30 MB           7 min
#   nsub20         20    8000   3.54e6       142 MB          30 min
#   fixed_nsub     10   20000   2.44e6        98 MB          21 min
#   two_level_hi   10   20000   2.40e5        10 MB           4 min
#   two_level_lo   10   20000   2.40e5        10 MB           2 min
#   single          -   20000   2.00e4         1 MB           1 min
#
# Per repetition of the whole sweep: ~400 MB and ~89 min.
# At 20 repetitions: ~8 GB and ~30 CPU-hours, doubling to ~60 on a cluster
# core. The walltime above gives the longest config (nsub20) about 3x margin,
# which is deliberate: the superlinear slowdown seen in earlier runs has never
# been quantified, and the per-interval it/s column in each log will say
# whether it is present long before the limit is reached.
#
# 8 GB belongs on a workspace, not in $HOME, and the workspace needs extending
# before it expires or the chains go with it:  ws_extend mlda_diag 30
#
# To run without chain files -- summaries only, a few kB per task -- add
# --no-chains. The summaries carry acceptance, ESS, pairing and the estimator
# decomposition, so everything except comparing estimator *formulations* can
# be done from them alone.
# ---------------------------------------------------------------------------
