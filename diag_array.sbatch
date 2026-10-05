#!/bin/bash
# Predator-prey sampler sweep: 9 configurations x 20 repetitions.
# Every task keeps its full chain. Total output ~8 GB in tinyDA/results.
#
#   sbatch diag_array.sbatch
#
#SBATCH --job-name=mlda_diag
#SBATCH --partition=cpu_il
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-179%20
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

REPO="/home/ka/ka_scc/ka_br6867/tinyDA"     

# One thread per task. numpy and scipy each default to grabbing every core on
# the node; with 20 concurrent array tasks that oversubscribes it and every
# task slows down together.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

module purge
module load devel/python/3.12.3-gnu-14.2
source ~/venvs/tinyda/bin/activate 

cd "$REPO"
mkdir -p logs

echo "task=$SLURM_ARRAY_TASK_ID host=$(hostname) started=$(date -Is)"
python mlda_diagnostics.py --task-id "$SLURM_ARRAY_TASK_ID"
echo "finished=$(date -Is)"

# Output goes to the script's own results/ directory, which is the default.
#
# Sizing, measured: 30-40 compressed bytes per link; a three-level run stores
# N(2 + 2J + J^2) links, two-level N(2 + J), single level N.
#
#   base/fixed_nsub  J=10  20000   98 MB each    21 min
#   nsub20           J=20   8000  142 MB         30 min
#   nsub5            J= 5  20000   30 MB          7 min
#   two_level_*      J=10  20000   10 MB each   2-4 min
#   nsub1/nsub2/single          4-8 MB each      1-2 min
#
# ~400 MB and ~89 min per repetition of the whole sweep, so ~8 GB and ~30
# CPU-hours at 20 repetitions, roughly doubling on a cluster core. The
# walltime gives the longest config about 3x margin, because the superlinear
# slowdown seen in earlier runs has never been quantified; the per-interval
# it/s column in each log will show it long before the limit is reached.
#
# --no-chains writes summaries only, a few kB per task. Those carry
# acceptance, ESS, pairing and the estimator decomposition, so everything
# except comparing estimator *formulations* can be done without the chains.
