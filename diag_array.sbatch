#!/bin/bash
#SBATCH --job-name=mlda_sweep
#SBATCH --partition=cpu_il
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=3000
#SBATCH --cpus-per-task=1
#SBATCH --array=0-59
#SBATCH --output=/home/ka/ka_scc/ka_br6867/tinyDA/results/sweep_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=louise.kluge@posteo.net

module purge
module load devel/python/3.12.3-gnu-14.2
source ~/venvs/tinyda/bin/activate
export OMP_NUM_THREADS=1

python ~/tinyDA/mlda_diagnostics.py --task-id ${SLURM_ARRAY_TASK_ID}
