#!/bin/bash
#SBATCH --job-name=mlda_diag
#SBATCH --partition=cpu_il
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=6000
#SBATCH --array=0-19
#SBATCH --output=diag_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=louise.kluge@posteo.net

module purge
module load devel/python/3.12.3-gnu-14.2
source ~/venvs/tinyda/bin/activate
export OMP_NUM_THREADS=1

python ~/tinyDA/mlda_diagnostics.py \
    --run-id ${SLURM_ARRAY_TASK_ID} \
    --iterations 20000 \
    --subchain-length 10 \
    --proposal am \
    --outdir ~/diag_results
