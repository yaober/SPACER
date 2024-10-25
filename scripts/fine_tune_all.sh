#!/bin/bash
#SBATCH --job-name fine_tune_model1

# Name of the SLURM partition that this job should run on.
#SBATCH -p GPUA100    # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

#SBATCH -t 100-23:0:00

#SBATCH -o job_%j_fine_tune_all20000+8000.out
#SBATCH -e job_%j_fine_tune_all20000+8000.err

#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

module load python/3.8.x-anaconda
conda activate spatial_tcr
cd /project/DPDS/Wang_lab/s439765/spatial_tcr/MIL_TCR
python fine_tune_model1_all.py