#!/bin/bash
#SBATCH --job-name tedd

# Name of the SLURM partition that this job should run on.
#SBATCH -p 512GB   # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

#SBATCH -t 100-23:0:00

#SBATCH -o job_%j_tcell.out
#SBATCH -e job_%j_tcell.err

#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

source activate spatial_tcr
cd /project/shared/cli_wang/s439765/spatial_tcr/MIL_TCR
python tedd.py 