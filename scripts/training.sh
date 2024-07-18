#!/bin/bash
#SBATCH --job-name mil_training

# Name of the SLURM partition that this job should run on.
#SBATCH -p GPUv100s    # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

#SBATCH -t 100-23:0:00

#SBATCH -o job_%j.out
#SBATCH -e job_%j.err

#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

module load gpu_prepare
module load python/3.8.x-anaconda
conda activate spatial_tcr
cd/project/DPDS/Wang_lab/s439765/spatial_tcr/MIL_TCR
python train.py --data training.csv --reference_gene human.csv --output_dir ./results/test_1 --immune_cell tcell --learning_rate 0.0001 --num_epochs 100 --patience 5 --delta 0.0001