#!/bin/bash
#SBATCH --job-name mil_20000_0.1 

# Name of the SLURM partition that this job should run on.
#SBATCH -p 512GB   # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

#SBATCH -t 100-23:0:00

#SBATCH -o job_%j_0.1.out
#SBATCH -e job_%j_0.1.err

#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

module load python/3.8.x-anaconda
conda activate spatial_tcr
cd /project/DPDS/Wang_lab/s439765/spatial_tcr/MIL_TCR
python train.py --data data/training_all.csv --reference_gene data/human.csv --output_dir ./test/all_cpu_20000_0.1 --immune_cell tcell --learning_rate 0.1 --num_epochs 1000 --patience 5 --delta 0.0001