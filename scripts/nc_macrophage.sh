#!/bin/bash
#SBATCH --job-name nc_macrophage

# Name of the SLURM partition that this job should run on.
#SBATCH -p 512GB   # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

#SBATCH -t 100-23:0:00

#SBATCH -o job_%j_macrophage.out
#SBATCH -e job_%j_macrophage.err

#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

source activate spatial_tcr

cd /project/DPDS/Wang_lab/s439765/spatial_tcr/MIL_TCR
python train.py --data data/macrophage.csv --reference_gene data/human.csv --output_dir ./finalize_model/macrophage --immune_cell macrophage --learning_rate 0.1 --num_epochs 1000 --patience 5 --delta 0.0001  --n_genes 10000    
