#!/bin/bash
#SBATCH --job-name aocrc

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
export CUDA_VISIBLE_DEVICES=0
cd /project/shared/cli_wang/s439765/spatial_tcr/MIL_TCR/
python train.py --data  /archive/DPDS/Xiao_lab/shared/jia_yao/xenium32/data/data4spacer/aocrc.csv --reference_gene data/human_filtered.csv --output_dir /archive/DPDS/Xiao_lab/shared/jia_yao/xenium32/data/data4spacer/spacer_results/aocrc --immune_cell fibroblast --learning_rate 0.05 --num_epochs 10 --patience 5 --delta 0.0001  --n_genes 372 --selection positive 
