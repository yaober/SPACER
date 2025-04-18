#!/bin/bash
#SBATCH --job-name tcell

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
conda activate spatial_tcr
export CUDA_VISIBLE_DEVICES=0
cd /project/DPDS/Wang_lab/s439765/spatial_tcr/MIL_TCR
python train.py --data  data/all_data/t_cell_2r.csv --reference_gene data/human_filtered.csv --output_dir inhibit_model/tcell_2r_clean --immune_cell tcell --learning_rate 0.01 --num_epochs 100 --patience 5 --delta 0.0001  --n_genes 500 --selection negative 
