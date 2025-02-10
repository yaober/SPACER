#!/bin/bash
#SBATCH --job-name all

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
cd /project/DPDS/Wang_lab/s439765/spatial_tcr/MIL_TCR
python train.py --data  data/hyun/all/tcell.csv --reference_gene data/all_mouse_genes.csv --output_dir mouse_heart/all/tcell --immune_cell tcell --learning_rate 0.05 --num_epochs 10 --patience 5 --delta 0.0001  --n_genes 3000 
python train.py --data  data/hyun/all/bcell.csv --reference_gene data/all_mouse_genes.csv --output_dir mouse_heart/all/bcell --immune_cell bcell --learning_rate 0.05 --num_epochs 10 --patience 5 --delta 0.0001  --n_genes 3000 
python train.py --data  data/hyun/all/macrophage.csv --reference_gene data/all_mouse_genes.csv --output_dir mouse_heart/all/macrophage --immune_cell macrophage --learning_rate 0.05 --num_epochs 10 --patience 5 --delta 0.0001  --n_genes 3000    

