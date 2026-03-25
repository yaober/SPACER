#!/bin/bash
#SBATCH --job-name benchmark

# Name of the SLURM partition that this job should run on.
#SBATCH -p 512GB   # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

#SBATCH -t 100-23:0:00

#SBATCH -o job_%j_sigmod.out
#SBATCH -e job_%j_sigmod.err

#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

source activate spatial_tcr
cd /project/shared/cli_wang/s439765/spatial_tcr/MIL_TCR/

python benchmark_per.py   --data data/all_data/simulation.csv   --reference_gene data/human_filtered.csv   --output_dir benchmark_ml_fastimp_simulation_2noise   --immune_cell tcell   --n_genes 500   --selection positive   --cache_features   --perm_on_tree   --perm_topk 100000000   --perm_repeats 1
