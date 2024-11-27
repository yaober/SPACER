#!/bin/bash
#SBATCH --job-name fine_tune_fibroblast

# Name of the SLURM partition that this job should run on.
#SBATCH -p 512GB    # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

#SBATCH -t 100-23:0:00

#SBATCH -o job_%j_fine_tune_fibroblast.out
#SBATCH -e job_%j_fine_tune_indivual_fibroblast.err

#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

module load python/3.8.x-anaconda
conda activate spatial_tcr
cd /project/DPDS/Wang_lab/s439765/spatial_tcr/MIL_TCR
conda activate spatial_tcr

python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/s439765/spatial_tcr/negative_control/high_res/HumanColorectalCancerFibroblast_negative_control.h5ad' --reference_gene_path 'data/human_filtered.csv' --pretrained_gene_path 'data/human_filtered.csv' --model_path 'finalize_model/fibroblast/best_model.pth' --output_dir './fine_tuned_model_fibroblast/HumanColorectalCancer' --resolution high --radius 150 --learning_rate 0.1 --num 200 --n_genes 18085 --immune_cell fibroblast
python fine_tune_model1_indiviual.py --data_path '/project/DPDS/Wang_lab/s439765/spatial_tcr/negative_control/high_res/HumanLungCancerFibroblast_negative_control.h5ad' --reference_gene_path 'data/human_filtered.csv' --pretrained_gene_path 'data/human_filtered.csv' --model_path 'finalize_model/fibroblast/best_model.pth' --output_dir './fine_tuned_model_fibroblast/HumanLungCancerHD' --resolution high --radius 150 --learning_rate 0.1 --num 200 --n_genes 18085 --immune_cell fibroblast


