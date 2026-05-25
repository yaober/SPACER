#!/bin/bash
#SBATCH --job-name=spacer_seed
#SBATCH -p 512GB
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --array=0-9
#SBATCH -o /project/shared/cli_wang/s439765/spatial_tcr/MIL_TCR/logs/stability_seed/job_%A_%a.out
#SBATCH -e /project/shared/cli_wang/s439765/spatial_tcr/MIL_TCR/logs/stability_seed/job_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jia.yao@utsouthwestern.edu

source activate spatial_tcr
export CUDA_VISIBLE_DEVICES=0
cd /project/shared/cli_wang/s439765/spatial_tcr/MIL_TCR
mkdir -p /project/shared/cli_wang/s439765/spatial_tcr/MIL_TCR/logs/stability_seed
SEED=${SLURM_ARRAY_TASK_ID:-0}
OUTPUT_DIR=/project/shared/cli_wang/s439765/spatial_tcr/MIL_TCR/stability/seed/seed${SEED}
mkdir -p ${OUTPUT_DIR}
echo "seed=${SEED}  output=${OUTPUT_DIR}"
python /project/shared/cli_wang/s439765/spatial_tcr/MIL_TCR/train.py \
    --data /project/shared/cli_wang/s439765/spatial_tcr/MIL_TCR/data/all_data/t_cell.csv \
    --output_dir ${OUTPUT_DIR} \
    --reference_gene /project/shared/cli_wang/s439765/spatial_tcr/MIL_TCR/data/human_filtered.csv --immune_cell tcell --learning_rate 0.05 --num_epochs 10 --patience 5 --delta 0.0001 --n_genes 500 --selection positive \
    --seed ${SEED}
