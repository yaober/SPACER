#!/bin/bash
#SBATCH --job-name mil_training

# Name of the SLURM partition that this job should run on.
#SBATCH -p 512GB   # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

#SBATCH -t 1000-23:0:00

#SBATCH -o job_%j.out
#SBATCH -e job_%j.err

#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

for (( ; ; ))
do
  sleep 10
done
