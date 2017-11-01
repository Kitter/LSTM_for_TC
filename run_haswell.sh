#!/bin/bash -l
#SBATCH -p regular
#SBATCH -N 1
#SBATCH -t 16:00:00
#SBATCH -L project
#SBATCH -C haswell
 
module load tensorflow

python main_2.py
