#!/bin/bash
#SBATCH -o ./ccv/logs/train_job-%j.out
#SBATCH -e ./ccv/logs/train_job-%j.err

#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gpus=1

module load miniconda3
conda activate vlad

# Expecting arguments: combination, num_trials, num_iterations
combination=$1
num_trials=$2

# Call the helper script with the provided arguments.
./mega_sweep.sh "${combination}" "${num_trials}"
