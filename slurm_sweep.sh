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

# SWEEP_TYPE=fixed_uniform # Running
SWEEP_TYPE="fixed_structured" # Running 
# SWEEP_TYPE=changing_uniform # Running 
# SWEEP_TYPE=changing_structured # Done 

if [ "$SWEEP_TYPE" == "fixed_uniform" ]; then
    ~/.conda/envs/vlad/bin/python random_sweeps.py \
        --config "reproduce_configs/vjepa/${SWEEP_TYPE}/sweep_${SWEEP_TYPE}.(0).vjepa.yaml" \
        --num_trials 5 \
        > "./ccv/logs/sweep_${SWEEP_TYPE}.(0).vjepa.out" \
        2>&1 &
fi

~/.conda/envs/vlad/bin/python random_sweeps.py \
    --config "reproduce_configs/vjepa/${SWEEP_TYPE}/sweep_${SWEEP_TYPE}.(0.25).vjepa.yaml" \
    --num_trials 5 \
    > "./ccv/logs/sweep_${SWEEP_TYPE}.(0.25).vjepa.out" \
    2>&1 &
# Hacky way to get cifar datasets to not clobber each other on download I am assuming?
sleep 30

~/.conda/envs/vlad/bin/python random_sweeps.py \
    --config "reproduce_configs/vjepa/${SWEEP_TYPE}/sweep_${SWEEP_TYPE}.(0.50).vjepa.yaml" \
    --num_trials 5 \
    > "./ccv/logs/sweep_${SWEEP_TYPE}.(0.50).vjepa.out" \
    2>&1 &
~/.conda/envs/vlad/bin/python random_sweeps.py \
    --config "reproduce_configs/vjepa/${SWEEP_TYPE}/sweep_${SWEEP_TYPE}.(0.75).vjepa.yaml" \
    --num_trials 5 \
    > "./ccv/logs/sweep_${SWEEP_TYPE}.(0.75).vjepa.out" \
    2>&1 &
~/.conda/envs/vlad/bin/python random_sweeps.py \
    --config "reproduce_configs/vjepa/${SWEEP_TYPE}/sweep_${SWEEP_TYPE}.(1.5).vjepa.yaml" \
    --num_trials 5 \
    > "./ccv/logs/sweep_${SWEEP_TYPE}.(1.5).vjepa.out" \
    2>&1 &
~/.conda/envs/vlad/bin/python random_sweeps.py \
    --config "reproduce_configs/vjepa/${SWEEP_TYPE}/sweep_${SWEEP_TYPE}.(1.25).vjepa.yaml" \
    --num_trials 5 \
    > "./ccv/logs/sweep_${SWEEP_TYPE}.(1.25).vjepa.out" \
    2>&1 &
~/.conda/envs/vlad/bin/python random_sweeps.py \
    --config "reproduce_configs/vjepa/${SWEEP_TYPE}/sweep_${SWEEP_TYPE}.(1).vjepa.yaml" \
    --num_trials 5 \
    > "./ccv/logs/sweep_${SWEEP_TYPE}.(1).vjepa.out" \
    2>&1 &
~/.conda/envs/vlad/bin/python random_sweeps.py \
    --config "reproduce_configs/vjepa/${SWEEP_TYPE}/sweep_${SWEEP_TYPE}.(2.5).vjepa.yaml" \
    --num_trials 5 \
    > "./ccv/logs/sweep_${SWEEP_TYPE}.(2.5).vjepa.out" \
    2>&1 &

wait
