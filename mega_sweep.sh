#!/bin/bash
# run_sweeps.sh
# Usage: ./run_sweeps.sh <fixed|changing> <num_trials>
#   where num_trials is the number of learning rate sweeps per experiment (e.g. 20)
#
# Example: To run 30 sweeps for the "fixed" combination:
#   ./run_sweeps.sh fixed 30

# Load conda and activate the environment
module load miniconda3
conda activate vlad

# Parse input arguments
combination=$1     # fixed or changing
num_trials=$2      # number of learning rate sweeps

# Create timestamped log directory
timestamp=$(date +"%Y-%m-%d-%H-%M-%S")
log_dir="./ccv/logs/mega_sweep_${combination}_${timestamp}" # More specific log dir name
mkdir -p "${log_dir}"

# Define the array of noise levels
# NOTE These need to be an exact string match to the config file names
# noise_levels=(0.25 0.50 0.75 1.00 1.25 1.5 1.75 2.0 2.25 2.5)
noise_levels=(1.75)
# noise_levels=(0.25 0.50 1.25 1.5 1.75)
# Define arrays for tubelet sizes and masking ratios
tubelet_sizes=(3)
# tubelet_sizes=(1 2 3)
masking_ratios=(0.7)

echo "Starting mega sweep for '${combination}' combination with ${num_trials} LR trials per setting."
echo "Log directory: ${log_dir}"

job_count=0

# for structure in "structured"; do
# for structure in "uniform" "structured"; do
for structure in "uniform"; do
  for noise in "${noise_levels[@]}"; do
    for ts in "${tubelet_sizes[@]}"; do
      for mr in "${masking_ratios[@]}"; do

        # Build the configuration file path.
        config_path="reproduce_configs/vjepa/${combination}_${structure}/sweep_${combination}_${structure}.(${noise}).vjepa.yaml"

        # Check if config file exists before launching
        if [ ! -f "${config_path}" ]; then
            echo "Warning: Config file not found, skipping: ${config_path}"
            continue
        fi

        # Update log file name to include ts and mr
        log_file="${log_dir}/${combination}_${structure}_noise${noise}_ts${ts}_mr${mr}.out"

        echo "Launching sweep for: config=${config_path}, ts=${ts}, mr=${mr}. Log: ${log_file}"

        # Pass tubelet_size and masking_ratio to random_sweeps.py
        ~/.conda/envs/vlad/bin/python random_sweeps.py \
            --config "${config_path}" \
            --num_trials "${num_trials}" \
            --tubelet_size "${ts}" \
            --masking_ratio "${mr}" \
            > "${log_file}" 2>&1 &

        job_count=$((job_count + 1))

        # This is so it downloads CIFAR10 and doesnt destroy the other processes.
        # Apply sleep only for the first job of each 'structured' config to handle potential dataset download contention
        if [ "$structure" = "structured" ] && [ "$ts" = "${tubelet_sizes[0]}" ] && [ "$mr" = "${masking_ratios[0]}" ]; then
          echo "Waiting 60 seconds before starting next structured noise level group to avoid CIFAR10 contention..."
          sleep 60
        fi
      done # end masking_ratio loop
    done # end tubelet_size loop
  done # end noise loop
done # end structure loop

echo "Launched ${job_count} sweep jobs in the background."
# Wait for all background processes launched in this script to finish.
wait
echo "All sweep jobs completed."
