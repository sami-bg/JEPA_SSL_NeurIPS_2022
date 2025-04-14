import argparse
import os
import random
import subprocess
import sys
import math
from pathlib import Path

PYTHON_EXECUTABLE = '/users/sboughan/.conda/envs/vlad/bin/python'

def parse_args():
    parser = argparse.ArgumentParser(description="Run training with different learning rates")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of random learning rates to try")
    parser.add_argument("--min_lr_exp", type=float, default=-3, help="Minimum learning rate exponent (10^x)")
    parser.add_argument("--max_lr_exp", type=float, default=-2.5, help="Maximum learning rate exponent (10^x)")
    parser.add_argument("--output_dir", type=str, default="results/param_sweep", help="Base output directory for all sweeps")
    parser.add_argument("--tubelet_size", type=int, required=True, help="Tubelet size to use")
    parser.add_argument("--masking_ratio", type=float, required=True, help="Masking ratio to use")
    parser.add_argument("--quick_debug", action='store_true', help="Enable quick debug mode")
    return parser.parse_args()

def generate_random_lr(min_exp, max_exp):
    """Generate a random learning rate on a log scale."""
    exponent = random.uniform(min_exp, max_exp)
    return 10 ** exponent

def detect_model_type(config_path):
    """Detect model type from config file name."""
    config_name = Path(config_path).stem
    
    if "vjepa" in config_name.lower():
        return "VJEPA"
    elif "vicreg" in config_name.lower():
        return "VICReg"
    elif "simclr" in config_name.lower():
        return "SimCLR"
    elif "rssm" in config_name.lower():
        return "RSSM"
    else:
        raise ValueError(f"Could not detect model type from config name: {config_name}")

def main():
    args = parse_args()
    
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} does not exist")
        sys.exit(1)
    
    # Extract base name for run naming
    config_basename = Path(config_path).stem
    
    # Detect model type
    model_type = detect_model_type(config_path)
    print(f"Detected model type: {model_type}")
    
    # Create a more specific output directory including tubelet size and masking ratio
    sweep_params_str = f"ts{args.tubelet_size}_mr{args.masking_ratio}"
    output_base_dir = os.path.join(args.output_dir, config_basename, sweep_params_str)
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Log file to track experiments for this specific sweep config
    log_file = os.path.join(output_base_dir, "experiments.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            # Add tubelet_size and masking_ratio to header
            f.write("config,model_type,learning_rate,tubelet_size,masking_ratio,output_dir\n")
    
    # Include tubelet size and masking ratio in the run group
    run_group = f"{config_basename}_ts{args.tubelet_size}_mr{args.masking_ratio}_lr_sweep"
    # Run trials with different learning rates
    for trial in range(args.num_trials):
        # Generate random learning rate
        lr = generate_random_lr(args.min_lr_exp, args.max_lr_exp)
        
        # Create trial-specific output directory (relative to the new output_base_dir)
        trial_output_dir = os.path.join(output_base_dir, f"lr_{lr:.8f}")
        os.makedirs(trial_output_dir, exist_ok=True)
        
        # Construct run name for wandb, including all swept parameters
        run_name = f"{config_basename}_lr_{lr:.8f}_ts{args.tubelet_size}_mr{args.masking_ratio}"

        # Build the command
        cmd = [
            PYTHON_EXECUTABLE, "train.py",
            "--configs", config_path,
            "--values",
            f"wandb=True",
            f"output_path={trial_output_dir}",
            f"model_type={model_type}",
            # Pass the specific lr, tubelet_size, and masking_ratio
            f"{model_type.lower()}.base_lr={lr}",
            f"{model_type.lower()}.tubelet_size={args.tubelet_size}",
            f"{model_type.lower()}.masking_ratio={args.masking_ratio}",
            f"run_name={run_name}",
            f"run_group={run_group}",
            # Pass quick_debug correctly
            f"quick_debug={str(args.quick_debug).lower()}"
        ]
        
        print(f"\nTrial {trial+1}/{args.num_trials}: Running with lr={lr:.8f}, ts={args.tubelet_size}, mr={args.masking_ratio}")
        print(f"Command: {' '.join(cmd)}")
        
        # Run the command
        process = subprocess.run(cmd)
        
        # Log the experiment, including tubelet_size and masking_ratio
        with open(log_file, "a") as f:
            f.write(f"{config_basename},{model_type},{lr},{args.tubelet_size},{args.masking_ratio},{trial_output_dir}\n")
        
        if process.returncode != 0:
            print(f"Warning: Command exited with return code {process.returncode}")

if __name__ == "__main__":
    main()