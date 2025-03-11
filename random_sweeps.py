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
    parser.add_argument("--min_lr_exp", type=float, default=-2, help="Minimum learning rate exponent (10^x)")
    parser.add_argument("--max_lr_exp", type=float, default=0, help="Maximum learning rate exponent (10^x)")
    parser.add_argument("--output_dir", type=str, default="results/lr_sweep", help="Base output directory")
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
    
    # Create output directory
    output_base_dir = os.path.join(args.output_dir, config_basename)
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Log file to track experiments
    log_file = os.path.join(args.output_dir, "experiments.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("config,model_type,learning_rate,output_dir\n")
    
    run_group = f"{config_basename}_lr_sweep"
    # Run trials with different learning rates
    for trial in range(args.num_trials):
        # Generate random learning rate
        lr = generate_random_lr(args.min_lr_exp, args.max_lr_exp)
        
        # Create trial-specific output directory
        trial_output_dir = os.path.join(output_base_dir, f"lr_{lr:.8f}")
        os.makedirs(trial_output_dir, exist_ok=True)
        
        # Construct run name for wandb
        run_name = f"{config_basename}_lr_{lr:.8f}"

        # Build the command
        cmd = [
            PYTHON_EXECUTABLE, "train.py",
            "--configs", config_path,
            "--values",
            f"wandb=True",
            f"output_path={trial_output_dir}",
            f"model_type={model_type}",
            f"{model_type.lower()}.base_lr={lr}",
            f"run_name={run_name}",
            f"run_group={run_group}"
        ]
        
        print(f"\nTrial {trial+1}/{args.num_trials}: Running with learning rate {lr:.8f}")
        print(f"Command: {' '.join(cmd)}")
        
        # Run the command
        process = subprocess.run(cmd)
        
        # Log the experiment
        with open(log_file, "a") as f:
            f.write(f"{config_basename},{model_type},{lr},{trial_output_dir}\n")
        
        if process.returncode != 0:
            print(f"Warning: Command exited with return code {process.returncode}")

if __name__ == "__main__":
    main()