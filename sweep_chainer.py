#!/usr/bin/env python3
"""
chain_jobs.py

Usage:
    python chain_jobs.py <combination> <num_trials_job1> [<num_trials_job2> ...]

This script submits a chain of SLURM jobs by calling the run_sweeps.sh script with
the specified combination (fixed or changing) and each provided num_trials value.
Each job is submitted so that it will only start after the previous job completes successfully.
"""

import argparse
import subprocess
import sys

def submit_job(script, args, dependency=None):
    """
    Submits a job via sbatch, optionally setting a dependency.
    Returns the submitted job ID.
    """
    cmd = ["sbatch"]
    if dependency:
        cmd.append("--dependency=afterok:" + dependency)
    cmd.append(script)
    cmd.extend(args)
    
    print("Submitting: " + " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("Error submitting job:", result.stderr)
        sys.exit(1)
    # Expect output of the form "Submitted batch job <jobid>"
    out = result.stdout.strip()
    job_id = out.split()[-1]
    return job_id

def main():
    parser = argparse.ArgumentParser(
        description="Chain SLURM job submissions for mega_sweep.sh with dependency chaining."
    )
    parser.add_argument("combination", choices=["fixed", "changing"],
                        help="Combination type: fixed or changing")
    parser.add_argument("num_trials", nargs="+",
                        help="A list of num_trials values for each job in the chain, e.g. 30 20 for 60 then 40 sweeps.")
    parser.add_argument("model_type", choices=["hjepa", "vjepa"],
                        help="Model type: hjepa or vjepa")
    args = parser.parse_args()
    
    dependency = None  # No dependency for the first job
    
    for i, num_trials in enumerate(args.num_trials, start=1):
        print(f"Submitting job {i} for combination '{args.combination}' with num_trials={num_trials} ...")
        job_id = submit_job("slurm_mega_sweep.sh", [args.combination, num_trials, args.model_type], dependency=dependency)
        print(f"Job {i} submitted with job ID {job_id}")
        # Set dependency for the next job to be after this job completes successfully
        dependency = job_id

if __name__ == "__main__":
    main()
