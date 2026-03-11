#!/bin/bash
#SBATCH --job-name=ER_Challenge1c                # Job name
#SBATCH --output=logs/ER_Challenge1c_%j.out      # Standard output file (%j will be replaced by the job ID)
#SBATCH --error=logs/ER_Challenge1c_%j.err       # Standard error file
#SBATCH --nodes=1                                # Number of nodes requested
#SBATCH --ntasks=1                               # Number of tasks requested
#SBATCH --cpus-per-task=64                       # Number of CPU cores per task (adjust based on your parallel environment)
#SBATCH --mem=32G                                # Memory allocation
#SBATCH --time=48:00:00                          # Expected runtime (hours:minutes:seconds)
#SBATCH --partition=academic                     # Partition to submit to academic partition
#SBATCH --account=micro-515                      # Account name

# Activate virtual environment
 source .venv/bin/activate

# Create logs directory
mkdir -p logs

# Run Python script
python Challenge1c.py

