#!/bin/bash
# Build the "racoon" conda env on a SLURM CPU node (not on the login node).
# Submit from the RACOON repo root:  mkdir -p log && sbatch scripts/run_setup_env.sbatch
#SBATCH --job-name=racoon_env
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=log/setup_env.out

set -e
echo "host=$(hostname) date=$(date)"
bash scripts/setup_racoon_env.sh
echo "finished $(date)"
