#!/bin/bash
#SBATCH --job-name=starter
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16g
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# The account is passed on the command line, not set here:
#     sbatch --account=<your account ending in -gpu> job.slurm
# Run `accounts` on the login node to see yours.

module reset
module load pytorch-conda
module list

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

srun python3 train.py --out results

echo "finished $(date)"
