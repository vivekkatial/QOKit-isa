#!/bin/bash

#SBATCH --job-name=run-experiment
###SBATCH -p cloud

# For R need to run on single CPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Set your minimum acceptable walltime, format=hours:minutes:seconds
#SBATCH --time=0-01:00:00  

# Specify your email address to be notified of progress.
### SBATCH --mail-user=your-email-address
### SBATCH --mail-type=ALL

module purge
module load GCCcore/11.3.0
module load Apptainer/1.1.8

# Load in environment modules
source ~/.bashrc

# The script expects two arguments: graph_type and weight_type
echo "Running instance for graph type: $1, weight type: $2, and $3 layers"
apptainer exec \
    --env-file .env \
    --pwd /mnt \
    --bind $(pwd):/mnt \
    qaoa.sif python src/run_instance.py \
    --track True \
    --graph_type "$1" \
    --num_nodes 8 \
    --weight_type "$2" \
    --n_layers $3