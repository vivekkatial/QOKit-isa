#!/bin/bash

# Define the file path to the evolved graph types
inst_directory="data/isa-instances"

# Loop over each node count
i=1
for file in "$inst_directory"/*; do
    # Extract the instance name from the file path
    instance=$(basename "$file")
    # Format the logfile name with the instance
    logfile="logs/run-$instance-%x-%j-%N.out"
    # Submit the Slurm job for each combination with logging
    echo "Run $i: Submitting job for file: $file"
    sbatch --output="$logfile" bin/run_instances_n_layers.slurm "$file"
    ((i++))
done
