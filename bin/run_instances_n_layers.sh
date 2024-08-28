#!/bin/bash

# Define the file path to the evolved graph types
inst_directory="data/isa-instances"

# Loop over each node count
for file in "$inst_directory"/*; do
    # Submit the Slurm job for each combination with logging
    echo "Run $i: Submitting job for file: $file"
    # sbatch --output=logs/run-$file-%x-%j-%N.out bin/run_evolved_instance.slurm "$file"
done
