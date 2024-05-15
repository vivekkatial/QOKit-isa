#!/bin/bash

# Define the array of graph types
declare -a graph_types=("Nearly Complete BiPartite" "Uniform Random" "Power Law Tree" "Watts-Strogatz small world" "3-Regular Graph" "4-Regular Graph" "Geometric")

# Define the array of weight types including None
declare -a weight_types=("uniform" "uniform_plus" "normal" "exponential" "log-normal" "cauchy" "None")

# Loop over each graph type and weight type
for graph_type in "${graph_types[@]}"; do
    for weight_type in "${weight_types[@]}"; do
        # Submit the Slurm job for each combination with logging
        echo "Submitting job for graph type: $graph_type, weight type: $weight_type"
        sbatch --output=logs/%x-%j-%N.out bin/build_qibpi.slurm "$graph_type" "$weight_type"
    done
done