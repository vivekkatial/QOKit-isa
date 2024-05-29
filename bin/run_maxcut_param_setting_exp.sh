#!/bin/bash

# Define the array of graph types
declare -a graph_types=("Nearly Complete BiPartite" "Uniform Random" "Power Law Tree" "Watts-Strogatz small world" "3-Regular Graph" "4-Regular Graph" "Geometric")

# Define the array of weight types including None
declare -a weight_types=("uniform" "uniform_plus" "normal" "exponential" "log-normal" "cauchy" "None")

# Define the array of num_nodes
declare -a node_counts=(9 11 13)

# Define the number of repetitions per node size
declare -i num_repeats=30

# Loop over each node count
for num_nodes in "${node_counts[@]}"; do
    # Repeat job submissions 100 times for each node count
    for (( i=1; i<=num_repeats; i++ )); do
        # Loop over each combination of graph type and weight type
        for graph_type in "${graph_types[@]}"; do
            for weight_type in "${weight_types[@]}"; do
                # Submit the Slurm job for each combination with logging
                echo "Run $i: Submitting job for graph type: $graph_type, weight type: $weight_type, num_nodes: $num_nodes"
                sbatch --output=logs/run-$num_nodes-$i-%x-%j-%N.out bin/run_maxcut_instance.slurm "$graph_type" "$weight_type" "$num_nodes"
            done
        done
    done
done
