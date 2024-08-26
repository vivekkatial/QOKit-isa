# Define the array of graph types
declare -a graph_types=("Nearly Complete BiPartite" "Uniform Random" "Power Law Tree" "Watts-Strogatz small world" "3-Regular Graph" "4-Regular Graph" "Geometric")

# Define the array of weight types including None
declare -a weight_types=("uniform" "uniform_plus" "normal" "exponential" "log-normal" "cauchy")

# Define the specific layer counts
declare -a layer_counts=(11 12 13 14 15 16 17 18 19 20)

# Repeat the entire process 49 times
for i in $(seq 1 20); do
    # Loop over each graph type and weight type
    for graph_type in "${graph_types[@]}"; do
        for weight_type in "${weight_types[@]}"; do
            # Loop over the specified layer counts
            for layers in "${layer_counts[@]}"; do
                # Submit the Slurm job for each combination with logging
                echo "Submitting job for graph type: $graph_type, weight type: $weight_type, layers: $layers, iteration: $i"
                sbatch --output=logs/%x-%j-%N.out bin/build_qibpi.slurm "$graph_type" "$weight_type" "$layers"
            done
        done
    done
done