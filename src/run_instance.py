import json
import time
import pandas as pd
import numpy as np
import networkx as nx
from functools import partial
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import mlflow
import argparse


# QOKIT Imports
from qokit.qaoa_circuit_maxcut import get_qaoa_circuit
from qokit.maxcut import maxcut_obj, get_adjacency_matrix
from qokit.utils import brute_force
from qokit.qaoa_objective_maxcut import get_qaoa_maxcut_objective

from graph_instance import create_graphs_from_all_sources
from initialisation import Initialisation
from features import get_graph_features, get_weighted_graph_features
from visualizations import plot_approximation_ratio, plot_facet_approximation_ratio, plot_graph_with_weights, plot_edge_weights_histogram
from utils import make_temp_directory, str2bool, convert_list_attributes_to_string

import logging
logging.basicConfig(level=logging.INFO)

# Start the timer
start_time = time.time()

# Constants
LAYER_INDEPENDENT_METHODS = ['random', 'tqa', 'qibpi']
LAYER_DEPENDENT_METHODS = []
# LAYER_DEPENDENT_METHODS = ['interp', 'fourier']

def create_graph():
    """Generate and return a connected graph instance."""
    # Generate graph instances
    G_instances = create_graphs_from_all_sources(instance_size=NUM_NODES, sources="ALL")
    
    # Filter for the specific graph type
    G_instances = [g for g in G_instances if g.graph_type == GRAPH_TYPE]
    
    # Select the first instance (or another based on your criteria)
    graph_instance = G_instances[0]
    
    # Check if the selected graph instance is connected
    if not nx.is_connected(graph_instance.G):
        print("Generated graph is not connected, generating again.")
        return create_graph()  # Recursively generate another graph if not connected
    
    # Allocate weights to the edges if the graph is connected
    graph_instance.weight_type = WEIGHT_TYPE
    graph_instance.allocate_random_weights()
    
    return graph_instance.G

def get_optimal_cut(G):
    """Compute the optimal cut for the graph G."""
    obj = partial(maxcut_obj, w=get_adjacency_matrix(G))
    return brute_force(obj, G.number_of_nodes(), function_takes="bits")[0]

def run_qaoa(G, init_class, method, layer):
    """Run QAOA with specified initializations, directly to max layers for non-dependent methods."""
    optimal_cut = get_optimal_cut(G)
    init_class.current_layer = layer
    init_params = init_class.init_params_for_layer(method, layer)
    gamma_init, beta_init = init_params[:layer], init_params[layer:]

    logging.info(f"Optimizing for Layer: {layer} with {method} initialization.")

    qc = get_qaoa_circuit(G, gamma_init, beta_init, save_statevector=False)
    qc.measure_all()

    f = get_qaoa_maxcut_objective(G.number_of_nodes(), layer, G, parameterization="theta")
    theta = np.hstack([gamma_init, beta_init])

    evals = {'num_evals': 0, 'iterations': []}
    def callback(xk):
        evals['num_evals'] += 1
        # Each time num_evals is 100; print the current function value
        if evals['num_evals'] % 100 == 0:
            print(f"Function value at iteration {evals['num_evals']}: {f(xk)}")
        evals['iterations'].append({'variables': xk.copy(), 'function_value': f(xk), 'num_evals': evals['num_evals']})

    res = minimize(f, theta, method='SLSQP', options={'disp': True, 'maxiter': 10000}, callback=callback)

    approximation_ratio = -f(res.x) / optimal_cut
    logging.info(f"Approximation ratio with {method} at p = {layer}: {approximation_ratio}")

    return approximation_ratio, evals

def main(**kwargs):
    logging.info("Starting QAOA with different initialization methods")
    
    # Unpack kwargs
    GRAPH_TYPE = kwargs.get("graph_type")
    NUM_NODES = kwargs.get("num_nodes")
    WEIGHT_TYPE = kwargs.get("weight_type")
    N_LAYERS = kwargs.get("n_layers")
    TRACK = kwargs.get("track")
    CUSTOM_GRAPH = kwargs.get("custom_graph")
    

    print(f"Custom Graph: {CUSTOM_GRAPH}")

    if not CUSTOM_GRAPH:
        G = create_graph()
    else:
        # Read the graph from the custom
        logging.info(f"Reading graph from {CUSTOM_GRAPH}")
        G = nx.read_graphml(CUSTOM_GRAPH)
        # Convert the edges to integers
        G = nx.convert_node_labels_to_integers(G)
        # Get the source from the graph attributes
        GRAPH_TYPE = G.graph['instance_class']
        if GRAPH_TYPE == "nearly_complete_bipartite":
            GRAPH_TYPE = "nearly_complete_bi_partite"
        # Get the weight type from the graph attributes
        WEIGHT_TYPE = G.graph['weight_type']
        # Get the number of nodes from the graph attributes
        NUM_NODES = G.number_of_nodes()

    # Print global variables
    logging.info(f"Graph Type: {GRAPH_TYPE}")
    logging.info(f"Number of Nodes: {NUM_NODES}")
    logging.info(f"Weight Type: {WEIGHT_TYPE}")
    logging.info(f"Max Layers: {N_LAYERS}")
    

    graph_features = get_graph_features(G)
    weighted_features = get_weighted_graph_features(G)
    # Log the graph features and format the output as a json
    logging.info("Graph Features:")
    logging.info(json.dumps(graph_features, indent=4))
    # Log the weighted graph features and format the output as a json
    logging.info("Weighted Graph Features:")
    logging.info(json.dumps(weighted_features, indent=4))

    # If tracking is enabled, start the MLflow run
    if TRACK:
        # Set experiment name to be INFORMS 2024 IJOC 
        mlflow.set_experiment("QAOA-Parameter-Initialisation")
        mlflow.start_run(run_name=f"{GRAPH_TYPE} Graph")
        mlflow.log_params(graph_features)
        mlflow.log_params(weighted_features)
        mlflow.log_param("num_nodes", NUM_NODES)
        mlflow.log_param("graph_type", GRAPH_TYPE)
        mlflow.log_param("weight_type", WEIGHT_TYPE)
        mlflow.log_param("max_layers", N_LAYERS)
        # if custom graph is used, log the path
        if CUSTOM_GRAPH:
            mlflow.log_param("custom_graph", CUSTOM_GRAPH)
            mlflow.log_param("evolved_instance", True)


    init_class = Initialisation(num_qubits=NUM_NODES, max_layers=N_LAYERS, source=GRAPH_TYPE, weight_type=WEIGHT_TYPE)
    
    results = {}
    evaluations = {}
    # Run QAOA for non-layer dependent methods at maximum layer only
    if len(LAYER_INDEPENDENT_METHODS) > 0:
        for method in LAYER_INDEPENDENT_METHODS:
            approximation_ratio, evals = run_qaoa(G, init_class, method, N_LAYERS)
            results[method] = approximation_ratio
            evaluations[method] = evals

    if len(LAYER_DEPENDENT_METHODS) > 0:
        for method in LAYER_DEPENDENT_METHODS:
            for layer in range(1, N_LAYERS + 1):
                key = f"{method}_p{layer}"
                approximation_ratio, evals = run_qaoa(G, init_class, method, layer)            
                results[key] = approximation_ratio
                evaluations[key] = evals

    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
    logging.info("Results:")
    for k, v in sorted_results.items():
        logging.info(f"{k}: {v}")

    # Flattening the data structure for easier analysis
    rows = []
    for method, data in evaluations.items():
        for iteration in data['iterations']:
            row = {
                'method': method,
                'num_evals': iteration['num_evals'],
                'function_value': iteration['function_value'],
                'variables': iteration['variables']
            }
            rows.append(row)

    # Creating a DataFrame
    df = pd.DataFrame(rows)
    # Add column for approximation ratio
    df['approximation_ratio'] = -df['function_value'] / get_optimal_cut(G)
    # First, create a new column that generalizes the method names to 'interp' or 'fourier'
    df['new_method'] = df['method'].replace(to_replace=r'^(interp|fourier)_.*$', value=r'\1', regex=True)
    # Use this new column to compute cumulative counts for the `num_evals` for each group
    df['num_evals'] = df.groupby('new_method').cumcount() + 1
    
    # Find which method has the best approximation ratio
    best_method = df.groupby('new_method')['approximation_ratio'].max().idxmax()
    logging.info(f"Best method: {best_method}")
    # Find the approximation ratio of the best method
    best_approximation_ratio = df[df['new_method'] == best_method]['approximation_ratio'].max()
    
    # Acceptable approximation ratio
    acceptable_ratio = 0.99*best_approximation_ratio
    # Find the number of evaluations each method took to reach the acceptable ratio and store in a dictionary
    metrics = {}
    for method in df['new_method'].unique():
        num_evals = df[(df['new_method'] == method) & (df['approximation_ratio'] >= acceptable_ratio)]['num_evals'].min()

        if num_evals is np.nan:
            # Allocate penalty value if the method did not reach the acceptable ratio, the penalty value is 2^NUM_NODES
            logging.info(f"Method: {method} did not reach an approximation ratio of {acceptable_ratio}.")
            metrics[method] = 100000
            if TRACK:
                # Log the penalty value as True
                mlflow.log_param(f"penalty_{method}", True)
        else:
            logging.info(f"Method: {method} reached an approximation ratio of {acceptable_ratio} in {num_evals} evaluations.")
            metrics[method] = num_evals
            if TRACK:
                # Log the penalty value as False
                mlflow.log_param(f"penalty_{method}", False)
    
    logging.info(f"Metrics: {metrics}")

    ## Create plots
    with make_temp_directory() as temp_dir:
        # Plot approximation ratio
        plt1 = plot_approximation_ratio(df, acceptable_ratio)
        plt1.savefig(f'{temp_dir}/iteration_vs_approx_ratio.png', dpi=300)
        plt1.clf()

        # Plot facet approximation ratio
        g = plot_facet_approximation_ratio(df, acceptable_ratio)
        g.savefig(f'{temp_dir}/facet_iteration_vs_approx_ratio.png', dpi=300)
        plt.clf()

        # Plot graph with weights
        plt3 = plot_graph_with_weights(G)
        plt3.savefig(f'{temp_dir}/graph_with_weights.png', dpi=300)
        plt3.clf()

        # Plot edge weights histogram
        plt4 = plot_edge_weights_histogram(G, WEIGHT_TYPE)
        if plt4 is not None:
            plt4.savefig(f'{temp_dir}/edge_weights_histogram.png', dpi=300)
            plt4.clf()

        convert_list_attributes_to_string(G)

        # Save networkx graph as a GraphML file
        nx.write_graphml(G, f'{temp_dir}/graph.graphml')

        # Find the highest value of the approximation ratio -- and its row
        max_approximation_ratio = df['approximation_ratio'].max() 
        max_row = df[df['approximation_ratio'] == max_approximation_ratio]
        # If more than one row take the highest number of evaluations
        if len(max_row) > 1:
            max_row = max_row[max_row['num_evals'] == max_row['num_evals'].max()]
        # Extract the variables from the row
        max_variables = max_row['variables'].values
        # Create a dict (gamma_i, beta_i) for each layer and convert to float
        max_variables_dict = {f"gamma_{i}": float(max_variables[0][i]) for i in range(N_LAYERS)}
        # Make sure beta also starts from 0
        max_variables_dict.update({f"beta_{i - N_LAYERS}": float(max_variables[0][i]) for i in range(N_LAYERS, 2*N_LAYERS)})
        
        # Save the DataFrame as a CSV file
        df.to_csv(f'{temp_dir}/evaluations.csv', index=False)

        # Add the plots to the MLflow run
        if TRACK:
            mlflow.log_artifacts(temp_dir)

    # Add algo prefix to the metrics
    metrics = {f"algo_{k}": float(v) for k, v in metrics.items()}
    # Get the approximation ratio for each method
    approximation_ratio = {f"approximation_ratio_{k}": v for k, v in results.items()}
    # Get max func evaluations for each method
    max_func_evals = {f"max_func_evals_{k}": max([iteration['num_evals'] for iteration in data['iterations']]) for k, data in evaluations.items()}

    # If tracking is enabled, log the metrics
    if TRACK:
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(approximation_ratio)
        mlflow.log_metrics(max_func_evals)
        # Log optimal parameters
        mlflow.log_metrics(max_variables_dict)
        mlflow.end_run()

    # json dump the metrics
    logging.info(json.dumps(metrics, indent=4))
    logging.info(json.dumps(approximation_ratio, indent=4))
    logging.info(json.dumps(max_func_evals, indent=4))
    logging.info("best_method: " + best_method)
    logging.info(json.dumps(max_variables_dict, indent=4))



if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Running QAOA for a single instance")
    parser.add_argument("--track", type=str2bool, default=False, help="Enable or disable tracking of the process")
    parser.add_argument("--graph_type", type=str, default="Uniform Random", help="The type of graph to generate")
    parser.add_argument("--num_nodes", type=int, default=8, help="The number of nodes in the graph")
    parser.add_argument("--weight_type", type=str, default=None, help="The type of weights to allocate to the edges")
    parser.add_argument("--n_layers", type=int, default=15, help="The maximum number of layers to run QAOA")
    # Add argument for a custom_graph file
    parser.add_argument("--custom_graph", type=str, default=None, help="Path to a custom graph file")

    args = parser.parse_args()
    GRAPH_TYPE = args.graph_type
    NUM_NODES = args.num_nodes
    WEIGHT_TYPE = args.weight_type
    N_LAYERS = args.n_layers
    TRACK = args.track
    CUSTOM_GRAPH = args.custom_graph

    main(
        graph_type = GRAPH_TYPE,
        num_nodes = NUM_NODES,
        weight_type = WEIGHT_TYPE,
        n_layers = N_LAYERS,
        track = TRACK,
        custom_graph = CUSTOM_GRAPH
    )
    # End the timer
    end_time = time.time()
    logging.info(f"Time taken: {end_time - start_time} seconds")
