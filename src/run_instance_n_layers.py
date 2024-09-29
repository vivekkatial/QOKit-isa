import json
import time
import pandas as pd
import numpy as np
import networkx as nx
from functools import partial
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import argparse
import os
import requests
from dotenv import load_dotenv

# QOKIT Imports
from qokit.qaoa_circuit_maxcut import get_qaoa_circuit
from qokit.maxcut import maxcut_obj, get_adjacency_matrix
from qokit.utils import brute_force
from qokit.qaoa_objective_maxcut import get_qaoa_maxcut_objective

from graph_instance import create_graphs_from_all_sources, GraphInstance
from initialisation import Initialisation
from features import get_graph_features, get_weighted_graph_features
from visualizations import plot_approximation_ratio, plot_graph_with_weights, plot_edge_weights_histogram
from utils import make_temp_directory, str2bool, convert_list_attributes_to_string

import logging
logging.basicConfig(level=logging.INFO)

# Start the timer
start_time = time.time()

def extract_and_save_optimized_parameters(results, evaluations, output_file='optimized_parameters.json'):
    optimized_parameters = []
    
    for layer, data in evaluations.items():
        # Get the last iteration, which should be the optimized result
        optimized_result = data['iterations'][-1]
        
        # Extract the variables (which contain both gamma and beta)
        variables = optimized_result['variables']
        
        # The number of gamma and beta values is equal to the layer number
        layer_number = int(layer.split('_')[1])
        
        # Split the variables into gamma and beta
        gamma = variables[:layer_number].tolist()
        beta = variables[layer_number:].tolist()
        
        # Create a dictionary for this layer
        layer_data = {
            "layer": layer_number,
            "gamma": gamma,
            "beta": beta
        }
        
        optimized_parameters.append(layer_data)
    
    # Sort the results by layer number
    optimized_parameters.sort(key=lambda x: x['layer'])
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(optimized_parameters, f, indent=2)
    
    print(f"Optimized parameters saved to {output_file}")
    
    return optimized_parameters

load_dotenv()
BASE_URL = "http://115.146.94.114:5000"
AUTH = (os.environ.get("BASIC_AUTH_USERNAME"), os.environ.get("BASIC_AUTH_PASSWORD"))

# Have a mapping that maps the graph type what the API expects
GRAPH_TYPE_MAPPING = {
    "Uniform Random": "uniform_random",
    "Nearly Complete BiPartite": "nearly_complete_bi_partite",
    "Erdos Renyi": "uniform_random",
    "Watts-Strogatz small world": "watts_strogatz_small_world",
    "Geometric": "geometric",
    "Power Law Tree": "power_law_tree",
    "3-Regular Graph": "three_regular_graph",
    "4-Regular Graph": "four_regular_graph"
}


def make_request(endpoint, data):
    """
    Make a request to the API endpoint with the given data.
    """
    url = f"{BASE_URL}{endpoint}"
    response = requests.post(url, json=data, auth=AUTH)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

def initialize_using_qibpi(G, layer):
    """Initialize the QAOA parameters using QIBPI."""
    endpoint = '/graph/QIBPI'
    # Prepare the data for the API request
    adj_matrix = nx.to_numpy_array(G).tolist()
    graph_type = GRAPH_TYPE_MAPPING.get(G.graph.get('graph_type'), "None")
    weight_type = G.graph.get('weight_type')
    data = {
        "adjacency_matrix": adj_matrix,
        "p": layer,
        "graph_type": graph_type,
        "weight_type": weight_type
    }
    
    result = make_request(endpoint, data)
    if result is not None and 'beta' in result and 'gamma' in result:
        gamma_init = result['gamma']
        beta_init = result['beta']
        return gamma_init, beta_init
    else:
        # Fallback to random initialization if the API request fails
        print("QIBPI initialization failed. Falling back to random initialization.")
        gamma_init = np.random.uniform(0, 2*np.pi, layer)
        beta_init = np.random.uniform(0, np.pi, layer)
        return gamma_init, beta_init

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

def run_qaoa(G, layer):
    """Run QAOA with QIBPI initialization for a specific number of layers."""
    optimal_cut = get_optimal_cut(G)
    # Use QIBPI initialization
    gamma_init, beta_init = initialize_using_qibpi(G, layer)

    logging.info(f"Optimizing for Layer: {layer} with QIBPI initialization.")

    qc = get_qaoa_circuit(G, gamma_init, beta_init, save_statevector=False)
    qc.measure_all()

    f = get_qaoa_maxcut_objective(G.number_of_nodes(), layer, G, parameterization="theta")
    theta = np.hstack([gamma_init, beta_init])

    evals = {'num_evals': 0, 'iterations': []}
    def callback(xk):
        evals['num_evals'] += 1
        if evals['num_evals'] % 100 == 0:
            print(f"Function value at iteration {evals['num_evals']}: {f(xk)}")
        evals['iterations'].append({'variables': xk.copy(), 'function_value': f(xk), 'num_evals': evals['num_evals']})

    res = minimize(f, theta, method='SLSQP', options={'disp': True, 'maxiter': 10000}, callback=callback)

    approximation_ratio = -f(res.x) / optimal_cut
    logging.info(f"Approximation ratio at p = {layer}: {approximation_ratio}")

    return approximation_ratio, evals, res

def main(**kwargs):
    logging.info("Starting QAOA layer exploration")
    # Unpack the keyword arguments
    NUM_NODES = kwargs.get("NUM_NODES", 8)
    WEIGHT_TYPE = kwargs.get("WEIGHT_TYPE", "uniform")
    GRAPH_TYPE = kwargs.get("GRAPH_TYPE", "Uniform Random")
    MAX_LAYERS = kwargs.get("MAX_LAYERS", 2)
    TRACK = kwargs.get("TRACK", False)
    CUSTOM_GRAPH = kwargs.get("CUSTOM_GRAPH", None)

    if not CUSTOM_GRAPH:
        G = create_graph()
    else:
        logging.info(f"Reading graph from {CUSTOM_GRAPH}")
        G = nx.read_graphml(CUSTOM_GRAPH)
        G = nx.convert_node_labels_to_integers(G)
        GRAPH_TYPE = G.graph['graph_type']
        if GRAPH_TYPE == "nearly_complete_bipartite":
            GRAPH_TYPE = "nearly_complete_bi_partite"
        WEIGHT_TYPE = G.graph['weight_type']
        # Check that there are weights for the edges
        if not all([G[u][v].get('weight') for u, v in G.edges]):
            # if not, check if a weight type is specified
            if not WEIGHT_TYPE:
                raise ValueError("All edges must have weights and no weight type specified")
            else:
                logging.warn(f"Weight type specified: {WEIGHT_TYPE}")
                logging.warn("Assigning weights to the edges")
                G_inst = GraphInstance(
                    G = G,
                    graph_type = GRAPH_TYPE,
                    weight_type = WEIGHT_TYPE
                )
                G_inst.allocate_random_weights()
                G = G_inst.G
        NUM_NODES = G.number_of_nodes()

    logging.info(f"Number of Nodes: {NUM_NODES}")
    logging.info(f"Graph Type: {GRAPH_TYPE}")
    logging.info(f"Weight Type: {WEIGHT_TYPE}")
    logging.info(f"Max Layers: {MAX_LAYERS}")
    graph_features = get_graph_features(G)
    weighted_features = get_weighted_graph_features(G)
    logging.info("Graph Features:")
    logging.info(json.dumps(graph_features, indent=4))
    logging.info("Weighted Graph Features:")
    logging.info(json.dumps(weighted_features, indent=4))

    if TRACK:
        mlflow.set_experiment("QAOA-Layer-Exploration")
        mlflow.start_run(run_name=f"{GRAPH_TYPE} Graph")
        mlflow.log_param("initialization", "QIBPI")
        mlflow.log_params(graph_features)
        mlflow.log_params(weighted_features)
        mlflow.log_param("num_nodes", NUM_NODES)
        mlflow.log_param("graph_type", GRAPH_TYPE)
        mlflow.log_param("weight_type", WEIGHT_TYPE)
        mlflow.log_param("max_layers", MAX_LAYERS)
        if CUSTOM_GRAPH:
            mlflow.log_param("custom_graph", CUSTOM_GRAPH)
            mlflow.log_param("evolved_instance", True)

    results = {}
    evaluations = {}
    fevals = {}
    
    for layer in range(1, MAX_LAYERS + 1):
        approximation_ratio, evals, res = run_qaoa(G, layer)
        results[f"layer_{layer}"] = approximation_ratio
        evaluations[f"layer_{layer}"] = evals
        fevals[f"layer_{layer}"] = res.nfev
        
    optimized_parameters = extract_and_save_optimized_parameters(results, evaluations)

    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
    logging.info("Results:")
    for k, v in sorted_results.items():
        logging.info(f"{k}: {v}")

    rows = []
    for layer, data in evaluations.items():
        for iteration in data['iterations']:
            row = {
                'layer': layer,
                'num_evals': iteration['num_evals'],
                'function_value': iteration['function_value'],
                'variables': iteration['variables']
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df['approximation_ratio'] = -df['function_value'] / get_optimal_cut(G)
    df['layer'] = df['layer'].str.extract('(\d+)').astype(int)

    best_layer = df.groupby('layer')['approximation_ratio'].max().idxmax()
    logging.info(f"Best layer: {best_layer}")
    best_approximation_ratio = df[df['layer'] == best_layer]['approximation_ratio'].max()
    
    acceptable_ratio = 0.99 * best_approximation_ratio
    metrics = {}
    for layer in range(1, MAX_LAYERS + 1):
        layer_df = df[df['layer'] == layer]
        num_evals = layer_df[layer_df['approximation_ratio'] >= acceptable_ratio]['num_evals'].min()

        if pd.isna(num_evals):
            logging.info(f"Layer {layer} did not reach an approximation ratio of {acceptable_ratio}.")
            metrics[f"algo_layer_{layer}"] = 100000  # Penalty value
            if TRACK:
                mlflow.log_param(f"penalty_layer_{layer}", True)
        else:
            logging.info(f"Layer {layer} reached an approximation ratio of {acceptable_ratio} in {num_evals} evaluations.")
            metrics[f"algo_layer_{layer}"] = num_evals
            if TRACK:
                mlflow.log_param(f"penalty_layer_{layer}", False)

    logging.info(f"Metrics: {metrics}")

    with make_temp_directory(local=False) as temp_dir:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='layer', y='approximation_ratio')
        plt.title('Approximation Ratio vs Number of Layers')
        plt.xlabel('Number of Layers')
        plt.ylabel('Approximation Ratio')
        plt.savefig(f'{temp_dir}/layer_vs_approx_ratio.png', dpi=300)
        plt.clf()

        plt_graph = plot_graph_with_weights(G)
        plt_graph.savefig(f'{temp_dir}/graph_with_weights.png', dpi=300)
        plt_graph.clf()

        plt_hist = plot_edge_weights_histogram(G, WEIGHT_TYPE)
        if plt_hist is not None:
            plt_hist.savefig(f'{temp_dir}/edge_weights_histogram.png', dpi=300)
            plt_hist.clf()

        convert_list_attributes_to_string(G)
        nx.write_graphml(G, f'{temp_dir}/graph.graphml')

        df.to_csv(f'{temp_dir}/evaluations.csv', index=False)
        
        # Save the optimized parameters to a JSON file
        with open(f'{temp_dir}/optimized_parameters.json', 'w') as f:
            json.dump(optimized_parameters, f, indent=2)

        if TRACK:
            mlflow.log_artifacts(temp_dir)

    # Extract the AR for each layer and log it
    metrics = {f"approx_ratio_layer_{k}": v for k, v in results.items()}
    # Extract the maximum number of function evaluations for each layer and log it
    max_func_evals = {f"max_func_evals_layer_{k}": max([iteration['num_evals'] for iteration in data['iterations']]) for k, data in evaluations.items()}
    # Fevals extraction code
    metrics_fevals = {f"fevals_{k}": v for k, v in fevals.items()}

    if TRACK:
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(max_func_evals)
        mlflow.log_metrics(metrics_fevals)
        mlflow.log_metric("best_layer", best_layer)
        mlflow.log_metric("best_approximation_ratio", best_approximation_ratio)
        mlflow.end_run()

    logging.info(json.dumps(metrics, indent=4))
    logging.info(json.dumps(max_func_evals, indent=4))
    logging.info(f"Best layer: {best_layer}")
    logging.info(f"Best approximation ratio: {best_approximation_ratio}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running QAOA layer exploration for a single instance")
    parser.add_argument("--track", type=str2bool, default=False, help="Enable or disable tracking of the process")
    parser.add_argument("--graph_type", type=str, default="Uniform Random", help="The type of graph to generate")
    parser.add_argument("--num_nodes", type=int, default=8, help="The number of nodes in the graph")
    parser.add_argument("--weight_type", type=str, default='uniform', help="The type of weights to allocate to the edges")
    parser.add_argument("--max_layers", type=int, default=2, help="The maximum number of layers to explore")
    parser.add_argument("--custom_graph", type=str, default=None, help="Path to a custom graph file")

    args = parser.parse_args()
    NUM_NODES = args.num_nodes
    WEIGHT_TYPE = args.weight_type
    GRAPH_TYPE = args.graph_type
    MAX_LAYERS = args.max_layers
    TRACK = args.track
    CUSTOM_GRAPH = args.custom_graph

    main(
        NUM_NODES=NUM_NODES,
        WEIGHT_TYPE=WEIGHT_TYPE,
        GRAPH_TYPE=GRAPH_TYPE,
        MAX_LAYERS=MAX_LAYERS,
        TRACK=TRACK,
        CUSTOM_GRAPH=CUSTOM_GRAPH
    )
    end_time = time.time()
    logging.info(f"Time taken: {end_time - start_time} seconds")