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
from qokit.qaoa_maxcut_objective import get_qaoa_maxcut_objective

from graph_instance import create_graphs_from_all_sources
from initialisation import Initialisation
from features import get_graph_features, get_weighted_graph_features
from visualizations import plot_approximation_ratio, plot_graph_with_weights, plot_edge_weights_histogram
from utils import make_temp_directory, str2bool, convert_list_attributes_to_string

import logging
logging.basicConfig(level=logging.INFO)

# Start the timer
start_time = time.time()

def create_graph():
    """Generate and return a connected graph instance."""
    G_instances = create_graphs_from_all_sources(instance_size=NUM_NODES, sources="ALL")
    G_instances = [g for g in G_instances if g.graph_type == GRAPH_TYPE]
    graph_instance = G_instances[0]
    
    if not nx.is_connected(graph_instance.G):
        print("Generated graph is not connected, generating again.")
        return create_graph()
    
    graph_instance.weight_type = WEIGHT_TYPE
    graph_instance.allocate_random_weights()
    
    return graph_instance.G

def get_optimal_cut(G):
    """Compute the optimal cut for the graph G."""
    obj = partial(maxcut_obj, w=get_adjacency_matrix(G))
    return brute_force(obj, G.number_of_nodes(), function_takes="bits")[0]

def run_qaoa(G, layer):
    """Run QAOA with random initialization for a specific number of layers."""
    optimal_cut = get_optimal_cut(G)
    
    # Random initialization
    gamma_init = np.random.uniform(0, 2*np.pi, layer)
    beta_init = np.random.uniform(0, np.pi, layer)

    logging.info(f"Optimizing for Layer: {layer} with random initialization.")

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

    return approximation_ratio, evals

def main():
    logging.info("Starting QAOA layer exploration")

    print(f"Custom Graph: {CUSTOM_GRAPH}")

    if not CUSTOM_GRAPH:
        G = create_graph()
    else:
        logging.info(f"Reading graph from {CUSTOM_GRAPH}")
        G = nx.read_graphml(CUSTOM_GRAPH)
        G = nx.convert_node_labels_to_integers(G)
        GRAPH_TYPE = G.graph['instance_class']
        if GRAPH_TYPE == "nearly_complete_bipartite":
            GRAPH_TYPE = "nearly_complete_bi_partite"
        WEIGHT_TYPE = G.graph['weight_type']
        NUM_NODES = G.number_of_nodes()

    logging.info(f"Graph Type: {GRAPH_TYPE}")
    logging.info(f"Number of Nodes: {NUM_NODES}")
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
    
    for layer in range(1, MAX_LAYERS + 1):
        approximation_ratio, evals = run_qaoa(G, layer)
        results[f"layer_{layer}"] = approximation_ratio
        evaluations[f"layer_{layer}"] = evals

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

    with make_temp_directory() as temp_dir:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='layer', y='approximation_ratio', ci='sd')
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

        if TRACK:
            mlflow.log_artifacts(temp_dir)

    metrics = {f"approx_ratio_layer_{k}": v for k, v in results.items()}
    max_func_evals = {f"max_func_evals_layer_{k}": max([iteration['num_evals'] for iteration in data['iterations']]) for k, data in evaluations.items()}

    if TRACK:
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(max_func_evals)
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
    parser.add_argument("--weight_type", type=str, default=None, help="The type of weights to allocate to the edges")
    parser.add_argument("--max_layers", type=int, default=2, help="The maximum number of layers to explore")
    parser.add_argument("--custom_graph", type=str, default=None, help="Path to a custom graph file")

    args = parser.parse_args()
    GRAPH_TYPE = args.graph_type
    NUM_NODES = args.num_nodes
    WEIGHT_TYPE = args.weight_type
    MAX_LAYERS = args.max_layers
    TRACK = args.track
    CUSTOM_GRAPH = args.custom_graph

    main()
    end_time = time.time()
    logging.info(f"Time taken: {end_time - start_time} seconds")