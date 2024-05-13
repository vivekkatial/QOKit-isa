import time
import numpy as np
import networkx as nx
import pandas as pd
from functools import partial

# QOKIT Imports
from qokit.qaoa_circuit_maxcut import get_qaoa_circuit
from qokit.maxcut import maxcut_obj, get_adjacency_matrix
from qokit.utils import brute_force, objective_from_counts, invert_counts
from scipy.optimize import minimize
from qokit.qaoa_objective_maxcut import get_qaoa_maxcut_objective

# Custom imports
from initialisation import Initialisation
from graph_instance import create_graphs_from_all_sources
from utils import to_snake_case

import logging
logging.basicConfig(level=logging.INFO)


# Parameters for the graph and QAOA
graph_type = "3-Regular Graph"
N = 8
N_LAYERS = 3

# Generate a random regular graph
G_instances = create_graphs_from_all_sources(instance_size=N, sources="ALL")
G_instances = [g for g in G_instances if g.graph_type == graph_type]
graph_instance = G_instances[0]
G = graph_instance.G
num_nodes = G.number_of_nodes()

# Get the optimal cut
obj = partial(maxcut_obj, w=get_adjacency_matrix(G))
optimal_cut = brute_force(obj, num_nodes, function_takes="bits")


def run_qaoa(init_class, init_type):
    best_ratio = 0
    # Initialize parameters for the first layer to start the loop
    init_class.current_layer = 1  # Ensure starting from the first layer
    init_params = init_class.init_params_for_layer(init_type)

    for layer in range(1, N_LAYERS + 1):
        # Update parameters for the current layer number
        if layer > 1:
            init_class.current_layer = layer  # Set the current layer explicitly
            init_params = init_class.init_params_for_layer(init_type)  # Initialize for the current layer

        # Split parameters based on the current layer
        gamma_init, beta_init = init_params[:layer], init_params[layer:]

        logging.info(f"Optimising for Layer: {layer}/{N_LAYERS}")
        qc = get_qaoa_circuit(G, gamma_init, beta_init, save_statevector=False)
        qc.measure_all()

        f = get_qaoa_maxcut_objective(num_nodes, layer, G, parameterization="theta")
        theta = np.hstack([gamma_init, beta_init])

        res = minimize(f, theta, method='SLSQP', options={'disp': True})
        approximation_ratio = -f(res.x) / optimal_cut[0]
        best_ratio = max(best_ratio, approximation_ratio)

        logging.info(f"Approximation ratio after optimization at p = {layer}: {approximation_ratio}")
    return best_ratio


# Create an instance of the Initialization class
init_class = Initialisation(num_qubits=N, max_layers=N_LAYERS, source=graph_instance.graph_type)

# Start timer
logging.info("Starting QAOA with different initialization methods")
start_time = time.time()


# Run QAOA using different initialization methods for each layer
logging.info("Running QAOa with different initialization methods")

logging.info("Running QAOA with random initialization")
random_best_ratio = run_qaoa(init_class, 'random')
logging.info(f"Best approximation ratio with random initialization: {random_best_ratio}")

logging.info("Running QAOA with Fixed Angle Constant initialization")
fixed_best_ratio = run_qaoa(init_class, 'fixed_angles_constant')
logging.info(f"Best approximation ratio with Fixed Angle Constant initialization: {fixed_best_ratio}")

logging.info("Running QAOA with TQA initialization")
tqa_best_ratio = run_qaoa(init_class, 'tqa')
logging.info(f"Best approximation ratio with TQA initialization: {tqa_best_ratio}")

logging.info("Running QAOA with qibpi initialization")
qibpi_best_ratio = run_qaoa(init_class, 'qibpi')
logging.info(f"Best approximation ratio with qibpi initialization: {qibpi_best_ratio}")

logging.info("Running QAOA with interp initialization")
interp_best_ratio = run_qaoa(init_class, 'interp')
logging.info(f"Best approximation ratio with interp initialization: {interp_best_ratio}")


logging.info("Running QAOA with fourier initialization")
fourier_best_ratio = run_qaoa(init_class, 'fourier')
logging.info(f"Best approximation ratio with fourier initialization: {fourier_best_ratio}")


# Print out formatted results (ordered by best approximation ratio)
results = {
    "Random": random_best_ratio,
    "Constant Fixed Angles": fixed_best_ratio,
    "TQA": tqa_best_ratio,
    "QIBPI": qibpi_best_ratio,
    "Interp": interp_best_ratio,
    "Fourier": fourier_best_ratio
}

sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
logging.info("Results:")
for k, v in sorted_results.items():
    logging.info(f"{k}: {v}")

# End script and state time taken
end_time = time.time()
# Format time taken
time_taken = end_time - start_time
logging.info(f"Time taken: {time_taken} seconds")
logging.info("Finished running QAOA with different initialization methods")