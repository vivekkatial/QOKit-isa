import numpy as np
import networkx as nx
import math
from typing import List, Optional
import pandas as pd
from utils import to_snake_case

# Load the optimal parameters DataFrame from the csv file
optimal_parameters_df = pd.read_csv('data/qibpi_data.csv')

# Define the Initialisation class
class Initialisation:
    def __init__(self, num_qubits, max_layers, random_seed=None, source=None, weight_type=None):
        self.num_qubits = num_qubits
        self.max_layers = max_layers
        self.current_layer = 1
        self.random_seed = random_seed
        self.parameter_history = []
        self.last_parameters = None
        self.source = to_snake_case(source)
        self.weight_type = weight_type
        self.allowed_sources = [
            "four_regular_graph",
            "geometric",
            "nearly_complete_bi_partite",
            "power_law_tree",
            "three_regular_graph",
            "uniform_random",
            "watts_strogatz_small_world",
        ]
        self.optimal_parameters_df = optimal_parameters_df
        if random_seed is not None:
            np.random.seed(random_seed)

    def __str__(self):
        return f"Initialisation(num_qubits={self.num_qubits}, max_layers={self.max_layers}, current_layer={self.current_layer}, random_seed={self.random_seed})"

    def random_init(self, num_layers=None):
        if num_layers is None:
            num_layers = self.current_layer
        gamma = np.random.uniform(-np.pi, np.pi, num_layers)
        beta = np.random.uniform(-np.pi, np.pi, num_layers)
        params = np.hstack([gamma, beta])
        self.parameter_history.append(params)
        self.last_parameters = params
        return params

    def fixed_angles_constant_init(self, num_layers=None):
        if num_layers is None:
            num_layers = self.current_layer
        gamma = np.full(num_layers, -0.2)
        beta = np.full(num_layers, 0.2)
        params = np.hstack([gamma, beta])
        self.parameter_history.append(params)
        self.last_parameters = params
        return params
    
    def update_layer(self):
        if self.current_layer < self.max_layers:
            self.current_layer += 1

    def qibpi_init(self, source=None):
        if self.optimal_parameters_df is None:
            return "No data available."
        
        df = self.optimal_parameters_df
        effective_source = source if source is not None else self.source

        if self.weight_type == 'None' or self.weight_type is None:
            # Filter for weight_type is NaN
            filtered_df = df[(df['graph_type'] == effective_source) & (df['layer'] == self.current_layer) & (df['weight_type'].isnull())]
        else:
            # Filter for weight_type is not NaN
            filtered_df = df[(df['graph_type'] == effective_source) & (df['layer'] == self.current_layer) & (df['weight_type'] == self.weight_type)]
        
        if not filtered_df.empty:
            beta_values = []
            gamma_values = []
            for i in range(0, self.current_layer):
                beta_key = f'median_beta_{i}'
                gamma_key = f'median_gamma_{i}'
                beta_values.append(filtered_df.iloc[0][beta_key])
                gamma_values.append(filtered_df.iloc[0][gamma_key])

            # Flatten gamma and beta values
            params = np.hstack([gamma_values, beta_values])
            self.parameter_history.append(params)
            self.last_parameters = params

            return params
        else:
            # Update the error message to include the source, weight_type and number of layers 
            breakpoint()
            raise ValueError(f"No optimal parameters found for source: {effective_source}, weight_type: {self.weight_type}, and number of layers: {self.current_layer}.")
            

    def three_regular_init(self):
        return self.qibpi_init(source="three_regular_graph")

        
    def tqa_init(self, evolution_time=5):
        """
        Initialize QAOA parameters using the TQA (Trotterized Quantum Annealing) heuristic.
        
        Args:
            p (int): Number of QAOA layers.
            noise (float): Noise level for perturbing the initial points, defaults to 0.1.
            previous_layer_initial_point (list or None): If given, perturbs this point for the new initial point.

        Returns:
            np.array: Array of initial QAOA parameters (gamma and beta values).
        """

        # If the first layer, initialize using random
        if self.current_layer == 1:
            return self.random_init()

        # Number of layers
        p = self.current_layer

        # Initialize delta_t for the evolution time divided by the number of layers
        delta_t = evolution_time / p

        # Linearly spaced gamma and betas
        gamma = np.linspace(delta_t / p, delta_t, num=p)
        beta = np.linspace(delta_t * (1 - 1 / p), 0, num=p)

        # Combine alphas and betas into a single array
        return np.concatenate([gamma, beta])

    def interp_init(self):
        """
        Initialize parameters for QAOA at the next level using linear interpolation from the last parameters.
        Assumes self.last_parameters stores the last used parameters and is not None.
        """
        if self.current_layer == 1:
            # Initialise using QIBPI if no parameters are set yet
            return self.random_init()
        
        p = len(self.last_parameters) // 2  # Current number of layers, assuming gamma and beta are equally split
        last_gamma = self.last_parameters[:p]
        last_beta = self.last_parameters[p:]
        
        new_gamma = np.zeros(p + 1)
        new_beta = np.zeros(p + 1)
        
        # Interpolate new values
        for i in range(1, p + 1):
            new_gamma[i] = ((p - i) / p) * last_gamma[i - 1] + (i / p) * last_gamma[min(i, p - 1)]
            new_beta[i] = ((p - i) / p) * last_beta[i - 1] + (i / p) * last_beta[min(i, p - 1)]
        
        # Ensure the final element is the same as the last element
        new_gamma[-1] = last_gamma[-1]
        new_beta[-1] = last_beta[-1]

        # Store and return the new parameters
        self.parameter_history.append(np.hstack([new_gamma, new_beta]))
        self.last_parameters = np.hstack([new_gamma, new_beta])
        return self.last_parameters
    
    def fourier_init(self):
        """
        Apply a Fourier transform to the last parameter set and convert back, adjusting for the new parameter size.
        This method assumes that last_parameters is already set and current_layer is used to determine dimensions.
        """
        if self.current_layer == 1:
            # Handle the base case where no parameters are set yet
            return self.random_init()
        
        # Convert to Fourier space and back to the real space with the new dimension
        fourier_point = self.convert_to_fourier_point(point = self.last_parameters, num_params_in_fourier_point= 2 * (self.current_layer - 1))
        new_initial_parameters = self.convert_from_fourier_point(
            fourier_point, 
            num_params_in_point=2 * self.current_layer
            )

        return new_initial_parameters
    

    def convert_to_fourier_point(self, point, num_params_in_fourier_point):
        """Converts a point to fourier space.
        Args:
            point: The point to convert.
            num_params_in_fourier_point: The length of the resulting fourier point. Must be even.
        Returns:
            The converted point in fourier space.
        """
        fourier_point = [0] * num_params_in_fourier_point
        reps = int(len(point) / 2)  # point should always be even
        max_frequency = int(
            num_params_in_fourier_point / 2
        )  # num_params_in_fourier_point should always be even
        for i in range(max_frequency):
            fourier_point[i] = 0
            for k in range(reps):
                fourier_point[i] += point[k] * math.sin(
                    (k + 0.5) * (i + 0.5) * math.pi / max_frequency
                )
            fourier_point[i] = 2 * fourier_point[i] / reps

            fourier_point[i + max_frequency] = 0
            for k in range(reps):
                fourier_point[i + max_frequency] += point[k + reps] * math.cos(
                    (k + 0.5) * (i + 0.5) * math.pi / max_frequency
                )
            fourier_point[i + max_frequency] = 2 * fourier_point[i + max_frequency] / reps
        return fourier_point


    def convert_from_fourier_point(self, fourier_point, num_params_in_point):
        """Converts a point in Fourier space back to QAOA angles.
        Args:
            fourier_point: The point in Fourier space to convert.
            num_params_in_point: The length of the resulting point. Must be even.
        Returns:
            The converted point in the form of QAOA rotation angles.
        """
        new_point = [0] * num_params_in_point
        reps = int(num_params_in_point / 2)  # num_params_in_result should always be even
        max_frequency = int(len(fourier_point) / 2)  # fourier_point should always be even
        for i in range(reps):
            new_point[i] = 0
            for k in range(max_frequency):
                new_point[i] += fourier_point[k] * math.sin(
                    (k + 0.5) * (i + 0.5) * math.pi / reps
                )

            new_point[i + reps] = 0
            for k in range(max_frequency):
                new_point[i + reps] += fourier_point[k + max_frequency] * math.cos(
                    (k + 0.5) * (i + 0.5) * math.pi / reps
                )
        return new_point
    
    def init_params_for_layer(self, init_type, df=None):
        """
        Initialize parameters for the current layer based on the specified type.
        """
        if init_type == "random":
            return self.random_init()
        elif init_type == "fixed_angles_constant":
            return self.fixed_angles_constant_init()
        elif init_type == "tqa":
            return self.tqa_init()
        elif init_type == "qibpi":
            return self.qibpi_init()
        elif init_type == "three_regular":
            return self.three_regular_init()
        elif init_type == "interp":
            return self.interp_init()
        elif init_type == "fourier":
            return self.fourier_init()
        else:
            raise ValueError("Unknown initialization type specified.")