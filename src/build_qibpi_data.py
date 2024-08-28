import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def extract_parameters(all_dicts, params=('gamma_', 'beta_')):
    extracted_values = []
    for d in all_dicts:
        params_dict = {key: value for key, value in d.items() if key.startswith(params)}
        extracted_values.append(params_dict)
    return extracted_values

def main():
    try:
        # Connect to MLFlow experiment
        EXPERIMENT_NAME = "QAOA-Parameter-Initialisation"
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info(f"Connected to MLFlow experiment: {EXPERIMENT_NAME}")

        # Initialize MLflow client
        client = MlflowClient()

        # Retrieve the experiment ID
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            logger.error(f"Experiment '{EXPERIMENT_NAME}' not found.")
            return

        experiment_id = experiment.experiment_id
        logger.info(f"Retrieved experiment ID: {experiment_id}")

        # Retrieve all finished runs from the experiment
        # Set of graph_types
        graph_types = [
            "Nearly Complete BiPartite", "Uniform Random","Power Law Tree","Watts-Strogatz small world","3-Regular Graph","4-Regular Graph","Geometric"
        ]
        
        all_runs = []
        for graph_type in graph_types:
            logging.info(f"Retrieving runs for graph type: {graph_type}")
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string="attributes.status = 'FINISHED' AND attributes.start_time > 1722485419000 AND params.graph_type = '%s'" % graph_type,
                max_results=1000,
            )
            all_runs.extend(runs)
        
        logging.info(f"Extracting parameters from {len(all_runs)} runs.")
        
        
        # Build QIBPI data for each depth 
        MAX_LAYERS = 20
        for layer in range(MAX_LAYERS):
            # Filter runs where the depth is equal to the current layer
            runs_at_layer = [run for run in all_runs if run.data.params['max_layers'] == str(layer + 1)]
            #  Extract parameters for each depth
            extracted_values = extract_parameters([run.data.metrics for run in runs_at_layer], params=('gamma_', 'beta_'))
            if len(extracted_values) == 0:
                logging.info(f"No runs found for depth {layer+1}.")
                continue
            # Create a DataFrame
            df = pd.DataFrame(extracted_values)
            # # Add a column for the graph type
            df['graph_type'] = [run.data.params['graph_type'] for run in runs_at_layer]
            # Add weight type
            df['weight_type'] = [run.data.params['weight_type'] for run in runs_at_layer]
            # # Add a column for the depth
            df['depth'] = layer + 1
            # Reorder the columns to have the graph type, weight type and layer at the beginning
            df = df[['graph_type', 'weight_type', 'depth'] + [col for col in df.columns if col not in ['graph_type', 'weight_type', 'depth']]]
            
            # Group by 'graph_type', 'weight_type', and 'depth' and calculate the median for each group
            summary_df = df.groupby(['graph_type', 'weight_type', 'depth']).median().reset_index()
            
            # Rename all columns with `beta_i` and `gamma_i` to `median_beta_i` and `median_gamma_i`
            summary_df = summary_df.rename(columns={col: f"median_{col}" for col in summary_df.columns if col.startswith('beta_') or col.startswith('gamma_')})
            # Display the resulting DataFrame
            logging.info(f"QIBPI data for depth {layer+1}:")
            logging.info(summary_df)
            
            # # Save the DataFrame to a CSV file
            summary_df.to_csv(f"data/qibpi/qibpi_data_{layer+1}.csv", index=False)
            logging.info(f"Saved QIBPI data for depth {layer+1}.")
    


    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
