import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import logging
import os

from utils import to_snake_case

# Configure logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Connect to MLFlow experiment
        EXPERIMENT_NAME = "QAOA-Parameter-Initialisation"
        NUM_NODES = 12
        # graph_types = [
        #     "Nearly Complete BiPartite",
        #     "Uniform Random",
        #     "Power Law Tree",
        #     "Watts-Strogatz small world",
        #     "3-Regular Graph",
        #     "4-Regular Graph",
        #     "Geometric",
        # ]
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        # Log experiment details
        logging.info(f"Experiment: {EXPERIMENT_NAME}")
        logging.info(f"MLFlow Tracking Server URI: {mlflow.get_tracking_uri()}")
        logging.info(f"MLFlow Version: {mlflow.version.VERSION}")
        logging.info(f"MLFlow Timeout: {os.getenv('MLFLOW_HTTP_REQUEST_TIMEOUT')}")

        # Initialize MLflow client
        client = MlflowClient()

        # Retrieve the experiment ID
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            logger.error(f"Experiment '{EXPERIMENT_NAME}' not found.")
            return

        experiment_id = experiment.experiment_id
        logger.info(f"Retrieved experiment ID: {experiment_id}")

        all_runs = []
            
        # Retrieve all finished runs from the experiment
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"params.num_nodes = '{NUM_NODES}' and attributes.status = 'FINISHED' and params.evolved_instance = 'True'",
            max_results=5000
        )

        if runs:
            run_data = []
            for run in runs:
                run_info = {
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "artifact_uri": run.info.artifact_uri,
                    "lifecycle_stage": run.info.lifecycle_stage,
                }
                run_metrics = run.data.metrics
                run_params = run.data.params
                run_tags = run.data.tags
                
                # Combine all data into a single dictionary
                run_info.update(run_metrics)
                run_info.update(run_params)
                run_info.update(run_tags)
                
                run_data.append(run_info)

            runs_df = pd.DataFrame(run_data)
            all_runs.append(runs_df)
            # logging.info(f"Retrieved {len(runs_df)} runs for graph type: {graph_type}")
            logging.info(f"Retrieved {len(runs_df)} runs")

        if all_runs:
            d_results = pd.concat(all_runs, ignore_index=True)
            d_results.to_csv(f"data/initialisation_results_nodes-{NUM_NODES}-evolved.csv", index=False)
            logging.info(f"Saved {len(d_results)} runs to data/initialisation_results_nodes-{NUM_NODES}-evolved.csv")
        else:
            logging.info("No runs found for the specified parameters.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
