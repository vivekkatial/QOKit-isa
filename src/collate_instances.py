import mlflow
from mlflow.tracking import MlflowClient
import logging
import os
import shutil
import networkx as nx

# Configure logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def process_runs(client, runs, base_dir):
    for run in runs:
        run_id = run.info.run_id
        instance_type = run.data.params.get('graph_type', 'unknown')
        weight_type = run.data.params.get('weight_type', 'unknown')
        num_nodes = int(run.data.params.get('num_nodes', 0))
        
        # Create the directory if it doesn't exist
        output_dir = f"{base_dir}/{instance_type}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Download the artifact
        artifact_path = "graph.graphml"
        output_path = f"{output_dir}/{run_id}_graph.graphml"
        
        try:
            local_path = client.download_artifacts(run_id, artifact_path, output_dir)
            # Rename the file to include the run_id
            shutil.move(local_path, output_path)
            logger.info(f"Downloaded and saved artifact for run {run_id} to {output_path}")
            # Read the graph and add the `graph_type` and `weight_type` as attributes
            graph = nx.read_graphml(output_path)
            graph.graph['graph_type'] = instance_type
            graph.graph['weight_type'] = weight_type
            # Save the updated graph
            nx.write_graphml(graph, output_path)
        except Exception as e:
            logger.error(f"Failed to download artifact for run {run_id}: {e}")

def main():
    try:
        # Connect to MLFlow experiment
        EXPERIMENT_NAME = "QAOA-Parameter-Initialisation"
        MIN_NODES = 12
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

        # Define instance types
        instance_types = [
            "Nearly Complete BiPartite",
            "Uniform Random",
            "Power Law Tree",
            "Watts-Strogatz small world",
            "3-Regular Graph",
            "4-Regular Graph",
            "Geometric",
        ]

        # # Process non-evolved instances
        for instance_type in instance_types:
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=f"attributes.status = 'FINISHED' and params.num_nodes = '{MIN_NODES}' and params.graph_type = '{instance_type}'",
                max_results=10000
            )
            # Filter out evolved runs
            runs = [run for run in runs if run.data.params.get('evolved_instance', 'False') != 'True']
            logger.info(f"Processing {len(runs)} non-evolved runs for {instance_type}")
            process_runs(client, runs, "data/instances")

        # Process evolved instances
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"attributes.status = 'FINISHED' AND params.num_nodes = '{MIN_NODES}' AND params.evolved_instance = 'True'",
            max_results=10000
        )
        logger.info(f"Processing {len(runs)} evolved runs")
        process_runs(client, runs, "data/evolved_instances")

        logger.info("Finished processing all runs.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()