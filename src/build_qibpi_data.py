import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Connect to MLFlow experiment
        mlflow.set_experiment("QAOA-QIBPI-Generator")
        logger.info("Connected to MLFlow experiment: QAOA-QIBPI-Generator")

        # Initialize MLflow client
        client = MlflowClient()

        # Retrieve the experiment ID
        experiment = client.get_experiment_by_name('QAOA-QIBPI-Generator')
        if not experiment:
            logger.error("Experiment 'QAOA-QIBPI-Generator' not found.")
            return

        experiment_id = experiment.experiment_id
        logger.info(f"Retrieved experiment ID: {experiment_id}")

        # Retrieve all finished runs from the experiment
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="attributes.status = 'FINISHED'"
        )

        all_params = []

        for run in runs:
            data = run.data
            # Skip runs with no metrics
            if not data.metrics:
                logger.info(f"Run {run.info.run_id} skipped due to no metrics.")
                continue

            # Initialize dictionaries for beta and gamma values for this run
            params_gamma = {f"median_gamma_{i}": None for i in range(15)}
            params_beta = {f"median_beta_{i}": None for i in range(15)}

            # Extract beta and gamma values from data.metrics
            for i in range(15):
                gamma_i = data.metrics.get(f"gamma_{i}", None)
                beta_i = data.metrics.get(f"beta_{i}", None)
                if gamma_i is not None:
                    params_gamma[f"median_gamma_{i}"] = gamma_i
                if beta_i is not None:
                    params_beta[f"median_beta_{i}"] = beta_i

            # Combine beta and gamma dictionaries and include run UUID
            combined_params = {
                **params_beta,
                **params_gamma,
                'uuid': run.info.run_id,
                'graph_type': data.params.get('graph_type'),
                'weight_type': data.params.get('weight_type'),
            }
            # Append the combined dictionary to the list
            all_params.append(combined_params)

        # Construct a DataFrame from the list of dictionaries
        df = pd.DataFrame(all_params)

        # Filter for rows with non-null values
        df = df.dropna()

        # Select numeric columns and include grouping columns for operations
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        grouping_cols = ['graph_type', 'weight_type']
        columns_to_use = numeric_cols + grouping_cols

        # Ensure all required columns exist in the DataFrame
        missing_cols = [col for col in grouping_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Columns {missing_cols} are missing from the DataFrame.")
            return

        # Create a filtered DataFrame with only the columns of interest
        filtered_df = df[columns_to_use]

        # Group by 'graph_type' and 'weight_type', then compute the median for each group
        grouped_df = filtered_df.groupby(grouping_cols)
        median_df = grouped_df.median()

        # Reset the index to flatten the DataFrame, turning the multi-level index into columns
        flat_df = median_df.reset_index()
        # Add column for layer number
        flat_df['layer'] = 15
        # Reorder columns
        flat_df = flat_df[['graph_type', 'weight_type', 'layer'] + numeric_cols]

        # Display the final flattened DataFrame
        logger.info("Final flattened DataFrame:")
        logger.info(flat_df)

        # Write to csv file
        flat_df.to_csv("data/qibpi_data.csv", index=False)
        logger.info("Data saved to qibpi_data.csv")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
