import pandas as pd
import numpy as np
import pytest

# Hide pandas warnings
pd.options.mode.chained_assignment = None

def load_and_process_data(file_path, source_type='graph_weight', feature_filter=False, **kwargs):
    """
    Load and process the CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file.
    source_type (str): The type of source column to create. 
                       Options are 'graph_weight', 'graph', 'weight', 'weighted_unweighted'.
                       Default is 'graph_weight'.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Select all algo_ columns
    algo_cols = [col for col in df.columns if col.startswith('algo_')]

    # Check if `evolved` is in **kwargs
    if 'evolved' in kwargs:
        print(f"Filtering for evolved_instance = {kwargs['evolved']}")

        # Read in the original data
        # Check kwargs for `original_file_path`
        if 'original_file_path' in kwargs:
            original_file_path = kwargs['original_file_path']
            original_df = pd.read_csv(original_file_path)
            # Update the graph_type column in the evolved data to include `(Evolved)`
            if source_type == 'graph':
                df['graph_type'] = df['graph_type'] + ' (Evolved)'
                df['graph_type'] = 'Evolved'
            elif source_type == 'weight':
                df['weight_type'] = df['weight_type'] + ' (Evolved)'
                df['weight_type'] = 'Evolved'
            else:
                raise ValueError("Invalid source_type. Choose from 'graph', 'weight'.")
            # Concat the original_df and df
            df = pd.concat([original_df, df], ignore_index=True)
            #
        else:
            raise ValueError("original_file_path not found in kwargs.")

    # Drop the evolved_instance column and the custom_graph column
    df.drop(columns=['evolved_instance', 'custom_graph'], inplace=True)

    # Select all columns from 'acyclic' to 'weight_type'
    start_idx = df.columns.get_loc('acyclic')
    end_idx = df.columns.get_loc('weight_type') + 1
    range_cols = df.columns[start_idx:end_idx]

    
    

    # Select run_id column
    run_id_col = ['run_id']

    # Deselect the penalty columns
    penalty_cols = [col for col in df.columns if col.startswith('penalty_')]

    # Combine all selected columns and exclude penalty columns
    selected_cols = run_id_col + algo_cols + list(range_cols)
    selected_cols = [col for col in selected_cols if col not in penalty_cols]

    # Create a new DataFrame with the selected columns
    selected_df = df[selected_cols]

    # Rename the range columns to have 'feature_' prefix
    range_cols_prefixed = ['feature_' + col for col in range_cols]
    rename_dict = dict(zip(range_cols, range_cols_prefixed))
    selected_df.rename(columns=rename_dict, inplace=True)

    # Handle NaN in weight_type
    selected_df['feature_weight_type'] = selected_df['feature_weight_type'].fillna('None')

    # Filter out rows with 'feature_weight_type' equal to 'None'
    selected_df = selected_df[selected_df['feature_weight_type'] != 'None']
    
    # selected_df = selected_df[selected_df['feature_weight_type'] == 'None']

    # Create the 'Source' column based on the source_type parameter
    if source_type == 'graph_weight':
        selected_df['Source'] = selected_df['feature_graph_type'] + ' ' + selected_df['feature_weight_type']
    elif source_type == 'graph':
        selected_df['Source'] = selected_df['feature_graph_type']
    elif source_type == 'weight':
        selected_df['Source'] = selected_df['feature_weight_type']
    elif source_type == 'weighted_unweighted':
        selected_df['Source'] = selected_df['feature_weight_type'].apply(lambda x: 'unweighted' if x == 'None' else 'weighted')
    else:
        raise ValueError("Invalid source_type. Choose from 'graph_weight', 'graph', 'weight', 'weighted_unweighted'.")

    # Remove the original 'feature_weight_type' and 'feature_graph_type' columns
    selected_df.drop(columns=['feature_weight_type', 'feature_graph_type'], inplace=True)

    # Feature filtering (remove non-weight related columns)
    if feature_filter:
        
        features = [
            # Weighted features
            "feature_maximum_weighted_degree",
            "feature_max_weight",
            "feature_mean_weight",
            "feature_median_weight",
            "feature_minimum_weighted_degree",
            "feature_min_weight",
            "feature_range_weight",
            "feature_skewness_weight",
            "feature_std_dev_weight",
            "feature_variance_weight",
            "feature_weighted_average_clustering",
            "feature_weighted_average_shortest_path_length",
            "feature_weighted_diameter",
            "feature_weighted_radius",
            
            # Laplacian features
            "feature_laplacian_largest_eigenvalue",
            "feature_laplacian_second_largest_eigenvalue",
            "feature_ratio_of_two_largest_laplacian_eigenvaleus",
            
            # Symmetry-related features
            "feature_number_of_orbits",
            "feature_group_size",
            "feature_is_distance_regular",
            "feature_regular"
        ]

        selected_df = selected_df[['run_id', 'Source'] + algo_cols + features]

    # Check for missing values
    missing_values = selected_df.isnull().any(axis=1).sum()
    print(f"Missing values: {missing_values}/{selected_df.shape[0]} ({missing_values / selected_df.shape[0]:.2%})")

    # Filter rows with any missing values
    selected_df = selected_df.dropna()

    # Check for columns that have no-variance (i.e., all values are the same) and remove them
    no_variance_cols = selected_df.columns[selected_df.apply(lambda x: x.nunique()) == 1]
    print(f"No-variance columns: {no_variance_cols}")
    selected_df.drop(columns=no_variance_cols, inplace=True)

    # Convert all boolean columns to integers
    bool_cols = selected_df.select_dtypes(include=bool).columns
    selected_df[bool_cols] = selected_df[bool_cols].astype(int)

    # Shuffle the DataFrame
    selected_df = selected_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert run_id to `Instances`
    selected_df.rename(columns={'run_id': 'Instances'}, inplace=True)

    return selected_df

## Define tests
def test_instances_column():
    assert 'Instances' in d_matilda.columns

def test_instances_column_unique():
    assert d_matilda['Instances'].nunique() == len(d_matilda), "Checking Unique Instances"
    assert d_matilda['Instances'].dtype == 'object', "Checking Instances column is a string"

def test_valid_columns():
    valid_cols = ['Instances', 'Source'] + [col for col in d_matilda.columns if col.startswith('feature_') or col.startswith('algo_')]
    assert set(d_matilda.columns) == set(valid_cols), "Checking valid column names"

def test_no_missing_values():
    assert d_matilda.isnull().sum().sum() == 0

def test_unique_instances():
    assert d_matilda['Instances'].nunique() == len(d_matilda), "Checking Unique Instances"

def test_feature_variance():
    feature_cols = d_matilda.filter(like='feature_')
    no_variance_cols = feature_cols.columns[feature_cols.apply(lambda x: x.nunique()) == 1]
    assert len(no_variance_cols) == 0, "There are feature columns with no variance"

def test_at_least_one_source():
    assert d_matilda['Source'].nunique() > 1

def test_at_least_two_features():
    feature_cols = d_matilda.filter(like='feature_')
    assert feature_cols.shape[1] >= 2, "There are at least two feature columns"

def test_at_least_two_algorithms():
    algo_cols = d_matilda.filter(like='algo_')
    assert algo_cols.shape[1] >= 2, "There are at least two algorithm columns"

def test_no_boolean_columns():
    bool_cols = d_matilda.select_dtypes(include=bool).columns
    assert len(bool_cols) == 0, f"Boolean columns: {', '.join(bool_cols)}"
    
def test_only_numeric_features():
    numeric_cols = d_matilda.select_dtypes(include=np.number).columns
    non_numeric_cols = d_matilda.drop(columns=['Instances', 'Source']).select_dtypes(exclude=np.number).columns
    assert len(non_numeric_cols) == 0, f"Non-numeric columns: {', '.join(non_numeric_cols)}"

## Load and process the data
d_matilda = load_and_process_data(
            "data/initialisation_results_nodes-12-evolved.csv", 
            source_type="graph",
            evolved=True,
            original_file_path="data/initialisation_results_nodes-12.csv",
            feature_filter=False
        )

if __name__ == "__main__":
    file_path = "data/initialisation_results_nodes-12-evolved.csv"
    source_types = ['weight']
    for source_type in source_types:
        d_matilda = load_and_process_data(
            "data/initialisation_results_nodes-12-evolved.csv", 
            source_type=source_type,
            evolved=True,
            original_file_path="data/initialisation_results_nodes-12.csv",
            feature_filter=False
        )
        output_file = f"data/12-nodes/matilda_processed_{source_type}.csv"
        
        # Write to csv file
        d_matilda.to_csv(output_file, index=False)
        
        # Print the first few rows of the processed data
        print(f"Processed data for source type '{source_type}':")
        print(d_matilda.head())
        print(d_matilda.info())
        # Print the source distribution
        print(d_matilda['Source'].value_counts())
        print(f"Writing to {output_file}...")

        
    pytest.main([__file__])
