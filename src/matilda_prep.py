import pandas as pd
import numpy as np
import pytest

# Hide pandas warnings
pd.options.mode.chained_assignment = None

def load_and_process_data(file_path, source_type='graph_weight'):
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

    # Check for missing values
    missing_values = selected_df.isnull().any(axis=1).sum()
    print(f"Missing values: {missing_values}/{selected_df.shape[0]} ({missing_values / selected_df.shape[0]:.2%})")

    # Check for columns that have no-variance (i.e., all values are the same) and remove them
    no_variance_cols = selected_df.columns[selected_df.nunique() == 1]
    print(f"No-variance columns: {no_variance_cols}")
    selected_df.drop(columns=no_variance_cols, inplace=True)

    # Filter rows with any missing values
    selected_df = selected_df.dropna()

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
    no_variance_cols = feature_cols.columns[feature_cols.nunique() == 1]
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
d_matilda = load_and_process_data("data/initialisation_results-10-nodes.csv")

if __name__ == "__main__":
    file_path = "data/initialisation_results-10-nodes.csv"
    source_types = ['graph_weight', 'graph', 'weight', 'weighted_unweighted']
    
    for source_type in source_types:
        d_matilda = load_and_process_data(file_path, source_type)
        output_file = f"data/matilda_processed_{source_type}.csv"
        
        # Write to csv file
        d_matilda.to_csv(output_file, index=False)
        
        # Print the first few rows of the processed data
        print(f"Processed data for source type '{source_type}':")
        print(d_matilda.head())
        print(d_matilda.info())
        print(f"Writing to {output_file}...")
        
    pytest.main([__file__])
