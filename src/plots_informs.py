import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def plot_ar_distribution(df, output_file='ar_distribution.png'):
    # Select and rename columns to be more readable
    cols = ['approximation_ratio_fixed_angles_constant',
            'approximation_ratio_random',
            'approximation_ratio_three_regular',
            'approximation_ratio_qibpi',
            'approximation_ratio_tqa',
            'approximation_ratio_interp_p15',
            'approximation_ratio_fourier_p15']

    df = df[cols]
    
    new_cols = ['Constant', 'Random', 'Three Regular', 'QIBPI', 'TQA', 'INTERP', 'FOURIER']
    df.columns = new_cols

    # Melt the dataframe to long format for FacetGrid
    df_melted = df.melt(var_name='Approximation Type', value_name='Approximation Ratio')
    
    # Filter rows with NaN values
    df_melted = df_melted.dropna()

    # Create the FacetGrid
    g = sns.FacetGrid(df_melted, col='Approximation Type', col_wrap=4, height=4, sharex=True, sharey=True)
    g.map(sns.histplot, 'Approximation Ratio', bins=20, kde=False, color='blue', alpha=0.3)

    g.set(xlim=(0, 1))


    # Add titles and labels
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels('Approximation Ratio', 'Count')
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('')

    # Save the plot
    plt.savefig(output_file)
    plt.close()

# Read the CSV file
df = pd.read_csv('data/initialisation_results-10-nodes.csv')

# Print number of rows and columns
print(df.shape)

# Plot the distribution and save the plot
plot_ar_distribution(df)
