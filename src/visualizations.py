import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

def plot_approximation_ratio(df, acceptable_ratio):
    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='num_evals', y='approximation_ratio', hue='new_method', linewidth=1.5, alpha=0.8)
    plt.axhline(y=1, color='black', linestyle='--', linewidth=1.5)
    plt.axhline(y=acceptable_ratio, color='red', linestyle='--', linewidth=1.5)
    plt.ylim(0, 1.1)
    plt.title('Iteration vs Approximation Ratio by Initialisation', fontsize=16)
    plt.xlabel('Number of Evaluations', fontsize=14)
    plt.ylabel('Approximation Ratio', fontsize=14)
    plt.legend(title='Initialisation Method', title_fontsize='14', fontsize='12', loc='lower right')
    plt.tight_layout()
    
    return plt

def plot_facet_approximation_ratio(df, acceptable_ratio):
    sns.set(style="whitegrid")

    g = sns.FacetGrid(df, col="new_method", col_wrap=4, height=3, aspect=1.5)
    g.map_dataframe(sns.lineplot, x='num_evals', y='approximation_ratio', linewidth=1.5, alpha=0.8)
    g.map(plt.axhline, y=1, color='black', linestyle='--', linewidth=1.5)
    g.map(plt.axhline, y=acceptable_ratio, color='red', linestyle='--', linewidth=1.5)

    g.set(ylim=(0, 1.1))
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Iteration vs Approximation Ratio by Initialisation Method', fontsize=16)
    g.set_axis_labels("Number of Evaluations", "Approximation Ratio")
    
    return g

def plot_graph_with_weights(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='blue', node_size=500, font_size=10, font_weight='bold')
    labels = nx.get_edge_attributes(G, 'weight')
    labels = {k: round(v, 2) for k, v in labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8, font_weight='bold')
    
    return plt

def plot_edge_weights_histogram(G, weight_type):
    if weight_type is not None:
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        sns.histplot(weights, kde=True)
        plt.title(f'Histogram of Edge Weights for {weight_type}')
        plt.xlabel('Weight')
        plt.ylabel('Frequency')
        
        return plt
    return None