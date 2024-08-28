import os
import shutil
import networkx as nx
from collections import defaultdict

# Define the uniform mapping for graph types
GRAPH_TYPE_MAPPING = {
    '3-Regular Graph': '3-Regular Graph',
    '3_regular_graph': '3-Regular Graph',
    '4-Regular Graph': '4-Regular Graph',
    '4_regular_graph': '4-Regular Graph',
    'Geometric': 'Geometric',
    'geometric': 'Geometric',
    'Nearly Complete BiPartite': 'Nearly Complete BiPartite',
    'nearly_complete_bi_partite': 'Nearly Complete BiPartite',
    'Power Law Tree': 'Power Law Tree',
    'power_law_tree': 'Power Law Tree',
    'Uniform Random': 'Uniform Random',
    'uniform_random': 'Uniform Random',
    'Watts-Strogatz small world': 'Watts-Strogatz small world'
}

def copy_graphml_files(source_dir, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.graphml'):
                file_path = os.path.join(root, file)
                shutil.copy2(file_path, destination_dir)

def analyze_graph_properties(file_path):
    try:
        G = nx.read_graphml(file_path)
        graph_type = G.graph.get('graph_type', 'Unknown')
        weight_type = G.graph.get('weight_type', 'Unknown')
        
        # Map the graph_type to the uniform naming convention
        mapped_graph_type = GRAPH_TYPE_MAPPING.get(graph_type, 'Unknown')
        
        # If the mapping changed the graph_type, update the file
        if mapped_graph_type != graph_type:
            G.graph['graph_type'] = mapped_graph_type
            nx.write_graphml(G, file_path)
            print(f"Updated graph_type in {file_path} from '{graph_type}' to '{mapped_graph_type}'")
        
        return mapped_graph_type, weight_type
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return 'Error', 'Error'

# Define the paths
base_dir = 'data'
evolved_instances_dir = os.path.join(base_dir, 'evolved_instances')
instances_dir = os.path.join(base_dir, 'instances')
destination_dir = os.path.join(base_dir, 'isa-instances')

# Copy .graphml files
copy_graphml_files(evolved_instances_dir, destination_dir)
copy_graphml_files(instances_dir, destination_dir)

print("All .graphml files have been copied to", destination_dir)

# Analyze graph properties
graph_type_counts = defaultdict(int)
weight_type_counts = defaultdict(int)

for file in os.listdir(destination_dir):
    if file.endswith('.graphml'):
        file_path = os.path.join(destination_dir, file)
        graph_type, weight_type = analyze_graph_properties(file_path)
        
        graph_type_counts[graph_type] += 1
        weight_type_counts[weight_type] += 1

# Print results
print("\nGraph Type Counts:")
for graph_type, count in graph_type_counts.items():
    print(f"{graph_type}: {count}")

print("\nWeight Type Counts:")
for weight_type, count in weight_type_counts.items():
    print(f"{weight_type}: {count}")