import networkx as nx
from collections import defaultdict
import os
import glob

def graph_signature(G, consider_weights=True):
    """Compute a signature for the graph based on some invariants."""
    if consider_weights:
        return (G.number_of_nodes(), G.number_of_edges(),
                tuple(sorted(d for n, d in G.degree())),
                tuple(sorted(G.get_edge_data(e[0], e[1]).get('weight', 1) for e in G.edges())))
    else:
        return (G.number_of_nodes(), G.number_of_edges(),
                tuple(sorted(d for n, d in G.degree())))

def are_isomorphic(G1, G2, consider_weights=True):
    """Check if two graphs are isomorphic, optionally considering weights."""
    if consider_weights:
        return nx.is_isomorphic(G1, G2, edge_match=lambda e1, e2: e1.get('weight') == e2.get('weight'))
    else:
        return nx.is_isomorphic(G1, G2)

def find_unique_networks(networks, consider_weights=True):
    signature_groups = defaultdict(list)
    for i, (name, G) in enumerate(networks):
        sig = graph_signature(G, consider_weights)
        signature_groups[sig].append((i, name, G))
    
    unique_networks = []
    indices_of_unique_networks = []

    for group in signature_groups.values():
        if len(group) == 1:
            unique_networks.append((group[0][1], group[0][2]))
            indices_of_unique_networks.append(group[0][0])
        else:
            for i, name, G in group:
                if not any(are_isomorphic(G, H, consider_weights) for _, H in unique_networks):
                    unique_networks.append((name, G))
                    indices_of_unique_networks.append(i)
    
    return unique_networks, indices_of_unique_networks

def load_graphml_networks(folder_path):
    networks = []
    for file_path in glob.glob(os.path.join(folder_path, '*.graphml')):
        G = nx.read_graphml(file_path)
        name = os.path.basename(file_path)
        networks.append((name, G))
    return networks

def process_instance_type(instance_type, base_path):
    folder_path = os.path.join(base_path, instance_type)
    networks = load_graphml_networks(folder_path)
    
    unique_networks_weighted, _ = find_unique_networks(networks, consider_weights=True)
    unique_networks_unweighted, _ = find_unique_networks(networks, consider_weights=False)
    
    print(f"\nInstance Type: {instance_type}")
    print(f"Total networks: {len(networks)}")
    print(f"Unique networks (considering weights): {len(unique_networks_weighted)}")
    print(f"Unique networks (ignoring weights): {len(unique_networks_unweighted)}")
    print("Names of first 5 unique networks (weighted):")
    for name, _ in unique_networks_weighted[:5]:
        print(f"  {name}")
    print("..." if len(unique_networks_weighted) > 5 else "")
    
    return len(networks), len(unique_networks_weighted), len(unique_networks_unweighted)

def process_and_save_final(folder_path, output_folder_path='data/instances-processed'):
    networks = load_graphml_networks(folder_path)
    
    unique_networks_weighted, _ = find_unique_networks(networks, consider_weights=True)
    unique_networks_unweighted, _ = find_unique_networks(networks, consider_weights=False)
    
    print(f"\nInstance Type: ALL")
    print(f"Total networks: {len(networks)}")
    print(f"Unique networks (considering weights): {len(unique_networks_weighted)}")
    print(f"Unique networks (ignoring weights): {len(unique_networks_unweighted)}")
    print("Names of first 5 unique networks (weighted):")
    for name, _ in unique_networks_weighted[:5]:
        print(f"  {name}")
    print("..." if len(unique_networks_weighted) > 5 else "")
    
    # Save all the unique networks to a new folder called data/instances-processed
    os.makedirs(output_folder_path, exist_ok=True)
    for name, G in unique_networks_weighted:
        file_path = os.path.join(output_folder_path, name)
        nx.write_graphml(G, file_path)
    
    return len(networks), len(unique_networks_weighted), len(unique_networks_unweighted)

# Main script
# base_path = 'data/instances'
# instance_types = [
#     "Nearly Complete BiPartite",
#     "Uniform Random",
#     "Power Law Tree",
#     "Watts-Strogatz small world",
#     "3-Regular Graph",
#     "4-Regular Graph",
#     "Geometric",
# ]

# total_networks = 0
# total_unique_weighted = 0
# total_unique_unweighted = 0

# print("Analysis of Network Isomorphism")
# print("===============================")

# for instance_type in instance_types:
#     networks_count, unique_weighted, unique_unweighted = process_instance_type(instance_type, base_path)
#     total_networks += networks_count
#     total_unique_weighted += unique_weighted
#     total_unique_unweighted += unique_unweighted

# print("\nOverall Summary:")
# print(f"Total networks processed: {total_networks}")
# print(f"Total unique networks (considering weights): {total_unique_weighted}")
# print(f"Total unique networks (ignoring weights): {total_unique_unweighted}")


# Add another snippet that does the same analysis for instances in the `instances_raw` folder.

networks_count, unique_weighted, unique_unweighted = process_and_save_final('data/instances-raw', 'data/instances-processed')

print("\nOverall Summary:")
print(f"Total networks processed: {networks_count}")
print(f"Total unique networks (considering weights): {unique_weighted}")
print(f"Total unique networks (ignoring weights): {unique_unweighted}")

