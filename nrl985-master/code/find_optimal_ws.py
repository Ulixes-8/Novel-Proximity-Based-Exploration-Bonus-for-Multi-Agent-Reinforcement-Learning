import networkx as nx
import matplotlib.pyplot as plt

# Number of nodes (agents)
n = 12

# Each node is connected to k nearest neighbors in ring topology
k = 4  

# Rewiring probability
p = 0.5  # Typical value to maintain small-world properties

# Initialize the meets_requirement flag
meets_requirement = False

# Loop until a suitable network is found
count = 0

optimal_lambda = 2

while not meets_requirement:
    # Constructing the Watts-Strogatz small-world network
    if count % 10 == 0:
        print("Iteration: ", count)
    
    ws_network = nx.watts_strogatz_graph(n, k, p)

    # Checking if the network meets the requirement (maximum two steps to reach any node)
    max_distance = max(nx.eccentricity(ws_network).values())
    meets_requirement = max_distance <= optimal_lambda
    count += 1

print("Found a suitable network after", count, "iterations")

# Convert the adjacency matrix to a 2D list
adj_matrix_sparse = nx.adjacency_matrix(ws_network)
adj_matrix = adj_matrix_sparse.toarray().tolist()

print(f"adj_matrix {adj_matrix}")

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

def plot_graph(adj, num_of_agents, fig_name, k):
    G = nx.Graph()
    for i in range(num_of_agents):
        for j in range(num_of_agents):
            if adj[i][j] == 1:
                G.add_edge(i, j)

    # Ensure graph is displayed with nodes in numerical order and in a circle
    pos = {}
    for i in range(num_of_agents):
        pos[i] = [np.cos(2*np.pi*i/num_of_agents), np.sin(2*np.pi*i/num_of_agents)]
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')

    # Incremental integer for filename
    increment = 0
    filename = f'saved_data/watts_strogatz_figs/{fig_name}_K={k}_{increment}.png'
    while os.path.exists(filename):
        increment += 1
        filename = f'saved_data/watts_strogatz_figs/{fig_name}_K={k}_{increment}.png'

    print(f"Figure saved as {filename}")
    plt.savefig(filename)
    plt.clf()  # Clear the current figure after saving it


plot_graph(adj_matrix, 24, 'watts_strogatz_discovered', 6)



# #Optimal WS for lambda* = 2, K = 4, N = 12
# ws_12_adj = [[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0], 
#        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0], 
#        [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0], 
#        [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1], 
#        [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0], 
#        [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0], 
#        [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1], 
#        [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1], 
#        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0], 
#        [1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0], 
#        [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0], 
#        [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]]

# #Ring lattice, K = 4, N = 12
# ring_12_adj = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], 
#                [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], 
#                [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
#                [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0], 
#                [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0], 
#                [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0], 
#                [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0], 
#                [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0], 
#                [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0], 
#                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1], 
#                [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1], 
#                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]]

