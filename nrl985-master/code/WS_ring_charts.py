import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random 

def ws_ring_comparisons(k, p, n):
    # k = degree
    # p = probability of rewiring 
    # n = maximum number of agents considered 
    
    if n % 2 != 0 or n <= 10:
        return "N must be even and greater than 10"
    
    ws_average_path_lengths = []
    ws_diameters = []
    ring_average_path_lengths = []
    ring_diameters = []
    agent_numbers = [i for i in range(10, n+1, 2)]
    
    #Calculate the diameters and average path lengths as the number of agents go to n (i must be even)
    for i in range(10, n+1, 2):
        
        if i % 20 == 0:
            print(f"Producing result for N = {i}")
        #Create WS
        ws_graph = nx.watts_strogatz_graph(i,k,p)
        #Create Ring
        ring_lattice = nx.watts_strogatz_graph(i,k,0)
        #Compute the average_path length of ws
        ws_average_path_lengths.append(nx.average_shortest_path_length(ws_graph))
        #Compute the diameter of ws
        ws_diameters.append(nx.diameter(ws_graph))
        #Compute the average_path length of ring
        ring_average_path_lengths.append(nx.average_shortest_path_length(ring_lattice))
        #Compute the diameter of ring
        ring_diameters.append(nx.diameter(ring_lattice))
        
    #Create the average path lengths plot
    create_plot(ws_average_path_lengths, ring_average_path_lengths, "Average Path Lengths", agent_numbers)
    #Create the diameters plot
    create_plot(ws_diameters, ring_diameters, "Network Diameters", agent_numbers)
    

def create_plot(ws_data, ring_data, y_axis_name, agent_numbers):

    line_styles = ['-', '--']
    colors = ['blue', 'red']
    # Plot the results
    fig, ax = plt.subplots()
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel(y_axis_name)
    ax.grid(True)

    smoothed_ws_data = pd.Series(ws_data).rolling(50, min_periods=1).mean()
    smoothed_ring_data = pd.Series(ring_data).rolling(50, min_periods=1).mean()
    
    ax.plot(agent_numbers, smoothed_ws_data, line_styles[0], label="Watts-Strogatz", color=colors[0])
    ax.plot(agent_numbers, smoothed_ring_data, line_styles[1], label="Ring Lattice", color=colors[1])

    ax.legend()
    plt.tight_layout()  

    random_number = random.randint(0, 999999999)
    filename = f'saved_data/watts_strogatz_figs/ws_ring_comps{random_number}.png'
    print(f"Figure saved as {filename}")
    plt.savefig(filename)
    
    
ws_ring_comparisons(4, .5, 1000)

for i in range(5):
    ws_net = nx.watts_strogatz_graph(1000, 4, p=.5)
    print(f"Diameter of WS is: {nx.diameter(ws_net)}")
    print(f"Avg. Path Length of WS is: {nx.average_shortest_path_length(ws_net)}")
    ring_net = nx.watts_strogatz_graph(1000, 4, p=.0)
    print(f"Diameter of ring is: {nx.diameter(ring_net)}")
    print(f"Avg. Path Length of ring is: {nx.average_shortest_path_length(ring_net)}")