from enum import Enum

class AgentType(Enum):

    """Specifies the different agent types used"""

    RANDOM = 'RANDOM'
    IQL = 'IQL'
    ORIGINAL = 'ORIGINAL'
    EB_Lidard = 'EB_Lidard'

def line_graph(num_of_agents):

    """
    Creates a line graph adjacency table
    num_of_agents - The number of agents 

    Return an adjacency table for a line graph
    """

    adj = []

    for i in range(num_of_agents):
        neighbours = []
        for j in range(num_of_agents):
            if j == i -1 or j == i+1:
                neighbours.append(1)
            else:
                neighbours.append(0)

        adj.append(neighbours)

    return adj

def ring_graph(num_of_agents, k):

    """
    Creates a ring graph adjacency table
    num_of_agents - The number of agents
    k - Each node is connected to k nearest neighbors in ring topology.

    Return a ring graph adjacency table of degree k.

    """
    k = k // 2  # So k is the number of connections to each side
    
    adj = []

    for i in range(num_of_agents):
        neighbours = []
        for j in range(num_of_agents):
            found = False
            for neighbour in range(1, k+1):
                if j == (i-neighbour)%num_of_agents or j == (i+neighbour)%num_of_agents:
                    neighbours.append(1)
                    found =True
                    break
                    
            if not found:
                neighbours.append(0)
        adj.append(neighbours)

    return adj


import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os


def watts_strogatz_deterministic():

    adj = [[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0], 
       [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0], 
       [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0], 
       [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1], 
       [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0], 
       [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0], 
       [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1], 
       [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1], 
       [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0], 
       [1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0], 
       [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0], 
       [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]]

    return adj 

def watts_strogatz(num_of_agents, k, p):
    """
    Creates a Watts-Strogatz graph adjacency table.
    num_of_agents - The number of agents
    k - Each node is connected to k nearest neighbors in ring topology.
    p - The probability of rewiring each edge.

    Returns an adjacency table representing a Watts-Strogatz graph.
    
    """
    if k % 2 != 0 or k >= num_of_agents:
        raise ValueError("k must be even and less than num_of_agents")

    # Initialize adjacency matrix with zeros
    adj = [[0 for _ in range(num_of_agents)] for _ in range(num_of_agents)]
    
    # Create a ring lattice (regular graph)
    for i in range(num_of_agents):
        for j in range(1, k // 2 + 1): 
            right_neighbor = (i + j) % num_of_agents
            left_neighbor = (i - j) % num_of_agents
            adj[i][right_neighbor] = 1
            adj[right_neighbor][i] = 1  # Ensure symmetry
            adj[i][left_neighbor] = 1
            adj[left_neighbor][i] = 1  # Ensure symmetry
    

    plot_graph(adj, num_of_agents, 'wattsstrogatz_prewiring.png', k)

    # Rewire edges with probability p
    for i in range(num_of_agents):
        for j in range(1, k // 2 + 1):
            if random.random() < p:
                old_neighbor = (i + j) % num_of_agents
                potential_new_neighbors = [n for n in range(num_of_agents) if n != i and adj[i][n] == 0 and adj[n][i] == 0]
                if potential_new_neighbors:
                    new_neighbor = random.choice(potential_new_neighbors)

                    # Remove connection to the original neighbor
                    adj[i][old_neighbor] = 0
                    adj[old_neighbor][i] = 0

                    # Add new rewired connection
                    adj[i][new_neighbor] = 1
                    adj[new_neighbor][i] = 1
    
    plot_graph(adj, num_of_agents, 'wattsstrogatz_postwiring.png', k)


    return adj

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

def star_graph(num_of_agents, centre):

    """
    Creates a star graph adjacency table
    num_of_agents - The number of agents

    centre - The agent to be in a centre.  

    Return a star graph adjacency table with the agent being in the centre corresponding to arg centre
    """

    adj = []

    for i in range(num_of_agents):
        neighbours = []
        if i == centre:
            for j in range(num_of_agents):
                if i == j:
                    neighbours.append(0)
                else:
                    neighbours.append(1)

        else:
            for j in range(num_of_agents):
                if j == centre:
                    neighbours.append(1)
                else:
                    neighbours.append(0)

        adj.append(neighbours)

    return adj


def create_bipatrite(top_half, bottom_half):

    """
    Creates a Bipatrite adjacency table - Never used in project
    top_half - The agents which are in the top half
    bottom_half - The agents which are in the bottom half
    
    Returns a Bipatrite adjacency table"""


    adj = []
    # Assume agent numbers are 0 - (len-1)
    for i in range(len(top_half+bottom_half)):
        neighbours = []
        top = i in top_half  #Assume that if not in top half its in bottom

        for j in range(len(top_half+bottom_half)):
            if top and j in bottom_half:
                neighbours.append(1)
            elif top and j not in bottom_half:
                neighbours.append(0)
            elif not top and j in top_half:
                neighbours.append(1)
            elif not top and j not in top_half:
                neighbours.append(0)

        adj.append(neighbours)

    return adj

def fully_connected(num_of_agents):

    """
    Creates a Fully connected adjacency table
    num_of_agents - The number of agents

    Return 
    A fully connected adjacency table 
    """
    
    return [[0 if i == j else 1 for i in range(num_of_agents)] for j in range(num_of_agents)]

# GRAPH
# Adjacency graph & connection slow & gamma hop
multiple_graph_parameters = [
    {   # Fully Connected
        'graph': fully_connected(4),
        'connection_slow': False,
        'gamma_hop': 1
    },
    {   # Line Graph with gamma = 3
        'graph': line_graph(12),
        'connection_slow': True,
        'gamma_hop': 1
    },
    {   # No communication
        'graph': [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]],
        'connection_slow': True,
        'gamma_hop': 0
    },
    {   # Bipatrite K6,6 ## DO NOT USE THIS ONE
        'graph': create_bipatrite([0,1,2,3,4,5], [6,7,8,9,10,11]),
        'connection_slow': True,
        'gamma_hop': 1,
    },
    {   # Star centred on Agent 2
        'graph': star_graph(4, 2),
        'connection_slow': True,
        'gamma_hop': 1
    },
    {   # Ring with 10 connections
        'graph': ring_graph(12, 2),
        'connection_slow': True,
        'gamma_hop': 2
    },
]

# AGENT
# What agent to use
# Number of agents
# Size of state space
agent_multiple_parameters = [
    {
        'agent_choice': AgentType.RANDOM,
        'num_of_agents': 4,
        'size_of_state_space': 10**4
    },
    {
        'agent_choice': AgentType.RANDOM,
        'num_of_agents': 12,
        'size_of_state_space': 40**4
    },
    {
        'agent_choice': AgentType.IQL,
        'num_of_agents': 4,
        'size_of_state_space': 10**4
    },
    {
        'agent_choice': AgentType.IQL,
        'num_of_agents': 12,
        'size_of_state_space': 40**4
    },
    {
        'agent_choice': AgentType.ORIGINAL,
        'num_of_agents': 4,
        'size_of_state_space': 10**4
    },
    {
        'agent_choice': AgentType.ORIGINAL,
        'num_of_agents': 12,
        # 'size_of_state_space': 40**4
        'size_of_state_space': 10**4
    },
        {
        'agent_choice': AgentType.EB_Lidard,
        'num_of_agents': 4,
        'size_of_state_space': 10**4 
    },
    {
        'agent_choice': AgentType.EB_Lidard,
        'num_of_agents': 12,
        'size_of_state_space': 40**4
    },
]

# IQL
# Gamma, alpha, Greedy
iql_multiple_parameters = [
    {
        'gamma': 0.8,
        'greedy': 0.5,
        'alpha': 0.8
    }
]

# UCB
# c value
# prob value
ucb_marl_multiple_parameters = [
    {
        'c': 0.02,
        'probability': 0.1,
    }
]

eb_marl_multiple_parameters = [
    {
        'initial_decay_factor': 1, #.1 #10 #100
        'decay_rate': 0.5,
        'scaling_factor': 0.0001, 
        'probability': .1,
    }
]

# EVALUATION
# NUMBER_OF_TRIALS
# NUM_EVALUATION_EPISODES
# EVALUATION_INTERVALS
evaluation_multiple_parameters = [
    {
        'num_of_trials': 1,
        'num_evaluation_episodes': 8, 
        'evaluation_interval': 1,
    }
]

# REWARD
# Reward 'function' to use
reward_multiple_parameters = [
    {
        'reward': 'mean',
    },
    {
        'reward': 'split'
    },
    {
        'reward': 'split_all'
    },
    {
        'reward': 'split_all_7&8'
    }
]


# TRAIN
# NUM_OF_CYCLES
# NUM_OF_EPISODES
# LOCAL_RATIO = 0
train_multiple_parameters = [
    {
        'num_of_episodes': 50000,
        'num_of_cycles': 10,
        'local_ratio': 0,

    }
]

# Whether agents should be tested in the same position or switched.  
switch_multiple_parameters = [
    {
        'switch': True
    }
]

# Whether the graph should be a dynamic graph (so depending on distance and not on an adj table)
dynamic_parameters = [
    {
        'dynamic': False
    }
]


experiments_choice = [

#EXPERIMENT 1 ----------------------------------------------------------

    # {   # Fully Connected with Gamma = 0
    #     'graph': fully_connected(12),
    #     'connection_slow': False,
    #     'gamma_hop': 0,
    #     'experiment_name': 'Complete Graph, M = 12, γ = 0',
    #     'num_agents': 12
    # },
    
    # {   # Fully Connected with Gamma = 1
    #     'graph': fully_connected(12),
    #     'connection_slow': False,
    #     'gamma_hop': 1,
    #     'experiment_name': 'Complete Graph, M = 12, γ = 1',
    #     'num_agents': 12
    # },
    
    # {   # Fully Connected with Gamma = 1
    #     'graph': fully_connected(4),
    #     'connection_slow': False,
    #     'gamma_hop': 1,
    #     'experiment_name': 'Complete Graph, M = 4, γ = 1',
    #     'num_agents': 4
    # },
    
    # {   # Star with Gamma = 1, Center on Agent 1
    #     'graph': star_graph(12, 1),
    #     'connection_slow': True,
    #     'gamma_hop': 1,
    #     'experiment_name': 'Star, M = 12, γ = 1',
    #     'num_agents': 12
        
    # },
    
    # {   # Star with Gamma = 2, Center on Agent 1
    #     'graph': star_graph(12, 1),
    #     'connection_slow': True,
    #     'gamma_hop': 2,
    #     'experiment_name': 'Star, M = 12, γ = 2',
    #     'num_agents': 12
        
    # },
    
# #EXPERIMENT 2 ----------------------------------------------------------
    
    {   # Line Graph with Gamma = 3
        'graph': line_graph(12),
        'connection_slow': True,
        'gamma_hop': 3,
        'experiment_name': 'Line, γ = 3',
        'num_agents': 12
        
    },
    
    {   # Line Graph with Gamma = 6
        'graph': line_graph(12),
        'connection_slow': True,
        'gamma_hop': 6,
        'experiment_name': 'Line, γ = 6',
        'num_agents': 12
        
    },
    
    {   # Ring Degree 2 with Gamma = 1
        'graph': ring_graph(12, 2),
        'connection_slow': True,
        'gamma_hop': 1,
        'experiment_name': 'Lattice, γ = 1, K = 2',
        'num_agents': 12
         
    },
    
    {   # Ring Degree 2 with Gamma = 2
        'graph': ring_graph(12, 2),
        'connection_slow': True,
        'gamma_hop': 2,
        'experiment_name': 'Lattice, γ = 2, K = 2',
        'num_agents': 12
        
    },   
    
# # #EXPERIMENT 2.5 ----------------------------------------------------------
    
    # {   # Line Graph with Gamma = 1
    #     'graph': line_graph(12),
    #     'connection_slow': True,
    #     'gamma_hop': 1,
    #     'experiment_name': 'Line, γ = 1',
    #     'num_agents': 12
        
    # },
    
    # {   # Line Graph with Gamma = 6
    #     'graph': line_graph(12),
    #     'connection_slow': True,
    #     'gamma_hop': 6,
    #     'experiment_name': 'Line, γ = 6',
    #     'num_agents': 12
        
    # },
    
    # {   # Ring Degree 2 with Gamma = 1
    #     'graph': ring_graph(12, 2),
    #     'connection_slow': True,
    #     'gamma_hop': 1,
    #     'experiment_name': 'Lattice, γ = 1, K = 2',
    #     'num_agents': 12
         
    # },
    
    # {   # Ring Degree 2 with Gamma = 2
    #     'graph': ring_graph(12, 2),
    #     'connection_slow': True,
    #     'gamma_hop': 2,
    #     'experiment_name': 'Lattice, γ = 2, K = 2',
    #     'num_agents': 12
        
    # },   

# # #EXPERIMENT 3 ----------------------------------------------------------

    
# # #Watts-Strogatz Deterministic 

#     { #Watts-Strogatz with Gamma = 1, K = 4, P = 0.5
#          'graph': watts_strogatz_deterministic(), 
#          'connection_slow': True, 
#          'gamma_hop': 1, 
#          'experiment_name': 'Watts-Strogatz, γ = 1, K = 4, P = 0.5',
#          'num_agents': 12
         
#     },
    
#     { #Watts-Strogatz with Gamma = 2, K = 4, P = 0.5
#          'graph': watts_strogatz_deterministic(), 
#          'connection_slow': True, 
#          'gamma_hop': 2, 
#          'experiment_name': 'Watts-Strogatz, γ = 2, K = 4, P = 0.5',
#          'num_agents': 12
         
#     },    
    
#     {   # Ring Degree 4 with Gamma = 1
#         'graph': ring_graph(12, 4),
#         'connection_slow': True,
#         'gamma_hop': 1,
#         'experiment_name': 'Lattice, γ = 1, K = 4',
#         'num_agents': 12
         
#     },
    
#     {   # Ring Degree 4 with Gamma = 2
#         'graph': ring_graph(12, 4),
#         'connection_slow': True,
#         'gamma_hop': 2,
#         'experiment_name': 'Lattice, γ = 2, K = 4',
#         'num_agents': 12
        
#     },   
    
# # #EXPERIMENT 3.5 ----------------------------------------------------------

    
# # # #Watts-Strogatz Deterministic 

#     { #Watts-Strogatz with Gamma = 1, K = 4, P = 0.5
#          'graph': watts_strogatz_deterministic(), 
#          'connection_slow': True, 
#          'gamma_hop': 1, 
#          'experiment_name': 'Watts-Strogatz, γ = 1, K = 4, P = 0.5',
#          'num_agents': 12
         
#     },
    
#     { #Watts-Strogatz with Gamma = 2, K = 4, P = 0.5
#          'graph': watts_strogatz_deterministic(), 
#          'connection_slow': True, 
#          'gamma_hop': 2, 
#          'experiment_name': 'Watts-Strogatz, γ = 2, K = 4, P = 0.5',
#          'num_agents': 12
         
#     },    
    
#     {   # Ring Degree 4 with Gamma = 1
#         'graph': ring_graph(12, 4),
#         'connection_slow': True,
#         'gamma_hop': 1,
#         'experiment_name': 'Lattice, γ = 1, K = 4',
#         'num_agents': 12
         
#     },
    
#     {   # Ring Degree 4 with Gamma = 2
#         'graph': ring_graph(12, 4),
#         'connection_slow': True,
#         'gamma_hop': 2,
#         'experiment_name': 'Lattice, γ = 2, K = 4',
#         'num_agents': 12
        
#     },   
    
#     {   # Fully Connected with Gamma = 1
#         'graph': fully_connected(12),
#         'connection_slow': False,
#         'gamma_hop': 1,
#         'experiment_name': 'Complete Graph, γ = 1',
#         'num_agents': 12
    # },

#EXPERIMENT 4 ----------------------------------------------------------
    

#Watts-Strogatz Probabilistic 


    # { #Watts-Strogatz with Gamma = 1, K = 4, P = 0.5
    #     'graph': watts_strogatz(12, 4, 0.5), 
    #     'connection_slow': True, 
    #     'gamma_hop': 1, 
    #     'experiment_name': 'Watts-Strogatz, γ = 1, K = 4, P = 0.5'
    # },
    
    # { #Watts-Strogatz with Gamma = 2, K = 4, P = 0.5
    #     'graph': watts_strogatz(12, 4, 0.5), 
    #     'connection_slow': True, 
    #     'gamma_hop': 2, 
    #     'experiment_name': 'Watts-Strogatz, γ = 2, K = 4, P = 0.5'
    # },    
    
    # {   # Ring Degree 4 with Gamma = 1
    #     'graph': ring_graph(12, 4),
    #     'connection_slow': True,
    #     'gamma_hop': 1,
    #     'experiment_name': 'Ring, γ = 1, K = 4'
    # },
    
    # {   # Ring Degree 4 with Gamma = 2
    #     'graph': ring_graph(12, 4),
    #     'connection_slow': True,
    #     'gamma_hop': 2,
    #     'experiment_name': 'Ring, γ = 2, K = 4'
    # },   
    

]

graph_hyperparameters = multiple_graph_parameters[0] #Fully connected
# graph_hyperparameters = multiple_graph_parameters[1] #Line
# graph_hyperparameters = multiple_graph_parameters[4] #  Star
# graph_hyperparameters = multiple_graph_parameters[5] #Ring

agent_hyperparameters = agent_multiple_parameters[4] #Vanilla Lidard 4 agents
# agent_hyperparameters = agent_multiple_parameters[6] #EB Lidard 4 agents


# agent_hyperparameters = agent_multiple_parameters[5] #Vanilla Lidard 12 agents
# agent_hyperparameters = agent_multiple_parameters[7] #EB Lidard 12 agents

iql_hyperparameters = iql_multiple_parameters[0]

ucb_marl_hyperparameters = ucb_marl_multiple_parameters[0]

eb_marl_hyperparameters = eb_marl_multiple_parameters[0]

evaluation_hyperparameters = evaluation_multiple_parameters[0]

# reward_function = reward_multiple_parameters[0] #Mean reward 4

reward_function = reward_multiple_parameters[0] #12
# reward_function = reward_multiple_parameters[2] #12

train_hyperparameters = train_multiple_parameters[0]

switch_hyperparameters = switch_multiple_parameters[0]

dynamic_hyperparameters = dynamic_parameters[0]