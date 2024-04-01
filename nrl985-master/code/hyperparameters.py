from enum import Enum
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

class AgentType(Enum):

    """Specifies the different agent types used"""

    RANDOM = 'RANDOM'
    IQL = 'IQL'
    ORIGINAL = 'ORIGINAL'
    EB_Lidard = 'EB_Lidard'


##Network topolgies. Feel free to define your own, but it should look like the below (return an adjacency table)
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
    
    adj = [[0 for _ in range(num_of_agents)] for _ in range(num_of_agents)]

    for i in range(num_of_agents):
        for neighbour in range(1, k+1):
            right = (i + neighbour) % num_of_agents
            left = (i - neighbour) % num_of_agents
            adj[i][right] = 1
            adj[i][left] = 1

    return adj





def watts_strogatz_deterministic(): # This is a WS network we found with brute force that exhibits the small world property when we do fully random swapping scheme. 

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
#Legacy code. You do not need to use it if you're using the experiment pipeline, which is far more optimized, functional, and easier to use. 
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

#If you are using the experiment pipeline, you do not need to use this. This is legacy code. 
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
        'size_of_state_space': 40**4
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

# IQL hyperparamters. Legacy code. 
# Gamma, alpha, Greedy
iql_multiple_parameters = [
    {
        'gamma': 0.8,
        'greedy': 0.5,
        'alpha': 0.8
    }
]

# UCB hyperparameters 
# c value
# prob value
ucb_marl_multiple_parameters = [
    {
        'c': 0.02, # these have been shown to be the optimal hyperparameters over 1000 episodes for 4 agents in all topologies... 
        'probability': 0.1,
    }
]

#The hyperparamters of the PB algorithm. Tune them wisely. 
eb_marl_multiple_parameters = [
    {
        'initial_decay_factor': 1, #Keep this at 1. It is identical to the scaling factor. 
        'decay_rate': 0.25, # generally should be lower the faster communication is, unless it is not full. 
        'scaling_factor': .0001, # generally should be lower the more agents you have. 
        'probability': .1, #Keep at .1.
    }
]

# EVALUATION
# NUMBER_OF_TRIALS
# NUM_EVALUATION_EPISODES
# EVALUATION_INTERVALS
evaluation_multiple_parameters = [
    {
        'num_of_trials': 2, # How many trials do you want? i recommend not touching the others. 
        'num_evaluation_episodes': 8, 
        'evaluation_interval': 1,
    }
]

# REWARD
# Reward 'function' to use
# Keep it on mean reward unless you know what you're doing. 
reward_multiple_parameters = [
    {
        'reward': 'mean', #Stick with this. The rest is legacy code. 
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
        'num_of_episodes': 100, # Just change the number of episodes. Everything else is fine.
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

# Whether the graph should be a dynamic graph (so depending on distance and not on an adj table). Legacy code.
dynamic_parameters = [
    {
        'dynamic': False
    }
]














#### Define experiments in exactly this format (a list of dictionaries where each dictionary corresponds to a sub-experiment within the larger experiment.) 
### Do not include more the 4 or the final performance graph will look hideous.
### You can mix the number of agents in experiments, but please make sure that they first occupy the same state space and that their average euclidean distance from the goal state is about the same. Otherwise, you will get ugly charts. 
### If the agents occupy a different state space or do not have roughly the same euclidean distance from the goal state, then it is better just to put them in different experiments. 

experiment_1 = [
    {   
        'graph': star_graph(4, 2),
        'connection_slow': True,
        'gamma_hop': 2,
        'experiment_name': 'PEB - Star, M = 4, γ = 2',
        'num_agents': 4,
        'agent_type': AgentType.EB_Lidard
    },
    {   
        'graph': star_graph(4, 2),
        'connection_slow': True,
        'gamma_hop': 2,
        'experiment_name': 'UCB - Star, M = 4, γ = 2',
        'num_agents': 4,
        'agent_type': AgentType.ORIGINAL
    },
]

experiment_2 = [

    {   
        'graph': star_graph(8, 2),
        'connection_slow': True,
        'gamma_hop': 2,
        'experiment_name': 'PEB - Star, M = 8, γ = 2',
        'num_agents': 8,
        'agent_type': AgentType.EB_Lidard
    },
    {   
        'graph': star_graph(8, 2),
        'connection_slow': True,
        'gamma_hop': 2,
        'experiment_name': 'UCB - Star, M = 8, γ = 2',
        'num_agents': 8,
        'agent_type': AgentType.ORIGINAL
    },
]

experiment_3 = [
    {   
        'graph': star_graph(4, 2),
        'connection_slow': True,
        'gamma_hop': 1,
        'experiment_name': 'PEB - Star, M = 4, γ = 1',
        'num_agents': 4,
        'agent_type': AgentType.EB_Lidard
    },

    {   
        'graph': star_graph(4, 2),
        'connection_slow': True,
        'gamma_hop': 1,
        'experiment_name': 'UCB - Star, M = 4, γ = 1',
        'num_agents': 4,
        'agent_type': AgentType.ORIGINAL
    },
    
]

experiment_4 = [

    {   
        'graph': star_graph(8, 2),
        'connection_slow': True,
        'gamma_hop': 1,
        'experiment_name': 'PEB - Star, M = 8, γ = 1',
        'num_agents': 8,
        'agent_type': AgentType.EB_Lidard
    },

    {   
        'graph': star_graph(8, 2),
        'connection_slow': True,
        'gamma_hop': 1,
        'experiment_name': 'UCB - Star, M = 8, γ = 1',
        'num_agents': 8,
        'agent_type': AgentType.ORIGINAL
    },
]



experiment_5 = [
    {   
        'graph': fully_connected(4),
        'connection_slow': False,
        'gamma_hop': 1,
        'experiment_name': 'PEB - Complete Graph, M = 4, γ = 1',
        'num_agents': 4,
        'agent_type': AgentType.EB_Lidard
    },

    {  
        'graph': fully_connected(4),
        'connection_slow': False,
        'gamma_hop': 1,
        'experiment_name': 'UCB - Complete Graph, M = 4, γ = 1',
        'num_agents': 4,
        'agent_type': AgentType.ORIGINAL
    },

]

experiment_6 = [

    {   
        'graph': fully_connected(8),
        'connection_slow': False,
        'gamma_hop': 1,
        'experiment_name': 'PEB - Complete Graph, M = 8, γ = 1',
        'num_agents': 8,
        'agent_type': AgentType.EB_Lidard
    },

    {   
        'graph': fully_connected(8),
        'connection_slow': False,
        'gamma_hop': 1,
        'experiment_name': 'UCB - Complete Graph, M = 8, γ = 1',
        'num_agents': 8,
        'agent_type': AgentType.ORIGINAL
    },
]

experiment_7 = [    
    {  
        'graph': line_graph(4),
        'connection_slow': True,
        'gamma_hop': 3,
        'experiment_name': 'PEB - Line, M = 4, γ = 3',
        'num_agents': 4,
        'agent_type': AgentType.EB_Lidard
    },

    {   
        'graph': line_graph(4),
        'connection_slow': True,
        'gamma_hop': 3,
        'experiment_name': 'UCB - Line, M = 4, γ = 3',
        'num_agents': 4,
        'agent_type': AgentType.ORIGINAL
    },

]


experiment_8 = [    

    {   
        'graph': line_graph(8),
        'connection_slow': True,
        'gamma_hop': 4,
        'experiment_name': 'PEB - Line, M = 8, γ = 4',
        'num_agents': 8,
        'agent_type': AgentType.EB_Lidard
    },

    {  
        'graph': line_graph(8),
        'connection_slow': True,
        'gamma_hop': 4,
        'experiment_name': 'UCB - Line, M = 8, γ = 4',
        'num_agents': 8,
        'agent_type': AgentType.ORIGINAL
    },
]

experiment_9 = [    
    {  
        'graph': line_graph(4),
        'connection_slow': True,
        'gamma_hop': 2,
        'experiment_name': 'PEB - Line, M = 4, γ = 2',
        'num_agents': 4,
        'agent_type': AgentType.EB_Lidard
    },
    
    {   
        'graph': line_graph(4),
        'connection_slow': True,
        'gamma_hop': 2,
        'experiment_name': 'UCB - Line, M = 4, γ = 2',
        'num_agents': 4,
        'agent_type': AgentType.ORIGINAL
    },

]

experiment_10 = [    

    {   
        'graph': line_graph(8),
        'connection_slow': True,
        'gamma_hop': 7,
        'experiment_name': 'PEB - Line, M = 8, γ = 7',
        'num_agents': 8,
        'agent_type': AgentType.EB_Lidard
    },

    {   
        'graph': line_graph(8),
        'connection_slow': True,
        'gamma_hop': 7,
        'experiment_name': 'UCB - Line, M = 8, γ = 7',
        'num_agents': 8,
        'agent_type': AgentType.ORIGINAL
    },
]

experiment_11 = [    
    {   
        'graph': line_graph(4),
        'connection_slow': True,
        'gamma_hop': 1,
        'experiment_name': 'PEB - Line, M = 4, γ = 1',
        'num_agents': 4,
        'agent_type': AgentType.EB_Lidard
    },

    {   
        'graph': line_graph(4),
        'connection_slow': True,
        'gamma_hop': 1,
        'experiment_name': 'UCB - Line, M = 4, γ = 1',
        'num_agents': 4,
        'agent_type': AgentType.ORIGINAL
    },

]

experiment_12 = [    

    {   
        'graph': line_graph(8),
        'connection_slow': True,
        'gamma_hop': 1,
        'experiment_name': 'PEB - Line, M = 8, γ = 1',
        'num_agents': 8,
        'agent_type': AgentType.EB_Lidard
    },
 
    {   
        'graph': line_graph(8),
        'connection_slow': True,
        'gamma_hop': 1,
        'experiment_name': 'UCB - Line, M = 8, γ = 1',
        'num_agents': 8,
        'agent_type': AgentType.ORIGINAL
    },
]
experiment_13 = [
    {   
        'graph': ring_graph(4, 2),
        'connection_slow': True,
        'gamma_hop': 2,
        'experiment_name': 'PEB - Lattice, M = 4, γ = 2, K = 2',
        'num_agents': 4,
        'agent_type': AgentType.EB_Lidard
    },

    {   
        'graph': ring_graph(4, 2),
        'connection_slow': True,
        'gamma_hop': 2,
        'experiment_name': 'UCB - Lattice, M = 4, γ = 2, K = 2',
        'num_agents': 4,
        'agent_type': AgentType.ORIGINAL
    },

]

experiment_14 = [

    {   
        'graph': ring_graph(8, 4),
        'connection_slow': True,
        'gamma_hop': 2,
        'experiment_name': 'PEB - Lattice, M = 8, γ = 2, K = 4',
        'num_agents': 8,
        'agent_type': AgentType.EB_Lidard
    },   
 
    {   
        'graph': ring_graph(8, 4),
        'connection_slow': True,
        'gamma_hop': 2,
        'experiment_name': 'UCB - Lattice, M = 8, γ = 2, K = 4',
        'num_agents': 8,
        'agent_type': AgentType.ORIGINAL
    },   
]

experiment_15 = [
    {   
        'graph': ring_graph(4, 2),
        'connection_slow': True,
        'gamma_hop': 1,
        'experiment_name': 'PEB - Lattice, M = 4, γ = 1, K = 2',
        'num_agents': 4,
        'agent_type': AgentType.EB_Lidard
    },

    {   
        'graph': ring_graph(4, 2),
        'connection_slow': True,
        'gamma_hop': 1,
        'experiment_name': 'UCB - Lattice, M = 4, γ = 1, K = 2',
        'num_agents': 4,
        'agent_type': AgentType.ORIGINAL
    },

]


experiment_16 = [

    {   
        'graph': ring_graph(8, 4),
        'connection_slow': True,
        'gamma_hop': 1,
        'experiment_name': 'PEB - Lattice, M = 8, γ = 1, K = 4',
        'num_agents': 8,
        'agent_type': AgentType.EB_Lidard
    },  
    {   
        'graph': ring_graph(8, 4),
        'connection_slow': True,
        'gamma_hop': 1,
        'experiment_name': 'UCB - Lattice, M = 8, γ = 1, K = 4',
        'num_agents': 8,
        'agent_type': AgentType.ORIGINAL
    },   
]


experiment_17 = [
    { 
         'graph': watts_strogatz_deterministic(), 
         'connection_slow': True, 
         'gamma_hop': 2, 
         'experiment_name': 'PEB - Watts-Strogatz, γ = 2, K = 4, P = 0.5',
         'num_agents': 12,
        'agent_type': AgentType.EB_Lidard
    },
    {   
        'graph': ring_graph(12, 4),
        'connection_slow': True,
        'gamma_hop': 2,
        'experiment_name': 'PEB - Lattice, γ = 2, K = 4',
        'num_agents': 12,
        'agent_type': AgentType.EB_Lidard
    },
    { 
         'graph': watts_strogatz_deterministic(), 
         'connection_slow': True, 
         'gamma_hop': 2, 
         'experiment_name': 'UCB - Watts-Strogatz, γ = 2, K = 4, P = 0.5',
         'num_agents': 12,
        'agent_type': AgentType.ORIGINAL
    },    
    {   
        'graph': ring_graph(12, 4),
        'connection_slow': True,
        'gamma_hop': 2,
        'experiment_name': 'UCB - Lattice, γ = 2, K = 4',
        'num_agents': 12,
        'agent_type': AgentType.ORIGINAL
    },   
    
]
    
#Define a new experiment above as a list of dictionaries. Add that here. 
experiments_choice = [experiment_1, experiment_2, experiment_3, experiment_4, 
                      experiment_5, experiment_6, experiment_7, experiment_8, experiment_9,
                      experiment_10, experiment_11, experiment_12, experiment_13, experiment_14,
                      experiment_15, experiment_16, experiment_17]

#Hyperparmameters that feed into the experiment pipeline. 
ucb_marl_hyperparameters = ucb_marl_multiple_parameters[0]
eb_marl_hyperparameters = eb_marl_multiple_parameters[0]
evaluation_hyperparameters = evaluation_multiple_parameters[0]
reward_function = reward_multiple_parameters[0] 
train_hyperparameters = train_multiple_parameters[0]
switch_hyperparameters = switch_multiple_parameters[0]


#Legacy code from previous student. Do not worry about it if you're using the experiment pipeline, which you ought to be using. 
graph_hyperparameters = multiple_graph_parameters[0] #Fully connected
agent_hyperparameters = agent_multiple_parameters[4] #Vanilla Lidard 4 agents
iql_hyperparameters = iql_multiple_parameters[0]
dynamic_hyperparameters = dynamic_parameters[0]