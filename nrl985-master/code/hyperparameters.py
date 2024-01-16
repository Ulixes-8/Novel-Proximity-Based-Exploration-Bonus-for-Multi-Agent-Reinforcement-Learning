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
    k - The number of agents left or right to be close neighbours.  k = 1 means a ring graph of degree 2, k = 2, degree 4 etc.

    Return a ring graph adjacency table of degree k*2.

    """

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
        'graph': line_graph(4),
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
        'graph': ring_graph(4, 1),
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
        'num_of_episodes': 1000,
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

# graph_hyperparameters = multiple_graph_parameters[0] #Fully connected
# graph_hyperparameters = multiple_graph_parameters[1] #Line
# graph_hyperparameters = multiple_graph_parameters[4] #  Star
graph_hyperparameters = multiple_graph_parameters[5] #Ring

agent_hyperparameters = agent_multiple_parameters[4] #Vanilla Lidard 4 agents
# agent_hyperparameters = agent_multiple_parameters[6] #EB Lidard 4 agents


# agent_hyperparameters = agent_multiple_parameters[5] #Vanilla Lidard 12 agents
# agent_hyperparameters = agent_multiple_parameters[7] #EB Lidard 12 agents

iql_hyperparameters = iql_multiple_parameters[0]

ucb_marl_hyperparameters = ucb_marl_multiple_parameters[0]

eb_marl_hyperparameters = eb_marl_multiple_parameters[0]

evaluation_hyperparameters = evaluation_multiple_parameters[0]

reward_function = reward_multiple_parameters[0] #Mean reward 4
# reward_function = reward_multiple_parameters[2] #12

train_hyperparameters = train_multiple_parameters[0]

switch_hyperparameters = switch_multiple_parameters[0]

dynamic_hyperparameters = dynamic_parameters[0]