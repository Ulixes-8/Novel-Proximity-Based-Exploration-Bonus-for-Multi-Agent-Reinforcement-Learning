from copy import deepcopy

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


def fully_connected(num_of_agents):

    """
    Creates a Fully connected adjacency table
    num_of_agents - The number of agents

    Return 
    A fully connected adjacency table 
    """
    
    return [[0 if i == j else 1 for i in range(num_of_agents)] for j in range(num_of_agents)]


def node_degree(adj):

    """Returns the average node degree
    
    adj - The adjacency table of the graph
    
    Return The average node degree"""


    num_of_nodes = 0
    for row in adj:
        for neighbour in row:
            if neighbour != 0:
                num_of_nodes += 1

    return num_of_nodes/len(adj)


def local_clustering(adj):

    """Finds the average local cluster measure
    adj - The adjacency table of the graph
    
    Returns the average local clustering value"""

    sum_of_clusters = 0

    for node in adj:
        k_i = 0
        l_i = 0
        neighbour_set = set()
        for i, neighbour in enumerate(node):
            if neighbour != 0:
                k_i += 1
                neighbour_set.add(i)

        for neighbour in neighbour_set:
            neighbour_row = adj[neighbour]
            for j, neighbour_neighbour in enumerate(neighbour_row):
                if j in neighbour_set and neighbour_neighbour != 0:
                    l_i += 1
        l_i /= 2
        if k_i == 0 or k_i == 1:
            c_i = 0
        else:
            c_i = (2* l_i) / (k_i * (k_i-1))
        sum_of_clusters += c_i

    return sum_of_clusters / len(adj)#

def num_of_links(adj):

    """Finds the number of connections in the graph
    adj - The adjacency table of the graph
    
    Return - The number of edges in the graph"""

    num_of_nodes = 0
    for row in adj:
        for neighbour in row:
            if neighbour != 0:
                num_of_nodes += 1

    return num_of_nodes/2

def convert_adj_to_power_graph(adj_table, gamma_hop, connection_slow=False):

    """Converts an adjacency table for a graph into an adjacency table for the corresponding power graph of distance gamma_hop

    Parameters
    -------------

    adj_table - The adjacency graph to convert

    gamma_hop - the amount the graph is allowed to change by

    connection_slow - False means the amount of jumps is factored in (so if the node is 2 jumps away its set to 2)

    Returns
    --------
    adjacency table of the power graph"""

    # If Gamma hop = 0 there is no communication. So [[0,0], [0,0]] for 2 agents
    if gamma_hop == 0:
        for i in range(len(adj_table)):
            for j in range(len(adj_table[i])):
                adj_table[i][j] = 0
        return adj_table
    
    # Goes through adj_table and adds to own dictionary the neighbours for each node
    neighbours = {z: [] for z in range(len(adj_table))}
    for i, row in enumerate(adj_table):
        for j, col in enumerate(row):
            if col != 0:
                i_neighbours = neighbours[i]
                i_neighbours.append((j, col))
   
    # Updates the adjacency table to be power graph.
    for k in range(gamma_hop-1):                 
        new_table = deepcopy(adj_table)

        # Goes over each agent
        for i, row in enumerate(adj_table):
            # Goes for every distance for node i to node j
            for j, col in enumerate(row):
                if col != 0:
                    neighbours_of_j = neighbours[j]
                    for neighbour, dis in neighbours_of_j:
                        if adj_table[i][neighbour] == 0 and i != neighbour:
                            if connection_slow:
                                speed_of_others = adj_table[i][j]
                                new_table[i][neighbour] = dis+speed_of_others
                            else:
                                new_table[i][neighbour] = 1
        adj_table = new_table

    return adj_table


def main_network():

    """Will print the average node degree, average local clustering and number of links of each of the graphs used in the testing"""

    data = {
        'Line Gamma 0': convert_adj_to_power_graph(line_graph(12), 0),
        'Line Gamma 10': convert_adj_to_power_graph(line_graph(12), 10),
        'Fully Connected': fully_connected(12),
        'Ring Two': ring_graph(12, 1),
        'Ring Four': ring_graph(12, 2),
        'Ring Six': ring_graph(12, 3),
        'Ring Eight': ring_graph(12, 4),
        'Ring Ten': ring_graph(12, 5),
        'Star Centre Zero': star_graph(12, 0),
        'Star Centre One': star_graph(12, 1),
        'Star Centre Two': star_graph(12, 2),
        'Star Centre Three': star_graph(12, 3),
        'Star Centre Four': star_graph(12, 4),
        'Star Centre Five': star_graph(12, 5),
        'Star Centre Six': star_graph(12, 6),
        'Star Centre Seven': star_graph(12, 7),
        'Star Centre Eight': star_graph(12, 8),
        'Star Centre Nine': star_graph(12, 9),
        'Star Centre Ten': star_graph(12, 10),
        'Star Centre Eleven': star_graph(12, 11),
        'Star Centre Zero Gamma Two': convert_adj_to_power_graph(star_graph(12, 0), 2),
    }

    final_data = {}
    print("Graph | Av Node Degree | Av Local Clustering | Total num of links |")
    for key, values in data.items():
        print(f'{key} | {node_degree(values)} | {local_clustering(values)} | {num_of_links(values)}')
        final_data[key] = (node_degree(values), local_clustering(values), num_of_links(values))

    return final_data

main_network()