from collections import defaultdict
from statistics import mean, median, mode, variance
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from scipy.spatial.distance import euclidean
import numpy as np
# import torch 
import math 
import os
import uuid
import seaborn as sns

class Oracle:
    def __init__(self):
        # Initializes the universal n-table
        self.universal_nTable = defaultdict(lambda: defaultdict(int))
        self.real_state_map = {}  # Mapping from hashed states to real states
        self.episode_stats = [] # List of dictionaries containing statistics for each episode
    
    def create_universal_nTable(self):
        return self.universal_nTable

    def sum_universal_nTable(self):
        total_count = 0
        for state in self.universal_nTable:
            for action in self.universal_nTable[state]:
                total_count += self.universal_nTable[state][action]
        return total_count
    
    def sum_top_four_states(self):
        state_action_sums = {}

        # Sum actions for each state
        for state, actions in self.universal_nTable.items():
            state_action_sums[state] = sum(actions.values())

        # Sort states by total actions and select the top four
        top_four_states = sorted(state_action_sums, key=state_action_sums.get, reverse=True)[:4]

        # Sum the actions for the top four states
        total_actions_top_four = sum(state_action_sums[state] for state in top_four_states)

        return total_actions_top_four

    def calculate_and_store_stats(self, mean_reward):
        """
        Calculate and store statistics, including the bad exploration score.
        """
        stats = self.calculate_statistics()
        stats['bad_exp_score'] = self.calculate_bad_exp_score(mean_reward)
        # bes = stats['bad_exp_score']
        # cc, _ = self.calculate_clustering_coefficient()
        # stats['clustering_coefficient'] = cc
        # bcc = self.calculate_bcc(bes, cc)
        # stats['bad_clustering_coefficient'] = bcc
        self.episode_stats.append(stats)
        
    def calculate_bcc(self, cc, bes):
        return min(1, (cc * (1 + bes)))
    
    def calculate_bad_exp_score(self, mean_reward, alpha=0.2, beta=0.8, delta=1000):
        """
        Calculate the bad exploration score based on the mean reward of the episode and mean visit count.
        :param mean_reward: Mean reward of the episode.
        :param mean_visit_count: Mean visit count for actions.
        :return: Bad exploration score.
        """
        def modified_sigmoid(x, scale=delta):
            return 1 / (1 + scale * math.exp(-x))
        
        all_counts = []

        for state, actions in self.universal_nTable.items():
            for action, count in actions.items():
                all_counts.append(count)

        # Calculate entropy
        total_count = sum(all_counts)
        probabilities = [count / total_count for count in all_counts]
        entropy_value = entropy(probabilities)
        normalized_entropy_value = modified_sigmoid(entropy_value)

        bes = (alpha * (1-normalized_entropy_value) + beta * (1 - math.exp(mean_reward))) / (alpha + beta)
        
        return bes

    def update(self, state, action):
        # Updates the n-table with the given state-action pair
        self.universal_nTable[state][action] += 1

    def get_visit_count(self, state, action):
        # Retrieves the visit count for a specific state-action pair
        return self.universal_nTable[state][action]
    
    def calculate_statistics(self):
        """
        Calculate mean, median, mode, entropy, variance, number of unique states, 
        and number of unique state-action pairs for the values in the universal n-table.
        """
        all_counts = []
        unique_state_action_pairs = 0

        for state, actions in self.universal_nTable.items():
            for action, count in actions.items():
                all_counts.append(count)
                unique_state_action_pairs += 1  # Count each state-action pair

        unique_states = len(self.universal_nTable)

        # Calculate entropy
        total_count = sum(all_counts)
        probabilities = [count / total_count for count in all_counts]
        entropy_value = entropy(probabilities)

        return {
            "mean": mean(all_counts),
            "median": median(all_counts),
            "mode": mode(all_counts),
            "entropy": entropy_value,
            "variance": variance(all_counts),
            "unique_states": unique_states,
            "unique_state_action_pairs": unique_state_action_pairs,
            "USA-to-US-ratio": unique_state_action_pairs / unique_states
        }

    def update_real_state_map(self, hash_value, real_state):
        """Store the mapping from state hash to its original real state."""
        self.real_state_map[hash_value] = real_state
        
    def create_bubble_plot(self):
        """
        Plots and saves a bubble plot of the visit counts in the universal nTable.
        """
        directory = "saved_data/observer_figs"
        if not os.path.exists(directory):
            os.makedirs(directory)

        object_id_key_name = f"oracle_{id(self)}"  # Unique identifier for the Oracle instance

        # Aggregate visitation counts by (x, y) coordinates
        aggregated_counts = {}
        for hashed_state, actions in self.universal_nTable.items():
            real_state = self.real_state_map[hashed_state]
            x, y = real_state[-2], real_state[-1]  # Extract x and y coordinates

            if (x, y) not in aggregated_counts:
                aggregated_counts[(x, y)] = 0
            aggregated_counts[(x, y)] += sum(actions.values())  # Sum visitation counts

        # Create the bubble plot
        x_coords, y_coords, sizes = zip(*[(x, y, count) for (x, y), count in aggregated_counts.items()])
        
        plt.figure(figsize=(10, 6))
        # Normalize the sizes for color mapping
        norm = mcolors.Normalize(vmin=min(sizes), vmax=max(sizes))
        scatter = plt.scatter(x_coords, y_coords, s=sizes, c=sizes, alpha=0.5, norm=norm)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Agent Exploration Bubble Plot')
        plt.colorbar(scatter, label='Visitation Counts')

        # Construct unique filename
        unique_filename = f"{directory}/{object_id_key_name}_bubble_plot_{uuid.uuid4()}.png"
        plt.savefig(unique_filename)
        print(f"Saved figure: {unique_filename}")
        plt.close()
    
    def plot_episode_statistics(self):
        """
        Plots and saves the time series data for each statistic over the episodes.
        """
        directory = "saved_data/observer_figs"
        if not os.path.exists(directory):
            os.makedirs(directory)

        object_id_key_name = f"oracle_{id(self)}"  # Unique identifier for the Oracle instance

        for stat in ['mean', 'median', 'mode', 'entropy', 'variance', 'unique_states', 'unique_state_action_pairs', "USA-to-US-ratio" ,"bad_exp_score"]:
            stat_values = [episode_stat[stat] for episode_stat in self.episode_stats]
            plt.figure()
            plt.plot(stat_values)
            plt.title(f"Evolution of {stat} over Episodes")
            plt.xlabel("Episode")
            plt.ylabel(stat)

            # Construct unique filename
            unique_filename = f"{directory}/{object_id_key_name}_{stat}_{uuid.uuid4()}.png"
            plt.savefig(unique_filename)
            print(f"Saved figure: {unique_filename}")
            plt.close()

    def plot_visit_count_distribution(self):
        """
        Plots and saves a seaborn histogram of the visit counts in the universal nTable.
        """
        directory = "saved_data/observer_figs"
        if not os.path.exists(directory):
            os.makedirs(directory)

        object_id_key_name = f"oracle_{id(self)}"  # Unique identifier for the Oracle instance

        visit_counts = []
        for actions in self.universal_nTable.values():
            visit_counts.extend(actions.values())

        plt.figure(figsize=(10, 6))
        sns.histplot(visit_counts, bins=50, kde=False, color='blue')
        plt.title("Distribution of State-Action Visit Counts")
        plt.xlabel("Visit Counts")
        plt.ylabel("Frequency")
        plt.grid(True)

        # Construct unique filename
        unique_filename = f"{directory}/{object_id_key_name}_visit_count_distribution_{uuid.uuid4()}.png"
        plt.savefig(unique_filename)
        print(f"Saved figure: {unique_filename}")
        plt.close()
        
    def calculate_interaction_strength(self, real_state1, real_state2, visit_count1, visit_count2):
        # Euclidean distance
        
        # euclidean_dist = np.linalg.norm(real_state1 - real_state2)
        euclidean_dist = np.linalg.norm(real_state1[-2:] - real_state2[-2:])  # Consider only the x, y coordinates

        # Visitation count difference
        visitation_diff = abs(visit_count1 - visit_count2) / max(visit_count1, visit_count2, 1)

        # Interaction strength (balance as needed)
        interaction_strength = 1 / (1 + euclidean_dist + visitation_diff)
        # print(interaction_strength)
        return interaction_strength

    def build_state_graph(self, threshold=.5):
        G = nx.Graph()

        # Aggregate action counts by state
        action_counts_by_state = {state_hash: sum(actions.values()) for state_hash, actions in self.universal_nTable.items()}

        # Add nodes to the graph
        for state_hash in self.universal_nTable.keys():
            G.add_node(state_hash)

        for state_hash, actions in self.universal_nTable.items():
            real_state = np.array(self.real_state_map[state_hash])
            for other_state_hash, other_actions in self.universal_nTable.items():
                if other_state_hash != state_hash:
                    other_real_state = np.array(self.real_state_map[other_state_hash])
                    interaction_strength = self.calculate_interaction_strength(
                        real_state, other_real_state, 
                        action_counts_by_state[state_hash], action_counts_by_state[other_state_hash])

                    # Adding edge based on threshold
                    if interaction_strength >= threshold:
                        G.add_edge(state_hash, other_state_hash, weight=1+interaction_strength) 
                        
        return G
    
    def calculate_clustering_coefficient(self):
        # Build the state-action graph
        graph = self.build_state_graph()

        # Calculate the clustering coefficient for each node
        clustering_coeffs = nx.clustering(graph, weight='weight')

        # Calculate the average clustering coefficient
        avg_clustering_coeff = sum(clustering_coeffs.values()) / len(clustering_coeffs)
        return avg_clustering_coeff, graph
    
    def visualize_graph(self, G):
        """
        Visualize the state-action graph.
        """
        # G = self.build_state_action_graph()  # Assuming this method returns a NetworkX graph

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)  # positions for all nodes

        # nodes
        node_sizes = [700 for _ in range(len(G))]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes)

        # edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

        # labels (optional, you might want to skip this for large graphs)
        labels = {node: f"{node[0][-2:]}, {node[1]}" for node in G.nodes()}  # Customize as per your node structure
        nx.draw_networkx_labels(G, pos, labels, font_size=12, font_family="sans-serif")

        plt.axis("off")
        plt.title("State-Action Graph Visualization")
        plt.show()