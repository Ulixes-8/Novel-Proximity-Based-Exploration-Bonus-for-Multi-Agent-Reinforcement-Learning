import numpy as np
from copy import deepcopy

from hyperparameters import evaluation_hyperparameters, train_hyperparameters, agent_hyperparameters, AgentType
from train import _set_up
from show import episode_play_normal_marl
from file_management import save
from eb_marl_agent import EB_MARL_Comm
from hyperparameters import multiple_graph_parameters
from observer import Oracle
import matplotlib.pyplot as plt


NUMBER_OF_TRIALS = evaluation_hyperparameters['num_of_trials']
NUM_EVALUATION_EPISODES = evaluation_hyperparameters['num_evaluation_episodes']
EVALUATION_INTERVAL = evaluation_hyperparameters['evaluation_interval']
NUM_OF_EPISODES = train_hyperparameters['num_of_episodes']
NUM_OF_AGENTS = agent_hyperparameters['num_of_agents']
LOCAL_RATIO = train_hyperparameters['local_ratio']
NUM_OF_CYCLES = train_hyperparameters['num_of_cycles']

def evaluation_twelve(choice):
    # This is the reward from each episode
    reward_array_cumulative = np.array([np.zeros(6) for i in range(NUM_OF_EPISODES)])

    # This is the reward from evaluation runs
    reward_list_evaluation = np.array([np.zeros(6) for i in range(NUM_OF_EPISODES//EVALUATION_INTERVAL)])

    # This contains all the evaluation episodes
    reward_array_episode_num = np.array([episode_num for episode_num in range(1,NUM_OF_EPISODES+1) if episode_num % EVALUATION_INTERVAL == 0 ])

    
    episodes_array = np.array([i+1 for i in range(NUM_OF_EPISODES)])


    best_joint_policy = {}
    best_mean = -1000

    print(f'Training {NUM_OF_AGENTS} agents of type {choice} over {NUM_OF_EPISODES} episodes with trials {NUMBER_OF_TRIALS}')
    for trials_num in range(NUMBER_OF_TRIALS):
        agent_type, env, agents, train_choice = _set_up(choice)
        evaluation_pos = 0
        for episode_num in range(1, NUM_OF_EPISODES+1):
            reward = train_choice(env, agents, episode_num-1)
            reward_array_cumulative[episode_num-1] = reward
            if episode_num % 100 == 0:
                print(trials_num, episode_num)

            # Evaluate how good the agents are every EVALUATION_INTERVAL
            if episode_num % EVALUATION_INTERVAL == 0:
                # print(f"Evaluating at episode number {episode_num}")
                cumulative_reward = reward_list_evaluation[evaluation_pos]
                reward_here = 0
                for episode in range(NUM_EVALUATION_EPISODES):
                    reward = episode_play_normal_marl(env, agents, NUM_OF_CYCLES, NUM_OF_AGENTS, render=False )
                    reward_here += reward
                    
                cumulative_reward += (reward_here/NUM_EVALUATION_EPISODES)
                reward_list_evaluation[evaluation_pos] = cumulative_reward
                evaluation_pos += 1

        # After each trial we get a mean reward for the set of agents
        cumulative_reward = 0
        for run in range(NUM_EVALUATION_EPISODES *10):
            reward = episode_play_normal_marl(env, agents, NUM_OF_CYCLES, NUM_OF_AGENTS, render=False )
            cumulative_reward += reward
        mean_reward = cumulative_reward/(NUM_EVALUATION_EPISODES*10)

        if sum(mean_reward)/6 > best_mean:
            best_joint_policy = deepcopy(agents)
            best_mean = sum(mean_reward)/6

                
    reward_array = reward_array_cumulative/NUMBER_OF_TRIALS
    reward_array_evaluation = reward_list_evaluation / NUMBER_OF_TRIALS

    # Save all data
    save(best_joint_policy, episodes_array, [reward_array, reward_array_evaluation, reward_array_episode_num], agent_type, NUM_OF_AGENTS, NUM_OF_CYCLES, NUM_OF_EPISODES, LOCAL_RATIO)
    
    print(f'The Mean Reward is: {best_mean}')
    return best_mean 

import numpy as np


# def compute_aggregate_statistics_scaled(universal_nTable, num_agents):
#     all_visit_counts = [count/num_agents for state in universal_nTable.values() for count in state.values()]

#     average_visit_count = np.mean(all_visit_counts)
#     median_visit_count = np.median(all_visit_counts)

#     return average_visit_count, median_visit_count

# def plot_visit_count_distribution_scaled(universal_nTable, num_agents):
#     all_visit_counts = [count/num_agents for state in universal_nTable.values() for count in state.values()]

#     plt.hist(all_visit_counts, bins=30, edgecolor='black')
#     plt.xlabel('Average State-Action Visit Counts in Universal N-Table')
#     plt.ylabel('Frequency')
#     plt.title('Distribution of Scaled State-Action Visit Counts')
#     plt.show()

def evaluation(choice):
    """
    This will allow there to be evaluation of how well the code is working. 

    choice - The AgentType to use
    """

    if NUM_OF_AGENTS == 12:
        return evaluation_twelve(choice)
    
    reward_array_cumulative = np.zeros(NUM_OF_EPISODES)
    reward_list_evaluation = np.zeros(NUM_OF_EPISODES // EVALUATION_INTERVAL)
    reward_array_episode_num = np.array([episode_num for episode_num in range(1, NUM_OF_EPISODES + 1) if episode_num % EVALUATION_INTERVAL == 0 ])
    episodes_array = np.array([i + 1 for i in range(NUM_OF_EPISODES)])
    

    exploration_bonus_array_cumulative = np.zeros(NUM_OF_EPISODES)  # Initialize exploration bonus array

    best_joint_policy = {}
    best_mean = -1000
    
    print("The Oracle is created.")
    oracle = Oracle() # The Observer is trial persistent and will be used to track the universal n-table
    

    print(f'Training {NUM_OF_AGENTS} agents of type {choice} over {NUM_OF_EPISODES} episodes with trials {NUMBER_OF_TRIALS}')
    for trials_num in range(NUMBER_OF_TRIALS):
        agent_type, env, agents, train_choice = _set_up(choice)
        evaluation_pos = 0
        
        for episode_num in range(1, NUM_OF_EPISODES + 1):
            reward = train_choice(env, agents, episode_num - 1, oracle)
            reward_array_cumulative[episode_num - 1] += reward
            temp_bonus_list = []
            
            for agent_name, agent_obj in agents.items():
                if hasattr(agent_obj, 'exploration_bonuses'):
                    mean_bonus = np.mean(agent_obj.exploration_bonuses[-NUM_OF_CYCLES:])
                    temp_bonus_list.append(mean_bonus)

            if temp_bonus_list:
                exploration_bonus_array_cumulative[episode_num - 1] += np.mean(temp_bonus_list)

            if episode_num % 100 == 0:
                print(trials_num, episode_num)

            if episode_num % EVALUATION_INTERVAL == 0:
                cumulative_reward = 0
                for episode in range(NUM_EVALUATION_EPISODES):
                    reward = episode_play_normal_marl(env, agents, NUM_OF_CYCLES, NUM_OF_AGENTS, render=False)
                    cumulative_reward += reward

                reward_list_evaluation[evaluation_pos] += cumulative_reward / NUM_EVALUATION_EPISODES
                evaluation_pos += 1
                
            mean_reward_episode = reward_array_cumulative[episode_num - 1] / NUM_OF_AGENTS
            oracle.calculate_and_store_stats(mean_reward_episode)
        
        
        stats = oracle.calculate_statistics()
        oracle.create_bubble_plot()
        oracle.plot_episode_statistics()
        oracle.plot_visit_count_distribution()
        print("Statistics: ", stats)
        last_mean_reward = reward_array_cumulative[-1] / NUM_OF_AGENTS
        bes = oracle.calculate_bad_exp_score(last_mean_reward)
        print("Final Bad Exploration Score:", bes)
        cc, G = oracle.calculate_clustering_coefficient()
        print("Clustering Coefficient:", cc)
        # Assuming oracle.universal_nTable is your universal nTable
        total_actions = oracle.sum_universal_nTable()
        print("Total sum of the universal_nTable:", total_actions)
        sum_top_four_states = oracle.sum_top_four_states()        
        print("Sum of the top four states:", sum_top_four_states)
        
        
        # oracle.visualize_graph(G)
        # bcc = oracle.calculate_bcc(avg_clustering_coeff, bes)
        # print("Bad Clustering Coefficient:", bcc)

        # Compute the mean reward for this trial
        cumulative_reward = 0
        for run in range(NUM_EVALUATION_EPISODES * 10):
            # Switch agents 'agent_0' with 'agent_3', and 'agent_1' with 'agent_2'
            # print('Switching agents')
            agents['agent_0'], agents['agent_3'] = agents['agent_3'], agents['agent_0']
            agents['agent_1'], agents['agent_2'] = agents['agent_2'], agents['agent_1']
            
            reward = episode_play_normal_marl(env, agents, NUM_OF_CYCLES, NUM_OF_AGENTS, render=False)
            cumulative_reward += reward
            # print('Switching back')
            
                # Switch back agents to original configuration
            agents['agent_0'], agents['agent_3'] = agents['agent_3'], agents['agent_0']
            agents['agent_1'], agents['agent_2'] = agents['agent_2'], agents['agent_1']
            
        mean_reward = cumulative_reward / (NUM_EVALUATION_EPISODES * 10)

        if mean_reward > best_mean:
            best_joint_policy = deepcopy(agents)
            best_mean = mean_reward

    # Average rewards and exploration bonuses over all trials
    reward_array_cumulative /= NUMBER_OF_TRIALS
    exploration_bonus_array_cumulative /= NUMBER_OF_TRIALS


    # Save all data, including averaged exploration bonuses
    save(best_joint_policy, episodes_array, [reward_array_cumulative, reward_list_evaluation, reward_array_episode_num], agent_type, NUM_OF_AGENTS, NUM_OF_CYCLES, NUM_OF_EPISODES, LOCAL_RATIO, exploration_bonus_array_cumulative)
    # save(best_joint_policy, episodes_array, [reward_array_cumulative, reward_list_evaluation, reward_array_episode_num], agent_type, NUM_OF_AGENTS, NUM_OF_CYCLES, NUM_OF_EPISODES, LOCAL_RATIO, exploration_bonus_array_cumulative, detailed_exploration_bonus)

    print(f'The Mean Reward is: {best_mean}')
    return best_mean
