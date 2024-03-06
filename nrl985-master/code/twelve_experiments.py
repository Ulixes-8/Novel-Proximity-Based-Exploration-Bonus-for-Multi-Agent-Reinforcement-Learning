import numpy as np
from copy import deepcopy
from hyperparameters import evaluation_hyperparameters, train_hyperparameters, agent_hyperparameters, AgentType
from train import _set_up
from show import episode_play_normal_marl
from file_management import save
import matplotlib.pyplot as plt
import numpy as np
from hyperparameters import train_hyperparameters, agent_hyperparameters, dynamic_hyperparameters, AgentType
from env import create_env
from create_agents import create_agents
from utils import encode_state
from file_management import save
from reward_functions import final_reward
import numpy as np
import math
from adjacency import convert_adj_to_power_graph
from ucb_marl_agent import MARL_Comm
from observer import Oracle


NUMBER_OF_TRIALS = evaluation_hyperparameters['num_of_trials']
NUM_EVALUATION_EPISODES = evaluation_hyperparameters['num_evaluation_episodes']
EVALUATION_INTERVAL = evaluation_hyperparameters['evaluation_interval']
NUM_OF_EPISODES = train_hyperparameters['num_of_episodes']
# NUM_OF_AGENTS = agent_hyperparameters['num_of_agents']
LOCAL_RATIO = train_hyperparameters['local_ratio']
NUM_OF_CYCLES = train_hyperparameters['num_of_cycles']

def _episode_original_multiple(env, agents, episode_num, oracle=None):
    NUM_OF_AGENTS = len(agents)
    

    """
    This trains the original Lidard algorithm for MARL agents for one episode

    env - The parallel environment to be used

    agents - A dict containing the agents to be used

    episode_num - The episode number

    Return 
    The reward for that episode
    
    """
    
    
    
    agent_old_state = {agent: -1 for agent in agents.keys()}
    observations = env.reset()    

    t = 0
    while env.agents:
        t = t+1
        actions = {}
        for agent_name in agents.keys():        # Take action
            real_state = observations[agent_name]
            agent_old_state[agent_name] = encode_state(observations[agent_name], NUM_OF_AGENTS)
            action = _policy(agent_name, agents, observations[agent_name], False, t, episode_num)
            actions[agent_name] = action
            if oracle is not None: 
                oracle.update(agent_old_state[agent_name], action) #Update the oracle with the state-action pair
                oracle.update_real_state_map(agent_old_state[agent_name], real_state) # Update the oracle with the real state
            
            
        observations, rewards, terminations, truncations, infos = env.step(actions)
  
        for agent_name in agents.keys():        # Send messages
            agent_obj = agents[agent_name]
            agent_obj.message_passing(episode_num, t, agent_old_state[agent_name], actions[agent_name], 
                encode_state(observations[agent_name], NUM_OF_AGENTS), rewards[agent_name], agents)

        for agent_name in agents.keys():        # Update u and v
            agent_obj = agents[agent_name]
            agent_obj.update(episode_num, t, agent_old_state[agent_name], encode_state(observations[agent_name], NUM_OF_AGENTS),
                actions[agent_name], rewards[agent_name])
            
        for agent_name in agents.keys():       # Update the values
            agent_obj = agents[agent_name]
            agent_obj.update_values(episode_num, t)
    
    return final_reward(rewards)

def _set_up(experiment):

    """
    Sets up the environments and agents

    choice - The type of Agent to use

    Return
    The agent_type
    The environment set up
    Dictionary of agents
    The function to train the agents on
    """

    train_choice = _episode_original_multiple
    agent_type = AgentType.ORIGINAL
    multiple=True
    adj_table =  experiment['graph']
    NUM_OF_AGENTS = experiment['num_agents']
    
        
    
    env = create_env(NUM_OF_AGENTS, NUM_OF_CYCLES, LOCAL_RATIO, multiple)
    # agents = create_agents(NUM_OF_AGENTS, agent_type, num_of_episodes=NUM_OF_EPISODES, length_of_episode=NUM_OF_CYCLES)
    agents = create_exp_marl_agents(NUM_OF_AGENTS, NUM_OF_EPISODES, NUM_OF_CYCLES, 
        experiment['gamma_hop'], adj_table, experiment['connection_slow'])
    return agent_type, env, agents, train_choice


def create_exp_marl_agents(num_of_agents, num_of_episodes, length_of_episode, gamma_hop, adjacency_table, connection_slow):

    """
    Creates the MARL agents

    num_of_agents - The Number of agents to be created

    num_of_episodes - The number of episodes to play

    length_of_episode - The length of the episode

    gamma_hop - The gamma hop distance

    adjacency_table - The graph to be used

    connection_slow - Whether we want the connections to be instantaneous or whether a time delay should be incurred

    """

    agents = {f'agent_{i}': MARL_Comm(f'agent_{i}', num_of_agents, num_of_episodes, length_of_episode, gamma_hop) for i in range(num_of_agents)}

    
    power_graph = convert_adj_to_power_graph(adjacency_table, gamma_hop, connection_slow)
    if dynamic_hyperparameters['dynamic']:
        return agents
    
    print(power_graph)
    for i, row in enumerate(power_graph):
        for j, col in enumerate(row):
            if col != 0:
                agent_obj = agents[f'agent_{i}']
                agent_obj.update_neighbour(f'agent_{j}', col)

    return agents


# def twelve_experiments(experiment, choice):
    
#     NUM_OF_AGENTS = experiment['num_agents']
#             # This is the reward from each episode
#     # reward_array_cumulative = np.array([np.zeros(6) for i in range(NUM_OF_EPISODES)])

#     #     # This is the reward from evaluation runs
#     # reward_list_evaluation = np.array([np.zeros(6) for i in range(NUM_OF_EPISODES//EVALUATION_INTERVAL)])

#     #     # This contains all the evaluation episodes
#     # reward_array_episode_num = np.array([episode_num for episode_num in range(1,NUM_OF_EPISODES+1) if episode_num % EVALUATION_INTERVAL == 0 ])
#         # episodes_array = np.array([i+1 for i in range(NUM_OF_EPISODES)])
#     if NUM_OF_AGENTS == 12:
#         # This is the reward from each episode
#         reward_array_cumulative = np.array([np.zeros(12) for i in range(NUM_OF_EPISODES)])

#         # This is the reward from evaluation runs
#         reward_list_evaluation = np.array([np.zeros(12) for i in range(NUM_OF_EPISODES//EVALUATION_INTERVAL)])

#         # This contains all the evaluation episodes
#         reward_array_episode_num = np.array([episode_num for episode_num in range(1,NUM_OF_EPISODES+1) if episode_num % EVALUATION_INTERVAL == 0 ])
#         # episodes_array = np.array([i+1 for i in range(NUM_OF_EPISODES)])
#     else: 
#         reward_array_cumulative = np.array([np.zeros(4) for i in range(NUM_OF_EPISODES)])
#         reward_list_evaluation = np.array([np.zeros(4) for i in range(NUM_OF_EPISODES//EVALUATION_INTERVAL)])
#         reward_array_episode_num = np.array([episode_num for episode_num in range(1, NUM_OF_EPISODES + 1) if episode_num % EVALUATION_INTERVAL == 0 ])
    
#     # print("The Oracle is created.")
#     # oracle = Oracle() # The Observer is trial persistent and will be used to track the universal n-table
    

#     print(f'TOPOLOGY EXPERIMENTS: Training {NUM_OF_AGENTS} agents of type {choice} over {NUM_OF_EPISODES} episodes with trials {NUMBER_OF_TRIALS}')
#     for trials_num in range(NUMBER_OF_TRIALS):
#         agent_type, env, agents, train_choice = _set_up(experiment) #All we need is a custom set-up function
#         evaluation_pos = 0
        
#         for episode_num in range(1, NUM_OF_EPISODES+1):
#             reward = train_choice(env, agents, episode_num-1)
#             # reward = train_choice(env, agents, episode_num-1, oracle)
#             reward_array_cumulative[episode_num-1] += reward #There is a problem here....
#             if episode_num % 100 == 0:
#                 print(trials_num, episode_num)

#             # Evaluate how good the agents are every EVALUATION_INTERVAL
#             if episode_num % EVALUATION_INTERVAL == 0:
#                 # print(f"Evaluating at episode number {episode_num}")
#                 cumulative_reward = reward_list_evaluation[evaluation_pos]
#                 reward_here = 0
#                 for episode in range(NUM_EVALUATION_EPISODES):
#                     reward = episode_play_normal_marl(env, agents, NUM_OF_CYCLES, NUM_OF_AGENTS, render=False )
#                     reward_here += reward
                    
#                 cumulative_reward += (reward_here/NUM_EVALUATION_EPISODES)
#                 reward_list_evaluation[evaluation_pos] = cumulative_reward
#                 evaluation_pos += 1

#         # oracle.create_bubble_plot()
                
#     # reward_array = reward_array_cumulative/NUMBER_OF_TRIALS
#     reward_array_evaluation = reward_list_evaluation / NUMBER_OF_TRIALS
    
#     print(f"The reward array for {NUM_OF_AGENTS} is {reward_array_evaluation}")

#     return [reward_array_evaluation, reward_array_episode_num]
 
    
def twelve_experiments(experiment, choice):
    NUM_OF_EPISODES = train_hyperparameters['num_of_episodes']
    EVALUATION_INTERVAL = evaluation_hyperparameters['evaluation_interval']
    NUM_EVALUATION_EPISODES = evaluation_hyperparameters['num_evaluation_episodes']
    
    # Initialize an array to track average evaluation rewards
    reward_list_evaluation = np.zeros(NUM_OF_EPISODES // EVALUATION_INTERVAL)
    reward_array_episode_num = np.arange(EVALUATION_INTERVAL, NUM_OF_EPISODES + 1, EVALUATION_INTERVAL)
    
    print(f'TOPOLOGY EXPERIMENTS: Training over {NUM_OF_EPISODES} episodes with trials {NUMBER_OF_TRIALS}')
    for trials_num in range(NUMBER_OF_TRIALS):
        agent_type, env, agents, train_choice = _set_up(experiment)
        for episode_num in range(1, NUM_OF_EPISODES + 1):
            # Training phase
            # It's assumed your training function updates the agents based on interactions with the environment
            train_choice(env, agents, episode_num-1)  # Assuming this function trains the agents

            # Evaluation phase at specified intervals
            if episode_num % EVALUATION_INTERVAL == 0:
                total_evaluation_reward = 0
                for eval_episode in range(NUM_EVALUATION_EPISODES):
                    reward = episode_play_normal_marl(env, agents, NUM_OF_CYCLES, len(agents), render=False)
                    total_evaluation_reward += reward
                
                # Update the average evaluation reward for this evaluation point
                average_evaluation_reward = total_evaluation_reward / NUM_EVALUATION_EPISODES
                reward_list_evaluation[(episode_num // EVALUATION_INTERVAL) - 1] += average_evaluation_reward

            if episode_num % 100 == 0:
                print(f"Trial {trials_num}, Episode {episode_num}")

    # Average the rewards across all trials
    reward_list_evaluation /= NUMBER_OF_TRIALS
    
    print("Average Evaluation Rewards:", reward_list_evaluation)
    return reward_list_evaluation, reward_array_episode_num




def _policy(agent_name, agents, observation, done, time_step, episode_num=0):

    """
    Chooses the action for the agent

    agent_name - The agent names
    
    agents - The dictionary of agents

    observations - What the agent can see

    done - Whether the agent is finished

    time_step - The timestep on

    episode_num=0 - The episode number.  Not used

    returns - The action for the agent to run"""
    
    NUM_OF_AGENTS = len(agents)

    if time_step > NUM_OF_CYCLES:
        return None
    if done:
        return None
    agent = agents[agent_name]
    #print(f'observation: {observation}')
    return agent.policy(encode_state(observation, NUM_OF_AGENTS), time_step)

import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns

def run_experiment(experiment, choice, index, experiment_rewards):
    try:
        print(f"Starting experiment {experiment['experiment_name']}")
        experiment_reward = twelve_experiments(experiment, choice)
        experiment_rewards[index] = experiment_reward
        print(f"Completed experiment {experiment['experiment_name']}")
    except Exception as e:
        print(f"Error in experiment {experiment['experiment_name']}: {e}")


def experiment_pipeline(experiments, choice):
    manager = multiprocessing.Manager()
    experiment_rewards = manager.list([None] * len(experiments))
    processes = []

    # Create and start processes for each experiment
    for i, experiment in enumerate(experiments):
        process = multiprocessing.Process(target=run_experiment, args=(experiment, choice, i, experiment_rewards))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    # Convert managed list back to a regular list for plotting
    experiment_rewards = list(experiment_rewards)
    for i, reward in enumerate(experiment_rewards): 
        if reward is None:
            print(f"Warning: No results for experiment at index {i}")
            continue  # or handle the missing data appropriately


    # Define line styles and colors for different experiments
    # line_styles = ['-', '-', '--', '--', '-.', '-.']
    # colors = ['blue', 'green', 'red', 'orange', 'brown', 'purple']
    # # Plot the results
    # fig, ax = plt.subplots()
    # ax.set_xlabel('Episode Number')
    # ax.set_ylabel('Globally Averaged Reward')
    # ax.grid(True)

        
    # for i, experiment in enumerate(experiments):
        # rewards_array, episode_nums = experiment_rewards[i]
        
        # # Calculate average rewards
        # average_rewards_shaded = np.mean(rewards_array, axis=1)

        # # Smooth the average rewards
        # average_rewards = pd.Series(average_rewards_shaded).rolling(500, min_periods=1).mean()
        
        # line_style = line_styles[i % len(line_styles)]
        # color = colors[i % len(colors)]
        # label = experiment['experiment_name']

        # ax.plot(episode_nums, average_rewards, line_style, label=label, color=color)
        # ax.fill_between(episode_nums, rewards_array.min(axis=1), rewards_array.max(axis=1), alpha=0.2, color=color) 

    # ax.legend()
    # plt.tight_layout()  

    # random_number = random.randint(0, 999999999)
    # filename = f'saved_data/figs/test_time_rewards_{random_number}.png'
    # print(f"Figure saved as {filename}")
    # plt.savefig(filename)
    

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Episode Number')
    ax.set_ylabel('Globally Averaged Reward')
    ax.grid(True)

    # Define line styles and colors for visual distinction between experiments
    line_styles = ['-', '--', '-.', ':']
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']

    # Iterate over both experiment_rewards and experiments to get rewards and names
    for i, ((rewards, episode_nums), experiment) in enumerate(zip(experiment_rewards, experiments)):
        # Ensure rewards is a numpy array for consistency in plotting operations
        rewards = np.array(rewards)

        # Calculate the rolling average for a smooth line
        rolling_avg_rewards = pd.Series(rewards).rolling(window=5000, min_periods=1).mean()
        rolling_std_dev = pd.Series(rewards).rolling(window=5000, min_periods=1).std() #new

        # Setup for plot aesthetics
        line_style = line_styles[i % len(line_styles)]
        color = colors[i % len(colors)]
        label = experiment['experiment_name']  # Extract name from the experiment dict

        # Plot the rolling average line
        ax.plot(episode_nums, rolling_avg_rewards, line_style, label=label, color=color, lw=2)
        
        ax.fill_between(episode_nums, rolling_avg_rewards - rolling_std_dev, rolling_avg_rewards + rolling_std_dev, color=color, alpha=0.2) #new

        # # Create a shaded region around the rolling average
        # ax.fill_between(episode_nums, rolling_avg_rewards, rewards, color=color, alpha=0.2)
        # ax.fill_between(episode_nums, rewards, rolling_avg_rewards, color=color, alpha=0.2)

    ax.legend()
    plt.tight_layout()
    random_number = random.randint(0, 999999999)
    filename = f'saved_data/figs/test_time_rewards_{random_number}.png'
    print(f"Figure saved as {filename}")
    plt.savefig(filename)



import networkx as nx


def plot_graph(adj, num_of_agents, fig_name):
    G = nx.Graph()
    for i in range(num_of_agents):
        for j in range(num_of_agents):
            if adj[i][j] == 1:
                G.add_edge(i, j)

    #Ensure graph is displayed with nodes in numerical order and in a circle
    pos = {}
    for i in range(num_of_agents):
        pos[i] = [np.cos(2*np.pi*i/num_of_agents), np.sin(2*np.pi*i/num_of_agents)]
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    
    random_number = random.randint(0, 999999999)
    filename = f'saved_data/watts_strogatz_figs/{fig_name}_{random_number}.png'
    print(f"Figure saved as {filename}")
    
    plt.savefig(filename)
    plt.clf()  # Clear the current figure after saving it