import numpy as np
from hyperparameters import evaluation_hyperparameters
from train import _set_up
from show import episode_play_normal_marl
import matplotlib.pyplot as plt
from hyperparameters import train_hyperparameters, dynamic_hyperparameters, AgentType
from env import create_env
from utils import encode_state
from reward_functions import final_reward
from adjacency import convert_adj_to_power_graph
from ucb_marl_agent import MARL_Comm
from eb_marl_agent import EB_MARL_Comm
from observer import Oracle
import multiprocessing
import matplotlib.pyplot as plt
import random
import pandas as pd
import networkx as nx

NUMBER_OF_TRIALS = evaluation_hyperparameters['num_of_trials']
NUM_EVALUATION_EPISODES = evaluation_hyperparameters['num_evaluation_episodes']
EVALUATION_INTERVAL = evaluation_hyperparameters['evaluation_interval']
NUM_OF_EPISODES = train_hyperparameters['num_of_episodes']
LOCAL_RATIO = train_hyperparameters['local_ratio']
NUM_OF_CYCLES = train_hyperparameters['num_of_cycles']


def _episode_original_multiple_ucb(env, agents, episode_num, oracle=None):
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


## THIS VERSION IS FOR THE EB LIDARD AKA PB AGENTS ONLY!!!!!
def _episode_original_multiple_peb(env, agents, episode_num, oracle=None):
    NUM_OF_AGENTS = len(agents)
    
    agent_old_state = {agent: -1 for agent in agents.keys()}
    observations = env.reset()    

    t = 0
    while env.agents:
        t = t+1
        actions = {}
        for agent_name in agents.keys():        
            # Take action
            real_state = observations[agent_name]
            encoded_state = encode_state(real_state, NUM_OF_AGENTS)
            agent_old_state[agent_name] = encoded_state

            # Update the real state map in the EB Lidard agent
            agents[agent_name].update_real_state_map(encoded_state, real_state)

            action = _policy(agent_name, agents, real_state, False, t, episode_num)
            actions[agent_name] = action
            
            if oracle is not None: 
                oracle.update(encoded_state, action) # Update the oracle with the state-action pair
                oracle.update_real_state_map(encoded_state, real_state) # Update the oracle with the real state
            

        observations, rewards, terminations, truncations, infos = env.step(actions)
  
        for agent_name in agents.keys():
            # Send messages
            agent_obj = agents[agent_name]
            new_real_state = observations[agent_name]
            new_encoded_state = encode_state(new_real_state, NUM_OF_AGENTS)

            # Fetch the old real state from the agent's real_state_map
            old_real_state = agents[agent_name].real_state_map.get(agent_old_state[agent_name], [0] * len(agent_old_state[agent_name]))

            # Update the real state map with the new state
            agents[agent_name].update_real_state_map(new_encoded_state, new_real_state)

            # Updated message passing with real states
            agent_obj.message_passing(episode_num, t, agent_old_state[agent_name], old_real_state, actions[agent_name],
                                    new_encoded_state, new_real_state, rewards[agent_name], agents)

        for agent_name in agents.keys():
            # Update u and v
            agent_obj = agents[agent_name]
            new_real_state = observations[agent_name]
            new_encoded_state = encode_state(new_real_state, NUM_OF_AGENTS)

            # Update the real state map with the new state
            agents[agent_name].update_real_state_map(new_encoded_state, new_real_state)

            agent_obj.update(episode_num, t, agent_old_state[agent_name], new_encoded_state,
                             actions[agent_name], rewards[agent_name])
            
        for agent_name in agents.keys():
            # Update the values
            agent_obj = agents[agent_name]
            agent_obj.update_values(episode_num, t)
    
    return final_reward(rewards)

def _set_up(experiment):

    """
    Sets up the environments and agents

    
    """

    agent_type = experiment['agent_type'] 
    
    if agent_type == AgentType.EB_Lidard:
        train_choice = _episode_original_multiple_peb
    else:
        train_choice = _episode_original_multiple_ucb
    

    
    multiple=True
    adj_table =  experiment['graph']
    NUM_OF_AGENTS = experiment['num_agents']
    
        
    
    env = create_env(NUM_OF_AGENTS, NUM_OF_CYCLES, LOCAL_RATIO, multiple)
    agents = create_exp_marl_agents(NUM_OF_AGENTS, NUM_OF_EPISODES, NUM_OF_CYCLES, 
        experiment['gamma_hop'], adj_table, experiment['connection_slow'], agent_type)
    return agent_type, env, agents, train_choice


def create_exp_marl_agents(num_of_agents, num_of_episodes, length_of_episode, gamma_hop, adjacency_table, connection_slow, agent_type):

    """
    Creates the MARL agents

    num_of_agents - The Number of agents to be created

    num_of_episodes - The number of episodes to play

    length_of_episode - The length of the episode

    gamma_hop - The gamma hop distance

    adjacency_table - The graph to be used

    connection_slow - Whether we want the connections to be instantaneous or whether a time delay should be incurred

    """ 
    if agent_type == AgentType.ORIGINAL:
        # print(f"Using {num_of_agents} UCB agents.")
        agents = {f'agent_{i}': MARL_Comm(f'agent_{i}', num_of_agents, num_of_episodes, length_of_episode, gamma_hop) for i in range(num_of_agents)}
    else:
        # print(f"Using {num_of_agents} PEB agents.")
        agents = {f'agent_{i}': EB_MARL_Comm(f'agent_{i}', num_of_agents, num_of_episodes, length_of_episode, gamma_hop) for i in range(num_of_agents)}

    
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



def twelve_experiments(experiment, choice=None):
    NUM_OF_EPISODES = train_hyperparameters['num_of_episodes']
    EVALUATION_INTERVAL = evaluation_hyperparameters['evaluation_interval']
    NUM_EVALUATION_EPISODES = evaluation_hyperparameters['num_evaluation_episodes']
    
    # Initialize arrays to track shit
    reward_list_evaluation = np.zeros(NUM_OF_EPISODES // EVALUATION_INTERVAL)
    bes_scores = np.zeros(NUM_OF_EPISODES // EVALUATION_INTERVAL)
    reward_array_episode_num = np.arange(EVALUATION_INTERVAL, NUM_OF_EPISODES + 1, EVALUATION_INTERVAL)
    
    min_rewards = np.full(NUM_OF_EPISODES // EVALUATION_INTERVAL, np.inf)  # Initialize with infinities for minimums
    max_rewards = np.full(NUM_OF_EPISODES // EVALUATION_INTERVAL, -np.inf)  # Initialize with -infinities for maximums
    
    percentile_25_rewards = np.full(NUM_OF_EPISODES // EVALUATION_INTERVAL, np.inf)
    percentile_75_rewards = np.full(NUM_OF_EPISODES // EVALUATION_INTERVAL, -np.inf)
    
    oracle = Oracle() #Always create the oracle!
    print("The Observer is created.")
    
    
    print(f"TOPOLOGY EXPERIMENTS: EXPERIMENT {experiment['experiment_name']} training over {NUM_OF_EPISODES} episodes with trials {NUMBER_OF_TRIALS} with {experiment['num_agents']} agents of the type {experiment['agent_type']}\n")
    for trials_num in range(NUMBER_OF_TRIALS):
        agent_type, env, agents, train_choice = _set_up(experiment)
        for episode_num in range(1, NUM_OF_EPISODES + 1):
            # Training phase
            train_choice(env, agents, episode_num - 1, oracle)  
            # Evaluation phase at specified intervals
            if episode_num % EVALUATION_INTERVAL == 0:
                episode_rewards = []  # List to collect rewards for this evaluation
                
                for eval_episode in range(NUM_EVALUATION_EPISODES): #Evaluation/Test time!
                    reward = episode_play_normal_marl(env, agents, NUM_OF_CYCLES, len(agents), render=False)
                    episode_rewards.append(reward)  #  total reward per episode
                
                # Calculate and update average, min, and max rewards for this evaluation point
                average_evaluation_reward = np.mean(episode_rewards)
                reward_list_evaluation[(episode_num // EVALUATION_INTERVAL) - 1] += average_evaluation_reward
                min_rewards[(episode_num // EVALUATION_INTERVAL) - 1] = min(min(episode_rewards), min_rewards[(episode_num // EVALUATION_INTERVAL) - 1])
                max_rewards[(episode_num // EVALUATION_INTERVAL) - 1] = max(max(episode_rewards), max_rewards[(episode_num // EVALUATION_INTERVAL) - 1])
                # After calculating and updating average, min, and max rewards, calculate percentiles
                percentile_25 = np.percentile(episode_rewards, 25)
                percentile_75 = np.percentile(episode_rewards, 75)
                
                # Update the percentile arrays
                index = (episode_num // EVALUATION_INTERVAL) - 1  # Calculate the index for the current evaluation interval
                percentile_25_rewards[index] = percentile_25
                percentile_75_rewards[index] = percentile_75
                
                # Compute the bad exploration score after every training episode so we can take the average later. 
                bes = oracle.calculate_bad_exp_score(average_evaluation_reward)
                bes_scores[(episode_num // EVALUATION_INTERVAL) - 1] += bes
                
            if episode_num % 100 == 0:
                print(f"Trial {trials_num}, Episode {episode_num}")
      

    # Take the average universal N table across 5 trials. 
    for outer_key, inner_dict in oracle.universal_nTable.items():
        for inner_key in inner_dict:
            oracle.universal_nTable[outer_key][inner_key] /= NUMBER_OF_TRIALS
  
  # Compute the metrics we want to compute from the Observer's universal N table.
    stats = oracle.calculate_statistics()         
    if len(agents) == 4 or len(agents) == 8:
        oracle.create_bubble_plot(len(agents)) # Plots with more than 8 agents are cluttered. 
    print("Statistics: ", stats)
    last_mean_reward = reward_list_evaluation[-1] / len(agents)
    bes = oracle.calculate_bad_exp_score(last_mean_reward)
    print("Final Bad Exploration Score:", bes)
    cc, G = oracle.calculate_clustering_coefficient()
    print("Clustering Coefficient:", cc)

    # Average the rewards across all trials
    reward_list_evaluation /= NUMBER_OF_TRIALS
    
    average_bes = np.mean(bes_scores)
    print(f"Average BES: {average_bes}")
    
    
    return reward_list_evaluation, min_rewards, max_rewards, percentile_25_rewards, percentile_75_rewards, reward_array_episode_num, experiment





def run_experiment(experiment, choice, index, experiment_rewards):
    try:
        print(f"Starting experiment {experiment['experiment_name']}")
        experiment_reward = twelve_experiments(experiment)
        experiment_rewards[index] = experiment_reward
        print(f"Completed experiment {experiment['experiment_name']}")
    except Exception as e:
        print(f"Error in experiment {experiment['experiment_name']}: {e}")


def experiment_pipeline(experiments, choice=None):
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


    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Episode Number')
    ax.set_ylabel('Mean Reward')
    ax.grid(True)

    # Define line styles and colors for visual distinction between experiments
    # line_styles = ['-', '-', '--', '-.', '-.'] # Experiment 1
    # line_styles = ['-', '-', '--', '--'] # Experiment 2, 2.5, 3
    # line_styles = ['-', '-', '--', '--', '-.'] # Experiment 3.5
    line_styles = ['-', '-.'] # Experiment 2, 2.5, 3
    
    
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    
    
    
    # # Min max chart
    # for i, (average_rewards, min_rewards, max_rewards, percentile_25_rewards, percentile_75_rewards, episode_nums, experiment) in enumerate(experiment_rewards):
    
    # # for i, (average_rewards, min_rewards, max_rewards, episode_nums, experiment) in enumerate(experiment_rewards):
    #     average_rewards = np.array(average_rewards)
    #     min_rewards = np.array(min_rewards)
    #     max_rewards = np.array(max_rewards)

    #     rolling_avg_rewards = pd.Series(average_rewards).rolling(window=50, min_periods=1).mean()

    #     line_style = line_styles[i % len(line_styles)]
    #     color = colors[i % len(colors)]
    #     label = experiment['experiment_name']  # Extract name from the experiment dict
        

    #     ax.plot(episode_nums, rolling_avg_rewards, line_style, label=label, color=color, lw=2)
    #     ax.fill_between(episode_nums, min_rewards, max_rewards, color=color, alpha=.2)



    # ax.legend()
    # plt.tight_layout()
    # random_number = random.randint(0, 999999999)
    # filename = f'saved_data/figs/thesis/test_time_rewards_MINMAX{random_number}.png'
    # print(f"Figure saved as {filename}")
    # plt.savefig(filename)
    # plt.clf()

    # fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure and axes
    # ax.set_xlabel('Episode Number')
    # ax.set_ylabel('Mean Reward')
    # ax.grid(True)
    
    # for i, (average_rewards, min_rewards, max_rewards, percentile_25_rewards, percentile_75_rewards, episode_nums, experiment) in enumerate(experiment_rewards):
    #     average_rewards = np.array(average_rewards)
    #     quad1_rewards = np.array(percentile_25_rewards)
    #     quad2_rewards = np.array(percentile_75_rewards)

    #     rolling_avg_rewards = pd.Series(average_rewards).rolling(window=50, min_periods=1).mean()

    #     line_style = line_styles[i % len(line_styles)]
    #     color = colors[i % len(colors)]
    #     label = experiment['experiment_name']  # Extract name from the experiment dict
        

    #     ax.plot(episode_nums, rolling_avg_rewards, line_style, label=label, color=color, lw=2)
    #     ax.fill_between(episode_nums, quad1_rewards, quad2_rewards, color=color, alpha=.2)


    # ax.legend()
    # plt.tight_layout()
    # random_number = random.randint(0, 999999999)
    # filename = f'saved_data/figs/thesis/test_time_rewards_PERCENTILE{random_number}.png'
    # print(f"Figure saved as {filename}")
    # plt.savefig(filename)
    # plt.clf()

    # fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure and axes
    # ax.set_xlabel('Episode Number')
    # ax.set_ylabel('Mean Reward')
    # ax.grid(True)


    # Iterate over both experiment_rewards and experiments to get rewards and names
    for i, (average_rewards, min_rewards, max_rewards, percentile_25_rewards, percentile_75_rewards, episode_nums, experiment) in enumerate(experiment_rewards):
        # Ensure rewards is a numpy array for consistency in plotting operations
        rewards = np.array(average_rewards)

        # Calculate the rolling average for a smooth line
        rolling_avg_rewards = pd.Series(rewards).rolling(window=50, min_periods=1).mean()
        rolling_std_dev = pd.Series(rewards).rolling(window=50, min_periods=1).std() #new

        # Setup for plot aesthetics
        line_style = line_styles[i % len(line_styles)]
        color = colors[i % len(colors)]
        label = experiment['experiment_name']  # Extract name from the experiment dict

        # Plot the rolling average line
        ax.plot(episode_nums, rolling_avg_rewards, line_style, label=label, color=color, lw=2)
        
        ax.fill_between(episode_nums, rolling_avg_rewards - rolling_std_dev, rolling_avg_rewards + rolling_std_dev, color=color, alpha=.2) #new


    ax.legend()
    plt.tight_layout()
    random_number = random.randint(0, 999999999)
    filename = f'saved_data/figs/thesis/test_time_rewards_STDEV_{label}_{random_number}.png'
    print(f"Figure saved as {filename}")
    plt.savefig(filename)
    plt.clf()
    
    
 

#     # Plot the results
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.set_xlabel('Episode Number')
#     ax.set_ylabel('Mean Reward')
#     ax.grid(True)

#     # Define line styles and colors for visual distinction between experiments
#     # line_styles = ['-', '-', '--', '-.', '-.'] # Experiment 1
#     line_styles = ['-', '-', '--', '--'] # Experiment 2, 2.5, 3
#     # line_styles = ['-', '-', '--', '--', '-.'] # Experiment 3.5
    
#     colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    
    
#     # Min max chart
#     for i, (average_rewards, min_rewards, max_rewards, percentile_25_rewards, percentile_75_rewards, episode_nums, experiment) in enumerate(experiment_rewards):
    
#     # for i, (average_rewards, min_rewards, max_rewards, episode_nums, experiment) in enumerate(experiment_rewards):
#         average_rewards = np.array(average_rewards)
#         min_rewards = np.array(min_rewards)
#         max_rewards = np.array(max_rewards)

#         rolling_avg_rewards = pd.Series(average_rewards).rolling(window=50, min_periods=1).mean()

#         line_style = line_styles[i % len(line_styles)]
#         color = colors[i % len(colors)]
#         label = experiment['experiment_name']  # Extract name from the experiment dict
        

#         ax.plot(episode_nums, rolling_avg_rewards, line_style, label=label, color=color, lw=2)
#         ax.fill_between(episode_nums, min_rewards, max_rewards, color=color, alpha=.7)



#     ax.legend()
#     plt.tight_layout()
#     random_number = random.randint(0, 999999999)
#     filename = f'saved_data/figs/thesis/test_time_rewards_MINMAXDARK{random_number}.png'
#     print(f"Figure saved as {filename}")
#     plt.savefig(filename)
#     plt.clf()

#     fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure and axes
#     ax.set_xlabel('Episode Number')
#     ax.set_ylabel('Mean Reward')
#     ax.grid(True)
    
#     for i, (average_rewards, min_rewards, max_rewards, percentile_25_rewards, percentile_75_rewards, episode_nums, experiment) in enumerate(experiment_rewards):
#         average_rewards = np.array(average_rewards)
#         quad1_rewards = np.array(percentile_25_rewards)
#         quad2_rewards = np.array(percentile_75_rewards)

#         rolling_avg_rewards = pd.Series(average_rewards).rolling(window=50, min_periods=1).mean()

#         line_style = line_styles[i % len(line_styles)]
#         color = colors[i % len(colors)]
#         label = experiment['experiment_name']  # Extract name from the experiment dict
        

#         ax.plot(episode_nums, rolling_avg_rewards, line_style, label=label, color=color, lw=2)
#         ax.fill_between(episode_nums, quad1_rewards, quad2_rewards, color=color, alpha=.7)


#     ax.legend()
#     plt.tight_layout()
#     random_number = random.randint(0, 999999999)
#     filename = f'saved_data/figs/thesis/test_time_rewards_PERCENTILEDARK{random_number}.png'
#     print(f"Figure saved as {filename}")
#     plt.savefig(filename)
#     plt.clf()

#     fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure and axes
#     ax.set_xlabel('Episode Number')
#     ax.set_ylabel('Mean Reward')
#     ax.grid(True)


#     # Iterate over both experiment_rewards and experiments to get rewards and names
#     for i, (average_rewards, min_rewards, max_rewards, percentile_25_rewards, percentile_75_rewards, episode_nums, experiment) in enumerate(experiment_rewards):
#         # Ensure rewards is a numpy array for consistency in plotting operations
#         rewards = np.array(average_rewards)

#         # Calculate the rolling average for a smooth line
#         rolling_avg_rewards = pd.Series(rewards).rolling(window=50, min_periods=1).mean()
#         rolling_std_dev = pd.Series(rewards).rolling(window=50, min_periods=1).std() #new

#         # Setup for plot aesthetics
#         line_style = line_styles[i % len(line_styles)]
#         color = colors[i % len(colors)]
#         label = experiment['experiment_name']  # Extract name from the experiment dict

#         # Plot the rolling average line
#         ax.plot(episode_nums, rolling_avg_rewards, line_style, label=label, color=color, lw=2)
        
#         ax.fill_between(episode_nums, rolling_avg_rewards - rolling_std_dev, rolling_avg_rewards + rolling_std_dev, color=color, alpha=.7) #new


#     ax.legend()
#     plt.tight_layout()
#     random_number = random.randint(0, 999999999)
#     filename = f'saved_data/figs/thesis/test_time_rewards_STDEVDARK{random_number}.png'
#     print(f"Figure saved as {filename}")
#     plt.savefig(filename)
#     plt.clf()
    





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

