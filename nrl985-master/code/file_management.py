import dill
import os
import numpy as np
import matplotlib.pyplot as plt

DIR_NAME = 'saved_data'

def filename_creator(agent_type, num_of_agents, num_of_cycles, num_of_episodes, local_ratio):

    """ The filename to be created.  The folder trained_agents is assumed to contain all the currently saved data.
    If you have saved the reward data but not the agent then rerun the code with the same agent you will rewrite the reward data at the end

    agent_type - The type of agents to be saved

    num_of_agents - The number of agents to be saved

    num_of_cycles - The number of frames which the agents have been trained on

    num_of_episodes - The number of episodes the agents have been trained on

    local_ratio - The local ratio the agents have been saved on
    
    Return
    The filename to be used
    """

    file_name_first_half = f'{agent_type}_{num_of_agents}_{num_of_cycles}_{num_of_episodes}_{local_ratio}'
    max = -1
    # os.chdir('/Users/lleoardostella/Documents/MARL')
    for file in os.listdir(f'{DIR_NAME}/trained_agents'):
        final_underscore = file.rfind('_')
        if file[:final_underscore] == file_name_first_half:
            final_dot = file.rfind('.')
            num = int(file[final_underscore+1:final_dot])
            if num > max:
                max = num
    filename = file_name_first_half+ f'_{max+1}'

    return filename


def save_fig(episodes, reward, filename, title):

    """
    This takes in 2 arrays and then creates a plot of it.  
    
    episodes - An array which contains the episode number.  The x axis
    
    reward - An array which contains the reward per episode.  The y axis
    
    filename - The filename to be used
    
    title - The title of the graph
    """

    plt.figure()
    plt.plot(episodes, reward)
    plt.title(title)
    plt.ylabel("Reward")
    plt.xlabel("Episode Number")

    plt.savefig(f'{DIR_NAME}/figs/{filename}.png')
    print(f'Saved fig as {filename}.png')



def save_agents(agents, filename):
    
    """
    This saves the agent onto local machine.  The filename is saved in form agent_type_num_of_agents_num_of_cycles_num_of_episodes_local_ratio_copies.pkl

    agents - The dictionary of agents to be saved

    filename - The filename
    """

    with open(f'{DIR_NAME}/trained_agents/'+filename+'.pkl', 'wb') as output:
        dill.dump(agents, output, dill.HIGHEST_PROTOCOL)

    print('\n')
    print(f'Saved file as {filename}.pkl')
    

def save_rewards(episodes, reward, filename):
    """
    This takes in 2 arrays and then saves onto the local machine if its needed for later plots
    
    episodes - An array which contains the episode number.  The x axis
    
    reward - An array which contains the reward per episode.  The y axis
    
    filename - The filename to be used
    """

    dict_to_save = {'episodes': episodes, 'reward': reward}

    with open(f'{DIR_NAME}/training_rewards/'+filename+'.pkl', 'wb') as output:
        dill.dump(dict_to_save, output, dill.HIGHEST_PROTOCOL)

    print(f'Saved the rewards dicts as {filename}.pkl')

def save_bonuses(episodes, bonuses, filename):
    """
    This takes in 2 arrays and then saves onto the local machine if it's needed for later plots
    
    episodes - An array which contains the episode number. The x-axis.
    
    bonuses - An array which contains the exploration bonus per episode. The y-axis.
    
    filename - The filename to be used.
    """

    dict_to_save = {'episodes': episodes, 'bonuses': bonuses}

    with open(f'{DIR_NAME}/exploration_bonuses/'+filename+'.pkl', 'wb') as output:
        dill.dump(dict_to_save, output, dill.HIGHEST_PROTOCOL)

    print(f'Saved the bonuses dict as {filename}.pkl')

def save_eb_fig(episodes, bonuses, filename, title):
    """
    This takes in an array of episodes and bonuses, then creates and saves a plot of the exploration bonus over episodes.
    
    episodes - An array which contains the episode number. The x-axis.
    
    bonuses - An array which contains the exploration bonus per episode. The y-axis.
    
    filename - The filename to be used for saving the figure.
    
    title - The title of the graph.
    """
    plt.figure()
    plt.plot(episodes, bonuses, label='Exploration Bonus')
    plt.title(title)
    plt.ylabel("Exploration Bonus")
    plt.xlabel("Episode Number")
    plt.legend()

    plt.savefig(f'{DIR_NAME}/figs/{filename}_eb.png')
    print(f'Saved exploration bonus fig as {filename}_eb.png')

def save_eb_reward_fig(episodes, rewards, bonuses, filename, title):
    """
    This takes in an array of episodes, rewards, and bonuses, then creates and saves a plot with both 
    rewards and exploration bonus values over episodes.
    
    episodes - An array which contains the episode number. The x-axis.
    
    rewards - An array which contains the reward per episode.
    
    bonuses - An array which contains the exploration bonus per episode.
    
    filename - The filename to be used for saving the figure.
    
    title - The title of the graph.
    """
    plt.figure()
    plt.plot(episodes, rewards, label='Reward')
    plt.plot(episodes, bonuses, label='Exploration Bonus', linestyle='--')
    plt.title(title)
    plt.ylabel("Value")
    plt.xlabel("Episode Number")
    plt.legend()

    plt.savefig(f'{DIR_NAME}/figs/{filename}_eb_r.png')
    print(f'Saved reward and exploration bonus fig as {filename}_eb_r.png')
    
    
def save_detailed_eb_fig(mean_bonuses, filename, title):
    plt.figure()
    
    # mean_bonuses is now a list of mean bonuses per timestep
    timesteps = np.arange(1, len(mean_bonuses) + 1)
    plt.plot(timesteps, mean_bonuses, label='Mean Exploration Bonus')
    
    plt.title(title)
    plt.ylabel("Mean Exploration Bonus")
    plt.xlabel("Timestep")
    plt.legend()
    
    plt.savefig(f'{DIR_NAME}/figs/{filename}_EpisodeEB.png')
    print(f'Saved detailed exploration bonus fig as {filename}_EpisodeEB.png')



def save(agents, episodes, reward, agent_type, num_of_agents, num_of_cycles, num_of_episodes, local_ratio, bonuses=None, detailed_exploration_bonus=None):
    """ 
    Saves the agents and a fig of reward onto the local machine. If bonuses are provided, also saves a fig of the exploration bonus and a combined fig.

    agents - The dictionary of agents to be saved.
    
    episodes - The episode numbers which were run.

    reward - The rewards array to be saved - This can be a list of 3 elements.  
    Element 0 is the rewards array, element 1 is the evaluation rewards array, and element 2 is the evaluation episodes array.

    agent_type - The agent type which has been trained.

    num_of_agents - The number of agents to be saved.

    num_of_cycles - The number of frames which the agents have been trained on.

    num_of_episodes - The number of episodes the agents have been trained on.

    local_ratio - The local ratio the agents have been saved on.

    bonuses - An optional array containing the exploration bonuses per episode.
    """

    filename = filename_creator(agent_type, num_of_agents, num_of_cycles, num_of_episodes, local_ratio)

    print('Saving Agents')
    save_agents(agents, filename)  # COMMENT OUT IF YOU DON'T WANT TO SAVE AGENTS!!!

    print('Saving Reward Figures')
    # Check if we are saving the evaluation data as well
    if isinstance(reward, list):
        save_fig(episodes, reward[0], filename, 'The Average Reward per Episode')
        save_fig(reward[2], reward[1], f'{filename}_evaluation', 'The Evaluation Reward per Episode')
        if bonuses is not None:
            save_eb_fig(episodes, bonuses, filename, 'The Exploration Bonus per Episode')
            save_eb_reward_fig(episodes, reward[0], bonuses, filename, 'The Average Reward and Exploration Bonus per Episode')
    else:
        save_fig(episodes, reward, filename, 'The Average Reward per Episode')
        if bonuses is not None:
            save_eb_fig(episodes, bonuses, filename, 'The Exploration Bonus per Episode')
            save_eb_reward_fig(episodes, reward, bonuses, filename, 'The Average Reward and Exploration Bonus per Episode')

    # Handle detailed exploration bonus
    # if detailed_exploration_bonus is not None:

    #     print('Saving Detailed Exploration Bonus Figures')
    #     for episode_num, bonuses in detailed_exploration_bonus.items():
    #         title = f'Exploration Bonus for Episode {episode_num}'
    #         filename = f'{agent_type}_{num_of_agents}_{num_of_cycles}_{num_of_episodes}_{local_ratio}_eb_{episode_num}'
    #         save_detailed_eb_fig(bonuses, filename, title)
            
    print('Saving Rewards Array')
    save_rewards(episodes, reward, filename)
    
    if bonuses is not None:
        print('Saving Bonuses Array')
        save_bonuses(episodes, bonuses, filename)



# def save(agents, episodes, reward, agent_type, num_of_agents, num_of_cycles, num_of_episodes, local_ratio):

#     """ 
#     Saves the agents and a fig of reward onto the local 

#     agents - The dictionary of agents to be saved
    
#     episodes - The episode which were run

#     reward - The rewards array to be saved - This can be a list of 3 elements.  
#     Element 1 is the rewards array, element 2 is the evaluation rewards array and element 3 is the evaluation episodes array

#     agent_type - The agent type which has been trained

#     num_of_agents - The number of agents to be saved

#     num_of_cycles - The number of frames which the agents have been trained on

#     num_of_episodes - The number of episodes the agents have been trained on

#     local_ratio - The local ratio the agents have been saved on
#     """

#     filename = filename_creator(agent_type, num_of_agents, num_of_cycles, num_of_episodes, local_ratio)

#     print('Saving Agents')
#     save_agents(agents, filename)       # COMMENT OUT IF YOU DON'T WANT TO SAVE AGENTS!!!

#     print('Saving Fig')
#     # Checks if we are saving the evaluation data as well
#     if type(reward) == list:
#         save_fig(episodes, reward[0], filename, 'The Average Reward per Episode')
#         save_fig(reward[2], reward[1], f'{filename}_evaluation', 'The Evaluation Reward per Episode')
#     else:
#         save_fig(episodes, reward, filename, 'The Average Reward per Episode')

#     print('Saving rewards array')
#     save_rewards(episodes, reward, filename)


def load(filename):

    """
    Loads the pickled file

    filename - The file to be loaded

    returns - Dict of agents which was in the pickled file"""

    print('Loading Agents')
    return [agent for agent in _pickle_loader(f'{DIR_NAME}/trained_agents/'+filename)][0]


def show_all_files():

    """Returns the list of trained agents filenames"""
    
    return os.listdir(f'{DIR_NAME}/trained_agents')


def _pickle_loader(filename):

    """
    This loads a pickle file
    
    filename - The filename
    
    yields - The object which was saved
    """
    
    with open(filename, 'rb') as f:
        while True:
            try:
                yield dill.load(f)
            except EOFError:
                break
