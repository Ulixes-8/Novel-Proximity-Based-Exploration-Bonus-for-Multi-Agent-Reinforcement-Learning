from hyperparameters import reward_function
import numpy as np

## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 
## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 
## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 
## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 
## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 
## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 
## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 
## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 
## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 
## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 
## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 
## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 
## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 
## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 
## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 
## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 
## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 
## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 
## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 
## THIS IS LEGACY CODE. IF YOU ARE USING THE EXPERIMENT PIPELINE, YOU CAN SAFELY IGNORE THIS FILE. IT IS NOT USED IN THE PIPELINE. 


def final_reward(agent_old_reward):

    """
    This is the actual reward function to be used

    agents_old_reward - A dictionary of rewards with keys being agent names

    Returns 
    The reward 
    """
    if reward_function['reward'] == 'mean':
        return _mean_reward(agent_old_reward)

    if reward_function['reward'] == 'split':
        return switch_reward(agent_old_reward)
    
    if reward_function['reward'] == 'split_all':
        return switch_all_reward(agent_old_reward)
    
    if reward_function['reward'] == 'split_all_7&8':
        return switch_all_reward_and_extra(agent_old_reward)
    

def _mean_reward(agent_old_reward):

    """
    This will be the reward function for to be used for the average reward over all agents
    
    agents_old_reward - A dictionary of rewards with keys being agent names

    Returns
    The mean reward
    """

    total_reward = 0
    for reward_per_agent in agent_old_reward.values():
        total_reward += reward_per_agent
        
    return total_reward/len(agent_old_reward.keys())


def switch_reward(agent_old_reward):

    """
    This will average out the reward for agents 0 and 3 (A & D).

    agents_old_reward - A dictionary of rewards with keys being agent names

    Returns
    The mean reward of agents 0 and 3

    """

    agent_one = agent_old_reward['agent_0']
    agent_four = agent_old_reward['agent_3']

    return (agent_four+agent_one)/2


def switch_all_reward_and_extra(agent_old_reward):

    """
    This will create the reward array which has the different average rewards 
    
    agents_old_reward - A dictionary of rewards with keys being agent names
    
    Return  
    An array with average rewards like [0&6, 1&4, 2&10, 7&8, 3&5, 9&11]
    """

    agent_zero = agent_old_reward['agent_0']
    agent_six = agent_old_reward['agent_6']
    first_switch = (agent_zero+agent_six)/2

    agent_one = agent_old_reward['agent_1']
    agent_four = agent_old_reward['agent_4']
    second_switch = (agent_one+agent_four)/2

    agent_two = agent_old_reward['agent_2']
    agent_ten = agent_old_reward['agent_10']
    third_switch = (agent_two+agent_ten)/2

    agent_seven = agent_old_reward['agent_7']
    agent_eight = agent_old_reward['agent_8']
    fourth_switch = (agent_seven+agent_eight)/2

    agent_three = agent_old_reward['agent_3']
    agent_five = agent_old_reward['agent_5']
    fifth_switch = (agent_three+agent_five)/2

    agent_nine = agent_old_reward['agent_9']
    agent_eleven = agent_old_reward['agent_11']
    sixth_switch = (agent_nine+agent_eleven)/2

    return np.array([first_switch, second_switch, third_switch, fourth_switch, fifth_switch, sixth_switch, agent_seven, agent_eight])


def switch_all_reward(agent_old_reward):

    """
    This will create the reward array which has the different average rewards 
    
    agents_old_reward - A dictionary of rewards with keys being agent names
    
    Return  
    An array with average rewards like [0&6, 1&4, 2&10, 7&8, 3&5, 9&11]
    """

    agent_zero = agent_old_reward['agent_0']
    agent_six = agent_old_reward['agent_6']
    first_switch = (agent_zero+agent_six)/2

    agent_one = agent_old_reward['agent_1']
    agent_four = agent_old_reward['agent_4']
    second_switch = (agent_one+agent_four)/2

    agent_two = agent_old_reward['agent_2']
    agent_ten = agent_old_reward['agent_10']
    third_switch = (agent_two+agent_ten)/2

    agent_seven = agent_old_reward['agent_7']
    agent_eight = agent_old_reward['agent_8']
    fourth_switch = (agent_seven+agent_eight)/2

    agent_three = agent_old_reward['agent_3']
    agent_five = agent_old_reward['agent_5']
    fifth_switch = (agent_three+agent_five)/2

    agent_nine = agent_old_reward['agent_9']
    agent_eleven = agent_old_reward['agent_11']
    sixth_switch = (agent_nine+agent_eleven)/2

    return np.array([first_switch, second_switch, third_switch, fourth_switch, fifth_switch, sixth_switch])