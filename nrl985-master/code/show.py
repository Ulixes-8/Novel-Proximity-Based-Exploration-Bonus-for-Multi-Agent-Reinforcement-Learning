from file_management import load
from env import create_env
from utils import encode_state
from reward_functions import final_reward
from hyperparameters import switch_hyperparameters

from time import sleep

def _find_all(string, char):

    '''
    Returns all the indexes where a character is in the string

    string - The string to look through

    char - The character to find
    
    Returns
    A list of indexes of where the character is in the string
    '''
    return [index for index in range(len(string)) if string[index]==char]


def play_on_show(filename):

    """
    This will allow the agents to play with it being seen on the screen
    
    filename - The filename which has the pickled agents
    """

    
    # agent_type = AgentType.EB_Lidard
    # agent_type = AgentType.ORIGINAL
    num_of_agents = 4
    num_of_cycles = 10
    local_ratio = 0

    # agent_type_with_agent = filename[:indexes[0]]
    # indexes = _find_all(filename, '_')    
    # agent_type = AgentType(agent_type_with_agent[agent_type_with_agent.find('.')+1:indexes[0]])
    # num_of_cycles = int(filename[indexes[1]+1:indexes[2]])
    # num_of_agents = int(filename[indexes[0]+1:indexes[1]])
    # local_ratio = float(filename[indexes[3]+1:indexes[4]])
    

    agents = load(filename)
    env = create_env(num_of_agents, num_of_cycles, local_ratio, multiple=True, render_mode='human')
    print(episode_play_normal_marl(env, agents, num_of_cycles, num_of_agents))

    for i in range(10000):
        env.render()


def episode_play_normal_marl(env, agents, max_cycles, num_of_agents, render=True):

    """
    This plays a single episode of in the Parallel MDP environment.

    env - The environment to be used - A parallel env

    agents - A dictionary of agents to be used

    max_cycles - The length of an episode

    num_of_agents - The number of agents being played

    render - Whether it is being rendered onto a screen for a human.

    Return
    The reward for the episode
    """

    agent_old_state = {agent: -1 for agent in agents.keys()}
    # This will make sure the agents are set in either the switch position or not
    observations = env.reset(options=switch_hyperparameters['switch'])
    t = 0
    while env.agents:
        t = t+1
        actions = {}
        for agent_name in agents.keys():        # Take action
            agent_old_state[agent_name] = encode_state(observations[agent_name], num_of_agents)
            action = _policy(agent_name, agents, observations[agent_name], False, t, max_cycles, render=render)
            actions[agent_name] = action
        if render:
            sleep(0.05)
        observations, rewards, terminations, truncations, infos = env.step(actions)
        if render:
            sleep(0.1)

    return final_reward(rewards)
    

def _policy(agent_name, agents, observations, done, num_of_cycles_done, num_of_cycles_max, render=True):

    """
    This will find and play the correct action for the agent

    agent_name - The agent whose move it is

    agents - The dictionary of agents

    observations - What the agent can observe

    done - Whether the agent has finished

    num_of_cycles_done - The number of cycles which have been played
    
    num_of_cycles_max - The maximum number of cycles which can be played

    render - Whether rendering onto the screen

    Return 
    The action to be taken
    """

    if num_of_cycles_done > num_of_cycles_max:
        return None
    if done:
        return None
    agent = agents[agent_name]
    return agent.play_normal(encode_state(observations, len(agents.keys())), num_of_cycles_done, render)
