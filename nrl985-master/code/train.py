"""Contains the main algorithm which can train
the model"""
from env import create_env
from create_agents import create_agents
from utils import encode_state
from reward_functions import final_reward
from hyperparameters import train_hyperparameters, agent_hyperparameters, dynamic_hyperparameters, AgentType
import math

# Hyperparameters which can be changed to change what and how the agent learns
NUM_OF_CYCLES = train_hyperparameters['num_of_cycles']
NUM_OF_AGENTS = agent_hyperparameters['num_of_agents']
NUM_OF_EPISODES = train_hyperparameters['num_of_episodes']
LOCAL_RATIO = train_hyperparameters['local_ratio']


def _set_up(choice):

    """
    Sets up the environments and agents

    choice - The type of Agent to use

    Return
    The agent_type
    The environment set up
    Dictionary of agents
    The function to train the agents on
    """

    if choice == AgentType.RANDOM:
        agent_type = AgentType.RANDOM
        train_choice = _episode_random
        multiple=True
    elif choice == AgentType.IQL:
        agent_type = AgentType.IQL
        train_choice = _episode_IQL
        multiple=True
    elif choice == AgentType.ORIGINAL:
        if dynamic_hyperparameters['dynamic']:
            train_choice = _episode_dynamic_graph
        else:
            train_choice = _episode_original_multiple
        agent_type = AgentType.ORIGINAL
        multiple=True
    elif choice == AgentType.EB_Lidard:
        if dynamic_hyperparameters['dynamic']:
            train_choice = _episode_dynamic_graph
        else:
            train_choice = _episode_EB_Lidard
        agent_type = AgentType.EB_Lidard
        multiple=True
    else:
        print("Not a choice")
        return
    
    env = create_env(NUM_OF_AGENTS, NUM_OF_CYCLES, LOCAL_RATIO, multiple)
    agents = create_agents(NUM_OF_AGENTS, agent_type, num_of_episodes=NUM_OF_EPISODES, length_of_episode=NUM_OF_CYCLES)

    return agent_type, env, agents, train_choice

    

# def train(choice):

#     """
#     Creates, trains and saves agents. 
    
#     choice - The choice of the agent to train
#     """

#     print("Train is running")
#     agent_type, env, agents, train_choice = _set_up(choice)
#     detailed_exploration_bonus = {1: [], 333: [], 667: [], 1000: []}

#     exploration_bonus_dict = {agent_name: [] for agent_name in agents.keys()}
#     reward_array = np.zeros(NUM_OF_EPISODES)
#     episodes_array = np.array([i+1 for i in range(NUM_OF_EPISODES)])

#     print(f'Training {NUM_OF_AGENTS} agents of type {choice} over {NUM_OF_EPISODES} episodes')
#     for episode_num in range(NUM_OF_EPISODES):
#         reward = train_choice(env, agents, episode_num)
#         reward_array[episode_num] = reward
#         for agent_name, agent_obj in agents.items():
#             if hasattr(agent_obj, 'exploration_bonuses'):
#                 # Store the mean of exploration bonuses for the current episode
#                 mean_bonus = np.mean(agent_obj.exploration_bonuses[-NUM_OF_CYCLES:])  # Assuming NUM_OF_CYCLES is the number of timesteps per episode
#                 exploration_bonus_dict[agent_name].append(mean_bonus)
                
#         # if episode_num in [1, 333, 667, 1000]:
#         #     for agent in agents.values():
#         #         if isinstance(agent, EB_MARL_Comm):  # Check if the agent is of the correct type
#         #             timestep_bonuses = agent.get_exploration_bonuses_for_episode(episode_num, list(range(1, NUM_OF_CYCLES + 1)))
#         #             detailed_exploration_bonus[episode_num].append(timestep_bonuses)
                    

#         if episode_num % 100 == 0:
#             print(episode_num)

#     mean_exploration_bonus_per_episode = np.mean(list(exploration_bonus_dict.values()), axis=0)

#     save(agents, episodes_array, reward_array, agent_type, NUM_OF_AGENTS, NUM_OF_CYCLES, NUM_OF_EPISODES, LOCAL_RATIO, mean_exploration_bonus_per_episode, detailed_exploration_bonus=detailed_exploration_bonus)



def _episode_original_multiple(env, agents, episode_num, oracle=None):

    """
    This trains the UCB/Lidard algorithm for MARL agents for one episode

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


#  PB Algorithm training episode. 
def _episode_EB_Lidard(env, agents, episode_num, oracle=None):
    
    """
    This trains the PB algorithm for MARL agents for one episode

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

def _update_graph(agents, observations):

    """
    This updates the agents to contain the latest on who they can communicate to.
    
    agents - Dictionary of agents
    observations - The observations"""

    for agent_name in agents.keys():

        # Gets the agent position
        agent_obj = agents[agent_name]
        your_pos = observations[agent_name][2:]

        for other_agent in observations.keys():
            # Find every other agent observation
            other_pos = observations[other_agent][2:]
            euclidean = math.sqrt(sum((your_pos-other_pos)**2))

            rounded = int(euclidean)
            # Updates the neighbour distance 
            if rounded <= NUM_OF_CYCLES:
                if rounded == 0:
                    agent_obj.update_neighbour(other_agent, 1)    
                else:
                    agent_obj.update_neighbour(other_agent, rounded)
            else:
                agent_obj.update_neighbour(other_agent, 0)


def _episode_dynamic_graph(env, agents, episode_num):

    """
    This will allow a dynamic graph to be used where who the agent can talk to is dependent on how far they are
    env - The environment
    agents - Dictionary of agents to use
    episode_num - The episode number
    
    Return
    The reward for the episode
    """

    agent_old_state = {agent: -1 for agent in agents.keys()}
    observations = env.reset()    

    _update_graph(agents, observations)

    t = 0
    while env.agents:
        t = t+1
        actions = {}
        for agent_name in agents.keys():        # Take action
            agent_old_state[agent_name] = encode_state(observations[agent_name], NUM_OF_AGENTS)
            action = _policy(agent_name, agents, observations[agent_name], False, t, episode_num)
            actions[agent_name] = action
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Updates the agents with new distances for graphs
        _update_graph(agents, observations)
  
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


def _episode_IQL(env, agents, episode_num):

    """
    Runs an episode for the Independent Q Learning agent in a parallel environemtn

    env - The parallel environment to be played in

    agents - The dictionary of agents to be used
    
    episode_num - Unneeded argument

    Return
    The reward for the episode
    """
    agent_old_state = {agent: -1 for agent in agents.keys()}
    observations = env.reset() 

    t = 0
    while env.agents:
        t = t+1
        actions = {}
        for agent_name in agents.keys():        # Take action
            agent_old_state[agent_name] = encode_state(observations[agent_name], NUM_OF_AGENTS)
            action = _policy(agent_name, agents, observations[agent_name], False, t-1)
            actions[agent_name] = action
        observations, rewards, terminations, truncations, infos = env.step(actions)
            
        for agent_name in agents.keys():       # Update the values
            agent_obj = agents[agent_name]
            old_state = agent_old_state[agent_name]
            current_state = encode_state(observations[agent_name], NUM_OF_AGENTS)
            old_action = actions[agent_name]
            reward = rewards[agent_name]
            agent_obj.update_qTable(old_state, current_state, old_action, reward, t-1)

    return final_reward(rewards)


def _episode_random(env, agents, episode_num):

    """
    Runs an episode for the Random agent in a parallel environment

    env - The environment to be played in

    agents - The dictionary of agents to be used
    
    episode_num - Unneeded argument

    Return
    The reward for the episode
    """
    agent_old_state = {agent: -1 for agent in agents.keys()}
    observations = env.reset() 

    t = 0
    while env.agents:
        t = t+1
        actions = {}
        for agent_name in agents.keys():        # Take action
            agent_old_state[agent_name] = encode_state(observations[agent_name], NUM_OF_AGENTS)
            action = _policy(agent_name, agents, observations[agent_name], False, t-1)
            actions[agent_name] = action
        observations, rewards, terminations, truncations, infos = env.step(actions)

    return final_reward(rewards)


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

    if time_step > NUM_OF_CYCLES:
        return None
    if done:
        return None
    agent = agents[agent_name]
    #print(f'observation: {observation}')
    return agent.policy(encode_state(observation, NUM_OF_AGENTS), time_step)
