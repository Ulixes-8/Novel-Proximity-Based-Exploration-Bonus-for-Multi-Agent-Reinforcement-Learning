from collections import defaultdict
import math
import random
from agent import Agent
from hyperparameters import eb_marl_hyperparameters, agent_hyperparameters

## THIS FILE CONTAINS THE PB Exploration Algorithm applied to Multi Agent Q Learning, as described in algorithm 2 of the thesis.  

class EB_MARL_Comm(Agent):


    def __init__(self, agent_name, num_of_agents, num_of_episodes, length_of_episode, gamma_hop):
        self.exploration_bonuses_detailed = {}  # New attribute for detailed tracking
        self.real_state_map = {}  # Map to store the real state for each hash
        self.exploration_bonuses = [] # Stores exploration bonuses for visualization. 


        # The set of H number of Q-Tables.
        self.qTables = {i+1: defaultdict(lambda: defaultdict(lambda: length_of_episode)) for i in range(length_of_episode)}

        # This will contain the number of times each state action has been seen for each timestep
        self.nTables = {i+1: defaultdict(lambda: defaultdict(lambda: 0)) for i in range(length_of_episode)}

        # The u set (episode number is indexed from 0 whilst the timestep is indexed from 1)
        self.uSet = {i: {i+1: defaultdict(lambda: defaultdict(lambda: set())) for i in range(length_of_episode +gamma_hop+1)} for i in range(num_of_episodes)} 

        # The v set (episode number is indexed from 0 whilst the timestep is indexed from 1)
        self.vSet = {i: {i+1: defaultdict(lambda: defaultdict(lambda: set())) for i in range(length_of_episode+gamma_hop+1)} for i in range(-length_of_episode-1, num_of_episodes+round(gamma_hop%length_of_episode)+1)}

        self.next_add = set()

        # Of form agent_name, distance = 0 if no connection
        self._neighbours = {f'agent_{i}': 0 for i in range(num_of_agents)}   
        self._num_neighbours = 1
        super().__init__(agent_name)

        self.H = length_of_episode
        self.episodes = num_of_episodes

        # This corresponds to A 
        num_of_actions = 5 

        # This corresponds to S
        num_of_states = agent_hyperparameters['size_of_state_space']

        # This corresponds to T (H*K)
        T = num_of_episodes*length_of_episode
        
        #pb hyperparameters 
        self.initial_decay_factor = eb_marl_hyperparameters['initial_decay_factor']
        self.decay_rate = eb_marl_hyperparameters['decay_rate']
        self.scaling_factor = eb_marl_hyperparameters['scaling_factor']
        self.probability = eb_marl_hyperparameters['probability']
        self.log_term = math.log((num_of_states * num_of_actions * T * num_of_agents)/self.probability)  
        
        # The v-table values.  Set to H as this corresponds to the q tables currently.
        self.vTable = {j+1: defaultdict(lambda: self.H) for j in range(length_of_episode+1)}
        

    
    def exponential_decay(self, t):
        """
        Calculate the exponential decay factor based on the time step.

        t - The current time step or episode number.

        Return
        The decay factor at the given time step.
        """
        return self.initial_decay_factor * math.exp(-self.decay_rate * t)
    
    
    def update_real_state_map(self, hash_value, real_state):
        """Store the mapping from state hash to its original real state."""
        self.real_state_map[hash_value] = real_state

    def update_neighbour(self, agent_to_update, connection_quality):

        """
        This updates the connection value in the neighbours dictionary.
        
        agent_to_update - The agent with a new connection to be updated to
        
        connection_quality - How long the distance is between agents
        """

        if self._neighbours[agent_to_update] == 0 and connection_quality != 0:
            self._num_neighbours +=1
        elif self._neighbours[agent_to_update] != 0 and connection_quality == 0:
            self._num_neighbours -= 1

        self._neighbours[agent_to_update] = connection_quality

    def policy(self, state, time_step):

        """
        This gets the next move to be played
        
        state - The current state we are in
        
        time_step - The time step in the episode

        Return
        The action to be taken
        """

        # Choose the largest value
        max_value = float('-inf')
        move = -1
        q_table = self.qTables[time_step]
        values = [0,1,2,3,4]
        random.shuffle(values) # Shuffled to get randomness at start
        for i in values:        
            if q_table[state][i] > max_value:
                max_value = q_table[state][i]
                move = i 
        return move

    
    def play_normal(self, state, time_step, *args):

        """
        Plays the episode for showing.  Plays the best action in the q-table
        
        state - The current state we are in
        
        time_step - The time step in the episode

        *args - Spare arguments.

        Return 
        The action to be taken
        """

        q_table = self.qTables[time_step]
        max_value = float('-inf')
        move = -1
        values = [0,1,2,3,4]
        random.shuffle(values)
        for i in values:        
            if q_table[state][i] > max_value:
                max_value = q_table[state][i]
                move = i 
        return move


    def choose_smallest_value(self, state, time_step):

        """
        This chooses the smallest value - either from the max value from the Q-Table or H

        state - The current state we are in
        
        time_step - The time step in the episode

        Return
        The smaller value
        """

        max_value = float('-inf')
        values = [0,1,2,3,4]
        random.shuffle(values)
        for i in values:
            if self.qTables[time_step][state][i] > max_value:
                max_value = self.qTables[time_step][state][i]
        return min(self.H, max_value)


    def message_passing(self, episode_num, time_step, old_state, old_real_state, action, current_state, current_real_state, reward, agents_dict):
        """
        This passes messages to other agents which this agent can communicate to.

        episode_num - The episode number the agent is in
        time_step - The time step the agent is in
        old_state - The hashed state the agent was in
        action - The action taken
        current_state - The hashed state after the agent has taken
        reward - The reward for the agents move
        agents_dict - The agents dictionary
        """

        for agent in self._neighbours.keys():
            if self._neighbours[agent] != 0:
                agent_obj = agents_dict[agent]

                self.send_message(episode_num, time_step, old_state, old_real_state, action, current_state, current_real_state, reward, agent_obj, self._neighbours[agent])

    def update(self, episode_num, time_step, old_state, current_state, action, reward):

        """
        This updates the u set and v set using the set update rules

        episode_num - The episode number the agent is one

        time_step - The time step the agent is on

        old_state - The previous state the agent was on

        current_state - The state the agent is currently on

        action - The action the agent has taken

        reward - The reward gained by the action taken from the old_state
        """

        # Update the uset to contain the latest reward and current state
        old_set = self.uSet[episode_num][time_step][old_state][action]
        old_set.add((reward, current_state))
        self.uSet[episode_num][time_step][old_state][action] = old_set

        new_set = set()
        
        # Add the new data from other agents into vSet.  Only added in the vSet when it 'reaches' the agent (dis == 0).  Added into the vset at the episode and timestep of when the message was sent
        for message_data in self.next_add:
            time_step_other, episode_num_other, agent_name_other, current_state_other, action_other, next_state_other, reward_other, dis = message_data
            
            if dis == 0:

                old_set = self.vSet[episode_num_other][time_step_other][current_state_other][action_other]
                old_set.add((reward_other, next_state_other))
                self.vSet[episode_num_other][time_step_other][current_state_other][action_other] = old_set
            else:
                dis -= 1
                new_set.add(tuple([time_step_other, episode_num_other, agent_name_other, current_state_other, action_other, next_state_other, reward_other, dis]))

        self.next_add = new_set
        
        # Add everything into the vSet for the current episode and timestep
        for state in self.uSet[episode_num][time_step]:
            for act in self.uSet[episode_num][time_step][state]:
                for element in self.uSet[episode_num][time_step][state][act]:
                        reward_new, next_state = element
                        self.vSet[episode_num][time_step][state][action].add((reward_new, next_state))

        
        
    def receive_message(self, message, dis):
        """
        This is how the agent should receive a message

        message - The tuple containing the message

        dis - How far away the agent sending the message is
        """
        # Extracting information from the received message
        time_step, episode_num, sender_agent_name, old_state, old_real_state, action, new_state, new_real_state, reward = message

        # Update the real state map with both old and new real states
        self.update_real_state_map(old_state, old_real_state)
        self.update_real_state_map(new_state, new_real_state)

        # Modify the message to exclude the real states
        modified_message = (time_step, episode_num, sender_agent_name, old_state, action, new_state, reward, dis)
        self.next_add.add(modified_message)

    def send_message(self, episode_num, time_step, current_state, current_real_state, action, next_state, next_real_state, reward, agent_obj, distance):
        """
        How an agent sends a message

        [current parameters...]
        current_real_state - The real state corresponding to current_state
        next_real_state - The real state corresponding to next_state
        """

        message_tuple = (time_step, episode_num, self.agent_name(), current_state, current_real_state, action, next_state, next_real_state, reward)
        agent_obj.receive_message(message_tuple, distance)


# ## HERE IS THE PB EXPLORATION ALGO!!!!

    def update_values(self, episode_num_max, time_step_max):
        
        # Initialize a default state value vector based on the size of the state space.
        default_value = [0 for i in range(agent_hyperparameters['size_of_state_space'])]
        
        # Iterate over episodes from the second-to-last to the last.
        for episode_num in range(episode_num_max-1, episode_num_max+1):
            # Iterate through each time step within the given horizon.
            for time_step in range(1, self.H+1):
                # Iterate over all state hashes encountered at this episode and time step.
                for state_hash in self.vSet[episode_num][time_step].keys():
                    # Iterate over all actions taken from those states.
                    for action in self.vSet[episode_num][time_step][state_hash].keys():
                        # Iterate over all rewards and subsequent states resulting from those actions.
                        for reward, next_state_hash in self.vSet[episode_num][time_step][state_hash][action]:
                            # Increment the counter for how many times a given state-action pair has been visited.
                            self.nTables[time_step][state_hash][action] += 1
                            
                            # Retrieve the updated visitation count for the current state-action pair.
                            t = self.nTables[time_step][state_hash][action]
                            
                            # Calculate the current decay factor based on the time step.
                            current_decay_factor = self.exponential_decay(time_step)
                            # Initialize the bonus for exploration to zero.
                            b = 0
                            
                            # Retrieve the real state vector for the current state hash, defaulting if not found.
                            real_state = self.real_state_map.get(state_hash, default_value)
                            # Iterate over all other state hashes at the current time step.
                            for other_state_hash in self.nTables[time_step].keys():
                                # Retrieve the real state vector for the other state hash.
                                other_real_state = self.real_state_map.get(other_state_hash, default_value)
                                # Calculate the Euclidean distance between the current and other state.
                                distance = math.sqrt(sum([(s - o)**2 for s, o in zip(real_state, other_real_state)]))
                                # For each action available from the other state, accumulate the bonus.
                                for other_action in self.nTables[time_step][other_state_hash].keys():
                                    b += self.scaling_factor * self.nTables[time_step][other_state_hash][other_action] * distance
                            # Adjust the bonus based on the decay factor and a logarithmic term.
                            b *= current_decay_factor * self.log_term
                            
                            # Record the calculated bonus.
                            self.exploration_bonuses.append(b)

                            # Calculate the learning rate alpha, dependent on the visitation count.
                            alpha = (self.H + 1) / (self.H + t)
                            # Calculate the weighted current Q-value estimate.
                            initial = (1 - alpha) * self.qTables[time_step][state_hash][action]
                            # Calculate the updated Q-value incorporating the reward, estimated future value, and bonus.
                            expected_future = alpha * (reward + self.vTable[time_step + 1][next_state_hash] + b)
                            # Combine the current and future estimates to form the new Q-value.
                            new_score = initial + expected_future
                            # Update the Q-table with the new Q-value for the current state-action pair.
                            self.qTables[time_step][state_hash][action] = new_score
                            # Update the value table for the current state based on the smallest Q-value across all actions.
                            self.vTable[time_step][state_hash] = self.choose_smallest_value(state_hash, time_step)

                # Reset the visited state-action pairs for the next episode and time step to ensure fresh calculations.
                self.vSet[episode_num][time_step] = defaultdict(lambda: defaultdict(lambda: set()))
