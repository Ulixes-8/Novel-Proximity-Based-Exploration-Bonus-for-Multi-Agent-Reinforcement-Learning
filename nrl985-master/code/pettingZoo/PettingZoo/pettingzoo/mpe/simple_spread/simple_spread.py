# noqa
"""
# Simple Spread

```{figure} mpe_simple_spread.gif
:width: 140px
:name: simple_spread
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.mpe import simple_spread_v2` |
|----------------------|-----------------------------------------------|
| Actions              | Discrete/Continuous                           |
| Parallel API         | Yes                                           |
| Manual Control       | No                                            |
| Agents               | `agents= [agent_0, agent_1, agent_2]`         |
| Agents               | 3                                             |
| Action Shape         | (5)                                           |
| Action Values        | Discrete(5)/Box(0.0, 1.0, (5))                |
| Observation Shape    | (18)                                          |
| Observation Values   | (-inf,inf)                                    |
| State Shape          | (54,)                                         |
| State Values         | (-inf,inf)                                    |

```{figure} ../../_static/img/aec/mpe_simple_spread_aec.svg
:width: 200px
:name: simple_spread
```

This environment has N agents, N landmarks (default N=3). At a high level, agents must learn to cover all the landmarks while avoiding collisions.

More specifically, all agents are globally rewarded based on how far the closest agent is to each landmark (sum of the minimum distances). Locally, the agents are penalized if they collide with other agents (-1 for each collision). The relative weights of these rewards can be controlled with the
`local_ratio` parameter.

Agent observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_spread_v2.env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False)
```



`N`:  number of agents and landmarks

`local_ratio`:  Weight applied to local reward and global reward. Global reward weight will always be 1 - local reward weight.

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.utils.conversions import parallel_wrapper_fn

from .._mpe_utils.core import Agent, Landmark, World
from .._mpe_utils.scenario import BaseScenario
from .._mpe_utils.simple_env import SimpleEnv, make_env


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=3,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
    ):
        EzPickle.__init__(
            self, N, local_ratio, max_cycles, continuous_actions, render_mode
        )
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N)
        super().__init__(
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions
        )
        # for i, lm in enumerate(world.landmarks):
        #     print(lm.state.p_pos)
        #     lm.state.p_pos = np.array([0.5+ i, 0.5 + i])
        #     print(lm.state.p_pos)

        # for i, agent in enumerate(world.agents):
        #     print(agent.state.p_pos)
        #     agent.state.p_pos = np.array([1 + i, 1 + i])
        #     print(agent.state.p_pos)

        self.metadata["name"] = "simple_spread_v2"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, N=3):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = N
        num_landmarks = N
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world, np_random, test=False):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # set random initial states.  These are the continuous values.
        for i, agent in enumerate(world.agents):
            
            if len(world.agents) == 4:
                self.four_agent(i, agent, test)
            elif len(world.agents) == 12:
                self.twelve_agent(i, agent, test)
                
            #agent.state.p_vel = np.array([0,0])
            # These are the observed values - related to the continuous values but clipped to be in the state space
            agent.state.obs_pos = np.zeros(world.dim_p)
            agent.state.obs_vel = np.zeros(world.dim_p)
            self.convert_values(agent, world)
            agent.state.c = np.zeros(world.dim_c)

        #position = np_random.uniform(-2, +2, world.dim_p)
        position = np.array([0 for num in range(world.dim_p)])     # So all surrounding the origin.
        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = np_random.uniform(-2, +2, world.dim_p)
            landmark.state.p_pos = position
            #landmark.state.p_pos = [0.5, 0.5]
            landmark.state.p_vel = np.zeros(world.dim_p)

            landmark.state.obs_pos = np.zeros(world.dim_p)
            landmark.state.obs_vel = np.zeros(world.dim_p)

            self.convert_values(landmark, world)

    def twelve_agent(self, i, agent, test):
        if i == 0:
            if test:
                agent.state.p_pos = np.array([4/1.414, -4/1.414])
                agent.state.p_vel = np.array([0,0])
            else:
                agent.state.p_pos = np.array([-4/1.414, 4/1.414])
                agent.state.p_vel = np.array([0,0])
        
        elif i == 1:
            if test:
                agent.state.p_pos = np.array([4/1.414, 1/1.414])
                agent.state.p_vel = np.array([0,0])
            else:
                agent.state.p_pos = np.array([-1/1.414, 4/1.414])
                agent.state.p_vel = np.array([0,0])

        elif i == 2:
            if test:
                agent.state.p_pos = np.array([-4/1.414, -1/1.414])
                agent.state.p_vel = np.array([0,0])
            else:
                agent.state.p_pos = np.array([1/1.414, 4/1.414])
                agent.state.p_vel = np.array([0,0])

        elif i == 3:
            if test:
                agent.state.p_pos = np.array([4/1.414, -1/1.414])
                agent.state.p_vel = np.array([0,0])
            else:
                agent.state.p_pos = np.array([4/1.414, 4/1.414])
                agent.state.p_vel = np.array([0,0])

        elif i == 4:
            if test:
                agent.state.p_pos = np.array([-1/1.414, 4/1.414])
                agent.state.p_vel = np.array([0,0])
            else:
                agent.state.p_pos = np.array([4/1.414, 1/1.414])
                agent.state.p_vel = np.array([0,0])
        
        elif i == 5:
            if test:
                agent.state.p_pos = np.array([4/1.414, 4/1.414])
                agent.state.p_vel = np.array([0,0])
            else:
                agent.state.p_pos = np.array([4/1.414, -1/1.414])
                agent.state.p_vel = np.array([0,0])

        elif i == 6:
            if test:
                agent.state.p_pos = np.array([-4/1.414, 4/1.414])
                agent.state.p_vel = np.array([0,0])
            else:
                agent.state.p_pos = np.array([4/1.414, -4/1.414])
                agent.state.p_vel = np.array([0,0])

        elif i == 7:
            if test:
                agent.state.p_pos = np.array([-1/1.414, -4/1.414])
                agent.state.p_vel = np.array([0,0])
            else:
                agent.state.p_pos = np.array([1/1.414, -4/1.414])
                agent.state.p_vel = np.array([0,0])

        elif i == 8:
            if test:
                agent.state.p_pos = np.array([1/1.414, -4/1.414])
                agent.state.p_vel = np.array([0,0])
            else:
                agent.state.p_pos = np.array([-1/1.414, -4/1.414])
                agent.state.p_vel = np.array([0,0])

        elif i == 9:
            if test:
                agent.state.p_pos = np.array([-4/1.414, 1/1.414])
                agent.state.p_vel = np.array([0,0])
            else:
                agent.state.p_pos = np.array([-4/1.414, -4/1.414])
                agent.state.p_vel = np.array([0,0])

        elif i == 10:
            if test:
                agent.state.p_pos = np.array([1/1.414, 4/1.414])
                agent.state.p_vel = np.array([0,0])
            else:
                agent.state.p_pos = np.array([-4/1.414, -1/1.414])
                agent.state.p_vel = np.array([0,0])

        elif i == 11:
            if test:
                agent.state.p_pos = np.array([-4/1.414, -4/1.414])
                agent.state.p_vel = np.array([0,0])
            else:
                agent.state.p_pos = np.array([-4/1.414, 1/1.414])
                agent.state.p_vel = np.array([0,0])

        

    def four_agent(self, i, agent, test):
        if i == 0:
            # If test is true we swap A & D
            if test:
                agent.state.p_pos = np.array([1/1.414, 1/1.414]) #np_random.uniform(-1, +1, world.dim_p)
                agent.state.p_vel = np.array([0,0]) #agent.state.p_vel = np_random.uniform(-1, +1, world.dim_p)
            else:
                agent.state.p_pos = np.array([-1/1.414, 1/1.414]) #np_random.uniform(-1, +1, world.dim_p)
                agent.state.p_vel = np.array([0,0]) #agent.state.p_vel = np_random.uniform(-1, +1, world.dim_p)
        if i == 1:
            agent.state.p_pos = np.array([-1/1.414, -1/1.414]) #np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.array([0,0]) #agent.state.p_vel = np_random.uniform(-1, +1, world.dim_p)
        if i == 2:
            agent.state.p_pos = np.array([1/1.414, -1/1.414]) #np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.array([0,0]) #agent.state.p_vel = np_random.uniform(-1, +1, world.dim_p)
        if i == 3:
            # If test is true we swap A & D
            if test:
                agent.state.p_pos = np.array([-1/1.414, 1/1.414]) #np_random.uniform(-1, +1, world.dim_p)
                agent.state.p_vel = np.array([0,0]) #agent.state.p_vel = np_random.uniform(-1, +1, world.dim_p)
            else:
                agent.state.p_pos = np.array([1/1.414, 1/1.414]) #np_random.uniform(-1, +1, world.dim_p)
                agent.state.p_vel = np.array([0,0]) #agent.state.p_vel = np_random.uniform(-1, +1, world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents rewarded for how close they are to their landmark
        for lm in world.landmarks:
            # print(lm.state.p_pos)

            if lm.name[-1] == agent.name[-1]:
                self.convert_values(agent, world)
                self.convert_values(lm, world)
                # print(lm.state.p_pos)
                return -np.sqrt(np.sum(np.square(agent.state.obs_pos - lm.state.obs_pos)))


    def observation(self, agent, world):
        # # get positions of all entities in this agent's reference frame
        # entity_pos = []
        # for entity in world.landmarks:  # world.entities:
        #     entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # # entity colors
        # entity_color = []
        # for entity in world.landmarks:  # world.entities:
        #     entity_color.append(entity.color)
        # # communication of all other agents
        # comm = []
        # other_pos = []
        # for other in world.agents:
        #     if other is agent:
        #         continue
        #     comm.append(other.state.c)
        #     other_pos.append(other.state.p_pos - agent.state.p_pos)

        self.convert_values(agent,world)
        # print(np.concatenate(
        #     [agent.state.p_vel] + [agent.state.p_pos]
        # ))
        return np.concatenate(
            [agent.state.obs_vel] + [agent.state.obs_pos]
        )

    def convert_values(self, entity, world):
        # print("HELLO")
        # correct_values = np.array([-2, -1.56, -1.11, -0.67, -0.22, 
        #                 0.22, 0.67, 1.11, 1.56, 2])

        # These are the 10 values which we can match to.

        if len(world.agents) == 4:
            self.convert_values_four(entity)
        else:
            self.convert_values_twelve(entity)
        

    
    def convert_values_four(self, entity):
        correct_values = np.array([-2, -1.6, -1.2, -0.8, -0.4, 
                         0.0, 0.4, 0.8, 1.2, 1.6])
        
        # This converts the position value
        for i, value in enumerate(entity.state.p_pos):
            smallest_distance = abs(correct_values-value) #Something like this
            index_of_smallest = np.where(smallest_distance == min(smallest_distance))
            # print(type(index_of_smallest))
            # print(value, correct_values)
            # print(index_of_smallest)
            # print(index_of_smallest[0])
            if len(index_of_smallest[0]) > 1:
                index_of_smallest = index_of_smallest[0][0]
            # print(index_of_smallest)
            entity.state.obs_pos[i] = correct_values[index_of_smallest]

        # This converts the velocity value
        for i, value in enumerate(entity.state.p_vel):
            smallest_distance = abs(correct_values-value)#Something like this
            index_of_smallest_p_vel = np.where(smallest_distance == min(smallest_distance))
            # print(type(index_of_smallest_p_vel))
            # print(value, correct_values)
            # print(index_of_smallest_p_vel)
            # print(index_of_smallest_p_vel[0])
            if len(index_of_smallest_p_vel[0]) > 1:
                index_of_smallest_p_vel = index_of_smallest_p_vel[0][0]
            

            entity.state.obs_vel[i] = correct_values[index_of_smallest_p_vel]

    def convert_values_twelve(self, entity):
        correct_values = np.array([-8, -7.6, -7.2, -6.8, -6.4, 
                                   -6, -5.6, -5.2, -4.8, -4.4, 
                                   -4, -3.6, -3.2, -2.8, -2.4, 
                                   -2, -1.6, -1.2, -0.8, -0.4,  
                                    0.0, 0.4, 0.8, 1.2, 1.6,
                                    2.0, 2.4, 2.8, 3.2, 3.6,
                                    4.0, 4.4, 4.8, 5.2, 5.6,
                                    6.0, 6.4, 6.8, 7.2, 7.6,])
        

        # This converts the position value
        for i, value in enumerate(entity.state.p_pos):
            smallest_distance = abs(correct_values-value) #Something like this
            index_of_smallest = np.where(smallest_distance == min(smallest_distance))
            # print(type(index_of_smallest))
            # print(value, correct_values)
            # print(index_of_smallest)
            # print(index_of_smallest[0])
            if len(index_of_smallest[0]) > 1:
                index_of_smallest = index_of_smallest[0][0]
            # print(index_of_smallest)
            entity.state.obs_pos[i] = correct_values[index_of_smallest]

        # This converts the velocity value
        for i, value in enumerate(entity.state.p_vel):
            smallest_distance = abs(correct_values-value)#Something like this
            index_of_smallest_p_vel = np.where(smallest_distance == min(smallest_distance))
            # print(type(index_of_smallest_p_vel))
            # print(value, correct_values)
            # print(index_of_smallest_p_vel)
            # print(index_of_smallest_p_vel[0])
            if len(index_of_smallest_p_vel[0]) > 1:
                index_of_smallest_p_vel = index_of_smallest_p_vel[0][0]