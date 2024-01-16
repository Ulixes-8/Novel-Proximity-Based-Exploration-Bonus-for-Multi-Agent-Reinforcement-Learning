"""This will create the necessary environment for use of the rest of the program.  Plug & play with this this abstracting which environment is used"""
#from pettingzoo.mpe import simple_spread_v2
from pettingZoo.PettingZoo.pettingzoo.mpe import simple_spread_v2

def create_env(num_of_agent=3, num_of_cycles=25, local_ratio=0.5, multiple=False, render_mode='ansi'):
    """
    Creates the environment to be played in

    num_of_agent = 3 - The number of agents to be created.  Default is default for Simple Spread

    num_of_cycles = 25 - The number of frames (step for each agent).  Default is default for Simple Spread

    local_ratio = 0.5 - Weight applied to local and global reward.  Local is collisions global is distance to landmarks.  Default is default for simple spread

    multiple = False - Whether you need a parallel environment to be created
    
    render_mode = 'ansi' - Whether to be rendered on screen.  Default is not.
    """
    if multiple:
        return simple_spread_v2.parallel_env(render_mode=render_mode,N=num_of_agent, max_cycles=num_of_cycles, local_ratio=local_ratio)
    return simple_spread_v2.env(render_mode=render_mode,N=num_of_agent, max_cycles=num_of_cycles, local_ratio=local_ratio)
