import hashlib
import numpy as np

def encode_state(observation, num_of_agents):

    """
    Encodes a state which can be saved over Python instances.  The hash function changes after every Python instance
    Taken from *Gertjan Verhoeven* Notebook found on PettingZoo website (Which has been subsequently deleted).  
    
    observation - What the agent can see

    num_of_agents - The number of agents overall. Unneeded variable but was used in testing

    returns - an encoded state"""

    # encode observation as bytes           
    obs_bytes = str(observation).encode('utf-8')
    # create md5 hash
    m = hashlib.md5(obs_bytes)
    # return hash as hex digest
    state = m.hexdigest()
    return(state)