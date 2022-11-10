'''
Env to help debug RL algorithms
'''

import gym

# constants
CONFIG_KEY_ACTION_SIZE = 'action_size'
CONFIG_KEY_OBS_SIZE = 'obs_size'
CONFIG_KEY_TIMESTEPS = 'timesteps'
CONFIG_KEY_REWARD_TYPE = 'reward_type'

class DebugGym(gym.Env):
    def __init__(self. **kwargs):
        # action size, obs size, num time steps, reward value
        pass

    def reset(self):
        pass

    def step(self):
        pass
    
    def render(self):
        pass