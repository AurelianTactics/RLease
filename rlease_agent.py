'''
Class for creating agent compatible with RLease
'''

'''
working:
    so ray needs to register an env, do that in agent load
    here just make the env needed for the loop dont' worry about ray or whatever here

to do:
want an abstract class then to inherit
for now just a couple of basic things here then abstract when more functionality is figured out
'''

import gym
from ray.tune import register_env

class RLeaseAgent():
    def __init__(self, **kwargs):
        if 'is_gym_registered' in kwargs and 'env_name' in kwargs:
            env = gym.make(kwargs['env_name'])
        else:
            pass
            #create env based on env type
            # if 'env_create_type' in kwargs:
            #     env_create_type = kwargs['env_create_type']
            #     if env_create_type == 'rllib':

        # then fill out functions needed by traj loop