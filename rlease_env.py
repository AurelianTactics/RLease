'''
Class for creating and return an env to work with rlease_env

goal: user specifies the env name and the env type (gym, DeepMind, etc), env created here and handled properly
'''

'''
working:
see what functionality is needed for this. probably some sort of step but idk maybe not needed now if just default gym    

to do:
not sure on structure for custom envs. maybe inherit from the class?
deepmind env, other envs types, custom env types, etc

'''

import gym

ENV_TYPE_GYM = 'env_type_gym'


class RLeaseEnv():
  def __init__(self, **kwargs):
    if 'env_type' in kwargs:
      if kwargs['env_type'] == ENV_TYPE_GYM:
        self.env = gym.make(kwargs['env_name'])
    else:
      raise ValueError('env type not supported')

  def reset(self):
    # to do, handle non gym tpes
    return self.env.reset()

  def step(self, action):
    # to do: handle non gym types
    obs, reward, done, info = self.env.step(action)

    return obs, reward, done, info