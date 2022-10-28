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

RL_LIBRARY_RAY_2v0 = 'rl_library_ray_2v0'

class RLeaseAgent():
    def __init__(self, **kwargs):
        if 'rl_library' in kwargs:
            self.rl_library = kwargs['rl_library']
        else:
            raise KeyError('rl_library key must be specified')

        if 'is_gym_registered' in kwargs and 'env_name' in kwargs:
            env = gym.make(kwargs['env_name'])
        else:
            pass
            #create env based on env type
            # if 'env_create_type' in kwargs:
            #     env_create_type = kwargs['env_create_type']
            #     if env_create_type == 'rllib':

        self.agent = self.load_agent(**kwargs)

    def get_action_vf(self, obs):
        '''
        return an action and vf score from an obs
        '''
        if self.rl_library == RL_LIBRARY_RAY_2v0:
            # get action
            action = self.agent.compute_single_action(o)
            # get value function value
            policy = self.agent.get_policy() #might need to specify a policy id policy_id=policy_id #like policy_id = "main" or sometimes ray has a default value for this
            vf_tensor = policy.model.value_function()  # this is updated after every self.trainer.compute_single_action()
            assert vf_tensor.shape[0] == 1  # should only be a tensor of one value
            vf = vf_tensor.cpu().detach().numpy()[0]

            # to do: confirm this is proper way to do it with ray2.0
            # to do: option to return logits maybe
            # get logits for this obs
            # logits, _ = policy.model.forward({"obs": obs_tensor}, None, None)
            # logits = logits.cpu().detach().numpy().flatten()
        else:
            raise ValueError("Error: this library is not supported")

        return action, vf_score

    def load_agent(self, kwargs):
        '''
        return an agent
        '''