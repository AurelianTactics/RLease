'''
Class for creating agent compatible with RLease
'''

'''
working:
    so ray needs to register an env, do that in agent load
    here just make the env needed for the loop dont' worry about ray or whatever here

to do:
how to handle imports of rl packages that people might not have installed. ie don't want ray dependencies if using cleanrl etc
want an abstrac class then to inherit
for no just a fe basic things here then abstract when more functionality is figured out
cleanrl jax
cleanrl torch load test
    also deterministic vs. stochastic
cleanrl is alg specific apparently. need a wrapper around model
    also need to better add stochastic or not like for PPO. can hard code it here but idk in future
        people training would ahve to modify their code or not
        maybe do an example

'''

import abc
from abc import ABC, abstractmethod
import gym
import numpy as np
# from ray.rllib import agents
# import ray.rllib.algorithms.ppo as ppo
# from ray.tune.registry import register_env

RL_LIBRARY_RAY_2v0 = 'rl_library_ray_2v0'
RL_LIBRARY_CLEANRL_1v0_TORCH = 'rl_library_cleanrl_1v0_TORCH'

class RLeaseAgentBaseClass():

    @abstractmethod
    def get_action_vf(self):
        pass

class RLeaseAgentProvided(RLeaseAgentBaseClass):
    '''
    agent is provided in init and has get_action
    '''

    def __init__(self, agent):
        self.agent = agent

    def get_action_vf(self, obs, info=None):
        action, vf = self.agent.get_action_vf(obs, info)

        return action, vf

class RLeaseAgentTorchModel(RLeaseAgentBaseClass):
    '''
    agent is a loaded torch model. below example works with cleanrl_ppo default (if action choice is not deterministic)
    '''
    def __init__(self, model_class, state_dict_path, **kwargs):
        '''
        model_class: torch model class

        state_dict_path: str
            path to torch model.state_dict(
        '''


        self.agent = self.load_agent(model_class, state_dict_path, **kwargs)

        if 'action_choice' in kwargs:
            self.action_choice = kwargs['action_choice']
        else:
            self.action_choice = 'deterministic'

    def get_action_vf(self, obs, info=None):
        '''
        return an action and vf score from an obs
        '''
        import torch

        obs = torch.tensor(obs, dtype=torch.float32)
        if self.action_choice == 'deterministic':
            action_stochastic, lobprob, probs_entropy, vf, logits = self.agent.get_action_and_value(obs)
            # to do decide between equal outputs if maxes are the same
            action = np.argmax(logits.cpu().detach().numpy())
        else:
            # default, stochastic
            action, logprob, probs_entropy, vf = self.agent.get_action_and_value(obs)
            action = action.cpu().detach().numpy()[0]

        vf = vf.cpu().detach().numpy()[0]

        return action, vf

    def load_agent(self, model, state_dict_path, **kwargs):
        '''
        return an agent

        '''
        import torch

        model.load_state_dict(torch.load(state_dict_path))
        model.eval()

        return model


class RLeaseAgent(RLeaseAgentBaseClass):
    '''
    loads agent and provides wrapper for rest of RLease
    to do: probably abstract classes for different alg types
    '''
    def __init__(self, load_path=None, **kwargs):
        if 'rl_library' in kwargs:
            self.rl_library = kwargs['rl_library']
        else:
            raise KeyError('Error: RLeaseAgent init rl_library key must be specified')

        self.agent = self.load_agent(load_path, **kwargs)

    def get_action_vf(self, obs, info=None):
        '''
        return an action and vf score from an obs
        '''
        if self.rl_library == RL_LIBRARY_RAY_2v0:
            # get action
            action = self.agent.compute_single_action(obs)
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

    def load_agent(self, load_path, kwargs):
        '''
        return an agent

        to do: abstract this away into an inherited class
        to do read from kwargs (or an agent yaml config)?
        to do: abstract custom gym env
        to do code for registered gym env
        ray: may need to do a custom env
        '''
        if self.rl_library == RL_LIBRARY_RAY_2v0:
            # to do: code for registered gym env

            # if 'is_gym_registered' in kwargs and 'env_name' in kwargs:
            #     self.env = gym.make(kwargs['env_name'])
            # else:
            #     raise ValueError("ERROR: functionality not yet implemented")
            #
            # restore_config = {
            #     "framework": "torch",
            #     "num_workers": 1,
            # }

            if load_path is not None:
                restore_config = {
                    #"env": registered_env_name,
                    #"env_config": env_config,
                    "framework": "torch",
                    "model": {
                        "fcnet_hiddens": [32, 32],
                        "fcnet_activation": "relu",
                    },
                }
                keys_to_check_list =["env_config", 'num_workers', 'evaluation_duration', 'evaluation_interval']
                for k in keys_to_check_list:
                    if k in kwargs:
                        restore_config[k] = kwargs[k]

                # to do this is a ray only thing but not srue how to abstract it properly on the import
                # maybe leave it to suer with custom code where they specify
                def env_creater(env_config):
                    pass
                    # import my_custom_env; return my_custom_env.MyCustomEnv(env_config) # return an env instance

                if 'is_custom_env' in kwargs and kwargs['is_custom_env']:
                    registered_env_name = kwargs.get('custom_env_name', 'my_custom_env')
                    register_env(registered_env_name, env_creater)
                    restore_config["env"] = registered_env_name

                agent = ppo.PPO(config=restore_config)

                agent.restore(load_path)

            return agent
        else:
            raise ValueError("Error: this library is not supported")


class RLeaseAgentPredictModel(RLeaseAgentBaseClass):
    '''
    UNFINISHED agent is provided in init and has get_action
    '''

    def __init__(self, model):
        self.model = model

    def get_action_vf(self, obs, info=None):
        # might need to combine obs and info
        vf = None
        action = self.model.predict(obs)

        return action, vf