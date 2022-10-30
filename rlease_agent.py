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

        to do: abstract this away into an inherited class
        to do read from kwargs (or an agent yaml config)?
        ray: may need to do a custom env
        '''
        if self.rl_library == RL_LIBRARY_RAY_2v0:
            if 'is_gym_registered' in kwargs and 'env_name' in kwargs:
                self.env = gym.make(kwargs['env_name'])
            else:
                raise ValueError("ERROR: functionality not yet implemented")
            restore_config = {
                "framework": "torch",
                "num_workers": 1,
            }

            trainer = PPOTrainer(config=restore_config, env=self.env)
        else:
            raise ValueError("Error: this library is not supported")
        # # custom env example
        # def env_creator_sa(args):
        #     schema = 'citylearn_challenge_2022_phase_1'
        #     env = CityLearnEnv(schema=schema)
        #     sa_env = SingleAgentSingleBuildingCityLearnEnv(env)
        #
        #     return sa_env
        #
        # # env = env_creator_sa({})
        # register_env("citylearn_rllib_env", env_creator_sa)
        #
        # restore_config = {
        #     "framework": "torch",
        #     # "model": model_config,
        #     # "multiagent": {
        #     #     "policies": rc_policies,
        #     #     "policy_mapping_fn": rc_policy_mapping_fn,
        #     # },
        #     "num_workers": 1,
        # }
        #
        # trainer = PPOTrainer(config=restore_config, env="citylearn_rllib_env")