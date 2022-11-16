'''
Env to help debug RL algorithms

Based on ideas from Andy Jones RL Debugging
'''

'''
working
test this
document this better
write up ideas for future envs

ideas
can scale this up behond these simple examples (maybe more abastracted env)
ie more than 2 actions, more the 2 obs, etc
'''

import gym

# constants
CONFIG_KEY_ACTION_SIZE = 'action_size'
CONFIG_KEY_OBS_TYPE = 'obs_size'
CONFIG_KEY_TIMESTEP_MAX = 'timestep_max'
CONFIG_KEY_REWARD_TYPE = 'reward_type'
# reward types
REWARD_TYPE_1 = 'reward_1'
REWARD_TYPE_GOAL_1 = 'reward_1_at_goal'
REWARD_TYPE_ACTION_DEPENDENT = 'reward_action_dependent'
REWARD_TYPE_OBS_DEPENDENT = 'reward_obs_dependent'
REWARD_TYPE_ACTION_OBS_DEPENDENT = 'reward_action_obs_dependent'
# obs types
OBS_TYPE_ZERO_OBS = 'obs_type_zero'
OBS_TYPE_RANDOM_PLUS_MINUS_1 = 'obs_type_random_plus_minus_1'
OBS_TYPE_GOAL_1 = 'obs_type_1_at_goal'


class DebugGym(gym.Env):
    def __init__(self. **kwargs):
        self.timestep_max = kwargs.get(CONFIG_KEY_TIMESTEP_MAX, 1)
        self.timestep_current = 0

        if CONFIG_KEY_REWARD_TYPE in kwargs:
            self.reward_type = kwargs[CONFIG_KEY_REWARD_TYPE]
        else:
            self.reward_type = REWARD_TYPE_1

        # define action type
        if CONFIG_KEY_ACTION_SIZE in kwargs:
            self.action_size = kwargs[CONFIG_KEY_ACTION_SIZE]
        else:
            self.action_size = 1
        self.action_space = gym.spaces.Discrete(self.action_size)

        self.obs_type = kwargs.get(CONFIG_KEY_OBS_TYPE, OBS_TYPE_ZERO_OBS)
        self.observation_space = self._define_obs_space(self.obs_type)

    def reset(self):
        self.timestep_current = 0
        is_goal = self._calculate_done(self.timestep_current, self.timestep_max)
        obs = self._calculate_obs(self.obs_type, is_goal)

        return obs

    def step(self, action):
        self.timestep_current += 1
        obs = self._define_obs_space(self.obs_type)
        done = self._calculate_done(self.timestep_current, self.timestep_max)
        reward = self._calculate_reward(action, obs, done, self.reward_type)
        info = {}

        return obs, reward, done, info
    
    def render(self):
        pass

    def _define_obs_space(self, obs_type):
        return gym.spaces.Discrete(1)

    def _calculate_done(self, ts_current, ts_max):
        if ts_current >= ts_max:
            return True
        return False
    
    def _calculate_obs(self, obs_type, is_goal):
        if obs_type == OBS_TYPE_ZERO_OBS:
            return np.array(0, dtype=np.int32)
        elif obs_type == OBS_TYPE_RANDOM_PLUS_MINUS_1:
            if np.random.rand() <= 0.5:
                return np.array(-1, dtype=np.int32)
            else:
                return np.array(1, dtype=np.int32)
        elif obs_type == OBS_TYPE_GOAL_1:
            if is_goal:
                return np.array(1, dtype=np.int32)
            else:
                return np.array(0, dtype=np.int32)
        else:
            raise Exception("Invalid obs type")

    def _calculate_reward(self, action, obs, is_goal, reward_type):
        if reward_type == REWARD_TYPE_1:
            # always return 1
            reward = 1.0
        elif reward_type == REWARD_TYPE_GOAL_1:
            # return 1.0 at the goal, else 0
            if is_goal:
                reward = 1.0
            else:
                reward = 0.
        elif reward_type == REWARD_TYPE_ACTION_DEPENDENT:
            if action == 1:
                reward = 1.0
            elif action == 0:
                reward = -1.0
            else:
                raise Exception('Invalid action for action dependent reward type')
        elif reward_type == REWARD_TYPE_OBS_DEPENDENT:
            if obs == 1:
                reward = 1.0
            elif obs == -1:
                reward = -1.0
            else:
                raise Exception('Invalid action for obs dependent reward type')
        elif reward_type == REWARD_TYPE_ACTION_OBS_DEPENDENT:
            if action == 1 and obs == 1:
                reward = 1.0
            elif action == 0 and obs == -1:
                reward = 1.0
            else:
                reward = -1.
        else:
            raise Exception('Invalid reward type')

        return reward