'''
Build Trajectories with stats for debugging and checking on performance of things like
basic stats: reward and episode length
Value Function
Action Logits

make into a class?
    class vs. function
    i'm thinking function for now
        agents passed as a class so can call soemthing like get action
        if I ever need to functionize i'll have a better idea of that later

LLL
better ways to parallelize
uses for debugging
stat performance
wandb logging
mlflow (or other alternative) logging?
eval code: not just mean len and reward but min and max probably helpful as well
'''

import rlease_trajectory_builder
import rlease_trajectory_plot
import time
from rlease_utils import save_pickle_file

def get_trajectory_stats(agent, env, args_dict, experiment_save_dir=None):
    '''
    Get stats from a collected trajectory and log to trajectory_builder object
    Can log every step to one builder or the end stats to another builder

    Parameters
    ----------
    agent:
        RL agent that is invoked to produce actions

    env: gym.env object

    args_dict: dict
        dictionary of arguments
        to do: list and explain the args

    experiment_save_dir: str, None
        directory to save results to
    
    Returns
    -------
    None
    
    '''

    # get arguments
    args_dict = args_dict['trajectory_args']
    project_name = args_dict.get('project_name', '')
    num_episodes = args_dict.get('num_episodes', 10)
    save_interval = args_dict.get('save_interval', 100)
    logging_args = args_dict.get('logging_args', None)
    is_log_obs = args_dict.get('is_log_obs', False)
    is_log_info_dict_all = args_dict.get('is_log_info_dict_all', False)
    log_info_dict = args_dict.get('log_info_dict', None)
    is_log_trajectory = args_dict.get('is_log_trajectory', False)
    is_eval_agent_basic = args_dict.get('is_eval_agent_basic', True)
    is_log_action_counts = args_dict.get('is_log_action_counts', True)
    seed = args_dict.get('seed', 0)

    if experiment_save_dir is None:
        experiment_save_dir = 'RLease_results/{}_{}/'.format(project_name, int(time.time()))

    # constants
    TRAJECTORY_SAVE_FP = "trajectory_results"
    EVAL_SAVE_FP = "eval_basic_results"
    ACTION_FP = "action_basic_results"
    ACTION_COUNTER_FP = "action_dict_counter.pkl"

    # create builders for logging various stats
    if is_log_trajectory:
        trajectory_builder = rlease_trajectory_builder.TrajectoryBuilder(save_path=experiment_save_dir,
                                                                         project_name=project_name)

    if is_eval_agent_basic:
        eval_agent_basic_builder = rlease_trajectory_builder.TrajectoryBuilder(save_path=experiment_save_dir,
                                                                               project_name=EVAL_SAVE_FP)

    if is_log_action_counts:
        action_builder = rlease_trajectory_builder.TrajectoryBuilder(save_path=experiment_save_dir,
                                                                     project_name=ACTION_FP)
        action_dict_counter = {}

    # run episodes and get results
    for ep in range(num_episodes):
        done = False
        obs = env.reset()
        timestep = 0
        reward_total = 0.
        info = {}

        while not done:
            action, vf = agent.get_action_vf(obs, info)

            next_obs, reward, done, info = env.step(action)
            reward_total += reward

            if is_log_trajectory:
                # trajectory stats collected every time steps
                step_dict = {
                    'episode': ep,
                    'timestep': timestep,
                    'action': action,
                    'value_function_output': vf,
                    'reward': reward,
                    'done': done,
                }

                if is_log_obs:
                    step_dict['obs'] = obs
                    step_dict['next_obs'] = next_obs

                    step_dict = add_info_to_dict(step_dict, info, is_log_info_dict_all, log_info_dict)
                    trajectory_builder.add_values(values=step_dict)

            if is_log_action_counts:
                # action stats
                action_dict_counter = increment_action_counter(action_dict_counter, action)
                action_dict = {
                    'action': action
                }
                action_builder.add_values(values=action_dict)

            if done:
                # if episode is done
                if is_eval_agent_basic:
                    # end of episode stats for basic evaluation
                    episode_dict ={
                        'reward_total': reward_total,
                        'timestep_end': timestep,
                    }

                    episode_dict = add_info_to_dict(episode_dict, info, is_log_info_dict_all, log_info_dict)
                    eval_agent_basic_builder.add_values(values=episode_dict)

                if ep % save_interval == 0 and ep > 0:
                    # save and empty the trajectory_builder objects periodically to prevent from getting too large
                    # this will affect the end of episode basic information
                    if is_log_trajectory:
                        trajectory_builder.save_and_clear()
                    if is_eval_agent_basic:
                        eval_agent_basic_builder.save_and_clear()
                    if is_log_action_counts:
                        action_builder.save_and_clear()
            else:
                timestep += 1
                obs = next_obs

    # save and empty the trajectory_builder objects
    if is_log_trajectory:
        trajectory_builder.save_and_clear()

    if is_eval_agent_basic:
        # turn builder into data frame
        print('----- Eval Agent Basic Stats -----')
        df_eval = eval_agent_basic_builder.get_dataframe() # if builder had been saved and cleared early this will be incorrect
        print(df_eval.describe())

        # eval plots
        rlease_trajectory_plot.plot_episode_stats(df_eval, experiment_save_dir=experiment_save_dir, **args_dict)

        # save results
        eval_agent_basic_builder.save_and_clear()

    if is_log_action_counts:
        # print and log the actions
        print('----- Action Dict Counter -----')
        print(action_dict_counter)
        save_pickle_file(experiment_save_dir+ACTION_COUNTER_FP, action_dict_counter)

        # turn builder into dataframe and plot
        df_action = action_builder.get_dataframe() # if the builder had been saved and cleared early this will be incorrect
        print('----- Action Distribution Stats -----')
        print(df_action.describe())
        rlease_trajectory_plot.plot_action_distribution(df_action, experiment_save_dir)

        # save results
        action_builder.save_and_clear()

    return df_eval # idk if I should return this or no in general but using it in one specific case


def add_info_to_dict(input_dict, info_dict, is_log_info_dict_all, log_info_dict):
    '''
    Add additional items from info_dict to input_dict depending on the arguments

    to do: check for potential collisions from info dict with rlease defaults and rename accordingly
    '''

    if is_log_info_dict_all:
        for k in info_dict.keys():
            input_dict[k] = info_dict[k]
    else:
        if log_info_dict is not None:
            for k in log_info_dict.keys():
                if k in info_dict.keys():
                    input_dict[k] = info_dict[k]

    return input_dict


def increment_action_counter(action_dict, action):
    if action in action_dict:
        action_dict[action] += 1
    else:
        action_dict[action] = 1

    return action_dict