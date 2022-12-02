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
'''

import trajectory_builder

def get_trajectory_stats(agent, env, args_dict):
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
    log_info_dict = args_dict.get('log_info_dict', None)
    is_log_trajectory = args_dict.get('is_log_trajectory', False)
    is_eval_agent_basic = args_dict.get('is_eval_agent_basic', True)
    
    if is_log_trajectory:
        trajectory_builder = TrajectoryBuilder(project_name=project_name)

    if is_eval_agent_basic:
        eval_agent_basic_builder = TrajectoryBuilder(project_name=project_name+"_eval_basic")
    
    for ep in range(num_episodes):
        done = False
        obs = env.reset()
        timestep = 0
        reward_total = 0.

        while not done:
            action, vf = agent.get_action_vf(obs)

            next_obs, reward, done, info = env.step(action)
            reward_total += reward

            if is_log_trajectory:
                # build dictionary of stats wanted to store
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

                if log_info_dict is not None:
                    for k in log_info_dict.keys():
                        if k in info:
                            step_dict[k] = info[k]

                trajectory_builder.add_values(**step_dict)

            if done:
                if is_eval_agent_basic:
                    episode_dict ={
                        'reward_total': reward_total,
                        'timestep_end': timestep,
                    }

                    if log_info_dict is not None:
                        for k in log_info_dict.keys():
                            if k in info:
                                episode_dict[k] = info[k]

                    eval_agent_basic_builder.add_values(**episode_dict)

                if ep % save_interval == 0 and ep > 0:
                    # save and empty the trajectory_builder objects
                    if is_log_trajectory:
                        trajectory_builder.save_and_clear()
                    if is_eval_agent_basic:
                        eval_agent_basic_builder.save_and_clear()
            else:
                timestep += 1
                obs = next_obs

    # save and empty the trajectory_builder objects
    if is_log_trajectory:
        trajectory_builder.save_and_clear()
    if is_eval_agent_basic:
        eval_agent_basic_builder.save_and_clear()