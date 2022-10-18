'''
Build Trajectories with stats for debugging and checking on performance of things like
Value Function
Action Logits

DONE take args
DONE run loop
DONE log stats
DON save stats

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
    project_name = args_dict.get('project_name', '')
    num_episodes = args_dict.get('num_episodes', 10)
    save_interval = args_dict.get('save_interval', 100)
    logging_args = args_dict.get('logging_args', None)
    is_log_obs = args_dict.get('is_log_obs', False)
    log_info_dict = args_dict.get('log_info_dict', None)

    trajectory_builder = TrajectoryBuilder(project_name=project_name)

    for ep in range(num_episodes):
        done = False
        obs = env.reset()
        timestep = 0

        while not done:
            action, vf = agent.get_action_vf(obs)

            next_obs, reward, done, info = env.step(action)

            # build dictionary of stats wanted to store
            # to do: functionize, add arguments for storing more/less
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
                if ep % save_interval == 0 and ep > 0:
                    trajectory_builder.save_and_clear()
            else:
                timestep += 1
                obs = next_obs

    trajectory_builder.save_and_clear()