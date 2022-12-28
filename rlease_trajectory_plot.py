'''
stand in for plotting trajectories

will abstract at some point

plot action distribution
plot VF traj
save the plots
wrapper to call the plots based on args

to do:
action distribution by differetn subsets (over time, subsets etc)
VF traj for multiple ones,a ggregate by relation to end point, difference from reward etc
make metrics for this rather than have to look at plots for vf
    like x steps from end how close
make metrics for this rather than have to look at action dist for actions
interactive plots
better method than saving to file, something more like tensorplot
non plots things but some numerics (5 number summary of action dist)
  vf vs, reward as a %

'''
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from pathlib import Path

def plot_reward_vs_vf(df_input, fp, num_plots, is_filter_zero_reward=False, vf_col='vf', 
  reward_col='reward', y_min=-1.2, y_max=1.2):
    plt.figure()
    df_in = df_input[["episode", "timestep", vf_col, reward_col]]
    saved_plots = 0
    time_int = int(time.time())
    episodes = df_in['episode'].drop_duplicates()
    for i in range(len(episodes)):
        e = episodes[i]
        df = df_in[df_in['episode'] == e].copy()
        df.timestep = df.timestep - df.timestep.min()
        fig, ax = plt.subplots()
        ax.set(ylim=(y_min, y_max), xlabel='timestep', title="Episode {}".format(e))

        if is_filter_zero_reward:
            # remove any rewards are zero for plotting
            df_r = df.copy()
            df_r.loc[df_r[reward_col] == 0, 'reward'] = np.NaN
            xtick_index = df[df[reward_col] != 0].timestep
        else:
            df_r = df
            ax.set_xticks([])
            xtick_index = np.arange(0, df.timestep.max(), 50)

        if df_r.shape[0] > 0:
            sns.pointplot(y=reward_col, x=df_r.timestep, data=df_r, ax=ax, color='orange')
            # ax.set_xticks([df_r.index])

        sns.lineplot(y=vf_col, x=df.timestep, data=df, ax=ax)
        ax.set_xticks(xtick_index)
        ax.set_ylabel('{} / {}'.format(vf_col, reward_col))
        plt.savefig(fp + "vf_plot_{}.png".format(saved_plots))
        if saved_plots > num_plots:
            break
    # ax.set_xticks([]) based on the line plot or based on the rewards
    plt.clf()

# plot_reward_vs_vf(df_2)
# plot_reward_vs_vf(df_2, is_filter_zero_reward=True)

def plot_action_distribution(df_input, experiment_save_dir=None):
    '''
    Save a plot of action distributions
    '''
    experiment_save_dir = get_experiment_save_dir(experiment_save_dir)

    plt.figure()
    sns.displot(data=df_input, x='action', kind='hist', kde=True)
    plt.title('Action Distribution')
    plt.xlabel('actions')
    #plt.ylabel()
    plt.savefig(experiment_save_dir + "action_plot.png")
    plt.clf()


def plot_difference_between_cols(df_input, col_1, col_2, experiment_save_dir=None):
    '''
    Save a plot of difference between columns

    to do
        args for plots
    '''
    df = df_input.copy()
    new_col = col_1 + "_minus_" + col_2
    df[new_col] = df[col_1] - df[col_2]
    experiment_save_dir = get_experiment_save_dir(experiment_save_dir)

    plt.figure()
    sns.histplot(data=df, x=new_col, bins=20, binwidth=0.01, stat="percent")
    plt.title('Difference Plot {}'.format(new_col))
    plt.xlabel(new_col)
    #plt.ylabel()
    plt.savefig(experiment_save_dir+"difference_between_columns_{}.png".format(new_col))
    plt.clf()


def plot_trajectory(df_input, **kwargs):
    time_int = int(time.time())
    fp = 'trajectory_plot_{}'.format(time_int)
    plot_reward_vs_vf(df_input.copy(), fp=fp, num_episodes=10)
    plot_action_distribution(df_input, fp)


def plot_episode_stats(df_input, experiment_save_dir=None, **kwargs):
    '''
    Basic plots of end episode states
    Bunch of to dos, in the roadmap for expanding
        add labels, test, etc

    Parameters
    ----------
    df_input: pandas DataFrame
        Contains end of episode stats

    experiment_save_dir: str, None
        Directory to save files to. All fiels from teh same experiment should share this directory

    Returns
    -------
    None

    '''

    experiment_save_dir = get_experiment_save_dir(experiment_save_dir)

    running_average_n = kwargs.get('running_average_n', 100)
    if df_input.shape[0] < running_average_n:
        running_average_n = min(1, df_input.shape[0])

    plt.figure()
    # reward plot running average
    #sns.lineplot(y=df.rolling(window=running_average_n)['reward_total'].mean())
    plt.plot(df_input.rolling(window=running_average_n)['reward_total'].mean()) # to do better plot for this
    plt.title('Running Mean of Reward --- {} episodes'.format(running_average_n))
    plt.xlabel('Episode')
    plt.ylabel('Running Reward Mean')
    plt.savefig(experiment_save_dir +"episode_reward_plot.png")
    plt.clf()

    plt.figure()
    # episode length plot running average
    # sns.lineplot(y=df.rolling(window=running_average_n)['reward_total'].mean())
    plt.plot(df_input.rolling(window=running_average_n)['timestop_end'].mean())  # to do better plot for this
    plt.title('Running Mean of Episode Length --- {} episodes'.format(running_average_n))
    plt.xlabel('Episode')
    plt.ylabel('Running Episode Length Mean')
    plt.savefig(experiment_save_dir+"episode_length_plot.png")
    plt.clf()


def get_experiment_save_dir(experiment_save_dir=None):
    '''

    '''
    if experiment_save_dir is None:
        experiment_save_dir = 'RLease_results/{}_{}/'.format("default_project_name", int(time.time()))

    Path(experiment_save_dir).mkdir(parents=True, exist_ok=True)

    return experiment_save_dir