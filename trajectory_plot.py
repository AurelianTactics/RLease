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

def plot_reward_vs_vf(df_input, fp, num_plots, is_filter_zero_reward=False, vf_col='vf', 
  reward_col='reward', y_min=-1.2, y_max=1.2):
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

# plot_reward_vs_vf(df_2)
# plot_reward_vs_vf(df_2, is_filter_zero_reward=True)

def plot_action_distribution(df_input, fp):
    sns.displot(data=df_input, x='actions', kind='hist', kde=True)
    plt.savefig(fp + "action_plot.png")


def plot_trajectory(df_input, **kwargs):
    time_int = int(time.time())
    fp = 'trajectory_plot_{}'.format(time_int)
    plot_reward_vs_vf(df_input.copy(), fp=fp, num_episodes=10)
    plot_action_distribution(df_input, fp)


def plot_episode_stats(df_input, **kwargs):
    '''
    Plot mean reward and episode length
    Bunch of to dos, in the roadmap for expanding

    add labels, test, etc
    '''
    running_average_n = kwargs.get('running_average_n', 100)
    if df_input.shape[0] < running_average_n:
        running_average_n = min(10, df_input.shape[0])

    # reward plot running average
    sns.lineplot(y=df.rolling(window=running_average_n)['reward_total'].mean())
    plt.savefig(fp + "episode_reward_plot_{}.png".format(saved_plots))

    # episode length plot running average
    sns.lineplot(y=df.rolling(window=running_average_n)['timestep_end'].mean())
    plt.savefig(fp + "episode_length_plot_{}.png".format(saved_plots))