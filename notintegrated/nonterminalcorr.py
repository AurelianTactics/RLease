'''
given a traj of vfs and rewards, find the corr
'''

from scipy.stats.stats import pearsonr

def non_terminal_corr(df, corr_step=3):
  print("overall corr", df['reward'].corr(df['vf']))
  # non zero reward corr
  df_nz = df[df.reward != 0].copy()
  print("non zero corr", df_nz['reward'].corr(df_nz['vf']))
  # corr x steps removed
  # to do loop through values up to x
  for i in range(corr_step):
    x = i
    # size check
    if df.shape[0] <= x:
      continue
    reward_array = df.reward.to_numpy(copy=True)
    reward_array = reward_array[x:]
    vf_array = df.vf.to_numpy(copy=True)
    vf_array = vf_array[:-x]
    # pearson corr code
    print(i, pearsonr(reward_array, vf_array))

  # non zero thing as above
  for i in range(corr_step):
    x = i
    # size check
    if df_nz.shape[0] <= x:
      continue
    # to do: check this code
    # can only take rewards that have a vf x many steps earlier
    df_nz_copy = df_nz.copy()
    df_nz_copy = df_nz_copy[df_nz.index > x]
    reward_array = df_nz_copy.reward.to_numpy(copy=True)
    index_list = df_nz_copy.index.to_list()
    # now shift the indices
    index_list = index_list - x
    # get vf by that shift
    vf_array = df.loc[df.index[index_list], 'vf'].to_numpy(copy=True)
    # pearson corr code
    print(i, pearsonr(reward_array, vf_array))