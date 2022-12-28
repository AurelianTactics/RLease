'''
Utilities for various scripts
'''

from os.path import expanduser
import pandas as pd
import pickle
from pathlib import Path

def get_save_path_and_make_save_directory(file_name, file_dir = "/RLEase_results/"):
    """
    Makes directory and returns save path
    :param file_name: file name to save to
    :param file_dir: directory to put file of file name in
    :return: a file path represented bya  string
    """
    #save_dir = expanduser('~') + file_dir
    save_dir = file_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = save_dir + file_name

    return save_path


def save_pickle_file(file_name, save_object):
    with open(filename, 'wb') as handle:
        pickle.dump(save_object, handle, protocol=pickle.HIGHEST_PROTOCOL)

def make_directory(file_dir):
    Path(file_dir).mkdir(parents=True, exist_ok=True)


def save_multiagent_results(file_name, save_object, file_dir="/rlease_results/"):
  save_path = get_save_path_and_make_save_directory(file_name, file_dir)
  pickle_file(save_path, save_object)


def print_rankings(rankings_dict):
  # dict for agents with elo or equivalent
  # turn into df and print
  print(pd.DataFrame(rankings_dict))

def print_head_to_head_table(stats_dict):
  pass

def print_multiagent_eval(results_dict, rankings_dict, args_dict):
    '''
    results_dict: outcome of games
    args_dict: from yaml/argparse, specifications for what to display
    '''
    # print league table
    print_league_table(rankings_dict)
    # print head to head
    if 'is_print_hth' in args_dict:
      print_head_to_head_table(results_dict)
    #arg to print head to head
    pass


# print WLD dict
# placeholder
def print_rankings_wld_dict(w_dict, a_dict, r_dict):
    GAME_RESULT_WIN = "W"
    GAME_RESULT_DRAW = "D"
    GAME_RESULT_LOSS = "L"
    GAME_RESULT_KEY = "result"
    GAME_HOME_KEY = "is_home"
    GAME_SCORE_KEY = "score"
    GAME_SCORE_OPP_KEY = "opponent_score"
    agent_names = list(a_dict.keys())

    if r_dict is not None:
        print("{} --- Mu --- Sigma".format("RANKINGS".ljust(40)))
        # import pdb; pdb.set_trace()
        for an in agent_names:
            print("{} --- {} --- {}".format(an.ljust(40), round(r_dict[an].mu,2), round(r_dict[an].sigma,2) ) )
    print("---------------------------------------------------------")
    print("{} --- Won / Loss / Draw --- Score Avg. / Opp Score Avg".format("STANDINGS".ljust(40)))
    for an in agent_names:
        avg_td = np.round(np.mean(w_dict[an][GAME_SCORE_KEY]),3)
        opp_avg_td = np.round(np.mean(w_dict[an][GAME_SCORE_OPP_KEY]), 3)
        print("{} --- {} / {} / {} ----- {} / {} ".format(an.ljust(40), str(w_dict[an][GAME_RESULT_WIN]).ljust(3),
                                                          str(w_dict[an][GAME_RESULT_LOSS]).ljust(3),
                                           str(w_dict[an][GAME_RESULT_DRAW]).ljust(3),
                                                          str(avg_td).ljust(5), str(opp_avg_td).ljust(5)))

# placeholder
def print_head_to_head_dict(hth_dict, is_print_hth):
    if not is_print_hth:
        return
    # keys are the matchup of team_0_team_1, sorted in alphabetical order
    # values are a list of hth_results
    # hth_results are a dict for each entry in the list
    for k, v in hth_dict.items():
        team_0 = v[0]["team_0"]
        team_1 = v[0]["team_1"]
        num_games = len(v)

        win_per_team_0 = 0
        win_per_team_1 = 0
        draw_per = 0
        avg_score_team_0 = 0
        avg_score_team_1 = 0
        avg_casualties_team_0 = 0
        avg_casualties_team_1 = 0
        for i in range(num_games):
            win_per_team_0 += v[i]['team_0_win']
            win_per_team_1 += v[i]['team_1_win']
            if v[i]['team_0_win'] == 0 and v[i]['team_1_win'] == 0:
                draw_per += 1
            avg_score_team_0 += v[i]['team_0_score']
            avg_score_team_1 += v[i]['team_1_score']
            # avg_casualties_team_0 += v[i]['team_0_casualties']
            # avg_casualties_team_1 += v[i]['team_1_casualties']

        win_per_team_0 = np.round(win_per_team_0 / num_games, 3)
        win_per_team_1 = np.round(win_per_team_1 / num_games, 3)
        draw_per = np.round(draw_per / num_games, 3)
        avg_score_team_0 = np.round(avg_score_team_0 / num_games, 3)
        avg_score_team_1 = np.round(avg_score_team_1 / num_games, 3)
        # avg_casualties_team_0 = np.round(avg_casualties_team_0 / num_games, 3)
        # avg_casualties_team_1 = np.round(avg_casualties_team_1 / num_games, 3)

        print("{} v {} | W%: {} v {} | D%: {} | pts: {} v {} | cas: {} v {} ".format(
            team_0, team_1, win_per_team_0, win_per_team_1, draw_per, avg_score_team_0, avg_score_team_1,)
            avg_casualties_team_0, avg_casualties_team_1))
        )