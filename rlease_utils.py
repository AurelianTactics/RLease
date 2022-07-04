'''
Utilities for various scripts
'''

from os.path import expanduser
import pandas as pd

def get_save_path_and_make_save_directory(file_name, file_dir = "/rlease_results/"):
    """
    Makes directory and returns save path
    :param file_name: file name to save to
    :param file_dir: directory to put file of file name in
    :return: a file path represented bya  string
    """
    save_dir = expanduser('~') + file_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = save_dir + file_name
    return save_path


def pickle_file(file_name, save_object):
    with open(filename, 'wb') as handle:
        pickle.dump(save_object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_multiagent_results(file_name, save_object, file_dir="/rlease_results/"):
  save_path = get_save_path_and_make_save_directory(file_name, file_dir)
  pickle_file(save_path, save_object)


def print_league_table(stats_dict):
  #df =
  pass

def print_head_to_head_table(stats_dict):
  pass

def print_multiagent_eval(args_dict):
    # print overview

    #arg to print head to head
    pass

