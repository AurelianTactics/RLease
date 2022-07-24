#!/usr/bin/env python3

import numpy as np
import argparse
import pickle
from collections import OrderedDict
from trueskill import Rating, quality_1vs1, rate_1vs1
from rlease_utils import print_league_table, print_head_to_head, save_multiagent_results
import yaml

'''
Tools to help analyze the performance of agents in multi-agent zero sum games  
'''

'''
working
dicts I want
agent_dict: list of agents. used to determine match ups and agent specific info
results: w, l, score, various other match specific stats
hth dict: could be combined with results, lot of overlap
rankings: scored based on true skill or whatever. needed because this can help determine match up distribution
stats: overall stats, could just be meta organized from results


last action:
  need to do play and print functions but kind of already have those on linux


conceptionally,
want an outer loop that takes agents, plays them against each other, gets results, display results

I'm thinking want an abstract class, then examples for specific classes since different envs, algs, and agents will be loaded and played against each other in different ways

Taking agents:
  load from json file or dict I think
    agent name, load dir, SP steps/version (then classess can do optional)
  Then class specific load
    abstract function that passes this

Play them against each other
  the trueskill loop, though trueskill can be a mode
  get trueskill, W-L-D, score, other stats from it

Display results
  print table
  head to head for each type
  order by trueskill with SP steps/version
  way to zoom in on each type
  plots

so minimal version of this is above with display results being minimal


to do
should be able to parallelize: multiple versions of script or parallelize games
  maybe make just this script write results do meta stuff elsewhere
this runs but might want a class to stub out certain envs
some hard coded keys to fill in with constants
fill out the args and the args in the yaml dict

saving and loading
smarter save and load
better print



save different outputs
wandb, tensorboard, pandas, logging
seed everything
more eval types: ELO, None, Trueskill 2 etc 
sampling type
zero sum or not

more effective way to load
more optional args to add
scan dir for more opponents or opponent descriptions from a json or python dict or something

way to tell if loop where unstable learning
  some sort of version number, ideally later versions are better
  tracking elo?
condition games on some variable to see if learning betterworse in some ways


'''


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--config-file",
        default=None,
        type=str,
        help="If specified, load arguments from yaml file specified by this path. Overrides all other argparse arguments.",
    )
    parser.add_argument('project-name', type=str, default='multiagent_zero_sum_analysis',
                        help='Name of project. Used in save file')
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for environment if applicable. If set to 0 defaults to None seed")
    parser.add_argument("--load-path", type=str, default="", help="Path to load results from")
    parser.add_argument("--print-and-quit", type=int, default=0, help="Load results, print them and quit")
    parser.add_argument("--num-games", type=int, default=1)
    parser.add_argument("--eval-type", type=str, default='trueskill', help="evaluation method to use", choices=['trueskill'])
    parser.add_argument("--eval-path", type=str, default="", help="Path to load results from")

    return parser


def print_results():
    pass


def run(args, parser):
    # Get arguments
    if args.config_file:
        with open(args.config_file) as f:
            arg_dict = yaml.safe_load(f)
    else:
        arg_dict = {
            # to do, fill in args from dict
            'agents_path': args.agents_path,

        }

    # load agents: either inputted directly in yaml or a yaml file to load or a directory to scan for agent files
    if 'agents_dict' in arg_dict:
        pass
    elif 'agents_path' in arg_dict:
        pass
    elif 'agent_file' in arg_dict:
        agent_dict = yaml.safe_load(args_dict['agent_file'])
    else:
        print("ERROR: No agents specified to load, quitting")
        quit()

    # load prior results dict if key
    if 'results_filepath' in arg_dict:
      prior_results_dict = yaml.safe_load(arg_dict['results_filepath'])
    else:
      prior_results_dict = None 

    # create new dicts for new agents if necessary

    # play agents against each other
    results_dict, rankings_dict = play_game(args_dict, agent_dict, prior_results_dict)

    # display results
    # league table sorted by skill, arg to show head to head stats, args/env specific for more skills
    # show skill, mu, WLD, score
    # I'm thinking just do pandas
    print_multiagent_eval(results_dict, results_dict[RANKINGS_KEY])

    # save results
    save_multiagent_results(TBD, TBD)


def add_new_agents(agent_dict, prior_results_dict):
    # iterate through new agents adding them to any dicts
    if prior_results_dict is None:
        agent_dict = create_new_results_dict(agent_dict)
    else:
        agent_dict = add_new_agents_to_existing_dict(agent_dict, prior_results_dict[AGENT_DICT_KEY], prior_results_dict[STATS_DICT_KEY],
          prior_results_dict[RESULTS_DICT_KEY], prior_results_dict[WLD_DICT_KEY])

    return agent_dict
    

def play_game(args_dict, agent_dict, prior_results_dict):
    # create new agent dict or add agents to existing agent dict
    agent_dict = add_new_agents(agent_dict, prior_results_dict)
    
    # iterate through opponent combinations
    agent_names = list(agent_dict.keys())
    for i in range(len(agent_names)-1):
        current_agent_name = agent_names[i]
        for j in range(i+1, len(agent_names)):
            opp_agent_name = agent_names[j]
            # check if these bots have already played
            if check_if_already_played(match_up_dict, current_agent_name, opp_agent_name):
                continue
            # play args many games
            for _ in range(args.num_games):
                # game_number used as key for some dicts
                game_number += 1
                # special args and stats for different envs 
                temp_stats_dict, temp_hth_dict = get_game_result(game_number) # env specific
                # update dicts with results
                #rankings, wl, head to head, if already played
                update_dicts_with_results(temp_stats_dict, temp_hth_dict, game_number)


def update_dicts_with_results():
    pass 

def get_game_result(args):
    pass

# when loading from prior results, need to add new keys to dictionaries
def add_new_agents_to_existing_dict(temp_agents_dict, args_dict, a_dict, s_dict, r_dict, w_dict):
    agent_names = list(temp_agents_dict.keys())
    # game numbers are the keys
    # incremented before an entry is added to the dict 
    game_number = max(list(s_dict.keys()))
    for an in agent_names:
        if an in a_dict or an in s_dict or an in r_dict or an in w_dict:
            print("WARNING: agent already exists {} ".format(an))
            # to do: maybe add option to make the agent again with a timestamp?
            continue
            # an = an + "_{}".format(int(time.time()))
            # print("adding timestamp to agent name {}".format(an))

        a_dict[an] = temp_agents_dict[an]
        r_dict = get_eval_type(args_dict, r_dict)
        w_dict[an] = {
                GAME_RESULT_WIN: 0,
                GAME_RESULT_LOSS: 0,
                GAME_RESULT_DRAW: 0,
                GAME_SCORE_KEY: [],
                GAME_SCORE_OPP_KEY: [],
                GAME_HOME_KEY: [],
            }

    return a_dict, s_dict, r_dict, w_dict, game_number

def create_new_results_dict(agent_dict):
    rankings_dict = {}
    wld_dict = {}
    for k, v in agent_dict.items():
        rankings_dict[k] = get_eval_type(args_dict, r_dict)
        wld_dict[k] = {
            GAME_RESULT_WIN: 0,
            GAME_RESULT_LOSS: 0,
            GAME_RESULT_DRAW: 0,
            GAME_SCORE_KEY: [],
            GAME_SCORE_OPP_KEY: [],
            GAME_HOME_KEY: [],
        }
    stats_dict = {}
    head_to_head_dict = {}

def get_eval_type(args_dict, r_dict):
    if args.eval_type == EVAL_TYPE_TRUESKILL:
      r_dict[an] = Rating() # new rating
    else:
      r_dict[an] = Rating() # new rating

def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)


if __name__ == "__main__":
    main()