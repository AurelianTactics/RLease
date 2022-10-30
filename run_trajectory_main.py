#!/usr/bin/env python3

'''
Given an agent and env, run a trajectory and collect stats

to do

longer term
this should work with other things that run on an env loop.
    not exactly sure but like if you wanted to do like self play stats in the loop do that here as well
'''

import yaml
import argparse
import trajectory_stats
import rlease_agent
import rlease_env

parser = argparse.ArgumentParser()
parser.add_argument("--yaml-filepath", type=str, default="", help="load arguments from this file", required=True)
args = parser.parse_args()

def main():
    if args.yaml_filepath != "":
        with open(args.yaml_filepath, "r") as stream:
            try:
                yaml_dict = yaml.safe_load(stream)
                args_dict = yaml_dict['settings']
            except yaml.YAMLError as exc:
                print(exc)
                raise Exception("yaml args not loaded")
    else:
        print("ERROR Not yet implemented, load from yaml file, quitting")
        quit()
        args_dict = make_args_dict(args)

    rlease_agent = RLeaseAgent(args_dict)
    rlease_env = load_env(args_dict)

    trajectory_stats.get_trajectory_stats(agent, env, args_dict)


if __name__ == "__main__":
    main()