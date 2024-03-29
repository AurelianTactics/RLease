#!/usr/bin/env python3

'''
Given an agent and env, run a trajectory and collect stats


further improvements
this module works with others
better plotting
wandb integration
tb integration
abstract env class
abstract agent class
more examples
more envs
more rl libraries 
'''

import yaml
import argparse
import trajectory_stats
from rlease_agent import RLeaseAgent
from rlease_env import RLeaseEnv


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

    agent = RLeaseAgent(args_dict)
    env = RLeaseEnv(args_dict)

    trajectory_stats.get_trajectory_stats(agent, env, args_dict)


if __name__ == "__main__":
    main()
