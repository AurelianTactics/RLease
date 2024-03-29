#!/usr/bin/env python3

'''
NOT IMPLEMENTED
Given an agent and env, run basic evaluation
for now folding into run_trajectory_main.py
'''

import yaml
import argparse
import evaluate_single_agent
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

    rlease_agent = RLeaseAgent(args_dict)
    rlease_env = RLeaseEnv(args_dict)

    evaluate_single_agent.evaluate(agent, env, args_dict)


if __name__ == "__main__":
    main()