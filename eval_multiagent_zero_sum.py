#!/usr/bin/env python3

import numpy as np
import argparse
import pickle
from collections import OrderedDict
from trueskill import Rating, quality_1vs1, rate_1vs1
from rlease_utils import print_league_table, print_head_to_head, save_multiagent_results
import yaml
from multi_agent_zero_sum import MultiAgentZeroSum

def create_arg_dict(args):
    arg_dict = {
            # to do, fill in args from dict
            'agents_path': args.agents_path,

        }
    return arg_dict

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


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file) as f:
            arg_dict = yaml.safe_load(f)
    else:
        create_arg_dict(args)
    # unclear to me if I want to make a MAZS object and feed args into it for specifics inherit from MAZS classes
    MAZS = MultiAgentZeroSum(arg_dict)
    run(args, parser)




if __name__ == "__main__":
    main()