#!/usr/bin/env python3

import numpy as np
import argparse
import pickle
from collections import OrderedDict
from trueskill import Rating, quality_1vs1, rate_1vs1

'''
Tools to help analyze the performance of agents in multi-agent zero sum games
'''

'''
to do
make this a class that can be stubbed for specific examples

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
'''

parser = argparse.ArgumentParser()
parser.add_argument('project-name', type=str, default='multiagent_zero_sum_analysis',
                    help='Name of project. Used in save file')
parser.add_argument("--seed", type=int, default=0, help="Seed for environment if applicable. If set to 0 defaults to None seed")
parser.add_argument("--load-path", type=str, default="", help="Path to load results from")
parser.add_argument("--print-and-quit", type=int, default=0, help="Load results, print them and quit")
parser.add_argument("--num-games", type=int, default=1)

parser.add_argument("--eval-type", type=str, default="", help="Path to load results from")


args = parser.parse_args()


def print_results():
    pass


def main():
    pass

if __name__ == "__main__":
    main()