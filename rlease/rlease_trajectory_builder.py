'''
Class for building trajectories
'''

import collections
from rlease_utils import save_pickle_file, get_save_path_and_make_save_directory
import time
from typing import Any, Dict, List
import pandas as pd


class TrajectoryBuilder:
    """
    store episodes of RL trajectories for later analysis
    store each row as a new dictionary value
    modelled after https://github.com/ray-project/ray/blob/master/rllib/evaluation/sample_batch_builder.py#L31
    """
    def __init__(self, save_path="/rlease_results/trajectory_builder/", project_name=""):
        self.buffers: Dict[str, List] = collections.defaultdict(list)
        self.count = 0
        if project_name == "":
            project_name = "trajectory_{}".format(int(time.time()))
        self.save_path = get_save_path_and_make_save_directory(project_name, save_path)
        self.save_number = 0

    def add_values(self, **values: Any) -> None:
        """Add the given dictionary (row) of values to this batch."""

        for k, v in values.items():
            self.buffers[k].append(v)
        self.count += 1

    def save_and_clear(self):
        if self.count > 0:
            # save
            file_path = self.save_path + "_{}".format(self.save_number)
            pickle_file(file_path, self.buffers)
            self.save_number += 1
            # clear buffers
            self.buffers.clear()
            self.count = 0

    def get_dataframe(self):
        '''
        turn a builder object into a datframe

        Returns
        -------
        df_out: pandas DataFrame
            DataFrame of self.buffers
        '''
        try:
            if 'values' in self.buffers.keys():
                df_out = pd.DataFrame(self.buffers['values'])
            else:
                df_out = pd.DataFrame(self.buffers)
        except Exception as e:
            print("ERROR: unable to create dtaframe from builder, returning empty df ", e)
            df_out = pd.DataFrame()

        return df_out