"""Module containing mechanism for calculating standard deviation between datasets.
"""

import glob
import os
import numpy as np

from inflammation import models, views

class CSVDataSource:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_inflammation_data(self):
        """Loads the inflamation data data from CSV files within a directory

        Returns
        ------
        list of 2D NumPy array with inflammation data
        """
        data_file_paths = glob.glob(os.path.join(self.data_dir, 'inflammation*.csv'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data CSV files found in path {self.data_dir}")
        data = map(models.load_csv, data_file_paths)
        return data

class JSONDataSource:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_inflammation_data(self):
        """Loads the inflamation data data from CSV files within a directory

        Returns
        ------
        list of 2D NumPy array with inflammation data
        """
        data_file_paths = glob.glob(os.path.join(self.data_dir, 'inflammation*.json'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data json files found in path {self.data_dir}")
        data = map(models.load_json, data_file_paths)
        return data


def analyse_data(data_source):
    """Calculates the standard deviation by day between datasets.

    Gets all the inflammation data from CSV files within a directory,
    works out the mean inflammation value for each day across all datasets,
    then plots the graphs of standard deviation of these means.
    
    Parameters
    ----------
    data_source: an object providing data

    Returns
    -------
    None
    """
    data = data_source.load_inflammation_data()

    daily_standard_deviation = models.compute_standard_deviation_by_day(data)
    
    return daily_standard_deviation
