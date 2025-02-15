import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluators.metrics import Metrics
import evaluators.metrics_functions as mf
import loggers.logger_agents as logger
import evaluators.evaluation_plotting as eplot

"""
Evaluation plots:
    - cohesion
    - order
"""

class EvaluatorBasicMovementMulti:
    def __init__(self, data_file_paths, data_labels, base_save_path, animal_types, max_iters=None):
        self.data_file_paths = data_file_paths
        self.data_labels = data_labels
        self.base_save_path = base_save_path
        self.animal_types = animal_types
        self.max_iters = max_iters

        self.data = self.load_data()

    def load_data(self):
        data = []
        for path in self.data_file_paths:
            data.append(logger.load_log_data(path, max_iters=self.max_iters))
        return data

    def evaluate_and_visualise(self, metrics, normalise_cohesion=False):
        for metric in metrics:
            data = []
            for i in range(len(self.data)):
                subdata = self.data[i]
                animal_type = self.animal_types[i]
                if metric == Metrics.COHESION:
                    data.append(mf.evaluate_cohesion(data=subdata, animal_type=animal_type, normalised=normalise_cohesion))
                    y_label = 'cohesion (width/average distance from centroid)'
                if metric == Metrics.ORDER:
                    data.append(mf.evaluate_order(data=subdata))   
                    y_label = 'global order'
            match metric:
                case Metrics.COHESION:
                    eplot.create_line_plot(data=data, labels=self.data_labels)
                case Metrics.ORDER:
                    eplot.create_line_plot(data=data, labels=self.data_labels)
            eplot.plot(metric=metric, base_save_path=self.base_save_path, x_label='timesteps', y_label=y_label)
