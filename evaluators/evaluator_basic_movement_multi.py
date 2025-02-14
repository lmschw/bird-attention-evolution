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
    def __init__(self, data_file_paths, data_labels, base_save_path, max_iters=None):
        self.data_file_paths = data_file_paths
        self.data_labels = data_labels
        self.base_save_path = base_save_path
        self.max_iters = max_iters

        self.data = self.load_data()

    def load_data(self):
        data = []
        for path in self.data_file_paths:
            data.append(logger.load_log_data(path, max_iters=self.max_iters))
        return data

    def evaluate_and_visualise(self, metrics):
        for metric in metrics:
            data = []
            for i in range(len(self.data)):
                subdata = self.data[i]
                if metric == Metrics.COHESION:
                    data.append(mf.evaluate_cohesion(data=subdata))
                if metric == Metrics.ORDER:
                    data.append(mf.evaluate_order(data=subdata))   
            match metric:
                case Metrics.COHESION:
                    eplot.create_line_plot(data=data, labels=self.data_labels)
                case Metrics.ORDER:
                    eplot.create_line_plot(data=data, labels=self.data_labels)
            eplot.plot(metric=metric)
