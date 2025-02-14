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

class EvaluatorBasicMovement:
    def __init__(self, data_file_path, base_save_path, max_iters=None):
        self.data_file_path = data_file_path
        self.base_save_path = base_save_path
        self.max_iters = max_iters
        self.data = logger.load_log_data(self.data_file_path, max_iters=max_iters)

    def evaluate_and_visualise(self, metric=None):
        if metric in [None, Metrics.COHESION]:
            data = mf.evaluate_cohesion(data=self.data)
            eplot.create_line_plot(data=data, labels=["average distance to centroid"])
            eplot.plot(metric=Metrics.COHESION)
        if metric in [None, Metrics.ORDER]:
            data = mf.evaluate_order(data=self.data)
            eplot.create_line_plot(data=data, labels=["global order"])
            eplot.plot(metric=Metrics.ORDER)
