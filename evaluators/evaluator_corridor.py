import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluators.evaluator_basic_movement import EvaluatorBasicMovement
from evaluators.metrics import Metrics
import evaluators.metrics_functions as mf
import loggers.logger_agents as logger
import evaluators.evaluation_plotting as eplot

"""
Evaluation plots:
    - cohesion
    - order
    - corridor distribution
    - success proportion
    - duration until the agents reach the end of the corridor
"""

class EvaluatorCorridor(EvaluatorBasicMovement):
    def __init__(self, data_file_path, base_save_path, animal_type, max_iters=None, 
                 corridor_centers=[], corridor_endpoints=[]):
        super().__init__(data_file_path=data_file_path,
                         base_save_path=base_save_path,
                         animal_type=animal_type,
                         max_iters=max_iters)
        self.corridor_centers = corridor_centers
        self.corridor_endpoints = corridor_endpoints

    def evaluate_and_visualise(self, metric=None):
        super().evaluate_and_visualise(metric=metric)
        if metric in [None, Metrics.CORRIDOR_DISTRIBUTION]:
            if len(self.corridor_centers) != 2:
                print("need to specify exactly two corridors upon instantiation to evaluate CORRIDOR_DISTRIBUTION")
            else:
                data = mf.evaluate_corridor_selection(data=self.data, corridor_centers=self.corridor_centers)
                eplot.create_pie_plot(data=data, labels=['same corridor', 'split'])
                xlim = plt.gca().get_xlim()
                ylim = plt.gca().get_ylim()
                eplot.plot(metric=Metrics.CORRIDOR_DISTRIBUTION, base_save_path=self.base_save_path, xlim=xlim, ylim=ylim)
        if metric in [None, Metrics.SUCCESS_PERCENTAGE]:
            if len(self.corridor_endpoints) != 2:
                print("need to specify the endpoint of the corridors upon instantiation to evaluate CORRIDOR_DISTRIBUTION")
            else:
                data = mf.evaluate_success_percentage(data=self.data, corridor_endpoints=self.corridor_endpoints)
                eplot.create_pie_plot(data=data, labels=['percentage agents got through', 'percentage agents left behind'])
                xlim = plt.gca().get_xlim()
                ylim = plt.gca().get_ylim()
                eplot.plot(metric=Metrics.SUCCESS_PERCENTAGE, base_save_path=self.base_save_path, xlim=xlim, ylim=ylim)
        if metric in [None, Metrics.DURATION]:
            if len(self.corridor_endpoints) != 2:
                print("need to specify the endpoint of the corridors upon instantiation to evaluate CORRIDOR_DISTRIBUTION")
            else:
                data = mf.evaluate_duration(data=self.data, corridor_endpoints=self.corridor_endpoints)
                eplot.create_bar_plot(data=data, labels=['min', 'avg', 'max'])
                xlim = plt.gca().get_xlim()
                ylim = plt.gca().get_ylim()
                eplot.plot(metric=Metrics.DURATION, base_save_path=self.base_save_path, xlim=xlim, ylim=ylim)
