import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluators.evaluator_basic_movement import EvaluatorBasicMovement
from evaluators.metrics import Metrics
import evaluators.metrics_functions as mf
import loggers.logger_agents as logger

"""
Evaluation plots:
    - cohesion
    - order
    - corridor distribution
    - success proportion
    - duration until the agents reach the end of the corridor
"""

class EvaluatorCorridor(EvaluatorBasicMovement):
    def __init__(self, data_file_path, base_save_path, max_iters=None, corridor_centers=[], corridor_endpoints=[]):
        super().__init__(data_file_path=data_file_path,
                         base_save_path=base_save_path,
                         max_iters=max_iters)
        self.corridor_centers = corridor_centers
        self.corridor_endpoints = corridor_endpoints

    def evaluate_and_visualise(self, metric=None):
        super().evaluate_and_visualise(metric=metric)
        if metric in [None, Metrics.CORRIDOR_DISTRIBUTION]:
            if len(self.corridor_centers) != 2:
                print("need to specify exactly two corridors upon instantiation to evaluate CORRIDOR_DISTRIBUTION")
            else:
                data = self.evaluate_corridor_selection()
                self.create_pie_plot(data=data, labels=['same corridor', 'split'])
                xlim = plt.gca().get_xlim()
                ylim = plt.gca().get_ylim()
                self.plot(metric=Metrics.CORRIDOR_DISTRIBUTION, xlim=xlim, ylim=ylim)
        if metric in [None, Metrics.SUCCESS_PERCENTAGE]:
            if len(self.corridor_endpoints) != 2:
                print("need to specify the endpoint of the corridors upon instantiation to evaluate CORRIDOR_DISTRIBUTION")
            else:
                data = self.evaluate_success_percentage()
                self.create_pie_plot(data=data, labels=['percentage agents got through', 'percentage agents left behind'])
                xlim = plt.gca().get_xlim()
                ylim = plt.gca().get_ylim()
                self.plot(metric=Metrics.SUCCESS_PERCENTAGE, xlim=xlim, ylim=ylim)
        if metric in [None, Metrics.DURATION]:
            if len(self.corridor_endpoints) != 2:
                print("need to specify the endpoint of the corridors upon instantiation to evaluate CORRIDOR_DISTRIBUTION")
            else:
                data = self.evaluate_duration()
                self.create_bar_plot(data=data, labels=['min', 'avg', 'max'])
                xlim = plt.gca().get_xlim()
                ylim = plt.gca().get_ylim()
                self.plot(metric=Metrics.DURATION, xlim=xlim, ylim=ylim)

    def evaluate_corridor_selection(self):
        corridor_diff_x = np.absolute(self.corridor_centers[0][0]-self.corridor_centers[1][0])
        corridor_diff_y = np.absolute(self.corridor_centers[0][1]-self.corridor_centers[1][1])

        point_check = 'x' # the axis along which we're moving
        if corridor_diff_x > corridor_diff_y:
            point_check = 'y'

        count_all_same_corridor = 0
        for iter in range(len(self.data)):
            for t in range(len(self.data[iter])):
                if point_check == 'x' and self.data[iter][t][0,0] >= self.corridor_centers[0][0]:
                    result = np.absolute(self.corridor_centers[0][1]-self.data[iter][t][:,1]) < np.absolute(self.corridor_centers[1][1]-self.data[iter][t][:,1])
                    corridor_1_percentage = np.count_nonzero(result) / len(result)
                    corridor_0_percentage = 1-corridor_1_percentage
                    if corridor_0_percentage == 1 or corridor_1_percentage == 1:
                        count_all_same_corridor += 1
                    break
                if point_check == 'y' and self.data[iter][t][0,1] >= self.corridor_centers[0][1]:
                    result = np.absolute(self.corridor_centers[0][0]-self.data[iter][t][:,0]) < np.absolute(self.corridor_centers[1][0]-self.data[iter][t][:,0])
                    corridor_1_percentage = np.count_nonzero(result) / len(result)
                    corridor_0_percentage = 1-corridor_1_percentage
                    if corridor_0_percentage == 1 or corridor_1_percentage == 1:
                        count_all_same_corridor += 1
                    break
        return np.array([count_all_same_corridor/len(self.data), (len(self.data)-count_all_same_corridor)/len(self.data)])

    def evaluate_success_percentage(self):
        corridor_diff_x = np.absolute(self.corridor_endpoints[0][0]-self.corridor_endpoints[1][0])
        corridor_diff_y = np.absolute(self.corridor_endpoints[0][1]-self.corridor_endpoints[1][1])

        point_check = 'x' # the axis along which we're moving
        if corridor_diff_x > corridor_diff_y:
            point_check = 'y'

        success_percentages = []
        for iter in range(len(self.data)):
            t = -1
            if point_check == 'x':
                result = (self.corridor_endpoints[0][0] < self.data[iter][t][:,0]) & (self.corridor_endpoints[1][0] < self.data[iter][t][:,0])
                success_percentage = np.count_nonzero(result) / len(result)
                success_percentages.append(success_percentage)
            else:
                result = (self.corridor_endpoints[0][1] < self.data[iter][t][:,1]) & (self.corridor_endpoints[1][0] < self.data[iter][t][:,1])
                success_percentage = np.count_nonzero(result) / len(result)
                success_percentages.append(success_percentage)
        avg_success = np.average(success_percentages)
        return np.array([avg_success, 1-avg_success])
    
    def evaluate_duration(self):
        corridor_diff_x = np.absolute(self.corridor_endpoints[0][0]-self.corridor_endpoints[1][0])
        corridor_diff_y = np.absolute(self.corridor_endpoints[0][1]-self.corridor_endpoints[1][1])

        point_check = 'x' # the axis along which we're moving
        if corridor_diff_x > corridor_diff_y:
            point_check = 'y'

        durations = []
        for iter in range(len(self.data)):
            for t in range(len(self.data[iter])):
                if point_check == 'x':
                    result = (self.corridor_endpoints[0][0] < self.data[iter][t][:,0]) & (self.corridor_endpoints[1][0] < self.data[iter][t][:,0])
                    if np.count_nonzero(result) == len(result):
                        durations.append(t)
                        break
        return np.array([np.min(durations), np.average(durations), np.max(durations)])

    def create_pie_plot(self, data, labels):
        patches, _ = plt.pie(data, labels=[f"{int(p*100)}%" for p in data])
        plt.legend(patches, labels, loc="best")

    def create_bar_plot(self, data, labels):
        plt.bar(x=[1, 3, 5], height=data, tick_label=labels)
