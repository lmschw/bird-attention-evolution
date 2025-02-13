import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

class EvaluatorCorridor:
    def __init__(self, data_file_path, base_save_path, max_iters=None, corridor_centers=[], corridor_endpoints=[]):
        self.data_file_path = data_file_path
        self.base_save_path = base_save_path
        self.max_iters = max_iters
        self.corridor_centers = corridor_centers
        self.corridor_endpoints = corridor_endpoints
        self.data = logger.load_log_data(self.data_file_path, max_iters=max_iters)

    def evaluate_and_visualise(self, metric=None):
        if metric in [None, Metrics.COHESION]:
            data = self.evaluate_cohesion()
            self.create_line_plot(data=data, labels=["average distance to centroid"])
            self.plot(metric=Metrics.COHESION)
        if metric in [None, Metrics.ORDER]:
            data = self.evaluate_order()
            self.create_line_plot(data=data, labels=["global order"])
            self.plot(metric=Metrics.ORDER)
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

    def evaluate_cohesion(self):
        cohesion_results = {t: [] for t in range(len(self.data[0]))}
        for iter in range(len(self.data)):
            for t in range(len(self.data[iter])):
                result = mf.compute_cohesion(self.data[iter][t])
                cohesion_results[t].append(result)
        return {t: np.average(cohesion_results[t]) for t in range(len(self.data[0]))}

    def evaluate_order(self):
        order_results = {t: [] for t in range(len(self.data[0]))}
        for iter in range(len(self.data)):
            for t in range(len(self.data[iter])):
                result = mf.compute_global_order(self.data[iter][t])
                order_results[t].append(result)
        return {t: np.average(order_results[t]) for t in range(len(self.data[0]))}

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
        avg_success = np.average(success_percentages)
        return np.array([avg_success, 1-avg_success])

    def create_line_plot(self, data, labels=[""], xlim=None, ylim=None):
        sorted(data.items())
        df = pd.DataFrame(data, index=labels).T
        if xlim != None and ylim != None:
            df.plot.line(xlim=xlim, ylim=ylim)
        elif xlim != None:
            df.plot.line(xlim=xlim)
        elif ylim != None:
            df.plot.line(ylim=ylim)
        else:
            df.plot.line()

    def create_pie_plot(self, data, labels):
        patches, _ = plt.pie(data, labels=[f"{int(p*100)}%" for p in data])
        plt.legend(patches, labels, loc="best")
        #plt.gca().pie(data, labels=labels)

    def plot(self, metric, x_label=None, y_label=None, subtitle=None, xlim=None, ylim=None):
        ax = plt.gca()
        # reset axis to start at (0.0)
        if xlim == None:
            xlim = ax.get_xlim()
            ax.set_xlim((0, xlim[1]))
        if ylim == None:
            ylim = ax.get_ylim()
            ax.set_ylim((0, ylim[1]))

        if x_label != None:
            plt.xlabel(x_label)
        if y_label != None:
            plt.ylabel(y_label)
        if subtitle != None:
            plt.title(f"""{subtitle}""")
        if self.base_save_path != None:
            plt.savefig(f"{self.base_save_path}_{metric.value}.jpeg")
        plt.show()
        plt.close()