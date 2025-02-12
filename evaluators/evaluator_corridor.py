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
"""

class EvaluatorCorridor:
    def __init__(self, data_file_path, base_save_path, max_iters=None):
        self.data_file_path = data_file_path
        self.base_save_path = base_save_path
        self.data = logger.load_log_data(self.data_file_path, max_iters=max_iters)

    def evaluate_and_visualise(self, metric=None):
        if metric in [None, Metrics.COHESION]:
            data = self.evaluate_cohesion()
            self.create_line_plot(data=data, labels=["average distance to centroid"])
            self.plot(data=data, metric=Metrics.COHESION)
        if metric in [None, Metrics.ORDER]:
            data = self.evaluate_order()
            self.create_line_plot(data=data, labels=["global order"])
            self.plot(data=data, metric=Metrics.ORDER)
        # if metric in [None, Metrics.CORRIDOR_DISTRIBUTION]:
        #     data = self.evaluate_corridor_selection()
        #     self.create_pie_plot(data=data)
        #     self.plot(data=data, metric=Metrics.CORRIDOR_DISTRIBUTION, labels=["corridor distribution"])
        # if metric in [None, Metrics.SUCCESS_PERCENTAGE]:
        #     self.plot_success_percentage()

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
        order_results = {t: [] for t in range(len(self.data[0]))}
        for iter in range(len(self.data)):
            for t in range(len(self.data[iter])):
                result = mf.compute_global_order(self.data[iter][t])
                order_results[t].append(result)
        return {t: np.average(order_results[t]) for t in range(len(self.data[0]))}

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

    def plot(self, metric, x_label=None, y_label=None, subtitle=None, xlim=None, ylim=None):
        ax = plt.gca()
        # reset axis to start at (0.0)
        xlim = ax.get_xlim()
        ax.set_xlim((0, xlim[1]))
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