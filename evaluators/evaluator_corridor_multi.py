import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluators.metrics import Metrics
import evaluators.metrics_functions as mf
import loggers.logger_agents as logger
import evaluators.evaluation_plotting as eplot
from evaluators.evaluator_basic_movement_multi import EvaluatorBasicMovementMulti

"""
Evaluation plots:
    - cohesion
    - order
"""

class EvaluatorCorridorMulti(EvaluatorBasicMovementMulti):
    def __init__(self, data_file_paths, data_labels, base_save_path, max_iters=None, corridor_centers=[], corridor_endpoints=[]):
        super().__init__(data_file_paths=data_file_paths,
                         data_labels=data_labels,
                         base_save_path=base_save_path,
                         max_iters=max_iters)
        self.corridor_centers = corridor_centers
        self.corridor_endpoints = corridor_endpoints

    def evaluate_and_visualise(self, metrics):
        for metric in metrics:
            data = []
            for i in range(len(self.data)):
                subdata = self.data[i]
                match metric:
                    case Metrics.COHESION:
                        data.append(mf.evaluate_cohesion(data=subdata))
                    case Metrics.ORDER:
                        data.append(mf.evaluate_order(data=subdata))   
                    case Metrics.CORRIDOR_DISTRIBUTION:
                        if len(self.corridor_centers) != 2:
                            print("need to specify exactly two corridors upon instantiation to evaluate CORRIDOR_DISTRIBUTION")
                        else:
                            data.append(mf.evaluate_corridor_selection(data=subdata, corridor_centers=self.corridor_centers)[0])
                    case Metrics.SUCCESS_PERCENTAGE:
                        if len(self.corridor_endpoints) != 2:
                            print("need to specify the endpoint of the corridors upon instantiation to evaluate SUCCESS_PERCENTAGE")
                        else:
                            data.append(mf.evaluate_success_percentage(data=subdata, corridor_endpoints=self.corridor_endpoints)[0])
                    case Metrics.DURATION:
                        if len(self.corridor_endpoints) != 2:
                            print("need to specify the endpoint of the corridors upon instantiation to evaluate CORRIDOR_DISTRIBUTION")
                        else:
                            data.append(mf.evaluate_duration(data=subdata, corridor_endpoints=self.corridor_endpoints)[0])
            self.create_plot(data=data, metric=metric)

    def create_plot(self, data, metric):
            set_lims = False
            match metric:
                case Metrics.COHESION:
                    eplot.create_line_plot(data=data, labels=self.data_labels)
                case Metrics.ORDER:
                    eplot.create_line_plot(data=data, labels=self.data_labels)
                case Metrics.CORRIDOR_DISTRIBUTION:
                    eplot.create_bar_plot(data=data, labels=self.data_labels)
                    set_lims = True
                case Metrics.CORRIDOR_DISTRIBUTION:
                    eplot.create_bar_plot(data=data, labels=self.data_labels)
                case Metrics.SUCCESS_PERCENTAGE:
                    eplot.create_bar_plot(data=data, labels=self.data_labels)
                case Metrics.DURATION:
                    eplot.create_bar_plot(data=data, labels=self.data_labels)
            if set_lims:
                xlim = plt.gca().get_xlim()
                ylim = plt.gca().get_ylim()
                eplot.plot(metric=Metrics.CORRIDOR_DISTRIBUTION, xlim=xlim, ylim=ylim)
            else:
                eplot.plot(metric=metric)