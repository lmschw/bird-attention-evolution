import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluators.metrics import Metrics
import evaluators.metrics_functions as mf
import loggers.logger_agents as logger
import evaluators.evaluation_plotting as eplot
from evaluators.evaluator_basic_movement import EvaluatorBasicMovement

"""
Evaluation plots:
    - cohesion
    - order
"""

class EvaluatorPredator(EvaluatorBasicMovement):
    def load_data(self):
        self.data, self.predator_data = logger.load_log_data(self.data_file_path, max_iters=self.max_iters, is_predator_scenario=True, load_predator=True)

    def evaluate_and_visualise(self, metric=None, normalise_cohesion=False):
        if metric in [None, Metrics.SPLIT_TURN]:
            data = mf.evaluate_splits_and_turns(prey_data=self.data, predator_data=self.predator_data)
            eplot.create_bar_plot(data=data, labels=['splits', 'turns', 'avg angle splits', 'avg angle turns'])
            eplot.plot(metric=Metrics.SPLIT_TURN, base_save_path=self.base_save_path, x_label='timesteps', y_label='')
