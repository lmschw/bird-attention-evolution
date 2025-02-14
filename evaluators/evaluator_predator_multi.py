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

class EvaluatorPredatorMulti(EvaluatorBasicMovementMulti):
    def load_data(self):
        data = []
        for path in self.data_file_paths:
            data.append(logger.load_log_data(path, max_iters=self.max_iters, is_predator_scenario=True))
        return data

