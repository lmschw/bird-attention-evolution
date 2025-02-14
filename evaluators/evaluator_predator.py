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
        self.data = logger.load_log_data(self.data_file_path, max_iters=self.max_iters, is_predator_scenario=True)

