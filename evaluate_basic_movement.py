from evaluators.evaluator_basic_movement import EvaluatorBasicMovement
from evaluators.metrics import Metrics

evaluator = EvaluatorBasicMovement(data_file_path="log_agents_navigation_through_narrow_hole.csv", 
                                   base_save_path="eval_navigation_through_narrow_hole",
                                   max_iters=2)

evaluator.evaluate_and_visualise()