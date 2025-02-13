from evaluators.evaluator_corridor import EvaluatorCorridor
from evaluators.metrics import Metrics

evaluator = EvaluatorCorridor(data_file_path="log_agents_navigation_through_narrow_hole.csv", 
                              base_save_path="eval_navigation_through_narrow_hole",
                              max_iters=100,
                              corridor_centers=[[125, 27.5], [125, 67.5]])

evaluator.evaluate_and_visualise(metric=Metrics.CORRIDOR_DISTRIBUTION)