from evaluators.evaluator_corridor import EvaluatorCorridor

evaluator = EvaluatorCorridor(data_file_path="log_agents_navigation_through_narrow_hole.csv", 
                              base_save_path="eval_navigation_through_narrow_hole",
                              max_iters=2)

evaluator.evaluate_and_visualise()