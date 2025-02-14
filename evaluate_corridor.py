from evaluators.evaluator_corridor import EvaluatorCorridor
from evaluators.metrics import Metrics
from animal_models.hawk import Hawk
from animal_models.pigeon import Pigeon
from animal_models.rabbit import Rabbit
from animal_models.wolf import Wolf
from animal_models.zebra import Zebra
from animal_models.zebrafinch import Zebrafinch

for n in [7, 5]:
    for animal_type in [Pigeon(), Hawk()]:
        base_save_path = f"corridor_{animal_type.name}_n={n}"
        save_path_agents = f"log_agents_{base_save_path}.csv"
        evaluator = EvaluatorCorridor(data_file_path=save_path_agents, 
                                        base_save_path=base_save_path,
                                        max_iters=500,
                                        corridor_centers=[[125, 27.5], [125, 67.5]],
                                        corridor_endpoints=[[150, 27.5], [150, 67.5]])
        evaluator.evaluate_and_visualise()