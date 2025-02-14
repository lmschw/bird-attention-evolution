from evaluators.evaluator_basic_movement import EvaluatorBasicMovement
from evaluators.metrics import Metrics

from animal_models.hawk import Hawk
from animal_models.pigeon import Pigeon
from animal_models.rabbit import Rabbit
from animal_models.wolf import Wolf
from animal_models.zebra import Zebra
from animal_models.zebrafinch import Zebrafinch

for n in [5, 7, 10, 20]:
    for animal_type in [Pigeon(), Hawk()]:
        base_save_path = f"basic_normed_{animal_type.name}_n={n}"
        save_path_agents = f"log_agents_basic_{animal_type.name}_n={n}.csv"
        evaluator = EvaluatorBasicMovement(data_file_path=save_path_agents, 
                                        base_save_path=base_save_path,
                                        animal_type=animal_type,
                                        max_iters=None)
        evaluator.evaluate_and_visualise(metric=Metrics.COHESION)