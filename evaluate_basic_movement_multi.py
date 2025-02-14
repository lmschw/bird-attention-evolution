from evaluators.evaluator_basic_movement_multi import EvaluatorBasicMovementMulti
from evaluators.metrics import Metrics

from animal_models.hawk import Hawk
from animal_models.pigeon import Pigeon
from animal_models.rabbit import Rabbit
from animal_models.wolf import Wolf
from animal_models.zebra import Zebra
from animal_models.zebrafinch import Zebrafinch

for n in [5]:
        base_save_path = f"basic_comparison_n={n}"
        save_path_agents = [f"log_agents_basic_Pigeon_n={n}.csv", f"log_agents_basic_Hawk_n={n}.csv"]
        evaluator = EvaluatorBasicMovementMulti(data_file_paths=save_path_agents,
                                                data_labels=["Pigeon", "Hawk"],
                                                base_save_path=base_save_path,
                                                max_iters=2)
        evaluator.evaluate_and_visualise(metrics=[Metrics.COHESION, Metrics.ORDER])