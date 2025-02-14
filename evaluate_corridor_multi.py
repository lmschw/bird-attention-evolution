from evaluators.evaluator_corridor_multi import EvaluatorCorridorMulti
from evaluators.metrics import Metrics

from animal_models.hawk import Hawk
from animal_models.pigeon import Pigeon
from animal_models.rabbit import Rabbit
from animal_models.wolf import Wolf
from animal_models.zebra import Zebra
from animal_models.zebrafinch import Zebrafinch

for n in [7]:
        base_save_path = f"basic_comparison_n={n}"
        save_path_agents = [f"log_agents_basic_Pigeon_n={n}.csv", f"log_agents_basic_Hawk_n={n}.csv"]
        evaluator = EvaluatorCorridorMulti(data_file_paths=save_path_agents,
                                                data_labels=["Pigeon", "Hawk"],
                                                base_save_path=base_save_path,
                                                animal_types=[Pigeon(), Hawk()],
                                                max_iters=2,
                                                corridor_centers=[[125, 27.5], [125, 67.5]],
                                                corridor_endpoints=[[150, 27.5], [150, 67.5]])
        evaluator.evaluate_and_visualise(metrics=[Metrics.COHESION, 
                                                  Metrics.ORDER, 
                                                  Metrics.CORRIDOR_DISTRIBUTION,
                                                  Metrics.SUCCESS_PERCENTAGE,
                                                  Metrics.DURATION])