from evaluators.evaluator_basic_movement import EvaluatorBasicMovement
from evaluators.metrics import Metrics

from simulator.enum_neighbour_selection import NeighbourSelectionMechanism

from animal_models.hawk import Hawk
from animal_models.pigeon import Pigeon
from animal_models.rabbit import Rabbit
from animal_models.wolf import Wolf
from animal_models.zebra import Zebra
from animal_models.zebrafinch import Zebrafinch

for n in [7]:
    for nsm in [NeighbourSelectionMechanism.NEAREST, NeighbourSelectionMechanism.FARTHEST]:
        for animal_type in [Pigeon()]:
                base_save_path = f"test_ns_normed_{animal_type.name}_nsm={nsm.value}_n={n}"
                save_path_agents = f"log_agents_ns_{animal_type.name}_nsm={nsm.value}_n={n}.csv"
                evaluator = EvaluatorBasicMovement(data_file_path=save_path_agents, 
                                                base_save_path=base_save_path,
                                                animal_type=animal_type,
                                                min_iters=1,
                                                max_iters=2)
                evaluator.evaluate_and_visualise(metric=Metrics.ORDER)