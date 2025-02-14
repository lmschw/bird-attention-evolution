from evaluators.evaluator_predator import EvaluatorPredator
from evaluators.metrics import Metrics

from animal_models.hawk import Hawk
from animal_models.pigeon import Pigeon
from animal_models.rabbit import Rabbit
from animal_models.wolf import Wolf
from animal_models.zebra import Zebra
from animal_models.zebrafinch import Zebrafinch

for num_prey in [7]:
    for num_predators in [1]:
        for animal_type_combo in [[Pigeon(), Hawk()]]:
            for kill in [True]:
                animal_type_prey, animal_type_predator = animal_type_combo
                base_save_path = f"predator_{animal_type_prey.name}_{animal_type_predator.name}_n_prey={num_prey}_n=preds={num_predators}_kill={kill}"
                save_path_agents = f"log_agents_{base_save_path}.csv"
                evaluator = EvaluatorPredator(data_file_path=save_path_agents, 
                                                base_save_path=base_save_path,
                                                animal_type=animal_type_prey,
                                                max_iters=500)
                evaluator.evaluate_and_visualise()