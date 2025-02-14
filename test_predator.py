import numpy as np

from simulator.orientation_perception_free_zone_model_with_predators import OrientationPerceptionFreeZoneModelSimulatorWithPredators
from animal_models.pigeon import Pigeon
from animal_models.hawk import Hawk

import loggers.logger_agents as logger
import loggers.logger_model_params as logger_params


n_steps = 5000
domain_size = (300, 300)
noise_amplitude = 0.2
graph_freq = 10
visualize = False
visualize_vision_fields_prey = 0
visualize_vision_fields_predators = 0

follow = False
single_speed = True

start_position_prey = (150, 150)
start_position_predators = (10, 10)

limit_turns = True
dist_based_zone_factors = True
pack_hunting = True
killing_frenzy = False

social_weights = [0.5, 0.5] 
environment_weights = [0,0]
other_type_weights = [0.5, 0.5]

landmarks = []

num_iters = 500
for num_prey in [7, 5, 10, 20]:
    for num_predators in [1]:
        for animal_type_combo in [[Pigeon(), Hawk()]]:
            for kill in [True, False]:
                animal_type_prey, animal_type_predator = animal_type_combo
                base_save_path = f"predator_{animal_type_prey.name}_{animal_type_predator.name}_n_prey={num_prey}_n=preds={num_predators}_kill={kill}"
                save_path_params = f"log_params_{base_save_path}.csv"
                save_path_agents = f"log_agents_{base_save_path}.csv"
                save_path_centroid = f"log_centroid_{base_save_path}.csv"

                model_params = {
                    "num_prey": num_prey,
                    "animal_type_prey": animal_type_prey.name,
                    "num_predators": num_predators,
                    "animal_type_predators": animal_type_predator.name,
                    "tmax": n_steps,
                    "domain_size": domain_size,
                    "noise": noise_amplitude,
                    "start_position_prey": start_position_prey,
                    "start_position_predators": start_position_predators,
                    "single_speed": single_speed,
                    "dist_based_zone_factors": dist_based_zone_factors,
                    "limit_turns": limit_turns,
                    "social_weights": social_weights,
                    "environment_weights": environment_weights,
                    "other_type_weights": other_type_weights,
                    "pack_hunting": pack_hunting,
                    "kill": kill,
                    "killing_frenzy": killing_frenzy
                }

                logger_params.log_model_params(model_params_dict=model_params, save_path=save_path_params)
                logger.initialise_log_file_with_headers(['iter', 't', 'type', 'i', 'x', 'y', 'h'], save_path=save_path_agents)
                logger.initialise_log_file_with_headers(['iter', 't', 'x', 'y'], save_path=save_path_centroid)

                for iter in range(num_iters):
                    print(f"{animal_type_prey.name} ({num_prey}) - {animal_type_predator.name} ({num_predators}) - kill: {kill} - {iter} / {num_iters}")
                    sim = OrientationPerceptionFreeZoneModelSimulatorWithPredators(num_prey=num_prey,
                                                                                animal_type_prey=animal_type_prey,
                                                                                num_predators=num_predators,
                                                                                animal_type_predator=animal_type_predator,
                                                                                domain_size=domain_size,
                                                                                start_position_prey=start_position_prey,
                                                                                start_position_predator=start_position_predators,
                                                                                pack_hunting=pack_hunting,
                                                                                landmarks=landmarks,
                                                                                noise_amplitude=noise_amplitude,
                                                                                social_weights=social_weights,
                                                                                environment_weights=environment_weights,
                                                                                other_type_weights=other_type_weights,
                                                                                limit_turns=limit_turns,
                                                                                use_distant_dependent_zone_factors=dist_based_zone_factors,
                                                                                single_speed=single_speed,
                                                                                kill=kill,
                                                                                killing_frenzy=killing_frenzy,
                                                                                visualize=visualize,
                                                                                visualize_vision_fields_prey=visualize_vision_fields_prey,
                                                                                visualize_vision_fields_predator=visualize_vision_fields_predators,
                                                                                follow=follow,
                                                                                graph_freq=graph_freq,
                                                                                save_path_agents=save_path_agents,
                                                                                save_path_centroid=save_path_centroid,
                                                                                iter=iter)
                    sim.run(tmax=n_steps)

