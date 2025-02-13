import numpy as np

from simulator.orientation_perception_free_zone_model import OrientationPerceptionFreeZoneModelSimulator
from animal_models.pigeon import Pigeon
from animal_models.hawk import Hawk
from animal_models.zebrafinch import Zebrafinch
from animal_models.zebra import Zebra
from animal_models.wolf import Wolf
from animal_models.rabbit import Rabbit

from area_models.landmark import Landmark
import loggers.logger_agents as logger
import loggers.logger_model_params as logger_params

n_agents = 7
n_steps = 5000
domain_size = (300, 100)
noise_amplitude = 0.2
start_position = (25, 20)
graph_freq = 10
visualize = False
visualize_vision_fields = 0
follow = False
single_speed = True
animal_type = Pigeon()

y = np.random.randint(30, 60)
start_position = (10, y)

dist_based_zone_factors = True

social_weight = 0.25
environment_weight = 0.75
other_type_weight = None

landmarks = []

num_iters = 500

for n_agents in [7, 5, 10, 20]:
        for animal_type in [Pigeon(), Hawk(), Zebrafinch(), Zebra(), Wolf(), Rabbit()]:
            base_save_path = f"basic_{animal_type.name}_n={n_agents}"
            save_path_params = f"log_params_{base_save_path}.csv"
            save_path_agents = f"log_agents_{base_save_path}.csv"
            save_path_centroid = f"log_centroid_{base_save_path}.csv"

            model_params = {
                "n": n_agents,
                "tmax": n_steps,
                "domain_size": domain_size,
                "noise": noise_amplitude,
                "start_position": start_position,
                "single_speed": single_speed,
                "animal_type": animal_type.name,
                "dist_based_zone_factors": dist_based_zone_factors,
                "social_weight": social_weight,
                "environment_weight": environment_weight,
                "landmarks": "-".join(",".join(f"[{corner[0]},{corner[1]}]" for corner in landmark.corners) for landmark in landmarks)
            }

            logger_params.log_model_params(model_params_dict=model_params, save_path=save_path_params)
            logger.initialise_log_file_with_headers(['iter', 't', 'i', 'x', 'y', 'h'], save_path=save_path_agents)
            logger.initialise_log_file_with_headers(['iter', 't', 'x', 'y'], save_path=save_path_centroid)

            for iter in range(num_iters):
                print(f"{animal_type.name} - {iter} / {num_iters}")
                sim = OrientationPerceptionFreeZoneModelSimulator(num_agents=n_agents,
                                    animal_type=animal_type,
                                    domain_size=domain_size,
                                    start_position=start_position,
                                    use_distant_dependent_zone_factors=dist_based_zone_factors,
                                    landmarks=landmarks,
                                    social_weight=social_weight,
                                    environment_weight=environment_weight,
                                    single_speed=single_speed,
                                    visualize=visualize,
                                    visualize_vision_fields=visualize_vision_fields,
                                    follow=follow,
                                    graph_freq=graph_freq,
                                    save_path_agents=save_path_agents,
                                    save_path_centroid=save_path_centroid,
                                    iter=iter)
                sim.run(tmax=n_steps)

