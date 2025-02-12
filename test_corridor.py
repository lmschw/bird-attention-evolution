import numpy as np

from simulator.orientation_perception_free_zone_model import OrientationPerceptionFreeZoneModelSimulator
from animal_models.pigeon import Pigeon
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

border = Landmark('', corners=[[1,1], [1, 99], [299, 99], [299,1], [1,1], [0,0], [300, 0], [300, 100], [0,100], [0,0]])

landmark_1 = Landmark('1', corners=[[100, 0], [100, 25], [150, 25], [150, 0]])
landmark_2 = Landmark('2', corners=[[100, 30], [100, 65], [150, 65], [150, 35]])
landmark_3 = Landmark('3', corners=[[100, 70], [100, 100], [150, 100], [150, 75]])


landmarks = [border, landmark_1, landmark_2, landmark_3]

base_save_path = "navigation_through_narrow_hole"
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

for iter in range(100):
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

