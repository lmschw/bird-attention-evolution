from simulator.orientation_perception_free_zone_model import OrientationPerceptionFreeZoneModelSimulator
from simulator.ae_simulator import ActiveElasticSimulator
from simulator.ae_simulator_with_perception_strength import ActiveElasticWithPerceptionStrengthSimulator
from simulator.couzin_simulator import CouzinZoneModelSimulator
from simulator.couzin_with_perception_strength_simulator import CouzinZoneModelWithPerceptionStrengthSimulator
from simulator.vicsek_simulator import VicsekSimulator
from simulator.vicsek_with_perception_strength_simulator import VicsekWithPerceptionStrengthSimulator
from animal_models.pigeon import Pigeon
from area_models.landmark import Landmark
import loggers.logger_agents as logger
import loggers.logger_model_params as logger_params

n_agents = 7
n_steps = 10000
domain_size = (50, 50)
noise_amplitude = 0
start_position = (25, 20)
graph_freq = 10
visualize = True
visualize_vision_fields = 0
follow = False
single_speed = True
animal_type = Pigeon()
start_position = (10, 10)

dist_based_zone_factors = True

social_weight = 0.5
environment_weight = 0.5
other_type_weight = None

landmark_1 = Landmark('1', corners=[[20, 10], [20, 15], [25, 15], [25, 10]])
landmark_2 = Landmark('2', corners=[[20, 0], [20, 5], [25, 5], [25, 0]])
landmark_3 = Landmark('3', corners=[[20, 20], [20, 25], [25, 25], [25, 20]])


landmarks = [landmark_1, landmark_2, landmark_3]

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

iter = 0
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

