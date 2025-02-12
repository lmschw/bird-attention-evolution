from simulator.orientation_perception_free_zone_model import OrientationPerceptionFreeZoneModelSimulator
from animal_models.pigeon import Pigeon
from area_models.landmark import Landmark

n_agents = 7
n_steps = 10000
domain_size = (50, 50)
noise_amplitude = 0
start_position = (25, 25)
graph_freq = 10
visualize = True
visualize_vision_fields = 0
follow = False
single_speed = True
animal_type = Pigeon()
start_position = (10, 10)

dist_based_zone_factors = True

social_weight = 0.5
environment_weight = 1

border = Landmark('', corners=[[1,1], [1, 49], [49, 49], [49,1], [1,1], [0,0], [50, 0], [50, 50], [0,50], [0,0]])
wall_1 = Landmark('', [[20, 0], [20, 20], [25, 20], [25, 0]])
wall_2 = Landmark('', [[5, 50], [15, 20], [15, 50], [10, 50]])


landmarks = [border, wall_1, wall_2]

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
                      graph_freq=graph_freq)
sim.run(tmax=n_steps)

