from simulator.orientation_perception_free_zone_model import OrientationPerceptionFreeZoneModelSimulator
from animal_models.pigeon import Pigeon
from area_models.landmark import Landmark
from simulator.enum_neighbour_selection import NeighbourSelectionMechanism

n_agents = 7
n_steps = 10000
domain_size = (500, 500)
noise_amplitude = 0
start_position = (250, 20)
graph_freq = 10
visualize = True
visualize_vision_fields = 0
follow = True
single_speed = True
animal_type = Pigeon()
start_position = (250, 250)

dist_based_zone_factors = True

social_weight = 0.5
environment_weight = 0

landmark_1 = Landmark('1', corners=[[20, 10], [20, 15], [25, 15], [25, 10]])
landmark_2 = Landmark('2', corners=[[20, 0], [20, 5], [25, 5], [25, 0]])
landmark_3 = Landmark('3', corners=[[20, 20], [20, 25], [25, 25], [25, 20]])


landmarks = [landmark_1, landmark_2, landmark_3]
landmarks = []

sim = OrientationPerceptionFreeZoneModelSimulator(num_agents=n_agents,
                      animal_type=animal_type,
                      domain_size=domain_size,
                      start_position=start_position,
                      use_distant_dependent_zone_factors=dist_based_zone_factors,
                      landmarks=landmarks,
                      social_weight=social_weight,
                      environment_weight=environment_weight,
                      single_speed=single_speed,
                      neighbour_selection=NeighbourSelectionMechanism.CLOSEST_TO_FOVEA,
                      k=2,
                      visualize=visualize,
                      visualize_vision_fields=visualize_vision_fields,
                      follow=follow,
                      graph_freq=graph_freq)
sim.run(tmax=n_steps)

