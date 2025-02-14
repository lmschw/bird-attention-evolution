from simulator.orientation_perception_free_zone_model_with_predators import OrientationPerceptionFreeZoneModelSimulatorWithPredators
from animal_models.pigeon import Pigeon
from animal_models.hawk import Hawk
from area_models.landmark import Landmark

n_steps = 10000
domain_size = (500, 500)
noise_amplitude = 0.2
graph_freq = 10
visualize = True
follow = False
single_speed = True
limit_turns = True
use_distant_dependent_zone_factors = True


landmark_1 = Landmark('1', corners=[[20, 10], [20, 15], [25, 15], [25, 10]])
landmark_2 = Landmark('2', corners=[[20, 0], [20, 5], [25, 5], [25, 0]])
landmark_3 = Landmark('3', corners=[[20, 20], [20, 25], [25, 25], [25, 20]])


landmarks = [landmark_1, landmark_2, landmark_3]
landmarks = []

num_prey = 10
animal_type_prey = Pigeon()
num_predators = 1
animal_type_predator = Hawk()
start_position_prey = (150, 150)
start_position_predator = (200, 100)
pack_hunting = False
visualize_vision_fields_prey = 1
visualize_vision_fields_predator = 1

social_weights = [0.5, 0.5]
environment_weights = [0, 0]
other_type_weights = [0.5, 0.5]

kill = True
killing_frenzy = False

sim = OrientationPerceptionFreeZoneModelSimulatorWithPredators(num_prey=num_prey,
                                   animal_type_prey=animal_type_prey,
                                   num_predators=num_predators,
                                   animal_type_predator=animal_type_predator,
                                   domain_size=domain_size,
                                   start_position_prey=start_position_prey,
                                   start_position_predator=start_position_predator,
                                   landmarks=landmarks,
                                   noise_amplitude=noise_amplitude,
                                   social_weights=social_weights,
                                   environment_weights=environment_weights,
                                   other_type_weights=other_type_weights,
                                   limit_turns=limit_turns,
                                   use_distant_dependent_zone_factors=use_distant_dependent_zone_factors,
                                   single_speed=single_speed,
                                   kill=kill,
                                   killing_frenzy=killing_frenzy,
                                   visualize=visualize,
                                   visualize_vision_fields_prey=visualize_vision_fields_prey,
                                   visualize_vision_fields_predator=visualize_vision_fields_predator,
                                   follow=follow,
                                   graph_freq=graph_freq)
sim.run(tmax=n_steps)

