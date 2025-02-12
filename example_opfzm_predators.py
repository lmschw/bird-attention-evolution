import numpy as np

from simulator.orientation_perception_free_zone_model_with_predators import OrientationPerceptionFreeZoneModelSimulatorWithPredators
from simulator.head_movement.enum_weight_options import WeightOptions
from animal_models.pigeon import Pigeon
from animal_models.zebrafinch import Zebrafinch
from animal_models.hawk import Hawk
from area_models.landmark import Landmark

from neural_network.activation_layer import ActivationLayer
from neural_network.fully_connected_layer import FullyConnectedLayer
from neural_network.neural_network import NeuralNetwork
import neural_network.activation_functions as snn

n_steps = 10000
domain_size = (500, 500)
noise_amplitude = 0.2
graph_freq = 10
visualize = True
follow = True
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

social_weight = 0.5
environment_weight = 0
other_type_weight = 0.5

sim = OrientationPerceptionFreeZoneModelSimulatorWithPredators(num_prey=num_prey,
                                   animal_type_prey=animal_type_prey,
                                   num_predators=num_predators,
                                   animal_type_predator=animal_type_predator,
                                   domain_size=domain_size,
                                   start_position_prey=start_position_prey,
                                   start_position_predator=start_position_predator,
                                   landmarks=landmarks,
                                   noise_amplitude=noise_amplitude,
                                   social_weight=social_weight,
                                   environment_weight=environment_weight,
                                   other_type_weight=other_type_weight,
                                   limit_turns=limit_turns,
                                   use_distant_dependent_zone_factors=use_distant_dependent_zone_factors,
                                   single_speed=single_speed,
                                   visualize=visualize,
                                   visualize_vision_fields_prey=visualize_vision_fields_prey,
                                   visualize_vision_fields_predator=visualize_vision_fields_predator,
                                   follow=follow,
                                   graph_freq=graph_freq)
sim.run(tmax=n_steps)

