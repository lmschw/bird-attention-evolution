import numpy as np

from simulator.pigeon_simulator_2_with_predators import PigeonSimulatorWithPredators
from simulator.enum_weight_options import WeightOptions
from bird_models.pigeon import Pigeon
from bird_models.zebrafinch import Zebrafinch
from bird_models.hawk import Hawk
from area_models.landmark import Landmark

from neural_network.activation_layer import ActivationLayer
from neural_network.fully_connected_layer import FullyConnectedLayer
from neural_network.neural_network import NeuralNetwork
import neural_network.activation_functions as snn

"""
weight_options = [WeightOptions.CLOSEST_DISTANCES, 
                  WeightOptions.AVG_BEARINGS, 
                  WeightOptions.NUM_VISIBLE_AGENTS,
                  WeightOptions.PREVIOUS_HEAD_ANGLES]
# closest distance, average bearings, num visible agents, previous head angle
weights = [0.24373, 0.90672, 1.,      0.31082]
weights = [0.52398, 0.70982, 0.98804, 0.50911]
weights = [0,0,0,0]
"""
weight_options = [WeightOptions.CLOSEST_DISTANCES,
                  WeightOptions.CLOSEST_BEARINGS,
                  WeightOptions.AVG_DISTANCES,
                  WeightOptions.AVG_BEARINGS,
                  WeightOptions.NUM_VISIBLE_AGENTS,
                  WeightOptions.PREVIOUS_HEAD_ANGLES,
                  WeightOptions.AVG_PERCEPTION_STRENGTHS]
weight_size = len(weight_options)
output_size = 1

weights = [0, 0, 0, 0, 0, 0, 0]

nn = NeuralNetwork()
fully_connected_layer = FullyConnectedLayer(input_size=weight_size, output_size=output_size)
fully_connected_layer.set_weights(weights=weights)
nn.add(fully_connected_layer)
nn.add(ActivationLayer(activation=snn.tanh, activation_prime=snn.tanh_prime))

n_steps = 10000
env_size = (500, 500)
noise_amplitude = 0.2
graph_freq = 10
visualize = True
visualize_head_directions = False
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
bird_type_prey = Pigeon()
num_predators = 1
bird_type_predator = Hawk()
start_position_prey = (150, 150)
start_position_predator = (200, 100)
pack_hunting = False
visualize_vision_fields_prey = 1
visualize_vision_fields_predator = 1

social_weight = 0.5
environment_weight = 0
other_type_weight = 0.5

sim = PigeonSimulatorWithPredators(num_prey=num_prey,
                                   bird_type_prey=bird_type_prey,
                                   num_predators=num_predators,
                                   bird_type_predator=bird_type_predator,
                                   domain_size=env_size,
                                   start_position_prey=start_position_prey,
                                   start_position_predator=start_position_predator,
                                   landmarks=landmarks,
                                   noise_amplitude=noise_amplitude,
                                   social_weight=social_weight,
                                   environment_weight=environment_weight,
                                   other_type_weight=other_type_weight,
                                   limit_turns=limit_turns,
                                   use_distant_dependent_zone_factors=use_distant_dependent_zone_factors,
                                   weight_options=weight_options,
                                   model=nn,
                                   single_speed=single_speed,
                                   visualize=visualize,
                                   visualize_vision_fields_prey=visualize_vision_fields_prey,
                                   visualize_vision_fields_predator=visualize_vision_fields_predator,
                                   visualize_head_direction=visualize_head_directions, follow=follow,
                                   graph_freq=graph_freq)
sim.run(tmax=n_steps)

