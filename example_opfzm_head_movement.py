import numpy as np

from simulator.head_movement.orientation_perception_free_zone_model_with_head_movement import OrientationPerceptionFreeZoneModelWithHeadMovementSimulator
from simulator.head_movement.enum_weight_options import WeightOptions
from animal_models.pigeon import Pigeon
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
weights = [0.04894279039946401,0.00667812943656855,0.10353310279886091,0.21765670588565983,0.20047645873708853,0.20505610685669823,0.21765670588565983]

#weights = [0, 0, 0, 0, 0, 0, 0]

nn = NeuralNetwork()
fully_connected_layer = FullyConnectedLayer(input_size=weight_size, output_size=output_size)
fully_connected_layer.set_weights(weights=weights)
nn.add(fully_connected_layer)
nn.add(ActivationLayer(activation=snn.tanh, activation_prime=snn.tanh_prime))

n_agents = 7
n_steps = 10000
domain_size = (500, 500)
noise_amplitude = 0
start_position = (250, 20)
graph_freq = 10
visualize = True
visualize_vision_fields = 0
visualize_head_directions = True
follow = True
single_speed = True
animal_type = Pigeon()
start_position = (10, 10)

dist_based_zone_factors = True

social_weight = 0.5
environment_weight = 0

landmark_1 = Landmark('1', corners=[[20, 10], [20, 15], [25, 15], [25, 10]])
landmark_2 = Landmark('2', corners=[[20, 0], [20, 5], [25, 5], [25, 0]])
landmark_3 = Landmark('3', corners=[[20, 20], [20, 25], [25, 25], [25, 20]])


landmarks = [landmark_1, landmark_2, landmark_3]
landmarks = []

sim = OrientationPerceptionFreeZoneModelWithHeadMovementSimulator(num_agents=n_agents,
                      animal_type=animal_type,
                      domain_size=domain_size,
                      start_position=start_position,
                      use_distant_dependent_zone_factors=dist_based_zone_factors,
                      weight_options=weight_options,
                      model=nn,
                      landmarks=landmarks,
                      social_weight=social_weight,
                      environment_weight=environment_weight,
                      single_speed=single_speed,
                      visualize=visualize,
                      visualize_vision_fields=visualize_vision_fields,
                      visualize_head_direction=visualize_head_directions,
                      follow=follow,
                      graph_freq=graph_freq)
sim.run(tmax=n_steps)

