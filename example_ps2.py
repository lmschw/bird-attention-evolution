import numpy as np

from simulator.pigeon_simulator_2 import PigeonSimulator
from simulator.enum_weight_options import WeightOptions
from bird_models.pigeon import Pigeon
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

to_my_left = -1
to_my_right = 1

n_agents = 7
n_steps = 10000
env_size = (50, 50)
start_position = (25, 25)
graph_freq = 10
visualize = True
visualize_vision_fields = 1
follow = True
single_speed = True
bird_type = Pigeon()
start_position = (10, 10)
target_position = (40, 40)
target_radius = 10
target_attraction_range = 0 

dist_based_zone_factors = True

social_weight = 1
path_weight = 0

landmark_1 = Landmark("1", [10, 15])
landmark_2 = Landmark("2", [15, 40])
landmark_3 = Landmark("3", [25, 20])
landmark_4 = Landmark("4", [40, 10])
landmark_5 = Landmark("5", [48, 25])
landmark_6 = Landmark("6", [30, 35])
target = Landmark('target', target_position)
landmarks = [landmark_1, landmark_2, landmark_3, landmark_4, landmark_5, landmark_6]

landmark_7 = Landmark("7", [30, 35])

#landmarks = [landmark_7]
#landmarks = [landmark_1]

path_options = [np.array([to_my_left, 0, to_my_right, 0, 0, to_my_left]),
                np.array([to_my_left, to_my_left, 0, 0, 0, to_my_right]),
                np.array([to_my_left, 0, to_my_left, 0, to_my_right, 0]),
                np.array([0, 0, to_my_left, 0, to_my_right, 0]),
                np.array([to_my_left, 0, 0, to_my_right, to_my_right, 0]),
                np.array([to_my_left, 0, 0, to_my_right, 0, to_my_left]),
                np.array([to_my_left, to_my_left, 0, to_my_right, 0, 0])]

path_options = [path_options[0]]

#path_options = [np.array([to_my_left, to_my_left, to_my_right, to_my_right, to_my_right, to_my_left])]

""" path_options = [{landmark_1: 'r', landmark_3: 'l', landmark_6: 'r', target: 's'},
         {landmark_1: 'r', landmark_3: 'l', target: 's'},
         {landmark_1: 'l', landmark_2: 'r', landmark_6: 'l', target: 's'},
         {landmark_1: 'l', landmark_3: 'l', landmark_5: 'l', target: 's'},
         {landmark_3: 'l', landmark_6: 'r', target: 's'},
         {landmark_3: 'r', landmark_4: 'l', landmark_5: 'l', target: 's'},
         {landmark_3: 'r', landmark_4: 'l', target: 's'}
         ] """
sim = PigeonSimulator(num_agents=n_agents,
                      bird_type=bird_type,
                      domain_size=env_size,
                      start_position=start_position,
                      target_position=target_position,
                      target_radius=target_radius,
                      target_attraction_range=target_attraction_range,
                      use_distant_dependent_zone_factors=dist_based_zone_factors,
                      weight_options=weight_options,
                      model=nn,
                      landmarks=landmarks,
                      path_options=path_options,
                      social_weight=social_weight,
                      path_weight=path_weight,
                      single_speed=single_speed,
                      visualize=visualize,
                      visualize_vision_fields=visualize_vision_fields,
                      follow=follow,
                      graph_freq=graph_freq)
sim.run(tmax=n_steps)

