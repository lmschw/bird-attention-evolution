import numpy as np

from simulator.pigeon_simulator_2 import PigeonSimulator
from bird_models.pigeon import Pigeon
from area_models.landmark import Landmark

to_my_left = -1
to_my_right = 1

n_agents = 6
n_steps = 1000
env_size = (50, 50)
start_position = (25, 25)
graph_freq = 10
visualize = True
visualize_vision_fields = 3
follow = False
single_speed = True
bird_type = Pigeon()
start_position = (10, 10)
target_position = (40, 40)
target_radius = 10
target_attraction_range = 0 

social_weight = 0.2
path_weight = 0.8

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

#path_options = [path_options[0], path_options[6]]

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

