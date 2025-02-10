from bird_models.pigeon import Pigeon
from area_models.landmark import Landmark
from simulator.ae_simulator import PigeonSimulatorAe


n_agents = 10
n_steps = 10000
env_size = (50, 50)
graph_freq = 10
visualize = True
follow = True
animal_type = Pigeon()
start_position = (10, 10)
target_position = (40, 40)
target_radius = 10
target_success_percentage = 50 

landmark_1 = Landmark("1", [10, 15])
landmark_2 = Landmark("2", [15, 40])
landmark_3 = Landmark("3", [25, 20])
landmark_4 = Landmark("4", [40, 10])
landmark_5 = Landmark("5", [48, 25])
landmark_6 = Landmark("6", [30, 35])
target = Landmark('target', target_position)
landmarks = [landmark_1, landmark_2, landmark_3, landmark_4, landmark_5, landmark_6, target]

landmark_7 = Landmark("7", [30, 35])

#landmarks = [landmark_7]
landmarks = [landmark_1]

path_options = [{landmark_1: 'r', landmark_3: 'l', landmark_6: 'r', target: 's'},
         {landmark_1: 'r', landmark_3: 'l', target: 's'},
         {landmark_1: 'l', landmark_2: 'r', landmark_6: 'l', target: 's'},
         {landmark_1: 'l', landmark_3: 'l', landmark_5: 'l', target: 's'},
         {landmark_3: 'l', landmark_6: 'r', target: 's'},
         {landmark_3: 'r', landmark_4: 'l', landmark_5: 'l', target: 's'},
         {landmark_3: 'r', landmark_4: 'l', target: 's'}
         ]
sim = PigeonSimulatorAe(animal_type=animal_type, num_agents=n_agents, env_size=env_size,
                        start_position=start_position, 
                        model=None,
                        visualize=visualize, follow=follow, graph_freq=graph_freq)
sim.run(tmax=n_steps)
