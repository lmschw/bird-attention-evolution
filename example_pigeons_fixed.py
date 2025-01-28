

from area_models.area import Area
from area_models.landmark import Landmark
from simulator.pigeon_simulator import PigeonSimulator

from neural_network.neural_network import NeuralNetwork
from neural_network.activation_layer import ActivationLayer
from neural_network.fully_connected_layer import FullyConnectedLayer
import neural_network.activation_functions as af

weights_size = 2
weights = [0.5, 0.5]
nn = NeuralNetwork()
fully_connected_layer = FullyConnectedLayer(input_size=weights_size, output_size=5)
fully_connected_layer.set_weights(weights=weights)
nn.add(fully_connected_layer)
nn.add(ActivationLayer(activation=af.tanh, activation_prime=af.tanh_prime))

area_size = (50, 50)
landmark_1 = Landmark("1", [10, 15])
landmark_2 = Landmark("2", [15, 40])
landmark_3 = Landmark("3", [25, 20])
landmark_4 = Landmark("4", [40, 10])
landmark_5 = Landmark("5", [48, 25])
landmark_6 = Landmark("6", [30, 35])
landmarks = [landmark_1, landmark_2, landmark_3, landmark_4, landmark_5, landmark_6]

start_position = (5, 5)
target_position = (45, 40)

paths = [{landmark_1: 'r', landmark_3: 'l', landmark_6: 'r'},
         {landmark_1: 'r', landmark_3: 'l'},
         {landmark_1: 'l', landmark_2: 'r', landmark_6: 'l'},
         {landmark_1: 'l', landmark_3: 'l', landmark_5: 'l'},
         {landmark_3: 'l', landmark_6: 'r'},
         {landmark_3: 'r', landmark_4: 'l', landmark_5: 'l'},
         {landmark_3: 'r', landmark_4: 'l'}
         ]

area = Area(area_size=area_size, landmarks=landmarks)
area.set_paths(paths=paths)

simulator = PigeonSimulator(n=5, 
                            area=area, 
                            start_position=start_position, 
                            target_position=target_position,
                            neural_network=nn)

simulator.simulate(tmax=1000)
