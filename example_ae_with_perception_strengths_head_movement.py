from animal_models.pigeon import Pigeon
from simulator.head_movement.ae_simulator_with_perception_strength_head_movement import ActiveElasticWithPerceptionStrengthAndHeadMovementSimulator
from simulator.head_movement.enum_weight_options import WeightOptions

from neural_network.activation_layer import ActivationLayer
from neural_network.fully_connected_layer import FullyConnectedLayer
from neural_network.neural_network import NeuralNetwork
import neural_network.activation_functions as snn

weight_options = [WeightOptions.CLOSEST_DISTANCES,
                  WeightOptions.CLOSEST_BEARINGS,
                  WeightOptions.AVG_DISTANCES,
                  WeightOptions.AVG_BEARINGS,
                  WeightOptions.NUM_VISIBLE_AGENTS,
                  WeightOptions.PREVIOUS_HEAD_ANGLES,
                  WeightOptions.AVG_PERCEPTION_STRENGTHS]
weight_size = len(weight_options)
output_size = 1
weights = [0.0,0.41323149558909394,0.0,0.0,0.48122849443372817,0.10554000997717788,0.0]

nn = NeuralNetwork()
fully_connected_layer = FullyConnectedLayer(input_size=weight_size, output_size=output_size)
fully_connected_layer.set_weights(weights=weights)
nn.add(fully_connected_layer)
nn.add(ActivationLayer(activation=snn.tanh, activation_prime=snn.tanh_prime))

n_agents = 7
n_steps = 5000
domain_size = (500, 500)
graph_freq = 10
visualize = True
follow = True
visualize_head_direction = True
visualize_vision_fields = True
animal_type = Pigeon()
start_position = (250, 250)

sim = ActiveElasticWithPerceptionStrengthAndHeadMovementSimulator(animal_type=animal_type, 
                                                                  num_agents=n_agents, 
                                                                  domain_size=domain_size,
                                                                  start_position=start_position, 
                                                                  model=nn,
                                                                  weight_options=weight_options,
                                                                  visualize=visualize, 
                                                                  follow=follow, 
                                                                  visualize_head_direction=visualize_head_direction,
                                                                  visualize_vision_fields=visualize_vision_fields,
                                                                  graph_freq=graph_freq)
sim.run(tmax=n_steps)
