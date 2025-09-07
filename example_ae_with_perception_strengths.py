from animal_models.pigeon import Pigeon
from simulator.ae_simulator_with_perception_strength import ActiveElasticWithPerceptionStrengthSimulator
from animal_models.hawk import Hawk

n_agents = 10
n_steps = 10000
domain_size = (1000, 1000)
graph_freq = 10
visualize = True
follow = True
animal_type = Hawk()
start_position = (500, 500)
occlusion_active = False

sim = ActiveElasticWithPerceptionStrengthSimulator(animal_type=animal_type, num_agents=n_agents, domain_size=domain_size,
                        start_position=start_position, occlusion_active=occlusion_active,
                        visualize=visualize, follow=follow, graph_freq=graph_freq)
sim.run(tmax=n_steps)
