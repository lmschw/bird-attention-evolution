from animal_models.pigeon import Pigeon
from simulator.ae_simulator_with_perception_strength import ActiveElasticWithPerceptionStrengthSimulator

n_agents = 10
n_steps = 10000
domain_size = (50, 50)
graph_freq = 10
visualize = True
follow = True
animal_type = Pigeon()
start_position = (10, 10)
occlusion_active = False

sim = ActiveElasticWithPerceptionStrengthSimulator(animal_type=animal_type, num_agents=n_agents, domain_size=domain_size,
                        start_position=start_position, occlusion_active=occlusion_active,
                        visualize=visualize, follow=follow, graph_freq=graph_freq)
sim.run(tmax=n_steps)
