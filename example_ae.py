from animal_models.pigeon import Pigeon
from simulator.ae_simulator import ActiveElasticSimulator
from animal_models.hawk import Hawk

n_agents = 10
n_steps = 10000
domain_size = (500, 500)
graph_freq = 10
visualize = True
follow = True
animal_type = Hawk()
start_position = (250, 250)

sim = ActiveElasticSimulator(animal_type=animal_type, num_agents=n_agents, domain_size=domain_size,
                        start_position=start_position, 
                        visualize=visualize, follow=follow, graph_freq=graph_freq)
sim.run(tmax=n_steps)
