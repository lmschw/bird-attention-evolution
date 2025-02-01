from simulator.pigeon_simulator_2 import PigeonSimulator
from bird_models.pigeon import Pigeon

n_agents = 3
n_steps = 1000
env_size = 100
start_position = (25, 25)
graph_freq = 10
visualize = True
follow = True
bird_type = Pigeon()

sim = PigeonSimulator(num_agents=n_agents,
                        domain_size=env_size,
                        start_position=start_position,
                        bird_type=bird_type,
                        visualize=visualize,
                        follow=follow,
                        graph_freq=graph_freq)
sim.run(tmax=n_steps)

