from animal_models.pigeon import Pigeon
from area_models.landmark import Landmark
from simulator.vicsek_with_perception_strength_simulator import VicsekWithPerceptionStrengthSimulator


n_agents = 10
n_steps = 10000
domain_size = (50, 50)
graph_freq = 10
visualize = True
follow = True
animal_type = Pigeon()
start_position = (10, 10)
noise_amplitude = 0.1

sim = VicsekWithPerceptionStrengthSimulator(animal_type=animal_type,
                      num_agents=n_agents,
                      domain_size=domain_size,
                      start_position=start_position,
                      noise_amplitude=noise_amplitude,
                      visualize=visualize,
                      follow=follow,
                      graph_freq=graph_freq)
sim.run(tmax=n_steps)
