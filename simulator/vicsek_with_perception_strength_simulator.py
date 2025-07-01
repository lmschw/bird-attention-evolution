import numpy as np

import vision.perception_strength as pstrength

from simulator.vicsek_simulator import VicsekSimulator

"""
Implementation of the Vicsek model with perception strengths.
"""

class VicsekWithPerceptionStrengthSimulator(VicsekSimulator):
    def __init__(self, animal_type, num_agents, domain_size, start_position, 
                 noise_amplitude, landmarks=[], visualize=True, visualize_ids=False, follow=True, graph_freq=5):
        """
        Params:
            - animal_type (Animal): the type of animal 
            - num_agents (int): the number of animals within the domain
            - domain_size (tuple of ints): the size of the domain, though it is not strictly bounded and used for display only
            - start_position (tuple of 2 ints): the position around which the agents are initially distributed
            - noise_amplitude (float) [optional, default=0]: the amount of noise that is added to the orientation updates
            - visualize (boolean) [optional, default=True]: whether the simulation should be visualized immediately
            - follow (boolean) [optional, default=True]: whether the visualization should follow the centroid of the swarm or whether it should show the whole domain
            - graph_freq (int) [optional, default=5]: how often the visualization should be updated
        """
        super().__init__(animal_type=animal_type,
                         num_agents=num_agents,
                         domain_size=domain_size,
                         start_position=start_position,
                         landmarks=landmarks,
                         noise_amplitude=noise_amplitude,
                         visualize=visualize,
                         visualize_ids=visualize_ids,
                         follow=follow,
                         graph_freq=graph_freq)

    def compute_new_orientations(self, neighbours, orientations):
        """
        Computes the new orientations for all agents.
        """
        dists_conspecifics, angles_conspecifics = self.compute_distances_and_angles()
        perception_strengths_conspecifics = pstrength.compute_perception_strengths(distances=dists_conspecifics, angles=angles_conspecifics, shape=(self.num_agents, self.num_agents), animal_type=self.animal_type)
        orientations_grid = np.concatenate([[orientations]]*len(orientations))
        orientations_grid = np.where(neighbours, orientations_grid, 0)
        orientations_grid = orientations_grid * perception_strengths_conspecifics
        return np.average(orientations_grid, axis=1) + self.generate_noise()
    