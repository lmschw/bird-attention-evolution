import numpy as np
import matplotlib.pyplot as plt

import general.angle_conversion as ac

from simulator.base_simulator import BaseSimulator

"""
Implementation of the Vicsek model.
"""

class VicsekSimulator(BaseSimulator):
    def __init__(self, animal_type, num_agents, domain_size, start_position, 
                 noise_amplitude, visualize=True, follow=True, graph_freq=5):
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
                         noise_amplitude=noise_amplitude,
                         visualize=visualize,
                         follow=follow,
                         graph_freq=graph_freq)

    def get_neighbours(self, agents):
        """
        Returns an array of booleans representing whether other agents are neighbours.
        """
        pos_xs = agents[:, 0]
        pos_ys = agents[:, 1]
        xx1, xx2 = np.meshgrid(pos_xs, pos_xs)
        yy1, yy2 = np.meshgrid(pos_ys, pos_ys)

        x_diffs = xx1 - xx2
        y_diffs = yy1 - yy2
        distances = np.sqrt(np.multiply(x_diffs, x_diffs) + np.multiply(y_diffs, y_diffs))  
        return distances <= self.animal_type.sensing_range

    def compute_new_orientations(self, neighbours, orientations):
        """
        Computes the new orientations for all agents.
        """
        orientations_grid = np.concatenate([[orientations]]*len(orientations))
        orientations_grid = np.where(neighbours, orientations_grid, 0)
        return np.average(orientations_grid, axis=1) + self.generate_noise()
    
    def compute_new_positions(self, agents):
        """
        Update the new position based on the current position and orientation
        """
        positions = np.column_stack((agents[:,0], agents[:,1]))
        orientations = ac.compute_u_v_coordinates_for_angles(angles=agents[:,2])
        positions += self.dt*orientations
        return positions[:,0], positions[:,1]
    
    def run(self, tmax):
        """
        Runs the simulation for tmax timesteps
        """
        agent_history = []
        agents = self.initialize()
        self.dt = 1

        for t in range(tmax):
            self.current_step = t

            self.agents = agents

            agents[:,0], agents[:,1] = self.compute_new_positions(agents=agents)
            self.agents = agents
            neighbours = self.get_neighbours(agents=agents)
            agents[:,2]= self.compute_new_orientations(neighbours=neighbours, orientations=agents[:,2])

            if not (self.current_step % self.graph_freq) and self.visualize and self.current_step > 0:
                 self.graph_agents()

            agent_history.append(agents)
            
        plt.close()
        return np.array(agent_history)