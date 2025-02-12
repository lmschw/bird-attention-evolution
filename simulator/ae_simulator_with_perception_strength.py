import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import levy_stable, circmean, circvar
import pickle
from datetime import datetime
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

import vision.perception_strength as pstrength
import simulator.head_movement.weight_options as wo
import general.angle_conversion as ac

from simulator.ae_simulator import ActiveElasticSimulator

# AE Constants
EPSILON = 12
SIGMA = 0.7
SIGMA_MIN = 0.7
SIGMA_MAX = 0.7
UC = 0.05
UMAX = 0.1
WMAX = np.pi / 2
ALPHA = 2.0
BETA = 0.5
BETA_LEADER = 4
GAMMA = 1.0
G_GAIN = 1
K1 = 0.6
K2 = 0.05
L_0 = 0.5
K_REP = 2.0
DT = 1
DES_DIST = SIGMA * 2**(1/2)
PERCEPTION_STRENGTH_MODIFIER = 5

"""
An implementation of Active Elastic including perception strengths.
"""

class ActiveElasticWithPerceptionStrengthSimulator(ActiveElasticSimulator):
    def __init__(self, animal_type, num_agents, domain_size, start_position,
                 visualize=True, follow=True, graph_freq=5):
        """
        Params:
            - animal_type (Animal): the type of animal
            - num_agents (int): the number of agents
            - domain_size (tuple of ints): how big the domain is (not bounded by these values though)
            - start_position (tuple of ints): around which points the agents should initially be placed
            - visualize (boolean) [optional, default=True]: whether the results should be visualized directly
            - follow (boolean) [optional, default=True]: whether the visualization should follow the centroid of the swarm or whether it should show the whole domain
            - graph_freq (int) [optional, default=5]: how often the visualization should be updated
        """

        super().__init__(animal_type=animal_type,
                         num_agents=num_agents,
                         domain_size=domain_size,
                         start_position=start_position,
                         visualize=visualize,
                         follow=follow,
                         graph_freq=graph_freq)
    

    def get_pi_elements(self, distances_conspecifics, angles_conspecifics, perception_strengths_conspecifics):
        """
        Computes the proximal control vector.
        """
        forces_conspecifics = -EPSILON * perception_strengths_conspecifics * PERCEPTION_STRENGTH_MODIFIER * (2 * (self.sigmas[:, np.newaxis] ** 4 / distances_conspecifics ** 5) - (self.sigmas[:, np.newaxis] ** 2 / distances_conspecifics ** 3))
        forces_conspecifics[distances_conspecifics == np.inf] = 0.0

        p_x_conspecifics = np.sum(np.multiply(forces_conspecifics, np.cos(angles_conspecifics)), axis=1)
        p_y_conspecifics = np.sum(np.multiply(forces_conspecifics, np.sin(angles_conspecifics)), axis=1)

        p_x = p_x_conspecifics
        p_y = p_y_conspecifics

        return p_x, p_y

    def compute_fi(self):
        """
        Computes the force components.
        """
        dists_conspecifics, angles_conspecifics = self.compute_distances_and_angles()
        perception_strengths_conspecifics = pstrength.compute_perception_strengths(distances=dists_conspecifics, angles=angles_conspecifics, shape=(self.num_agents, self.num_agents), animal_type=self.animal_type)

        p_x, p_y = self.get_pi_elements(distances_conspecifics=dists_conspecifics,
                                        angles_conspecifics=angles_conspecifics,
                                        perception_strengths_conspecifics=perception_strengths_conspecifics)
        h_x, h_y = self.get_hi_elements()

        f_x = ALPHA * p_x + BETA * h_x 
        f_y = ALPHA * p_y + BETA * h_y 
        
        return f_x, f_y



        
