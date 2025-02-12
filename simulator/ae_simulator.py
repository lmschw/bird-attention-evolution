import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import levy_stable, circmean, circvar
import pickle
from datetime import datetime
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

import general.angle_conversion as ac

from simulator.base_simulator import BaseSimulator

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
DT = 0.05
#DT = 1
DES_DIST = SIGMA * 2**(1/2)

"""
An implementation of Active Elastic
"""

class ActiveElasticSimulator(BaseSimulator):
    def __init__(self, animal_type, num_agents, domain_size, start_position, visualize=True, follow=True, graph_freq=5):
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
        
    def initialize(self):
        self.sigmas = np.full(self.num_agents, SIGMA)
        return super().initialize()

    def get_pi_elements(self, distances, angles):
        """
        Computes the proximal control vector.
        """
        forces = -EPSILON * (2 * (self.sigmas[:, np.newaxis] ** 4 / distances ** 5) - (self.sigmas[:, np.newaxis] ** 2 / distances ** 3))
        forces[distances == np.inf] = 0.0

        p_x = np.sum(np.multiply(forces, np.cos(angles)), axis=1)
        p_y = np.sum(np.multiply(forces, np.sin(angles)), axis=1)

        return p_x, p_y

    def get_hi_elements(self):
        """
        Computes the heading component.
        """
        headings = self.curr_agents[:, 2]

        alignment_coss = np.sum(np.cos(headings))
        alignment_sins = np.sum(np.sin(headings))
        alignment_angs = np.arctan2(alignment_sins, alignment_coss)
        alignment_mags = np.sqrt(alignment_coss**2 + alignment_sins**2)

        h_x = alignment_mags * np.cos(alignment_angs - headings)
        h_y = alignment_mags * np.sin(alignment_angs - headings)

        return h_x, h_y

    def compute_fi(self):
        """
        Computes the force components.
        """
        dists_conspecifics, angles_conspecifics = self.compute_distances_and_angles()

        p_x, p_y = self.get_pi_elements(dists_conspecifics, angles_conspecifics)
        h_x, h_y = self.get_hi_elements()

        f_x = ALPHA * p_x + BETA * h_x 
        f_y = ALPHA * p_y + BETA * h_y 

        return f_x, f_y
    
    def compute_u_w(self, f_x, f_y):
        """
        Computes u and w based on the force components.
        """
        u = K1 * f_x + UC
        u[u > UMAX] = UMAX
        u[u < 0] = 0.0

        w = K2 * f_y
        w[w > WMAX] = WMAX
        w[w < -WMAX] = -WMAX

        return u, w
    
    def update_agents(self):
        """
        Updates the agents' positions and orientations based on the spring force.
        """
        f_x, f_y = self.compute_fi()
        u, w = self.compute_u_w(f_x, f_y)

        headings = self.curr_agents[:, 2]
        x_vel = np.multiply(u, np.cos(headings))
        y_vel = np.multiply(u, np.sin(headings))

        self.curr_agents[:, 0] = self.curr_agents[:, 0] + x_vel * DT
        self.curr_agents[:, 1] = self.curr_agents[:, 1] + y_vel * DT
        self.curr_agents[:, 2] = ac.wrap_to_pi(self.curr_agents[:, 2] + w * DT)

    def run(self, tmax):
        """
        Runs the simulation for tmax time steps.
        """
        self.initialize()
        while self.current_step < tmax / DT:
            self.update_agents()
            self.current_step +=1

            self.states.append(self.curr_agents.copy())
            centroid_x, centroid_y = np.mean(self.curr_agents[:, 0]), np.mean(self.curr_agents[:, 1])
            self.centroid_trajectory.append((centroid_x, centroid_y))

            if not (self.current_step % self.graph_freq) and self.visualize and self.current_step > 0:
                self.graph_agents()

        plt.close()
        return self.states



        
