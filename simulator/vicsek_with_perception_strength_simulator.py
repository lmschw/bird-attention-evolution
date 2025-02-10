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
import simulator.weight_options as wo
import general.angle_conversion as ac

class VicsekSimulator:
    def __init__(self, animal_type, num_agents, domain_size, start_position, 
                 noise_amplitude, visualize=True, follow=True, graph_freq=5):
        self.animal_type = animal_type
        self.num_agents = num_agents
        self.domain_size = domain_size
        self.start_position = start_position
        self.noise_amplitude = noise_amplitude
        self.visualize = visualize
        self.follow = follow
        self.graph_freq = graph_freq
        self.curr_agents = None
        self.states = []

    def initialize(self):
        agents = self.init_agents()
        self.current_step = 0

        # Setup graph
        self.fig, self.ax = plt.subplots()
        self.ax.set_facecolor((0, 0, 0))  
        centroid_x, centroid_y = np.mean(self.curr_agents[:, 0]), np.mean(self.curr_agents[:, 1])
        if self.follow:
            self.ax.set_xlim(centroid_x - 7.5, centroid_x + 7.5)
            self.ax.set_ylim(centroid_y - 7.5, centroid_y + 7.5)
        else:
            self.ax.set_xlim(0, self.domain_size[0])
            self.ax.set_ylim(0, self.domain_size[1])

        return agents

    def init_agents(self):
        rng = np.random
        n_points_x = int(np.ceil(np.sqrt(self.num_agents)))
        n_points_y = int(np.ceil(np.sqrt(self.num_agents)))
        spacing = 0.8
        init_x = 0
        init_y = 0

        x_values = np.linspace(init_x, init_x + (n_points_x - 1) * spacing, n_points_x)
        y_values = np.linspace(init_y, init_y + (n_points_y - 1) * spacing, n_points_y)
        xx, yy = np.meshgrid(x_values, y_values)

        pos_xs = self.start_position[0] + xx.ravel() + (rng.random(n_points_x * n_points_y) * spacing * 0.5) - spacing * 0.25
        pos_ys = self.start_position[1] + yy.ravel() + (rng.random(n_points_x * n_points_y) * spacing * 0.5) - spacing * 0.25
        pos_hs = (rng.random(n_points_x * n_points_x) * 2 * np.pi) - np.pi

        indices = np.random.choice(range(len(pos_xs)), self.num_agents, replace=False)
        pos_xs = pos_xs[indices]
        pos_ys = pos_ys[indices]
        pos_hs = pos_hs[indices]

        num_agents = len(pos_xs)
        self.num_agents = num_agents

        self.curr_agents = np.column_stack([pos_xs, pos_ys, pos_hs])
        return self.curr_agents

    def graph_agents(self):
        self.ax.clear()

        self.ax.scatter(self.curr_agents[:, 0], self.curr_agents[:, 1], color="white", s=15)
        self.ax.quiver(self.curr_agents[:, 0], self.curr_agents[:, 1],
                    np.cos(self.curr_agents[:, 2]), np.sin(self.curr_agents[:, 2]),
                    color="white", width=0.005, scale=40)
    
        self.ax.set_facecolor((0, 0, 0))

        centroid_x, centroid_y = np.mean(self.curr_agents[:, 0]), np.mean(self.curr_agents[:, 1])
        if self.follow:
            self.ax.set_xlim(centroid_x-10, centroid_x+10)
            self.ax.set_ylim(centroid_y-10, centroid_y+10)
        else:
            self.ax.set_xlim(0, self.domain_size[0])
            self.ax.set_ylim(0, self.domain_size[1])

        plt.pause(0.000001)

    def generate_noise(self):
        return np.random.normal(scale=self.noise_amplitude, size=self.num_agents)

    def get_neighbours(self, agents):
        pos_xs = agents[:, 0]
        pos_ys = agents[:, 1]
        xx1, xx2 = np.meshgrid(pos_xs, pos_xs)
        yy1, yy2 = np.meshgrid(pos_ys, pos_ys)

        x_diffs = xx1 - xx2
        y_diffs = yy1 - yy2
        distances = np.sqrt(np.multiply(x_diffs, x_diffs) + np.multiply(y_diffs, y_diffs))  
        return distances <= self.animal_type.sensing_range
    
    def compute_distances_and_angles(self):
        headings = self.curr_agents[:, 2]

        pos_xs = self.curr_agents[:, 0]
        pos_ys = self.curr_agents[:, 1]
        xx1, xx2 = np.meshgrid(pos_xs, pos_xs)
        yy1, yy2 = np.meshgrid(pos_ys, pos_ys)

        x_diffs = xx1 - xx2
        y_diffs = yy1 - yy2
        distances = np.sqrt(np.multiply(x_diffs, x_diffs) + np.multiply(y_diffs, y_diffs))  
        distances[distances > self.animal_type.sensing_range] = np.inf
        distances[distances == 0.0] = np.inf        

        headings = self.curr_agents[:, 2]
        angles = ac.wrap_to_pi(np.arctan2(y_diffs, x_diffs) - headings[:, np.newaxis])

        return distances, angles

    def compute_new_orientations(self, neighbours, orientations):
        dists_conspecifics, angles_conspecifics = self.compute_distances_and_angles()
        perception_strengths_conspecifics, min_agent = pstrength.compute_perception_strengths(angles_conspecifics, dists_conspecifics, self.animal_type)
        orientations_grid = np.concatenate([[orientations]]*len(orientations))
        orientations_grid = np.where(neighbours, orientations_grid, 0)
        orientations_grid = orientations_grid * perception_strengths_conspecifics
        return np.average(orientations_grid, axis=1) + self.generate_noise()
    
    def compute_new_positions(self, agents):
        positions = np.column_stack((agents[:,0], agents[:,1]))
        orientations = ac.compute_u_v_coordinates_for_angles(angles=agents[:,2])
        positions += self.dt*orientations
        return positions[:,0], positions[:,1]
    
    def run(self, tmax):
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