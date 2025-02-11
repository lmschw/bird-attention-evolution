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

class CouzinZoneModelSimulator:

    def __init__(self, animal_type, num_agents, domain_size, start_position,
                 noise_amplitude,
                 visualize=True, follow=True, graph_freq=5):
        self.animal_type = animal_type
        self.num_agents = num_agents
        self.domain_size = domain_size
        self.start_position = start_position
        self.noise_amplitude = noise_amplitude
        self.visualize = visualize
        self.follow = follow
        self.graph_freq = graph_freq
        self.curr_agents = None
        self.centroid_trajectory = []
        self.states = []

    def compute_field_of_vision(self):
        min_angle = 0
        max_angle = 0
        for focus_area in self.animal_type.focus_areas:
            if (focus_area.azimuth_angle_position_horizontal - focus_area.angle_field_horizontal) < min_angle:
                min_angle = (focus_area.azimuth_angle_position_horizontal - focus_area.angle_field_horizontal)
            elif (focus_area.azimuth_angle_position_horizontal + focus_area.angle_field_horizontal) > max_angle:
                max_angle = (focus_area.azimuth_angle_position_horizontal + focus_area.angle_field_horizontal)
        return np.absolute((max_angle+2*np.pi)-(min_angle+2*np.pi))

    def initialize(self):
        agents = self.init_agents()
        self.current_step = 0

        self.field_of_vision_half = self.compute_field_of_vision() / 2

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

    def get_neighbour_unit_vectors(self, agents):
        positions = np.column_stack((agents[:,0], agents[:,1]))
        return positions[:,np.newaxis,:]-positions   
    
    def get_neighbours_repulsion_zone(self, distances):
        repulsion_radius = self.animal_type.preferred_distance_left_right[0]
        return distances < repulsion_radius
    
    def get_neighbours_alignment_zone(self, distances):
        repulsion_radius = self.animal_type.preferred_distance_left_right[0]
        attraction_radius = self.animal_type.preferred_distance_left_right[1]
        return (repulsion_radius < distances) & (distances < attraction_radius)
    
    def get_neighbours_attraction_zone(self, distances):
        attraction_radius = self.animal_type.preferred_distance_left_right[1]
        return (attraction_radius < distances) & (distances < self.animal_type.sensing_range)
    
    def computes_distances_and_angles(self, agents):
        # Build meshgrid 
        pos_xs = agents[:, 0]
        pos_ys = agents[:, 1]
        xx1, xx2 = np.meshgrid(pos_xs, pos_xs)
        yy1, yy2 = np.meshgrid(pos_ys, pos_ys)

        x_diffs = xx1 - xx2
        y_diffs = yy1 - yy2
        distances = np.sqrt(np.multiply(x_diffs, x_diffs) + np.multiply(y_diffs, y_diffs))  
        distances[distances > self.animal_type.sensing_range] = np.inf
        distances[distances == 0.0] = np.inf

        headings = agents[:,2]
        angles = ac.wrap_to_pi(np.arctan2(y_diffs, x_diffs) - headings[:, np.newaxis])

        return distances, angles

    def apply_vision_field(self, neighbours, bearings):
        return np.where(np.absolute(bearings) <= self.field_of_vision_half, neighbours, False)

    def compute_new_orientations(self, agents):
        distance_vectors = self.get_neighbour_unit_vectors(agents=agents)
        distances_x = distance_vectors[:,:,0]
        np.fill_diagonal(distances_x, np.inf)
        distances_y = distance_vectors[:,:,1]
        np.fill_diagonal(distances_y, np.inf)
        distance_vectors = np.dstack((distances_x, distances_y))

        distances, angles = self.computes_distances_and_angles(agents=agents)

        repulsed = self.get_neighbours_repulsion_zone(distances=distances)
        aligned = self.get_neighbours_alignment_zone(distances=distances)
        attracted = self.get_neighbours_attraction_zone(distances=distances)

        distances[distances == np.inf] = 0

        distances_repulsed = distances * repulsed
        norm_repulsed = np.linalg.norm(distances_repulsed, axis=1)
        new_orientations_repulsed = -np.sum(np.divide(distances_repulsed, norm_repulsed, out=np.zeros_like(distances_repulsed), where=norm_repulsed!=0), axis=1)
        #new_orientations_repulsed = -np.sum(distances_repulsed / np.linalg.norm(distances_repulsed, axis=1), axis=1)

        aligned = self.apply_vision_field(neighbours=aligned, bearings=angles)
        orientations_aligned = agents[:,2] * aligned
        norm_aligned = np.linalg.norm(orientations_aligned, axis=1)
        new_orientations_aligned = np.sum(np.divide(orientations_aligned, norm_aligned, out=np.zeros_like(orientations_aligned), where=norm_aligned!=0), axis=1)
        #new_orientations_aligned = np.sum(orientations_aligned / np.linalg.norm(orientations_aligned, axis=1), axis=1)

        attracted = self.apply_vision_field(neighbours=attracted, bearings=angles)
        distances_attracted = distances * attracted
        norm_attracted = np.linalg.norm(distances_attracted, axis=1)
        new_orientations_attracted = np.sum(np.divide(distances_attracted, norm_attracted, out=np.zeros_like(distances_attracted), where=norm_attracted!=0), axis=1)
        #new_orientations_attracted = -np.sum(distances_attracted / np.linalg.norm(distances_attracted, axis=1), axis=1)

        avg_orientations_aligned_attracted = (new_orientations_aligned + new_orientations_attracted) / 2

        print(f"rep: {np.count_nonzero(repulsed)}, align: {np.count_nonzero(aligned)}, attr: {np.count_nonzero(attracted)}, sums: {np.count_nonzero(repulsed) + np.count_nonzero(aligned) + np.count_nonzero(attracted)}")

        new_orientations = agents[:,2]
        new_orientations = np.where(((np.count_nonzero(aligned, axis=1) & np.count_nonzero(attracted, axis=1))), avg_orientations_aligned_attracted, new_orientations)
        new_orientations = np.where((np.count_nonzero(aligned, axis=1) == 0), new_orientations_attracted, new_orientations)
        new_orientations = np.where((np.count_nonzero(attracted, axis=1) == 0), new_orientations_aligned, new_orientations)
        new_orientations = np.where(np.count_nonzero(repulsed, axis=1), new_orientations_repulsed, new_orientations)


        """
        new_orientations = np.where(np.count_nonzero(repulsed, axis=1), new_orientations_repulsed, 0)
        new_orientations = np.where(((new_orientations == 0) & (np.count_nonzero(attracted, axis=1) == 0)), new_orientations_aligned, new_orientations)
        new_orientations = np.where(((new_orientations == 0) & (np.count_nonzero(aligned, axis=1) == 0)), new_orientations_attracted, new_orientations)
        new_orientations = np.where(((new_orientations == 0) & ((np.count_nonzero(aligned, axis=1) & np.count_nonzero(attracted, axis=1)))), avg_orientations_aligned_attracted, new_orientations)
        new_orientations = np.where((new_orientations == 0), agents[:,2], new_orientations)
        """
        new_orientations += self.generate_noise()
        new_orientations = np.where(((new_orientations < agents[:,2])&((agents[:,2]-new_orientations) > self.animal_type.max_turn_angle)), (agents[:,2]-self.animal_type.max_turn_angle), new_orientations)
        new_orientations = np.where(((new_orientations > agents[:,2])&((new_orientations-agents[:,2]) > self.animal_type.max_turn_angle)), (agents[:,2]+self.animal_type.max_turn_angle), new_orientations)
        return new_orientations
    
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
            agents[:,2] = self.compute_new_orientations(agents=agents)

            if not (self.current_step % self.graph_freq) and self.visualize and self.current_step > 0:
                 self.graph_agents()

            agent_history.append(agents)
            
        plt.close()
        return np.array(agent_history)