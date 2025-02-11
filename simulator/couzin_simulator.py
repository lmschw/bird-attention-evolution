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

"""
Implements Couzin et al.'s zone model.
"""

class CouzinZoneModelSimulator:
    def __init__(self, animal_type, num_agents, domain_size, start_position,
                 noise_amplitude,
                 visualize=True, follow=True, graph_freq=5):
        """
        Params:
            - animal_type (Animal): the type of animal
            - num_agents (int): the number of agents
            - domain_size (tuple of ints): how big the domain is (not bounded by these values though)
            - start_position (tuple of ints): around which points the agents should initially be placed
            - noise_amplitude (float): how much noise should be added to the orientation updates
            - visualize (boolean) [optional, default=True]: whether the results should be visualized directly
            - follow (boolean) [optional, default=True]: whether the visualization should follow the centroid of the swarm or whether it should show the whole domain
            - graph_freq (int) [optional, default=5]: how often the visualization should be updated
        """
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
        """
        Determines a single angle in radians that represents half the field of vision that is centered around 0.
        """
        min_angle = 0
        max_angle = 0
        for focus_area in self.animal_type.focus_areas:
            if (focus_area.azimuth_angle_position_horizontal - focus_area.angle_field_horizontal) < min_angle:
                min_angle = (focus_area.azimuth_angle_position_horizontal - focus_area.angle_field_horizontal)
            elif (focus_area.azimuth_angle_position_horizontal + focus_area.angle_field_horizontal) > max_angle:
                max_angle = (focus_area.azimuth_angle_position_horizontal + focus_area.angle_field_horizontal)
        return np.absolute((max_angle+2*np.pi)-(min_angle+2*np.pi))

    def initialize(self):
        """
        Initialises the agents, domain and field of vision.
        """
        agents = self.init_agents()
        self.current_step = 0

        self.field_of_vision_half = self.compute_field_of_vision() / 2 # todo add field of vision as limitation in base model

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
        """
        Initialises the agents (positions and orientations).
        """
        rng = np.random
        n_points_x = int(np.ceil(np.sqrt(self.num_agents)))
        n_points_y = int(np.ceil(np.sqrt(self.num_agents)))
        spacing = self.animal_type.preferred_distance_left_right[1]
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
        """
        Redraws the visualization for the current positions and orientations of the agents.
        """
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

    def compute_distances_and_angles(self):
        """
        Computes the distances and bearings between the agents.
        """
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
        angles = np.arctan2(y_diffs, x_diffs) - headings[:, np.newaxis]

        return distances, angles

    def get_repulsion_neighbours(self, distances):
        """
        Returns a boolean array representing which other agents are within the repulsion zone
        """
        return distances < self.animal_type.preferred_distance_left_right[0]
    
    def get_alignment_neighbours(self, distances):
        """
        Returns a boolean array representing which other agents are within the alignment zone
        """
        return (distances >= self.animal_type.preferred_distance_left_right[0]) & (distances < self.animal_type.preferred_distance_left_right[1])
    
    def get_attraction_neighbours(self, distances):
        """
        Returns a boolean array representing which other agents are within the attraction zone
        """
        return (distances >= self.animal_type.preferred_distance_left_right[0]) & (distances <= self.animal_type.sensing_range)
    
    def get_repulsion_orientations(self, positions, neighbours):
        """
        Computes the new orientation based on repulsion for every agent.
        """
        # doubles the neighbour boolean to fit the shape of the bearings
        neighbours_2d =  np.repeat(neighbours[:,:, np.newaxis], 2, axis=2) 

        bearings = positions[:,np.newaxis,:] - positions
        # limits the bearings to the relevant neighbours
        bearings = bearings * neighbours_2d

        bearings_norm = np.repeat(np.linalg.norm(bearings, axis=2)[:,:, np.newaxis], 2, axis=2)
        rij = np.divide(bearings, bearings_norm, out=np.zeros_like(bearings), where=bearings_norm!=0) # prevents divide by zero
        rij_norm = np.linalg.norm(rij)
        orientations = np.sum(np.divide(rij, rij_norm, out=np.zeros_like(rij), where=rij_norm!=0), axis=1) # prevents divide by zero
        return ac.compute_angles_for_orientations(orientations=orientations)
    
    def get_alignment_orientations(self, headings, neighbours):
        """
        Computes the new orientation based on alignment for every agent.
        """
        # doubles the neighbour boolean to fit the shape of the bearings
        neighbours_2d =  np.repeat(neighbours[:,:, np.newaxis], 2, axis=2)

        orientations = ac.compute_u_v_coordinates_for_angles(headings)

        # limits the orientations to the relevant neighbours
        orientations = orientations * neighbours_2d
        norm_aligned = np.linalg.norm(orientations, axis=1)
        new_orientations = np.sum(np.divide(orientations, norm_aligned, out=np.zeros_like(orientations), where=norm_aligned!=0), axis=1) # prevents divide by zero
        return ac.compute_angles_for_orientations(orientations=new_orientations)
    
    def get_attraction_orientations(self, positions, neighbours):
        """
        Computes the new orientation based on attraction for every agent.
        """
        # doubles the neighbour boolean to fit the shape of the bearings
        neighbours_2d =  np.repeat(neighbours[:,:, np.newaxis], 2, axis=2)

        bearings = positions[:,np.newaxis,:] - positions

        # limits the bearings to the relevant neighbours
        bearings = bearings * neighbours_2d
        bearings_norm = np.repeat(np.linalg.norm(bearings, axis=2)[:,:, np.newaxis], 2, axis=2)
        rij = np.divide(bearings, bearings_norm, out=np.zeros_like(bearings), where=bearings_norm!=0) # prevents divide by zero
        rij_norm = np.linalg.norm(rij)
        orientations = -np.sum(np.divide(rij, rij_norm, out=np.zeros_like(rij), where=rij_norm!=0), axis=1) # prevents divide by zero
        return ac.compute_angles_for_orientations(orientations=orientations)
    
    def compute_new_orientations(self, agents):
        """
        Computes the new orientations based on the following logic:
        1) if there are any neighbours within the repulsion zone, then the orientation based on those neighbours is used
        2) if there are neighbours in both the alignment and attraction zones, then the average of the orientations of these two zones is used
        3) if there are neighbours in the alignment or attraction zone but not both, then the respective orientation is used
        4) if there are no neighbours at all, the orientation remains unchanged
        """
        positions = np.column_stack((agents[:,0], agents[:,1]))
        distances, angles = self.compute_distances_and_angles()

        # determine which neighbour falls into which zone if any at all
        repulsed = self.get_repulsion_neighbours(distances=distances)
        aligned = self.get_alignment_neighbours(distances=distances)
        attracted = self.get_attraction_neighbours(distances=distances)

        # compute the orientation that each zone yields for each agent
        repulsion_orientations = self.get_repulsion_orientations(positions=positions, neighbours=repulsed)
        alignment_orientations = self.get_alignment_orientations(headings=agents[:,2], neighbours=aligned)
        attraction_orientations = self.get_attraction_orientations(positions=positions, neighbours=attracted)

        # compute the average of the alignment and attraction zones
        avg_orientations_aligned_attracted = (alignment_orientations + attraction_orientations) / 2

        # helper variables to make the code more readable representing whether there are neighbours in the respective zone
        # for each agent
        has_repulsed = np.count_nonzero(repulsed, axis=1) > 0
        has_aligned = np.count_nonzero(aligned, axis=1) > 0
        has_attracted = np.count_nonzero(attracted, axis=1) > 0

        # determine the new orientation
        new_orientations = np.where((has_aligned), alignment_orientations, agents[:,2])
        new_orientations = np.where((has_attracted), attraction_orientations, new_orientations)
        new_orientations = np.where((has_aligned & has_attracted), avg_orientations_aligned_attracted, new_orientations)
        new_orientations = np.where(has_repulsed, repulsion_orientations, new_orientations)

        return new_orientations
    
    def compute_new_positions(self, agents):
        """
        Update the new position based on the current position and orientation
        """
        positions = np.column_stack((agents[:,0], agents[:,1]))
        orientations = ac.compute_u_v_coordinates_for_angles(agents[:,2])
        positions += orientations * self.dt
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
            agents[:,2] = self.compute_new_orientations(agents=agents)

            if not (self.current_step % self.graph_freq) and self.visualize and self.current_step > 0:
                 self.graph_agents()

            agent_history.append(agents)
            
        plt.close()
        return np.array(agent_history) 