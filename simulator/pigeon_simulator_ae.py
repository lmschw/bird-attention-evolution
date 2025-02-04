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

class PigeonSimulatorAe:
    def __init__(self, bird_type, num_agents, env_size, start_position,
                 model=None,
                 visualize=True, visualize_head_direction=True, follow=True, graph_freq=5):
        self.bird_type = bird_type
        self.num_agents = num_agents
        self.env_size = env_size
        self.start_position = start_position
        self.model = model
        self.visualize = visualize
        self.visualize_head_direction = visualize_head_direction
        self.follow = follow
        self.graph_freq = graph_freq
        self.curr_agents = None
        self.centroid_trajectory = []
        self.fly_lengths = []
        self.collective_fly_lengths = []
        self.n_leaders = 0
        self.historic_leaders = 0
        self.flights_completed = 0
        self.leaders_demoted = 0
        self.states = []
        self.centroid_trajectory = []
        self.stabilities = np.array([])
        self.states = []
        self.initialize()

    def initialize(self):
        self.init_agents(self.num_agents)
        self.sigmas = np.full(self.num_agents, SIGMA)
        self.current_step = 0

        # Setup graph
        self.fig, self.ax = plt.subplots()
        self.ax.set_facecolor((0, 0, 0))  
        centroid_x, centroid_y = np.mean(self.curr_agents[:, 0]), np.mean(self.curr_agents[:, 1])
        if self.follow:
            self.ax.set_xlim(centroid_x - 7.5, centroid_x + 7.5)
            self.ax.set_ylim(centroid_y - 7.5, centroid_y + 7.5)
        else:
            self.ax.set_xlim(0, self.env_size[0])
            self.ax.set_ylim(0, self.env_size[1])


    def init_agents(self, n_agents):
        rng = np.random
        n_points_x = n_agents
        n_points_y = n_agents
        spacing = 0.8
        init_x = 0
        init_y = 0

        x_values = np.linspace(init_x, init_x + (n_points_x - 1) * spacing, n_points_x)
        y_values = np.linspace(init_y, init_y + (n_points_y - 1) * spacing, n_points_y)
        xx, yy = np.meshgrid(x_values, y_values)

        pos_xs = self.start_position[0] + xx.ravel() + (rng.random(n_points_x * n_points_y) * spacing * 0.5) - spacing * 0.25
        pos_ys = self.start_position[1] + yy.ravel() + (rng.random(n_points_x * n_points_y) * spacing * 0.5) - spacing * 0.25
        pos_hs = (rng.random(n_points_x * n_points_x) * 2 * np.pi) - np.pi

        num_agents = len(pos_xs)
        self.num_agents = num_agents

        head_angles = np.zeros(self.num_agents)

        self.curr_agents = np.column_stack([pos_xs, pos_ys, pos_hs, head_angles])

    def graph_agents(self):
        """
        Visualizes the state of the simulation with matplotlib

        """  
        self.ax.clear()

        # Draw agents
        self.ax.scatter(self.curr_agents[:, 0], self.curr_agents[:, 1], color="white", s=15)
        self.ax.quiver(self.curr_agents[:, 0], self.curr_agents[:, 1],
                    np.cos(self.curr_agents[:, 2]), np.sin(self.curr_agents[:, 2]),
                    color="white", width=0.005, scale=40)
        
        if self.visualize_head_direction:
            self.ax.quiver(self.curr_agents[:, 0], self.curr_agents[:, 1],
                        np.cos(self.curr_agents[:, 3]), np.sin(self.curr_agents[:, 3]),
                        color="yellow", width=0.005, scale=50)
        """ 
        # Draw Trajectory
        if len(self.centroid_trajectory) > 1:
            x_traj, y_traj = zip(*self.centroid_trajectory)
            self.ax.plot(x_traj, y_traj, color="orange")
        """


        self.ax.set_facecolor((0, 0, 0))

        centroid_x, centroid_y = np.mean(self.curr_agents[:, 0]), np.mean(self.curr_agents[:, 1])
        if self.follow:
            self.ax.set_xlim(centroid_x-10, centroid_x+10)
            self.ax.set_ylim(centroid_y-10, centroid_y+10)
        else:
            self.ax.set_xlim(0, self.env_size[0])
            self.ax.set_ylim(0, self.env_size[1])

        plt.pause(0.000001)

    def compute_distances_and_angles(self):
        """
        Computes and returns the distances and its x and y elements for all pairs of agents

        """
        headings = self.curr_agents[:, 2]

        # Build meshgrid 
        pos_xs = self.curr_agents[:, 0]
        pos_ys = self.curr_agents[:, 1]
        xx1, xx2 = np.meshgrid(pos_xs, pos_xs)
        yy1, yy2 = np.meshgrid(pos_ys, pos_ys)

        # Calculate distances
        x_diffs = xx1 - xx2
        y_diffs = yy1 - yy2
        distances = np.sqrt(np.multiply(x_diffs, x_diffs) + np.multiply(y_diffs, y_diffs))  
        distances[distances > self.bird_type.sensing_range] = np.inf
        distances[distances == 0.0] = np.inf
        #print(f"Dists: {distances}")
        

        # Calculate angles in the local frame of reference
        headings = self.curr_agents[:, 2]
        angles = self.wrap_to_pi(np.arctan2(y_diffs, x_diffs) - headings[:, np.newaxis])
        # print(f"Angles: {angles}")

        return distances, angles
    

    def get_pi_elements(self, distances_conspecifics, angles_conspecifics, perception_strengths_conspecifics):
        """
        Calculates the x and y components of the proximal control vector

        """  
        #forces_old = -EPSILON * (2 * (self.sigmas[:, np.newaxis] ** 4 / distances ** 5) - (self.sigmas[:, np.newaxis] ** 2 / distances ** 3))
        forces_conspecifics = -EPSILON * perception_strengths_conspecifics * PERCEPTION_STRENGTH_MODIFIER * (2 * (self.sigmas[:, np.newaxis] ** 4 / distances_conspecifics ** 5) - (self.sigmas[:, np.newaxis] ** 2 / distances_conspecifics ** 3))
        forces_conspecifics[distances_conspecifics == np.inf] = 0.0


        p_x_conspecifics = np.sum(np.multiply(forces_conspecifics, np.cos(angles_conspecifics)), axis=1)
        p_y_conspecifics = np.sum(np.multiply(forces_conspecifics, np.sin(angles_conspecifics)), axis=1)

        p_x = p_x_conspecifics
        p_y = p_y_conspecifics

        return p_x, p_y

    def get_hi_elements(self):
        """
        Calculates the x and y components of the heading alignment vector

        """  
        headings = self.curr_agents[:, 2]

        # All this is doing is getting the vectorial avg of the headings
        alignment_coss = np.sum(np.cos(headings))
        alignment_sins = np.sum(np.sin(headings))
        alignment_angs = np.arctan2(alignment_sins, alignment_coss)
        alignment_mags = np.sqrt(alignment_coss**2 + alignment_sins**2)

        h_x = alignment_mags * np.cos(alignment_angs - headings)
        h_y = alignment_mags * np.sin(alignment_angs - headings)

        return h_x, h_y
    
    def get_gi_elements(self, distances, angles):
        """
        Calculates the x and y components of the goal direction vector

        """  
        pass

    def compute_fi(self):
        """
        Computes the virtual force vector components

        """  
        dists_conspecifics, angles_conspecifics = self.compute_distances_and_angles()
        perception_strengths_conspecifics, min_agent = pstrength.compute_perception_strengths(angles_conspecifics, dists_conspecifics, self.bird_type)


        p_x, p_y = self.get_pi_elements(distances_conspecifics=dists_conspecifics,
                                        angles_conspecifics=angles_conspecifics,
                                        perception_strengths_conspecifics=perception_strengths_conspecifics)
        h_x, h_y = self.get_hi_elements()

        f_x = ALPHA * p_x + BETA * h_x 
        f_y = ALPHA * p_y + BETA * h_y 
        # print(f"Fx: {f_x}")
        # print(f"Fy: {f_y}")
        

        return f_x, f_y
    
    def compute_u_w(self, f_x, f_y):
        """
        Computes u and w given the components of Fi

        """
        u = K1 * f_x + UC
        u[u > UMAX] = UMAX
        u[u < 0] = 0.0

        w = K2 * f_y
        w[w > WMAX] = WMAX
        w[w < -WMAX] = -WMAX

        return u, w
    
    def update_head_orientations(self, agents):
        if self.model:
            distances, angles = self.compute_distances_and_angles()
            perception_strengths_conspecifics, min_agent = pstrength.compute_perception_strengths(angles, distances, self.bird_type)

            closest_distances = np.min(distances, axis=1)
            average_bearings = np.average(angles, axis=1)
            num_visible_agents = np.count_nonzero(perception_strengths_conspecifics, axis=1)
            previous_head_angles = agents[:,3]
            
            new_head_angles = []
            for i in range(self.num_agents):
                new_head_angles.append(self.model.predict([[closest_distances[i], average_bearings[i], num_visible_agents[i], previous_head_angles[i]]])[0][0][0])
            new_head_orientations = new_head_angles
        else:
            new_head_orientations = agents[:,3]
        return new_head_orientations
    
    def update_agents(self):
        """
        Updates agents duhh

        """  
        # Calculate forces
        f_x, f_y = self.compute_fi()
        u, w = self.compute_u_w(f_x, f_y)

        # Project to local frame
        headings = self.curr_agents[:, 2]
        x_vel = np.multiply(u, np.cos(headings))
        y_vel = np.multiply(u, np.sin(headings))
        # print(f"X add: {x_vel}")
        # print(f"Y add: {y_vel}")

        # Update agents
        self.curr_agents[:, 0] = self.curr_agents[:, 0] + x_vel * DT
        self.curr_agents[:, 1] = self.curr_agents[:, 1] + y_vel * DT
        self.curr_agents[:, 2] = self.wrap_to_pi(self.curr_agents[:, 2] + w * DT)
        self.curr_agents[:, 3] = self.update_head_orientations(self.curr_agents)

    def target_reached(self):
        pos_xs = self.curr_agents[:, 0]
        pos_ys = self.curr_agents[:, 1]
        xx1, xx2 = np.meshgrid(pos_xs, self.target_position[0])
        yy1, yy2 = np.meshgrid(pos_ys, self.target_position[1])
        x_diffs = xx1 - xx2
        y_diffs = yy1 - yy2
        distances = np.sqrt(np.multiply(x_diffs, x_diffs) + np.multiply(y_diffs, y_diffs))  
        distances[distances <= self.target_radius] = 0
        if (np.count_nonzero(distances) / self.num_agents) <= self.target_success_percentage:
            print(f"target reached by {(np.count_nonzero(distances) / self.num_agents)*100}% of agents")
            return True
        return False

    def run(self, tmax=1000):
        t = 0
        while t < tmax:
            t += DT
            self.current_step = t

            # Update simulation
            self.update_agents()
            
            # Update experiment data
            self.states.append(self.curr_agents.copy())
            centroid_x, centroid_y = np.mean(self.curr_agents[:, 0]), np.mean(self.curr_agents[:, 1])
            self.centroid_trajectory.append((centroid_x, centroid_y))

            # if not (self.current_step % 250):
            #     print(f"------------------------ Iteration {self.current_step} ------------------------")

            if not (self.current_step % self.graph_freq) and self.visualize and self.current_step > 0:
                 self.graph_agents()
        # if self.target_reached():
           # print(f"target reached after: {t} timesteps")

        plt.close()
        return self.states


    # --------------------------------------------------------------------- Utils ---------------------------------------------------------------------

    def wrap_to_pi(self, x):
        """
        Wraps the angles to [-pi, pi]
        """
        x = x % (2 * np.pi)
        x = (x + (2 * np.pi)) % (2 * np.pi)

        x[x > np.pi] = x[x > np.pi] - (2 * np.pi)

        return x
        
