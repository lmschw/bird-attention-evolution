
import numpy as np
import matplotlib.pyplot as plt

import general.angle_conversion as ac

class BaseSimulator:
    def __init__(self, animal_type, num_agents, domain_size, start_position, noise_amplitude=0, 
                 landmarks=[], visualize=True, follow=True, graph_freq=5):
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
        self.landmarks = landmarks
        self.visualize = visualize
        self.follow = follow
        self.graph_freq = graph_freq
        self.curr_agents = None
        self.centroid_trajectory = []
        self.states = []

    def initialize(self):
        """
        Initialises the graph and the agents.
        """
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

        for landmark in self.landmarks:
            self.ax.add_patch(landmark.get_patch_for_display())
            self.ax.annotate(landmark.id, landmark.get_annotation_point(), color="white")

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
    
    def generate_noise(self):
        """
        Generates noise.
        """
        return np.random.normal(scale=self.noise_amplitude, size=self.num_agents)
