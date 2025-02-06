import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shapely.ops as shpops
from shapely import Point

from bird_models.pigeon import Pigeon
from bird_models.focus_area import FocusArea
import geometry.normalisation as normal
import simulator.weight_options as wo
from simulator.pigeon_simulator_2 import PigeonSimulator

DIST_MOD = 0.001

class PigeonSimulatorWithPredators(PigeonSimulator):
    def __init__(self, num_prey, bird_type_prey, num_predators, bird_type_predator, domain_size, start_position_prey, 
                 start_position_predator, landmarks=[], noise_amplitude=0, social_weight=1, environment_weight=1,
                 other_type_weight=1, limit_turns=True, use_distant_dependent_zone_factors=True, weight_options=[], 
                 model=None, single_speed=True, visualize=True, visualize_vision_fields_prey=0, 
                 visualize_vision_fields_predator=0, visualize_head_direction=True, follow=False, graph_freq=5):
        super().__init__(num_agents=num_prey,
                         bird_type=bird_type_prey,
                         domain_size=domain_size,
                         start_position=start_position_prey,
                         landmarks=landmarks,
                         noise_amplitude=noise_amplitude,
                         social_weight=social_weight,
                         environment_weight=environment_weight,
                         limit_turns=limit_turns,
                         use_distant_dependent_zone_factors=use_distant_dependent_zone_factors,
                         weight_options=weight_options,
                         model=model,
                         single_speed=single_speed,
                         visualize=visualize,
                         visualize_vision_fields=visualize_vision_fields_prey,
                         visualize_head_direction=visualize_head_direction,
                         follow=follow,
                         graph_freq=graph_freq)
        
        self.num_prey = num_prey
        self.bird_type_prey = bird_type_prey
        self.num_predators = num_predators
        self.bird_type_predator = bird_type_predator
        self.start_position_prey = start_position_prey
        self.start_position_predator = start_position_predator
        self.other_type_weight = other_type_weight
        self.visualize_vision_fields_prey = visualize_vision_fields_prey
        self.visualize_vision_fields_predator = visualize_vision_fields_predator

    def initialize(self):
        prey, predators = self.init_agents()

        # Setup graph
        self.fig, self.ax = plt.subplots()
        self.ax.set_facecolor((0, 0, 0))  
        centroid_x, centroid_y = np.mean(prey[:, 0]), np.mean(prey[:, 1])
        if self.follow:
            self.ax.set_xlim(centroid_x - 7.5, centroid_x + 7.5)
            self.ax.set_ylim(centroid_y - 7.5, centroid_y + 7.5)
        else:
            self.ax.set_xlim(0, self.domain_size[0])
            self.ax.set_ylim(0, self.domain_size[1])

        return prey, predators
    
    def init_agents(self):
        prey, self.colours_prey = self.init_agents_for_agent_type(num_agents=self.num_prey, 
                                               bird_type=self.bird_type_prey, 
                                               start_position=self.start_position_prey,
                                               visualize_vision_fields=self.visualize_vision_fields_prey)
        predators, self.colours_predator = self.init_agents_for_agent_type(num_agents=self.num_predators, 
                                               bird_type=self.bird_type_predator, 
                                               start_position=self.start_position_predator,
                                               visualize_vision_fields=self.visualize_vision_fields_predator)
        return prey, predators

    def init_agents_for_agent_type(self, num_agents, bird_type, start_position, visualize_vision_fields):
        rng = np.random
        n_points_x = int(np.ceil(np.sqrt(num_agents)))
        n_points_y = int(np.ceil(np.sqrt(num_agents)))
        spacing = np.average(bird_type.preferred_distance_left_right)
        init_x = 0
        init_y = 0

        x_values = np.linspace(init_x, init_x + (n_points_x - 1) * spacing, n_points_x)
        y_values = np.linspace(init_y, init_y + (n_points_y - 1) * spacing, n_points_y)
        xx, yy = np.meshgrid(x_values, y_values)

        pos_xs = start_position[0] + xx.ravel() + (rng.random(n_points_x * n_points_y) * spacing * 0.5) - spacing * 0.25
        pos_ys = start_position[1] + yy.ravel() + (rng.random(n_points_x * n_points_y) * spacing * 0.5) - spacing * 0.25
        pos_hs = (rng.random(n_points_x * n_points_x) * 2 * np.pi) - np.pi

        indices = np.random.choice(range(len(pos_xs)), num_agents, replace=False)
        pos_xs = pos_xs[indices]
        pos_ys = pos_ys[indices]
        pos_hs = pos_hs[indices]

        if self.single_speed:
            speeds = np.full(num_agents, bird_type.speeds[1])
        else:
            speeds = np.random.uniform(bird_type.speeds[0], bird_type.speeds[2], num_agents)

        head_angles = np.zeros(num_agents)

        #print(f"Head angles: {head_angles}")

        colours = np.random.uniform(0, 1, (visualize_vision_fields, 3))

        return np.column_stack([pos_xs, pos_ys, pos_hs, speeds, head_angles]), colours
    
    def graph_vision_fields(self, agents, bird_type, visualize_vision_fields, colours):
        for i in range(visualize_vision_fields):
            for focus_area in bird_type.focus_areas:
                focus_angle = agents[i,2] + agents[i,4] + focus_area.azimuth_angle_position_horizontal + 2 * np.pi
                start_angle = np.rad2deg(focus_angle - focus_area.angle_field_horizontal) 
                end_angle = np.rad2deg(focus_angle + focus_area.angle_field_horizontal) 
                if focus_area.comfortable_distance[1] == np.inf:
                    distance = 1000
                else:
                    distance = focus_area.comfortable_distance[1]
                #print(f"az={focus_area.azimuth_angle_position_horizontal}, h={focus_area.angle_field_horizontal}, o={self.wrap_to_2_pi(agents[0,2])}, st={start_angle}, e={end_angle}")
                wedge = mpatches.Wedge((agents[i,0], agents[i,1]), distance, start_angle, end_angle, ec="none", color=colours[i])
                self.ax.add_patch(wedge)

    def graph_arrows(self, agents, colour):
        uv_coords = self.compute_u_v_coordinates_for_angles(agents[:,2])
        uv_coords_head = self.compute_u_v_coordinates_for_angles(agents[:,2] + agents[:,4])

        self.ax.scatter(agents[:, 0], agents[:, 1], color="white", s=15)

        self.ax.quiver(agents[:, 0], agents[:, 1],
                    uv_coords[:, 0], uv_coords[:, 1],
                    color=colour, width=0.005, scale=40)
        
        if self.visualize_head_direction:
            self.ax.quiver(agents[:, 0], agents[:, 1],
                        uv_coords_head[:, 0], uv_coords_head[:, 1],
                        color="yellow", width=0.005, scale=50)

    def graph_agents(self, prey, predators):
        """
        Visualizes the state of the simulation with matplotlib

        """  
        self.ax.clear()

        self.graph_vision_fields(prey, self.bird_type_prey, self.visualize_vision_fields_prey, self.colours_prey)
        self.graph_vision_fields(prey, self.bird_type_predator, self.visualize_vision_fields_predator, self.colours_predator)

        if self.environment_weight > 0:
            # Draw landmarks
            for landmark in self.landmarks:
                self.ax.add_patch(landmark.get_patch_for_display())
                self.ax.annotate(landmark.id, landmark.get_annotation_point(), color="white")

        # Draw prey
        self.graph_arrows(agents=prey, colour="white")
        self.graph_arrows(agents=predators, colour="red")

        self.ax.set_facecolor((0, 0, 0))

        centroid_x, centroid_y = np.mean(prey[:, 0]), np.mean(prey[:, 1])
        if self.follow:
            self.ax.set_xlim(centroid_x-10, centroid_x+10)
            self.ax.set_ylim(centroid_y-10, centroid_y+10)
        else:
            self.ax.set_xlim(0, self.domain_size[0])
            self.ax.set_ylim(0, self.domain_size[1])

        plt.pause(0.000001)    

    def compute_distances_and_angles_other_type(self, focus_group, other_group):
        """
        Computes and returns the distances and its x and y elements for all pairs of agents

        """
        # Build meshgrid 
        focus_group_xs = focus_group[:, 0]
        focus_group_ys = focus_group[:, 1]
        other_group_xs = other_group[:, 0]
        other_group_ys = other_group[:, 1]
        xx1, xx2 = np.meshgrid(focus_group_xs, other_group_xs)
        yy1, yy2 = np.meshgrid(focus_group_ys, other_group_ys)

        return self.compute_distances_and_angles(headings=focus_group[:,2], xx1=xx1, xx2=xx2, yy1=yy1, yy2=yy2)
    

    def compute_delta_orientations_away_from_predators(self, prey, predators):
        distances, angles = self.compute_distances_and_angles_other_type(prey, predators)
        match_factors = np.full((self.num_prey, self.num_predators), -1) # always repulsed
        side_factors = self.compute_side_factors(angles, shape=(self.num_prey, self.num_predators))
        vision_strengths = self.compute_vision_strengths(head_orientations=prey[:,4], distances=distances, angles=angles, shape=(self.num_prey, self.num_predators))
        return np.sum(match_factors * side_factors * vision_strengths, axis=1)

    def compute_delta_orientations_towards_prey(self, prey, predators):
        distances, angles = self.compute_distances_and_angles_other_type(predators, prey)
        match_factors = np.zeros((self.num_predators, self.num_prey)) # always attracted
        min_neighbours = np.min(distances.T, axis=1)
        match_factors = np.where(distances == min_neighbours, 1, match_factors)
        side_factors = self.compute_side_factors(angles, shape=(self.num_predators, self.num_prey))
        vision_strengths = self.compute_vision_strengths(head_orientations=predators[:,4], distances=distances, angles=angles, shape=(self.num_predators, self.num_prey))
        return np.sum(match_factors * side_factors * vision_strengths, axis=1)

    def compute_delta_orientations(self, prey, predators, is_prey):
        if is_prey:
            agents = prey
        else:
            agents = predators
        delta_orientations_conspecifics, distances, angles, vision_strengths = self.compute_delta_orientations_conspecifics(agents=agents)
        if len(self.landmarks) > 0:
            delta_orientations_landmarks = self.compute_delta_orientations_landmarks(agents=agents)
        else:
            delta_orientations_landmarks = 0

        if is_prey:
            delta_orientations_other_type = self.compute_delta_orientations_away_from_predators(prey, predators)
        else:
            delta_orientations_other_type = self.compute_delta_orientations_towards_prey(prey, predators)

        #print(delta_orientations_landmarks)
        delta_orientations = self.social_weight * delta_orientations_conspecifics + self.environment_weight * delta_orientations_landmarks + self.other_type_weight * delta_orientations_other_type
        delta_orientations = np.where((delta_orientations > self.bird_type.max_turn_angle), self.bird_type.max_turn_angle, delta_orientations)
        delta_orientations = np.where((delta_orientations < -self.bird_type.max_turn_angle), -self.bird_type.max_turn_angle, delta_orientations)
        return delta_orientations, distances, angles, vision_strengths
    
    def compute_new_orientations_for_type(self, prey, predators, is_prey):
        if is_prey:
            agents = prey
        else:
            agents = predators

        delta_orientations, distances, angles, vision_strengths = self.compute_delta_orientations(prey=prey, predators=predators, is_prey=is_prey)
        # add noise
        delta_orientations = delta_orientations + self.generate_noise()

        new_orientations = self.wrap_to_pi(agents[:,2] + delta_orientations)
        if self.model:
            new_head_orientations = self.move_heads(agents=agents, distances=distances, angles=angles, perception_strengths_conspecifics=vision_strengths)
        else:
            new_head_orientations = agents[:,4]
        return new_orientations, new_head_orientations

    def compute_new_orientations(self, prey, predators):
        self.num_agents = self.num_prey
        prey_orientations, prey_head_orientations = self.compute_new_orientations_for_type(prey, predators, is_prey=True)
        self.num_agents = self.num_predators
        predators_orientations, predators_head_orientations = self.compute_new_orientations_for_type(prey, predators, is_prey=False)
        return prey_orientations, prey_head_orientations, predators_orientations, predators_head_orientations
    
    def run(self, tmax):

        prey_history = []
        predator_history = []
        prey, predators = self.initialize()
        self.dt = 1

        for t in range(tmax):
            self.current_step = t

            self.prey = prey
            self.predators = predators

            prey[:,0], prey[:,1] = self.compute_new_positions(agents=prey)
            predators[:,0], predators[:,1] = self.compute_new_positions(agents=predators)

            self.prey = prey
            self.predators = predators
            
            prey_o, prey_ho, predator_o, predator_ho  = self.compute_new_orientations(prey=prey, predators=predators)

            prey[:,2] = prey_o
            prey[:,4] = prey_ho
            prey[:,2] = predator_o
            prey[:,4] = predator_ho

            if not (self.current_step % self.graph_freq) and self.visualize and self.current_step > 0:
                self.graph_agents(prey=prey, predators=predators)

            prey_history.append(prey)
            predator_history.append(predators)
            
        plt.close()
        return np.array(prey_history), np.array(predator_history)

    def wrap_to_pi(self, x):
        """
        Wraps the angles to [-pi, pi]
        """
        x = x % (2 * np.pi)
        x = (x + (2 * np.pi)) % (2 * np.pi)

        x[x > np.pi] = x[x > np.pi] - (2 * np.pi)

        return x
    
    def wrap_to_2_pi(self, x):
        return (2*np.pi*x) % (2*np.pi)
    
    
    def compute_u_v_coordinates_for_angles(self, angles):
        """
        Computes the (u,v)-coordinates based on the angle.

        Params:
            - angle (float): the angle in radians

        Returns:
            An array containing the [u, v]-coordinates corresponding to the angle.
        """
        # compute the uv-coordinates
        U = np.cos(angles)
        V = np.sin(angles)
    
        return np.column_stack((U,V))
    
    def compute_angles_for_orientations(self, orientations):
        """
        Computes the angle in radians based on the (u,v)-coordinates of the current orientation.

        Params:
            - orientation (array of floats): the current orientation in (u,v)-coordinates

        Returns:
            A float representin the angle in radians.
        """
        return np.arctan2(orientations[:, 1], orientations[:, 0])