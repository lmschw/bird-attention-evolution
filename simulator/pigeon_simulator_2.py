import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shapely.ops as shpops
from shapely import Point

from bird_models.pigeon import Pigeon
from bird_models.focus_area import FocusArea
import geometry.normalisation as normal
import simulator.weight_options as wo

DIST_MOD = 0.001

class PigeonSimulator:
    def __init__(self, num_agents, bird_type, domain_size, start_position, landmarks=[],
                 noise_amplitude=0, social_weight=1, environment_weight=1, limit_turns=True, 
                 use_distant_dependent_zone_factors=True, weight_options=[], model=None, 
                 single_speed=True, visualize=True, visualize_vision_fields=0, 
                 visualize_head_direction=True, follow=False, graph_freq=5):
        self.num_agents = num_agents
        self.bird_type = bird_type
        self.domain_size = domain_size
        self.start_position = start_position
        self.noise_amplitude = noise_amplitude
        self.landmarks = landmarks
        self.social_weight = social_weight,
        self.environment_weight = environment_weight
        self.limit_turns = limit_turns
        self.use_distant_dependent_zone_factors = use_distant_dependent_zone_factors
        self.weight_options = weight_options
        self.model = model
        self.single_speed = single_speed
        self.visualize = visualize
        self.visualize_vision_fields = visualize_vision_fields
        self.visualize_head_direction = visualize_head_direction
        self.follow = follow
        self.graph_freq = graph_freq
        self.centroid_trajectory = []

    def initialize(self):
        agents = self.init_agents()

        # Setup graph
        self.fig, self.ax = plt.subplots()
        self.ax.set_facecolor((0, 0, 0))  
        centroid_x, centroid_y = np.mean(agents[:, 0]), np.mean(agents[:, 1])
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
        spacing = np.average(self.bird_type.preferred_distance_left_right)
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

        if self.single_speed:
            speeds = np.full(self.num_agents, self.bird_type.speeds[1])
        else:
            speeds = np.random.uniform(self.bird_type.speeds[0], self.bird_type.speeds[2], self.num_agents)

        head_angles = np.zeros(self.num_agents)

        #print(f"Head angles: {head_angles}")

        self.colours = np.random.uniform(0, 1, (self.visualize_vision_fields, 3))

        return np.column_stack([pos_xs, pos_ys, pos_hs, speeds, head_angles])

    def graph_agents(self, agents):
        """
        Visualizes the state of the simulation with matplotlib

        """  
        self.ax.clear()

        for i in range(self.visualize_vision_fields):
            for focus_area in self.bird_type.focus_areas:
                focus_angle = agents[i,2] + agents[i,4] + focus_area.azimuth_angle_position_horizontal + 2 * np.pi
                start_angle = np.rad2deg(focus_angle - focus_area.angle_field_horizontal) 
                end_angle = np.rad2deg(focus_angle + focus_area.angle_field_horizontal) 
                if focus_area.comfortable_distance[1] == np.inf:
                    distance = 1000
                else:
                    distance = focus_area.comfortable_distance[1]
                #print(f"az={focus_area.azimuth_angle_position_horizontal}, h={focus_area.angle_field_horizontal}, o={self.wrap_to_2_pi(agents[0,2])}, st={start_angle}, e={end_angle}")
                wedge = mpatches.Wedge((agents[i,0], agents[i,1]), distance, start_angle, end_angle, ec="none", color=self.colours[i])
                self.ax.add_patch(wedge)

        if self.environment_weight > 0:
            # Draw landmarks
            for landmark in self.landmarks:
                self.ax.add_patch(landmark.get_patch_for_display())
                self.ax.annotate(landmark.id, landmark.get_annotation_point(), color="white")

        # Draw agents
        uv_coords = self.compute_u_v_coordinates_for_angles(agents[:,2])
        uv_coords_head = self.compute_u_v_coordinates_for_angles(agents[:,2] + agents[:,4])

        self.ax.scatter(agents[:, 0], agents[:, 1], color="white", s=15)

        self.ax.quiver(agents[:, 0], agents[:, 1],
                    uv_coords[:, 0], uv_coords[:, 1],
                    color="white", width=0.005, scale=40)
        
        if self.visualize_head_direction:
            self.ax.quiver(agents[:, 0], agents[:, 1],
                        uv_coords_head[:, 0], uv_coords_head[:, 1],
                        color="yellow", width=0.005, scale=50)

        # Draw Trajectory
        if len(self.centroid_trajectory) > 1:
            x_traj, y_traj = zip(*self.centroid_trajectory)
            self.ax.plot(x_traj, y_traj, color="orange")

        self.ax.set_facecolor((0, 0, 0))

        centroid_x, centroid_y = np.mean(agents[:, 0]), np.mean(agents[:, 1])
        if self.follow:
            self.ax.set_xlim(centroid_x-10, centroid_x+10)
            self.ax.set_ylim(centroid_y-10, centroid_y+10)
        else:
            self.ax.set_xlim(0, self.domain_size[0])
            self.ax.set_ylim(0, self.domain_size[1])

        plt.pause(0.000001)

    def move_heads(self, agents, distances, angles, perception_strengths_conspecifics):
        inputs = np.array([wo.get_input_value_for_weight_option(weight_option=option, bearings=agents[:,4], distances=distances, angles=angles, perception_strengths=perception_strengths_conspecifics) for option in self.weight_options])
        inputs = np.where(inputs == np.inf, wo.MAX_INPUT, inputs)
        new_head_angles = []
        for i in range(self.num_agents):
            new_head_angles.append(self.model.predict([inputs[:,i]])[0][0][0])
        return new_head_angles

    def compute_distances_and_angles(self, headings, xx1, xx2, yy1, yy2, transpose_for_angles=False):
        # Calculate distances
        x_diffs = xx1 - xx2
        y_diffs = yy1 - yy2
        distances = np.sqrt(np.multiply(x_diffs, x_diffs) + np.multiply(y_diffs, y_diffs))  
        distances[distances > self.bird_type.sensing_range] = np.inf
        distances[distances == 0.0] = np.inf
        #print(f"Dists: {distances}")
        

        # Calculate angles in the local frame of reference
        if transpose_for_angles:
            angles = self.wrap_to_pi(np.arctan2(y_diffs.T, x_diffs.T) - headings[:, np.newaxis])
        else:
            angles = self.wrap_to_pi(np.arctan2(y_diffs, x_diffs) - headings[:, np.newaxis])
        #print(f"Angles: {angles}")

        return distances, angles

    def compute_distances_and_angles_conspecifics(self, agents):
        """
        Computes and returns the distances and its x and y elements for all pairs of agents

        """
        # Build meshgrid 
        pos_xs = agents[:, 0]
        pos_ys = agents[:, 1]
        xx1, xx2 = np.meshgrid(pos_xs, pos_xs)
        yy1, yy2 = np.meshgrid(pos_ys, pos_ys)

        return self.compute_distances_and_angles(headings=agents[:,2], xx1=xx1, xx2=xx2, yy1=yy1, yy2=yy2)
    
    def compute_nearest_points_to_landmarks(self, agents):
        positions = np.column_stack((agents[:,0], agents[:,1]))
        nearest_points = []
        for landmark in self.landmarks:
            points_landmark = []
            for position in positions:
                point = shpops.nearest_points(Point(position), landmark.get_geometry())[1] # first point is the nearest point in the position, the second on the landmark
                points_landmark.append([point.x, point.y])
            nearest_points.append(points_landmark)
        return nearest_points
    
    def compute_distances_and_angles_landmarks(self, agents):
        """
        Computes and returns the distances and its x and y elements for all pairs of agents

        """
        nearest_points = np.array(self.compute_nearest_points_to_landmarks(agents=agents))
        # Build meshgrid 
        pos_xs = agents[:, 0]
        pos_ys = agents[:, 1]

        # Calculate distances
        x_diffs = nearest_points[:,:,0] - pos_xs[np.newaxis, :]
        x_diffs = x_diffs.T
        y_diffs = nearest_points[:,:,1] - pos_ys[np.newaxis, :]
        y_diffs = y_diffs.T
        distances = np.sqrt(np.multiply(x_diffs, x_diffs) + np.multiply(y_diffs, y_diffs))  
        distances[distances > self.bird_type.sensing_range] = np.inf
        distances[distances == 0.0] = np.inf
        #print(f"Dists: {distances}")
    
        headings = agents[:,2]
        angles = self.wrap_to_pi(np.arctan2(y_diffs, x_diffs) - headings[:, np.newaxis])
        #print(f"Angles: {angles}")

        return distances, angles
    
    def compute_conspecific_match_factors(self, distances):
        repulsion_zone = distances < self.bird_type.preferred_distance_left_right[0]
        attraction_zone = distances > self.bird_type.preferred_distance_left_right[1]
        match_factors = np.zeros((self.num_agents, self.num_agents))
        if self.use_distant_dependent_zone_factors:
            rep_factors = -(1/distances) # stronger repulsion the closer the other is
            att_factors = 1-(1/distances) # stronger attration the farther away the other is
        else:
            rep_factors = np.full(self.num_agents, -1)
            att_factors = np.ones(self.num_agents)

        match_factors = np.where(repulsion_zone, rep_factors, match_factors)
        match_factors = np.where(attraction_zone, att_factors, match_factors)
        return match_factors
    
    def compute_landmark_match_factors(self, distances):
        repulsion_zone = distances < self.bird_type.preferred_distance_left_right[0]
        match_factors = np.zeros((self.num_agents, len(self.landmarks)))
        if self.use_distant_dependent_zone_factors:
            rep_factors = -(1/distances) # stronger repulsion the closer the other is
        else:
            rep_factors = np.full(self.num_agents, -1)
        match_factors = np.where(repulsion_zone, rep_factors, match_factors)
        return match_factors
    
    def compute_side_factors(self, angles, shape):
        lefts = angles < 0
        rights = angles > 0
        side_factors = np.zeros(shape=shape)
        side_factors = np.where(lefts, -1, side_factors)
        side_factors = np.where(rights, 1, side_factors)   
        return side_factors
    
    def compute_vision_strengths(self, head_orientations, distances, angles, shape, bird_type=None):
        if bird_type == None:
            bird_type = self.bird_type
        dist_absmax = bird_type.sensing_range
        vision_strengths_overall = []
        for focus_area in self.bird_type.focus_areas:
            dist_min = focus_area.comfortable_distance[0]
            dist_max = focus_area.comfortable_distance[1]

            angles_2_pi = self.wrap_to_2_pi(angles)

            f_angle = self.wrap_to_2_pi(focus_area.azimuth_angle_position_horizontal + head_orientations)
            focus_angle = np.reshape(np.concatenate([f_angle for i in range(shape[1])]), shape)
            angle_diffs_focus = angles_2_pi - focus_angle

            perception_strengths = 1 - (np.absolute(angle_diffs_focus)/focus_area.angle_field_horizontal)
            
            vision_strengths = np.zeros(shape)
            # if the agent is within the cone of the field of vision, it is perceived
            vision_strengths = np.where(np.absolute(angle_diffs_focus) <= focus_area.angle_field_horizontal, perception_strengths, vision_strengths)
            
            # if an agent is outside of the comfortable viewing distance, it is perceived but with a very low percentage
            vision_strengths = np.where(np.absolute(distances) < dist_min, DIST_MOD * vision_strengths, vision_strengths)
            vision_strengths = np.where(((np.absolute(distances) > dist_max)&(np.absolute(distances) <=dist_absmax)), DIST_MOD * vision_strengths, vision_strengths)
        
            vision_strengths_overall.append(vision_strengths)
        vision_strengths_overall = np.array(vision_strengths_overall)
        vision_strengths_overall = np.sum(vision_strengths_overall.T, axis=2).T
        vision_strengths_overall = normal.normalise(vision_strengths_overall)
        return vision_strengths_overall

    def compute_delta_orientations_conspecifics(self, agents):
        distances, angles = self.compute_distances_and_angles_conspecifics(agents)
        match_factors = self.compute_conspecific_match_factors(distances=distances)
        side_factors = self.compute_side_factors(angles, shape=(self.num_agents, self.num_agents))
        vision_strengths = self.compute_vision_strengths(head_orientations=agents[:,4], distances=distances, angles=angles, shape=(self.num_agents, self.num_agents))
        return np.sum(match_factors * side_factors * vision_strengths, axis=1), distances, angles, vision_strengths
    
    def compute_delta_orientations_landmarks(self, agents):
        distances, angles = self.compute_distances_and_angles_landmarks(agents=agents)
        match_factors = self.compute_landmark_match_factors(distances=distances)
        side_factors = self.compute_side_factors(angles, shape=(self.num_agents, len(self.landmarks)))
        vision_strengths = self.compute_vision_strengths(head_orientations=agents[:,4], distances=distances, angles=angles, shape=(self.num_agents, len(self.landmarks)))
        return np.sum(match_factors * side_factors * vision_strengths, axis=1)
    
    def generate_noise(self):
        return np.random.normal(scale=self.noise_amplitude, size=self.num_agents)
    
    def compute_delta_orientations(self, agents):
        delta_orientations_conspecifics, distances, angles, vision_strengths = self.compute_delta_orientations_conspecifics(agents=agents)
        if len(self.landmarks) > 0:
            delta_orientations_landmarks = self.compute_delta_orientations_landmarks(agents=agents)
        else:
            delta_orientations_landmarks = 0
        #print(delta_orientations_landmarks)
        delta_orientations = self.social_weight * delta_orientations_conspecifics + self.environment_weight * delta_orientations_landmarks
        delta_orientations = np.where((delta_orientations > self.bird_type.max_turn_angle), self.bird_type.max_turn_angle, delta_orientations)
        delta_orientations = np.where((delta_orientations < -self.bird_type.max_turn_angle), -self.bird_type.max_turn_angle, delta_orientations)
        return delta_orientations, distances, angles, vision_strengths

    def compute_new_orientations(self, agents):
        delta_orientations, distances, angles, vision_strengths = self.compute_delta_orientations(agents=agents)
        # add noise
        delta_orientations = delta_orientations + self.generate_noise()

        new_orientations = self.wrap_to_pi(agents[:,2] + delta_orientations)
        if self.model:
            new_head_orientations = self.move_heads(agents=agents, distances=distances, angles=angles, perception_strengths_conspecifics=vision_strengths)
        else:
            new_head_orientations = agents[:,4]
        return new_orientations, new_head_orientations

    def compute_new_positions(self, agents):
        positions = np.column_stack((agents[:,0], agents[:,1]))
        orientations = self.compute_u_v_coordinates_for_angles(angles=agents[:,2])
        positions += self.dt*(orientations.T * agents[:,3]).T
        return positions[:,0], positions[:,1]
    
    
    def has_reached_the_target(self, agents):
        distances, angles = self.compute_distances_and_angles_to_target(agents=agents)
        return distances < self.target_radius

    def compute_target_reached(self, agents):
        distances, _ = self.compute_distances_and_angles_to_target(agents=agents)
        angles = np.arctan2(self.target_position[1]-agents[:,1], self.target_position[0]-agents[:,0])
        has_reached_target = distances < self.target_radius
        orientations = np.where(has_reached_target, angles, agents[:,2])
        speeds = np.where(has_reached_target, 0, agents[:,3])
        return orientations, speeds


    def has_reached_the_target(self, agents):
        distances, angles = self.compute_distances_and_angles_to_target(agents=agents)
        return distances < self.target_radius

    def compute_target_reached(self, agents):
        distances, _ = self.compute_distances_and_angles_to_target(agents=agents)
        angles = np.arctan2(self.target_position[1]-agents[:,1], self.target_position[0]-agents[:,0])
        has_reached_target = distances < self.target_radius
        orientations = np.where(has_reached_target, angles, agents[:,2])
        speeds = np.where(has_reached_target, 0, agents[:,3])
        return orientations, speeds

    def run(self, tmax):

        agent_history = []
        agents = self.initialize()
        self.dt = 1

        for t in range(tmax):
            self.current_step = t

            self.agents = agents

            agents[:,0], agents[:,1] = self.compute_new_positions(agents=agents)
            self.agents = agents
            agents[:,2], agents[:,4] = self.compute_new_orientations(agents=agents)

            if not (self.current_step % self.graph_freq) and self.visualize and self.current_step > 0:
                self.graph_agents(agents=agents)

            agent_history.append(agents)
            
        plt.close()
        return np.array(agent_history)

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