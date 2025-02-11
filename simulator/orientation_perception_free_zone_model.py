import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shapely.ops as shpops
from shapely import Point

from animal_models.pigeon import Pigeon
from animal_models.focus_area import FocusArea
import general.normalisation as normal
import general.angle_conversion as ac
import simulator.weight_options as wo

"""
Implementation of the orientation-perception-free zone model with landmarks.
"""

DIST_MOD = 0.001

class OrientationPerceptionFreeZoneModelSimulator:
    def __init__(self, num_agents, animal_type, domain_size, start_position, landmarks=[],
                 noise_amplitude=0, social_weight=1, environment_weight=1, limit_turns=True, 
                 use_distant_dependent_zone_factors=True, weight_options=[], model=None, 
                 single_speed=True, visualize=True, visualize_vision_fields=0, 
                 visualize_head_direction=True, follow=False, graph_freq=5):
        """
        Params:
            - num_agents (int): the number of animals within the domain
            - animal_type (Animal): the type of animal 
            - domain_size (tuple of ints): the size of the domain, though it is not strictly bounded and used for display only
            - start_position (tuple of 2 ints): the position around which the agents are initially distributed
            - landmarks (list of Landmark) [optional, default=[]]: the landmarks within the domain
            - noise_amplitude (float) [optional, default=0]: the amount of noise that is added to the orientation updates
            - social_weight (float) [optional, default=1]: how much the agents are influenced by the social information
            - environment_weight (float) [optional, default=1]: how much the agents are influenced by the landmark information
            - limit_turns (boolean) [optional, default=True]: whether the turns can be greater than the max turn angle defined for the animal type
            - use_distant_dependent_zone_factors (boolean) [optional, default=True]: whether the influence of neighbours should be dependent on their exact distance or only on the zone they're in
            - weight_options (list of WeightOption) [optional, default=[]]: which information is fed as input to the model
            - model (NeuralNetwork) [optional, default=None]: the neural network model that determines updates to the head orientations
            - single_speed (boolean) [optional, default=True]: whether the agents should have the same or slightly different speeds
            - visualize (boolean) [optional, default=True]: whether the simulation should be visualized immediately
            - visualize_vision_fields (int) [optional, default=0]: the field of vision of how many agents should be visualised. These will be superimposed if necessary
            - visualize_head_direction (boolean) [optional, default=True]: whether an additional arrow should point in the direction in whichere the head is pointing
            - follow (boolean) [optional, default=True]: whether the visualization should follow the centroid of the swarm or whether it should show the whole domain
            - graph_freq (int) [optional, default=5]: how often the visualization should be updated
        """
        self.num_agents = num_agents
        self.animal_type = animal_type
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
        """
        Initialises the agents, domain and field of vision.
        """
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
        """
        Initialises the agents (positions and orientations).
        """
        rng = np.random
        n_points_x = int(np.ceil(np.sqrt(self.num_agents)))
        n_points_y = int(np.ceil(np.sqrt(self.num_agents)))
        spacing = np.average(self.animal_type.preferred_distance_left_right)
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
            speeds = np.full(self.num_agents, self.animal_type.speeds[1])
        else:
            speeds = np.random.uniform(self.animal_type.speeds[0], self.animal_type.speeds[2], self.num_agents)

        head_angles = np.zeros(self.num_agents)

        self.colours = np.random.uniform(0, 1, (self.visualize_vision_fields, 3))

        return np.column_stack([pos_xs, pos_ys, pos_hs, speeds, head_angles])

    def graph_agents(self, agents):
        """
        Redraws the visualization for the current positions and orientations of the agents.
        """  
        self.ax.clear()

        for i in range(self.visualize_vision_fields):
            for focus_area in self.animal_type.focus_areas:
                focus_angle = agents[i,2] + agents[i,4] + focus_area.azimuth_angle_position_horizontal + 2 * np.pi
                start_angle = np.rad2deg(focus_angle - focus_area.angle_field_horizontal) 
                end_angle = np.rad2deg(focus_angle + focus_area.angle_field_horizontal) 
                if focus_area.comfortable_distance[1] == np.inf:
                    distance = 1000
                else:
                    distance = focus_area.comfortable_distance[1]
                #print(f"az={focus_area.azimuth_angle_position_horizontal}, h={focus_area.angle_field_horizontal}, o={ac.wrap_to_2_pi(agents[0,2])}, st={start_angle}, e={end_angle}")
                wedge = mpatches.Wedge((agents[i,0], agents[i,1]), distance, start_angle, end_angle, ec="none", color=self.colours[i])
                self.ax.add_patch(wedge)

        if self.environment_weight > 0:
            # Draw landmarks
            for landmark in self.landmarks:
                self.ax.add_patch(landmark.get_patch_for_display())
                self.ax.annotate(landmark.id, landmark.get_annotation_point(), color="white")

        # Draw agents
        uv_coords = ac.compute_u_v_coordinates_for_angles(agents[:,2])
        uv_coords_head = ac.compute_u_v_coordinates_for_angles(agents[:,2] + agents[:,4])

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
        """
        Moves the heads of all agents based on the output of the neural network model.
        """
        inputs = np.array([wo.get_input_value_for_weight_option(weight_option=option, bearings=agents[:,4], distances=distances, angles=angles, perception_strengths=perception_strengths_conspecifics) for option in self.weight_options])
        inputs = np.where(inputs == np.inf, wo.MAX_INPUT, inputs)
        new_head_angles = []
        for i in range(self.num_agents):
            new_head_angles.append(self.model.predict([inputs[:,i]])[0][0][0])
        return new_head_angles

    def compute_distances_and_angles(self, headings, xx1, xx2, yy1, yy2, transpose_for_angles=False):
        """
        Computes the distances and bearings between the agents.
        """
        x_diffs = xx1 - xx2
        y_diffs = yy1 - yy2
        distances = np.sqrt(np.multiply(x_diffs, x_diffs) + np.multiply(y_diffs, y_diffs))  
        distances[distances > self.animal_type.sensing_range] = np.inf
        distances[distances == 0.0] = np.inf

        if transpose_for_angles:
            angles = ac.wrap_to_pi(np.arctan2(y_diffs.T, x_diffs.T) - headings[:, np.newaxis])
        else:
            angles = ac.wrap_to_pi(np.arctan2(y_diffs, x_diffs) - headings[:, np.newaxis])

        return distances, angles

    def compute_distances_and_angles_conspecifics(self, agents):
        """
        Computes the distances and bearings between the conspecifics.
        """
        # Build meshgrid 
        pos_xs = agents[:, 0]
        pos_ys = agents[:, 1]
        xx1, xx2 = np.meshgrid(pos_xs, pos_xs)
        yy1, yy2 = np.meshgrid(pos_ys, pos_ys)

        return self.compute_distances_and_angles(headings=agents[:,2], xx1=xx1, xx2=xx2, yy1=yy1, yy2=yy2)
    
    def compute_nearest_points_to_landmarks(self, agents):
        """
        Computes the nearest point on every landmark to every agent.
        """
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
        Computes the distances and bearings between the agents and the landmarks.
        """
        nearest_points = np.array(self.compute_nearest_points_to_landmarks(agents=agents))
        pos_xs = agents[:, 0]
        pos_ys = agents[:, 1]

        x_diffs = nearest_points[:,:,0] - pos_xs[np.newaxis, :]
        x_diffs = x_diffs.T
        y_diffs = nearest_points[:,:,1] - pos_ys[np.newaxis, :]
        y_diffs = y_diffs.T
        distances = np.sqrt(np.multiply(x_diffs, x_diffs) + np.multiply(y_diffs, y_diffs))  
        distances[distances > self.animal_type.sensing_range] = np.inf
        distances[distances == 0.0] = np.inf
    
        headings = agents[:,2]
        angles = ac.wrap_to_pi(np.arctan2(y_diffs, x_diffs) - headings[:, np.newaxis])

        return distances, angles
    
    def compute_conspecific_match_factors(self, distances):
        """
        Computes whether the other agents are too close or too far away
        """
        repulsion_zone = distances < self.animal_type.preferred_distance_left_right[0]
        attraction_zone = distances > self.animal_type.preferred_distance_left_right[1]
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
        """
        Computes whether the landmarks are too close
        """
        repulsion_zone = distances < self.animal_type.preferred_distance_left_right[0]
        match_factors = np.zeros((self.num_agents, len(self.landmarks)))
        if self.use_distant_dependent_zone_factors:
            rep_factors = -(1/distances) # stronger repulsion the closer the other is
        else:
            rep_factors = np.full(self.num_agents, -1)
        match_factors = np.where(repulsion_zone, rep_factors, match_factors)
        return match_factors
    
    def compute_side_factors(self, angles, shape):
        """
        Computes whether the other agents or landmarks are to the left, right or straight ahead.
        """
        lefts = angles < 0
        rights = angles > 0
        side_factors = np.zeros(shape=shape)
        side_factors = np.where(lefts, -1, side_factors)
        side_factors = np.where(rights, 1, side_factors)   
        return side_factors
    
    def compute_vision_strengths(self, head_orientations, distances, angles, shape, animal_type=None):
        """
        Computes the vision strengths for every other agent or landmark. Every focus area of the animal_type
        is considered and the distance to the foveal projection is used to determine how well/strongly the
        entity is perceived. If an entity is perceived by multiple focus areas, their average strength is used.
        """
        if animal_type == None:
            animal_type = self.animal_type
        dist_absmax = animal_type.sensing_range
        vision_strengths_overall = []
        for focus_area in self.animal_type.focus_areas:
            dist_min = focus_area.comfortable_distance[0]
            dist_max = focus_area.comfortable_distance[1]

            angles_2_pi = ac.wrap_to_2_pi(angles)

            f_angle = ac.wrap_to_2_pi(focus_area.azimuth_angle_position_horizontal + head_orientations)
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
        """
        Computes the orientation difference that is caused by the conspecifics.
        """
        distances, angles = self.compute_distances_and_angles_conspecifics(agents)
        match_factors = self.compute_conspecific_match_factors(distances=distances)
        side_factors = self.compute_side_factors(angles, shape=(self.num_agents, self.num_agents))
        vision_strengths = self.compute_vision_strengths(head_orientations=agents[:,4], distances=distances, angles=angles, shape=(self.num_agents, self.num_agents))
        return np.sum(match_factors * side_factors * vision_strengths, axis=1), distances, angles, vision_strengths
    
    def compute_delta_orientations_landmarks(self, agents):
        """
        Computes the orientation difference that is caused by the landmarks.
        """
        distances, angles = self.compute_distances_and_angles_landmarks(agents=agents)
        match_factors = self.compute_landmark_match_factors(distances=distances)
        side_factors = self.compute_side_factors(angles, shape=(self.num_agents, len(self.landmarks)))
        vision_strengths = self.compute_vision_strengths(head_orientations=agents[:,4], distances=distances, angles=angles, shape=(self.num_agents, len(self.landmarks)))
        return np.sum(match_factors * side_factors * vision_strengths, axis=1)
    
    def generate_noise(self):
        """
        Generates noise.
        """
        return np.random.normal(scale=self.noise_amplitude, size=self.num_agents)
    
    def compute_delta_orientations(self, agents):
        """
        Computes the orientation difference for all agents.
        """
        delta_orientations_conspecifics, distances, angles, vision_strengths = self.compute_delta_orientations_conspecifics(agents=agents)
        if len(self.landmarks) > 0:
            delta_orientations_landmarks = self.compute_delta_orientations_landmarks(agents=agents)
        else:
            delta_orientations_landmarks = 0
        #print(delta_orientations_landmarks)
        delta_orientations = self.social_weight * delta_orientations_conspecifics + self.environment_weight * delta_orientations_landmarks
        delta_orientations = np.where((delta_orientations > self.animal_type.max_turn_angle), self.animal_type.max_turn_angle, delta_orientations)
        delta_orientations = np.where((delta_orientations < -self.animal_type.max_turn_angle), -self.animal_type.max_turn_angle, delta_orientations)
        return delta_orientations, distances, angles, vision_strengths

    def compute_new_orientations(self, agents):
        """
        Computes the new orientations and head orientations for all agents.
        """
        delta_orientations, distances, angles, vision_strengths = self.compute_delta_orientations(agents=agents)
        # add noise
        delta_orientations = delta_orientations + self.generate_noise()

        new_orientations = ac.wrap_to_pi(agents[:,2] + delta_orientations)
        if self.model:
            new_head_orientations = self.move_heads(agents=agents, distances=distances, angles=angles, perception_strengths_conspecifics=vision_strengths)
        else:
            new_head_orientations = agents[:,4]
        return new_orientations, new_head_orientations

    def compute_new_positions(self, agents):
        """
        Update the new position based on the current position and orientation
        """
        positions = np.column_stack((agents[:,0], agents[:,1]))
        orientations = ac.compute_u_v_coordinates_for_angles(angles=agents[:,2])
        positions += self.dt*(orientations.T * agents[:,3]).T
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
            agents[:,2], agents[:,4] = self.compute_new_orientations(agents=agents)

            if not (self.current_step % self.graph_freq) and self.visualize and self.current_step > 0:
                self.graph_agents(agents=agents)

            agent_history.append(agents)
            
        plt.close()
        return np.array(agent_history)

