import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shapely.ops as shpops
from shapely import Point

import general.angle_conversion as ac
import general.occlusion as occ
import vision.perception_strength as pstrength
import loggers.logger_agents as logger

from simulator.base_simulator import BaseSimulator
from simulator.enum_neighbour_selection import NeighbourSelectionMechanism

"""
Implementation of the orientation-perception-free zone model with landmarks.
"""

REPULSION_FACTOR = 50

class OrientationPerceptionFreeZoneModelSimulator(BaseSimulator):
    def __init__(self, num_agents, animal_type, domain_size, start_position, landmarks=[],
                 noise_amplitude=0, social_weight=1, environment_weight=1, limit_turns=True, 
                 use_distant_dependent_zone_factors=True, single_speed=True, neighbour_selection=None, 
                 k=None, occlusion_active=False, visualize=True, visualize_vision_fields=0, follow=False, graph_freq=5, 
                 save_path_agents=None, save_path_centroid=None, iter=0):
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
            - single_speed (boolean) [optional, default=True]: whether the agents should have the same or slightly different speeds
            - visualize (boolean) [optional, default=True]: whether the simulation should be visualized immediately
            - visualize_vision_fields (int) [optional, default=0]: the field of vision of how many agents should be visualised. These will be superimposed if necessary
            - follow (boolean) [optional, default=True]: whether the visualization should follow the centroid of the swarm or whether it should show the whole domain
            - graph_freq (int) [optional, default=5]: how often the visualization should be updated
        """
        super().__init__(animal_type=animal_type,
                    num_agents=num_agents,
                    domain_size=domain_size,
                    start_position=start_position,
                    noise_amplitude=noise_amplitude,
                    visualize=visualize,
                    follow=follow,
                    graph_freq=graph_freq)
        self.landmarks = landmarks
        self.social_weight = social_weight,
        self.environment_weight = environment_weight
        self.limit_turns = limit_turns
        self.use_distant_dependent_zone_factors = use_distant_dependent_zone_factors
        self.single_speed = single_speed
        self.neighbour_selection = neighbour_selection
        self.k = k
        self.occlusion_active = occlusion_active
        self.visualize_vision_fields = visualize_vision_fields
        self.save_path_agents = save_path_agents
        self.save_path_centroid = save_path_centroid
        self.iter = iter

    def init_agents(self):
        base = super().init_agents()
        pos_xs = base[:,0]
        pos_ys = base[:,1]
        pos_hs = base[:,2]
        if self.single_speed:
            speeds = np.full(self.num_agents, self.animal_type.speeds[1])
        else:
            speeds = np.random.uniform(self.animal_type.speeds[0], self.animal_type.speeds[2], self.num_agents)

        self.colours = np.random.uniform(0, 1, (self.visualize_vision_fields, 3))

        return np.column_stack([pos_xs, pos_ys, pos_hs, speeds])

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

        self.ax.scatter(agents[:, 0], agents[:, 1], color="white", s=15)

        self.ax.quiver(agents[:, 0], agents[:, 1],
                    uv_coords[:, 0], uv_coords[:, 1],
                    color="white", width=0.005, scale=40)

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
        #plt.pause(0.5)

    def save(self, t, agents):
        if self.save_path_agents:
            dict_list = logger.create_dicts(iter=self.iter, t=t, agents=agents)
            logger.log_results_to_csv(dict_list=dict_list, save_path=self.save_path_agents)
        if self.save_path_centroid:
            dict = logger.create_centroid_dict(iter=self.iter, t=t, centroid=self.centroid_trajectory[-1])
            logger.log_results_to_csv([dict], save_path=self.save_path_centroid)

    def compute_distances_and_angles(self, headings, xx1, xx2, yy1, yy2, transpose_for_angles=False):
        """
        Computes the distances and bearings between the agents.
        """
        x_diffs = xx1 - xx2
        x_diffs = x_diffs.astype(float)
        y_diffs = yy1 - yy2
        y_diffs = y_diffs.astype(float)
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
        nearest_points = np.array(self.compute_nearest_points_to_landmarks(agents=agents), dtype=float)
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
        match_factors = np.zeros((len(distances), len(distances)))
        if self.use_distant_dependent_zone_factors:
            rep_factors = -(1/distances) # stronger repulsion the closer the other is
            att_factors = 1-(1/distances) # stronger attration the farther away the other is
        else:
            rep_factors = np.full(self.num_agents, -REPULSION_FACTOR)
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
            rep_factors = np.full(self.num_agents, -REPULSION_FACTOR)
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
    
    def compute_vision_strengths(self, distances, angles, shape, animal_type=None):
        """
        Computes the vision strengths for every other agent or landmark. Every focus area of the animal_type
        is considered and the distance to the foveal projection is used to determine how well/strongly the
        entity is perceived. If an entity is perceived by multiple focus areas, their average strength is used.
        """
        if animal_type == None:
            animal_type = self.animal_type
        return pstrength.compute_perception_strengths(distances=distances, angles=angles, shape=shape, animal_type=animal_type)

    def get_neighbours(self, vision_strengths, distances):
        neighbours = vision_strengths > 0
        if self.neighbour_selection:
            if self.neighbour_selection == NeighbourSelectionMechanism.NEAREST:
                dists = np.where(neighbours, distances, np.inf)
                selected = np.argmin(dists, axis=1)[:self.k]
            else:
                dists = np.where(neighbours, distances, 0)
                selected = np.argmax(dists, axis=1)[:self.k]
            new_neighbours = np.full((self.num_agents, self.num_agents), False)
            new_neighbours[selected] = True
            neighbours = new_neighbours
        if self.occlusion_active:
            visibles = occ.compute_not_occluded_mask(agents=self.curr_agents, animal_type=self.animal_type)
            if len(self.landmarks) > 0:
                visibles_landmarks = occ.compute_not_occluded_mask_landmarks(agents=self.curr_agents, animal_type=self.animal_type, landmarks=self.landmarks)
                visibles = visibles & visibles_landmarks
            return neighbours & visibles
        return neighbours    
            

    def compute_delta_orientations_conspecifics(self, agents):
        """
        Computes the orientation difference that is caused by the conspecifics.
        """
        distances, angles = self.compute_distances_and_angles_conspecifics(agents)
        match_factors = self.compute_conspecific_match_factors(distances=distances)
        side_factors = self.compute_side_factors(angles, shape=(len(agents), len(agents)))
        vision_strengths = self.compute_vision_strengths(distances=distances, angles=angles, shape=(len(agents), len(agents)))
        neighbours = self.get_neighbours(vision_strengths, distances)
        return np.sum(match_factors * side_factors * vision_strengths * neighbours, axis=1), distances, angles, vision_strengths
    
    def compute_delta_orientations_landmarks(self, agents):
        """
        Computes the orientation difference that is caused by the landmarks.
        """
        distances, angles = self.compute_distances_and_angles_landmarks(agents=agents)
        match_factors = self.compute_landmark_match_factors(distances=distances)
        side_factors = self.compute_side_factors(angles, shape=(len(agents), len(self.landmarks)))
        vision_strengths = self.compute_vision_strengths(distances=distances, angles=angles, shape=(len(agents), len(self.landmarks)))
        return np.sum(match_factors * side_factors * vision_strengths, axis=1)
    
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
        #delta_orientations = np.where((delta_orientations > self.animal_type.max_turn_angle), self.animal_type.max_turn_angle, delta_orientations)
        #delta_orientations = np.where((delta_orientations < -self.animal_type.max_turn_angle), -self.animal_type.max_turn_angle, delta_orientations)
        return delta_orientations, distances, angles, vision_strengths

    def compute_new_orientations(self, agents):
        """
        Computes the new orientations for all agents.
        """
        delta_orientations, distances, angles, vision_strengths = self.compute_delta_orientations(agents=agents)
        # add noise
        delta_orientations = delta_orientations + self.generate_noise()

        new_orientations = ac.wrap_to_pi(agents[:,2] + delta_orientations)
        return new_orientations

    def compute_new_positions(self, agents):
        """
        Update the new position based on the current position and orientation
        """
        positions = np.column_stack((agents[:,0], agents[:,1]))
        orientations = ac.compute_u_v_coordinates_for_angles(angles=agents[:,2])
        positions += self.dt*(orientations.T * agents[:,3]).T
        return positions[:,0], positions[:,1]

    def run(self, tmax, dt=1):
        """
        Runs the simulation for tmax timesteps
        """
        agent_history = []
        agents = self.initialize()
        self.dt = dt

        tmax = int(tmax/dt)

        for t in range(tmax):
            self.current_step = t

            agents[:,0], agents[:,1] = self.compute_new_positions(agents=agents)
            agents[:,2] = self.compute_new_orientations(agents=agents)
            self.curr_agents = agents

            self.states.append(self.curr_agents.copy())
            centroid_x, centroid_y = np.mean(self.curr_agents[:, 0]), np.mean(self.curr_agents[:, 1])
            self.centroid_trajectory.append((centroid_x, centroid_y))

            if not (self.current_step % self.graph_freq) and self.visualize and self.current_step > 0:
                self.graph_agents(agents=agents)

            agent_history.append(agents)

            if self.save_path_agents or self.save_path_centroid:
                self.save(t=t, agents=agents)
            
        plt.close()
        return np.array(agent_history)

