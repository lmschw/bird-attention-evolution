import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import general.normalisation as normal
import general.angle_conversion as ac

from simulator.orientation_perception_free_zone_model import OrientationPerceptionFreeZoneModelSimulator
import simulator.head_movement.head_movement as hm

"""
Implementation of the orientation-perception-free zone model with landmarks.
"""

DIST_MOD = 0.001

class OrientationPerceptionFreeZoneModelWithHeadMovementSimulator(OrientationPerceptionFreeZoneModelSimulator):
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
        super().__init__(animal_type=animal_type,
                    num_agents=num_agents,
                    domain_size=domain_size,
                    start_position=start_position,
                    landmarks=landmarks,
                    noise_amplitude=noise_amplitude,
                    social_weight=social_weight,
                    environment_weight=environment_weight,
                    limit_turns=limit_turns,
                    use_distant_dependent_zone_factors=use_distant_dependent_zone_factors,
                    single_speed=single_speed,
                    visualize=visualize,
                    visualize_vision_fields=visualize_vision_fields,
                    follow=follow,
                    graph_freq=graph_freq)
        self.weight_options = weight_options
        self.model = model
        self.visualize_head_direction = visualize_head_direction

    def init_agents(self):
        base = super().init_agents()
        pos_xs = base[:,0]
        pos_ys = base[:,1]
        pos_hs = base[:,2]
        speeds = base[:,3]

        head_angles = np.zeros(self.num_agents)

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

    def compute_new_orientations_and_speeds(self, agents):
        """
        Computes the new orientations and head orientations for all agents.
        """
        delta_orientations, distances, angles, vision_strengths = self.compute_delta_orientations(agents=agents)
        # add noise
        delta_orientations = delta_orientations + self.generate_noise()

        new_orientations = ac.wrap_to_pi(agents[:,2] + delta_orientations)
        if self.model:
            new_head_orientations = hm.move_heads(model=self.model, 
                                                  weight_options=self.weight_options, 
                                                  animal_type=self.animal_type,
                                                  num_agents=self.num_agents, 
                                                  current_head_angles=agents[:,4], 
                                                  distances=distances, 
                                                  angles=angles, 
                                                  perception_strengths_conspecifics=vision_strengths)
        else:
            new_head_orientations = agents[:,4]
        return new_orientations, new_head_orientations

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
            agents[:,2], agents[:,4] = self.compute_new_orientations_and_speeds(agents=agents)

            if not (self.current_step % self.graph_freq) and self.visualize and self.current_step > 0:
                self.graph_agents(agents=agents)

            agent_history.append(agents)
            
        plt.close()
        return np.array(agent_history)

