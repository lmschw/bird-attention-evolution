import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import general.angle_conversion as ac
import simulator.head_movement.weight_options as wo
from simulator.orientation_perception_free_zone_model import OrientationPerceptionFreeZoneModelSimulator
import loggers.logger_agents as logger
import vision.perception_strength as pstrength

"""
Implementation of the orientation-perception-free zone model with predators and landmarks.
"""

DIST_MOD = 0.001

class OrientationPerceptionFreeZoneModelSimulatorWithPredators(OrientationPerceptionFreeZoneModelSimulator):
    def __init__(self, num_prey, animal_type_prey, num_predators, animal_type_predator, domain_size, start_position_prey, 
                 start_position_predator, pack_hunting=False, landmarks=[], noise_amplitude=0, social_weights=[1,1], environment_weights=[1,1],
                 other_type_weights=[1,1], limit_turns=True, use_distant_dependent_zone_factors=True, single_speed=True, 
                 neighbour_selection=None, k=None, occlusion_active=False,
                 kill=True, killing_frenzy=False, visualize=True, visualize_vision_fields_prey=0, visualize_vision_fields_predator=0, follow=False, 
                 graph_freq=5, save_path_agents=None, save_path_centroid=None, iter=0):
        """
        Params:
            - num_prey (int): the number of prey animals within the domain
            - animal_type_prey (Animal): the type of animal of the prey
            - num_predators (int): the number of predators within the domain
            - animal_type_predator (Animal): the type of animal of the predators
            - domain_size (tuple of ints): the size of the domain, though it is not strictly bounded and used for display only
            - start_position_prey (tuple of 2 ints): the position around which the prey are initially distributed
            - start_position_predator (tuple of 2 ints): the position around which the predators are initially distributed
            - pack_hunting (boolean) [optional, default=False]: whether the predators will consider social information from their conspecifics. Always False if there is only a single predator
            - landmarks (list of Landmark) [optional, default=[]]: the landmarks within the domain
            - noise_amplitude (float) [optional, default=0]: the amount of noise that is added to the orientation updates
            - social_weight (float) [optional, default=1]: how much the agents are influenced by the social information
            - environment_weight (float) [optional, default=1]: how much the agents are influenced by the landmark information
            - other_type_weight (float) [optional, default=1]: how much the agents are influenced by prey for predators and predators for prey
            - limit_turns (boolean) [optional, default=True]: whether the turns can be greater than the max turn angle defined for the animal type
            - use_distant_dependent_zone_factors (boolean) [optional, default=True]: whether the influence of neighbours should be dependent on their exact distance or only on the zone they're in
            - single_speed (boolean) [optional, default=True]: whether the agents should have the same or slightly different speeds
            - visualize (boolean) [optional, default=True]: whether the simulation should be visualized immediately
            - visualize_vision_fields_prey (int) [optional, default=0]: the field of vision of how many prey agents should be visualised. These will be superimposed if necessary
            - visualize_vision_fields_predator (int) [optional, default=0]: the field of vision of how many predator agents should be visualised. These will be superimposed if necessary
            - follow (boolean) [optional, default=True]: whether the visualization should follow the centroid of the swarm or whether it should show the whole domain
            - graph_freq (int) [optional, default=5]: how often the visualization should be updated
        """
        super().__init__(num_agents=num_prey,
                         animal_type=animal_type_prey,
                         domain_size=domain_size,
                         start_position=start_position_prey,
                         landmarks=landmarks,
                         noise_amplitude=noise_amplitude,
                         social_weight=social_weights[0],
                         environment_weight=environment_weights[0],
                         limit_turns=limit_turns,
                         use_distant_dependent_zone_factors=use_distant_dependent_zone_factors,
                         single_speed=single_speed,
                         neighbour_selection=neighbour_selection,
                         k=k,
                         occlusion_active=occlusion_active,
                         visualize=visualize,
                         visualize_vision_fields=visualize_vision_fields_prey,
                         follow=follow,
                         graph_freq=graph_freq,
                         save_path_agents=save_path_agents,
                         save_path_centroid=save_path_centroid,
                         iter=iter)
        
        self.num_prey = num_prey
        self.animal_type_prey = animal_type_prey
        self.num_predators = num_predators
        self.animal_type_predator = animal_type_predator
        self.start_position_prey = start_position_prey
        self.start_position_predator = start_position_predator
        self.pack_hunting = pack_hunting
        self.social_weights = social_weights
        self.environment_weights = environment_weights
        self.other_type_weights = other_type_weights
        self.kill = kill
        self.killing_frenzy = killing_frenzy
        self.visualize_vision_fields_prey = visualize_vision_fields_prey
        self.visualize_vision_fields_predator = visualize_vision_fields_predator

        # if there is only a single predator, then no social information is available and hence, 
        # pack hunting is impossible
        if self.num_predators == 1:
            self.pack_hunting = False

    def initialize(self):
        """
        Initialises the agents, domain and field of vision.
        """
        prey, predators = self.init_agents()

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
        """
        Initialises the agents (positions and orientations) both for prey and predators.
        """
        prey, self.colours_prey = self.init_agents_for_animal_type(num_agents=self.num_prey, 
                                               animal_type=self.animal_type_prey, 
                                               start_position=self.start_position_prey,
                                               visualize_vision_fields=self.visualize_vision_fields_prey,
                                               is_prey=True)
        predators, self.colours_predator = self.init_agents_for_animal_type(num_agents=self.num_predators, 
                                               animal_type=self.animal_type_predator, 
                                               start_position=self.start_position_predator,
                                               visualize_vision_fields=self.visualize_vision_fields_predator,
                                               is_prey=False)
        return prey, predators

    def init_agents_for_animal_type(self, num_agents, animal_type, start_position, visualize_vision_fields, is_prey):
        """
        Initialises the agents (positions and orientations) for either prey or predators.
        """
        rng = np.random
        n_points_x = int(np.ceil(np.sqrt(num_agents)))
        n_points_y = int(np.ceil(np.sqrt(num_agents)))
        spacing = np.average(animal_type.preferred_distance_left_right)
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
            speeds = np.full(num_agents, animal_type.speeds[1])
        else:
            speeds = np.random.uniform(animal_type.speeds[0], animal_type.speeds[2], num_agents)

        colours = np.random.uniform(0, 1, (visualize_vision_fields, 3))

        if is_prey:
            weights = np.full((num_agents, 3), [self.social_weights[0], self.environment_weights[0], self.other_type_weights[0]])
        else:    
            weights = np.full((num_agents, 3), [self.social_weights[1], self.environment_weights[1], self.other_type_weights[1]])
            if not self.pack_hunting:
                weights[:,0] = 0
        alive = np.ones(num_agents)

        return np.column_stack([pos_xs, pos_ys, pos_hs, speeds, weights, alive]), colours
    
    def graph_vision_fields(self, agents, animal_type, visualize_vision_fields, colours):
        """
        Redraws the vision fields for the selected agents.
        """
        for i in range(visualize_vision_fields):
            for focus_area in animal_type.focus_areas:
                focus_angle = agents[i,2] + focus_area.azimuth_angle_position_horizontal + 2 * np.pi
                start_angle = np.rad2deg(focus_angle - focus_area.angle_field_horizontal) 
                end_angle = np.rad2deg(focus_angle + focus_area.angle_field_horizontal) 
                if focus_area.comfortable_distance[1] == np.inf:
                    distance = 1000
                else:
                    distance = focus_area.comfortable_distance[1]
                #print(f"az={focus_area.azimuth_angle_position_horizontal}, h={focus_area.angle_field_horizontal}, o={ac.wrap_to_2_pi(agents[0,2])}, st={start_angle}, e={end_angle}")
                wedge = mpatches.Wedge((agents[i,0], agents[i,1]), distance, start_angle, end_angle, ec="none", color=colours[i])
                self.ax.add_patch(wedge)

    def graph_arrows(self, agents, colour):
        """
        Redraws an agent type (prey or predator)
        """

        alive_agents = agents[agents[:,7] == 1]

        uv_coords = ac.compute_u_v_coordinates_for_angles(alive_agents[:,2])

        self.ax.scatter(alive_agents[:, 0], alive_agents[:, 1], color="white", s=15)

        self.ax.quiver(alive_agents[:, 0], alive_agents[:, 1],
                    uv_coords[:, 0], uv_coords[:, 1],
                    color=colour, width=0.005, scale=40)

    def graph_agents(self, prey, predators):
        """
        Redraws the visualization for the current positions and orientations of the agents.
        """
        self.ax.clear()

        self.graph_vision_fields(prey, self.animal_type_prey, self.visualize_vision_fields_prey, self.colours_prey)
        self.graph_vision_fields(predators, self.animal_type_predator, self.visualize_vision_fields_predator, self.colours_predator)

        if self.environment_weight > 0:
            # Draw landmarks
            for landmark in self.landmarks:
                self.ax.add_patch(landmark.get_patch_for_display())
                self.ax.annotate(landmark.id, landmark.get_annotation_point(), color="white")

        # Draw prey and predators
        if len(prey) > 0:
            self.graph_arrows(agents=prey, colour="white")
        if len(predators) > 0:
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

    def save(self, t, prey, predators):
        if self.save_path_agents:
            dict_list = logger.create_dicts(iter=self.iter, t=t, agents=prey, is_prey=True)
            logger.log_results_to_csv(dict_list=dict_list, save_path=self.save_path_agents)
            dict_list = logger.create_dicts(iter=self.iter, t=t, agents=predators, is_prey=False)
            logger.log_results_to_csv(dict_list=dict_list, save_path=self.save_path_agents)
        if self.save_path_centroid:
            dict = logger.create_centroid_dict(iter=self.iter, t=t, centroid=self.centroid_trajectory[-1])
            logger.log_results_to_csv([dict], save_path=self.save_path_centroid)

    def compute_distances_and_angles_other_type(self, focus_group, other_group):
        """
        Computes the distances and bearings between predator and prey from either point of view.
        """
        focus_group_xs = focus_group[:, 0]
        focus_group_ys = focus_group[:, 1]
        other_group_xs = other_group[:, 0]
        other_group_ys = other_group[:, 1]
        xx1, xx2 = np.meshgrid(focus_group_xs, other_group_xs)
        yy1, yy2 = np.meshgrid(focus_group_ys, other_group_ys)

        return self.compute_distances_and_angles(headings=focus_group[:,2], xx1=xx1, xx2=xx2, yy1=yy1, yy2=yy2, transpose_for_angles=True)
    
    def compute_vision_strengths_other_type(self, distances, angles, shape, animal_type=None, is_prey=True):
        if animal_type == None:
            animal_type = self.animal_type
        if self.occlusion_active:
            return pstrength.compute_perception_strengths_with_occlusion_predation(prey=self.prey,
                                                                                   predators=self.predators,
                                                                                      distances=distances,
                                                                                      angles=angles,
                                                                                      shape=shape,
                                                                                      animal_type=animal_type,
                                                                                      is_prey=is_prey)
        else:
            return pstrength.compute_perception_strengths(distances=distances, angles=angles, shape=shape, animal_type=animal_type)


    def compute_delta_orientations_away_from_predators(self, prey, predators):
        """
        Computes the orientation difference that is caused by predators on prey (repulsion only).
        """
        distances, angles = self.compute_distances_and_angles_other_type(prey, predators)
        match_factors = np.full((len(prey), len(predators)), 1) # always repulsed
        side_factors = self.compute_side_factors(angles, shape=(len(prey), len(predators)))
        vision_strengths = self.compute_vision_strengths_other_type(distances=distances.T, angles=angles, shape=(len(prey), len(predators)), is_prey=True)
        return np.sum(match_factors * side_factors * vision_strengths, axis=1)

    def compute_delta_orientations_towards_prey(self, prey, predators):
        """
        Computes the orientation difference that is caused by prey on predators (attraction only).
        """
        distances, angles = self.compute_distances_and_angles_other_type(predators, prey)
        match_factors = np.full((len(predators),len(prey)), -1) # always attracted
        side_factors = self.compute_side_factors(angles, shape=(len(predators), len(prey)))
        vision_strengths = self.compute_vision_strengths_other_type(distances=distances.T, angles=angles, shape=(len(predators), len(prey)), is_prey=False)
        return np.sum(match_factors * side_factors * vision_strengths, axis=1)

    def compute_delta_orientations(self, prey, predators, is_prey):
        """
        Computes the orientation difference for a single agent type (predator or prey).
        """
        if is_prey:
            agents = prey
            animal_type = self.animal_type_prey
        else:
            agents = predators
            animal_type = self.animal_type_predator

        delta_orientations_conspecifics, distances, angles, vision_strengths = self.compute_delta_orientations_conspecifics(agents=agents)
        if len(self.landmarks) > 0 and self.environment_weight != 0:
            delta_orientations_landmarks = self.compute_delta_orientations_landmarks(agents=agents)
        else:
            delta_orientations_landmarks = 0

        if len(prey) > 0 and len(predators) > 0:
            if is_prey:
                delta_orientations_other_type = self.compute_delta_orientations_away_from_predators(prey, predators)
            else:
                delta_orientations_other_type = self.compute_delta_orientations_towards_prey(prey, predators)
        else:
            delta_orientations_other_type = 0

        delta_orientations = agents[:,4] * delta_orientations_conspecifics + agents[:,5] * delta_orientations_landmarks + agents[:,6] * delta_orientations_other_type
        delta_orientations = np.where((delta_orientations > animal_type.max_turn_angle), animal_type.max_turn_angle, delta_orientations)
        delta_orientations = np.where((delta_orientations < -animal_type.max_turn_angle), -animal_type.max_turn_angle, delta_orientations)
        return delta_orientations, distances, angles, vision_strengths
    
    def compute_new_orientations_for_type(self, prey, predators, is_prey):
        """
        Computes the new orientations for a single agent type (predator or prey).
        """
        if is_prey:
            agents = prey
        else:
            agents = predators

        delta_orientations, distances, angles, vision_strengths = self.compute_delta_orientations(prey=prey, predators=predators, is_prey=is_prey)
        # add noise
        delta_orientations = delta_orientations + self.generate_noise(len(agents))

        new_orientations = ac.wrap_to_pi(agents[:,2] + delta_orientations)
        return new_orientations

    def compute_new_orientations_and_speeds(self, prey, predators):
        """
        Computes the new orientations for all agents (predator and prey).
        """
        if len(prey) > 0:
            self.num_agents = self.num_prey
            prey_orientations = self.compute_new_orientations_for_type(prey, predators, is_prey=True)
        else:
            prey_orientations = prey[:,2]
        if len(predators) > 0:
            self.num_agents = self.num_predators
            predators_orientations = self.compute_new_orientations_for_type(prey, predators, is_prey=False)
        else:
            predators_orientations = predators[:,2]
        
        return prey_orientations, predators_orientations
    
    def check_kills(self, prey, predators):
        x_close = np.absolute(prey[:,0]-predators[:,0]) < 5
        y_close = np.absolute(prey[:,1]-predators[:,1]) < 5
        close = x_close & y_close
        prey[:,7] = np.where(close, 0, prey[:,7])
        if not self.killing_frenzy and np.count_nonzero(close):
            prey[:,6] = np.where(np.count_nonzero(close) > 0, 0, prey[:,6]) # if the predator has caught something, it is no longer an immediate danger
            predators[:,6] = np.where(np.count_nonzero(close) > 0, -1, predators[:,6]) # if they have caught something, they no longer hunt and are therefore not attracted to prey anymore
            #predators[:,7] = 0
            print(f"{self.current_step} - KILL!!!!!!!!")
        return prey, predators
    
    def run(self, tmax, dt=1):
        """
        Runs the simulation for tmax timesteps
        """
        prey_history = []
        predator_history = []
        prey, predators = self.initialize()
        self.dt = dt
        tmax = int(tmax/dt)

        print(tmax)

        for t in range(tmax):
            self.current_step = t

            # clean up the prey
            prey = prey[prey[:,7] == 1]
            predators = predators[predators[:,7] == 1]

            self.prey = prey
            self.predators = predators

            prey[:,0], prey[:,1] = self.compute_new_positions(agents=prey)
            predators[:,0], predators[:,1] = self.compute_new_positions(agents=predators)

            self.prey = prey
            self.predators = predators
            
            prey[:,2], predators[:,2]  = self.compute_new_orientations_and_speeds(prey=prey, predators=predators)

            if self.kill and len(predators) > 0:
                prey, predators = self.check_kills(prey=prey, predators=predators)


            centroid_x, centroid_y = np.mean(prey[:, 0]), np.mean(prey[:, 1])
            self.centroid_trajectory.append((centroid_x, centroid_y))

            if not (self.current_step % self.graph_freq) and self.visualize and self.current_step > 0:
                self.graph_agents(prey=prey, predators=predators)

            prey_history.append(prey)
            predator_history.append(predators)
            
            if self.save_path_agents or self.save_path_centroid:
                self.save(t=t, prey=prey, predators=predators)
            
        plt.close()
        return prey_history, predator_history
