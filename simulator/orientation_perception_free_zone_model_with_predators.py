import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shapely.ops as shpops
from shapely import Point

from animal_models.pigeon import Pigeon
from animal_models.focus_area import FocusArea
import general.normalisation as normal
import general.angle_conversion as ac
import simulator.head_movement.weight_options as wo
from simulator.orientation_perception_free_zone_model import OrientationPerceptionFreeZoneModelSimulator

"""
Implementation of the orientation-perception-free zone model with predators and landmarks.
"""

DIST_MOD = 0.001

class OrientationPerceptionFreeZoneModelSimulatorWithPredators(OrientationPerceptionFreeZoneModelSimulator):
    def __init__(self, num_prey, animal_type_prey, num_predators, animal_type_predator, domain_size, start_position_prey, 
                 start_position_predator, pack_hunting=False, landmarks=[], noise_amplitude=0, social_weight=1, environment_weight=1,
                 other_type_weight=1, limit_turns=True, use_distant_dependent_zone_factors=True, single_speed=True, 
                 visualize=True, visualize_vision_fields_prey=0, visualize_vision_fields_predator=0, follow=False, 
                 graph_freq=5):
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
                         social_weight=social_weight,
                         environment_weight=environment_weight,
                         limit_turns=limit_turns,
                         use_distant_dependent_zone_factors=use_distant_dependent_zone_factors,
                         single_speed=single_speed,
                         visualize=visualize,
                         visualize_vision_fields=visualize_vision_fields_prey,
                         follow=follow,
                         graph_freq=graph_freq)
        
        self.num_prey = num_prey
        self.animal_type_prey = animal_type_prey
        self.num_predators = num_predators
        self.animal_type_predator = animal_type_predator
        self.start_position_prey = start_position_prey
        self.start_position_predator = start_position_predator
        self.pack_hunting = pack_hunting
        self.other_type_weight = other_type_weight
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
                                               visualize_vision_fields=self.visualize_vision_fields_prey)
        predators, self.colours_predator = self.init_agents_for_animal_type(num_agents=self.num_predators, 
                                               animal_type=self.animal_type_predator, 
                                               start_position=self.start_position_predator,
                                               visualize_vision_fields=self.visualize_vision_fields_predator)
        return prey, predators

    def init_agents_for_animal_type(self, num_agents, animal_type, start_position, visualize_vision_fields):
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

        return np.column_stack([pos_xs, pos_ys, pos_hs, speeds]), colours
    
    def graph_vision_fields(self, agents, animal_type, visualize_vision_fields, colours):
        """
        Redraws the vision fields for the selected agents.
        """
        for i in range(visualize_vision_fields):
            for focus_area in animal_type.focus_areas:
                focus_angle = agents[i,2] + agents[i,4] + focus_area.azimuth_angle_position_horizontal + 2 * np.pi
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
        uv_coords = ac.compute_u_v_coordinates_for_angles(agents[:,2])

        self.ax.scatter(agents[:, 0], agents[:, 1], color="white", s=15)

        self.ax.quiver(agents[:, 0], agents[:, 1],
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
        Computes the distances and bearings between predator and prey from either point of view.
        """
        focus_group_xs = focus_group[:, 0]
        focus_group_ys = focus_group[:, 1]
        other_group_xs = other_group[:, 0]
        other_group_ys = other_group[:, 1]
        xx1, xx2 = np.meshgrid(focus_group_xs, other_group_xs)
        yy1, yy2 = np.meshgrid(focus_group_ys, other_group_ys)

        return self.compute_distances_and_angles(headings=focus_group[:,2], xx1=xx1, xx2=xx2, yy1=yy1, yy2=yy2, transpose_for_angles=True)
    
    def compute_delta_orientations_away_from_predators(self, prey, predators):
        """
        Computes the orientation difference that is caused by predators on prey (repulsion only).
        """
        distances, angles = self.compute_distances_and_angles_other_type(prey, predators)
        match_factors = np.full((self.num_prey, self.num_predators), 1) # always repulsed
        side_factors = self.compute_side_factors(angles, shape=(self.num_prey, self.num_predators))
        vision_strengths = self.compute_vision_strengths(distances=distances.T, angles=angles, shape=(self.num_prey, self.num_predators))
        return np.sum(match_factors * side_factors * vision_strengths, axis=1)

    def compute_delta_orientations_towards_prey(self, prey, predators):
        """
        Computes the orientation difference that is caused by prey on predators (attraction only).
        """
        distances, angles = self.compute_distances_and_angles_other_type(predators, prey)
        match_factors = np.full((self.num_predators, self.num_prey), -1) # always attracted
        side_factors = self.compute_side_factors(angles, shape=(self.num_predators, self.num_prey))
        vision_strengths = self.compute_vision_strengths(distances=distances.T, angles=angles, shape=(self.num_predators, self.num_prey))
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

        if self.other_type_weight != 0:
            if is_prey:
                delta_orientations_other_type = self.compute_delta_orientations_away_from_predators(prey, predators)
            else:
                delta_orientations_other_type = self.compute_delta_orientations_towards_prey(prey, predators)
        else:
            delta_orientations_other_type = 0

        social_weight = self.social_weight
        if not is_prey and not self.pack_hunting:
            social_weight = 0

        delta_orientations = social_weight * delta_orientations_conspecifics + self.environment_weight * delta_orientations_landmarks + self.other_type_weight * delta_orientations_other_type
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
        delta_orientations = delta_orientations + self.generate_noise()

        new_orientations = ac.wrap_to_pi(agents[:,2] + delta_orientations)
        return new_orientations

    def compute_new_orientations(self, prey, predators):
        """
        Computes the new orientations for all agents (predator and prey).
        """
        self.num_agents = self.num_prey
        prey_orientations = self.compute_new_orientations_for_type(prey, predators, is_prey=True)
        self.num_agents = self.num_predators
        predators_orientations = self.compute_new_orientations_for_type(prey, predators, is_prey=False)
        return prey_orientations, predators_orientations
    
    def run(self, tmax):
        """
        Runs the simulation for tmax timesteps
        """
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
            
            prey_o, predator_o  = self.compute_new_orientations(prey=prey, predators=predators)

            prey[:,2] = prey_o
            predators[:,2] = predator_o

            if not (self.current_step % self.graph_freq) and self.visualize and self.current_step > 0:
                self.graph_agents(prey=prey, predators=predators)

            prey_history.append(prey)
            predator_history.append(predators)
            
        plt.close()
        return np.array(prey_history), np.array(predator_history)
