import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shapely.ops as shpops
from shapely import Point

import general.angle_conversion as ac
import vision.perception_strength as pstrength
import loggers.logger_agents as logger
import evaluators.metrics_functions as mf

from simulator.orientation_perception_free_zone_model import OrientationPerceptionFreeZoneModelSimulator
from simulator.enum_neighbour_selection import NeighbourSelectionMechanism
from simulator.enum_switchtype import SwitchType

"""
Implementation of the orientation-perception-free zone model with landmarks.
"""

REPULSION_FACTOR = 50

class OrientationPerceptionFreeZoneModelNeighbourSelectionSimulator(OrientationPerceptionFreeZoneModelSimulator):

    def __init__(self, num_agents, animal_type, domain_size, start_position, switch_type, switch_options, threshold,
                 num_previous_steps=100, stress_delta=0.05, num_ideal_neighbours=9, landmarks=[], noise_amplitude=0, 
                 social_weight=1, environment_weight=1, limit_turns=True, use_distant_dependent_zone_factors=True, 
                 single_speed=True, neighbour_selection=None, k=None, visualize=True, visualize_vision_fields=0, 
                 follow=False, graph_freq=5, save_path_agents=None, save_path_centroid=None, iter=0):
        super().__init__(num_agents, animal_type, domain_size, start_position, landmarks, noise_amplitude, 
                         social_weight, environment_weight, limit_turns, use_distant_dependent_zone_factors, 
                         single_speed, neighbour_selection, k, visualize, visualize_vision_fields, follow, 
                         graph_freq, save_path_agents, save_path_centroid, iter)
        self.switch_type = switch_type
        self.switch_options = switch_options
        self.disorder_value = switch_options[0]
        self.order_value = switch_options[1]
        self.disorder_placeholder = 0
        self.order_placeholder = 1
        self.threshold = threshold
        self.num_previous_steps = num_previous_steps
        self.stress_delta = stress_delta
        self.num_ideal_neighbours = num_ideal_neighbours

    def init_agents(self):
        base = super().init_agents()
        pos_xs = base[:,0]
        pos_ys = base[:,1]
        pos_hs = base[:,2]
        if self.single_speed:
            speeds = np.full(self.num_agents, self.animal_type.speeds[1])
        else:
            speeds = np.random.uniform(self.animal_type.speeds[0], self.animal_type.speeds[2], self.num_agents)

        if self.switch_type == SwitchType.NEIGHBOUR_SELECTION_MECHANISM:
            if self.neighbour_selection == self.order_value:
                switchVals = np.full(self.num_agents, self.order_placeholder)
            else:
                switchVals = np.full(self.num_agents, self.disorder_placeholder)
        else:
            if self.k == self.order_value:
                switchVals = np.full(self.num_agents, self.order_placeholder)
            else:
                switchVals = np.full(self.num_agents, self.disorder_placeholder)

        stress_levels = np.zeros(self.num_agents)

        self.colours = np.random.uniform(0, 1, (self.visualize_vision_fields, 3))

        return np.column_stack([pos_xs, pos_ys, pos_hs, speeds, switchVals, stress_levels])
    
    def get_selected(self, neighbour_selection, k, neighbours, distances):
        match neighbour_selection:
            case NeighbourSelectionMechanism.NEAREST:
                dists = np.where(neighbours, distances, np.inf)
                return np.argmin(dists, axis=1)[:k]
            case NeighbourSelectionMechanism.FARTHEST:
                dists = np.where(neighbours, distances, 0)
                return np.argmax(dists, axis=1)[:k]

    def get_neighbours(self, vision_strengths, distances, decisions):
        neighbours = vision_strengths > 0
        if self.neighbour_selection:
            if self.switch_type != None:
                if self.switch_type == SwitchType.NEIGHBOUR_SELECTION_MECHANISM:
                    vals_order = self.get_selected(neighbour_selection=self.order_value, k=self.k, neighbours=neighbours, distances=distances)
                    vals_disorder = self.get_selected(neighbour_selection=self.disorder_value, k=self.k, neighbours=neighbours, distances=distances)
                else:
                    vals_order = self.get_selected(neighbour_selection=self.neighbour_selection, k=self.order_value, neighbours=neighbours, distances=distances)
                    vals_disorder = self.get_selected(neighbour_selection=self.neighbour_selection, k=self.disorder_value, neighbours=neighbours, distances=distances)

            selected = np.where(decisions == self.order_placeholder, vals_order, vals_disorder)

            new_neighbours = np.full((self.num_agents, self.num_agents), False)
            new_neighbours[selected] = True
            return new_neighbours
        return neighbours    
    
    def compute_stress_levels(self, agents, neighbours):
        num_neighbours = np.count_nonzero(neighbours, axis=1)
        stress_levels = agents[:,5]
        stress_levels = np.where(num_neighbours > self.num_ideal_neighbours, stress_levels - self.stress_delta, stress_levels)
        stress_levels = np.where(num_neighbours < self.num_ideal_neighbours, stress_levels + self.stress_delta, stress_levels)
        return stress_levels
    
    def get_decisions(self, agents, neighbours, stress_levels):
        local_orders = mf.compute_local_orders(agents=agents, neighbours=neighbours)
        self.local_order_history.append(local_orders)
        eval_vals = local_orders + stress_levels
        min_t = max(0, self.current_step-self.num_previous_steps)
        if self.current_step - min_t > 0:
            average_prev_local_orders = np.average(self.local_order_history[min_t:self.current_step], axis=0)
            switch_difference_threshold_lower = self.threshold
            switch_difference_threshold_upper = 1-self.threshold

            old_with_new_order_values = np.where(((eval_vals >= switch_difference_threshold_upper) & (average_prev_local_orders <= switch_difference_threshold_upper)), np.full(len(agents), self.order_placeholder), agents[:,4])
            updated_switch_values = np.where(((eval_vals <= switch_difference_threshold_lower) & (average_prev_local_orders >= switch_difference_threshold_lower)), np.full(len(agents), self.disorder_placeholder), old_with_new_order_values)
            neighbour_counts = np.count_nonzero(neighbours, axis=1)
            updated_switch_values = np.where((neighbour_counts <= 1), agents[:,4], updated_switch_values)
        else:
            updated_switch_values = agents[:,4]
        return updated_switch_values

    def compute_delta_orientations_conspecifics(self, agents):
        """
        Computes the orientation difference that is caused by the conspecifics.
        """
        distances, angles = self.compute_distances_and_angles_conspecifics(agents)
        match_factors = self.compute_conspecific_match_factors(distances=distances)
        side_factors = self.compute_side_factors(angles, shape=(len(agents), len(agents)))
        vision_strengths = self.compute_vision_strengths(distances=distances, angles=angles, shape=(len(agents), len(agents)))
        neighbours = vision_strengths > 0
        stress_levels = self.compute_stress_levels(agents, neighbours)
        decisions = self.get_decisions(agents, neighbours, stress_levels)
        neighbours = self.get_neighbours(vision_strengths, distances, decisions)
        return np.sum(match_factors * side_factors * vision_strengths * neighbours, axis=1), distances, angles, vision_strengths, decisions, stress_levels
    
    def compute_delta_orientations(self, agents):
        """
        Computes the orientation difference for all agents.
        """
        delta_orientations_conspecifics, distances, angles, vision_strengths, decisions, stress_levels = self.compute_delta_orientations_conspecifics(agents=agents)
        if len(self.landmarks) > 0:
            delta_orientations_landmarks = self.compute_delta_orientations_landmarks(agents=agents)
        else:
            delta_orientations_landmarks = 0
        #print(delta_orientations_landmarks)
        delta_orientations = self.social_weight * delta_orientations_conspecifics + self.environment_weight * delta_orientations_landmarks
        #delta_orientations = np.where((delta_orientations > self.animal_type.max_turn_angle), self.animal_type.max_turn_angle, delta_orientations)
        #delta_orientations = np.where((delta_orientations < -self.animal_type.max_turn_angle), -self.animal_type.max_turn_angle, delta_orientations)
        return delta_orientations, distances, angles, vision_strengths, decisions, stress_levels

    def compute_new_orientations(self, agents):
        """
        Computes the new orientations for all agents.
        """
        delta_orientations, distances, angles, vision_strengths, decisions, stress_levels = self.compute_delta_orientations(agents=agents)
        # add noise
        delta_orientations = delta_orientations + self.generate_noise()

        new_orientations = ac.wrap_to_pi(agents[:,2] + delta_orientations)
        return new_orientations, decisions, stress_levels

    def run(self, tmax, dt=1):
        """
        Runs the simulation for tmax timesteps
        """
        self.agent_history = []
        self.local_order_history = []
        agents = self.initialize()
        self.dt = dt

        tmax = int(tmax/dt)

        for t in range(tmax):
            self.current_step = t

            agents[:,0], agents[:,1] = self.compute_new_positions(agents=agents)      
            agents[:,2], agents[:,4], agents[:,5] = self.compute_new_orientations(agents=agents)
            
            self.curr_agents = agents

            self.states.append(self.curr_agents.copy())
            centroid_x, centroid_y = np.mean(self.curr_agents[:, 0]), np.mean(self.curr_agents[:, 1])
            self.centroid_trajectory.append((centroid_x, centroid_y))

            if not (self.current_step % self.graph_freq) and self.visualize and self.current_step > 0:
                self.graph_agents(agents=agents)

            self.agent_history.append(agents)

            if self.save_path_agents or self.save_path_centroid:
                self.save(t=t, agents=agents)
            
        plt.close()
        return np.array(self.agent_history)

