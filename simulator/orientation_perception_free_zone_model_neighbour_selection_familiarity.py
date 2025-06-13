import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shapely.ops as shpops
from shapely import Point

import general.angle_conversion as ac
import general.normalisation as normal
import vision.perception_strength as pstrength
import loggers.logger_agents as logger
import evaluators.metrics_functions as mf

from simulator.orientation_perception_free_zone_model_neighbour_selection import OrientationPerceptionFreeZoneModelNeighbourSelectionSimulator
from simulator.enum_neighbour_selection import NeighbourSelectionMechanism
from simulator.enum_switchtype import SwitchType

"""
Implementation of the orientation-perception-free zone model with landmarks.
"""

REPULSION_FACTOR = 50

class OrientationPerceptionFreeZoneModelNeighbourSelectionFamiliaritySimulator(OrientationPerceptionFreeZoneModelNeighbourSelectionSimulator):

    def __init__(self, num_agents, animal_type, domain_size, start_position, switch_type=None, switch_options=[], threshold=0,
                 num_previous_steps=100, stress_delta=0.05, num_ideal_neighbours=9, landmarks=[], noise_amplitude=0, 
                 social_weight=1, environment_weight=1, limit_turns=True, use_distant_dependent_zone_factors=True, 
                 single_speed=True, speed_delta=0.001, neighbour_selection=None, k=None, visualize=True, visualize_vision_fields=0, 
                 follow=False, graph_freq=5, save_path_agents=None, save_path_centroid=None, iter=0, familiarity_weight=0.1):
        super().__init__(num_agents=num_agents,
                         animal_type=animal_type,
                         domain_size=domain_size,
                         start_position=start_position,
                         switch_type=switch_type,
                         switch_options=switch_options,
                         threshold=threshold,
                         num_previous_steps=num_previous_steps,
                         stress_delta=stress_delta,
                         num_ideal_neighbours=num_ideal_neighbours,
                         landmarks=landmarks,
                         noise_amplitude=noise_amplitude,
                         social_weight=social_weight,
                         environment_weight=environment_weight,
                         limit_turns=limit_turns,
                         use_distant_dependent_zone_factors=use_distant_dependent_zone_factors,
                         single_speed=single_speed,
                         speed_delta=speed_delta,
                         neighbour_selection=neighbour_selection,
                         k=k,
                         visualize=visualize,
                         visualize_vision_fields=visualize_vision_fields,
                         follow=follow,
                         graph_freq=graph_freq,
                         save_path_agents=save_path_agents,
                         save_path_centroid=save_path_centroid,
                         iter=iter)
        self.familiarity_weight = familiarity_weight

    def init_agents(self):
        self.familiarites = np.zeros((self.num_agents, self.num_agents))
        return super().init_agents()
    
    def get_selected(self, neighbour_selection, k, neighbours, distances):
        familiarity_selected = np.flip(np.argsort(self.familiarites * neighbours), axis=1)[:,:k]
        ns_selected = super().get_selected(neighbour_selection=neighbour_selection, k=k, neighbours=neighbours, distances=distances)
        fams = [[self.familiarites[i][j] for j in range(len(familiarity_selected[i]))] for i in range(len(familiarity_selected))]
        for i in range(len(neighbours)):
            fams[i].extend([1 for _ in range(len(ns_selected[i]))])
        probs = normal.normalise(np.array(fams))
        idx = [list(familiarity_selected[i]) for i in range(len(familiarity_selected))]
        for i in range(len(neighbours)):
            idx[i].extend(list(ns_selected[i]))
        selected = []
        for i in range(len(probs)):
            selected.append(np.random.choice(idx[i], size=k, replace=False, p=probs[i]))
        mask = np.full(neighbours.shape, False)
        for i in range(len(neighbours)):
            for j in selected[i]:
                mask[i][j] = True
        self.familiarites[mask] += 1

        return selected


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
            agents[:,2], agents[:, 3], agents[:,4], agents[:,5] = self.compute_new_orientations_and_speeds(agents=agents)
            
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

