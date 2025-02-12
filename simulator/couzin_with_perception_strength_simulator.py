import numpy as np

import vision.perception_strength as pstrength
import general.angle_conversion as ac

from simulator.couzin_simulator import CouzinZoneModelSimulator

"""
Implements Couzin et al.'s zone model with added perception strength.
"""

class CouzinZoneModelWithPerceptionStrengthSimulator(CouzinZoneModelSimulator):
    def __init__(self, animal_type, num_agents, domain_size, start_position,
                 noise_amplitude,
                 visualize=True, follow=True, graph_freq=5):
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
        super().__init__(animal_type=animal_type,
                         num_agents=num_agents,
                         domain_size=domain_size,
                         start_position=start_position,
                         noise_amplitude=noise_amplitude,
                         visualize=visualize,
                         follow=follow,
                         graph_freq=graph_freq)

    def get_repulsion_orientations(self, positions, neighbours, perception_strengths):
        """
        Computes the new orientation based on repulsion for every agent.
        """
        neighbours_2d =  np.repeat(neighbours[:,:, np.newaxis], 2, axis=2)
        bearings = positions[:,np.newaxis,:] - positions
        bearings = bearings * neighbours_2d
        bearings_norm = np.repeat(np.linalg.norm(bearings, axis=2)[:,:, np.newaxis], 2, axis=2)
        rij = np.divide(bearings, bearings_norm, out=np.zeros_like(bearings), where=bearings_norm!=0)
        rij_norm = np.linalg.norm(rij)
        orientations = np.sum(perception_strengths * np.divide(rij, rij_norm, out=np.zeros_like(rij), where=rij_norm!=0), axis=1)
        return ac.compute_angles_for_orientations(orientations=orientations)
    
    def get_alignment_orientations(self, headings, neighbours, perception_strengths):
        """
        Computes the new orientation based on alignment for every agent.
        """
        neighbours_2d =  np.repeat(neighbours[:,:, np.newaxis], 2, axis=2)
        orientations = ac.compute_u_v_coordinates_for_angles(headings)
        orientations = orientations * neighbours_2d
        norm_aligned = np.linalg.norm(orientations, axis=1)
        new_orientations = np.sum(perception_strengths * np.divide(orientations, norm_aligned, out=np.zeros_like(orientations), where=norm_aligned!=0), axis=1)
        return ac.compute_angles_for_orientations(orientations=new_orientations)
    
    def get_attraction_orientations(self, positions, neighbours, perception_strengths):
        """
        Computes the new orientation based on attraction for every agent.
        """
        neighbours_2d =  np.repeat(neighbours[:,:, np.newaxis], 2, axis=2)
        bearings = positions[:,np.newaxis,:] - positions
        bearings = bearings * neighbours_2d
        bearings_norm = np.repeat(np.linalg.norm(bearings, axis=2)[:,:, np.newaxis], 2, axis=2)
        rij = np.divide(bearings, bearings_norm, out=np.zeros_like(bearings), where=bearings_norm!=0)
        rij_norm = np.linalg.norm(rij)
        orientations = -np.sum(perception_strengths * np.divide(rij, rij_norm, out=np.zeros_like(rij), where=rij_norm!=0), axis=1)
        return ac.compute_angles_for_orientations(orientations=orientations)
    
    def compute_new_orientations(self, agents):
        """
        Computes the new orientations based on the following logic:
        1) if there are any neighbours within the repulsion zone, then the orientation based on those neighbours is used
        2) if there are neighbours in both the alignment and attraction zones, then the average of the orientations of these two zones is used
        3) if there are neighbours in the alignment or attraction zone but not both, then the respective orientation is used
        4) if there are no neighbours at all, the orientation remains unchanged
        """
        positions = np.column_stack((agents[:,0], agents[:,1]))
        distances, angles = self.compute_distances_and_angles()

        repulsed = self.get_repulsion_neighbours(distances=distances)
        aligned = self.get_alignment_neighbours(distances=distances)
        attracted = self.get_attraction_neighbours(distances=distances)

        perception_strengths = pstrength.compute_perception_strengths(distances=distances, angles=angles, shape=(self.num_agents, self.num_agents), animal_type=self.animal_type)
        perception_strengths = np.repeat(perception_strengths[:,:, np.newaxis], 2, axis=2)

        repulsion_orientations = self.get_repulsion_orientations(positions=positions, neighbours=repulsed, perception_strengths=perception_strengths)
        alignment_orientations = self.get_alignment_orientations(headings=agents[:,2], neighbours=aligned, perception_strengths=perception_strengths)
        attraction_orientations = self.get_attraction_orientations(positions=positions, neighbours=attracted, perception_strengths=perception_strengths)

        avg_orientations_aligned_attracted = (alignment_orientations + attraction_orientations) / 2

        has_repulsed = np.count_nonzero(repulsed, axis=1) > 0
        has_aligned = np.count_nonzero(aligned, axis=1) > 0
        has_attracted = np.count_nonzero(attracted, axis=1) > 0

        new_orientations = np.where((has_aligned), alignment_orientations, agents[:,2])
        new_orientations = np.where((has_attracted), attraction_orientations, new_orientations)
        new_orientations = np.where((has_aligned & has_attracted), avg_orientations_aligned_attracted, new_orientations)
        new_orientations = np.where(has_repulsed, repulsion_orientations, new_orientations)

        return new_orientations