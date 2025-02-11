import numpy as np

import general.normalisation as normal
import general.angle_conversion as ac

"""
Contains methods to compute perception strengths.
"""

def compute_perception_strengths(azimuth_angles_positions, distances, animal_type, is_conspecifics=True):
    """
    Computes the perception strengths based on the bearings, distances and animal_type provided.
    """
    azimuth_angles_positions_pi = ac.wrap_to_pi(azimuth_angles_positions)
    overall_perception_strengths = []
    min_distances = []
    min_angles = []
    for focus_area in animal_type.focus_areas:
        # The closer to the focus, the stronger the perception of the input
        focus = focus_area.azimuth_angle_position_horizontal
        min_angle = ac.wrap_angle_to_pi(focus - focus_area.angle_field_horizontal)
        max_angle = ac.wrap_angle_to_pi(focus + focus_area.angle_field_horizontal)

        # the base perception strength is equal to the percentage of the visual field based around the focus
        strengths = 1-(np.absolute(focus - azimuth_angles_positions_pi) / focus_area.angle_field_horizontal)

        # print("initial strengths")
        # print(strengths)

        # if a conspecific is not within the field of vision, then its strength must be set to zero
        in_field_of_vision_min = azimuth_angles_positions_pi >= min_angle
        in_field_of_vision_max = azimuth_angles_positions_pi <= max_angle
        perception_strengths = np.where((in_field_of_vision_min & in_field_of_vision_max), strengths, 0)
        # print("in field of vision:")
        # print(perception_strengths)

        # if the conspecific is not at a distance where the eye can comfortably focus, we set the strength to 0.1
        in_comfortable_distance_min = distances >= focus_area.comfortable_distance[0]
        in_comfortable_distance_max = distances <= focus_area.comfortable_distance[1]
        perception_strengths = np.where(((in_comfortable_distance_min & in_comfortable_distance_max) | (perception_strengths == 0)), perception_strengths, 0.1)
        # print("comfortable distance")
        # print(perception_strengths)

        # we set the agents' own perception strength to zero
        if is_conspecifics:
            np.fill_diagonal(perception_strengths, 0)
        # print("diagonals removed:")
        # print(perception_strengths)
        overall_perception_strengths.append(perception_strengths)

        dist_eval = np.where((perception_strengths > 0), distances, np.inf)
        min_dist_idx = np.argmin(dist_eval, axis=1)
        min_distances.append(dist_eval[min_dist_idx])
        min_angles.append(azimuth_angles_positions_pi[min_dist_idx])

    min_distances = np.concatenate(min_distances).T
    min_angles = np.concatenate(min_angles).T
    min_dist_idx_basic = np.argmin(min_distances, axis=1)
    min_dist_idx = [[i, min_dist_idx_basic[i]] for i in range(len(min_dist_idx_basic))]

    min_dists_final = [min_distances[idx[0], idx[1]] for idx in min_dist_idx]
    min_angles_final = [min_angles[idx[0], idx[1]] for idx in min_dist_idx]

    normalised_perception_strengths = normal.normalise(np.concatenate(overall_perception_strengths, axis=1))

    normalised_reshaped_perception_strengths = np.sum(normalised_perception_strengths.reshape((1, len(azimuth_angles_positions),len(azimuth_angles_positions[0]),len(animal_type.focus_areas))), axis=3)
    normalised_reshaped_perception_strengths = normalised_reshaped_perception_strengths.reshape(distances.shape)
    return normalised_reshaped_perception_strengths, (min_dists_final, min_angles_final, min_dist_idx_basic)

