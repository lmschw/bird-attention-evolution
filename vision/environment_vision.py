import numpy as np
import geometry.angle_conversion as aconv
import geometry.normalisation as normal

def get_landmarks(self, bird, current_landmarks):
    # TODO: determine the landmarks that are currently within the field of vision and recognised by the bird
    pass

def compute_distances(agents):
    relative_positions = aconv.get_relative_positions(agents=agents)
    rij = relative_positions[:,np.newaxis,:]-relative_positions   
    return np.sum(rij**2,axis=2)

def compute_perception_strengths_conspecifics(agents, bird_type):
    azimuth_angles_positions = aconv.get_relative_positions(agents=agents)
    azimuth_angles_positions_pi = wrap_to_pi(azimuth_angles_positions)
    overall_perception_strengths = []
    min_distances = []
    min_angles = []
    for focus_area in bird_type.focus_areas:
        # The closer to the focus, the stronger the perception of the input
        focus = focus_area.azimuth_angle_position_horizontal
        min_angle = wrap_angle_to_pi(focus - focus_area.angle_field_horizontal)
        max_angle = wrap_angle_to_pi(focus + focus_area.angle_field_horizontal)

        # the base perception strength is equal to the percentage of the visual field based around the focus
        strengths = np.absolute(focus - azimuth_angles_positions_pi) / focus_area.angle_field_horizontal

        # print("initial strengths")
        # print(strengths)

        # if a conspecific is not within the field of vision, then its strength must be set to zero
        in_field_of_vision_min = azimuth_angles_positions_pi >= min_angle
        in_field_of_vision_max = azimuth_angles_positions_pi <= max_angle
        perception_strengths = np.where((in_field_of_vision_min & in_field_of_vision_max), strengths, 0)
        # print("in field of vision:")
        # print(perception_strengths)

        # if the conspecific is not at a distance where the eye can comfortably focus, we set the strength to 0.1
        distances = compute_distances(agents=agents)
        in_comfortable_distance_min = distances >= focus_area.comfortable_distance[0]
        in_comfortable_distance_max = distances <= focus_area.comfortable_distance[1]
        perception_strengths = np.where(((in_comfortable_distance_min & in_comfortable_distance_max) | (perception_strengths == 0)), perception_strengths, 0.1)
        # print("comfortable distance")
        # print(perception_strengths)

        # we set the agents' own perception strength to zero
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
    min_dist_idx = np.argmin(min_distances, axis=1)
    min_dist_idx = [[i, min_dist_idx[i]] for i in range(len(min_dist_idx))]

    min_dists_final = [min_distances[idx[0], idx[1]] for idx in min_dist_idx]
    min_angles_final = [min_angles[idx[0], idx[1]] for idx in min_dist_idx]

    normalised_perception_strengths = normal.normalise(np.concatenate(overall_perception_strengths, axis=1))

    num_agents = len(normalised_perception_strengths/len(bird_type.focus_areas))
    normalised_reshaped_perception_strengths = [normalised_perception_strengths[:, i * num_agents: (i+1) * num_agents] for i in range(len(bird_type.focus_areas))]

    return normalised_reshaped_perception_strengths, (min_dists_final, min_angles_final)


def wrap_to_pi(arr):
    """
    Wrapes the angles to [-pi, pi]

    """
    arr = arr % (3.1415926 * 2)
    arr = (arr + (3.1415926 * 2)) % (3.1415926 * 2)

    arr[arr > 3.1415926] = arr[arr > 3.1415926] - (3.1415926 * 2)

    return arr

def wrap_angle_to_pi(x):
    if x < 0:
        x += 2 * np.pi
    x = x % (2*np.pi)
    if x > np.pi:
        return -(2*np.pi - x)
    return x
