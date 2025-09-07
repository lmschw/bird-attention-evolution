import numpy as np

import general.normalisation as normal
import general.angle_conversion as ac
import general.occlusion as occ

"""
Contains methods to compute perception strengths.
"""

DIST_MOD = 0.001

def compute_perception_strengths_and_foveal_distances(distances, angles, shape, animal_type, head_orientations=[]):
    """
    Computes the perception strengths based on the bearings, distances and animal_type provided.
    """
    dist_absmax = animal_type.sensing_range
    vision_strengths_overall = []
    distances_to_closest_fovea = []
    for focus_area in animal_type.focus_areas:
        dist_min = focus_area.comfortable_distance[0]
        dist_max = focus_area.comfortable_distance[1]


        angles_2_pi = ac.wrap_to_2_pi(angles)
        if len(head_orientations) > 0:
            f_angle = ac.wrap_to_2_pi(focus_area.azimuth_angle_position_horizontal + head_orientations)
            focus_angle = np.reshape(np.repeat(f_angle, shape[0]), shape)
        else:
            f_angle = ac.wrap_to_2_pi(focus_area.azimuth_angle_position_horizontal)
            focus_angle = np.reshape(np.concatenate([[f_angle] * shape[0] for i in range(shape[1])]), shape)
        angle_diffs_focus = angles_2_pi - focus_angle

        if len(distances_to_closest_fovea) == 0:
            distances_to_closest_fovea = angle_diffs_focus
        else:
            for i in range(len(distances_to_closest_fovea)):
                for j in range(len(distances_to_closest_fovea[i])):
                    if np.absolute(distances_to_closest_fovea[i][j]) > np.absolute(angle_diffs_focus[i][j]):
                        distances_to_closest_fovea[i][j] = angle_diffs_focus[i][j]

        perception_strengths = 1 - (np.absolute(angle_diffs_focus)/focus_area.angle_field_horizontal)
        
        vision_strengths = np.zeros(shape)
        # if the agent is within the cone of the field of vision, it is perceived
        vision_strengths = np.where(np.absolute(angle_diffs_focus) <= focus_area.angle_field_horizontal, perception_strengths, vision_strengths)

        # if the agent is within the foveal region, it is observed with full acuity, outside it can be blurry
        vision_strengths = np.where((np.absolute(distances) > focus_area.angle_fovea_horizontal), animal_type.fovea_discount_factor * vision_strengths, animal_type.foveal_acuity * vision_strengths)
      
        # if an agent is outside of the comfortable viewing distance, it is perceived but with a very low percentage
        vision_strengths = np.where(np.absolute(distances) < dist_min, DIST_MOD * vision_strengths, vision_strengths)
        vision_strengths = np.where(((np.absolute(distances) > dist_max)&(np.absolute(distances) <=dist_absmax)), DIST_MOD * vision_strengths, vision_strengths)
    
        vision_strengths_overall.append(vision_strengths)
    vision_strengths_overall = np.array(vision_strengths_overall)
    vision_strengths_overall = np.sum(vision_strengths_overall.T, axis=2).T
    vision_strengths_overall = normal.normalise(vision_strengths_overall)
    return vision_strengths_overall, distances_to_closest_fovea

def compute_perception_strengths(distances, angles, shape, animal_type, head_orientations=[]):
    perception_strengths, _ = compute_perception_strengths_and_foveal_distances(distances=distances,
                                                                                angles=angles,
                                                                                shape=shape,
                                                                                animal_type=animal_type,
                                                                                head_orientations=head_orientations)
    return perception_strengths

def compute_distances_to_closest_fovea(distances, angles, shape, animal_type, head_orientations=[]):
    _, distances_to_closest_fovea = compute_perception_strengths_and_foveal_distances(distances=distances,
                                                                                        angles=angles,
                                                                                        shape=shape,
                                                                                        animal_type=animal_type,
                                                                                        head_orientations=head_orientations)
    return distances_to_closest_fovea

def compute_perception_strengths_with_occlusion_conspecifics(agents, distances, angles, shape, animal_type, head_orientations=[], landmarks=[]):
    perception_strengths = compute_perception_strengths(distances=distances,
                                                        angles=angles,
                                                        shape=shape,
                                                        animal_type=animal_type,
                                                        head_orientations=head_orientations)
    occluded_conspecifics = occ.compute_occluded_mask(agents=agents, animal_type=animal_type)
    if len(landmarks) > 0:
        occluded_landmarks = occ.compute_occluded_mask_landmarks(agents=agents, animal_type=animal_type, landmarks=landmarks)
    else:
        occluded_landmarks = np.full(shape, False)
    
    perception_strengths[occluded_conspecifics] = 0
    perception_strengths[occluded_landmarks] = 0
    return perception_strengths

def compute_perception_strengths_with_occlusion_predation(predators, prey, distances, angles, shape, animal_type, head_orientations=[], is_prey=True):
    perception_strengths = compute_perception_strengths(distances=distances,
                                                        angles=angles,
                                                        shape=shape,
                                                        animal_type=animal_type,
                                                        head_orientations=head_orientations)
    if is_prey:
        agents = np.concatenate((prey, predators))
    else:
        agents = np.concatenate((predators, prey))

    occluded = occ.compute_occluded_mask(agents=agents, animal_type=animal_type)
    relevant_occluded = occluded[shape[0]:, :shape[0]]
    perception_strengths[relevant_occluded.T] = 0
    return perception_strengths
