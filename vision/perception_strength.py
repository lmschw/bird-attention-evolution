import numpy as np

import general.normalisation as normal
import general.angle_conversion as ac
import general.occlusion as occ

"""
Contains methods to compute perception strengths.
"""

DIST_MOD = 0.001

def compute_perception_strengths(distances, angles, shape, animal_type, head_orientations=[]):
    """
    Computes the perception strengths based on the bearings, distances and animal_type provided.
    """
    dist_absmax = animal_type.sensing_range
    vision_strengths_overall = []
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
    return vision_strengths_overall

def compute_perception_strengths_with_occlusion_conspecifics(agents, distances, angles, shape, animal_type, head_orientations=[], landmarks=[]):
    perception_strengths = compute_perception_strengths(distances=distances,
                                                        angles=angles,
                                                        shape=shape,
                                                        animal_type=animal_type,
                                                        head_orientations=head_orientations)
    positions = np.column_stack((agents[:,0], agents[:,1]))
    occluded_conspecifics = occ.get_occlusion_mask(positions=positions,
                                                   orientations=agents[:,2],
                                                   animal_type=animal_type)
    if len(landmarks) > 0:
        occluded_landmarks = occ.get_occlusion_mask_landmarks(positions=positions,
                                                              orientations=agents[:,2],
                                                              landmarks=landmarks,
                                                              animal_type=animal_type)
    else:
        occluded_landmarks = np.full(shape, True)
    
    perception_strengths[occluded_conspecifics] = 0
    perception_strengths[occluded_landmarks] = 0
    return perception_strengths