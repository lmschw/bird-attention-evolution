import numpy as np

import general.normalisation as normal
import general.angle_conversion as ac

"""
Contains methods to compute perception strengths.
"""

DIST_MOD = 0.001

def compute_perception_strengths(distances, angles, shape, animal_type):
    """
    Computes the perception strengths based on the bearings, distances and animal_type provided.
    """
    dist_absmax = animal_type.sensing_range
    vision_strengths_overall = []
    for focus_area in animal_type.focus_areas:
        dist_min = focus_area.comfortable_distance[0]
        dist_max = focus_area.comfortable_distance[1]

        angles_2_pi = ac.wrap_to_2_pi(angles)

        f_angle = ac.wrap_to_2_pi(focus_area.azimuth_angle_position_horizontal)
        focus_angle = np.reshape(np.concatenate([[f_angle] * shape[0] for i in range(shape[1])]), shape)
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
