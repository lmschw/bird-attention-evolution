import numpy as np
import random, copy
from simulator.head_movement.enum_weight_options import WeightOptions

"""
Describes the type of input data to the neural network updating the head direction.
"""

MAX_INPUT = 1_000_000

def get_input_value_for_weight_option(weight_option, current_head_angles, distances, angles, perception_strengths):
    closest_neighbour = np.argmin(distances, axis=1)
    pstrength = copy.deepcopy(perception_strengths)
    np.fill_diagonal(pstrength, 0)
    closest_neighbour_fovea = np.argmax(pstrength, axis=1)
    match weight_option:
        case WeightOptions.CLOSEST_DISTANCES:
            closest_distances = distances[np.arange(len(distances)), closest_neighbour]
            return closest_distances
        case WeightOptions.CLOSEST_BEARINGS:
            closest_bearings = angles[np.arange(len(angles)),closest_neighbour]
            return closest_bearings
        case WeightOptions.AVG_DISTANCES:
            average_distances = np.average(distances, axis=1)
            return average_distances
        case WeightOptions.AVG_BEARINGS:
            average_bearings = np.average(angles, axis=1)
            return average_bearings
        case WeightOptions.DISTANCE_CLOSEST_FOVEA:
            distances_fovea = distances[np.arange(len(distances)),closest_neighbour_fovea]
            return distances_fovea
        case WeightOptions.BEARING_CLOSEST_FOVEA:
            bearings_fovea = angles[np.arange(len(angles)),closest_neighbour_fovea]
            return bearings_fovea
        case WeightOptions.NUM_VISIBLE_AGENTS:
            num_visible_agents = np.count_nonzero(perception_strengths, axis=1)
            return num_visible_agents
        case WeightOptions.NUM_AGENTS_LEFT:
            num_left = np.count_nonzero(perception_strengths * ((angles + current_head_angles) < 0), axis=1)
            return num_left
        case WeightOptions.NUM_AGENTS_RIGHT:
            num_right = np.count_nonzero(perception_strengths * ((angles + current_head_angles) > 0), axis=1)
            return num_right
        case WeightOptions.PREVIOUS_HEAD_ANGLES:
            previous_head_angles = current_head_angles
            return previous_head_angles
        case WeightOptions.AVG_PERCEPTION_STRENGTHS:
            average_perception_strengths = np.average(perception_strengths, axis=1)
            return average_perception_strengths
        case WeightOptions.RANDOM:
            return np.random.uniform(0, 2*np.pi, (len(distances)))