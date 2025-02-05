import numpy as np
from simulator.enum_weight_options import WeightOptions

MAX_INPUT = 1_000_000

def get_input_value_for_weight_option(weight_option, bearings, distances, angles, perception_strengths):
    closest_neighbour = np.argmin(distances, axis=1)
    match weight_option:
        case WeightOptions.CLOSEST_DISTANCES:
            closest_distances = distances[closest_neighbour]
            return closest_distances[0]
        case WeightOptions.CLOSEST_BEARINGS:
            closest_bearings = angles[closest_neighbour]
            return closest_bearings[0]
        case WeightOptions.AVG_DISTANCES:
            average_distances = np.average(distances, axis=1)
            return average_distances
        case WeightOptions.AVG_BEARINGS:
            average_bearings = np.average(angles, axis=1)
            return average_bearings
        case WeightOptions.NUM_VISIBLE_AGENTS:
            num_visible_agents = np.count_nonzero(perception_strengths, axis=1)
            return num_visible_agents
        case WeightOptions.PREVIOUS_HEAD_ANGLES:
            previous_head_angles = bearings
            return previous_head_angles
        case WeightOptions.AVG_PERCEPTION_STRENGTHS:
            average_perception_strengths = np.average(perception_strengths, axis=1)
            return average_perception_strengths