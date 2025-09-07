
from enum import Enum

"""
Enum containing the options for input for the neural network for head orientation updates.
"""

class WeightOptions(str, Enum):
    CLOSEST_DISTANCES = "closest_distance"
    CLOSEST_BEARINGS = "closest_bearings"
    AVG_DISTANCES = "average_distances"
    AVG_BEARINGS = "average_bearings"
    DISTANCE_CLOSEST_FOVEA ="closest_dist_fovea"
    BEARING_CLOSEST_FOVEA = "closest_bear_fovea"
    NUM_VISIBLE_AGENTS = "num_visible_agents"
    NUM_AGENTS_LEFT = "num_agents_left"
    NUM_AGENTS_RIGHT = "num_agents_right"
    PREVIOUS_HEAD_ANGLES = "previous_head_angles"
    AVG_PERCEPTION_STRENGTHS = "average_perception_strengths"
    RANDOM = "random"