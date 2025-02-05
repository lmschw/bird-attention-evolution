
from enum import Enum

class WeightOptions(str, Enum):
    CLOSEST_DISTANCES = "closest_distance"
    CLOSEST_BEARINGS = "closest_bearings"
    AVG_DISTANCES = "average_distances"
    AVG_BEARINGS = "average_bearings"
    NUM_VISIBLE_AGENTS = "num_visible_agents"
    PREVIOUS_HEAD_ANGLES = "previous_head_angles"
    AVG_PERCEPTION_STRENGTHS = "average_perception_strengths"