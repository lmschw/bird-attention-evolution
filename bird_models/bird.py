import numpy as np

class Bird:
    # TODO implement flapping frequency and energy level
    def __init__(self, name, speeds, wingspan, length, head_range_half,
                 focus_areas, preferred_distance_front_back, preferred_distance_left_right,
                 sensing_range):
        self.name = name
        self.speeds = speeds
        self.wingspan = wingspan
        self.length = length
        self.head_range_half = head_range_half
        self.focus_areas = focus_areas
        self.preferred_distance_front_back = preferred_distance_front_back
        self.preferred_distance_left_right = preferred_distance_left_right
        self.sensing_range = sensing_range

    def set_path(self, path):
        self.path = path
