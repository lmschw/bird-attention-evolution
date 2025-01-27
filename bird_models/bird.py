import numpy as np

class Bird:
    # TODO implement flapping frequency and energy level
    def __init__(self, average_speed, wingspan, length, head_range_half,
                 focus_areas, preferred_distance_front_back, preferred_distance_left_right):
        self.average_speed = average_speed
        self.wingspan = wingspan
        self.length = length
        self.head_range_half = head_range_half
        self.focus_areas = focus_areas
        self.preferred_distance_front_back = preferred_distance_front_back
        self.preferred_distance_left_right = preferred_distance_left_right

    def set_path(self, path):
        self.path = path
