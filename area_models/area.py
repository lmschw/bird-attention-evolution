import numpy as np

"""
Represents the area in which the agents move with the relevant area size and landmarks.

Attributes:
area_size   -   tuple of floats     -   the size of the area
landmarks   -   list of Landmark    -   a list of all the landmarks within the area
"""

class Area:
    def __init__(self, area_size, landmarks):
        self.area_size = area_size
        self.landmarks = landmarks

    def set_paths(self, paths):
        self.paths = paths

    def create_paths(self, n_landmarks, start_position, target_area_center):
        # TODO find an automatic way to create paths
        pass

    def pick_paths(self, n_agents):
        return np.random.choice(self.paths, n_agents)
