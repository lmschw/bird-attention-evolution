"""
The base class for all animals, i.e. birds, fish etc.

Attributes:
name                            -   string              -   the name of the animal, mainly intended for saving and printing
speeds                          -   array of floats     -   the min, average and max speed of the animal
width                           -   float[m]            -   the width of the animal, i.e. its size on the left-right axis, e.g. the wingspan for a bird
length                          -   float[m]            -   the length of an animal, e.g. nose to tail for a fish
head_range_half                 -   float[rad]          -   how much the animal can turn its head to either side starting from a position looking straight ahead
max_turn_angle                  -   float[rad]          -   how much the animal can turn at any given timestep
focus_areas                     -   list of FocusArea   -   the visual field of the animal defined by its foveal projections and associated ranges
preferred_distance_front_back   -   array of floats     -   how close the animal likes others to get to its arse and head described by a min (its minimum personal space) and a max (when it starts feeling lonely or exposed)
preferred_distance_left_right   -   array of floats     -   how close the animal likes others to get to its left and right side described by a min (its minimum personal space) and a max (when it starte feeling lonely or exposed)
sensing_range                   -   float               -   how far the animal can see overall
"""

class Animal:
    # TODO implement energy level
    def __init__(self, name, speeds, width, length, head_range_half, max_turn_angle,
                 focus_areas, preferred_distance_front_back, preferred_distance_left_right,
                 sensing_range):
        self.name = name
        self.speeds = speeds
        self.width = width
        self.length = length
        self.head_range_half = head_range_half
        self.max_turn_angle = max_turn_angle
        self.focus_areas = focus_areas
        self.preferred_distance_front_back = preferred_distance_front_back
        self.preferred_distance_left_right = preferred_distance_left_right
        self.sensing_range = sensing_range

    def set_path(self, path):
        self.path = path
