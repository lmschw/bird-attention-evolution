from enum import Enum

"""
Contains all the different ways of selecting a subset of neighbours from all possible neighbours.
"""
class NeighbourSelectionMechanism(str, Enum):
    NEAREST = "N",
    FARTHEST = "F",
    CLOSEST_TO_FOVEA = "CF"