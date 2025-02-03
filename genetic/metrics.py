from enum import Enum

"""
Indicates which particles should be coloured during the experiment to facilitate better understanding in the video rendering.
"""
class Metrics(str, Enum):
    COHESION = "C",
    ORDER = "O",
    COHESION_AND_ORDER = "CAO"


