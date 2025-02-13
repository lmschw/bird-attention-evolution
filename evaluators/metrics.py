from enum import Enum

"""
Indicates how the success of a simulation should be measured
"""
class Metrics(str, Enum):
    COHESION = "C",
    ORDER = "O",
    COHESION_AND_ORDER = "CAO",
    CORRIDOR_DISTRIBUTION = "CD",
    SUCCESS_PERCENTAGE = "SP"


