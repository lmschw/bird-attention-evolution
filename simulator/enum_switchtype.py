from enum import Enum

"""
Contains the different hyperparameters that can be used to switch between behaviours.
"""
class SwitchType(Enum):
    NEIGHBOUR_SELECTION_MECHANISM = "N", "nsms",
    K = "K", "ks", 

    def __init__(self, val, switchTypeValueKey):
        self.val = val
        self.switchTypeValueKey = switchTypeValueKey