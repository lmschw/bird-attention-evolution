import numpy as np

class Landmark:
    def __init__(self, name, position):
        self.name = name
        self.position = np.array(position)
