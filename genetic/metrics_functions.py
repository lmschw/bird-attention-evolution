import numpy as np
import math

import general.angle_conversion as ac

"""
Contains methods for metrics evaluations
"""

def compute_global_order(agents):
    """
    Computes the global order within a group of agents
    """
    orientations = ac.compute_u_v_coordinates_for_angles(agents[:,2])
    sum_orientation = np.sum(orientations[np.newaxis,:,:],axis=1)
    return np.divide(np.sqrt(np.sum(sum_orientation**2,axis=1)), len(orientations))[0]

def compute_cohesion(agents):
    """
    Computes the average distance to the centroid in order to measure cohesion
    """
    positions = np.column_stack((agents[:,0], agents[:,1]))
    centroid = np.mean(positions, axis=0)
    distances = [math.dist(pos, centroid) for pos in positions]
    return np.average(distances)