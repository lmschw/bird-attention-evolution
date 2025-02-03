import numpy as np
import math

def compute_u_v_coordinates_for_angles(angles):
    """
    Computes the (u,v)-coordinates based on the angle.

    Params:
        - angle (float): the angle in radians

    Returns:
        An array containing the [u, v]-coordinates corresponding to the angle.
    """
    # compute the uv-coordinates
    U = np.cos(angles)
    V = np.sin(angles)

    return np.column_stack((U,V))
    

def compute_global_order(agents):
    orientations = compute_u_v_coordinates_for_angles(agents[:,2])
    sum_orientation = np.sum(orientations[np.newaxis,:,:],axis=1)
    return np.divide(np.sqrt(np.sum(sum_orientation**2,axis=1)), len(orientations))[0]

def compute_cohesion(agents):
    positions = np.column_stack((agents[:,0], agents[:,1]))
    centroid = np.mean(positions, axis=0)
    distances = [math.dist(pos, centroid) for pos in positions]
    return np.average(distances)