import math
import numpy as np

"""
Contains methods to convert angles and orientations
"""

def get_relative_positions(agents):
    """
    Returns the relative positions between all agents.
    """
    x_diffs = agents[:,np.newaxis,0]-agents[:,0]    
    y_diffs = agents[:,np.newaxis,1]-agents[:,1]    
    return np.arctan2(y_diffs, x_diffs)

def get_relative_positions_landmarks(agents, landmarks):
    """
    Returns the relative positions of the landmarks as viewed by the agents
    """
    x_diffs = []
    y_diffs = []
    for agent_idx in range(len(agents)):
        landmark_x_diffs = []
        landmark_y_diffs = []
        for landmark in landmarks:
            landmark_x_diffs.append(agents[agent_idx,0]-landmark.position[0])   
            landmark_y_diffs.append(agents[agent_idx,1]-landmark.position[1])   
        x_diffs.append(np.array(landmark_x_diffs).flatten())   
        y_diffs.append(np.array(landmark_y_diffs).flatten())   
    return np.arctan2(y_diffs, x_diffs).T

def get_relative_headings(agents): 
    """
    Returns the relative orientations of the agents.
    """ 
    return agents[:,np.newaxis,2]-agents[:,2]    

def wrap_to_pi(x):
    """
    Wraps the angles to [-pi, pi]
    """
    x = x % (2 * np.pi)
    x = (x + (2 * np.pi)) % (2 * np.pi)

    x[x > np.pi] = x[x > np.pi] - (2 * np.pi)

    return x

def wrap_angle_to_pi(x):
    """
    Wraps a single angle to [-pi, pi]
    """
    if x < 0:
        x += 2 * np.pi
    x = x % (2*np.pi)
    if x > np.pi:
        return -(2*np.pi - x)
    return x

def wrap_to_2_pi(x):
    """
    Wraps the angle(s) to [0, 2pi]
    """
    return (2*np.pi*x) % (2*np.pi)

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

def compute_angles_for_orientations(orientations):
    """
    Computes the angle in radians based on the (u,v)-coordinates of the current orientation.

    Params:
        - orientation (array of floats): the current orientation in (u,v)-coordinates

    Returns:
        A float representin the angle in radians.
    """
    return np.arctan2(orientations[:, 1], orientations[:, 0])
