import math
import numpy as np

def get_relative_positions(agents):
    x_diffs = agents[:,np.newaxis,0]-agents[:,0]    
    y_diffs = agents[:,np.newaxis,1]-agents[:,1]    
    return np.arctan2(y_diffs, x_diffs)

def get_relative_positions_landmarks(agents, landmarks):
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
    return agents[:,np.newaxis,2]-agents[:,2]    

def wrap_to_pi(self, x):
    """
    Wraps the angles to [-pi, pi]
    """
    x = x % (2 * np.pi)
    x = (x + (2 * np.pi)) % (2 * np.pi)

    x[x > np.pi] = x[x > np.pi] - (2 * np.pi)

    return x

if __name__ == "__main__":
    x = np.array([ 1,  2,  3])
    y = np.array([1, 2, 3])
    h = np.array([np.pi, 0.5*np.pi, 1.5*np.pi])

    agents = np.column_stack([x, y, h])

    print(get_relative_positions(agents=agents))
    print(get_relative_headings(agents=agents))