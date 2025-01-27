import math
import numpy as np

def get_relative_positions(agents):
    x_diffs = agents[:,np.newaxis,0]-agents[:,0]    
    y_diffs = agents[:,np.newaxis,1]-agents[:,1]    
    return np.arctan2(y_diffs, x_diffs)

def get_relative_headings(agents):  
    return agents[:,np.newaxis,2]-agents[:,2]    

if __name__ == "__main__":
    x = np.array([ 1,  2,  3])
    y = np.array([1, 2, 3])
    h = np.array([np.pi, 0.5*np.pi, 1.5*np.pi])

    agents = np.column_stack([x, y, h])

    print(get_relative_positions(agents=agents))
    print(get_relative_headings(agents=agents))