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

def compute_cohesion(agents, animal_type, normalised=False):
    """
    Computes the average distance to the centroid in order to measure cohesion
    """
    positions = np.column_stack((agents[:,0], agents[:,1]))
    centroid = np.mean(positions, axis=0)
    distances = [math.dist(pos, centroid) for pos in positions]
    if normalised:
        return 1/(np.average(distances) / animal_type.width)
    else:
        return np.average(distances)

def evaluate_cohesion(data, animal_type, normalised=False):
    cohesion_results = {t: [] for t in range(len(data[0]))}
    for iter in range(len(data)):
        for t in range(len(data[iter])):
            result = compute_cohesion(data[iter][t], animal_type, normalised)
            cohesion_results[t].append(result)
    return {t: np.average(cohesion_results[t]) for t in range(len(data[0]))}

def evaluate_order(data):
    order_results = {t: [] for t in range(len(data[0]))}
    for iter in range(len(data)):
        for t in range(len(data[iter])):
            result = compute_global_order(data[iter][t])
            order_results[t].append(result)
    return {t: np.average(order_results[t]) for t in range(len(data[0]))}

def evaluate_corridor_selection(data, corridor_centers):
    corridor_diff_x = np.absolute(corridor_centers[0][0]-corridor_centers[1][0])
    corridor_diff_y = np.absolute(corridor_centers[0][1]-corridor_centers[1][1])

    point_check = 'x' # the axis along which we're moving
    if corridor_diff_x > corridor_diff_y:
        point_check = 'y'

    count_all_same_corridor = 0
    for iter in range(len(data)):
        for t in range(len(data[iter])):
            if point_check == 'x' and data[iter][t][0,0] >= corridor_centers[0][0]:
                result = np.absolute(corridor_centers[0][1]-data[iter][t][:,1]) < np.absolute(corridor_centers[1][1]-data[iter][t][:,1])
                corridor_1_percentage = np.count_nonzero(result) / len(result)
                corridor_0_percentage = 1-corridor_1_percentage
                if corridor_0_percentage == 1 or corridor_1_percentage == 1:
                    count_all_same_corridor += 1
                break
            if point_check == 'y' and data[iter][t][0,1] >= corridor_centers[0][1]:
                result = np.absolute(corridor_centers[0][0]-data[iter][t][:,0]) < np.absolute(corridor_centers[1][0]-data[iter][t][:,0])
                corridor_1_percentage = np.count_nonzero(result) / len(result)
                corridor_0_percentage = 1-corridor_1_percentage
                if corridor_0_percentage == 1 or corridor_1_percentage == 1:
                    count_all_same_corridor += 1
                break
    return np.array([count_all_same_corridor/len(data), (len(data)-count_all_same_corridor)/len(data)])

def evaluate_success_percentage(data, corridor_endpoints):
    corridor_diff_x = np.absolute(corridor_endpoints[0][0]-corridor_endpoints[1][0])
    corridor_diff_y = np.absolute(corridor_endpoints[0][1]-corridor_endpoints[1][1])

    point_check = 'x' # the axis along which we're moving
    if corridor_diff_x > corridor_diff_y:
        point_check = 'y'

    success_percentages = []
    for iter in range(len(data)):
        t = -1
        if point_check == 'x':
            result = (corridor_endpoints[0][0] < data[iter][t][:,0]) & (corridor_endpoints[1][0] < data[iter][t][:,0])
            success_percentage = np.count_nonzero(result) / len(result)
            success_percentages.append(success_percentage)
        else:
            result = (corridor_endpoints[0][1] < data[iter][t][:,1]) & (corridor_endpoints[1][0] < data[iter][t][:,1])
            success_percentage = np.count_nonzero(result) / len(result)
            success_percentages.append(success_percentage)
    avg_success = np.average(success_percentages)
    return np.array([avg_success, 1-avg_success])

def evaluate_duration(data, corridor_endpoints):
    corridor_diff_x = np.absolute(corridor_endpoints[0][0]-corridor_endpoints[1][0])
    corridor_diff_y = np.absolute(corridor_endpoints[0][1]-corridor_endpoints[1][1])

    point_check = 'x' # the axis along which we're moving
    if corridor_diff_x > corridor_diff_y:
        point_check = 'y'

    durations = []
    for iter in range(len(data)):
        for t in range(len(data[iter])):
            if point_check == 'x':
                result = (corridor_endpoints[0][0] < data[iter][t][:,0]) & (corridor_endpoints[1][0] < data[iter][t][:,0])
                if np.count_nonzero(result) == len(result):
                    durations.append(t)
                    break
    return np.array([np.min(durations), np.average(durations), np.max(durations)])