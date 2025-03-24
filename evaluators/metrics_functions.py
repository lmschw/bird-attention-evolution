import numpy as np
import math
from sklearn.cluster import AgglomerativeClustering

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

def compute_predator_angle_to_centroid(centroid, predator_position):
    centroid_position = [centroid[0], centroid[1]]
    centroid_orientation = centroid[2]
    return np.arctan2(centroid_position[1]-predator_position[1], centroid_position[0]-predator_position[0]) - centroid_orientation

def find_clusters(positions, orientations, threshold):
    """
    Find clusters in the data using AgglomerativeClustering.

    Params:
        - positions (array of arrays of float): the position of every particle at every timestep
        - orientations (array of arrays of float): the orientation of every particle at every timestep
        - threshold (float): the threshold used to cut the tree in AgglomerativeClustering

    Returns:
        The number of clusters, the labels of the clusters
    """
    cluster = AgglomerativeClustering(n_clusters=None, metric='euclidean', linkage='single', compute_full_tree=True, distance_threshold=threshold)

    # Cluster the data
    cluster.fit_predict(orientations)

    # number of clusters
    nClusters = 1+np.amax(cluster.labels_)

    return nClusters, cluster.labels_

def evaluate_splits_and_turns(prey_data, predator_data):
    splits = []
    turns = []
    angles = []
    for iter in range(len(prey_data)):
        splits_iter = 0
        turns_iter = 0
        angle_iter = 0
        pred = predator_data[iter]
        last_dist = np.inf
        for t in range(len(prey_data[iter])):
            centroid= np.average(prey_data[iter][t].T, axis=1)
            centroid_predators = np.average(predator_data[iter][t].T, axis=1)
            dist = math.dist(centroid[:2], centroid_predators[:2])
            if dist < 25 and last_dist >= dist:
                orientations = ac.compute_u_v_coordinates_for_angles(prey_data[iter][t])
                nclusters, _ = find_clusters(positions=prey_data[iter][t][:,:1], orientations=orientations, threshold=0.1)
                if nclusters > 1:
                    splits_iter = 1
                    turns_iter = 0
                else:
                    splits_iter = 0
                    turns_iter = 1
                angle_iter = compute_predator_angle_to_centroid(centroid=centroid, predator_position=[centroid_predators[0],centroid_predators[1]])
            last_dist = dist
        splits.append(splits_iter)
        turns.append(turns_iter)
        angles.append(angle_iter)
    angles_splits = np.where(splits, angles, 0)
    angles_turns = np.where(turns, angles, 0)
    avg_angle_splits = np.average(angles_splits)
    avg_angle_turns = np.average(angles_turns)

    return np.sum(splits)/len(prey_data), np.sum(turns)/len(prey_data), avg_angle_splits, avg_angle_turns

