import csv, os
import numpy as np
import pandas as pd

"""
Service containing methods to do with logging.
"""

def initialise_log_file_with_headers(headers, save_path):
    """
    Appends the headers to the csv file.

    Params:
        - headers (list of strings): the headers to be inserted into the file
        - save_path (string): the path of the file where the headers should be inserted

    Returns:
        Nothing.
    """
    with open(save_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)

def create_dicts(iter, t, agents, is_prey=None):
    agents_dicts = []
    for i, agent in enumerate(agents):
        if is_prey == None:
            agents_dicts.append({'iter': iter, 't': t, 'i': i, 'x': agent[0], 'y': agent[1], 'h': agent[2]})
        elif is_prey:
            agents_dicts.append({'iter': iter, 't': t, 'type': 'prey', 'i': i, 'x': agent[0], 'y': agent[1], 'h': agent[2]})
        else:
            agents_dicts.append({'iter': iter, 't': t, 'type': 'predator', 'i': i, 'x': agent[0], 'y': agent[1], 'h': agent[2]})

    return agents_dicts

def create_centroid_dict(iter, t, centroid):
    return {'iter': iter, 't': t, 'x': centroid[0], 'y': centroid[1]}

def log_results_to_csv(dict_list, save_path):
    """
    Logs the results to a csv file.

    Params:
        - dict_list (list of dictionaries): A list containing a dictionary for every data point (individual)
        - save_path (string): the path of the file where the headers should be inserted
        - prepare (boolean) [optional, default=False]: whether or not the dictionaries need to be prepared, i.e. if they still contain numpy arrays

    Returns:
        Nothing.
    """
    with open(save_path, 'a', newline='') as f:
        w = csv.writer(f)
        for dict in dict_list:
            w.writerow(dict.values())

def load_log_data(filepath, max_iters=None):
    df = pd.read_csv(filepath)
    if max_iters == None:
        max_iters = df['iter'].max()
    max_iter = min(df['iter'].max(),max_iters)
    data = []
    for iter in range(max_iter):
        print(f"loading data for iter {iter+1}/{max_iter}")
        data_iter = []
        df_iter = df[df['iter'] == iter]
        tmax = df_iter['t'].max()
        for t in range(tmax):
            df_t = df_iter[df_iter['t'] == t]
            data_iter.append(np.column_stack((df_t['x'], df_t['y'], df_t['h'])))
        data.append(data_iter)
    return data

def delete_csv_file(filepath):
    if(os.path.exists(filepath) and os.path.isfile(filepath)):
        os.remove(filepath)
    else:
        raise Exception(f"could not delete file: {filepath}")