import csv, os
import numpy as np

"""
Service containing methods to do with logging.
"""

def create_headers(len_weights, is_best=False):
    """
    Creates the headers for the csv file.

    Params:
        - len_weights (int): how many values are in the weights
        - is_best (boolean): if the log is meant for the best result or for individual results

    Returns:
        A list containing all the headers.
    """
    if is_best:
        headers = ['iter']
    else:
        headers = ['iter', 'ind']
    
    individual_headers = [f"individual_{i}" for i in range(len_weights)]
    headers.extend(individual_headers)
    headers.append('fitness')
    return headers

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

def create_dicts_for_logging(iter, individuals, fitnesses):
    """
    Creates the dictionaries from the data.

    Params:
        - iter (int): the generation/iteration
        - individuals (list of np.array): the individuals present in the current population
        - fitnesses (list of floats): the fitness value of every individual in the population

    Returns:
        A list of dictionaries containing the information for all the individuals.
    """
    dict_list = []
    for ind in range(len(individuals)):
        dict_list.append(create_dict(iter, ind, individuals[ind], fitnesses[ind]))
    return prepare_individuals_for_csv_logging(dict_list)

def create_dict(iter, ind, weights, fitness):
    """
    Creates a dictionary for an individual.

    Params:
        - iter (int): the generation
        - ind (int): the index of the individual within the population
        - c-values (np.array): the weights for all the orientations
        - fitness (float): the fitness of the individual

    Returns:
        A dictionary.
    """
    return {'iter': iter, 'ind': ind, 'individual': weights, 'fitness': fitness}

def prepare_individuals_for_csv_logging(dict_list):
    """
    Transforms all np.arrays into separate entries.

    Params:
        - dict_list (list of dictionaries): A list containing a dictionary for every data point (individual)

    Returns:
        A list of dictionaries containing no numpy arrays but instead spread to more items.
    """
    new_dict_list = []
    for dict in dict_list:
        new_dict = {}
        for k, v in dict.items():
            if isinstance(v, np.ndarray):
                for i in range(len(v)):
                    new_dict[f"individual_{i}"] = v[i]
            else:
                new_dict[k] = v
        new_dict_list.append(new_dict)
    return new_dict_list

def log_results_to_csv(dict_list, save_path, prepare=False):
    """
    Logs the results to a csv file.

    Params:
        - dict_list (list of dictionaries): A list containing a dictionary for every data point (individual)
        - save_path (string): the path of the file where the headers should be inserted
        - prepare (boolean) [optional, default=False]: whether or not the dictionaries need to be prepared, i.e. if they still contain numpy arrays

    Returns:
        Nothing.
    """
    if prepare == True:
        dict_list = prepare_individuals_for_csv_logging(dict_list)
    with open(save_path, 'a', newline='') as f:
        w = csv.writer(f)
        for dict in dict_list:
            w.writerow(dict.values())

def delete_csv_file(filepath):
    if(os.path.exists(filepath) and os.path.isfile(filepath)):
        os.remove(filepath)
    else:
        raise Exception(f"could not delete file: {filepath}")