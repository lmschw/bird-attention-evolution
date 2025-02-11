import csv

"""
Contains methods to log model params
"""

def log_model_params(save_path, model_params_dict):
    """
    Logs the model params as a single row with headers.
    """
    with open(f"{save_path}.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(model_params_dict.keys())
        w.writerow(model_params_dict.values())