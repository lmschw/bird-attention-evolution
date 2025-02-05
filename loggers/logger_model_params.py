import csv

def log_model_params(save_path, model_params_dict):
    with open(f"{save_path}.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(model_params_dict.keys())
        w.writerow(model_params_dict.values())