import numpy as np

from simulator.head_movement.enum_weight_options import WeightOptions
from animal_models.pigeon import Pigeon

from genetic.genetic_algorithm_ae import DifferentialEvolution
from evaluators.metrics import Metrics

import loggers.logger_evolution as logger_evolution
import loggers.logger_model_params as logger_params
import general.normalisation as normal

weight_options = [WeightOptions.CLOSEST_DISTANCES,
                  WeightOptions.CLOSEST_BEARINGS,
                  WeightOptions.AVG_DISTANCES,
                  WeightOptions.AVG_BEARINGS,
                  WeightOptions.NUM_VISIBLE_AGENTS,
                  WeightOptions.PREVIOUS_HEAD_ANGLES,
                  WeightOptions.AVG_PERCEPTION_STRENGTHS]
len_weights = len(weight_options)

n_agents = 7
n_steps = 1000
domain_size = (500, 500)
start_position = (250, 250)
graph_freq = 10
visualize = False
visualize_vision_fields = 1
follow = True
single_speed = True
animal_type = Pigeon()
start_position = (0, 0)

social_weight = 1
path_weight = 0

num_iters = 25
num_gens = 30
num_ind = 1
use_norm = True
pop_size = 10
bounds = [0,1]
metric = Metrics.COHESION

model_params = {'num_agents': n_agents,
                'tmax': n_steps,
                'domain_size': domain_size,
                'start_position': start_position,
                'social_weight': social_weight,
                'weight_options': [option.value for option in weight_options],
                'metric': metric.value}

postfix = f"_test_ae_tmax={n_steps}_n={n_agents}_bt={animal_type.name}_domain={domain_size}_m={metric.value}"
save_path_best = f"best{postfix}.csv"
save_path_best_normalised = f"best{postfix}_normalised.csv"
save_path_general = f"all{postfix}"
save_path_plot = f"plot{postfix}"
save_path_model_params = f"model_params{postfix}"

logger_params.log_model_params(model_params_dict=model_params, save_path=save_path_model_params)

logger_evolution.initialise_log_file_with_headers(logger_evolution.create_headers(weight_options=weight_options, is_best=True), save_path=save_path_best)
logger_evolution.initialise_log_file_with_headers(logger_evolution.create_headers(weight_options=weight_options, is_best=True), save_path=save_path_best_normalised)

for i in range(num_iters):
    print(f"{i}/{num_iters}")
    evo = DifferentialEvolution(tmax=n_steps,
                            num_agents=n_agents,
                            animal_type=animal_type,
                            domain_size=domain_size,
                            weight_options=weight_options,
                            num_generations=num_gens,
                            num_iterations_per_individual=num_ind,
                            use_norm=use_norm,
                            population_size=pop_size,
                            bounds=bounds,
                            metric=metric)

    best = evo.run(save_path_log=save_path_general, save_path_plots=save_path_plot)
    print(f"BEST overall: {best}")


    logger_evolution.log_results_to_csv([{'iter': i, 'individual': np.array(best[0]), 'fitness': best[1]}], prepare=True, save_path=save_path_best)
    logger_evolution.log_results_to_csv([{'iter': i, 'individual': normal.normalise(np.array(best[0])), 'fitness': best[1]}], prepare=True, save_path=save_path_best_normalised)

