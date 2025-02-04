import numpy as np

from simulator.pigeon_simulator_2 import PigeonSimulator
from bird_models.pigeon import Pigeon
from area_models.landmark import Landmark

from genetic.genetic_algorithm_ae import DifferentialEvolution
from genetic.metrics import Metrics

import loggers.logger as logger
import geometry.normalisation as normal

n_agents = 7
n_steps = 10000
env_size = (50, 50)
start_position = (25, 25)
graph_freq = 10
visualize = True
visualize_vision_fields = 1
follow = True
single_speed = True
bird_type = Pigeon()
start_position = (0, 0)

social_weight = 1
path_weight = 0

num_iters = 25
num_gens = 30
num_ind = 10
use_norm = True
pop_size = 30
bounds = [0,1]
metric = Metrics.ORDER

len_weights = 4

postfix = f"_test_ae_tmax={n_steps}_n={n_agents}_bt={bird_type.name}_domain={env_size}_m={metric.value}"
save_path_best = f"best{postfix}.csv"
save_path_best_normalised = f"best{postfix}_normalised.csv"
save_path_general = f"all{postfix}"
save_path_plot = f"plot{postfix}"

logger.initialise_log_file_with_headers(logger.create_headers(len_weights=len_weights, is_best=True), save_path=save_path_best)
logger.initialise_log_file_with_headers(logger.create_headers(len_weights=len_weights, is_best=True), save_path=save_path_best_normalised)

for i in range(num_iters):

    evo = DifferentialEvolution(tmax=n_steps,
                            num_agents=n_agents,
                            bird_type=bird_type,
                            domain_size=env_size,
                            num_generations=num_gens,
                            num_iterations_per_individual=num_ind,
                            use_norm=use_norm,
                            population_size=pop_size,
                            bounds=bounds,
                            metric=metric)

    best = evo.run(save_path_log=save_path_general, save_path_plots=save_path_plot)
    print(f"BEST overall: {best}")


    logger.log_results_to_csv([{'iter': i, 'individual': np.array(best[0]), 'fitness': best[1]}], prepare=True, save_path=save_path_best)
    logger.log_results_to_csv([{'iter': i, 'individual': normal.normalise(np.array(best[0])), 'fitness': best[1]}], prepare=True, save_path=save_path_best_normalised)

