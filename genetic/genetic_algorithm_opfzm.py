import numpy as np
import csv
import matplotlib.pyplot as plt

from neural_network.activation_layer import ActivationLayer
from neural_network.fully_connected_layer import FullyConnectedLayer
from neural_network.neural_network import NeuralNetwork
import neural_network.activation_functions as snn

from genetic.metrics import Metrics
import genetic.metrics_functions as metfunc
import general.normalisation as normal
import loggers.logger_evolution as logger_evolution

from simulator.orientation_perception_free_zone_model import OrientationPerceptionFreeZoneModelSimulator
from animal_models.pigeon import Pigeon

"""
Implements Differential Evolution for the Orientation Perception Free Zone Model implementation evolving the head movement via a neural network.
"""

class DifferentialEvolution:
    def __init__(self, tmax, num_agents=None, animal_type=Pigeon(), domain_size=(50,50), weight_options=[],
                 num_generations=1000, num_iterations_per_individual=1, 
                 use_norm=True, population_size=100, bounds=[0, 1], update_to_zero_bounds=[0,0], 
                 mutation_scale_factor=1, crossover_rate=0.5, early_stopping_after_gens=None, 
                 metric=Metrics.COHESION):
        """
        Models the DE approach.

        Params:
            - tmax (int): the number of timesteps for each simulation
            - num_agents (int): how many agents are within the domain
            - animal_type (Animal) [optional, default=Pigeon]: the type of animal
            - domain_size (tuple of floats) [optional, default=(50,50)]: the dimensions of the domain
            - weight_options (list of WeightOption) [optional, default=[]]: which weight options are active, i.e. which inputs will be considered by the neural network
            - num_generations (int) [optional, default=1000]: how many generations are generated and validated
            - num_iterations_per_individual (int) [optional, default=10]: how many times the simulation is run for every individual
            - use_norm (boolean) [optional, default=True]: whether the weights should be normalised
            - population_size (int) [optional, default=100]: how many individuals are generated per generation
            - bounds (list of 2 ints) [optional, default=[-1, 1]]: the bounds for the c_value generation
            - update_to_zero_bounds (list of 2 floats) [optional, default=[0,0]]: within the given range, all weight values will be set to zero
            - mutation_scale_factor (int) [optional, default=1]: the mutation scale factor
            - crossover_rate (float) [optional, default=0.5]: the crossover rate
            - early_stopping_after_gens (int) [optional, default=None]: stops the evolution after x number of generations without an improvement
            - metric (Metrics) [optional, default=COHESION]: how the fitness should be evaluated
        """

        self.num_generations = num_generations
        self.num_iterations_per_individual = num_iterations_per_individual

        self.tmax = tmax
        self.num_agents = num_agents
        self.animal_type = animal_type
        self.domain_size = domain_size
        self.weight_options = weight_options

        self.use_norm = use_norm
        self.population_size = population_size
        self.bounds = bounds
        self.update_to_zero_bounds = update_to_zero_bounds
        self.mutation_scale_factor = mutation_scale_factor
        self.crossover_rate = crossover_rate
        self.early_stopping_after_gens = early_stopping_after_gens

        self.metric = metric

        self.weight_size = len(self.weight_options)
        self.output_size = 1

    def create_initial_population(self):
        """
        Creates the initial random population.
        """
        return np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=((self.population_size, self.weight_size)))

    def create_neural_network(self, weights):
        """
        Creates the neural network with fixed weights to be used to determine the head movement updates.
        """
        nn = NeuralNetwork()
        fully_connected_layer = FullyConnectedLayer(input_size=self.weight_size, output_size=self.output_size)
        fully_connected_layer.set_weights(weights=weights)
        nn.add(fully_connected_layer)
        nn.add(ActivationLayer(activation=snn.tanh, activation_prime=snn.tanh_prime))
        return nn
    
    def evaluate_result(self, result):
        """
        Evaluates the results according to the predefined metric. 
        Returns the average of the evaluations across all timesteps.
        """
        timestep_results = []
        for t in range(self.tmax):
            match self.metric:
                case Metrics.ORDER:
                    timestep_results.append(1-metfunc.compute_global_order(result[t])) # for minimisation
                case Metrics.COHESION:
                    timestep_results.append(metfunc.compute_cohesion(result[t]))
                case Metrics.COHESION_AND_ORDER:
                    order = 1-metfunc.compute_global_order(result[t]) # for minimisation
                    cohesion = metfunc.compute_cohesion(result[t])
                    timestep_results.append(order*cohesion)
        return np.average(timestep_results)

    def fitness_function(self, weights):
        """
        Evaluates the success of the individual by running the simulations and evaluating the results 
        against the predefined metric. Returns the average fitness value across all runs.
        """
        results = []
        weights = self.update_weights(weights)
        model = self.create_neural_network(weights=weights)
        for i in range(self.num_iterations_per_individual):
            simulator = OrientationPerceptionFreeZoneModelSimulator(
                                        num_agents=self.num_agents,
                                        animal_type=self.animal_type,
                                        domain_size=self.domain_size,
                                        start_position=(0,0),
                                        social_weight=1,
                                        environment_weight=0,
                                        weight_options=self.weight_options,
                                        model=model,
                                        visualize=False)
            result = simulator.run(tmax=self.tmax)
            results.append(self.evaluate_result(result))
        fitness = np.average(results)
        return fitness
    
    def mutation(self, x, F):
        """
        Mutates an individual
        """
        return x[0] + F * (x[1] - x[2])
    
    def check_bounds(self, mutated, bounds):
        """
        Makes sure that the mutated individual only has values within the bound.
        """
        mutated_bound = np.clip(mutated, bounds[0], bounds[1])
        return mutated_bound
    
    def crossover(self, mutated, target, cr):
        """
        Performs the crossover operation.
        """
        # generate a uniform random value for every dimension
        p = np.random.rand(self.weight_size)
        # generate trial vector by binomial crossover
        trial = [mutated[i] if p[i] < cr else target[i] for i in range(self.weight_size)]
        return np.array(trial)
    
    def update_weights(self, weights):
        """
        Updates the weights for the neural network.
        """
        weights = np.where(((weights >= self.update_to_zero_bounds[0]) & (weights <= self.update_to_zero_bounds[1])), 0, weights)
        if self.use_norm == True:
            weights = normal.normalise(weights, norm='l1')
        return weights
    
    def plot_fitnesses(self, fitnesses, save_path_plots=None):
        """
        Plots the fitness values
        """
        plt.plot(fitnesses)
        if save_path_plots:
            plt.savefig(f"{save_path_plots}.svg")
            plt.savefig(f"{save_path_plots}.jpeg")
        else:
            plt.show()         

    def run(self, save_path_plots=None, save_path_log=None, log_depth='all'):
        """
        Performs the DE evolution for the weights.
        """
        with open(f"{save_path_log}.csv", 'a', newline='') as log:
            w = csv.writer(log)
            headers = logger_evolution.create_headers(self.weight_options)
            w.writerow(headers)
            log.flush()
            population  = self.create_initial_population()
            fitnesses = [self.fitness_function(individual) for individual in population]
            best_individual = population[np.argmin(fitnesses)]
            best_fitness = min(fitnesses)
            prev_fitness = best_fitness
            best_fitnesses_for_generations = [best_fitness]
            # saving the fitnesses
            if log_depth == 'all':
                log_dict_list = logger_evolution.create_dicts_for_logging(-1, population, fitnesses)
            else:
                log_dict_list = logger_evolution.create_dicts_for_logging(-1, [best_individual], [best_fitness])
            for dict in log_dict_list:
                w.writerow(dict.values())
            log.flush()
            last_improvement_at_gen = 0
            for iter in range(self.num_generations):
                print(f"gen {iter+1}/{self.num_generations}")
                for ind in range(self.population_size):
                    candidates = [candidate for candidate in range(self.population_size) if candidate != ind]
                    a, b, c = population[np.random.choice(candidates, 3, replace=False)]
                    mutated = self.mutation([a, b, c], self.mutation_scale_factor)
                    mutated = self.check_bounds(mutated, self.bounds)
                    trial = self.crossover(mutated, population[ind], self.crossover_rate)
                    target_existing = fitnesses[ind]
                    target_trial = self.fitness_function(trial)
                    if target_trial < target_existing:
                        population[ind] = trial
                        fitnesses[ind] = target_trial
                best_fitness = min(fitnesses)
                if best_fitness < prev_fitness:
                    best_individual = population[np.argmin(fitnesses)]
                    prev_fitness = best_fitness
                    last_improvement_at_gen = iter
                    print('Iteration: %d f([%s]) = %.5f' % (iter, np.around(best_individual, decimals=5), best_fitness))
                # saving the fitnesses
                if log_depth == 'all':
                    log_dict_list = logger_evolution.create_dicts_for_logging(iter, population, fitnesses)
                else:
                    log_dict_list = logger_evolution.create_dicts_for_logging(iter, [best_individual], [best_fitness])
                for dict in log_dict_list:
                    w.writerow(dict.values())
                log.flush()
                best_fitnesses_for_generations.append(best_fitness)
                if self.early_stopping_after_gens != None and iter-last_improvement_at_gen > self.early_stopping_after_gens:
                    print(f"Early stopping at iteration {iter} after {self.early_stopping_after_gens} generations without improvement")
                    break

            self.plot_fitnesses(best_fitnesses_for_generations, save_path_plots)
            return [best_individual, best_fitness]
