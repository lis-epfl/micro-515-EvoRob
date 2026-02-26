from typing import Dict

import numpy as np
from evorob.algorithms.base_ea import EA

ES_opts = {
    "min": -4,
    "max": 4,
    "num_parents": 16,
    "num_generations": 100,
    "mutation_sigma": 0.3,
    "min_sigma": 0.1,
    "sigma_decay_rate": 0.95
}


class ES(EA):

    def __init__(self, n_pop, n_params, opts: Dict = ES_opts, log_every: int = 5, output_dir: str = "./results/ES"):
        """
        Evolutionary Strategy

        :param n_pop: population size
        :param n_params: number of parameters
        :param opts: algorithm options
        :log_every: log every n generations
        :param output_dir: output directory Default = "./results/ES"
        """
        # % EA options
        self.n_params = n_params
        self.n_pop = n_pop
        self.n_gen = opts["num_generations"]
        self.n_parents = opts["num_parents"]
        self.min = opts["min"]
        self.max = opts["max"]

        self.current_gen = 0
        self.current_mean = [(self.min + self.max) / 2]*self.n_params
        self.current_sigma = opts["mutation_sigma"]
        self.min_sigma = opts["min_sigma"]
        self.sigma_decay_rate = opts["sigma_decay_rate"]

        # % bookkeeping
        self.log_every = log_every
        self.directory_name = output_dir
        self.full_x = []
        self.full_f = []
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = None
        self.f = None

    def ask(self):
        if self.current_gen==0:
            new_population = self.initialise_x0()
        else:
            new_population = self.generate_mutated_offspring(self.n_pop)
        new_population = np.clip(new_population, self.min, self.max)
        return new_population

    def tell(self, solutions, function_values, save_checkpoint=False):
        parents_population, parents_fitness = self.sort_and_select_parents(
            solutions, function_values, self.n_parents
        )
        self.update_population_mean(parents_population, parents_fitness)
        self.update_sigma()

        #% Some bookkeeping
        self.full_f.append(function_values)
        self.full_x.append(solutions)
        self.x = parents_population
        self.f = parents_fitness

        best_index = np.argmax(function_values)
        if function_values[best_index] > self.f_best_so_far:
            self.f_best_so_far = function_values[best_index]
            self.x_best_so_far = solutions[best_index]

        if self.current_gen % self.log_every == 0:
            print(f"Best in generation {self.current_gen: 3d}: {function_values[best_index]:.2f}\n"
                  f"Best fitness so far   : {self.f_best_so_far:.2f}\n"
                  f"Mean pop fitness      : {np.mean(self.f):.2f} +- {np.std(self.f):.2f}\n"
                  f"Sigma: {self.current_sigma:.2f} \n"
            )
        if save_checkpoint:
            self.save_checkpoint()
        self.current_gen += 1

    def initialise_x0(self):
        """Initialises the first population."""
        # TODO: generate the initial population mean vector (current_mean)
        mean_vector = ...
        return mean_vector

    def update_sigma(self):
        """Update the perturbation strength (sigma)."""
        # TODO: implement a decay of the sigma value over generations, ensuring it does not go below min_sigma
        self.current_sigma = ...

    def sort_and_select_parents(self, population, fitness, num_parents):
        """Sorts the population based on fitness and selects the top individuals as parents."""
        # TODO: sort the population and fitness based on fitness values, and select the top num_parents individuals as parents
        parent_population = ...
        parent_fitness = ...

        return parent_population, parent_fitness

    def update_population_mean(self, parent_population, parent_fitness):
        # TODO: compute the new population mean as a weighted average of the parent population, where the weights are based on the parent fitness
        # (you can use rank or raw fitness values)
        # Normalise parent fitness scores
        normed_parents_fitness = ...

        # Compute population weighted to the normed fitness scores
        weighted_parents_population = ...

        # Calculate the sum of weighted parents population
        updated_mean_vector = ...

        return updated_mean_vector

    def generate_mutated_offspring(self, population_size):
        """Generates a new population by adding Gaussian noise to the current mean."""
        # TODO: generate a new population by adding Gaussian noise to the current mean, where the noise is scaled by the current sigma value
        perturbation = ...
        mutated_population = ...

        return mutated_population

