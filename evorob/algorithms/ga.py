from typing import Dict

import numpy as np

from evorob.algorithms.base_ea import EA

GA_opts = {
    "min": -4,
    "max": 4,
    "num_generations": 100,
    "tournament_size": 16,
    "mutation_prob": 0.3,
    "crossover_prob": 0.1,
    "log_interval": 5,
}

class GA(EA):

    def __init__(self, n_pop, n_params, opts: Dict = GA_opts, output_dir: str = "./results"):
        self.n_params = n_params
        self.n_pop = n_pop
        self.n_gen = opts["num_generations"]
        self.tournament_size = opts["tournament_size"]
        self.min = opts["min"]
        self.max = opts["max"]

        self.current_gen = 0
        self.mutation_prob = opts["mutation_prob"]
        self.crossover_prob = opts["crossover_prob"]

        self.log_interval = opts.get("log_interval", 5)

        # % bookkeeping
        self.directory_name = output_dir
        self.full_x = []
        self.full_f = []
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = None
        self.f = None

    def ask(self):
        """Generates a new population based on the current one."""
        if self.current_gen == 0:
            new_population = self.initialise_x0()
        else:
            parents = self.select_parents()
            parents1 = parents[::2]
            parents2 = parents[1::2]
            offspring1, offspring2 = self.crossover_efficient(parents1, parents2)
            new_population = self.mutate_efficient(np.concatenate((offspring1, offspring2)))
        new_population = np.clip(new_population, self.min, self.max)
        return new_population

    def ask_slow(self):
        if self.current_gen == 0:
            new_population = self.initialise_x0()
        else:
            new_population = []
            # Generate new population through crossover and mutation
            for _ in range(self.n_pop // 2):  # Produce pairs of children
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                offspring1, offspring2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(offspring1))
                new_population.append(self.mutate(offspring2))
        new_population = np.clip(new_population, self.min, self.max)
        return new_population

    def tell(self, solutions, function_values, save_checkpoint=False):
        """Updates the current population given the individuals and fitnesses."""
        #% Some bookkeeping
        self.full_f.append(function_values)
        self.full_x.append(solutions)
        self.f = function_values
        self.x = solutions

        if np.nanmax(function_values) > self.f_best_so_far:
            best_index = np.nanargmax(function_values)
            self.f_best_so_far = function_values[best_index]
            self.x_best_so_far = solutions[best_index]

        if self.current_gen % self.log_interval == 0:
            n_nan = np.sum(np.isnan(function_values))
            print(
                f"Generation {self.current_gen} | best: {self.f_best_so_far:.2f} | #nan {n_nan}\n"
                f"Mean fitness:\t{np.nanmean(function_values):.2f} +- {np.nanstd(function_values):.2f}"
            )

        if save_checkpoint:
            self.save_checkpoint()
        self.current_gen += 1

    def initialise_x0(self):
        """Initialises the first population."""
        population = np.random.uniform(self.min, self.max, (self.n_pop, self.n_params))
        return population

    def select_parent(self):
        """Tournament selection: choose a random individual and return it."""
        tournament = np.random.choice(self.n_pop, self.tournament_size, replace=False)
        tournament_ind = np.nanargmax(self.f[tournament]) # type: ignore
        return self.x[tournament[tournament_ind]] # type: ignore

    def select_parents(self):
        tournaments = np.random.choice(self.n_pop, (self.n_pop, self.tournament_size))
        tournament_inds = np.nanargmax(self.f[tournaments], axis=1)  # type: ignore
        return self.x[tournaments[np.arange(self.n_pop), tournament_inds]]  # type: ignore

    def crossover(self, parent1, parent2):
        """Single-point crossover."""
        if np.random.random() < self.crossover_prob:
            idx = np.random.randint(1, self.n_params)
            offspring1 = np.concatenate([parent1[:idx], parent2[idx:]])
            offspring2 = np.concatenate([parent2[:idx], parent1[idx:]])
            return offspring1, offspring2
        else:
            return parent1, parent2

    def crossover_efficient(self, parents1, parents2):
        n_parents = parents1.shape[0]
        crossover_parents = np.random.rand(n_parents) < self.crossover_prob
        idxs = np.random.randint(1, self.n_params, size=n_parents)
        idxs[~crossover_parents] = self.n_params  # No crossover for these
        crossover_mask = np.arange(self.n_params) < idxs[:, None]
        offspring1 = np.where(crossover_mask, parents1, parents2)
        offspring2 = np.where(crossover_mask, parents2, parents1)
        return offspring1, offspring2

    def mutate(self, individual):
        """Mutate an individual by flipping bits with a given mutation rate."""
        for i in range(self.n_params):
            if np.random.random() < self.mutation_prob:
                individual[i] = individual[i] + np.random.uniform(-1, 1)
        return individual

    def mutate_efficient(self, individuals):
        mutation_mask = np.random.rand(*individuals.shape) < self.mutation_prob
        individuals += mutation_mask * np.random.uniform(-1, 1, individuals.shape)
        return individuals
