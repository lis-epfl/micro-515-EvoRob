import copy
import os
from typing import Tuple, List

import numpy as np

from evorob.utils.filesys import search_file_list


class NSGAII:
    """Non-dominated Sorting Genetic Algorithm II (NSGA-II).
    
    NSGA-II is a multi-objective evolutionary algorithm that uses:
    - Fast non-dominated sorting to rank solutions into Pareto fronts
    - Mutation and crossover operators
    - Fitness-proportional parent selection based on Pareto front rank
    
    The algorithm maintains a population of candidate solutions and evolves them
    over multiple generations to find a diverse set of non-dominated solutions
    approximating the Pareto front of the multi-objective optimization problem.
    
    Attributes:
        n_params (int): Number of optimization parameters per solution.
        n_pop (int): Population size.
        n_parents (int): Number of parents selected for reproduction.
        min (float): Lower bound for parameter values.
        max (float): Upper bound for parameter values.
        current_gen (int): Current generation counter.
        mutation_prob (float): Mutation probability.
        crossover_prob (float): Crossover probability.
        current_population (np.ndarray): Current parent population.

    References:
        Deb, K., et al. (2002). A fast and elitist multiobjective genetic
        algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation.
        
    Example:
        >>> nsga = NSGAII(population_size=100, num_opt_params=10, num_parents=20)
        >>> for generation in range(100):
        ...     population = nsga.ask()
        ...     fitness = evaluate_objectives(population)  # Shape: (100, num_objectives)
        ...     nsga.tell(population, fitness)
    """
    def __init__(
        self,
        population_size: int,
        num_opt_params: int,
        num_parents: int = 16,
        bounds: Tuple[float, float] = (-4, 4),
        mutation_prob: float = 0.3,
        crossover_prob: float = 0.1,
    ) -> None:
        """
        Initializes the NSGA-II algorithm.

        :param population_size: population size
        :param num_opt_params: number of parameters
        :param num_generations: number of generations
        :param num_parents: number of parents
        :param bounds: parameter bounds
        :param mutation_prob: mutation probability
        :param crossover_prob: crossover probability
        """
        # % EA options
        self.n_params = num_opt_params
        self.n_pop = population_size
        self.n_parents = num_parents
        self.min = bounds[0]
        self.max = bounds[1]

        self.current_gen = 0
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob

    def ask(self) -> np.ndarray:
        """Generates a new population of candidate solutions.

        Returns:
            np.ndarray: The new population of candidate solutions.
        """
        if self.current_gen == 0:
            new_population = self.initialise_x0()
        else:
            new_population = self.create_children(self.n_pop)
        new_population = np.clip(new_population, self.min, self.max)
        return new_population

    def tell(self, population: np.ndarray, fitness: np.ndarray) -> None:
        """Updates the algorithm with the evaluated solutions and their fitness values.
        
        Performs non-dominated sorting on the combined population, ranks solutions
        into Pareto fronts, and selects parents for the next generation using
        fitness-proportional selection based on front ranks.
        
        Args:
            solutions (np.ndarray): Population of candidate solutions. Shape: (n_pop, n_params)
            fitness (np.ndarray): Objective values for each solution. 
                                          Shape: (n_pop, n_objectives)
                                          
        Note:
            The algorithm assumes maximization of all objectives. For minimization,
            negate the objective values before calling tell().
            solutions, function_values, self.n_parents
        )
        """
        parents_population, parents_fitness = self.sort_and_select_parents(
            population, fitness, self.n_parents
        )

        self.current_population = parents_population
        self.fitness = parents_fitness

        self.current_gen += 1

    def initialise_x0(self) -> np.ndarray:
        """Initializes the population with random uniform samples.
        
        Returns:
            np.ndarray: Initial population with shape (n_pop, n_params).
        """
        return np.random.uniform(
            low=self.min, high=self.max, size=(self.n_pop, self.n_params)
        )

    def create_children(self, population_size: int) -> np.ndarray:
        """Creates offspring using mutation and crossover.
        
        Args:
            population_size (int): Number of offspring to generate.
            
        Returns:
            np.ndarray: Mutated and clipped offspring population.
        """
        new_offspring = np.empty((population_size, self.n_params))
        for i in range(population_size):
            r0 = i
            while r0 == i:
                r0 = np.floor(np.random.random() * self.n_pop).astype(int)
            r1 = r0
            while r1 == r0 or r1 == i:
                r1 = np.floor(np.random.random() * self.n_pop).astype(int)
            r2 = r1
            while r2 == r1 or r2 == r0 or r2 == i:
                r2 = np.floor(np.random.random() * self.n_pop).astype(int)

            jrand = np.floor(np.random.random() * population_size).astype(int)

            for j in range(self.n_params):
                if np.random.random() <= self.crossover_prob or j == jrand:
                    # Mutation
                    new_offspring[i][j] = copy.deepcopy(
                        self.current_population[r0][j]
                        + self.mutation_prob
                        * (
                            self.current_population[r1][j]
                            - self.current_population[r2][j]
                        )
                    )
                else:
                    new_offspring[i][j] = copy.deepcopy(self.current_population[r0][j])

        mutated_population = np.clip(new_offspring, self.min, self.max)
        return mutated_population

    def sort_and_select_parents(
        self, population: np.ndarray, fitness: np.ndarray, n_parents: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sorts solutions by Pareto dominance and selects parents.
        
        Uses fast non-dominated sorting to rank solutions, then performs
        fitness-proportional selection where solutions in better fronts
        have higher probability of being selected.
        
        Args:
            population (np.ndarray): Candidate solutions.
            fitness (np.ndarray): Objective values.
            n_parents (int): Number of parents to select.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Selected parent solutions and their fitness.
        """
        # TODO!
        draw_ind = ...
        population[draw_ind] = ...
        fitness[draw_ind] = ...

        return population[draw_ind], fitness[draw_ind]

    def dominates(self, individual: np.ndarray, other_individual: np.ndarray) -> bool:
        """Checks if one solution dominates another (for maximization).
        
        Solution A dominates solution B if:
        - A is at least as good as B in all objectives
        - A is strictly better than B in at least one objective
        
        Args:
            individual: Objective values of first solution.
            other_individual: Objective values of second solution.
            
        Returns:
            bool: True if individual dominates other_individual.
        """
        return all(x >= y for x, y in zip(individual, other_individual)) and any(
            x > y for x, y in zip(individual, other_individual)
        )

    def fast_nondominated_sort(self, fitness: np.ndarray) -> Tuple[List[List[int]], List[int]]:
        """Performs fast non-dominated sorting to rank solutions into Pareto fronts.
        
        Implements the fast non-dominated sorting algorithm from Deb et al. (2002).
        Solutions are assigned to fronts based on Pareto dominance:
        - Front 0: Non-dominated solutions
        - Front 1: Solutions dominated only by Front 0
        - Front i: Solutions dominated only by Fronts 0 to i-1
        
        Args:
            fitness (np.ndarray): Objective values for all solutions.
                                  Shape: (population_size, num_objectives)
                                  
        Returns:
            Tuple[List[List[int]], List[int]]:
                - pareto_fronts: List of fronts, each containing solution indices
                - population_rank: Front number for each solution
                
        
        """
        domination_lists: List[List[int]] = [[] for _ in range(len(fitness))]
        domination_counts: List[int] = [0 for _ in range(len(fitness))]
        population_rank: List[int] = [0 for _ in range(len(fitness))]
        pareto_fronts: List[List[int]] = [[]]

        for individual_a in range(len(fitness)):
            for individual_b in range(len(fitness)):
                # does candidate 1 dominate candidate 2?
                if self.dominates(fitness[individual_a], fitness[individual_b]):
                    # append index of dominating solution
                    domination_lists[individual_a].append(individual_b)

                # does candidate 2 dominate candidate 1?
                elif self.dominates(fitness[individual_b], fitness[individual_a]):
                    #
                    domination_counts[individual_a] += 1

            # if solution dominates all
            if domination_counts[individual_a] == 0:
                # placeholder solution rank
                population_rank[individual_a] = 0

                # add solution to first Pareto front
                pareto_fronts[0].append(individual_a)

        # iterates until there are no more items appended in the last front
        i: int = 0
        while pareto_fronts[i]:
            # open next front
            next_front: List[int] = []

            # iterate through all items in previous front
            for individual_a in pareto_fronts[i]:
                # check all other items which are dominated by this item
                for individual_b in domination_lists[individual_a]:
                    # reduce domination count
                    domination_counts[individual_b] -= 1

                    # every now nondominated item are append to next front
                    if domination_counts[individual_b] == 0:
                        # add solution rank
                        population_rank[individual_b] = i + 1
                        next_front.append(individual_b)

            i += 1

            pareto_fronts.append(next_front)

        # removes last empty front
        pareto_fronts.pop()

        return pareto_fronts, population_rank
    
    def save_checkpoint(self):
        curr_gen_path = os.path.join(self.directory_name, str(self.current_gen))
        os.makedirs(curr_gen_path, exist_ok=True)
        np.save(os.path.join(self.directory_name, 'full_f'), np.array(self.full_fitness))
        np.save(os.path.join(self.directory_name, 'full_x'), np.array(self.full_x))
        np.save(os.path.join(curr_gen_path, 'f_best'), np.array(self.f_best_so_far))
        np.save(os.path.join(curr_gen_path, 'x_best'), np.array(self.x_best_so_far))
        np.save(os.path.join(curr_gen_path, 'x'), np.array(self.x))
        np.save(os.path.join(curr_gen_path, 'f'), np.array(self.f))

    def load_checkpoint(self):
        dir_path = search_file_list(self.directory_name, 'f_best.npy')
        assert len(dir_path) > 0;
        "No files are here, check the directory_name!!"

        self.current_gen = int(dir_path[-1].split('/')[-2])
        curr_gen_path = os.path.join(self.directory_name, str(self.current_gen))

        self.full_fitness = np.load(os.path.join(self.directory_name, 'full_f.npy'))
        self.full_x = np.load(os.path.join(self.directory_name, 'full_x.npy'))
        self.f_best_so_far = np.load(os.path.join(curr_gen_path, 'f_best.npy'))
        self.x_best_so_far = np.load(os.path.join(curr_gen_path, 'x_best.npy'))
        self.x = np.load(os.path.join(curr_gen_path, 'x.npy'))
        self.f = np.load(os.path.join(curr_gen_path, 'f.npy'))