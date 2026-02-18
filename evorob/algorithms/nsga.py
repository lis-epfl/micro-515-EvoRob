from typing import Tuple, List

import numpy as np

from evorob.algorithms.base_ea import EA


class NSGAII(EA):
    """Non-dominated Sorting Genetic Algorithm II (NSGA-II).
    
    NSGA-II is a multi-objective evolutionary algorithm that uses:
    - Fast non-dominated sorting to rank solutions into Pareto fronts
    - Crowding distance to maintain diversity within fronts
    - Tournament selection based on rank and crowding distance
    - Mutation and crossover operators
    
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
        >>> nsga = NSGAII(population_size=100, n_opt_params=10, n_parents=20)
        >>> for generation in range(100):
        ...     population = nsga.ask()
        ...     fitness = evaluate_objectives(population)  # Shape: (100, n_objectives)
        ...     nsga.tell(population, fitness)
    """
    def __init__(
            self,
            population_size: int,
            n_opt_params: int,
            n_parents: int = 16,
            num_generations: int = 100,
            bounds: Tuple[float, float] = (-4, 4),
            mutation_prob: float = 0.3,
            crossover_prob: float = 0.1,
            output_dir: str = "./results/NSGA",
    ) -> None:
        """
        Initializes the NSGA-II algorithm.

        :param population_size: population size
        :param n_opt_params: number of parameters
        :param n_parents: number of parents
        :param num_generations: number of generations
        :param bounds: parameter bounds
        :param mutation_prob: mutation probability
        :param crossover_prob: crossover probability
        :param output_dir: output directory for checkpoints
        """
        # % EA options
        self.n_params = n_opt_params
        self.n_pop = population_size
        self.n_parents = n_parents
        self.min = bounds[0]
        self.max = bounds[1]
        self.n_gen = num_generations
        self.current_gen = 0
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob

        # Bookkeeping for checkpointing (used by base EA)
        self.directory_name = output_dir
        self.full_x = []
        self.full_f = []
        self.x_best_so_far = None
        self.f_best_so_far = None
        self.x = None
        self.f = None

        # Initialize current_population for first generation
        self.current_population = None
        self.fitness = None

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

    def tell(self, population: np.ndarray, fitness: np.ndarray, save_checkpoint=False) -> None:
        """Updates the algorithm with the evaluated solutions and their fitness values.
        
        Implements NSGA-II elitism by combining the current parent population with 
        the new offspring population, then selecting the best n_pop individuals 
        using non-dominated sorting and crowding distance.
        
        Args:
            population (np.ndarray): Population of candidate solutions. Shape: (n_pop, n_params)
            fitness (np.ndarray): Objective values for each solution. 
                                  Shape: (n_pop, n_objectives)
            save_checkpoint (bool): Whether to save checkpoint after update
                                          
        Note:
            The algorithm assumes maximization of all objectives. For minimization,
            negate the objective values before calling tell().
        """
        # NSGA-II elitism: combine parent and offspring populations
        # For the first generation, use population directly
        # For subsequent generations, combine current parents with new offspring
        if self.current_population is None:
            combined_population = population
            combined_fitness = fitness
        else:
            combined_population = np.vstack([self.current_population, population])
            combined_fitness = np.vstack([self.fitness, fitness])

        # Select best n_pop individuals from combined population
        parents_population, parents_fitness = self.sort_and_select_parents(
            combined_population, combined_fitness, self.n_pop
        )

        self.current_population = parents_population
        self.fitness = parents_fitness

        #% Some bookkeeping
        self.full_f.append(fitness)
        self.full_x.append(population)
        self.f = fitness
        self.x = population

        fitness_sums = fitness.sum(axis=1)
        best_in_current_gen_idx = np.argmax(fitness_sums)

        current_best_fitness = fitness[best_in_current_gen_idx]
        current_best_x = population[best_in_current_gen_idx]

        if self.current_gen == 0:
            self.f_best_so_far = current_best_fitness
            self.x_best_so_far = current_best_x
        else:
            if np.all(current_best_fitness >= self.f_best_so_far):
                if np.any(current_best_fitness > self.f_best_so_far):
                    self.f_best_so_far = current_best_fitness
                    self.x_best_so_far = current_best_x

        if self.current_gen % 5 == 0:
            print(f"Generation {self.current_gen}:\t{self.f_best_so_far}")
            print(f"Mean fitness:\t{self.f.mean():.2f} +- {self.f.std():.2f}")
            means = np.mean(fitness, axis=0)
            stds = np.std(fitness, axis=0)
            print(f"Mean fitness per obj: {[f'{m:.2f} +-{s:.2f}' for m, s in zip(means, stds)]}")

        if save_checkpoint:
            self.save_checkpoint()

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
        """Creates offspring using tournament selection, mutation and crossover.
        
        Uses tournament selection based on Pareto rank and crowding distance
        to select parents, then applies differential evolution mutation.
        
        Args:
            population_size (int): Number of offspring to generate.
            
        Returns:
            np.ndarray: Mutated and clipped offspring population.
        """
        new_offspring = np.empty((population_size, self.n_params))

        # Compute ranks and crowding distances for tournament selection
        fronts, ranks = self.fast_nondominated_sort(self.fitness)
        crowding = np.zeros(len(self.fitness))
        for front in fronts:
            dist = self.compute_crowding_distance(self.fitness, front)
            for i, idx in enumerate(front):
                crowding[idx] = dist[i]

        for i in range(population_size):
            # Select parent using tournament selection
            parent_idx = self.tournament_selection(ranks, crowding, tournament_size=2)

            # Select 3 different individuals for differential evolution
            r0 = parent_idx
            while r0 == parent_idx:
                r0 = np.random.randint(0, population_size)
            r1 = r0
            while r1 == r0 or r1 == parent_idx:
                r1 = np.random.randint(0, population_size)
            r2 = r1
            while r2 == r1 or r2 == r0 or r2 == parent_idx:
                r2 = np.random.randint(0, population_size)

            jrand = np.random.randint(0, self.n_params)
            for j in range(self.n_params):
                if np.random.random() <= self.crossover_prob or j == jrand:
                    new_offspring[i][j] = (
                            self.current_population[parent_idx][j]
                            + self.mutation_prob
                            * (self.current_population[r1][j] - self.current_population[r2][j])
                    )
                else:
                    new_offspring[i][j] = self.current_population[parent_idx][j]
        mutated_population = np.clip(new_offspring, self.min, self.max)
        return mutated_population

    def sort_and_select_parents(
            self, population: np.ndarray, fitness: np.ndarray, n_parents: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sorts solutions by Pareto dominance and selects parents using crowding distance.
        
        Uses fast non-dominated sorting to rank solutions, computes crowding
        distance for diversity, then selects best individuals front-by-front.
        If a front doesn't fit entirely, uses crowding distance to select
        the most diverse individuals.
        
        Args:
            population (np.ndarray): Candidate solutions.
            fitness (np.ndarray): Objective values.
            n_parents (int): Number of parents to select.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Selected parent solutions and their fitness.
        """
        fronts, population_rank = self.fast_nondominated_sort(fitness)

        # Compute crowding distance for all fronts
        crowding_distances = np.zeros(len(population))
        for front in fronts:
            if len(front) > 0:
                distances = self.compute_crowding_distance(fitness, front)
                for idx, individual in enumerate(front):
                    crowding_distances[individual] = distances[idx]
        
        # Select individuals front by front, using crowding distance for tie-breaking
        selected_indices = []
        for front in fronts:
            if len(selected_indices) + len(front) <= n_parents:
                # Add entire front
                selected_indices.extend(front)
            else:
                # Front doesn't fit entirely - select best by crowding distance
                remaining = n_parents - len(selected_indices)
                if remaining > 0:
                    # Sort front by crowding distance (descending)
                    front_crowding = [(idx, crowding_distances[idx]) for idx in front]
                    front_crowding.sort(key=lambda x: x[1], reverse=True)
                    selected_indices.extend([idx for idx, _ in front_crowding[:remaining]])
                break

        return population[selected_indices], fitness[selected_indices]

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
        # TODO: Implement Pareto dominance check
        # Use all() and any() to check the two conditions for dominance
        
        raise NotImplementedError(
            "TODO: Implement dominance check.\n"
            "Return True if 'individual' dominates 'other_individual'.\n"
            "See Exercise 2a in challenge2.md for guidance."
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
                                  Shape: (population_size, n_objectives)
                                  
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
                # does individual_a dominate individual_b?
                if self.dominates(fitness[individual_a], fitness[individual_b]):
                    # TODO: Track that individual_a dominates individual_b
                    pass  # Replace with your code

                # does individual_b dominate individual_a?
                elif self.dominates(fitness[individual_b], fitness[individual_a]):
                    # TODO: Track that individual_a is dominated by individual_b
                    pass  # Replace with your code

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
                    # TODO: Update domination count and check if individual_b
                    # should be added to the next front
                    pass  # Replace with your code

            i += 1

            pareto_fronts.append(next_front)

        # removes last empty front
        pareto_fronts.pop()

        return pareto_fronts, population_rank

    def compute_crowding_distance(self, fitness: np.ndarray, front: List[int]) -> np.ndarray:
        """Computes crowding distance for solutions in a given front.
        
        Crowding distance estimates the density of solutions surrounding a particular
        solution. Boundary solutions (extremes in any objective) receive infinite
        distance to preserve diversity. Interior solutions receive distance based on
        the average side length of the cuboid formed by their nearest neighbors.
        
        Args:
            fitness (np.ndarray): Objective values for all solutions.
                                  Shape: (population_size, n_objectives)
            front (List[int]): Indices of solutions in the current front.
            
        Returns:
            np.ndarray: Crowding distance for each solution in the front.
                       Shape: (len(front),)
        """
        n_solutions = len(front)
        n_objectives = fitness.shape[1]

        # Initialize distances to zero
        distance = np.zeros(n_solutions)

        # TODO: For each objective:
        # 1. Sort the front by that objective
        # 2. Assign infinite distance to boundary solutions
        # 3. Compute normalized distance for interior solutions
        
        raise NotImplementedError(
            "TODO: Implement crowding distance calculation.\n"
            "See Exercise 2c in challenge2.md for guidance."
        )

    def crowding_operator(self, individual_idx: int, other_individual_idx: int,
                          population_rank: List[int], crowding_distances: np.ndarray) -> int:
        """Compares two individuals based on rank and crowding distance.
        
        The crowding operator defines a partial order on solutions:
        1. If ranks differ, prefer solution with better (lower) rank
        2. If ranks are equal, prefer solution with larger crowding distance
           (to maintain diversity)
        
        Args:
            individual_idx (int): Index of first individual.
            other_individual_idx (int): Index of second individual.
            population_rank (List[int]): Front rank for each solution.
            crowding_distances (np.ndarray): Crowding distance for each solution.
            
        Returns:
            int: Index of the preferred individual.
        """
        # TODO: Compare two individuals
        # 1. Prefer lower rank (better Pareto front)
        # 2. If same rank, prefer larger crowding distance
        
        raise NotImplementedError(
            "TODO: Implement crowding operator.\n"
            "See Exercise 2d in challenge2.md for guidance."
        )

    def tournament_selection(self, population_rank: List[int],
                             crowding_distances: np.ndarray,
                             tournament_size: int) -> int:
        """Selects an individual using tournament selection.
        
        Randomly selects tournament_size individuals and returns the best one
        according to the crowding operator (rank first, then crowding distance).
        
        Args:
            population_rank (List[int]): Front rank for each solution.
            crowding_distances (np.ndarray): Crowding distance for each solution.
            tournament_size (int): Number of individuals in tournament.
            
        Returns:
            int: Index of the tournament winner.
        """
        possible_contestants = np.arange(len(population_rank))
        contestants = np.random.choice(possible_contestants, size=tournament_size, replace=False)

        best_idx = contestants[0]
        for i in range(1, len(contestants)):
            competitor_idx = contestants[i]
            winner_idx = self.crowding_operator(best_idx, competitor_idx,
                                                population_rank, crowding_distances)
            best_idx = winner_idx

        return best_idx
