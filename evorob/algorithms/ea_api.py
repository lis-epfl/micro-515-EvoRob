import numpy as np

from evorob.algorithms.base_ea import EA


class EvoAlgAPI(EA):
    """Evolutionary algorithm API wrapper.

    This class provides an interface to wrap any EA framework that uses
    the ask-tell pattern (CMA-ES, pyribs, evosax, etc.).

    Example frameworks to use:
    - CMA-ES: https://github.com/CMA-ES/pycma
    - pyribs: https://github.com/icaros-usc/pyribs/
    - evosax: https://github.com/RobertTLange/evosax/
    - EvoJAX: https://github.com/google/evojax
    """

    def __init__(self, n_params: int, population_size: int = 100,
                 output_dir: str = "./results/EA", **kwargs):
        """Initialize the evolutionary algorithm.

        Args:
            n_params: Dimensionality of the search space
            population_size: Number of solutions per generation
            output_dir: Directory for saving checkpoints
            **kwargs: Additional arguments for the EA framework
        """
        # TODO: Initialize your chosen EA framework here
        self.n_params = n_params
        self.population_size = population_size
        
        # % bookkeeping for base EA
        self.directory_name = output_dir
        self.current_gen = 0
        self.full_x = []
        self.full_f = []
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = None
        self.f = None

        raise NotImplementedError(
            "TODO: Initialize your chosen EA framework.\n"
            "Recommended: pip install cma, then import cma and create CMAEvolutionStrategy.\n"
            "See https://github.com/CMA-ES/pycma for documentation."
        )

    def ask(self) -> np.ndarray:
        """Sample population from the algorithm.

        Returns:
            population: Array of shape (population_size, n_params)
                       Each row is a candidate solution
        """
        # TODO: Get new population from your EA
        # Make sure the returned array has shape (population_size, n_params)

        raise NotImplementedError(
            "TODO: Implement ask() to sample new population.\n"
            "This should return an array of shape (population_size, n_params)."
        )

    def tell(self, population: np.ndarray, fitnesses: np.ndarray, save_checkpoint: bool = False) -> None:
        """Update the algorithm with evaluated population.

        Args:
            population: Array of shape (population_size, n_params)
            fitnesses: Array of shape (population_size,) with fitness values
                      Higher is better (maximization)
            save_checkpoint: Whether to save checkpoint after update
        """
        # TODO: Update your EA with the evaluated population
        # Note: Some algorithms minimize, others maximize.
        # Adjust accordingly (negate fitnesses if needed).
        
        # After updating the EA, do bookkeeping for checkpointing:
        self.full_f.append(fitnesses)
        self.full_x.append(population)
        self.f = fitnesses
        self.x = population
        
        # Track best individual
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > self.f_best_so_far:
            self.f_best_so_far = fitnesses[best_idx]
            self.x_best_so_far = population[best_idx].copy()
        
        if save_checkpoint:
            self.save_checkpoint()
        self.current_gen += 1

        raise NotImplementedError(
            "TODO: Implement tell() to update the EA.\n"
            "Pass the population and their fitness values to update the search distribution.\n"
            "Don't forget to add the bookkeeping code shown above for checkpointing!"
        )
