import os
from typing import Tuple

import cma
import jax
import numpy as np
from evosax.algorithms import CMA_ES
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.schedulers import Scheduler

from evorob.algorithms.base_ea import EA
from evorob.utils.filesys import search_file_list


class CMAESAPI(EA):
    """Wrapper for the original CMA-ES library (cma package)."""

    def __init__(
        self,
        n_params: int,
        population_size: int,
        num_generations: int = 100,
        sigma: float = 0.3,
        bounds: Tuple[int, int] = (-1, 1),
        output_dir: str = "./results/CMAES",
    ):
        self.population_size = population_size
        self.n_gen = num_generations
        self.n_params = n_params

        # % bookkeeping for base EA
        self.directory_name = output_dir
        self.current_gen = 0
        self.full_x = []
        self.full_f = []
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = None
        self.f = None

        # Initialize with random mean
        initial_mean = np.random.uniform(bounds[0], bounds[1], n_params)

        # Create CMA-ES optimizer
        opts = {"popsize": population_size, "bounds": bounds}
        self.es = cma.CMAEvolutionStrategy(x0=initial_mean, sigma0=sigma, inopts=opts)

    def ask(self):
        """Sample population from CMA-ES."""
        population = self.es.ask()
        return np.array(population)

    def tell(self, population, fitnesses, save_checkpoint: bool = False):
        """Update CMA-ES with evaluated population
        Note: CMA-ES minimizes, so negate fitnesses."""
        self.es.tell(population.tolist(), (-fitnesses).tolist())

        # % bookkeeping for checkpointing
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

    def load_checkpoint(self):
        dir_path = search_file_list(self.directory_name, 'f_best.npy')
        assert len(dir_path) > 0;
        "No files are here, check the directory_name!!"

        self.current_gen = int(dir_path[-1].split('/')[-2])
        curr_gen_path = os.path.join(self.directory_name, str(self.current_gen))
        print(f"Loading from: {curr_gen_path}")
        self.full_f = np.load(os.path.join(self.directory_name, 'full_f.npy'))
        self.full_x = np.load(os.path.join(self.directory_name, 'full_x.npy'))
        self.f_best_so_far = np.load(os.path.join(curr_gen_path, 'f_best.npy'))
        self.x_best_so_far = np.load(os.path.join(curr_gen_path, 'x_best.npy'))
        self.x = np.load(os.path.join(curr_gen_path, 'x.npy'))
        self.f = np.load(os.path.join(curr_gen_path, 'f.npy'))

        self.cmaes = self.load_cmeas()
        for x, f in zip(self.full_x, self.full_f):
            self.cmaes.tell(x, f)

class EvosaxAPI(EA):
    """Wrapper for Evosax library (JAX-based evolutionary strategies)."""

    def __init__(
        self, population_size: int, n_params: int, num_generations: int = 100, output_dir: str = "./results/Evosax"
    ):
        self.population_size = population_size
        self.n_gen = num_generations
        self.n_params = n_params
        self.rng = jax.random.key(0)

        # % bookkeeping for base EA
        self.directory_name = output_dir
        self.current_gen = 0
        self.full_x = []
        self.full_f = []
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = None
        self.f = None

        # Initialize strategy
        self.strategy = CMA_ES(popsize=population_size, num_dims=n_params)
        self.rng, rng_init = jax.random.split(self.rng)
        self.es_params = self.strategy.default_params
        self.state = self.strategy.initialize(rng_init, self.es_params)

    def ask(self):
        """Sample population from Evosax strategy."""
        self.rng, rng_ask = jax.random.split(self.rng)
        population, self.state = self.strategy.ask(rng_ask, self.state, self.es_params)
        return np.array(population)

    def tell(self, population, fitnesses, save_checkpoint: bool = True):
        """Update Evosax strategy with evaluated population."""
        import jax.numpy as jnp

        # Evosax expects fitnesses as JAX array
        self.state = self.strategy.tell(
            population, jnp.array(fitnesses), self.state, self.es_params
        )

        # % bookkeeping for checkpointing
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


class PyribsAPI(EA):
    """Wrapper for pyribs library (single-objective optimization without diversity)."""

    def __init__(
        self,
        population_size: int,
        n_params: int,
        num_generations: int = 100,
        sigma: float = 0.5,
        output_dir: str = "./results/Pyribs",
    ):
        self.n_params = n_params
        self.n_gen = num_generations
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

        # Create a minimal archive (1D grid with 1 cell for single-objective)
        self.archive = GridArchive(
            solution_dim=n_params,
            dims=[1],  # Single cell
            ranges=[(0, 1)],  # Dummy range
        )

        # Create emitter with Gaussian mutations
        initial_solution = np.random.uniform(-1, 1, n_params)
        self.emitter = GaussianEmitter(
            self.archive,
            sigma=sigma,
            x0=initial_solution,
            batch_size=population_size,
        )

        # Create scheduler
        self.scheduler = Scheduler(self.archive, [self.emitter])

    def ask(self):
        """Sample population from pyribs scheduler."""
        return self.scheduler.ask()

    def tell(self, population, fitnesses, save_checkpoint: bool = True):
        """Update pyribs scheduler with evaluated population (using dummy measures)."""
        # For single-objective, use dummy measures (all zeros)
        measures = np.zeros((len(fitnesses), 1))
        self.scheduler.tell(fitnesses, measures)

        # % bookkeeping for checkpointing
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


EvoAlgAPI = CMAESAPI