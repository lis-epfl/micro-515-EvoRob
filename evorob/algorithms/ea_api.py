import numpy as np

class EvoAlgAPI:
    """Evolutionary algorithm API."""

    def __init__(self, num_params: int, population_size: int, **kwargs):
        # TODO!
        raise NotImplementedError

    def ask(self) -> np.ndarray:
        """Sample population from the algorithm."""
        # TODO!
        raise NotImplementedError

    def tell(self, population, fitnesses) -> None:
        """Update the algorithm with evaluated population."""

        # TODO!
        raise NotImplementedError