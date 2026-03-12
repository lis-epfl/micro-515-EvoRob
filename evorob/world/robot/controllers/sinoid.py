import numpy as np

from evorob.world.robot.controllers.base import Controller

class OscillatoryController(Controller):
    """Simple oscillatory controller using sine waves for each actuator.

    This controller generates periodic motion patterns without using observations.
    Each joint oscillates with its own amplitude, frequency, and phase.
    """

    def __init__(
        self, input_size: int = 0, output_size: int = None, hidden_size: int = 0
    ):
        """Initialize the oscillatory controller.

        Args:
            output_size: Number of actuators to control
            input_size: Not used, kept for API compatibility
        """
        assert output_size is not None, (
            "output_size must be specified for OscillatoryController"
        )
        self.output_size = output_size
        self.time_step = 0.0
        self.n_params = self.get_num_params()

        # TODO: Initialize parameters for oscillatory control
        # You need 3 parameters per actuator: amplitude, frequency, phase
        # - self.amplitudes: uniform random in [0.1, 1.0] (shape: output_size)
        # - self.frequencies: uniform random in [0.5, 2.0] (shape: output_size)
        # - self.phases: uniform random in [0, 2*pi] (shape: output_size)
        # Hint: Use np.random.uniform(low, high, size)
        self.amplitudes = np.random.uniform(0.1, 1.0, size=output_size)
        self.frequencies = np.random.uniform(0.5, 2.0, size=output_size)
        self.phases = np.random.uniform(0, 2 * np.pi, size=output_size)
        # raise NotImplementedError("TODO: Initialize oscillatory parameters")

    def get_action(self, state):
        """Generate oscillatory actions based on time.

        Args:
            state: Observation (not used by this controller)

        Returns:
            actions: Array of actuator commands, shape (output_size,) or (batch_size, output_size)
        """
        # TODO: Compute oscillatory actions using sine waves
        # Compute action with parameterized sine wave.
        # Then increment self.time_step by e.g. 0.01
        # Clip actions to [-1.0, 1.0]
        #
        # For vectorized environments (batch of observations):
        # Check if state is 2D, if so replicate actions for each environment
        # Hint: Use np.tile(actions, (batch_size, 1))
        #raise NotImplementedError("TODO: Implement oscillatory action generation")
        actions = self.amplitudes * np.sin(
            2 * np.pi * self.frequencies * self.time_step + self.phases
        )
        self.time_step += 0.01
        actions = np.clip(actions, -1.0, 1.0)
        if state.ndim == 2:  # Batch of observations
            batch_size = state.shape[0]
            actions = np.tile(actions, (batch_size, 1))
        return actions
    def set_weights(self, weights):
        """Set controller parameters from flat array.

        Args:
            weights: Flat array of size (3 * output_size,)
                    [amplitudes, frequencies, phases]
        """
        # TODO: Extract parameters from weights array
        # weights structure: [amp1, amp2, ..., freq1, freq2, ..., phase1, phase2, ...]
        # Update self.amplitudes, self.frequencies, self.phases accordingly
        # Reset time to 0
        self.amplitudes = weights[: self.output_size]
        self.frequencies = weights[self.output_size : 2 * self.output_size]
        self.phases = weights[2 * self.output_size :]
        self.reset_controller() 

    def geno2pheno(self, genotype):
        """Alias for set_weights."""
        self.set_weights(genotype)
        self.reset_controller()

    def get_num_params(self):
        """Return total number of parameters.

        Returns:
            int: 3 * output_size (amplitude, frequency, phase for each actuator)
        """
        # TODO: Return the total number of parameters
        return 3 * self.output_size

    def reset_controller(self):
        """Reset the controller state (time)."""
        self.time_step = 0.0
