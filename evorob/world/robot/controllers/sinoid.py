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
        raise NotImplementedError("TODO: Initialize oscillatory parameters")

    def get_action(self, state):
        """Generate oscillatory actions based on time.

        Args:
            state: Observation (not used by this controller)

        Returns:
            actions: Array of actuator commands, shape (output_size,) or (batch_size, output_size)
        """
        # TODO: Compute oscillatory actions using sine waves
        # Formula: amplitude * sin(2*pi*frequency*time + phase)
        # Then increment self.time_step by 0.01
        # Clip actions to [-1.0, 1.0]
        #
        # For vectorized environments (batch of observations):
        # Check if state is 2D, if so replicate actions for each environment
        # Hint: Use np.tile(actions, (batch_size, 1))
        raise NotImplementedError("TODO: Implement oscillatory action generation")

    def set_weights(self, weights):
        """Set controller parameters from flat array.

        Args:
            weights: Flat array of size (3 * output_size,)
                    [amplitudes, frequencies, phases]
        """
        # TODO: Extract parameters from weights array
        # weights structure: [amp1, amp2, ..., freq1, freq2, ..., phase1, phase2, ...]
        # - self.amplitudes = weights[0:output_size]
        # - self.frequencies = 5 * weights[output_size:2*output_size] (scale to ~[0, 5] Hz)
        # - self.phases = pi * weights[2*output_size:3*output_size] (scale to ~[0, pi])
        # Reset time to 0
        raise NotImplementedError("TODO: Implement parameter setting")

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
        raise NotImplementedError("TODO: Compute number of parameters")

    def reset_controller(self):
        """Reset the controller state (time)."""
        self.time_step = 0.0
