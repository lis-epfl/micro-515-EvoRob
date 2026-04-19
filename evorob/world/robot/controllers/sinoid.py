import numpy as np
from evorob.world.robot.controllers.base import Controller


class PhaseOscillatorController(Controller):
    """
    Per-joint phase oscillator used ONLY to generate rhythmic phase features.

    It does NOT output motor actions.
    It outputs:
        [sin(phi_i), cos(phi_i)] for each joint i

    Parameters per joint:
        - frequency
        - phase offset

    Total params = 2 * output_size
    """

    def __init__(
        self,
        output_size: int = 8,
        dt: float = 0.01,
        default_frequency: float = 1.0,
    ):
        self.output_size = int(output_size)
        self.dt = float(dt)
        self.time_step = 0.0

        self.frequencies = np.full(
            self.output_size, default_frequency, dtype=np.float32
        )
        self.phases = np.zeros(self.output_size, dtype=np.float32)

    def reset_controller(self, batch_size=1):
        self.time_step = 0.0

    def step_time(self):
        self.time_step += self.dt

    def get_phase(self):
        return 2.0 * np.pi * self.frequencies * self.time_step + self.phases

    def get_phase_features(self, state=None):
        """
        Returns:
            - (2 * output_size,) for single input / no input
            - (batch_size, 2 * output_size) for batched input

        If state is batched, tiles the same phase feature vector across the batch.
        """
        phase = self.get_phase()
        sin_phase = np.sin(phase).astype(np.float32)
        cos_phase = np.cos(phase).astype(np.float32)
        feat = np.concatenate([sin_phase, cos_phase], axis=0)

        if state is None:
            return feat

        x = np.asarray(state)
        if x.ndim == 2:
            return np.tile(feat[None, :], (x.shape[0], 1))

        return feat

    def get_action(self, state):
        """
        Kept for API compatibility.
        Returns phase features, not actions.
        """
        feat = self.get_phase_features(state)
        self.step_time()
        return feat

    def set_weights(self, weights):
        """
        weights format:
            first output_size  -> frequencies
            last output_size   -> phases
        """
        weights = np.asarray(weights, dtype=np.float32).ravel()
        expected = 2 * self.output_size
        if len(weights) != expected:
            raise ValueError(f"Expected {expected} params, got {len(weights)}")

        self.frequencies = weights[:self.output_size].copy().astype(np.float32)
        self.phases = weights[self.output_size:].copy().astype(np.float32)
        self.reset_controller()

    def get_weights(self):
        return np.concatenate([self.frequencies, self.phases]).astype(np.float32)

    def get_num_params(self):
        return 2 * self.output_size

    def geno2pheno(self, genotype):
        self.set_weights(genotype)