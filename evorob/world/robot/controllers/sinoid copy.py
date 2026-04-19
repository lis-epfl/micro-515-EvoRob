import numpy as np
from evorob.world.robot.controllers.base import Controller


class PhaseOscillatorController(Controller):
    """
    Global-phase oscillator.

    Generates one shared phase signal:
        phi(t) = 2*pi*f*t + phi0

    And returns features:
        [sin(phi), cos(phi)]

    Optional fixed offsets can be added later for leg groups, but start simple.
    """

    def __init__(self, dt: float = 0.01, default_frequency: float = 1.0, default_phase: float = 0.0):
        self.dt = float(dt)
        self.time_step = 0.0

        self.frequency = np.float32(default_frequency)
        self.phase = np.float32(default_phase)

    def reset_controller(self, batch_size=1):
        self.time_step = 0.0

    def step_time(self):
        self.time_step += self.dt

    def get_phase_features(self, state):
        phi = 2.0 * np.pi * self.frequency * self.time_step + self.phase
        feat = np.array([np.sin(phi), np.cos(phi)], dtype=np.float32)

        state = np.asarray(state)
        if state.ndim == 2:
            return np.tile(feat[None, :], (state.shape[0], 1))
        return feat

    def get_action(self, state):
        feat = self.get_phase_features(state)
        self.step_time()
        return feat

    def set_weights(self, weights):
        weights = np.asarray(weights, dtype=np.float32).ravel()
        if len(weights) != 2:
            raise ValueError(f"Expected 2 params [frequency, phase], got {len(weights)}")

        self.frequency = np.float32(weights[0])
        self.phase = np.float32(weights[1])
        self.reset_controller()

    def get_num_params(self):
        return 2

    def geno2pheno(self, genotype):
        self.set_weights(genotype)

    def get_weights(self):
        return np.array([self.frequency, self.phase], dtype=np.float32)